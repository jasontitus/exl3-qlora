"""CPU-only native target offset/category and paired receipt integrity tests."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exl3_mlx.agent_edits import TOOL, render_command, unified_patch
from exl3_mlx.data import load_tokenizer, tokenize_row

SCRIPT = Path(__file__).parents[1] / "scripts/summarize_agent_kernel_eval.py"
spec = importlib.util.spec_from_file_location("agent_eval", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
MODEL = Path.home() / "models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"


@pytest.fixture(scope="module")
def tokenizer():
    if not (MODEL / "tokenizer.json").exists():
        pytest.skip("Real local tokenizer unavailable")
    return load_tokenizer(MODEL)


def native_row():
    before = {
        "kernel.metal": b"// old comment\nfloat y = x * 1.0f;\n",
        "host.cpp": b"launch(1);\n",
        "tests/test_kernel.cpp": b"check(1);\n",
        "docs/results.md": "Old π result.\n".encode(),
        "kernel.cu": b"run(1);\n",
    }
    after = {
        "kernel.metal": b"// new comment\nfloat y = x;\n",
        "host.cpp": b"launch(2);\n",
        "tests/test_kernel.cpp": b"check(2);\n",
        "docs/results.md": "New π result.\n".encode(),
        "kernel.cu": b"run(2);\n",
    }
    return {
        "case_id": "fixture1",
        "source_id": "heldout-family",
        "tools": [TOOL],
        "messages": [
            {"role": "user", "content": 'Edit this source; retain "π" and semantics.'},
            {
                "role": "assistant",
                "content": "Remove redundant work.",
                "tool_calls": [
                    {
                        "id": "edit_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": {"cmd": render_command(unified_patch(before, after))},
                        },
                    }
                ],
            },
        ],
    }


def normalize(row):
    return {k: row[k] for k in ("messages", "tools")}


@pytest.fixture
def evidence(tmp_path, tokenizer):
    raw = native_row()
    prepared = [tokenize_row(normalize(raw), tokenizer)]
    labels = audit.aligned_categories([raw], prepared, tokenizer)
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(json.dumps(raw) + "\n")
    data = tmp_path / "prepared.jsonl"
    data.write_text(json.dumps(prepared[0]) + "\n")
    targets = labels[0]["target_ids"]

    def report(pid, adapter):
        values = [0.25 if adapter is None else 0.2] * len(targets)
        row = {
            "target_ids": targets,
            "predicted_ids": list(targets),
            "token_nll": values,
            "loss": values[0],
            "correct_tokens": len(targets),
            "supervised_tokens": len(targets),
            "logits_sha256": "1" * 64,
        }
        argv = [
            "evaluate",
            "--model",
            str(MODEL.resolve()),
            "--data",
            str(data),
            "--token-metrics",
            "--chunk-size",
            "2048",
            "--activation-mode",
            "checkpoint",
        ]
        if adapter is not None:
            argv += ["--adapter", adapter]
        return {
            "rows": [row],
            "examples": 1,
            "loss": values[0],
            "supervised_tokens": len(targets),
            "logits_sha256": "2" * 64,
            "adapter": adapter,
            "checkpoint": str(MODEL.resolve()),
            "activation_mode": "checkpoint",
            "evaluation_context": {
                "argv": argv,
                "process_id": pid,
                "data": str(data),
                "input_identity": {"data_sha256": audit.digest(data.read_bytes())},
                "selected_examples_sha256": audit.digest(audit.canonical(prepared)),
                "seq_length": len(prepared[0]["input_ids"]),
                "requested_max_examples": 0,
                "implementation": {
                    "source": {"workflow.py": "a" * 64},
                    "runtime": {"mlx": "test"},
                    "python": "3.12",
                },
            },
        }

    base = report(1234001, None)
    adapted = report(1234002, str(tmp_path / "adapter"))
    kwargs = {
        "prepared_path": data,
        "prepared_sha256": audit.digest(data.read_bytes()),
        "model": MODEL,
    }
    return SimpleNamespace(
        raw=raw,
        prepared=prepared,
        labels=labels,
        base=base,
        adapted=adapted,
        kwargs=kwargs,
        raw_path=raw_path,
        data=data,
        tmp=tmp_path,
    )


def test_real_offsets_cover_native_dsml_unicode_and_every_category(evidence):
    e = evidence
    labels = e.labels[0]
    families = {s.split("/")[0] for s in labels["categories"]}
    assert {
        "metal",
        "host_support_source",
        "documentation",
        "tests_benchmarks",
        "other_device_source",
        "tool_boilerplate",
        "patch_metadata",
        "rationale",
        "eos",
    } <= families
    assert "metal/added/code_or_other" in labels["categories"]
    assert "metal/removed/code_or_other" in labels["categories"]
    assert "metal/added/comment_line" in labels["categories"]
    assert labels["categories"][-1] == "eos/terminator/metadata"
    assert labels["case_id"] == "fixture1"
    assert len(labels["categories"]) == len(labels["target_ids"])
    audit.validate_pair(e.base, e.adapted, e.prepared, e.labels, **e.kwargs)
    summary = audit.summarize(e.base, e.adapted, e.labels)
    assert summary["aggregate"]["all"]["nll_delta"] == pytest.approx(-0.05)
    assert summary["aggregate"]["family:metal"]["nll_delta"] == pytest.approx(-0.05)
    assert summary["aggregate"]["all"]["tokens"] == len(labels["target_ids"])
    assert len(summary["rows"][0]["tokens"]) == len(labels["target_ids"])


@pytest.mark.parametrize(
    "suffix",
    ["garbage\n", "--- a/other.metal\n", "@@ -1,8 +1,8 @@\n incomplete\n", "+++ b/other.metal\n"],
)
def test_malformed_or_trailing_diff_sections_rejected(suffix):
    patch = "--- a/a.metal\n+++ b/a.metal\n@@ -1 +1 @@\n-x\n+y\n"
    with pytest.raises(ValueError):
        audit.patch_spans(patch + suffix)


def test_new_file_and_header_like_removed_lines():
    patch = "--- /dev/null\n+++ b/new.metal\n@@ -0,0 +1 @@\n+float x;\n"
    assert audit.patch_spans(patch)[-1][2] == ("metal", "added", "code_or_other")
    patch = "--- a/a.metal\n+++ b/a.metal\n@@ -1 +1 @@\n--- not a header\n+text\n"
    assert audit.patch_spans(patch)[-2][2][1] == "removed"


@pytest.mark.parametrize(
    "path,family",
    [
        ("test/kernel.metal", "tests_benchmarks"),
        ("docs/a.cpp", "documentation"),
        ("a.mm", "host_support_source"),
        ("a.metal.h", "metal"),
        ("a.h", "host_support_source"),
        ("a.cuh", "other_device_source"),
        ("a.json", "other_file"),
    ],
)
def test_path_category_is_explicit_heuristic(path, family):
    assert audit.file_category(path) == family


@pytest.mark.parametrize("path", ["../outside", "/tmp/out", "a/../out", "a/.git/config"])
def test_unsafe_patch_path_rejected(path):
    with pytest.raises(ValueError):
        audit.file_category(path)


def test_raw_order_can_differ_but_duplicate_or_truncated_data_rejected(evidence, tokenizer):
    e = evidence
    duplicate = copy.deepcopy(e.raw)
    duplicate["case_id"] = "alias"
    with pytest.raises(ValueError, match="Duplicate"):
        audit.aligned_categories([e.raw, duplicate], e.prepared, tokenizer)
    truncated = copy.deepcopy(e.prepared)
    truncated[0]["input_ids"].pop()
    with pytest.raises(ValueError, match="exact complete"):
        audit.aligned_categories([e.raw], truncated, tokenizer)
    shifted = copy.deepcopy(e.prepared)
    shifted[0]["loss_start"] += 1
    with pytest.raises(ValueError, match="exact complete"):
        audit.aligned_categories([e.raw], shifted, tokenizer)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["rows"][0]["target_ids"].__setitem__(0, 123),
        lambda r: r["rows"][0]["predicted_ids"].pop(),
        lambda r: r["rows"][0]["predicted_ids"].__setitem__(0, 1 << 30),
        lambda r: r["rows"][0]["token_nll"].__setitem__(0, float("nan")),
        lambda r: r["rows"][0].__setitem__("loss", 9),
        lambda r: r.__setitem__("loss", 9),
        lambda r: r["rows"][0].__setitem__("correct_tokens", 0),
        lambda r: r.__setitem__("examples", True),
        lambda r: r["evaluation_context"].__setitem__("selected_examples_sha256", "b" * 64),
        lambda r: r["evaluation_context"]["input_identity"].__setitem__("data_sha256", "b" * 64),
        lambda r: r["evaluation_context"].__setitem__("process_id", 1234001),
        lambda r: r["evaluation_context"].__setitem__("seq_length", 2),
        lambda r: r["evaluation_context"].__setitem__("requested_max_examples", 2),
        lambda r: r["evaluation_context"]["implementation"]["runtime"].__setitem__(
            "mlx", "changed"
        ),
        lambda r: r.__setitem__("activation_mode", "bf16"),
        lambda r: r["evaluation_context"]["argv"].remove("--token-metrics"),
        lambda r: r["evaluation_context"]["argv"].__setitem__(
            r["evaluation_context"]["argv"].index("2048"), "8192"
        ),
        lambda r: r.__setitem__("adapter", None),
        lambda r: r.__setitem__("checkpoint", "/other/model"),
    ],
)
def test_invalid_pair_cannot_generate_summary(evidence, change):
    e = evidence
    bad = copy.deepcopy(e.adapted)
    change(bad)
    with pytest.raises(ValueError):
        audit.validate_pair(e.base, bad, e.prepared, e.labels, **e.kwargs)


def test_source_gains_not_confused_with_tool_gains(evidence):
    e = evidence
    after = copy.deepcopy(e.base)
    for i, category in enumerate(e.labels[0]["categories"]):
        after["rows"][0]["token_nll"][i] -= 0.1 if category.startswith("tool_boilerplate/") else 0
    report = audit.summarize(e.base, after, e.labels)
    assert report["aggregate"]["all"]["nll_delta"] < 0
    assert report["aggregate"]["family:metal"]["nll_delta"] == 0
    assert report["aggregate"]["metal/added/code_or_other"]["nll_delta"] == 0


def test_exclusive_publication_and_category_only(evidence):
    e = evidence
    args = SimpleNamespace(
        raw=e.raw_path,
        prepared=e.data,
        model=MODEL,
        baseline=None,
        adapter=None,
        categories_only=True,
        output=e.tmp / "categories.json",
    )
    result = audit.run(args)
    assert result["evaluation_completed"] is False
    assert result["quality_certified"] is False
    assert result["categories"] == e.labels
    saved = args.output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.run(args)
    assert args.output.read_bytes() == saved


def test_full_cli_report_binds_input_hashes(evidence):
    e = evidence
    base_path, adapter_path = e.tmp / "base.json", e.tmp / "adapter.json"
    base_path.write_text(json.dumps(e.base))
    adapter_path.write_text(json.dumps(e.adapted))
    args = SimpleNamespace(
        raw=e.raw_path,
        prepared=e.data,
        model=MODEL,
        baseline=base_path,
        adapter=adapter_path,
        categories_only=False,
        output=e.tmp / "summary.json",
    )
    result = audit.run(args)
    assert result["evaluation_completed"] is True
    assert result["inputs"][str(base_path)] == audit.digest(base_path.read_bytes())
    assert all(str(p) in result["inputs"] for p in audit.HELPER_SOURCES)
    assert result["quality_certified"] is False
