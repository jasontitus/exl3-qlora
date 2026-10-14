"""CPU-only offline source intake tests; no model construction or GPU work."""

import copy
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src/exl3_mlx"
PACKAGE = "upstream_data_cpu_tests"
namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = namespace
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.upstream_data", ROOT / "upstream_data.py")
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


def test_upstream_identity_supports_embedded_template_and_detects_override(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"})
    )
    identity = data._helpers()[2]
    embedded = identity(tmp_path)
    assert set(embedded) == {"tokenizer.json", "tokenizer_config.json"}
    (tmp_path / "chat_template.jinja").write_text("{{ messages | length }}")
    separate = identity(tmp_path)
    assert set(separate) == {*embedded, "chat_template.jinja"}
    assert separate != embedded
    (tmp_path / "chat_template.jinja").unlink()
    assert identity(tmp_path) == embedded
    (tmp_path / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="No valid explicit local chat template"):
        identity(tmp_path)


def git(repo, *args):
    return (
        subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )


def save(path, value):
    path.write_text(json.dumps(value))


def manifest(path, value):
    value = copy.deepcopy(value)
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = data._sha(data._canonical(value))
    save(path, value)


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


@pytest.fixture
def case(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "fixture@example.test")
    git(repo, "config", "user.name", "Fixture")
    (repo / "kernel.metal").write_text(
        "constant int lanes=32;\nfloat f(float x) { return x * 1; }\nfloat g(float y) { return y; }\n"
    )
    (repo / "dispatch.cc").write_text("launch(32);\n")
    (repo / "common.h").write_text("#define TYPE float\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Baseline")
    parent = git(repo, "rev-parse", "HEAD")
    (repo / "kernel.metal").write_text(
        "constant int lanes=32;\nfloat f(float x) { return x; }\nfloat g(float y) { return y; }\n"
    )
    (repo / "dispatch.cc").write_text("launch(16);\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Optimize launch: author reports 10% faster on M5")
    head = git(repo, "rev-parse", "HEAD")
    m = {
        "format": data.FORMAT,
        "repositories": {"demo": {"path": str(repo), "url": "https://github.com/owner/demo"}},
        "candidates": [
            {
                "id": "demo-change",
                "repository": "demo",
                "commit": head,
                "parent": parent,
                "dependency_paths": ["common.h"],
                "outcome": "reported_improvement",
            }
        ],
    }
    source = tmp_path / "manifest.json"
    manifest(source, m)
    model = tmp_path / "model"
    model.mkdir()
    save(model / "config.json", {"vocab_size": 300, "text_config": {"vocab_size": 300}})

    def tokenize(pair, tokenizer):
        p, t = pair["prompt"].encode(), pair["completion"].encode()
        return {
            "input_ids": [0, *[b + 4 for b in p], 2, *[b + 4 for b in t], 1],
            "loss_start": len(p) + 2,
        }

    def validate(row, *, seq_length, vocab_size, truncate):
        assert truncate is False
        assert 2 <= seq_length <= 32768
        assert len(row["input_ids"]) <= seq_length + 1
        assert 1 <= row["loss_start"] < len(row["input_ids"])
        return row

    monkeypatch.setattr(
        data,
        "_helpers",
        lambda: (
            lambda _: types.SimpleNamespace(eos_token_id=1, decode=lambda ids: str(ids)),
            tokenize,
            lambda _: {"tokenizer": "fixture"},
            validate,
        ),
    )
    return types.SimpleNamespace(
        repo=repo,
        source=source,
        model=model,
        output=tmp_path / "out",
        manifest=m,
        candidate=m["candidates"][0],
        parent=parent,
        head=head,
    )


def run(c, **kwargs):
    manifest(c.source, c.manifest)
    return data.import_upstream(c.source, c.model, c.output, **kwargs)


def test_full_source_and_host_dispatch_preserved_without_hash_targets(case):
    report = run(case)
    assert report["counts"] == {
        "candidates": 1,
        "source_reconstructed": 1,
        "unique_normalized_patches": 1,
        "train": 1,
        "valid": 0,
        "test": 0,
        "excluded": 0,
    }
    row = rows(case.output / "train.jsonl")[0]
    prompt, answer = json.loads(row["prompt"]), json.loads(row["completion"])
    assert "* 1" in prompt["baseline_files"]["kernel.metal"]["content"]
    assert answer["optimized_files"]["dispatch.cc"]["content"] == "launch(16);\n"
    assert "common.h" in prompt["baseline_files"] and "common.h" not in answer["optimized_files"]
    assert "10% faster" not in row["prompt"] and "10% faster" in row["completion"]
    assert "sha256" not in row["completion"] and "blob" not in row["completion"]
    assert case.head not in row["completion"] and case.parent not in row["completion"]
    assert answer["verification_metadata"]["compile"] == "not_run"
    assert (
        answer["verification_metadata"]["source_claims"]["classification"]
        == "curator_declared_reported_improvement"
    )
    provenance = rows(case.output / "provenance.jsonl")[0]
    assert "index " in provenance["diff"]
    assert provenance["head"]["commit"] == case.head
    assert provenance["files"]["kernel.metal"]["after"]["blob"]
    checksum = provenance.pop("record_sha256")
    assert data._sha(data._canonical(provenance)) == checksum
    assert (
        report["execution_verified_examples"] == 0
        and report["quality_evaluation_available"] is False
    )


def test_post_or_inferred_annotations_never_become_prospective_prompt(case):
    case.candidate["annotations"] = {
        "hardware": {
            "text": "M5",
            "basis": "source",
            "evidence": [{"source": "commit_message", "quote": "M5"}],
        },
        "bottleneck": {
            "text": "inferred bandwidth",
            "basis": "inferred",
            "evidence": [{"source": "commit_message", "quote": "Optimize launch"}],
        },
    }
    run(case)
    row = rows(case.output / "train.jsonl")[0]
    prompt, answer = json.loads(row["prompt"]), json.loads(row["completion"])
    assert prompt["task"]["hardware"]["status"] == "unknown"
    assert "M5" not in row["prompt"] and "inferred bandwidth" not in row["prompt"]
    assert answer["analysis"]["bottleneck"]["status"] == "inferred"


def test_pre_change_bound_task_evidence_can_condition_prompt(case):
    path = case.source.parent / "task.txt"
    path.write_text("Target hardware is M1 Ultra.\n")
    case.manifest["evidence"] = {
        "task": {
            "path": "task.txt",
            "sha256": data._file_sha(path),
            "url": "https://example.test/task",
            "phase": "pre_change",
        }
    }
    case.candidate["annotations"] = {
        "hardware": {
            "text": "M1 Ultra",
            "basis": "source",
            "evidence": [{"source": "task", "quote": "M1 Ultra"}],
        }
    }
    run(case)
    assert (
        json.loads(rows(case.output / "train.jsonl")[0]["prompt"])["task"]["hardware"]["text"]
        == "M1 Ultra"
    )


def test_fabricated_quote_is_excluded(case):
    case.candidate["annotations"] = {
        "hardware": {
            "text": "imaginary",
            "basis": "source",
            "evidence": [{"source": "commit_message", "quote": "not present"}],
        }
    }
    report = run(case)
    assert report["counts"]["train"] == 0
    assert "quote" in rows(case.output / "ledger.jsonl")[0]["error"]


@pytest.mark.parametrize(
    "mutation",
    [
        "checksum",
        "duplicate_id",
        "wrong_parent",
        "root",
        "unknown_policy",
        "binary",
        "symlink",
        "unsafe_dependency",
    ],
)
def test_bad_manifest_or_source_never_admitted(case, mutation):
    if mutation == "checksum":
        case.source.write_text(case.source.read_text().replace("demo-change", "foreign-change"))
        with pytest.raises(ValueError, match="checksum"):
            data.import_upstream(case.source, case.model, case.output)
        return
    if mutation == "duplicate_id":
        case.manifest["candidates"].append(copy.deepcopy(case.candidate))
        with pytest.raises(ValueError, match="identities"):
            run(case)
        return
    if mutation == "wrong_parent":
        case.candidate["parent"] = case.head
    elif mutation == "root":
        case.candidate["commit"] = case.parent
        case.candidate.pop("parent")
    elif mutation == "unknown_policy":
        case.candidate["context_policy"] = "crop"
    elif mutation == "unsafe_dependency":
        case.candidate["dependency_paths"] = ["../secret"]
    else:
        path = case.repo / "bad"
        if mutation == "binary":
            path.write_bytes(b"a\x00b")
        else:
            path.symlink_to("kernel.metal")
        git(case.repo, "add", ".")
        git(case.repo, "commit", "-qm", "Additional change")
        case.candidate["parent"] = case.head
        case.candidate["commit"] = git(case.repo, "rev-parse", "HEAD")
    report = run(case)
    assert report["counts"]["train"] == 0
    assert report["exclusion_counts"]["source_reconstruction_failed"] == 1


def test_strict_execution_gate_ignores_manifest_success_booleans(case):
    case.candidate["verification"] = {
        "execution_verified": True,
        "compile": True,
        "numerical": True,
        "runtime": True,
    }
    report = run(case, minimum_verification="execution_verified")
    assert report["counts"]["train"] == 0
    assert report["exclusion_counts"] == {"bound_execution_verifier_not_implemented": 1}


def test_overlength_retains_whole_source_exclusion_never_truncates(case):
    report = run(case, max_length=512)
    assert report["counts"]["train"] == 0
    assert report["truncated"] == 0
    assert report["exclusion_counts"]["full_serialized_example_exceeds_context"] == 1
    assert rows(case.output / "provenance.jsonl")[0]["files"]["kernel.metal"]["after"][
        "content"
    ].endswith("float g(float y) { return y; }\n")


def test_alias_equivalent_patch_dedup_and_protection_propagate(case):
    duplicate = copy.deepcopy(case.candidate)
    duplicate.update(id="alias", protected=True)
    case.manifest["candidates"].append(duplicate)
    report = run(case)
    assert report["counts"]["unique_normalized_patches"] == 1
    assert report["counts"]["train"] == 0
    assert report["exclusion_counts"]["protected_lineage"] == 2
    assert report["exclusion_counts"].get("duplicate_normalized_patch", 0) == 0


def test_failed_protected_candidate_still_quarantines_repository(case):
    duplicate = copy.deepcopy(case.candidate)
    duplicate.update(id="bad-heldout", protected=True, parent="0" * 40)
    case.manifest["candidates"].append(duplicate)
    report = run(case)
    assert report["counts"]["train"] == 0
    assert report["exclusion_counts"]["protected_lineage"] == 1
    assert report["exclusion_counts"]["source_reconstruction_failed"] == 1


def test_negative_equivalent_does_not_leave_positive_alias(case):
    duplicate = copy.deepcopy(case.candidate)
    duplicate.update(id="negative", outcome="negative")
    case.manifest["candidates"].append(duplicate)
    report = run(case)
    assert report["counts"]["train"] == 0
    assert (
        report["exclusion_counts"]["equivalent_patch_has_negative_or_contradictory_evidence"] == 1
    )


def test_external_protected_commit_key(case):
    case.manifest["protected_keys"] = ["commit:" + case.parent]
    assert run(case)["counts"]["train"] == 0


def unit_binding(content, fragment):
    start = content.index(fragment)
    return {"start": start, "end": start + len(fragment), "sha256": data._sha(fragment)}


def reviewed(case):
    record = data.reconstruct_candidate(case.candidate, case.manifest["repositories"], {})
    units = []
    for path in ("kernel.metal", "dispatch.cc"):
        pair = record["files"][path]
        before, after = pair["before"]["content"].encode(), pair["after"]["content"].encode()
        b = before.splitlines(keepends=True)[1] if path == "kernel.metal" else before
        a = after.splitlines(keepends=True)[1] if path == "kernel.metal" else after
        units.append(
            {
                "path": path,
                "kind": "complete_function" if path == "kernel.metal" else "whole_file",
                "before": unit_binding(before, b),
                "after": unit_binding(after, a),
            }
        )
    case.candidate.update(
        context_policy="reviewed_units",
        reviewed_units={
            "reviewed_by": "Fixture review",
            "dependency_review": "common.h retained; build contracts unknown",
            "units": units,
        },
    )
    return units


def test_reviewed_complete_units_replay_all_changes_and_keep_raw_source(case):
    reviewed(case)
    report = run(case)
    assert report["counts"]["train"] == 1
    row = rows(case.output / "train.jsonl")[0]
    assert "float g" not in row["prompt"] and "float g" not in row["completion"]
    assert "float f" in row["prompt"] and "float f" in row["completion"]
    record = rows(case.output / "provenance.jsonl")[0]
    assert "float g" in record["files"]["kernel.metal"]["after"]["content"]
    assert record["reviewed_context"]["coverage"] == "all changed file bytes reconstructed exactly"
    assert record["verification"]["compile"] == "not_run"


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "indices",
        "boolean_indices",
        "omit_file",
        "omit_change",
        "partial_whole_file",
        "overlap",
        "kind",
        "reviewer",
        "unbound_path",
        "missing_binding",
        "wrong_policy",
    ],
)
def test_reviewed_unit_binding_negatives(case, mutation):
    units = reviewed(case)
    u = units[0]
    if mutation == "hash":
        u["before"]["sha256"] = "0" * 64
    elif mutation == "indices":
        u["after"]["end"] = 10**9
    elif mutation == "boolean_indices":
        u["before"]["start"] = True
    elif mutation == "omit_file":
        units.pop()
    elif mutation == "omit_change":
        r = data.reconstruct_candidate(
            {
                k: v
                for k, v in case.candidate.items()
                if k not in {"context_policy", "reviewed_units"}
            },
            case.manifest["repositories"],
            {},
        )
        for side in ("before", "after"):
            raw = r["files"]["kernel.metal"][side]["content"].encode()
            u[side] = unit_binding(raw, raw.splitlines(keepends=True)[0])
    elif mutation == "partial_whole_file":
        u["kind"] = "whole_file"
    elif mutation == "overlap":
        units.append(copy.deepcopy(u))
    elif mutation == "kind":
        u["kind"] = "arbitrary_crop"
    elif mutation == "reviewer":
        case.candidate["reviewed_units"]["reviewed_by"] = ""
    elif mutation == "unbound_path":
        u["path"] = "other.metal"
    elif mutation == "missing_binding":
        u["after"].pop("sha256")
    else:
        case.candidate["context_policy"] = "complete_changed_files"
    report = run(case)
    assert report["counts"]["train"] == 0
    assert report["exclusion_counts"]["source_reconstruction_failed"] == 1


def test_git_hooks_external_diff_and_lazy_network_disabled(case):
    marker = case.source.parent / "executed"
    command = f"touch {marker}"
    git(case.repo, "config", "diff.external", command)
    git(case.repo, "config", "core.fsmonitor", command)
    assert run(case)["counts"]["train"] == 1
    assert not marker.exists()


def test_output_cannot_replace_existing_directory(case):
    case.output.mkdir()
    (case.output / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        run(case)
    assert (case.output / "keep").read_text() == "keep"


def test_historical_task_conditioning_is_explicit_and_still_source_grounded(case):
    case.candidate["task_conditioning"] = "historical_source"
    case.candidate["annotations"] = {
        "hardware": {
            "text": "M5",
            "basis": "source",
            "evidence": [{"source": "commit_message", "quote": "M5"}],
        },
        "bottleneck": {
            "text": "bandwidth inferred",
            "basis": "inferred",
            "evidence": [{"source": "commit_message", "quote": "Optimize launch"}],
        },
    }
    run(case)
    pair = rows(case.output / "train.jsonl")[0]
    prompt = json.loads(pair["prompt"])
    assert prompt["task"]["hardware"]["text"] == "M5"
    assert prompt["task"]["bottleneck"]["status"] == "unknown"
    assert "target-derived" in prompt["task_conditioning_scope"]
    assert "10% faster" not in pair["prompt"]


def test_duplicate_ineligible_record_does_not_block_eligible_equivalent(case):
    duplicate = copy.deepcopy(case.candidate)
    duplicate.update(id="second", outcome="source_only")
    case.candidate["annotations"] = {
        "rationale": {
            "text": "x" * 6000,
            "basis": "inferred",
            "evidence": [{"source": "commit_message", "quote": "Optimize"}],
        }
    }
    case.manifest["candidates"].append(duplicate)
    report = run(case, max_length=5000)
    assert report["selected_case_ids"] == ["second"]
    assert report["counts"]["unique_normalized_patches"] == 1


def test_byte_ranges_reject_split_utf8(case):
    # Unit metadata is source-byte based, never character-index based.
    side = {"content": "é"}
    with pytest.raises(UnicodeError):
        data._slice(side, {"start": 0, "end": 1, "sha256": data._sha(b"\xc3")})


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, True)])
def test_invalid_reviewed_indices_rejected(start, end):
    with pytest.raises(ValueError, match="indices"):
        data._slice({"content": "ab"}, {"start": start, "end": end, "sha256": data._sha(b"ab")})


def test_multiple_length_changes_replay_and_uncovered_middle_edit_rejects(case):
    b = b"head\nold-one\ngap\nold-two\ntail\n"
    a = b"head\nnew-long-one\ngap\nx\ntail\n"

    def binding(raw, text):
        return unit_binding(raw, text)

    units = [
        {
            "path": "kernel.metal",
            "kind": "complete_function",
            "before": binding(b, old),
            "after": binding(a, new),
        }
        for old, new in [(b"old-one\n", b"new-long-one\n"), (b"old-two\n", b"x\n")]
    ]
    record = {
        "files": {
            "kernel.metal": {
                "before": {"content": b.decode()},
                "after": {"content": a.decode()},
                "role": "changed",
            }
        }
    }
    candidate = {
        "context_policy": "reviewed_units",
        "reviewed_units": {
            "reviewed_by": "fixture",
            "dependency_review": "unknown",
            "units": units,
        },
    }
    data._reviewed_context(record, candidate)
    bad = copy.deepcopy(record)
    bad["files"]["kernel.metal"]["after"]["content"] = a.decode().replace("gap", "GAP")
    with pytest.raises(ValueError, match="gap"):
        data._reviewed_context(bad, candidate)


@pytest.mark.parametrize("deleted", [False, True])
def test_reviewed_new_and_deleted_file_empty_sides(case, deleted):
    text = "float new_kernel(float x) { return x; }\n"
    before, after = (text, "") if deleted else ("", text)
    record = {
        "files": {
            "new.metal": {
                "before": {"content": before, "mode": "100644"} if before else None,
                "after": {"content": after, "mode": "100644"} if after else None,
                "role": "changed",
            }
        }
    }
    unit = {
        "path": "new.metal",
        "kind": "whole_file",
        "before": unit_binding(before.encode(), before.encode()),
        "after": unit_binding(after.encode(), after.encode()),
    }
    candidate = {
        "context_policy": "reviewed_units",
        "reviewed_units": {
            "reviewed_by": "fixture",
            "dependency_review": "unknown",
            "units": [unit],
        },
    }
    data._reviewed_context(record, candidate)
    rendered = data._render_files(record, "after" if deleted else "before")
    assert rendered["new.metal"] is None


def test_source_mutation_during_tokenization_aborts_without_output(case, monkeypatch):
    helpers = data._helpers()

    def mutate(pair, tokenizer):
        case.source.write_text("{}")
        return helpers[1](pair, tokenizer)

    monkeypatch.setattr(data, "_helpers", lambda: (helpers[0], mutate, helpers[2], helpers[3]))
    with pytest.raises(ValueError, match="changed during import"):
        run(case)
    assert not case.output.exists()


def test_cpu_pair_serialization_matches_installed_mlx_lm_completion_contract():
    import ast
    import importlib.metadata

    source = Path(importlib.metadata.distribution("mlx-lm").locate_file("mlx_lm/tuner/datasets.py"))
    tree = ast.parse(source.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CompletionsDataset"
    )
    namespace = {}
    # This class is pure tokenizer Python; extracting it avoids mlx_lm's Metal initializer.
    exec(
        compile("from __future__ import annotations\n" + ast.unparse(node), str(source), "exec"),
        namespace,
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            text = "BOS" + "".join(
                m["role"] + ":" + m["content"] + ("EOS" if m["role"] == "assistant" else "")
                for m in messages
            )
            if kwargs.get("add_generation_prompt"):
                text += "assistant:"
            return list(text.encode())

    tokenizer = Tokenizer()
    row = {"prompt": "kernel λ", "completion": "improved λ"}
    ids, offset = namespace["CompletionsDataset"](
        [], tokenizer, "prompt", "completion", True
    ).process(row)
    assert data._tokenize_pair(row, tokenizer) == {"input_ids": ids, "loss_start": offset}


def test_cpu_pair_prefix_failure_and_text_only_validation():
    class Bad:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages) == 1 else [3, 4]

    with pytest.raises(ValueError, match="prefix"):
        data._tokenize_pair({"prompt": "x", "completion": "y"}, Bad())
    with pytest.raises(ValueError, match="text"):
        data._tokenize_pair({"prompt": "x", "completion": [1]}, Bad())


@pytest.mark.parametrize(
    "ids,start", [([1, 129264, 2], 1), ([1, 2], True), ([1, 2], 2), ([1, -1], 1), ([1, False], 1)]
)
def test_cpu_token_validator_rejects_invalid_text_record(ids, start):
    with pytest.raises(ValueError):
        data._validate_tokens(
            {"input_ids": ids, "loss_start": start}, seq_length=5, vocab_size=200000, truncate=False
        )


def test_output_envelope_and_metadata_preserve_standard_sft_data_contract(case, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.sft_data", ROOT / "sft_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = run(case)
    assert report["manifest_sha256"] == data._file_sha(case.output / "ktune-import.json")
    assert (
        json.loads((case.output / "ktune-import.json").read_text())["format"]
        == "exl3-mlx-upstream-import-v1"
    )
    monkeypatch.setattr(module, "_helpers", data._helpers)
    prepared = case.output.with_name("prepared")
    manifest_result = module.prepare_sft_data(case.output, case.model, prepared)
    assert manifest_result["counts"] == {"train": 1, "valid": 0, "test": 0}
    raw = rows(case.output / "train.jsonl")[0]
    tokenized = rows(prepared / "train.jsonl")[0]
    helpers = data._helpers()
    expected = helpers[1]({k: raw[k] for k in ("prompt", "completion")}, helpers[0](case.model))
    assert tokenized == expected
    provenance = rows(prepared / "provenance.jsonl")[0]
    assert provenance["metadata"]["case_id"] == raw["case_id"]
    assert provenance["metadata"]["evidence_tier"] == "source_reconstructed"
    assert raw["source_id"] in str(provenance)


def scoped_case(case):
    units = reviewed(case)
    units.pop()  # Deliberately scope to arithmetic, excluding host dispatch edit.
    case.candidate.update(
        context_policy="reviewed_operation",
        operation_scope={
            "description": "Only the complete f arithmetic function; not host dispatch",
            "omitted_path_reasons": {"dispatch.cc": "Separate host launch change excluded"},
        },
    )
    return units


def test_scoped_operation_binds_omissions_without_rendering_unrelated_source(case):
    scoped_case(case)
    assert run(case)["counts"]["train"] == 1
    record = rows(case.output / "provenance.jsonl")[0]
    pair = rows(case.output / "train.jsonl")[0]
    assert record["context_policy"] == "reviewed_operation"
    assert record["operation_scope"]["whole_commit_replayed"] is False
    omissions = record["operation_scope"]["omitted_changed_regions"]
    assert len(omissions) == 1 and omissions[0]["path"] == "dispatch.cc"
    assert omissions[0]["before"]["sha256"] == data._sha(b"launch(32);\n")
    assert "launch(32)" in record["files"]["dispatch.cc"]["before"]["content"]
    assert "launch(" not in pair["prompt"] + pair["completion"]
    assert "float f" in pair["prompt"] and "float f" in pair["completion"]
    assert "common.h" in pair["prompt"]  # Explicit unchanged dependency retained.
    assert "selected units only" in record["reviewed_context"]["coverage"]


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "empty", "no_change", "overlap", "wrong_policy"]
)
def test_scoped_operation_fail_closed(case, mutation):
    units = scoped_case(case)
    scope = case.candidate["operation_scope"]
    if mutation == "missing":
        scope["omitted_path_reasons"] = {}
    elif mutation == "extra":
        scope["omitted_path_reasons"]["kernel.metal"] = "No omitted edit exists"
    elif mutation == "empty":
        scope["description"] = ""
    elif mutation == "overlap":
        units.append(copy.deepcopy(units[0]))
    elif mutation == "wrong_policy":
        case.candidate["context_policy"] = "reviewed_units"
    else:
        for side, rev in (("before", case.parent), ("after", case.head)):
            raw = subprocess.check_output(
                ["git", "-C", str(case.repo), "show", f"{rev}:kernel.metal"]
            )
            units[0][side] = unit_binding(raw, raw.splitlines(keepends=True)[2])
        scope["omitted_path_reasons"]["kernel.metal"] = "Excluded f change"
    result = run(case)
    assert result["counts"]["train"] == 0
    assert result["exclusion_counts"] == {"source_reconstruction_failed": 1}


def test_scoped_operation_retains_protection_and_execution_gate(case):
    scoped_case(case)
    case.candidate["protected"] = True
    report = run(case, minimum_verification="execution_verified")
    assert report["counts"]["train"] == 0
    assert report["exclusion_counts"]["protected_lineage"] == 1
    assert report["exclusion_counts"]["bound_execution_verifier_not_implemented"] == 1
