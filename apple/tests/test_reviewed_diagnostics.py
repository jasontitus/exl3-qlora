"""Mixed admission is a consistency check, not a semantic claim verifier."""

import copy
import json

import pytest
import test_reviewed_corpus as fixtures
from test_reviewed_corpus import seal, write

from exl3_mlx import reviewed_corpus as intake
from exl3_mlx.data import load_tokenizer, tokenize_row

bundle = fixtures.bundle


def bind(root, name):
    return {"path": name, "sha256": intake.sha((root / name).read_bytes())}


def text_span(text, refs, **extra):
    return {
        "start": 0,
        "end": len(text.encode()),
        "sha256": intake.sha(text.encode()),
        "refs": refs,
        **extra,
    }


def diagnostic(bundle, split="train", case_id="diagnostic", source_id="one"):
    root, model, manifest, vals = bundle
    manifest["format"] = intake.MIXED_FORMAT
    manifest.setdefault("multi_views", []).append([source_id, case_id])
    baseline = vals["source"]["files"]["kernel.metal"]["before"]["content"]
    (root / (case_id + "-baseline.txt")).write_text(baseline)
    raw = {
        "messages": [
            {"role": "user", "content": "What should be measured? " + baseline},
            {
                "role": "assistant",
                "content": "Measure latency before assuming this arithmetic change helps.",
            },
        ]
    }
    evidence = {
        "format": "diagnostic-evidence-v1",
        "task_type": "choose_measurement",
        "prompt_mode": "before_only",
        "references": [
            {
                "id": "before",
                "kind": "before_source",
                "source_path": "kernel.metal",
                "file": bind(root, case_id + "-baseline.txt"),
                "start": 0,
                "end": len(baseline.encode()),
                "sha256": intake.sha(baseline.encode()),
            }
        ],
        "prompt_spans": [text_span(raw["messages"][0]["content"], ["before"])],
        "claims": [
            text_span(
                raw["messages"][1]["content"],
                ["before"],
                **{"class": "inferred", "limits": "Hypothesis; no timing measurement performed."},
            )
        ],
    }
    values = {
        "source": vals["source"],
        "raw": raw,
        "prepared": tokenize_row(raw, load_tokenizer(model)),
        "evidence": evidence,
    }
    entry = {"id": case_id, "kind": "diagnostic", "split": split, "inputs": {}}
    for role, value in values.items():
        name = case_id + "-" + role + ".json"
        write(root / name, value)
        entry["inputs"][role] = bind(root, name)
    review = {
        "format": "diagnostic-review-v1",
        "status": "approved",
        "reviewer": "test assertion",
        "cases": [{"id": entry["id"], "inputs": entry["inputs"]}],
    }
    write(root / (case_id + "-review.json"), review)
    entry["review"] = bind(root, case_id + "-review.json")
    manifest["entries"].append(entry)
    if split != "train":
        for suffix, value in ((".jsonl", raw), (".prepared.jsonl", values["prepared"])):
            (root / (split + suffix)).write_bytes(intake.canonical(value) + b"\n")
        manifest["frozen_splits"][split] = {
            "raw": bind(root, split + ".jsonl"),
            "prepared": bind(root, split + ".prepared.jsonl"),
        }
    seal(root, manifest)
    return values


def update(bundle, values):
    root, _, manifest, _ = bundle
    entry = manifest["entries"][-1]
    for role, value in values.items():
        name = "diagnostic-" + role + ".json"
        write(root / name, value)
        entry["inputs"][role] = bind(root, name)
    review = json.loads((root / "diagnostic-review.json").read_bytes())
    review["cases"][0]["inputs"] = entry["inputs"]
    write(root / "diagnostic-review.json", review)
    entry["review"] = bind(root, "diagnostic-review.json")
    seal(root, manifest)


def test_mixed_normal_project_reproduces_tokens(bundle, tmp_path):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    result = intake.prepare_reviewed_corpus(
        root / "corpus.json", model, tmp_path / "data", project=tmp_path / "project"
    )
    assert result["counts_by_kind"] == {"exact_edit": 1, "diagnostic": 1}
    assert result["explicit_same_source_views"] == [["diagnostic", "one"]]
    rows = intake.lines((tmp_path / "project/data/train.jsonl").read_bytes())
    assert rows[1] == values["prepared"]
    provenance = intake.lines((tmp_path / "data/provenance.jsonl").read_bytes())[1]
    assert "replay" not in provenance
    assert provenance["diagnostic_admission"]["semantic_truth_certified"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "unknown_ref",
        "span_hash",
        "span_bounds",
        "unbound_target",
        "overlap",
        "after_prompt",
        "wrong_source_bytes",
        "unknown_class",
        "class_mismatch",
        "unused_ref",
        "bad_range_file",
        "missing_evidence",
        "source_protected",
        "missing_multi",
        "extra_multi",
        "old_format",
    ],
)
def test_diagnostic_refusals_publish_nothing(bundle, tmp_path, damage):
    values = diagnostic(bundle)
    root, model, manifest, _ = bundle
    evidence = values["evidence"]
    claim = evidence["claims"][0]
    ref = evidence["references"][0]
    if damage == "unknown_ref":
        claim["refs"] = ["unbound"]
    elif damage == "span_hash":
        claim["sha256"] = "0" * 64
    elif damage == "span_bounds":
        claim["end"] = 999999
    elif damage == "unbound_target":
        claim["end"] -= 1
        claim["sha256"] = intake.sha(values["raw"]["messages"][1]["content"].encode()[:-1])
    elif damage == "overlap":
        evidence["claims"].append(copy.deepcopy(claim))
    elif damage == "after_prompt":
        after = values["source"]["files"]["kernel.metal"]["after"]["content"]
        (root / "after.txt").write_text(after)
        ref.update(
            kind="after_source",
            file=bind(root, "after.txt"),
            sha256=intake.sha(after.encode()),
            end=len(after.encode()),
        )
    elif damage == "wrong_source_bytes":
        ref["source_path"] = "run.sh"
    elif damage == "unknown_class":
        claim["class"] = "verified_fast"
    elif damage == "class_mismatch":
        claim["class"] = "local_gpu"
    elif damage == "unused_ref":
        extra = copy.deepcopy(ref)
        extra["id"] = "unused"
        evidence["references"].append(extra)
    elif damage == "bad_range_file":
        ref["file"]["sha256"] = "0" * 64
    elif damage == "missing_evidence":
        evidence["references"] = []
    elif damage == "source_protected":
        values["source"]["protected"] = True
    elif damage == "missing_multi":
        manifest["multi_views"] = []
    elif damage == "extra_multi":
        manifest["multi_views"] = [["one", "diagnostic"], ["diagnostic", "one"]]
    elif damage == "old_format":
        manifest["format"] = intake.FORMAT
    update(bundle, values)
    with pytest.raises(ValueError):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_same_source_cross_split_still_refused(bundle, tmp_path):
    diagnostic(bundle, split="test")
    root, model, _, _ = bundle
    with pytest.raises(ValueError, match="crosses explicit splits"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")


def test_supplied_observation_and_bound_execution_assertion(bundle, tmp_path):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    output = b"CPU counterexample: expected 1, observed 2.\n"
    (root / "cpu-output.txt").write_bytes(output)
    execution = {
        "executed": True,
        "kind": "local_cpu",
        "backend": "cpu",
        "argv": ["python", "check.py"],
        "inputs": [bind(root, "diagnostic-baseline.txt")],
        "outputs": [bind(root, "cpu-output.txt")],
        "environment": {"python": "fixture"},
    }
    write(root / "execution.json", execution)
    reference = {
        "id": "cpu",
        "kind": "local_cpu",
        "file": bind(root, "cpu-output.txt"),
        "start": 0,
        "end": len(output),
        "sha256": intake.sha(output),
        "execution": bind(root, "execution.json"),
    }
    e = values["evidence"]
    e["references"].append(reference)
    e["prompt_mode"] = "supplied_observations"
    e["prompt_spans"][0]["refs"] = ["before", "cpu"]
    e["claims"][0].update(refs=["cpu"], **{"class": "local_cpu"})
    update(bundle, values)
    result = intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    assert result["quality_certified"] is False
    assert "execution.json" in result["source_files_sha256"]
    # The importer deliberately does not authenticate the claim or execute check.py.
    assert not (root / "check.py").exists()
    execution["outputs"] = [bind(root, "diagnostic-baseline.txt")]
    write(root / "execution.json", execution)
    reference["execution"] = bind(root, "execution.json")
    update(bundle, values)
    with pytest.raises(ValueError, match="bound execution output"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "bad")


def test_arbitrary_mixed_sizes_preserve_frozen_bytes(bundle, tmp_path):
    fixtures.add_case(bundle, "second", "train")
    fixtures.add_case(bundle, "heldout", "test")
    diagnostic(bundle)
    root, model, _, _ = bundle
    result = intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    assert result["counts"] == {"train": 3, "valid": 0, "test": 1}
    for name in ("test.jsonl", "test.prepared.jsonl"):
        assert (root / name).read_bytes() == (tmp_path / "data" / name).read_bytes()
    index = json.loads((tmp_path / "data/captured-inputs.json").read_bytes())
    record = index["diagnostic-baseline.txt"]
    assert (tmp_path / "data" / record["file"]).read_bytes() == (
        root / "diagnostic-baseline.txt"
    ).read_bytes()


def test_diagnostic_alone_cannot_bypass_protected_lineage(bundle, tmp_path):
    diagnostic(bundle)
    root, model, manifest, _ = bundle
    manifest["entries"] = manifest["entries"][1:]
    manifest["multi_views"] = []
    policy = json.loads((root / "policy.json").read_bytes())
    policy["protected_lineages"] = ["change:one"]
    write(root / "policy.json", policy)
    manifest["policy"] = bind(root, "policy.json")
    seal(root, manifest)
    with pytest.raises(ValueError, match="Protected source"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")


def test_evidence_mutation_after_snapshot_prevents_publication(bundle, tmp_path, monkeypatch):
    diagnostic(bundle)
    root, model, _, _ = bundle
    import exl3_mlx.data as data

    actual = data.tokenize_row

    def mutate(row, tokenizer):
        result = actual(row, tokenizer)
        if len(row.get("messages", [])) == 2 and row["messages"][1].get("content"):
            (root / "diagnostic-baseline.txt").write_text("changed")
        return result

    monkeypatch.setattr(data, "tokenize_row", mutate)
    with pytest.raises(ValueError, match="Input changed"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_type", []),
        ("prompt_mode", {}),
        ("references", [None]),
        ("claims", [None]),
        ("prompt_spans", [False]),
    ],
)
def test_malformed_evidence_cli_refuses_cleanly(bundle, tmp_path, capsys, field, value):
    from exl3_mlx.ktune_cli import main

    values = diagnostic(bundle)
    root, model, _, _ = bundle
    values["evidence"][field] = value
    update(bundle, values)
    assert (
        main(
            [
                "prepare-reviewed",
                "--source",
                str(root / "corpus.json"),
                "--model",
                str(model),
                "--output",
                str(tmp_path / "data"),
            ]
        )
        == 1
    )
    assert "Error:" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "bad",
    [
        "missing_limits",
        "empty_limits",
        "missing_origin",
        "origin_url",
        "cpu_as_gpu",
        "gpu_backend_cpu",
        "cpu_backend_metal",
    ],
)
def test_claim_caveats_primary_identity_and_execution_backend(bundle, tmp_path, bad):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    e = values["evidence"]
    claim = e["claims"][0]
    if bad == "missing_limits":
        del claim["limits"]
    elif bad == "empty_limits":
        claim["limits"] = " "
    elif bad.startswith("origin") or bad == "missing_origin":
        (root / "report.txt").write_text("Author reports a measured change.")
        ref = {
            "id": "report",
            "kind": "author_report",
            "file": bind(root, "report.txt"),
            "start": 0,
            "end": len((root / "report.txt").read_bytes()),
            "sha256": intake.sha((root / "report.txt").read_bytes()),
        }
        if bad == "origin_url":
            ref["origin"] = {"url": "not a URL", "identity": "author commit"}
        e["references"].append(ref)
        claim.update(refs=["report"], **{"class": "author_reported"})
    else:
        output = b"measurement"
        (root / "observed.txt").write_bytes(output)
        execution = {
            "executed": True,
            "kind": "local_cpu",
            "backend": "cpu",
            "argv": ["test"],
            "inputs": [bind(root, "diagnostic-baseline.txt")],
            "outputs": [bind(root, "observed.txt")],
            "environment": {"host": "fixture"},
        }
        ref = {
            "id": "measured",
            "kind": "local_gpu",
            "file": bind(root, "observed.txt"),
            "start": 0,
            "end": len(output),
            "sha256": intake.sha(output),
        }
        if bad == "gpu_backend_cpu":
            execution["kind"] = "local_gpu"
        if bad == "cpu_backend_metal":
            ref["kind"] = "local_cpu"
            execution["backend"] = "metal"
        write(root / "execution.json", execution)
        ref["execution"] = bind(root, "execution.json")
        e["references"].append(ref)
        claim.update(refs=["measured"], **{"class": ref["kind"]})
    update(bundle, values)
    with pytest.raises(ValueError, match="limits|origin|URL|kind/backend"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    assert not (tmp_path / "data").exists()


def test_primary_author_report_positive_and_honestly_limited(bundle, tmp_path):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    text = "Author reports a measured change."
    (root / "report.txt").write_text(text)
    e = values["evidence"]
    e["references"].append(
        {
            "id": "author",
            "kind": "author_report",
            "file": bind(root, "report.txt"),
            "start": 0,
            "end": len(text.encode()),
            "sha256": intake.sha(text.encode()),
            "origin": {
                "url": "https://github.com/example/kernels/commit/" + "1" * 40,
                "identity": "Commit author; historical report",
            },
        }
    )
    e["claims"][0].update(
        refs=["author"],
        **{"class": "author_reported", "limits": "Author report only; not locally reproduced."},
    )
    update(bundle, values)
    result = intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    assert result["quality_certified"] is False


def test_ignored_raw_metadata_cannot_disguise_duplicate_prepared_view(bundle, tmp_path):
    root, model, manifest, vals = bundle
    manifest["format"] = intake.MIXED_FORMAT
    manifest["multi_views"] = [["one", "copy"]]
    entry = copy.deepcopy(manifest["entries"][0])
    entry["id"] = "copy"
    raw = copy.deepcopy(vals["raw"])
    raw["tools"][0]["function"]["description"] = (
        "A different tool description omitted by this template"
    )
    write(root / "copy-raw.json", raw)
    entry["inputs"]["raw"] = bind(root, "copy-raw.json")
    review = {
        "format": "exact-edit-review-v1",
        "status": "approved",
        "reviewer": "fixture assertion",
        "cases": [{"id": "copy", "inputs": entry["inputs"]}],
    }
    write(root / "copy-review.json", review)
    entry["review"] = bind(root, "copy-review.json")
    manifest["entries"].append(entry)
    seal(root, manifest)
    with pytest.raises(ValueError, match="Duplicate prepared training view"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")


def test_multiple_exposure_groups_deterministic_artifacts(bundle, tmp_path):
    diagnostic(bundle)
    second = fixtures.add_case(bundle, "second", "train")
    root, model, manifest, _ = bundle
    diagnostic((root, model, manifest, second), case_id="z-diagnostic", source_id="second")
    first = intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "first")
    second = intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "second")
    assert first["explicit_same_source_views"] == [
        ["diagnostic", "one"],
        ["second", "z-diagnostic"],
    ]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["artifacts_sha256"] == second["artifacts_sha256"]


@pytest.mark.parametrize("mode", ["task_instruction_only", "supplied_observations"])
def test_explicit_system_message_preserves_exact_prepared_tokens(bundle, tmp_path, mode):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    content = (
        "Explain a historical optimization and label evidence limitations."
        if mode == "task_instruction_only"
        else "Supplied baseline: float x = 1;"
    )
    values["raw"]["messages"].insert(0, {"role": "system", "content": content})
    values["evidence"]["system_mode"] = mode
    if mode == "supplied_observations":
        values["evidence"]["prompt_mode"] = mode
        values["evidence"]["system_spans"] = [text_span(content, ["before"])]
    values["prepared"] = tokenize_row(values["raw"], load_tokenizer(model))
    update(bundle, values)
    intake.prepare_reviewed_corpus(
        root / "corpus.json", model, tmp_path / "data", project=tmp_path / "project"
    )
    assert (
        intake.lines((tmp_path / "project/data/train.jsonl").read_bytes())[1] == values["prepared"]
    )
    assert (
        intake.lines((tmp_path / "data/train.jsonl").read_bytes())[1]["messages"][0]["content"]
        == content
    )
    values["evidence"].pop("system_mode")
    update(bundle, values)
    with pytest.raises(ValueError, match="system information boundary"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "bad")


def test_executable_addition_and_mode_transition_inspected_without_replay(bundle, tmp_path):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    values["source"]["files"]["new-test.sh"] = {
        "role": "changed",
        "before": None,
        "after": fixtures.blob("echo test\n", "100755"),
    }
    values["source"]["files"]["run.sh"]["after"]["mode"] = "100644"
    update(bundle, values)
    intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
    admitted = intake.lines((tmp_path / "data/provenance.jsonl").read_bytes())[1][
        "diagnostic_admission"
    ]
    assert admitted["replay_performed"] is False
    assert {"path": "new-test.sh", "before": None, "after": "100755"} in admitted[
        "mode_transitions"
    ]
    assert {"path": "run.sh", "before": "100755", "after": "100644"} in admitted["mode_transitions"]
    assert admitted["source_transition_sha256"] == intake.sha(
        intake.canonical(admitted["source_transitions"])
    )
    assert admitted["text_patch_identity_excludes_modes"] is True


@pytest.mark.parametrize("damage", ["binary", "symlink", "deleted", "pure_mode"])
def test_unsupported_diagnostic_source_transitions_fail_closed(bundle, tmp_path, damage):
    values = diagnostic(bundle)
    root, model, _, _ = bundle
    pair = values["source"]["files"]["run.sh"]
    if damage == "binary":
        pair["after"] = fixtures.blob("binary\0data", "100755")
    elif damage == "symlink":
        pair["after"]["mode"] = "120000"
    elif damage == "deleted":
        pair["after"] = None
    else:
        for p in values["source"]["files"].values():
            p["after"] = copy.deepcopy(p["before"])
        pair["after"]["mode"] = "100644"
    update(bundle, values)
    with pytest.raises(ValueError, match="Binary|regular text modes|Deleted|Pure mode"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")


def test_same_text_different_modes_cannot_cross_splits(bundle, tmp_path):
    values = diagnostic(bundle, split="test")
    root, model, _, _ = bundle
    values["source"]["files"]["run.sh"]["after"]["mode"] = "100644"
    update(bundle, values)
    with pytest.raises(ValueError, match="crosses explicit splits"):
        intake.prepare_reviewed_corpus(root / "corpus.json", model, tmp_path / "data")
