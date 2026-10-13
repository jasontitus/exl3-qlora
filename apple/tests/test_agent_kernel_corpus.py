"""CPU-only publication, manifest trust and conservative split-overlap checks."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "agent_corpus_builder",
    Path(__file__).resolve().parents[1] / "scripts/build_agent_kernel_corpus.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def manifest(path, *, identity="case-train", split="train", **changes):
    url = (
        "https://github.com/ashhart/TensorFold"
        if split == "train"
        else "https://github.com/ggml-org/llama.cpp"
    )
    candidate = {
        "id": identity,
        "repository": "repo",
        "protected": split == "test",
        "split": split,
        "outcome": "reported_improvement",
        "task_kind": {"value": "kernel"},
        **changes,
    }
    raw = {
        "format": builder.FORMAT,
        "repositories": {"repo": {"url": url}},
        "candidates": [candidate],
    }
    path.write_bytes(builder.dump(builder.with_manifest_checksum(raw)))
    return path


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    train = manifest(tmp_path / "train-source.json")
    test = manifest(tmp_path / "test-source.json", identity="case-test", split="test")
    monkeypatch.setattr(builder, "load_tokenizer", lambda _: object())
    monkeypatch.setattr(builder, "tokenizer_identity", lambda _: {"tokenizer": "fixed"})
    monkeypatch.setattr(builder, "_evidence", lambda *_: ({}, {}))

    def reconstruct(c, repos, evidence):
        return {
            "id": c["id"],
            "repository": repos["repo"]["url"],
            "lineage_keys": ["repo:" + repos["repo"]["url"]],
            "commits": [c["id"]],
            "stable_patch_id": c["id"],
            "reviewed_context": {"units": {}},
        }

    monkeypatch.setattr(builder, "reconstruct_candidate", reconstruct)
    monkeypatch.setattr(
        builder,
        "agent_example",
        lambda record: (
            {"messages": [{"role": "user", "content": record["id"]}]},
            {"full_commit_replay": True},
        ),
    )
    monkeypatch.setattr(
        builder, "tokenize_row", lambda *_: {"input_ids": [1, 2, 3, 4], "loss_start": 2}
    )
    return train, test, tmp_path / "output"


def run(paths, **kwargs):
    train, test, out = paths
    return builder.build_corpus([train], [test], "mock-model", out, **kwargs)


def test_fresh_complete_publication_and_artifact_hashes(prepared):
    report = run(prepared)
    out = prepared[2]
    assert report["counts"] == {"train": 1, "valid": 0, "test": 1}
    assert report["max_length"] == 4
    for name, digest in report["artifacts_sha256"].items():
        assert builder.sha((out / name).read_bytes()) == digest
    assert (out / "valid.jsonl").read_bytes() == b""
    assert not list(out.parent.glob(".output.pending-*"))
    assert {Path(name).name for name in report["helper_source_sha256"]} == {
        "agent_edits.py",
        "data.py",
        "tool_data.py",
        "upstream_data.py",
        "ktune_data.py",
        "context.py",
    }
    assert all(
        builder.sha(Path(name).read_bytes()) == digest
        for name, digest in report["helper_source_sha256"].items()
    )


@pytest.mark.parametrize(
    "identity", ["../escape", "/absolute", "nested/name", "x.json", "..", "", "a" * 121, "nul\x00x"]
)
def test_unsafe_ids_never_escape_or_publish(prepared, identity):
    manifest(prepared[0], identity=identity)
    with pytest.raises(ValueError, match="Case IDs"):
        run(prepared)
    assert not prepared[2].exists()
    assert not (prepared[2].parent / "escape.provenance.json").exists()


@pytest.mark.parametrize("identity", ["case-train", "CASE-TRAIN"])
def test_case_ids_unique_across_manifests_and_case_insensitive(prepared, identity):
    manifest(prepared[1], identity=identity, split="test")
    with pytest.raises(ValueError, match="Duplicate case ID"):
        run(prepared)
    assert not prepared[2].exists()


def test_duplicate_manifest_rejected_before_processing(prepared):
    with pytest.raises(ValueError, match="only once"):
        builder.build_corpus([prepared[0], prepared[0]], [prepared[1]], "mock", prepared[2])


@pytest.mark.parametrize("change", ["missing", "wrong", "changed"])
def test_manifest_checksum_required_and_exact(prepared, change):
    value = json.loads(prepared[0].read_bytes())
    if change == "missing":
        value.pop("manifest_sha256")
    elif change == "wrong":
        value["manifest_sha256"] = "0" * 64
    else:
        value["candidates"][0]["outcome"] = "source_only"
    prepared[0].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="canonical manifest_sha256"):
        run(prepared)
    assert not prepared[2].exists()


def test_checksum_helper_does_not_mutate_or_write():
    source = {"a": ["é"], "manifest_sha256": "wrong"}
    old = copy.deepcopy(source)
    result = builder.with_manifest_checksum(source)
    assert source == old and result["manifest_sha256"] != "wrong"
    assert result["manifest_sha256"] == builder.sha(b'{"a":["\xc3\xa9"]}')


@pytest.mark.parametrize(
    "changes", [{"protected": False}, {"protected": 1}, {"split": "train"}, {"split": "valid"}]
)
def test_test_protection_and_declared_split_are_binding(prepared, changes):
    path = prepared[1]
    value = json.loads(path.read_bytes())
    value["candidates"][0].update(changes)
    path.write_bytes(builder.dump(builder.with_manifest_checksum(value)))
    with pytest.raises(ValueError):
        run(prepared)
    assert not prepared[2].exists()


def test_duplicate_json_keys_rejected(prepared):
    prepared[0].write_text('{"format":"x","format":"y"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        run(prepared)


@pytest.mark.parametrize(
    "changes", [{"outcome": "negative"}, {"task_kind": {"value": "integration"}}]
)
def test_nonpositive_or_unreviewed_task_not_published(prepared, changes):
    manifest(prepared[0], **changes)
    with pytest.raises(ValueError, match="nonempty train"):
        run(prepared)
    assert not prepared[2].exists()
    assert not list(prepared[2].parent.glob(".output.pending-*"))


def test_row_failure_does_not_leave_partial_success_directory(prepared, monkeypatch):
    monkeypatch.setattr(
        builder, "agent_example", lambda _: (_ for _ in ()).throw(ValueError("bad patch"))
    )
    with pytest.raises(ValueError, match="nonempty"):
        run(prepared)
    assert not prepared[2].exists()
    assert not list(prepared[2].parent.glob(".output.pending-*"))


def test_existing_output_never_replaced(prepared):
    prepared[2].mkdir()
    sentinel = prepared[2] / "mine"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        run(prepared)
    assert sentinel.read_text() == "keep"


def test_symlink_output_never_followed(prepared):
    target = prepared[2].parent / "target"
    target.mkdir()
    prepared[2].symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        run(prepared)
    assert list(target.iterdir()) == []


def test_atomic_publish_racing_existing_empty_directory_is_not_replaced(prepared, monkeypatch):
    publish = builder._publish

    def race(stage, out):
        out.mkdir()
        inode = out.stat().st_ino
        with pytest.raises(FileExistsError):
            publish(stage, out)
        assert out.stat().st_ino == inode and list(out.iterdir()) == []
        raise FileExistsError("race")

    monkeypatch.setattr(builder, "_publish", race)
    with pytest.raises(FileExistsError):
        run(prepared)
    assert not list(prepared[2].parent.glob(".output.pending-*"))


def test_manifest_drift_before_publication_rejected(prepared, monkeypatch):
    def drift(record):
        prepared[0].write_bytes(prepared[0].read_bytes() + b" ")
        return {"messages": []}, {}

    monkeypatch.setattr(builder, "agent_example", drift)
    with pytest.raises(ValueError, match="changed during build"):
        run(prepared)
    assert not prepared[2].exists()


def test_tokenizer_drift_before_publication_rejected(prepared, monkeypatch):
    calls = iter([{"version": 1}, {"version": 2}])
    monkeypatch.setattr(builder, "tokenizer_identity", lambda _: next(calls))
    with pytest.raises(ValueError, match="Tokenizer changed"):
        run(prepared)
    assert not prepared[2].exists()


@pytest.mark.parametrize("length", [True, 0, 2, 32769, 3.2])
def test_context_ceiling_is_validated(prepared, length):
    with pytest.raises(ValueError, match="max_length"):
        run(prepared, max_length=length)


def code_record(identity, split, *, delta="-", unchanged=False):
    before = "\n".join(f"value_{i} += input_{i} * 3;" for i in range(80))
    after = before if unchanged else before.replace("+=", delta + "=")
    return {
        "id": identity,
        "split": split,
        "source": {
            "reviewed_context": {
                "units": {
                    "kernel.metal": [
                        {
                            "kind": "complete_kernel",
                            "before_content": before,
                            "after_content": after,
                        }
                    ]
                }
            }
        },
    }


def test_identical_long_edited_unit_is_cross_split_blocker():
    report = builder.cross_split_code_report([code_record("a", "train"), code_record("b", "test")])
    assert report["automatic_blocker"] and report["exact_repeated_edited_units"]


def test_shared_baseline_shingles_reported_not_automatically_excluded():
    report = builder.cross_split_code_report(
        [code_record("a", "train"), code_record("b", "test", delta="/")]
    )
    assert not report["automatic_blocker"]
    assert report["near_duplicate_pairs"][0]["shared_64_token_shingles"] >= 4


def test_ubiquitous_unchanged_helper_does_not_block():
    report = builder.cross_split_code_report(
        [code_record("a", "train", unchanged=True), code_record("b", "test", unchanged=True)]
    )
    assert not report["automatic_blocker"] and not report["near_duplicate_pairs"]


def test_code_overlap_blocker_prevents_publication(prepared, monkeypatch):
    monkeypatch.setattr(
        builder,
        "cross_split_code_report",
        lambda _: {"automatic_blocker": True, "exact_repeated_edited_units": ["same edit"]},
    )
    with pytest.raises(ValueError, match="exact repeated"):
        run(prepared)
    assert not prepared[2].exists()


def policy_file(prepared, value):
    path = prepared[2].parent / "documentation-policy.json"
    path.write_bytes(builder.dump(value) + b"\n")
    return path


def omission(**changes):
    return {
        "before_sha256": "a" * 64,
        "after_sha256": "b" * 64,
        "reason": "Historical performance announcement is provenance, not a requested edit.",
        **changes,
    }


def test_explicit_policy_forwarded_default_empty_and_exact_copy_hashed(prepared, monkeypatch):
    policy = {"case-train": {"docs/performance.md": omission(before_sha256=None)}}
    path = policy_file(prepared, policy)
    original = path.read_bytes()
    received = {}

    def example(record, *, documentation_omissions):
        received[record["id"]] = documentation_omissions
        return {"messages": []}, {
            "full_commit_replay": False,
            "exact_selected_file_replay": True,
            "target_scope": "complete_non_documentation_commit_edits",
        }

    monkeypatch.setattr(builder, "agent_example", example)
    report = run(prepared, documentation_omissions=path)
    assert received == {"case-train": policy["case-train"], "case-test": {}}
    assert (prepared[2] / "documentation-omissions.json").read_bytes() == original
    assert report["documentation_omissions"]["sha256"] == builder.sha(original)
    assert report["documentation_omissions"]["declared_paths"] == 1
    assert report["artifacts_sha256"]["documentation-omissions.json"] == builder.sha(original)
    audit = json.loads((prepared[2] / "audit.json").read_text())
    assert all(x["replay"]["full_commit_replay"] is False for x in audit)
    assert path.read_bytes() == original


def test_empty_policy_is_opt_in_not_default_none(prepared, monkeypatch):
    path = policy_file(prepared, {})
    seen = []

    def example(record, *, documentation_omissions):
        seen.append(documentation_omissions)
        return {"messages": []}, {}

    monkeypatch.setattr(builder, "agent_example", example)
    report = run(prepared, documentation_omissions=path)
    assert seen == [{}, {}]
    assert report["documentation_omissions"]["declared_cases"] == 0


def test_default_omission_policy_leaves_old_api_and_report_unchanged(prepared):
    # Fixture's one-argument agent_example would reject even a None keyword.
    report = run(prepared)
    assert "documentation_omissions" not in report
    assert not (prepared[2] / "documentation-omissions.json").exists()
    assert json.loads((prepared[2] / "audit.json").read_text())[0]["replay"]["full_commit_replay"]


@pytest.mark.parametrize(
    "value", [[], None, 3, {"foreign-case": {}}, {"CASE-TRAIN": {}}, {"case-train": []}]
)
def test_malformed_or_foreign_document_policy_fails_globally(prepared, value, monkeypatch):
    path = policy_file(prepared, value)
    monkeypatch.setattr(
        builder, "load_tokenizer", lambda _: pytest.fail("must fail before processing")
    )
    with pytest.raises(ValueError, match="Documentation"):
        run(prepared, documentation_omissions=path)
    assert not prepared[2].exists()


@pytest.mark.parametrize(
    "path",
    [
        "../README.md",
        "/README.md",
        "docs//x.md",
        "./README.md",
        "docs/../README.md",
        ".git/a.md",
        "docs/a.metal",
        "docs/tests.cpp",
        "CMakeLists.txt",
        "docs/a.md\n",
        "",
    ],
)
def test_document_policy_cannot_omit_source_or_escape(prepared, path):
    policy = policy_file(prepared, {"case-train": {path: omission()}})
    with pytest.raises(ValueError, match="safe .md/.rst"):
        run(prepared, documentation_omissions=policy)
    assert not prepared[2].exists()


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        {},
        omission(before_sha256="wrong"),
        omission(after_sha256=None),
        omission(reason=" \n"),
        omission(reason=1),
        omission(extra="ignored?"),
        omission(before_sha256=True),
    ],
)
def test_document_policy_requires_exact_hash_schema_and_reason(prepared, declaration):
    policy = policy_file(prepared, {"case-train": {"README.md": declaration}})
    with pytest.raises(ValueError, match="Documentation omission"):
        run(prepared, documentation_omissions=policy)
    assert not prepared[2].exists()


@pytest.mark.parametrize(
    "changes",
    [{"outcome": "negative"}, {"outcome": "source_only"}, {"task_kind": {"value": "integration"}}],
)
def test_policy_cannot_name_nonpositive_or_ineligible_case(prepared, changes):
    manifest(prepared[0], **changes)
    path = policy_file(prepared, {"case-train": {"README.md": omission()}})
    with pytest.raises(ValueError, match="positive eligible"):
        run(prepared, documentation_omissions=path)


def test_duplicate_document_policy_keys_rejected(prepared):
    path = policy_file(prepared, {})
    path.write_text('{"case-train":{},"case-train":{}}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        run(prepared, documentation_omissions=path)


def test_policy_drift_aborts_atomic_publication(prepared, monkeypatch):
    path = policy_file(prepared, {})

    def example(record, *, documentation_omissions):
        path.write_bytes(path.read_bytes() + b" ")
        return {"messages": []}, {}

    monkeypatch.setattr(builder, "agent_example", example)
    with pytest.raises(ValueError, match="changed during build"):
        run(prepared, documentation_omissions=path)
    assert not prepared[2].exists()
    assert not list(prepared[2].parent.glob(".output.pending-*"))


def test_renderer_helper_drift_aborts_before_publication(prepared, monkeypatch):
    helper = prepared[2].parent / "renderer.py"
    helper.write_text("version=1\n")
    monkeypatch.setattr(
        builder, "_helper_source_identity", lambda: {str(helper): builder.sha(helper.read_bytes())}
    )

    def example(record):
        helper.write_text("version=2\n")
        return {"messages": []}, {}

    monkeypatch.setattr(builder, "agent_example", example)
    with pytest.raises(ValueError, match="changed during build"):
        run(prepared)
    assert not prepared[2].exists()
    assert not list(prepared[2].parent.glob(".output.pending-*"))


def test_source_hash_or_document_coverage_error_is_audited_per_row(prepared, monkeypatch):
    m = json.loads(prepared[0].read_bytes())
    m["candidates"].append({**m["candidates"][0], "id": "second-train"})
    prepared[0].write_bytes(builder.dump(builder.with_manifest_checksum(m)))
    path = policy_file(prepared, {"case-train": {"README.md": omission()}})

    def example(record, *, documentation_omissions):
        if record["id"] == "case-train":
            raise ValueError("Documentation source hash or complete coverage mismatch")
        assert documentation_omissions == {}
        return {"messages": []}, {"exact_selected_file_replay": True}

    monkeypatch.setattr(builder, "agent_example", example)
    report = run(prepared, documentation_omissions=path)
    assert report["counts"] == {"train": 1, "test": 1, "valid": 0}
    assert json.loads((prepared[2] / "exclusions.json").read_text()) == [
        {
            "id": "case-train",
            "split": "train",
            "reason": "Documentation source hash or complete coverage mismatch",
        }
    ]
