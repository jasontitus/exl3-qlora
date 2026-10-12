"""CPU-only archive import and snapshot-boundary tests; no model or GPU."""

import gzip
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src/exl3_mlx"
PACKAGE = "ktune_data_cpu_tests"
namespace = types.ModuleType(PACKAGE)
namespace.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = namespace
spec = importlib.util.spec_from_file_location(f"{PACKAGE}.ktune_data", ROOT / "ktune_data.py")
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


PATCH = "diff --git a/kernel.metal b/kernel.metal\nindex 1111111..2222222 100644\n--- a/kernel.metal\n+++ b/kernel.metal\n@@ -1,2 +1,2 @@\n-float old = x;\n+float optimized = x;\n return old;\n"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source, model = tmp_path / "source", tmp_path / "model"
    source.mkdir()
    model.mkdir()
    write(model / "config.json", {"vocab_size": 300})

    def tokenize(pair, tokenizer):
        prompt, target = pair["prompt"].encode(), pair["completion"].encode()
        return {
            "input_ids": [0, *[b + 4 for b in prompt], 2, *[b + 4 for b in target], 1],
            "loss_start": len(prompt) + 2,
        }

    def validate(row, *, seq_length, vocab_size, truncate):
        assert truncate is False
        assert 2 <= seq_length <= 32768
        assert len(row["input_ids"]) <= seq_length + 1
        assert 1 <= row["loss_start"] < len(row["input_ids"])
        assert all(0 <= token < vocab_size for token in row["input_ids"])
        return row

    monkeypatch.setattr(
        data,
        "_helpers",
        lambda: (lambda _: object(), tokenize, lambda _: {"tokenizer.json": "fixture"}, validate),
    )
    monkeypatch.setattr(
        data,
        "_official_audit",
        lambda row: {
            "valid": row.get("verified", True),
            "errors": [] if row.get("verified", True) else ["needs evidence equivalence"],
        },
    )
    return source, model, tmp_path / "output"


def official(source, entries=None, track="test-track"):
    entries = entries or [
        {
            "id": "aaaaaaaa-1",
            "status": "accepted",
            "createdAt": "2026-09-10T01:00:00Z",
            "submissionCommitSha": "a" * 40,
            "promotedSourceRef": "a" * 40,
            "note": "FUTURE SECRET: optimized achieved 200x",
        }
    ]
    folder = source / "library/official" / track
    (folder / "promoted").mkdir(parents=True)
    with gzip.open(folder / "all-submissions.json.gz", "wb") as stream:
        stream.write(json.dumps(entries).encode())
    accepted = sorted(
        [r for r in entries if r["status"] == "accepted"], key=lambda r: r["createdAt"]
    )
    for number, row in enumerate(accepted, 1):
        if not row.get("missing_diff"):
            (folder / "promoted" / f"{number:02d}-{row['id'][:8]}.diff").write_text(
                row.get("diff", PATCH)
            )
    return folder


def snapshot(source, entries):
    for row in entries:
        row["diff_sha256"] = data._sha(row["diff"].encode())
        row["example_sha256"] = data._sha(data._canonical(row))
    value = {"schema_version": 1, "purpose": "retrieval_only", "examples": entries, "excluded": []}
    value["snapshot_sha256"] = data._sha(data._canonical(value))
    write(source / "library/optimizations.json", value)


def historical(identity="historical", tier="historical_promoted", **changes):
    return {
        "id": identity,
        "diff": PATCH,
        "edit_format": "unified_diff",
        "outcome": tier,
        "evidence_level": "historical_scored_engine",
        "provenance": {"repo": "other-repo", "track": "other-track", "base_commit": "b" * 40},
        "admission": {"split": "train", "full_edit_training_admitted": False},
        **changes,
    }


def test_complete_patch_and_full_length_audit_are_preserved_without_future_notes(setup):
    source, model, output = setup
    official(source)
    report = data.import_ktune(source, model, output)
    row = rows(output / "train.jsonl")[0]
    assert row["completion"] == PATCH
    assert "float old = x;" in row["prompt"]
    assert "float optimized" not in row["prompt"]
    assert "FUTURE SECRET" not in row["prompt"]
    audit = rows(output / "length-audit.jsonl")[0]
    assert audit["total_tokens"] == audit["prompt_tokens"] + audit["target_tokens"]
    assert audit["truncated"] is False
    assert report["max_length"] == audit["total_tokens"]
    assert report["prompt_context"] == "target_selected_preimage_hunks"
    assert report["quality_evaluation_available"] is False
    assert report["counts"] == {"train": 1, "valid": 0, "test": 0}
    assert not (output / "valid.jsonl").read_text()
    assert not (output / "test.jsonl").read_text()
    assert report["artifacts_sha256"]["train.jsonl"] == data._file_sha(output / "train.jsonl")


def test_overlength_is_ledgered_without_partial_target_or_silent_truncation(setup):
    source, model, output = setup
    official(source)
    report = data.import_ktune(source, model, output, max_length=512)
    assert report["ready_for_sft"] is False
    assert report["length_audit"]["over_limit"] == 1
    assert report["length_audit"]["truncated"] == 0
    assert rows(output / "train.jsonl") == []
    assert "full_serialized_example_exceeds_context" in rows(output / "ledger.jsonl")[0]["reasons"]


def test_default_evidence_tiers_preserve_prior_retrieval_limits_and_never_select_negative(setup):
    source, model, output = setup
    snapshot(
        source,
        [historical(), historical("source", "source_only"), historical("negative", "negative")],
    )
    report = data.import_ktune(source, model, output)
    assert report["counts"]["train"] == 1
    assert rows(output / "train.jsonl")[0]["case_id"] == "snapshot:historical"
    provenance = rows(output / "provenance.jsonl")[0]
    assert provenance["evidence"]["admission"]["full_edit_training_admitted"] is False
    assert {r["tier"] for r in rows(output / "ledger.jsonl")} == {"source_only", "negative"}


def test_source_only_requires_explicit_tier_selection(setup):
    source, model, output = setup
    snapshot(source, [historical(tier="source_only")])
    report = data.import_ktune(source, model, output, tiers=("source_only",))
    assert report["counts"]["train"] == 1
    assert report["evidence_counts"] == {"source_only": 1}


@pytest.mark.parametrize("change", [{"verified": False}, {"missing_diff": True}])
def test_unbound_official_evidence_or_missing_diff_remains_nonnegative_ledger(setup, change):
    source, model, output = setup
    official(
        source,
        [{"id": "aaaaaaaa-1", "status": "accepted", "createdAt": "2026-09-10T01:00:00Z", **change}],
    )
    report = data.import_ktune(source, model, output)
    assert report["counts"]["train"] == 0
    assert len(rows(output / "ledger.jsonl")) == 1
    assert rows(output / "ledger.jsonl")[0]["tier"] != "negative"


def test_entire_track_remains_one_lineage_and_shortest_pilot_is_explicit(setup):
    source, model, output = setup
    official(
        source,
        [
            {
                "id": "aaaaaaaa-1",
                "status": "accepted",
                "createdAt": "2026-09-10T01:00:00Z",
                "diff": PATCH.replace("optimized", "long_name_" * 15),
            },
            {
                "id": "bbbbbbbb-2",
                "status": "accepted",
                "createdAt": "2026-09-11T01:00:00Z",
                "diff": PATCH,
            },
        ],
    )
    report = data.import_ktune(source, model, output, max_examples=1)
    assert report["counts"]["train"] == 1
    assert report["selected_case_ids"] == ["official:test-track:bbbbbbbb-2"]
    assert len({r["source_id"] for r in rows(output / "provenance.jsonl")}) == 1
    assert any(
        "explicit_shortest_complete_pilot_limit" in r.get("reasons", [])
        for r in rows(output / "ledger.jsonl")
    )


def test_exact_case_whitelist_is_preserved_in_manifest(setup):
    source, model, output = setup
    official(
        source,
        [
            {"id": "aaaaaaaa-1", "status": "accepted", "createdAt": "2026-09-10T01:00:00Z"},
            {
                "id": "bbbbbbbb-2",
                "status": "accepted",
                "createdAt": "2026-09-11T01:00:00Z",
                "diff": PATCH.replace("optimized", "alternative"),
            },
        ],
    )
    wanted = ["official:test-track:bbbbbbbb-2"]
    report = data.import_ktune(source, model, output, selected_case_ids=wanted)
    assert report["selected_case_ids"] == wanted
    assert report["requested_case_ids"] == wanted
    assert rows(output / "train.jsonl")[0]["case_id"] == wanted[0]
    assert "not_in_explicit_case_selection" in rows(output / "ledger.jsonl")[0]["reasons"]


@pytest.mark.parametrize("selection", [["missing"], ["x", "x"], [], "x"])
def test_invalid_or_unknown_case_selection_rejected(setup, selection):
    source, model, output = setup
    official(source)
    with pytest.raises(ValueError):
        data.import_ktune(source, model, output, selected_case_ids=selection)
    assert not output.exists()


def test_transferred_snapshot_manifest_detects_modified_official_archive(setup):
    source, model, output = setup
    official(source)
    write(
        source / "snapshot-manifest.json",
        {"format": "exl3-mlx-ktune-snapshot-v1", "files_sha256": {}},
    )
    with pytest.raises(ValueError, match="snapshot manifest"):
        data.import_ktune(source, model, output)
    assert not output.exists()


def test_cutoff_quarantines_entire_lineage_not_just_future_revision(setup):
    source, model, output = setup
    official(
        source,
        [
            {"id": "aaaaaaaa-1", "status": "accepted", "createdAt": "2026-09-10T01:00:00Z"},
            {"id": "bbbbbbbb-2", "status": "accepted", "createdAt": "2026-09-12T01:00:00Z"},
        ],
    )
    report = data.import_ktune(source, model, output, train_until="2026-09-11T00:00:00Z")
    assert report["counts"]["train"] == 0
    assert all(
        "lineage_unknown_or_after_cutoff" in r["reasons"] for r in rows(output / "provenance.jsonl")
    )


def test_protected_and_duplicate_edits_union_across_repositories(setup):
    source, model, output = setup
    snapshot(
        source,
        [
            historical(),
            historical("protected", provenance={"repo": "unrelated"}, admission={"split": "test"}),
        ],
    )
    report = data.import_ktune(source, model, output)
    assert report["counts"]["train"] == 0
    assert all("protected_lineage" in r["reasons"] for r in rows(output / "provenance.jsonl"))


def test_corrupt_snapshot_digest_rejected_before_tokenization(setup):
    source, model, output = setup
    snapshot(source, [historical()])
    path = source / "library/optimizations.json"
    value = json.loads(path.read_text())
    value["examples"][0]["diff"] += "changed"
    write(path, value)
    with pytest.raises(ValueError, match="Modified"):
        data.import_ktune(source, model, output)
    assert not output.exists()


def test_source_symlink_escape_rejected(setup, tmp_path):
    source, model, output = setup
    outside = tmp_path / "outside.json"
    write(outside, {})
    (source / "library").mkdir()
    (source / "library/optimizations.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        data.import_ktune(source, model, output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_length": True},
        {"max_length": 32769},
        {"max_examples": 0},
        {"max_examples": True},
        {"tiers": ("negative",)},
        {"tiers": "official_promoted"},
        {"context_mode": "parent_files"},
        {"train_until": "2026-09-10"},
    ],
)
def test_invalid_options_fail_before_output(setup, kwargs):
    source, model, output = setup
    with pytest.raises(ValueError):
        data.import_ktune(source, model, output, **kwargs)
    assert not output.exists()


def test_search_replace_context_excludes_replacement():
    target = "```edit kernel.metal\n<<<<<<< SEARCH\nold();\n=======\nnew();\n>>>>>>> REPLACE\n```"
    context, paths = data.preimage_context(target, "search_replace")
    assert "old();" in context and "new();" not in context
    assert paths == ["kernel.metal"]


@pytest.mark.parametrize(
    "patch",
    [
        PATCH.replace("@@ -1,2 +1,2 @@", "@@ -1,3 +1,2 @@"),
        PATCH.replace("kernel.metal", "../escape"),
        "GIT binary patch\n",
        "diff --git a/a b/a\nold mode 100644\nnew mode 100755\n",
        PATCH + "+extra code\n",
    ],
)
def test_incomplete_unsafe_or_nontext_patch_rejected(patch):
    with pytest.raises(ValueError):
        data.preimage_context(patch)


@pytest.mark.parametrize(
    "suffix",
    [
        "diff --git a/g.metal b/g.metal\n@@ -1 +1 @@\n-old2\n+new2\n",
        "diff --git a/g.metal b/g.metal\nold mode 100644\nnew mode 100755\n",
        "diff --git a/g.metal b/g.metal\nsimilarity index 100%\nrename from g.metal\nrename to h.metal\n",
        "Post-outcome arbitrary prose\n",
    ],
)
def test_valid_first_hunk_cannot_hide_malformed_or_unsupported_later_sections(suffix):
    with pytest.raises(ValueError):
        data.preimage_context(PATCH + suffix)


def archive(path, entries):
    with tarfile.open(path, "w:gz") as stream:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            if kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = value
                stream.addfile(member)
            else:
                member.size = len(value)
                stream.addfile(member, io.BytesIO(value))


@pytest.mark.parametrize(
    "entries",
    [
        [("../escape", "file", b"x")],
        [(".env", "file", b"x")],
        [("library/official/token", "symlink", "/private/secret")],
        [("README.md", "file", b"x"), ("README.md", "file", b"y")],
        [("runs/build.bin", "file", b"x")],
    ],
)
def test_snapshot_rejects_archive_traversal_secrets_links_duplicates_and_unlisted_paths(
    tmp_path, entries
):
    path, output = tmp_path / "a.tar.gz", tmp_path / "out"
    output.mkdir()
    archive(path, entries)
    with pytest.raises(ValueError, match="Unsafe"):
        data._extract_snapshot(path, output, max_bytes=1024)
    assert list(output.iterdir()) == []


def test_snapshot_expansion_bound_precedes_writing(tmp_path):
    path, output = tmp_path / "a.tar.gz", tmp_path / "out"
    output.mkdir()
    archive(path, [("README.md", "file", b"12345")])
    with pytest.raises(ValueError, match="bound"):
        data._extract_snapshot(path, output, max_bytes=4)
    assert list(output.iterdir()) == []


def test_snapshot_argv_allowlist_and_atomic_publish(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, ("a" * 40 + "\n").encode(), b"")

    def download(host, root, path, *, max_bytes):
        assert host == "macbook" and root == "/data/ktune"
        assert max_bytes == 64 << 20
        archive(path, [("README.md", "file", b"evidence")])

    monkeypatch.setattr(data.subprocess, "run", run)
    monkeypatch.setattr(data, "_download_snapshot", download)
    report = data.snapshot_ktune("macbook:/data/ktune", tmp_path / "out")
    assert calls[0][:4] == ["ssh", "-o", "BatchMode=yes", "macbook"]
    assert report["remote_revision"] == "a" * 40
    assert (tmp_path / "out/README.md").read_text() == "evidence"
    assert not list(tmp_path.glob(".ktune-snapshot-*"))
    with pytest.raises(FileExistsError):
        data.snapshot_ktune("macbook:/data/ktune", tmp_path / "out")


def test_complete_parent_files_mode_binds_diff_to_real_git_history(setup):
    source, model, output = setup
    repo = source / "parent"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for key, value in (("user.name", "CPU Test"), ("user.email", "cpu@example.invalid")):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    file = repo / "kernel.metal"
    file.write_text("// COMPLETE FILE PREAMBLE\nfloat old = x;\nreturn old;\n")
    subprocess.run(["git", "-C", str(repo), "add", "kernel.metal"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "parent"], check=True)
    file.write_text("// COMPLETE FILE PREAMBLE\nfloat optimized = x;\nreturn old;\n")
    subprocess.run(["git", "-C", str(repo), "commit", "-qam", "optimized"], check=True)
    head = data._git(repo, "rev-parse", "HEAD").decode().strip()
    patch = data._git(repo, "diff", "HEAD^", "HEAD").decode()
    official(
        source,
        [
            {
                "id": "aaaaaaaa-1",
                "status": "accepted",
                "createdAt": "2026-09-10T01:00:00Z",
                "submissionCommitSha": head,
                "promotedSourceRef": head,
                "diff": patch,
            }
        ],
    )
    report = data.import_ktune(source, model, output, context_mode="parent_files", parent_repo=repo)
    assert report["counts"]["train"] == 1
    assert "// COMPLETE FILE PREAMBLE" in rows(output / "train.jsonl")[0]["prompt"]
    assert report["prompt_context"] == "target_selected_complete_parent_files"
    assert "parent_source_binding" in rows(output / "provenance.jsonl")[0]


def test_search_replace_parent_binding_rejects_mismatched_head(monkeypatch):
    parent, head = "a" * 40, "b" * 40
    target = "```edit kernel.metal\n<<<<<<< SEARCH\nold();\n=======\nnew();\n>>>>>>> REPLACE\n```"
    record = {
        "head_commit": head,
        "parent_commit": parent,
        "edit_format": "search_replace",
        "diff": target,
    }

    def git(repo, *args):
        if args[0] == "diff":
            return b"kernel.metal\n"
        if args[0] == "ls-tree":
            return b"100644 blob hash\tkernel.metal\n"
        if args == ("show", parent + ":kernel.metal"):
            return b"old();\n"
        if args == ("show", head + ":kernel.metal"):
            return b"different();\n"
        pytest.fail(f"Unexpected Git request: {args}")

    monkeypatch.setattr(data, "_git", git)
    with pytest.raises(ValueError, match="differs from pinned head"):
        data._parent_context(record, ["kernel.metal"], Path("unused"))


def test_snapshot_download_uses_bounded_argv_and_cleans_child_on_overflow(tmp_path, monkeypatch):
    popen = subprocess.Popen
    seen = []

    def fake_ssh(command, **kwargs):
        seen.append(command)
        return popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x'*100); sys.stdout.flush()",
            ],
            **kwargs,
        )

    monkeypatch.setattr(data.subprocess, "Popen", fake_ssh)
    with pytest.raises(ValueError, match="compressed byte bound"):
        data._download_snapshot("macbook", "/data/ktune", tmp_path / "partial.tar.gz", max_bytes=16)
    assert seen[0][:4] == ["ssh", "-o", "BatchMode=yes", "macbook"]
    assert " .env" not in seen[0][4]
    assert "library/official" in seen[0][4]
    assert "library/corpus" not in seen[0][4]


def test_remote_git_query_preserves_argument_quoting_and_read_only_boundary(monkeypatch):
    import shlex

    seen = []
    monkeypatch.setattr(
        data,
        "_bounded_query",
        lambda command, **kwargs: seen.append((command, kwargs)) or b"source",
    )
    reference = "a" * 40 + ":kernel;name.metal"
    assert data._git("macbook:/source root/repo", "show", reference) == b"source"
    command, options = seen[0]
    assert command[:4] == ["ssh", "-o", "BatchMode=yes", "macbook"]
    assert shlex.split(command[4]) == [
        "git",
        "--no-pager",
        "-C",
        "/source root/repo",
        "show",
        reference,
    ]
    assert options["max_bytes"] == 16 << 20
    with pytest.raises(ValueError, match="read-only"):
        data._git("macbook:/source/repo", "checkout", "HEAD")


def test_parent_source_query_enforces_output_bound_without_model():
    with pytest.raises(ValueError, match="byte bound"):
        data._bounded_query(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x'*100)"], max_bytes=8
        )


@pytest.mark.parametrize(
    "remote",
    ["-oProxyCommand=bad:/root", "host:relative", "host:/root\ncommand", "host with spaces:/root"],
)
def test_unsafe_remote_address_rejected(remote):
    with pytest.raises(ValueError):
        data._remote_parts(remote)
