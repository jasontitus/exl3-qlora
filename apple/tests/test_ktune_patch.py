"""Real CPU Git apply/replay tests; no compiler, model, network or GPU usage."""

import difflib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SOURCE))
_SPEC = importlib.util.spec_from_file_location(
    "exl3_mlx.ktune_patch_cpu_tests", _SOURCE / "exl3_mlx/ktune_patch.py"
)
patch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patch)


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return (
        subprocess.run(
            [
                "git",
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "commit.gpgsign=false",
                "-C",
                str(repo),
                *args,
            ],
            env=env,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "-c", "init.templateDir=", "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "kernel.metal").write_text("float kernel(float x) {\n    return x * 2;\n}\n")
    (repo / "helper.h").write_text("#define TILE 16\n")
    git(repo, "add", "kernel.metal", "helper.h")
    git(repo, "commit", "-qm", "pinned parent")
    return repo, git(repo, "rev-parse", "HEAD")


def diff(name, before, after):
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="a/" + name,
            tofile="b/" + name,
        )
    )


def convert(tmp_path, repository, text, **kwargs):
    source = tmp_path / "generated.diff"
    source.write_text(text)
    return patch.export_ktune_edit(*repository, source, tmp_path / "candidate.edit", **kwargs)


def snapshot(repo):
    return {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in repo.rglob("*")
        if p.is_file()
    }


def test_pinned_source_conversion_ignores_dirty_checkout_and_writes_no_repo_files(
    tmp_path, repository
):
    repo, _ = repository
    original = (repo / "kernel.metal").read_text()
    changed = original.replace("x * 2", "x + x")
    (repo / "kernel.metal").write_text("Unrelated uncommitted work.\n")
    before = snapshot(repo)
    result = convert(
        tmp_path,
        repository,
        diff("kernel.metal", original, changed),
        allowed_paths=["kernel.metal"],
    )
    assert snapshot(repo) == before
    edit = Path(result["output"]).read_text()
    assert patch.replay_full_file_edit({"kernel.metal": original}, edit) == {
        "kernel.metal": changed
    }
    assert result["replay_equal"] and result["source_payload_verified"]
    assert not result["compiled"] and not result["quality_certified"]
    assert result["files"]["kernel.metal"]["required_editable_range"] == [1, 3]
    assert result["output_edit_sha256"] == hashlib.sha256(edit.encode()).hexdigest()
    assert json.loads(Path(result["receipt"]).read_text())["parent_revision"] == repository[1]


def test_multiple_existing_files_replay_exactly(tmp_path, repository):
    repo, _ = repository
    before = {name: (repo / name).read_text() for name in ("kernel.metal", "helper.h")}
    after = {
        "kernel.metal": before["kernel.metal"].replace("x * 2", "x + x"),
        "helper.h": "#define TILE 32\n",
    }
    result = convert(tmp_path, repository, "".join(diff(n, before[n], after[n]) for n in before))
    assert patch.replay_full_file_edit(before, Path(result["output"]).read_text()) == after


@pytest.mark.parametrize(
    "tail",
    [
        "diff --git a/a b/a\nnew file mode 100644\n--- /dev/null\n+++ b/a\n@@ -0,0 +1 @@\n+new\n",
        "diff --git a/helper.h b/helper.h\nold mode 100644\nnew mode 100755\n",
        "diff --git a/helper.h b/renamed.h\nsimilarity index 100%\nrename from helper.h\nrename to renamed.h\n",
        "GIT binary patch\nliteral 0\nHcmV?d00001\n",
        "diff --git a/helper.h b/helper.h\n@@ -1 +1 @@\n-old\n+new\n",
    ],
)
def test_unsupported_mixed_sections_are_rejected_before_output(tmp_path, repository, tail):
    before = (repository[0] / "kernel.metal").read_text()
    with pytest.raises(ValueError):
        convert(
            tmp_path,
            repository,
            diff("kernel.metal", before, before.replace("x * 2", "x+x")) + tail,
        )
    assert not (tmp_path / "candidate.edit").exists()


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        ".git/config",
        "has space",
        "a\\b",
        "a/kernel.metal",
        "b/kernel.metal",
    ],
)
def test_unsafe_paths_rejected(tmp_path, repository, name):
    with pytest.raises(ValueError):
        convert(tmp_path, repository, diff(name, "old\n", "new\n"))


def test_unknown_or_disallowed_file_rejected(tmp_path, repository):
    with pytest.raises(ValueError):
        convert(tmp_path, repository, diff("absent.metal", "old\n", "new\n"))
    before = (repository[0] / "kernel.metal").read_text()
    with pytest.raises(ValueError, match="outside allowed_paths"):
        convert(
            tmp_path,
            repository,
            diff("kernel.metal", before, before.replace("x * 2", "x+x")),
            allowed_paths=["helper.h"],
        )


def test_wrong_preimage_and_truncated_patch_rejected(tmp_path, repository):
    with pytest.raises(ValueError):
        convert(tmp_path, repository, diff("kernel.metal", "wrong parent\n", "new\n"))
    with pytest.raises(ValueError):
        convert(
            tmp_path,
            repository,
            "--- a/kernel.metal\n+++ b/kernel.metal\n@@ -1,3 +1,3 @@\n-old\n+new\n",
        )


def test_symlink_and_submodule_are_not_regular_source(tmp_path, repository):
    repo, _ = repository
    (repo / "link").symlink_to("helper.h")
    git(repo, "add", "link")
    git(repo, "commit", "-qm", "symlink fixture")
    revision = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="regular files"):
        convert(tmp_path, (repo, revision), diff("link", "helper.h\n", "other.h\n"))
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{revision},module")
    git(repo, "commit", "-qm", "gitlink fixture")
    with pytest.raises(ValueError, match="regular files"):
        convert(tmp_path, (repo, git(repo, "rev-parse", "HEAD")), diff("module", "old\n", "new\n"))


def test_converter_does_not_execute_repository_filters(tmp_path, repository):
    repo, _ = repository
    marker = tmp_path / "FILTER_EXECUTED"
    (repo / ".gitattributes").write_text("*.metal filter=unsafe\n")
    git(repo, "config", "filter.unsafe.clean", "touch " + str(marker))
    git(repo, "config", "filter.unsafe.smudge", "touch " + str(marker))
    before = (repo / "kernel.metal").read_text()
    convert(tmp_path, repository, diff("kernel.metal", before, before.replace("x * 2", "x+x")))
    assert not marker.exists()


def test_short_branch_ref_and_output_inside_source_repository_rejected(tmp_path, repository):
    before = (repository[0] / "kernel.metal").read_text()
    text = diff("kernel.metal", before, before.replace("x * 2", "x+x"))
    with pytest.raises(ValueError, match="pinned full"):
        convert(tmp_path, (repository[0], "HEAD"), text)
    source = tmp_path / "source.diff"
    source.write_text(text)
    with pytest.raises(ValueError, match="outside"):
        patch.export_ktune_edit(*repository, source, repository[0] / "candidate.edit")


def test_existing_output_never_overwritten(tmp_path, repository):
    output = tmp_path / "candidate.edit"
    output.write_text("preserve")
    before = (repository[0] / "kernel.metal").read_text()
    with pytest.raises(FileExistsError):
        convert(tmp_path, repository, diff("kernel.metal", before, before.replace("x * 2", "x+x")))
    assert output.read_text() == "preserve"


@pytest.mark.parametrize("delimiter", ["=======", "=======   ", ">>>>>>> REPLACE extra"])
def test_protocol_delimiter_collision_rejected(tmp_path, repository, delimiter):
    before = (repository[0] / "kernel.metal").read_text()
    after = "// malformed candidate\n" + delimiter + "\n" + before
    with pytest.raises(ValueError):
        convert(tmp_path, repository, diff("kernel.metal", before, after))


def test_whole_file_protocol_byte_limit(tmp_path, repository, monkeypatch):
    before = (repository[0] / "kernel.metal").read_text()
    monkeypatch.setattr(patch, "_MAX_EDIT", 10)
    with pytest.raises(ValueError, match="250,000"):
        convert(tmp_path, repository, diff("kernel.metal", before, before.replace("x * 2", "x+x")))
