import hashlib
import subprocess

import pytest

from exl3_mlx.agent_edits import parse_command, render_command, replay, scoped_images, unified_patch


def test_git_patch_roundtrip_and_omitted_edit_preserved():
    before = {"src/kernel.metal": b"// unchanged\nfloat x = 1;\nfloat y = 2;\n"}
    after = {"src/kernel.metal": b"// unchanged\nfloat x = 3;\nfloat y = 2;\n"}
    command = render_command(unified_patch(before, after))
    assert parse_command(command).startswith("--- a/src/kernel.metal")
    assert replay(command, before, after)["exact_scoped_replay"]
    with pytest.raises(ValueError, match="after-image"):
        replay(command, before, {"src/kernel.metal": b"wrong"})


@pytest.mark.parametrize("suffix", ["\nrm -rf x", "; echo injected", "\n", "\nPATCH_x"])
def test_never_executes_extra_shell(suffix):
    with pytest.raises(ValueError):
        parse_command(render_command("--- a/x\n+++ b/x\n") + suffix)


def test_rejects_undeclared_path():
    command = render_command("--- a/../x\n+++ b/../x\n@@ -1 +1 @@\n-a\n+b\n")
    with pytest.raises(ValueError):
        replay(command, {"x": b"a\n"}, {"x": b"b\n"})


def test_baseline_mismatch_fails_apply():
    patch = unified_patch({"x": b"original\n"}, {"x": b"changed\n"})
    with pytest.raises(subprocess.CalledProcessError):
        replay(render_command(patch), {"x": b"different\n"}, {"x": b"changed\n"})


def test_scoped_images_ignore_omitted_commit_changes():
    old = b"abc\ndef\n"
    new = b"ABC\nDEF\n"
    unit = {
        "before": {"start": 0, "end": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
        "after": {"sha256": hashlib.sha256(b"ABC").hexdigest()},
        "after_content": "ABC",
    }
    record = {
        "files": {"x": {"before": {"content": old.decode()}, "after": {"content": new.decode()}}},
        "reviewed_context": {"units": {"x": [unit]}},
    }
    before, after = scoped_images(record)
    assert before["x"] == old and after["x"] == b"ABC\ndef\n"


def test_new_file_roundtrip():
    before = {"src/new.metal": None}
    after = {"src/new.metal": b"kernel void test() {}\n"}
    assert replay(render_command(unified_patch(before, after)), before, after)[
        "exact_scoped_replay"
    ]


def test_delimiter_collision_uses_small_suffix():
    patch = "--- a/x\n+++ b/x\nKERNEL_PATCH\n"
    command = render_command(patch)
    assert "<<'KERNEL_PATCH_1'" in command
    assert parse_command(command) == patch


def _full_record(old, new, before_mode="100644", after_mode="100644"):
    return {
        "files": {
            "src/kernel.metal": {
                "role": "changed",
                "before": None if old is None else {"mode": before_mode, "content": old},
                "after": None if new is None else {"mode": after_mode, "content": new},
            }
        }
    }


@pytest.mark.parametrize(
    "old,new,bmode,amode",
    [
        ("same\n", "same\n", "100644", "100755"),
        ("before\n", "after\n", "100755", "100644"),
        (None, "new\n", "100644", "100755"),
    ],
)
def test_full_commit_rejects_mode_changes_even_without_text_diff(old, new, bmode, amode):
    from exl3_mlx.agent_edits import full_commit_images

    with pytest.raises(ValueError, match="mode"):
        full_commit_images(_full_record(old, new, bmode, amode))


def test_full_commit_rejects_deleted_file():
    from exl3_mlx.agent_edits import full_commit_images

    with pytest.raises(ValueError, match="Deleted"):
        full_commit_images(_full_record("old\n", None))


def test_full_commit_retains_unselected_host_change():
    from exl3_mlx.agent_edits import full_commit_images

    record = _full_record("old kernel\n", "new kernel\n")
    record["files"]["host.cpp"] = {
        "role": "changed",
        "before": {"mode": "100644", "content": "old dispatch\n"},
        "after": {"mode": "100644", "content": "new dispatch\n"},
    }
    record["reviewed_context"] = {"units": {"src/kernel.metal": []}}
    before, after = full_commit_images(record)
    assert set(after) == {"src/kernel.metal", "host.cpp"}
    assert replay(render_command(unified_patch(before, after)), before, after)[
        "exact_scoped_replay"
    ]


def test_full_commit_empty_new_file_unsupported_fail_closed():
    from exl3_mlx.agent_edits import full_commit_images

    with pytest.raises(ValueError, match="Empty-file additions"):
        full_commit_images(_full_record(None, ""))


def _documentation_record():
    record = _full_record("old kernel\n", "new kernel\n")
    for path in ("host.cpp", "tests/test.cpp", "CMakeLists.txt", "docs/measurements.md"):
        record["files"][path] = {
            "role": "changed",
            "before": {"mode": "100644", "content": "before\n"},
            "after": {"mode": "100644", "content": "after\n"},
        }
    record["annotations"] = {"operation": {"text": "test operation"}}
    record["reviewed_context"] = {"units": {}}
    review = {
        "docs/measurements.md": {
            "before_sha256": hashlib.sha256(b"before\n").hexdigest(),
            "after_sha256": hashlib.sha256(b"after\n").hexdigest(),
            "reason": "Reviewed non-runtime historical benchmark narrative",
        }
    }
    return record, review


def test_documentation_omissions_keep_every_source_host_build_test_edit():
    from exl3_mlx.agent_edits import agent_example

    record, review = _documentation_record()
    row, receipt = agent_example(record, review)
    command = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["cmd"]
    for path in ("src/kernel.metal", "host.cpp", "tests/test.cpp", "CMakeLists.txt"):
        assert path in command
    assert "docs/measurements.md" not in str(row)
    assert receipt["omitted_documentation"] == review
    assert receipt["exact_selected_file_replay"] is True
    assert receipt["full_commit_replay"] is False
    assert receipt["target_scope"] == "complete_commit_edits_except_reviewed_documentation"
    assert agent_example(record)[1]["full_commit_replay"] is True


@pytest.mark.parametrize("change", ["missing", "unknown", "source", "hash", "reason"])
def test_documentation_omissions_fail_closed(change):
    from exl3_mlx.agent_edits import agent_example

    record, review = _documentation_record()
    if change == "missing":
        review = {}
    elif change == "unknown":
        review["docs/new.md"] = dict(review["docs/measurements.md"])
    elif change == "source":
        review["host.cpp"] = dict(review["docs/measurements.md"])
    elif change == "hash":
        review["docs/measurements.md"]["after_sha256"] = "0" * 64
    else:
        review["docs/measurements.md"]["reason"] = ""
    with pytest.raises(ValueError):
        agent_example(record, review)


@pytest.mark.parametrize(
    "old,new", [("old", "new"), ("old\n", "new"), ("old", "new\n"), ("old\n", "")]
)
def test_complete_bytes_without_terminal_newline_or_empty_result(old, new):
    from exl3_mlx.agent_edits import full_commit_images

    before, after = full_commit_images(_full_record(old, new))
    assert replay(render_command(unified_patch(before, after)), before, after)[
        "exact_scoped_replay"
    ]


def test_empty_file_addition_explicitly_refused():
    from exl3_mlx.agent_edits import full_commit_images

    with pytest.raises(ValueError, match="Empty-file additions"):
        full_commit_images(_full_record(None, ""))


def test_unchanged_executable_mode_remains_source_bound():
    from exl3_mlx.agent_edits import full_commit_images

    before, after = full_commit_images(_full_record("echo old\n", "echo new\n", "100755", "100755"))
    assert replay(render_command(unified_patch(before, after)), before, after)[
        "exact_scoped_replay"
    ]
    with pytest.raises(ValueError, match="mode changes"):
        full_commit_images(_full_record("old\n", "new\n", "100644", "100755"))
