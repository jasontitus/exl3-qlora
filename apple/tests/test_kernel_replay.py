import difflib
import hashlib
import json
import subprocess
from unittest.mock import patch

import pytest

from exl3_mlx import kernel_replay as r

OLD = b"float x = 1;\nfloat y = x;\n"
NEW = b"float x = 2;\nfloat y = x;\n"
PATCH = "".join(
    difflib.unified_diff(
        OLD.decode().splitlines(True),
        NEW.decode().splitlines(True),
        fromfile="a/kernel.metal",
        tofile="b/kernel.metal",
    )
)


def command(patch=PATCH, dash=True):
    return (
        "git apply --whitespace=nowarn " + ("- " if dash else "") + "<<'PATCH'\n" + patch + "PATCH"
    )


def native(cmd):
    return ("A bounded edit.\n\n" + r.OPEN + cmd + r.CLOSE).encode()


@pytest.mark.parametrize("dash", [False, True])
@pytest.mark.parametrize(
    "headers", ["", "diff --git a/kernel.metal b/kernel.metal\nindex 123..456 100644\n"]
)
def test_equivalent_common_stdin_forms_replay_without_shell(dash, headers):
    with patch.object(subprocess, "Popen", side_effect=AssertionError("shell forbidden")):
        _, _, diff = r.parse_native(native(command(headers + PATCH, dash)))
        after, changed = r.apply_exact(diff, {"kernel.metal": OLD}, required_paths=["kernel.metal"])
    assert after == {"kernel.metal": NEW} and changed == ["kernel.metal"]


@pytest.mark.parametrize(
    "bad",
    [
        lambda c: c + "; touch /tmp/not-allowed",
        lambda c: c.replace("git apply", "sh -c git apply"),
        lambda c: c.replace("<<'PATCH'", "<<PATCH"),
        lambda c: c.replace("git apply", "git apply --unsafe-paths"),
        lambda c: c.replace("+float x = 2;\n", "PATCH\n+float x = 2;\n"),
    ],
)
def test_no_shell_language_or_early_heredoc(bad):
    with pytest.raises(ValueError):
        r.parse_command(bad(command()))


def test_no_recount_fuzz_or_path_escape():
    for bad in (
        PATCH.replace("@@ -1,2 +1,2 @@", "@@ -1,2 +1,20 @@"),
        PATCH.replace("float x = 1;", "float x = 3;"),
        PATCH.replace("kernel.metal", "../kernel.metal"),
        PATCH.replace("@@ -1,2 +1,2 @@", "@@ -1,2 +2,2 @@"),
    ):
        with pytest.raises(ValueError):
            r.apply_exact(bad, {"kernel.metal": OLD})


def test_duplicate_file_and_multiple_native_calls():
    with pytest.raises(ValueError):
        r.apply_exact(PATCH + PATCH, {"kernel.metal": OLD})
    with pytest.raises(ValueError):
        r.parse_native(native(command()) + native(command()))


def test_dot_is_not_a_file_name():
    with pytest.raises(ValueError, match="Unsafe patch path"):
        r.safe_path(".")


def test_real_file_cli_positive_negative_and_no_overwrite(tmp_path):
    baseline = tmp_path / "kernel.metal"
    baseline.write_bytes(OLD)
    generation = tmp_path / "generation.raw"
    generation.write_bytes(native(command(dash=False)))

    def pin(p):
        return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    spec = tmp_path / "input.json"
    spec.write_text(
        json.dumps(
            {
                "generation": pin(generation),
                "baseline": {"kernel.metal": pin(baseline)},
                "required_paths": ["kernel.metal"],
            }
        )
    )
    sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    with patch.object(subprocess, "Popen", side_effect=AssertionError("shell forbidden")):
        out = r.run(spec, sha, tmp_path / "result")
    assert out["stages"]["replay"]["status"] == "passed"
    assert (tmp_path / "result/candidate/kernel.metal").read_bytes() == NEW
    with pytest.raises(FileExistsError):
        r.run(spec, sha, tmp_path / "result")
    generation.write_bytes(native(command(PATCH.replace("+1,2", "+1,20"), False)))
    data = json.loads(spec.read_text())
    data["generation"] = pin(generation)
    spec.write_text(json.dumps(data))
    out = r.run(spec, hashlib.sha256(spec.read_bytes()).hexdigest(), tmp_path / "failure")
    assert out["stages"]["raw_format"]["status"] == "passed"
    assert out["stages"]["replay"]["status"] == "failed"
    assert "Hunk count mismatch" in out["stages"]["replay"]["reason"]
    assert not out["shell_executed"] and not out["compiler_executed"]
