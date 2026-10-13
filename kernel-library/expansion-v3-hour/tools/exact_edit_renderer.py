"""Isolated renderer supporting exact non-newline-terminated Git text blobs.

Never alters source bytes to make an example fit. Only the unified-diff control
line documents a missing terminal newline, per Git patch syntax. Existing
strict tool grammar and replay validation remain in use. No upstream module
or active training runtime is modified.
"""

import difflib
import importlib.util
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
SOURCE = WORK / "loss-fix/apple/src/exl3_mlx/agent_edits.py"
spec = importlib.util.spec_from_file_location(
    "_isolated_historical_edit_renderer", SOURCE
)
_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_impl)


def full_commit_images(record):
    before, after = {}, {}
    for path, pair in record["files"].items():
        if pair.get("role", "changed") != "changed":
            continue
        _impl._path(path)
        # Match the existing addition/edit contract; deleted files need separate support.
        if pair["after"] is None:
            raise ValueError("Deletion not supported by this renderer")
        if pair["after"]["mode"] != "100644" or (
            pair["before"] is not None and pair["before"]["mode"] != "100644"
        ):
            raise ValueError("Only regular non-executable text modes supported")
        old = pair["before"]["content"].encode() if pair["before"] is not None else None
        new = pair["after"]["content"].encode()
        if old == new:
            continue
        if b"\0" in (old or b"") or b"\0" in new:
            raise ValueError("Binary source unsupported")
        before[path], after[path] = old, new
    if not before:
        raise ValueError("No complete commit text edit")
    return before, after


def unified_patch(before, after):
    output = []
    for p in sorted(before):
        _impl._path(p)
        lines = difflib.unified_diff(
            (before[p] or b"").decode().splitlines(keepends=True),
            after[p].decode().splitlines(keepends=True),
            fromfile="a/" + p if before[p] is not None else "/dev/null",
            tofile="b/" + p,
            n=3,
        )
        for line in lines:
            if line.endswith("\n"):
                output.append(line)
            else:
                output.extend([line + "\n", "\\ No newline at end of file\n"])
    return "".join(output)


# Private module instance only; original import and installed runtime stay untouched.
_impl.full_commit_images = full_commit_images
_impl.unified_patch = unified_patch
agent_example = _impl.agent_example
replay = _impl.replay
parse_command = _impl.parse_command
render_command = _impl.render_command
