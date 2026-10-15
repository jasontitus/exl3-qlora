"""Replay a bounded tool edit without executing generated shell commands.

Accepts quoted git-apply heredocs with an optional stdin dash and ordinary
``diff --git`` headers. Hunk counts, coordinates and context remain exact:
no fuzz, offsets, recounting, source extraction, or patch repair.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

from .evaluation_audit import _json
from .evaluation_receipts import _read

MAX_BYTES = 256 * 1024
OPEN = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n<｜DSML｜ parameter name="cmd" string="true">'
CLOSE = "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_path(value):
    require(isinstance(value, str), "Patch path must be a string")
    p = PurePosixPath(value)
    require(
        value != "."
        and re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        and not p.is_absolute()
        and str(p) == value
        and not any(x in (".", "..", ".git") for x in p.parts),
        "Unsafe patch path",
    )
    return value


def parse_command(command):
    require(
        isinstance(command, str)
        and len(command.encode()) <= MAX_BYTES
        and "\r" not in command
        and "\0" not in command,
        "Invalid command encoding/size",
    )
    match = re.fullmatch(
        r"git apply(?: --whitespace=nowarn)?(?: -)? <<([\'\"])([A-Za-z_][A-Za-z0-9_]{0,63})\1\n([\s\S]+\n)\2",
        command,
    )
    require(match is not None, "Expected one literal quoted git-apply heredoc")
    require(match[2] not in match[3].splitlines(), "Early heredoc delimiter")
    return match[3]


def parse_native(raw):
    require(type(raw) is bytes and len(raw) <= MAX_BYTES, "Generation exceeds limit")
    text = raw.decode("utf-8")
    require(
        "\r" not in text and "\0" not in text and text.count(OPEN) == 1 and text.endswith(CLOSE),
        "Expected one complete native tool call and terminal EOS",
    )
    rationale, body = text.split(OPEN)
    require(
        not rationale or (rationale.endswith("\n\n") and len(rationale.encode()) <= 16384),
        "Invalid rationale boundary",
    )
    command = body[: -len(CLOSE)]
    for part in (rationale, command):
        require(
            not any(x in part for x in ("｜DSML｜", "<｜", "</｜", "<think>", "</think>")),
            "Unexpected native control markup",
        )
    return rationale.removesuffix("\n\n"), command, parse_command(command)


def apply_exact(patch, baseline, *, required_paths=()):
    require(isinstance(baseline, dict) and 0 < len(baseline) <= 16, "Invalid baseline inventory")
    for name, raw in baseline.items():
        safe_path(name)
        require(
            type(raw) is bytes and len(raw) <= MAX_BYTES and b"\0" not in raw,
            "Invalid baseline bytes",
        )
    require(set(required_paths) <= set(baseline), "Required edit path missing from baseline")
    require(isinstance(patch, str) and len(patch.encode()) <= MAX_BYTES, "Patch exceeds limit")
    lines = patch.splitlines(keepends=True)
    after, changed, i = dict(baseline), [], 0
    while i < len(lines):
        git_name = None
        if lines[i].startswith("diff --git "):
            m = re.fullmatch(r"diff --git a/(\S+) b/(\S+)\n", lines[i])
            require(m is not None and m[1] == m[2], "Unsupported Git diff header")
            git_name = safe_path(m[1])
            i += 1
            if i < len(lines) and lines[i].startswith("index "):
                require(
                    re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100(?:644|755))?\n", lines[i]),
                    "Unsupported Git index header",
                )
                i += 1
        require(i < len(lines), "Missing file header")
        m = re.fullmatch(r"--- a/(\S+)\n", lines[i])
        require(m is not None, "Expected existing-file patch header")
        name = safe_path(m[1])
        i += 1
        require(
            name in baseline and name not in changed and (git_name is None or git_name == name),
            "Unknown/duplicate/mismatched patch path",
        )
        require(i < len(lines) and lines[i] == "+++ b/" + name + "\n", "Mismatched after path")
        changed.append(name)
        i += 1
        old = baseline[name].decode("utf-8").splitlines(keepends=True)
        cursor, output, hunks = 0, [], 0
        while i < len(lines) and not lines[i].startswith(("--- a/", "diff --git ")):
            h = re.fullmatch(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: [^\n]*)?\n", lines[i])
            require(h is not None, "Invalid hunk header")
            start, nold, newstart, nnew = int(h[1]), int(h[2] or 1), int(h[3]), int(h[4] or 1)
            oldpos, newpos = start - 1 if nold else start, newstart - 1 if nnew else newstart
            require(cursor <= oldpos <= len(old), "Overlapping/out-of-range hunk")
            output.extend(old[cursor:oldpos])
            cursor = oldpos
            require(len(output) == newpos, "Incorrect new hunk coordinate")
            i += 1
            seenold = seennew = 0
            while i < len(lines) and not lines[i].startswith(("@@ ", "--- a/", "diff --git ")):
                line = lines[i]
                i += 1
                require(line.endswith("\n") and line[:1] in (" ", "+", "-"), "Invalid hunk line")
                content = line[1:]
                if line[0] in (" ", "-"):
                    require(
                        cursor < len(old) and old[cursor] == content,
                        "Exact baseline context mismatch",
                    )
                    cursor += 1
                    seenold += 1
                if line[0] in (" ", "+"):
                    output.append(content)
                    seennew += 1
            require(
                (seenold, seennew) == (nold, nnew),
                f"Hunk count mismatch: declared {nold}/{nnew}, observed {seenold}/{seennew}",
            )
            hunks += 1
        require(hunks > 0, "Empty file patch")
        output.extend(old[cursor:])
        data = "".join(output).encode()
        require(
            data and len(data) <= MAX_BYTES and b"\0" not in data and data != baseline[name],
            "Empty/oversized/no-op candidate",
        )
        after[name] = data
    require(changed and set(required_paths) <= set(changed), "Required source edit missing")
    return after, changed


def run(spec_path, expected_sha256, output):
    spec_path = Path(spec_path).absolute()
    raw, identity = _read(spec_path, 1 << 20)
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Specification changed")
    spec = _json(raw)
    require(
        set(spec) == {"generation", "baseline", "required_paths"}, "Unexpected replay specification"
    )
    pins = {}

    def bound(ref):
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "Invalid reference")
        path = Path(ref["path"])
        path = path if path.is_absolute() else spec_path.parent / path
        data, ident = _read(path.absolute(), MAX_BYTES)
        require(hashlib.sha256(data).hexdigest() == ref["sha256"], "Input hash mismatch")
        if path in pins:
            require(pins[path] == (data, ident), "Repeated input changed")
        pins[path] = data, ident
        return data

    generation = bound(spec["generation"])
    baseline = {safe_path(name): bound(ref) for name, ref in spec["baseline"].items()}
    out = Path(output).absolute()
    require(out.parent.resolve(strict=True) == out.parent, "Noncanonical output parent")
    out.mkdir()
    (out / "generation.raw").write_bytes(generation)
    report = {
        "format": "kernel-eval-replay-v1",
        "protocol": "safe-common-git-apply-exact-v1",
        "input_sha256": expected_sha256,
        "generation_sha256": hashlib.sha256(generation).hexdigest(),
        "stages": {},
        "shell_executed": False,
        "compiler_executed": False,
        "GPU_executed": False,
    }
    try:
        rationale, command, patch = parse_native(generation)
        report["stages"]["raw_format"] = {"status": "passed"}
        report["rationale"] = rationale
        (out / "patch.diff").write_text(patch)
        after, changed = apply_exact(patch, baseline, required_paths=spec["required_paths"])
        (out / "candidate").mkdir()
        for name, data in after.items():
            target = out / "candidate" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        report["changed_paths"] = changed
        report["after_sha256"] = {
            name: hashlib.sha256(data).hexdigest() for name, data in after.items()
        }
        report["stages"]["replay"] = {"status": "passed"}
    except (ValueError, UnicodeError) as exc:
        stage = "replay" if "raw_format" in report["stages"] else "raw_format"
        report["stages"][stage] = {
            "status": "failed",
            "outcome_class": "model_output",
            "reason": str(exc),
        }
        if stage == "raw_format":
            report["stages"]["replay"] = {"status": "not_run"}
    require(_read(spec_path, 1 << 20) == (raw, identity), "Late specification drift")
    for path, captured in pins.items():
        require(_read(path.absolute(), MAX_BYTES) == captured, "Late input drift")
    with (out / "replay.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.input, args.sha256, args.output)
    print(json.dumps(report))
    return 0 if report["stages"]["replay"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
