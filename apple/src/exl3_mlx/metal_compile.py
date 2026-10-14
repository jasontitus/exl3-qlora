"""Compile a reviewed, self-contained Metal source pair without running GPU code.

This deliberately narrow gate accepts translation units with metal_stdlib as
their only include. Inline MLX strings, project headers and host integration need
their actual build context; this command never claims to validate those paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

FORMAT = "exl3-metal-compile-pair-v1"
STANDARDS = ("metal3.1", "metal4.0")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _without_comments(text):
    """Preserve quoted literals/newlines while blanking C-style comments."""
    if re.search(r'\b(?:u8|u|U|L)?R"', text):
        raise ValueError("Raw string literals are unsupported by this narrow compile gate")
    result, index, quote = [], 0, None
    while index < len(text):
        char = text[index]
        if quote:
            result.append(char)
            if char == "\\" and index + 1 < len(text):
                index += 1
                result.append(text[index])
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            result.append(char)
        elif text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            result.append(" " * (end - index))
            index = end - 1
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("Unterminated source comment")
            end += 2
            result.append("".join("\n" if c == "\n" else " " for c in text[index:end]))
            index = end - 1
        else:
            result.append(char)
        index += 1
    return "".join(result)


def _source(path):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix != ".metal" or not path.is_file() or path.stat().st_size > 8 << 20:
        raise ValueError("Expected a .metal translation unit of at most 8 MiB")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    if not text.strip() or "\x00" in text:
        raise ValueError("Expected nonempty UTF-8 Metal source without NULs")
    # Normalize preprocessor line splicing before checking include directives.
    # This is a supported-format restriction, not a sandbox for hostile shaders.
    if re.search(r"\?\?[=/'()!<>-]", text):
        raise ValueError("Alternate preprocessor directive spellings are unsupported")
    logical = text.replace("\r\n", "\n").replace("\r", "\n")
    logical = re.sub(r"\\\n", "", logical)
    logical = _without_comments(logical)
    if "%:" in logical or "??=" in logical:
        raise ValueError("Alternate preprocessor directive spellings are unsupported")
    logical = logical.replace("\f", " ").replace("\v", " ")
    directives = re.findall(r"^[ \t]*#[ \t]*(\w+)([^\n]*)", logical, re.M)
    includes = []
    for name, body in directives:
        if name in {"include", "include_next", "import", "embed"}:
            if name != "include" or body.strip() != "<metal_stdlib>":
                raise ValueError(
                    "Only #include <metal_stdlib> is supported; use the project build for other dependencies"
                )
            includes.append(body.strip())
    if not includes:
        raise ValueError("A self-contained unit must include <metal_stdlib>")
    return path, raw


def _run(argv, *, timeout=180):
    started = time.time()
    env = os.environ.copy()
    # Do not let user shell include flags silently change the source closure.
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "SDKROOT"):
        env.pop(key, None)
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout, env=env)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
        error = None
    except subprocess.TimeoutExpired as exc:
        code, stdout, stderr = None, exc.stdout or b"", exc.stderr or b""
        error = "compiler_timeout"
    except OSError as exc:
        code, stdout, stderr = None, b"", str(exc).encode()
        error = "tool_unavailable"
    return {
        "argv": argv,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "exit_code": code,
        "error": error,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


def compile_pair(before, after, output, *, standard="metal3.1"):
    """Snapshot both complete inputs and retain successes and failures in a receipt."""
    if standard not in STANDARDS:
        raise ValueError(f"standard must be one of {STANDARDS}")
    inputs = {label: _source(path) for label, path in (("before", before), ("after", after))}
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "format": FORMAT,
        "scope": "self_contained_translation_unit_pair_compile_only",
        "standard": standard,
        "host": platform.platform(),
        "script_sha256": _sha(Path(__file__).read_bytes()),
        "toolchain": _run(["xcrun", "--sdk", "macosx", "metal", "--version"]),
        "sdk": _run(["xcrun", "--sdk", "macosx", "--show-sdk-path"]),
        "sources": {},
        "compiles": {},
        "numerical_parity": "not_run",
        "performance": "not_run",
        "execution_verified": False,
        "limitations": [
            "AIR compilation does not prove GPU execution, numerical parity or speed.",
            "The fixed validation flags are not asserted to match original project flags.",
            "Only this source pair is checked; surrounding host code or other changed files are not.",
            "SDK compilation does not establish support on the runtime GPU.",
            "Receipts record local results; checksums are not external authentication.",
        ],
    }
    for label, (original, raw) in inputs.items():
        snapshot = output / f"{label}.metal"
        snapshot.write_bytes(raw)
        report["sources"][label] = {
            "original_path": str(original),
            "snapshot_path": str(snapshot),
            "sha256": _sha(raw),
            "bytes": len(raw),
        }
    for label, (_, raw) in inputs.items():
        source = output / f"{label}.metal"
        air = output / f"{label}.air"
        result = _run(
            [
                "xcrun",
                "--sdk",
                "macosx",
                "metal",
                f"-std={standard}",
                "-O3",
                "-fno-fast-math",
                f"-fmodules-cache-path={output / 'module-cache'}",
                "-c",
                str(source),
                "-o",
                str(air),
            ]
        )
        result["source_unchanged"] = source.read_bytes() == raw
        result["air_sha256"] = _sha(air.read_bytes()) if air.is_file() else None
        result["air_bytes"] = air.stat().st_size if air.is_file() else 0
        result["passed"] = (
            result["exit_code"] == 0
            and result["error"] is None
            and result["source_unchanged"]
            and bool(result["air_sha256"])
            and result["air_bytes"] > 0
        )
        report["compiles"][label] = result
    report["all_snapshots_unchanged"] = all(
        (output / f"{label}.metal").read_bytes() == raw for label, (_, raw) in inputs.items()
    )
    report["compiled"] = (
        all(run["passed"] for run in report["compiles"].values())
        and all(report[name]["exit_code"] == 0 for name in ("toolchain", "sdk"))
        and report["all_snapshots_unchanged"]
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["receipt_sha256"] = _sha(canonical)
    (output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    import sys

    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == "closure":
        from .metal_compile_closure import main as closure_main

        return closure_main(args_in[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument(
        "--output", required=True, help="New directory for sources, AIR and receipt"
    )
    parser.add_argument("--standard", choices=STANDARDS, default="metal3.1")
    args = parser.parse_args(argv)
    try:
        report = compile_pair(args.before, args.after, args.output, standard=args.standard)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Compile input error: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["compiled"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
