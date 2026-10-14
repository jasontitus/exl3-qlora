"""Bounded compilation of reviewed LOCAL Metal closures; no repository execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import posixpath
import re
import stat
import subprocess
import time
from pathlib import Path

from .metal_compile import STANDARDS, _without_comments

FORMAT = "exl3-metal-local-closure-v1"
MAX_UNIT = 16 << 20
MAX_TOTAL = 64 << 20
METAL_HEADERS = frozenset(
    (
        "metal_stdlib metal_math metal_simdgroup metal_simdgroup_matrix "
        "metal_atomic metal_compute metal_texture metal_types metal_memory metal_common "
        "metal_integer metal_relational metal_geometric metal_graphics metal_pack metal_logging"
    ).split()
)
MPP = "MetalPerformancePrimitives/MetalPerformancePrimitives.h"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def bad(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("Invalid " + label + " fields")


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA-256")
    return value


def _name(value):
    if (
        not isinstance(value, str)
        or not value
        or value.startswith("/")
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or any(p in {"", ".", ".."} for p in value.split("/"))
    ):
        raise ValueError("Noncanonical closure path")
    return value


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _read(path, limit):
    """No symlinks/nonregular files, bounded reads, matching open/path identities."""
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("Symlink or noncanonical file path: " + str(path))
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("Nonregular or oversized file: " + str(path))
    with path.open("rb") as f:
        opened = os.fstat(f.fileno())
        raw = f.read(limit + 1)
        after = os.fstat(f.fileno())
    if (
        len(raw) > limit
        or _identity(before) != _identity(opened)
        or _identity(opened) != _identity(after)
        or _identity(after) != _identity(path.lstat())
    ):
        raise ValueError("File changed while reading: " + str(path))
    return raw


def _includes(raw, name, files, mpp):
    text = raw.decode("utf-8")
    if "\x00" in text:
        raise ValueError("NUL in source")
    if re.search(r"\?\?[=/'()!<>-]", text):
        raise ValueError("Alternate preprocessor spelling")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if re.search(r"\\[ \t\f\v]+\n", text):
        raise ValueError("Whitespace-separated preprocessor line splice is unsupported")
    text = re.sub(r"\\\n", "", text)
    text = _without_comments(text).replace("\f", " ").replace("\v", " ")
    if (
        "%:" in text
        or "__has_include" in text
        or re.search(r"\b(?:import|module)\s+[A-Za-z_:]", text)
    ):
        raise ValueError("Unsupported dependency probing/module syntax")
    links, uses_mpp = [], False
    for directive, body in re.findall(r"^[ \t]*#[ \t]*(\w+)([^\n]*)", text, re.M):
        if directive not in {"include", "include_next", "import", "embed"}:
            continue
        body = body.strip()
        if directive != "include":
            raise ValueError("Unsupported dependency directive")
        angle = re.fullmatch(r"<([^<>]+)>", body)
        if angle:
            header = angle[1]
            if header in METAL_HEADERS:
                continue
            if header == MPP and mpp:
                uses_mpp = True
                continue
            raise ValueError("Unsupported SDK include: " + header)
        quoted = re.fullmatch(r'"([^"\n]+)"', body)
        if not quoted:
            raise ValueError("Macro/unresolved include")
        target = quoted[1]
        if target.startswith("/") or not re.fullmatch(r"[A-Za-z0-9_./-]+", target):
            raise ValueError("Unsafe quoted include")
        options = [
            posixpath.normpath(posixpath.join(posixpath.dirname(name), target)),
            posixpath.normpath(target),
        ]
        # A parent-relative reference may normalize inside the declared root; it never escapes.
        if options[0].startswith("../") or options[0] == "..":
            raise ValueError("Quoted include escapes closure root")
        valid = [p for p in options if not p.startswith("../") and p != ".."]
        resolved = next((p for p in valid if p in files), None)
        if resolved is None:
            raise ValueError("Unresolved quoted include: " + target)
        links.append(resolved)
    return links, uses_mpp


def load_manifest(path, expected):
    _digest(expected)
    path = Path(path).absolute()
    raw = _read(path, 2 << 20)
    if sha(raw) != expected:
        raise ValueError("Manifest SHA-256 mismatch")
    manifest = _json(raw)
    _fields(manifest, {"format", "standard", "sdk_mpp", "units"}, "manifest")
    if (
        manifest["format"] != FORMAT
        or manifest["standard"] not in STANDARDS
        or type(manifest["sdk_mpp"]) is not bool
    ):
        raise ValueError("Unsupported manifest format/standard/SDK policy")
    if manifest["sdk_mpp"] and manifest["standard"] != "metal4.0":
        raise ValueError("MPP requires metal4.0")
    units = manifest["units"]
    if not isinstance(units, list) or not 1 <= len(units) <= 32:
        raise ValueError("Expected 1..32 units")
    captured, ids, total = [], set(), 0
    for unit in units:
        _fields(unit, {"id", "revision", "primary", "files"}, "unit")
        ident = unit["id"]
        if (
            not isinstance(ident, str)
            or not re.fullmatch("[a-zA-Z0-9_-]{1,80}", ident)
            or ident.casefold() in ids
        ):
            raise ValueError("Invalid/duplicate unit ID")
        ids.add(ident.casefold())
        if not isinstance(unit["revision"], str) or not re.fullmatch(
            "[0-9a-f]{40}", unit["revision"]
        ):
            raise ValueError("Expected revision label of 40 lowercase hex characters")
        primary = _name(unit["primary"])
        files = unit["files"]
        if (
            not isinstance(files, dict)
            or not 1 <= len(files) <= 128
            or primary not in files
            or not primary.endswith(".metal")
        ):
            raise ValueError("Invalid primary or 1..128 file inventory")
        if len({n.casefold() for n in files}) != len(files):
            raise ValueError("Case-aliased source keys")
        for name in files:
            _name(name)
            if name.casefold() == MPP.casefold():
                raise ValueError("Local source shadows installed MPP umbrella")
            if any("/".join(name.split("/")[:i]) in files for i in range(1, len(name.split("/")))):
                raise ValueError("File/directory source key collision")
        blobs, origins, size = {}, {}, 0
        for name, binding in files.items():
            _name(name)
            if Path(name).suffix not in {".metal", ".h", ".hpp", ".inc"}:
                raise ValueError("Unsupported source file suffix")
            _fields(binding, {"path", "sha256"}, "file binding")
            if not isinstance(binding["path"], str) or not binding["path"]:
                raise ValueError("Expected local file path")
            src = Path(binding["path"])
            src = src if src.is_absolute() else path.parent / src
            declared_size = src.lstat().st_size
            if size + declared_size > MAX_UNIT or total + declared_size > MAX_TOTAL:
                raise ValueError("Source byte budget exceeded before read")
            data = _read(src, MAX_UNIT)
            if sha(data) != _digest(binding["sha256"]):
                raise ValueError("Source SHA-256 mismatch")
            size += len(data)
            total += len(data)
            if size > MAX_UNIT or total > MAX_TOTAL:
                raise ValueError("Source byte budget exceeded")
            blobs[name], origins[name] = data, str(src.absolute())
        if not blobs[primary].strip():
            raise ValueError("Empty primary source")
        graph, uses_mpp = {}, False
        for name, data in blobs.items():
            graph[name], used = _includes(data, name, blobs, manifest["sdk_mpp"])
            uses_mpp |= used
        reached, todo = set(), [primary]
        while todo:
            name = todo.pop()
            if name not in reached:
                reached.add(name)
                todo.extend(graph[name])
        if reached != set(blobs):
            raise ValueError("Inventory contains unreachable source files")
        captured.append(
            dict(
                metadata=unit, blobs=blobs, origins=origins, uses_mpp=uses_mpp, include_edges=graph
            )
        )
    return manifest, raw, captured


def _run(argv, env, cwd, timeout=90):
    started = time.time()
    try:
        p = subprocess.run(argv, env=env, cwd=cwd, capture_output=True, timeout=timeout)
        code, stdout, stderr, error = p.returncode, p.stdout, p.stderr, None
    except subprocess.TimeoutExpired as e:
        code, stdout, stderr, error = None, e.stdout or b"", e.stderr or b"", "timeout"
    except OSError as e:
        code, stdout, stderr, error = None, b"", str(e).encode(), "tool_unavailable"
    return dict(
        argv=argv,
        exit_code=code,
        error=error,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
        elapsed_seconds=time.time() - started,
    )


def _sdk_headers(sdk):
    root = (sdk / "System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers").resolve(
        strict=True
    )
    files, total = {}, 0
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            if p.is_symlink():
                raise ValueError("SDK header directory symlink")
            continue
        remaining = MAX_UNIT - total
        if len(files) >= 64 or p.lstat().st_size > remaining:
            raise ValueError("SDK MPP header budget exceeded before read")
        data = _read(p, remaining)
        total += len(data)
        files[str(p)] = sha(data)
        if len(files) > 64 or total > MAX_UNIT:
            raise ValueError("SDK MPP header budget exceeded")
    if not files or not (root / "MetalPerformancePrimitives.h").is_file():
        raise ValueError("Missing SDK MPP umbrella")
    return files


def compile_closure(manifest_path, manifest_sha256, output):
    manifest, raw, units = load_manifest(manifest_path, manifest_sha256)
    output = Path(output).absolute()
    if output.parent.resolve(strict=True) != output.parent:
        raise ValueError("Noncanonical output parent")
    output.mkdir(exist_ok=False)
    for name in ("home", "tmp", "module-cache"):
        (output / name).mkdir()
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(output / "home"),
        "TMPDIR": str(output / "tmp"),
        "LC_ALL": "C",
        "LANG": "C",
    }
    (output / "manifest.json").write_bytes(raw)
    report = dict(
        format=FORMAT,
        manifest_sha256=manifest_sha256,
        standard=manifest["standard"],
        host=platform.platform(),
        environment=env,
        units=[],
        tools={},
        sdk_headers={},
        source_code_sha256={
            str(p): sha(p.read_bytes())
            for p in (Path(__file__), Path(__file__).with_name("metal_compile.py"))
        },
        compiled=False,
        execution_verified=False,
        numerical_parity="not_run",
        performance="not_run",
        limitations=[
            "Compilation only, not GPU execution, numerical correctness, performance or whole-project validation.",
            "Revision labels and checksums are not source/review authenticity certification.",
            "Fixed flags may differ from the upstream build; imported local source is never executed.",
            "Installed compiler and SDK are trusted; this is not a hostile shader sandbox.",
        ],
    )
    pins = {str(output / "manifest.json"): manifest_sha256}
    for unit in units:
        folder = output / unit["metadata"]["id"] / "sources"
        for name, data in unit["blobs"].items():
            dest = folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            pins[str(dest)] = sha(data)

    def guard():
        for unit in units:
            folder = output / unit["metadata"]["id"] / "sources"
            actual = {
                str(p.relative_to(folder))
                for p in folder.rglob("*")
                if not p.is_dir() or p.is_symlink()
            }
            if actual != set(unit["blobs"]):
                return False
        return all(
            sha(_read(Path(p), MAX_UNIT if not p.endswith("manifest.json") else 2 << 20)) == h
            for p, h in pins.items()
        )

    try:
        for name, argv in [
            ("compiler", ["/usr/bin/xcrun", "--sdk", "macosx", "--find", "metal"]),
            ("sdk", ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"]),
            ("sdk_version", ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        ]:
            result = _run(argv, env, output)
            report["tools"][name] = result
            if result["exit_code"] != 0 or result["error"]:
                raise ValueError("Toolchain discovery failed: " + name)
        compiler = Path(report["tools"]["compiler"]["stdout"].strip())
        sdk = Path(report["tools"]["sdk"]["stdout"].strip())
        if not compiler.is_absolute() or not sdk.is_absolute():
            raise ValueError("Nonabsolute toolchain path")
        compiler, sdk = compiler.resolve(strict=True), sdk.resolve(strict=True)
        if not compiler.is_absolute() or not sdk.is_dir():
            raise ValueError("Invalid installed toolchain paths")
        toolhash = sha(_read(compiler, 256 << 20))
        report["compiler_path"], report["compiler_sha256"] = str(compiler), toolhash
        report["tools"]["version"] = _run([str(compiler), "--version"], env, output)
        if report["tools"]["version"]["exit_code"] != 0 or report["tools"]["version"]["error"]:
            raise ValueError("Compiler version probe failed")
        if any(u["uses_mpp"] for u in units):
            report["sdk_headers"] = _sdk_headers(sdk)

        def sdk_guard():
            return sha(_read(compiler, 256 << 20)) == toolhash and (
                not report["sdk_headers"] or _sdk_headers(sdk) == report["sdk_headers"]
            )

        airpins = {}
        for unit in units:
            meta = unit["metadata"]
            folder = output / meta["id"]
            air = folder / "output.air"
            record = dict(
                id=meta["id"],
                revision_label=meta["revision"],
                primary=meta["primary"],
                files={
                    n: dict(sha256=sha(b), bytes=len(b), original_path=unit["origins"][n])
                    for n, b in unit["blobs"].items()
                },
                include_edges=unit["include_edges"],
                passed=False,
            )
            report["units"].append(record)
            try:
                if not guard() or not sdk_guard():
                    raise ValueError("Source/toolchain drift before compilation")
                argv = [
                    str(compiler),
                    "-std=" + manifest["standard"],
                    "-O3",
                    "-fno-fast-math",
                    "-I",
                    str(folder / "sources"),
                    "-isysroot",
                    str(sdk),
                    "-fmodules-cache-path=" + str(output / "module-cache"),
                ]
                if unit["uses_mpp"]:
                    argv += ["-F", str(sdk / "System/Library/Frameworks")]
                argv += ["-c", str(folder / "sources" / meta["primary"]), "-o", str(air)]
                record["compile"] = _run(argv, env, output)
                if not guard() or not sdk_guard():
                    raise ValueError("Source/toolchain drift after compilation")
                if record["compile"]["exit_code"] == 0 and not record["compile"]["error"]:
                    artifact = _read(air, 128 << 20)
                    if not artifact:
                        raise ValueError("Empty AIR")
                    record.update(passed=True, air_sha256=sha(artifact), air_bytes=len(artifact))
                    airpins[str(air)] = sha(artifact)
            except (ValueError, OSError) as e:
                record["failure"] = str(e)
        report["all_snapshots_unchanged"] = guard()
        report["toolchain_unchanged"] = sdk_guard()
        report["all_air_unchanged"] = all(
            sha(_read(Path(p), 128 << 20)) == h for p, h in airpins.items()
        )
        report["compiled"] = (
            report["all_air_unchanged"]
            and all(r["passed"] for r in report["units"])
            and report["all_snapshots_unchanged"]
            and report["toolchain_unchanged"]
        )
    except (ValueError, OSError) as e:
        report["failure"] = str(e)
    report["receipt_sha256"] = sha(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    (output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_closure(args.manifest, args.manifest_sha256, args.output)
    except (ValueError, OSError) as e:
        parser.exit(2, f"Closure input error: {e}\n")
    print(json.dumps(result, indent=2))
    return 0 if result["compiled"] else 1
