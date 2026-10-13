"""Bind known TensorFold projection sources, compiler AIR and a Swift parity runner.

CPU-only build. Runtime qualification is a separate explicit command. This is a
single reviewed operation harness, not a compiler for arbitrary repository code.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED = {
    "before": "5f6378c6e2c7a6b74bc9c6e6283a02a2b036693898f1227365e40437927177ba",
    "after": "216dc0cadf64b313283981b107288de42fb551237a1b4390585fc90b101c97de",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(value):
    if not value:
        raise ValueError("Compile/source identity mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compile-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = args.compile_dir.resolve()
    out = args.output.resolve()
    receipt_bytes = (source / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    digest = receipt.pop("receipt_sha256")
    require(
        sha(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        == digest
    )
    require(receipt["format"] == "exl3-metal-compile-pair-v1" and receipt["compiled"] is True)
    require(receipt["standard"] == "metal3.1")
    verified_air = {}
    for label, expected in EXPECTED.items():
        require(
            sha((source / f"{label}.metal").read_bytes())
            == expected
            == receipt["sources"][label]["sha256"]
        )
        air = (source / f"{label}.air").read_bytes()
        run = receipt["compiles"][label]
        require(sha(air) == run["air_sha256"] and len(air) > 0)
        require(run["exit_code"] == 0 and run["passed"] is True and run["error"] is None)
        require(
            run["argv"][:7]
            == ["xcrun", "--sdk", "macosx", "metal", "-std=metal3.1", "-O3", "-fno-fast-math"]
        )
        require(len(run["argv"]) == 12 and run["argv"][7].startswith("-fmodules-cache-path="))
        require(
            run["argv"][8:]
            == [
                "-c",
                receipt["sources"][label]["snapshot_path"],
                "-o",
                str(source / f"{label}.air"),
            ]
        )
        verified_air[label] = air
    out.mkdir(parents=True, exist_ok=False)
    swift = Path(__file__).with_suffix(".swift").with_name("verify_tensorfold_exlr8.swift")
    swift_bytes = swift.read_bytes()
    (out / swift.name).write_bytes(swift_bytes)
    report = dict(
        format="exl3-tensorfold-compile-link-v1",
        commit="51426b8e668b181d28c0bb6964c1882183e292fd",
        parent="e463e48bc7d796101a49fab5a8116016fc33c527",
        compile_receipt_sha256=sha(receipt_bytes),
        builder_sha256=sha(Path(__file__).read_bytes()),
        harness_source_sha256=sha(swift_bytes),
        libraries={},
    )
    (out / "compile-receipt.json").write_bytes(receipt_bytes)
    for label, expected in EXPECTED.items():
        air = verified_air[label]
        (out / f"{label}.air").write_bytes(air)
        argv = [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            str(out / f"{label}.air"),
            "-o",
            str(out / f"{label}.metallib"),
        ]
        r = subprocess.run(argv, capture_output=True, timeout=180)
        report["libraries"][label] = dict(
            source_sha256=expected,
            air_sha256=sha(air),
            argv=argv,
            link_exit_code=r.returncode,
            stderr=r.stderr.decode(errors="replace"),
            library_sha256=sha((out / f"{label}.metallib").read_bytes())
            if r.returncode == 0
            else None,
        )
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if r.returncode:
            raise SystemExit("Metal link failed; see manifest.json")
    argv = [
        "swiftc",
        "-O",
        "-module-cache-path",
        str(out / "swift-module-cache"),
        str(out / swift.name),
        "-o",
        str(out / "verify_tensorfold_exlr8"),
    ]
    r = subprocess.run(argv, capture_output=True, timeout=180)
    report["swift_build"] = dict(
        argv=argv, exit_code=r.returncode, stderr=r.stderr.decode(errors="replace")
    )
    report["harness_executable_sha256"] = (
        sha((out / "verify_tensorfold_exlr8").read_bytes()) if r.returncode == 0 else None
    )
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    if r.returncode:
        raise SystemExit("Swift build failed; see manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
