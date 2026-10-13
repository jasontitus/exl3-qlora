"""CPU regression checks for the reviewed operation harness's artifact binding."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def build(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "scripts/build_tensorfold_qualification.py"
    spec = importlib.util.spec_from_file_location("tensorfold_build", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "compiled", tmp_path / "build"
    source.mkdir()
    report = {
        "format": "exl3-metal-compile-pair-v1",
        "compiled": True,
        "standard": "metal3.1",
        "sources": {},
        "compiles": {},
    }
    hashes = {}
    for label in ("before", "after"):
        raw = f"source {label}".encode()
        (source / f"{label}.metal").write_bytes(raw)
        air = f"AIR {label}".encode()
        (source / f"{label}.air").write_bytes(air)
        hashes[label] = module.sha(raw)
        report["sources"][label] = {
            "sha256": hashes[label],
            "snapshot_path": str(source / f"{label}.metal"),
        }
        report["compiles"][label] = {
            "air_sha256": module.sha(air),
            "passed": True,
            "error": None,
            "exit_code": 0,
            "argv": [
                "xcrun",
                "--sdk",
                "macosx",
                "metal",
                "-std=metal3.1",
                "-O3",
                "-fno-fast-math",
                "-fmodules-cache-path=/tmp/example",
                "-c",
                str(source / f"{label}.metal"),
                "-o",
                str(source / f"{label}.air"),
            ],
        }
    monkeypatch.setattr(module, "EXPECTED", hashes)
    monkeypatch.setattr(
        sys, "argv", [str(script), "--compile-dir", str(source), "--output", str(output)]
    )

    def save():
        payload = dict(report)
        payload["receipt_sha256"] = module.sha(
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        (source / "receipt.json").write_text(json.dumps(payload))

    save()
    return module, source, output, report, save


def test_verified_air_is_captured_once_even_if_original_changes_after_first_link(
    build, monkeypatch
):
    module, source, output, _, _ = build
    original_after = (source / "after.air").read_bytes()

    def run(argv, **kwargs):
        if argv[0] == "xcrun":
            (source / "after.air").write_bytes(b"changed after verification")
        Path(argv[-1]).write_bytes(b"linked artifact:" + argv[-1].encode())
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    module.main()
    assert (output / "after.air").read_bytes() == original_after
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["libraries"]["after"]["air_sha256"] == module.sha(original_after)


@pytest.mark.parametrize("corruption", ["source", "air", "flags", "compiler_failure", "receipt"])
def test_bad_binding_rejected_before_tools_or_output(build, monkeypatch, corruption):
    module, source, output, report, save = build
    if corruption in ("source", "air"):
        suffix = "metal" if corruption == "source" else "air"
        (source / f"after.{suffix}").write_bytes(b"unbound replacement")
    elif corruption == "flags":
        report["compiles"]["after"]["argv"][6] = "-ffast-math"
        save()
    elif corruption == "compiler_failure":
        report["compiles"]["after"]["exit_code"] = 1
        save()
    else:
        payload = json.loads((source / "receipt.json").read_text())
        payload["receipt_sha256"] = "0" * 64
        (source / "receipt.json").write_text(json.dumps(payload))

    def forbidden(*args, **kwargs):
        pytest.fail("Compiler/linker must not run with an invalid input binding")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="identity mismatch"):
        module.main()
    assert not output.exists()
