"""CPU contracts for truthful compile-only receipts; no model/GPU is imported."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from exl3_mlx import metal_compile as mc

GOOD = b"#include <metal_stdlib>\nusing namespace metal;\nkernel void example() {}\n"


@pytest.fixture
def pair(tmp_path):
    files = [tmp_path / name for name in ("parent.metal", "head.metal")]
    files[0].write_bytes(GOOD)
    files[1].write_bytes(GOOD.replace(b"example", b"changed"))
    return *files, tmp_path / "result"


def compiler(monkeypatch, *, fail=None, missing_air=False):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        label = Path(argv[-1]).stem if "-c" in argv else "tool"
        if "-c" in argv and not missing_air:
            Path(argv[-1]).write_bytes(b"AIR:" + label.encode())
        return dict(
            argv=argv,
            started_unix=1,
            elapsed_seconds=1,
            exit_code=1 if label == fail else 0,
            error=None,
            stdout="",
            stderr="syntax error" if label == fail else "",
        )

    monkeypatch.setattr(mc, "_run", run)
    return calls


def test_success_snapshots_both_sources_and_records_bound_flags(pair, monkeypatch):
    calls = compiler(monkeypatch)
    result = mc.compile_pair(*pair, standard="metal4.0")
    assert result["compiled"] is True
    assert result["execution_verified"] is False
    assert result["performance"] == result["numerical_parity"] == "not_run"
    for label, original in zip(("before", "after"), pair[:2], strict=True):
        info = result["sources"][label]
        assert Path(info["snapshot_path"]).read_bytes() == original.read_bytes()
        assert info["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    for argv in calls[2:]:
        assert "-std=metal4.0" in argv and "-O3" in argv and "-fno-fast-math" in argv
        assert str(pair[2]) in argv[-1]
    receipt = json.loads((pair[2] / "receipt.json").read_text())
    digest = receipt.pop("receipt_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


@pytest.mark.parametrize("label", ["before", "after"])
def test_failed_source_is_retained_and_cannot_pass(pair, monkeypatch, label):
    calls = compiler(monkeypatch, fail=label)
    result = mc.compile_pair(*pair)
    assert result["compiled"] is False
    assert result["compiles"][label]["stderr"] == "syntax error"
    assert len(calls) == 4  # Both attempts retained, including after a failed parent.
    assert (pair[2] / "receipt.json").is_file()


def test_success_exit_without_output_is_not_compilation(pair, monkeypatch):
    compiler(monkeypatch, missing_air=True)
    assert mc.compile_pair(*pair)["compiled"] is False


def test_zero_length_air_is_not_success(pair, monkeypatch):
    compiler(monkeypatch)
    original_run = mc._run

    def run(argv, **kwargs):
        result = original_run(argv, **kwargs)
        if "-c" in argv:
            Path(argv[-1]).write_bytes(b"")
        return result

    monkeypatch.setattr(mc, "_run", run)
    assert mc.compile_pair(*pair)["compiled"] is False


def test_earlier_snapshot_mutation_during_later_compile_fails(pair, monkeypatch):
    compiler(monkeypatch)
    original_run = mc._run

    def run(argv, **kwargs):
        result = original_run(argv, **kwargs)
        if "-c" in argv and Path(argv[-1]).stem == "after":
            (pair[2] / "before.metal").write_bytes(b"mutated")
        return result

    monkeypatch.setattr(mc, "_run", run)
    result = mc.compile_pair(*pair)
    assert result["all_snapshots_unchanged"] is False
    assert result["compiled"] is False


def test_cannot_overwrite_receipt_or_source(pair, monkeypatch):
    compiler(monkeypatch)
    mc.compile_pair(*pair)
    receipt = (pair[2] / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        mc.compile_pair(*pair)
    assert (pair[2] / "receipt.json").read_bytes() == receipt


@pytest.mark.parametrize(
    "include",
    [
        b'#include "local.h"',
        b"#include SOME_MACRO",
        b"#include_next <metal_stdlib>",
        b"#import <metal_stdlib>",
        b'#embed "data"',
        b'#inc\\\nlude "local.h"',
        b'#include /*comment*/ "local.h"',
        b'\f#include "local.h"',
        b'\v#include "local.h"',
        b'#define MARKER "/*"\n#include "local.h"\n// */',
        b'\r#include "local.h"',
    ],
)
def test_unsupported_dependencies_fail_before_any_compiler_or_output(pair, monkeypatch, include):
    pair[0].write_bytes(GOOD + include + b"\n")
    calls = compiler(monkeypatch)
    with pytest.raises(ValueError, match="Only #include"):
        mc.compile_pair(*pair)
    assert not calls and not pair[2].exists()


@pytest.mark.parametrize("directive", [b'%:include "local.h"', b'??=include "local.h"'])
def test_alternate_directive_spellings_rejected(pair, monkeypatch, directive):
    pair[0].write_bytes(GOOD + directive + b"\n")
    calls = compiler(monkeypatch)
    with pytest.raises(ValueError, match="Alternate preprocessor"):
        mc.compile_pair(*pair)
    assert not calls and not pair[2].exists()


def test_recorded_input_is_snapshot_not_mutable_original(pair, monkeypatch):
    calls = compiler(monkeypatch)
    original_run = mc._run

    def run(argv, **kwargs):
        if "-c" in argv:
            pair[0].write_bytes(b"original changed after snapshot")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(mc, "_run", run)
    result = mc.compile_pair(*pair)
    assert result["compiled"]
    assert (pair[2] / "before.metal").read_bytes() == GOOD
    assert str(pair[0]) not in calls[2]


def test_timeout_and_missing_tool_are_failures(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 180, output=b"partial", stderr=b"timeout")

    monkeypatch.setattr(subprocess, "run", timeout)
    report = mc._run(["xcrun"])
    assert report["exit_code"] is None and report["error"] == "compiler_timeout"
    assert report["stdout"] == "partial"

    def missing(*args, **kwargs):
        raise FileNotFoundError("xcrun")

    monkeypatch.setattr(subprocess, "run", missing)
    report = mc._run(["xcrun"])
    assert report["exit_code"] is None and report["error"] == "tool_unavailable"


def test_no_silent_include_environment(monkeypatch):
    for key in ("CPATH", "SDKROOT", "CPLUS_INCLUDE_PATH"):
        monkeypatch.setenv(key, "/external")

    def check(argv, **kwargs):
        assert all(key not in kwargs["env"] for key in ("CPATH", "SDKROOT", "CPLUS_INCLUDE_PATH"))
        assert kwargs["timeout"] == 180 and "shell" not in kwargs
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", check)
    assert mc._run(["xcrun"])["exit_code"] == 0
