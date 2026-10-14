"""Local synthetic files only: no model, MLX, child engine, or lease actions."""

import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import test_sft_project as fixtures

receipts = importlib.import_module(fixtures.sft.__package__ + ".evaluation_receipts")


def report(tmp_path, value=None):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"rows": [{"loss": 1.0}]} if value is None else value))
    return path.resolve()


def test_report_and_log_capture_binds_exact_bytes_and_is_immutable(tmp_path):
    path = report(tmp_path)
    log = tmp_path / "child.log"
    log.write_bytes(b"CPU fixture\n")
    snapshot = receipts.capture_receipts({"scores": path, "log": log})
    assert snapshot.hashes == {
        "scores": hashlib.sha256(path.read_bytes()).hexdigest(),
        "log": hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.files[0].data = b"changed"
    seen = []
    result = snapshot.validate_json("scores", seen.append)
    seen[0]["rows"][0]["loss"] = 2.0
    assert result["rows"][0]["loss"] == 1.0
    snapshot.recheck()
    log.write_bytes(b"altered log\n")
    with pytest.raises(ValueError, match="Receipt changed"):
        snapshot.recheck()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":{"loss":1,"loss":2}}',
        b'{"loss":NaN}',
        b'{"loss":Infinity}',
        b'{"loss":-Infinity}',
        b'{"rows":[',
        b"\xff",
    ],
)
def test_strict_json_before_validator(tmp_path, raw):
    path = report(tmp_path)
    path.write_bytes(raw)
    snapshot = receipts.capture_receipts({"report": path})
    with pytest.raises(ValueError):
        snapshot.validate_json("report", lambda _: pytest.fail("Must not validate invalid JSON"))


def test_callback_mutation_is_rejected(tmp_path):
    snapshot = receipts.capture_receipts({"report": report(tmp_path)})
    with pytest.raises(ValueError, match="validator mutated"):
        snapshot.validate_json("report", lambda value: value["rows"][0].update(loss=2.0))


@pytest.mark.parametrize("target", ["report", "log"])
def test_file_mutation_during_validation_is_rejected(tmp_path, target):
    path = report(tmp_path)
    log = tmp_path / "child.log"
    log.write_text("original")
    paths = {"report": path, "log": log}
    snapshot = receipts.capture_receipts(paths)
    with pytest.raises(ValueError, match="Receipt changed"):
        snapshot.validate_json("report", lambda _: paths[target].write_bytes(b"changed"))


@pytest.mark.parametrize("kind", ["file", "parent", "relative", "duplicate", "directory", "fifo"])
def test_path_aliases_and_nonregular_files_rejected(tmp_path, monkeypatch, kind):
    path = report(tmp_path)
    paths = {"report": path}
    if kind == "file":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        paths["report"] = alias
    elif kind == "parent":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        paths["report"] = alias / path.name
    elif kind == "relative":
        monkeypatch.chdir(tmp_path)
        paths["report"] = Path(path.name)
    elif kind == "duplicate":
        paths["second"] = path
    elif kind == "directory":
        paths["report"] = tmp_path
    else:
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        paths["report"] = fifo
    with pytest.raises((ValueError, OSError)):
        receipts.capture_receipts(paths)


def test_caps_are_exact_and_aggregate(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"1234")
    b.write_bytes(b"1234")
    receipts.capture_receipts({"a": a, "b": b}, max_file_bytes=4, max_total_bytes=8)
    for limits in ({"max_file_bytes": 3}, {"max_file_bytes": 4, "max_total_bytes": 7}):
        with pytest.raises(ValueError, match="byte limit"):
            receipts.capture_receipts({"a": a, "b": b}, **limits)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, None])
def test_invalid_limits_reject_before_read(tmp_path, monkeypatch, limit):
    path = report(tmp_path)
    monkeypatch.setattr(receipts.os, "open", lambda *a: pytest.fail("No read"))
    with pytest.raises(ValueError, match="byte limit"):
        receipts.capture_receipts({"report": path}, max_file_bytes=limit)


@pytest.mark.parametrize("kind", ["grow", "truncate", "replace", "parent_replace"])
def test_descriptor_and_path_continuity(tmp_path, monkeypatch, kind):
    parent = tmp_path / "reports"
    parent.mkdir()
    path = report(parent)
    original = path.read_bytes()
    real_read = receipts.os.read
    mutated = False

    def read(fd, count):
        nonlocal mutated
        value = real_read(fd, count)
        if not mutated:
            mutated = True
            if kind == "grow":
                path.write_bytes(original + b" " * 100)
            elif kind == "truncate":
                path.write_bytes(b"{}")
            elif kind == "replace":
                other = parent / "replacement"
                other.write_bytes(original)
                other.replace(path)
            else:
                parent.rename(tmp_path / "old")
                parent.mkdir()
                path.write_bytes(original)
        return value

    monkeypatch.setattr(receipts.os, "read", read)
    with pytest.raises(ValueError, match="Receipt (changed|pathname changed|exceeds)"):
        receipts.capture_receipts({"report": path}, max_file_bytes=len(original))


def test_same_bytes_replacement_after_capture_is_rejected(tmp_path):
    path = report(tmp_path)
    snapshot = receipts.capture_receipts({"report": path})
    replacement = tmp_path / "replacement"
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ValueError, match="Receipt changed"):
        snapshot.recheck()


def test_no_mlx_import_in_fresh_cpu_process():
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import importlib.abc, sys
class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'mlx', 'mlx_lm'}:
            raise AssertionError('Unexpected MLX import')
sys.meta_path.insert(0, BlockMLX())
from exl3_mlx.evaluation_receipts import capture_receipts
assert not any(n.split('.')[0] in {'mlx', 'mlx_lm'} for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-S", "-c", code],
        check=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(source)},
    )
