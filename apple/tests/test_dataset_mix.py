import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from exl3_mlx import dataset_mix as m
from exl3_mlx.data import load_dataset, sha256, tokenizer_identity


@pytest.fixture
def setup(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 32}}))
    (model / "tokenizer.json").write_text("{}")
    (model / "tokenizer_config.json").write_text('{"chat_template":"explicit template"}')
    source = tmp_path / "rows.jsonl"
    source.write_bytes(
        b'{ "input_ids": [0, 2, 3, 1], "loss_start": 2 }\r\n'
        b'{"input_ids":[0,4,1],"loss_start":2}\n'
        b'{"loss_start":1,"input_ids":[0,5,6,1]}'
    )
    return tmp_path, model, source


def plan_file(root, sources, **extra):
    path = root / "plan.json"
    path.write_text(json.dumps({"format": m.FORMAT, "sources": sources, **extra}))
    return path


def run(setup, sources=None, **extra):
    root, model, source = setup
    plan = plan_file(root, sources or [{"path": source.name, "repeat": 1}], **extra)
    return m.mix(plan, model, root / "mixed", seq_length=8)


def test_531_exact_lines_masks_counts_shares_and_standard_loader(setup):
    root, model, source = setup
    report = run(
        setup, [{"path": source.name, "rows": [i], "repeat": n} for i, n in enumerate([5, 3, 1])]
    )
    lines = source.read_bytes().splitlines(keepends=True)
    assert (root / "mixed/train.jsonl").read_bytes() == lines[0] * 5 + lines[1] * 3 + lines[
        2
    ] + b"\n"
    rows = load_dataset(root / "mixed/train.jsonl", model, seq_length=8)
    assert len(rows) == 9 and [r["loss_start"] for r in rows] == [2] * 8 + [1]
    assert report["counts"] == {"train": 9, "valid": 0, "test": 0}
    assert report["supervised_tokens"]["train"] == 16
    assert report["duplicates_removed"] == report["truncated_rows"] == 0
    assert not (root / "mixed/valid.jsonl").exists() and not (root / "mixed/test.jsonl").exists()
    mixture = json.loads((root / "mixed/mixture.json").read_bytes())
    assert [s["row_share"] for s in mixture["sources"]] == [5 / 9, 3 / 9, 1 / 9]
    assert [s["target_share"] for s in mixture["sources"]] == [10 / 16, 3 / 16, 3 / 16]
    trace = [json.loads(x) for x in (root / "mixed/mixture.jsonl").read_text().splitlines()]
    assert [x["output_row"] for x in trace] == list(range(9))
    assert [x["source_row"] for x in trace] == [0] * 5 + [1] * 3 + [2]
    assert trace[-1]["newline_added"] is True
    assert report["files"]["train.jsonl"] == sha256(root / "mixed/train.jsonl")
    for name, checksum in report["artifacts_sha256"].items():
        assert sha256(root / "mixed" / name) == checksum


def test_selection_order_repeats_whole_sequence_and_keeps_existing_duplicates(setup):
    root, model, source = setup
    source.write_text(
        '{"input_ids":[0,2,3,1],"loss_start":1}\n{"input_ids":[0,2,3,1],"loss_start":3}\n'
    )
    run(setup, [{"path": source.name, "rows": [1, 0], "repeat": 2}])
    rows = load_dataset(root / "mixed/train.jsonl", model, seq_length=8)
    assert [r["loss_start"] for r in rows] == [3, 1, 3, 1]


@pytest.mark.parametrize("key", ["protected", "exclude"])
def test_protected_ids_match_even_when_loss_mask_differs(setup, key):
    root, _, source = setup
    protected = root / "protected.jsonl"
    protected.write_text('{"input_ids":[0,2,3,1],"loss_start":1}\n')
    with pytest.raises(ValueError, match="token overlap"):
        run(setup, **{key: [protected.name]})
    assert not (root / "mixed").exists() and not list(root.glob(".mixture-*"))


@pytest.mark.parametrize(
    "source_spec",
    [
        {"path": "rows.jsonl", "repeat": True},
        {"path": "rows.jsonl", "repeat": 1.0},
        {"path": "rows.jsonl", "repeat": 0},
        {"path": "rows.jsonl", "repeat": -1},
        {"path": "rows.jsonl"},
        {"path": "rows.jsonl", "repeat": 1, "shuffle": True},
        {"path": "rows.jsonl", "repeat": 1, "rows": []},
        {"path": "rows.jsonl", "repeat": 1, "rows": [True]},
        {"path": "rows.jsonl", "repeat": 1, "rows": [3]},
        {"path": "rows.jsonl", "repeat": 1, "rows": [-1]},
        {"path": "rows.jsonl", "repeat": 1, "rows": [0, 0]},
    ],
)
def test_bad_source_specs_fail_before_output(setup, source_spec):
    with pytest.raises(ValueError):
        run(setup, [source_spec])
    assert not (setup[0] / "mixed").exists()


@pytest.mark.parametrize(
    "row",
    [
        {"text": "raw"},
        {"input_ids": [0, True, 1]},
        {"input_ids": [0, 32, 1]},
        {"input_ids": [0, 129264, 1]},
        {"input_ids": [0, 2, 1], "loss_start": True},
        {"input_ids": [0, 2, 1], "loss_start": 3},
        {"input_ids": [0, 2, 1], "other": 1},
        {"input_ids": [0] + [2] * 9 + [1]},
    ],
)
def test_raw_invalid_and_oversized_rows_are_not_retokenized_or_truncated(setup, row):
    setup[2].write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        run(setup)
    assert not (setup[0] / "mixed").exists()


def test_manifest_binding_then_output_loader_detects_drift(setup):
    root, model, source = setup
    manifest = {
        "format": "exl3-mlx-dataset-v1",
        "tokenizer": tokenizer_identity(model),
        "files": {source.name: sha256(source)},
        "model_config_sha256": sha256(model / "config.json"),
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    run(setup)
    (root / "mixed/train.jsonl").write_text('{"input_ids":[0,3,1]}\n')
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(root / "mixed/train.jsonl", model, seq_length=8)


@pytest.mark.parametrize("field", ["tokenizer", "files", "model_config_sha256"])
def test_source_manifest_mismatch_refused(setup, field):
    root, model, source = setup
    manifest = {
        "format": "exl3-mlx-dataset-v1",
        "tokenizer": tokenizer_identity(model),
        "files": {source.name: sha256(source)},
        "model_config_sha256": sha256(model / "config.json"),
    }
    manifest[field] = "changed" if field == "model_config_sha256" else {}
    (root / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run(setup)
    assert not (root / "mixed").exists()


def test_empty_duplicate_json_and_output_bounds(setup, monkeypatch):
    root, model, source = setup
    source.write_bytes(b"\n")
    with pytest.raises(ValueError, match="Empty"):
        run(setup)
    source.write_text('{"input_ids":[0,2,1],"loss_start":1,"loss_start":2}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        run(setup)
    source.write_text('{"input_ids":[0,2,1]}\n')
    monkeypatch.setattr(m, "MAX_ROWS", 2)
    with pytest.raises(ValueError, match="row limit"):
        run(setup, [{"path": source.name, "repeat": 2}, {"path": source.name, "repeat": 1}])
    monkeypatch.setattr(m, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        run(setup)
    assert not (root / "mixed").exists() and not list(root.glob(".mixture-*"))


def test_failure_after_stage_and_source_drift_cleanup(setup, monkeypatch):
    root, _, source = setup
    original = m.load_dataset

    def changed(path, *args, **kwargs):
        rows = original(path, *args, **kwargs)
        if path.parent.name.startswith(".mixture-"):
            source.write_bytes(source.read_bytes().replace(b"5,6", b"5,7"))
        return rows

    monkeypatch.setattr(m, "load_dataset", changed)
    with pytest.raises(ValueError, match="Input changed"):
        run(setup)
    assert not (root / "mixed").exists() and not list(root.glob(".mixture-*"))


def test_atomic_publish_never_replaces_concurrent_empty_directory(setup, monkeypatch):
    root = setup[0]
    original = m.publish_new

    def collide(stage, output):
        output.mkdir()
        original(stage, output)

    monkeypatch.setattr(m, "publish_new", collide)
    with pytest.raises(FileExistsError):
        run(setup)
    assert (root / "mixed").is_dir() and list((root / "mixed").iterdir()) == []
    assert not list(root.glob(".mixture-*"))


@pytest.mark.parametrize("change", ["fifo", "symlink", "replace", "growth"])
def test_final_input_check_is_bounded_and_regular(tmp_path, change):
    p = tmp_path / "input"
    p.write_bytes(b"123")
    inputs = m.Inputs()
    inputs.read(p)
    p.unlink()
    if change == "fifo":
        os.mkfifo(p)
    elif change == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"123")
        p.symlink_to(target)
    else:
        p.write_bytes(b"1234" if change == "growth" else b"123")
    with pytest.raises((ValueError, OSError)):
        inputs.check()


@pytest.mark.parametrize("entry", ["workflow", "cli"])
def test_actual_entrypoint_is_cpu_only_under_import_trap(setup, entry):
    root, model, source = setup
    plan = plan_file(root, [{"path": source.name, "repeat": 2}])
    script = """
import importlib.abc, runpy, sys
class NoDevice(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'mlx','mlx_lm'} or name in {'exl3_mlx.memory','exl3_mlx.deepseek','exl3_mlx.training','exl3_mlx.train'}:
   raise AssertionError('Device import: '+name)
sys.meta_path.insert(0,NoDevice())
entry=sys.argv[1];sys.argv=['exl3-mlx','mix',*sys.argv[2:]]
if entry=='workflow':runpy.run_module('exl3_mlx.workflow',run_name='__main__')
else:
 from exl3_mlx.cli import main
 main()
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(m.__file__).resolve().parents[1]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            entry,
            "--model",
            str(model),
            "--plan",
            str(plan),
            "--output",
            str(root / "mixed"),
            "--seq-length",
            "8",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["counts"]["train"] == 6
