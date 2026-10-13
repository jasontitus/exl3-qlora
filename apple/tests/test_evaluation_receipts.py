import hashlib
import json
from types import SimpleNamespace

import pytest

from exl3_mlx import sft, workflow


class Budget:
    limit = 1 << 30

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def check(self):
        pass

    def usage(self):
        return 0


@pytest.mark.parametrize("mutate", [None, "data", "implementation", "during_load"])
def test_evaluate_binds_actual_input_selection_and_rejects_midrun_changes(
    tmp_path, monkeypatch, mutate
):
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 12}}))
    data = tmp_path / "data.jsonl"
    rows = [
        {"input_ids": [0, 2, 3, 4], "loss_start": 2},
        {"input_ids": [0, 5, 6], "loss_start": 1},
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    target = tmp_path / "evaluation.json"
    argv = [
        "evaluate",
        "--model",
        str(root),
        "--data",
        str(data),
        "--seq-length",
        "3",
        "--max-examples",
        "1",
        "--report",
        str(target),
    ]
    monkeypatch.setattr(workflow, "MemoryBudget", Budget)
    monkeypatch.setattr(
        workflow,
        "make_model",
        lambda *_: SimpleNamespace(report=lambda: {"activation_mode": "checkpoint"}),
    )
    identity_state = {"changed": False}
    real_identity = sft.implementation_identity
    monkeypatch.setattr(
        sft,
        "implementation_identity",
        lambda: {"changed": True} if identity_state["changed"] else real_identity(),
    )
    real_load = workflow.load_resolved_datasets

    def load(*args, **kwargs):
        result = real_load(*args, **kwargs)
        if mutate == "during_load":
            data.write_text(data.read_text() + "\n")
        return result

    monkeypatch.setattr(workflow, "load_resolved_datasets", load)

    def score(model, selected, **kwargs):
        assert selected == rows[:1]
        if mutate == "data":
            data.write_text(data.read_text() + "\n")
        elif mutate == "implementation":
            identity_state["changed"] = True
        return {"loss": 0.25}

    monkeypatch.setattr(workflow, "evaluate", score)
    if mutate:
        with pytest.raises(SystemExit, match="changed during execution"):
            workflow.main(argv)
        assert not target.exists()
    else:
        workflow.main(argv)
        receipt = json.loads(target.read_text())["evaluation_context"]
        assert receipt["argv"] == argv
        assert receipt["process_id"] > 0
        assert receipt["data"] == str(data.resolve())
        assert receipt["seq_length"] == 3
        assert receipt["requested_max_examples"] == 1
        assert receipt["input_identity"] == {"data_sha256": original_hash}
        expected = hashlib.sha256(
            json.dumps(rows[:1], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert receipt["selected_examples_sha256"] == expected
        assert receipt["implementation"]["source"]["workflow.py"]


@pytest.mark.parametrize("changed", ["train", "valid"])
def test_training_binds_data_before_loading(tmp_path, monkeypatch, changed):
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 12}}))
    train, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    train.write_text(json.dumps({"input_ids": [0, 2, 3, 4], "loss_start": 2}) + "\n")
    valid.write_text(json.dumps({"input_ids": [0, 5, 6], "loss_start": 1}) + "\n")
    original = workflow.load_resolved_datasets

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        path = train if changed == "train" else valid
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(workflow, "load_resolved_datasets", load)
    monkeypatch.setattr(workflow, "MemoryBudget", Budget)
    output = tmp_path / "output"
    with pytest.raises(SystemExit, match="data changed during loading"):
        workflow.main([
            "train", "--model", str(root), "--data", str(train),
            "--valid-data", str(valid), "--output", str(output), "--steps", "1",
        ])
    assert not output.exists()
