"""CPU-only substitutes exercise publication/retention and update boundaries."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exl3_mlx import workflow


@pytest.fixture
def engine(tmp_path, monkeypatch):
    train, valid = tmp_path / "train.jsonl", tmp_path / "valid.jsonl"
    train.write_text('{"input_ids":[0,2,3,4],"loss_start":2}\n')
    valid.write_text('{"input_ids":[0,8,9,4,1],"loss_start":2}\n')
    tracker = SimpleNamespace(updates=0, evaluation_losses={}, after_update=None, models=[])

    def load_data(path, root, *, seq_length, truncate=False):
        rows = [json.loads(row) for row in Path(path).read_text().splitlines()]
        for row in rows:
            if len(row["input_ids"]) > seq_length + 1:
                if not truncate:
                    raise ValueError("Examples exceed sequence length")
                row["input_ids"] = row["input_ids"][: seq_length + 1]
        return rows

    def identity(path, root):
        return {"data_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}

    def make_model(args, log):
        model = SimpleNamespace(value=0)
        model.trainable_parameters = lambda: {"adapter": SimpleNamespace(nbytes=8)}
        model.report = lambda: {"trainable_parameters": 1}
        tracker.models.append(model)
        return model

    def update(model, optimizer, rows, *, clip, check):
        check()
        model.value += sum(sum(row["input_ids"]) for row in rows)
        optimizer.value += model.value
        tracker.updates += 1
        if tracker.after_update:
            tracker.after_update(tracker.updates)
        return {"loss": 1.0}

    def save(model, optimizer, directory, state):
        directory.mkdir()
        (directory / "trainer_state.json").write_text(json.dumps(state))
        (directory / "values.json").write_text(json.dumps([model.value, optimizer.value]))

    def restore(model, optimizer, directory, contract):
        state = json.loads((directory / "trainer_state.json").read_text())
        if state["contract"] != contract:
            raise ValueError("Resume settings/data differ")
        model.value, optimizer.value = json.loads((directory / "values.json").read_text())
        return state

    def evaluate(model, rows, *, check):
        return {"loss": tracker.evaluation_losses.get(tracker.updates, 1.0)}

    monkeypatch.setattr(workflow, "load_dataset", load_data)
    monkeypatch.setattr(workflow, "training_data_identity", identity)
    monkeypatch.setattr(workflow, "make_model", make_model)
    monkeypatch.setattr(workflow, "optimizer_step", update)
    monkeypatch.setattr(workflow, "save_checkpoint", save)
    monkeypatch.setattr(workflow, "load_checkpoint", restore)
    monkeypatch.setattr(workflow, "load_adapter_config", lambda directory: {})
    monkeypatch.setattr(workflow, "sample_indices", lambda count, cursor, size, seed: [0] * count)
    monkeypatch.setattr(workflow, "evaluate", evaluate)
    monkeypatch.setattr(workflow, "tree_flatten", lambda tree: list(tree.items()))
    monkeypatch.setattr(
        workflow,
        "mx",
        SimpleNamespace(
            __version__="test",
            random=SimpleNamespace(seed=lambda seed: None),
            get_peak_memory=lambda: 0,
            device_info=lambda: {"device_name": "CPU fixture"},
        ),
    )
    monkeypatch.setattr(
        workflow,
        "optim",
        SimpleNamespace(
            Adam=lambda learning_rate: SimpleNamespace(value=0),
        ),
    )
    monkeypatch.setattr(workflow.importlib.metadata, "version", lambda name: "test")
    budget = SimpleNamespace(check=lambda: None, limit=2**30, usage=lambda: 0)

    def run(name, *flags, log=lambda text: None):
        args = workflow.parse_args(
            [
                "train",
                "--model",
                str(tmp_path / "base"),
                "--data",
                str(train),
                "--valid-data",
                str(valid),
                "--output",
                str(tmp_path / name),
                "--steps",
                "3",
                "--save-every",
                "1",
                "--keep-last",
                "1",
                "--eval-every",
                "1",
                *map(str, flags),
            ]
        )
        return workflow.train_run(args, budget, log)

    return SimpleNamespace(run=run, tracker=tracker, train=train, valid=valid, root=tmp_path)


def test_best_is_pinned_before_pruning_and_survives_resume(engine):
    engine.tracker.evaluation_losses = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    report = engine.run("first")
    first = engine.root / "first"
    assert sorted(p.name for p in first.glob("step-*")) == ["step-000001", "step-000003"]
    assert (first / "latest.txt").read_text().strip() == "step-000003"
    assert (first / "best.txt").read_text().strip() == report["best_checkpoint"]
    state = json.loads((first / "step-000001/trainer_state.json").read_text())
    assert state["metrics"]["validation"]["loss"] == 1.0
    assert state["selection"]["best"]["checkpoint"] == str(first / "step-000001")
    resumed = engine.run("resumed", "--resume", report["checkpoint"], "--steps", 1)
    assert resumed["best_checkpoint"] == report["best_checkpoint"]
    assert resumed["best_validation_loss"] == 1.0
    assert resumed["completed_step"] == 4
    assert Path(resumed["best_checkpoint"]).is_dir()
    # Exact model/optimizer continuation remains independent of selection loss.
    uninterrupted = engine.run("uninterrupted", "--steps", 4)
    assert (Path(resumed["checkpoint"]) / "values.json").read_text() == (
        Path(uninterrupted["checkpoint"]) / "values.json"
    ).read_text()


def test_stop_before_update_has_no_manufactured_checkpoint(engine):
    stop = engine.root / "stop"
    stop.touch()
    report = engine.run("stopped", "--stop-file", stop)
    assert report["status"] == "stopped"
    assert report["completed_step"] == report["additional_steps"] == 0
    assert report["checkpoint"] is None
    assert engine.tracker.updates == 0
    assert not list((engine.root / "stopped").glob("step-*"))


def test_new_best_keeps_previous_best_needed_by_an_older_retained_checkpoint(engine):
    engine.tracker.evaluation_losses = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 0.5}
    report = engine.run("first", "--steps", 5, "--keep-last", 2)
    first = engine.root / "first"
    assert Path(report["best_checkpoint"]).name == "step-000005"
    assert sorted(p.name for p in first.glob("step-*")) == [
        "step-000001",
        "step-000004",
        "step-000005",
    ]
    resumed = engine.run("resumed", "--resume", first / "step-000004", "--steps", 1)
    assert resumed["completed_step"] == 5
    assert Path(resumed["best_checkpoint"]).name == "step-000001"


def test_stop_during_update_saves_completed_state_and_exactly_resumes(engine):
    stop = engine.root / "stop"
    engine.tracker.after_update = lambda step: stop.touch() if step == 2 else None
    stopped = engine.run("stopped", "--stop-file", stop, "--save-every", 10)
    assert stopped["status"] == "stopped"
    assert stopped["completed_step"] == stopped["additional_steps"] == 2
    assert Path(stopped["checkpoint"]).name == "step-000002"
    state = json.loads((Path(stopped["checkpoint"]) / "trainer_state.json").read_text())
    assert state["step"] == state["cursor"] == 2
    assert "validation" not in stopped["history"][-1]
    stop.unlink()
    engine.tracker.after_update = None
    resumed = engine.run("resumed", "--resume", stopped["checkpoint"], "--steps", 1)
    whole = engine.run("whole")
    assert resumed["status"] == "completed"
    assert (Path(resumed["checkpoint"]) / "values.json").read_text() == (
        Path(whole["checkpoint"]) / "values.json"
    ).read_text()


def test_stop_before_resumed_update_returns_existing_checkpoint(engine):
    report = engine.run("first", "--steps", 1)
    stop = engine.root / "stop"
    stop.touch()
    resumed = engine.run("stopped", "--resume", report["checkpoint"], "--stop-file", stop)
    assert resumed["status"] == "stopped"
    assert resumed["additional_steps"] == 0
    assert resumed["checkpoint"] == report["checkpoint"]
    assert resumed["best_checkpoint"] == report["best_checkpoint"]


@pytest.mark.parametrize("change", ["subset", "validation"])
def test_resume_rejects_selection_change_before_model_allocation(engine, change):
    report = engine.run("first", "--steps", 1)
    flags = ["--eval-examples", "3"] if change == "subset" else []
    if change == "validation":
        engine.valid.write_text('{"input_ids":[0,9,9,4,1],"loss_start":2}\n')
    before = len(engine.tracker.models)
    with pytest.raises(ValueError, match="validation data/subset differ"):
        engine.run("resumed", "--resume", report["checkpoint"], *flags)
    assert len(engine.tracker.models) == before


def test_old_checkpoint_without_selection_preserves_exact_contract(engine):
    report = engine.run("old", "--seq-length", 16, "--steps", 1)
    state_path = Path(report["checkpoint"]) / "trainer_state.json"
    state = json.loads(state_path.read_text())
    del state["selection"]
    state_path.write_text(json.dumps(state))
    resumed = engine.run("resumed", "--resume", report["checkpoint"], "--steps", 1)
    assert resumed["seq_length"] == 16
    assert resumed["completed_step"] == 2
    assert resumed["contract"] == report["contract"]
    with pytest.raises(ValueError, match="settings/data.*differ"):
        engine.run("changed", "--resume", report["checkpoint"], "--seq-length", 17)


def test_inferred_lengths_cover_all_splits_and_prepared_configuration(engine):
    length, _ = workflow.load_resolved_datasets([engine.train, engine.valid], engine.root)
    assert length == 4
    (engine.root / "dataset.json").write_text('{"seq_length":128}')
    length, _ = workflow.load_resolved_datasets([engine.train, engine.valid], engine.root)
    assert length == 128
    length, _ = workflow.load_resolved_datasets([engine.train], engine.root, seq_length=8)
    assert length == 8
    engine.train.write_text(json.dumps({"input_ids": [2] * 32770, "loss_start": 1}))
    with pytest.raises(ValueError, match="exceed"):
        workflow.load_resolved_datasets([engine.train], engine.root)
    length, rows = workflow.load_resolved_datasets([engine.train], engine.root, truncate=True)
    assert length == 32768
    assert len(rows[0][0]["input_ids"]) == 32769


@pytest.mark.parametrize("length", [1, 32769, True, 1.5])
def test_inferred_length_rejects_invalid_prepared_bound(engine, length):
    (engine.root / "dataset.json").write_text(json.dumps({"seq_length": length}))
    with pytest.raises(ValueError, match="2..32768"):
        workflow.load_resolved_datasets([engine.train], engine.root)


def test_atomic_pointer_replacement_failure_preserves_existing_value(tmp_path, monkeypatch):
    path = tmp_path / "latest.txt"
    path.write_text("step-000001\n")

    def fail(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(workflow.os, "replace", fail)
    with pytest.raises(OSError, match="replacement failure"):
        workflow.atomic_text(path, "step-000002\n")
    assert path.read_text() == "step-000001\n"
    assert list(tmp_path.iterdir()) == [path]


def test_train_and_evaluate_default_to_inferred_sequence_length():
    train = workflow.parse_args(["train", "--model", "M", "--data", "D", "--output", "O"])
    evaluate = workflow.parse_args(["evaluate", "--model", "M", "--data", "D"])
    assert train.seq_length is evaluate.seq_length is None


def test_generation_messages_preserve_whole_conversation(tmp_path, monkeypatch):
    messages = [
        {"role": "system", "content": "Use the conversation history."},
        {"role": "user", "content": "My marker is cedar."},
        {"role": "assistant", "content": "Recorded."},
        {"role": "user", "content": "What is my marker?"},
    ]
    path = tmp_path / "messages.json"
    path.write_text(json.dumps(messages))
    seen = []

    def template(history, **kwargs):
        seen.append(history)
        assert kwargs == {"add_generation_prompt": True, "return_dict": False}
        return [0, 2, 3]

    tokenizer = SimpleNamespace(
        apply_chat_template=template, eos_token_id=1, decode=lambda ids, **kwargs: "cedar"
    )
    monkeypatch.setattr(workflow, "load_tokenizer", lambda root: tokenizer)
    monkeypatch.setattr(
        workflow,
        "mx",
        SimpleNamespace(
            array=lambda values, dtype: values,
            int32="int32",
            isfinite=lambda values: True,
            all=lambda values: SimpleNamespace(item=lambda: True),
            argmax=lambda values: SimpleNamespace(item=lambda: 1),
        ),
    )

    class Model:
        def __call__(self, ids):
            assert ids == [[0, 2, 3]]
            return self

        def __getitem__(self, key):
            return [0.0, 1.0]

        def eval(self):
            pass

        def report(self):
            return {}

    args = workflow.parse_args(
        ["generate", "--model", "M", "--messages-file", str(path), "--max-tokens", "1"]
    )
    result = workflow.generate(args, Model(), SimpleNamespace(check=lambda: None))
    assert seen == [messages]
    assert result["prompt_messages"] == messages
    assert result["prompt_ids"] == [0, 2, 3]


@pytest.mark.parametrize(
    "messages",
    [
        None,
        [],
        {},
        [1],
        [{"role": "tool", "content": "answer"}],
        [{"role": "assistant", "content": "answer"}],
        [{"role": "user", "content": [{"type": "image"}]}],
        [{"role": "user", "content": "question", "tool_calls": []}],
    ],
)
def test_generation_messages_reject_unsupported_history(tmp_path, messages):
    path = tmp_path / "messages.json"
    path.write_text(json.dumps(messages))
    args = SimpleNamespace(
        max_tokens=1,
        messages_file=str(path),
        raw_prompt=False,
        prompt=None,
    )
    with pytest.raises(ValueError, match="plain-text"):
        workflow.generate(args, None, None)


def test_generation_prompt_sources_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        workflow.parse_args(
            ["generate", "--model", "M", "--prompt", "Q", "--messages-file", "messages.json"]
        )
    args = workflow.parse_args(
        ["generate", "--model", "M", "--raw-prompt", "--messages-file", "messages.json"]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        workflow.generate(args, None, None)


def test_resume_old_arithmetic_rejects_before_model_allocation(engine):
    report = engine.run("old-arithmetic", "--steps", 1)
    state_path = Path(report["checkpoint"]) / "trainer_state.json"
    state = json.loads(state_path.read_text())
    del state["contract"]["engine_semantics"]
    state_path.write_text(json.dumps(state))
    before = len(engine.tracker.models)
    with pytest.raises(ValueError, match="engine semantics differ"):
        engine.run("new-arithmetic", "--resume", report["checkpoint"], "--steps", 1)
    assert len(engine.tracker.models) == before
