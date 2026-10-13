"""CPU-only final probe configuration, binding, and numerical receipt checks."""

import ast
import dataclasses
import importlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_sft_project as project_tests
from test_resource_profiles import workflow_namespace
from test_sft_project import _arg, _checkpoint, _evaluation_result, _read, _write, cli, sft

from exl3_mlx.fit_probe import (
    compare_final_fit_probe,
    file_sha256,
    run_final_fit_probe,
    selection_sha256,
    validate_final_fit_examples,
)

project = project_tests.project


def scores(rows):
    values = []
    for row in rows:
        target = row["input_ids"][row["loss_start"] :]
        values.append(
            dict(
                target_ids=target,
                predicted_ids=target,
                supervised_tokens=len(target),
                token_nll=[0.25] * len(target),
                loss=0.25,
                logits_sha256=selection_sha256([row]),
            )
        )
    return dict(
        examples=len(rows),
        supervised_tokens=sum(v["supervised_tokens"] for v in values),
        loss=0.25,
        rows=values,
        logits_sha256=selection_sha256(rows),
    )


@pytest.mark.parametrize("value", [-1, 33, True, 1.5, "3", None])
def test_probe_count_is_bounded_and_strict(value, tmp_path):
    with pytest.raises(ValueError, match="final_fit_examples"):
        validate_final_fit_examples(value)
    with pytest.raises(ValueError, match="final_fit_examples"):
        sft.SFTConfig(output_dir=str(tmp_path), final_fit_examples=value)


def test_default_off_and_fresh_comparison_must_cover_probe(tmp_path):
    assert sft.SFTConfig(output_dir=str(tmp_path)).final_fit_examples == 0
    with pytest.raises(ValueError, match="comparison_examples"):
        sft.SFTConfig(output_dir=str(tmp_path), final_fit_examples=3, comparison_examples=2)
    parse = workflow_namespace()["parse_args"]
    assert parse(["train", "--model", "M", "--data", "D", "--output", "O"]).final_fit_examples == 0
    assert (
        parse(
            ["train", "--model", "M", "--data", "D", "--output", "O", "--final-fit-examples", "3"]
        ).final_fit_examples
        == 3
    )
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "finetune",
            "--model",
            "M",
            "--data",
            "D",
            "--output",
            str(tmp_path),
            "--final-fit-examples",
            "3",
        ]
    )
    assert cli.start_trainer(args, parser).args.final_fit_examples == 3


@pytest.fixture
def pair(tmp_path):
    checkpoint = tmp_path / "step-000012"
    checkpoint.mkdir()
    _write(checkpoint / "trainer_state.json", {"step": 12})
    for name in ("adapter.safetensors", "adapter_config.json"):
        (checkpoint / name).write_text("fixture")
    rows = [{"input_ids": [0, 2, i, 1], "loss_start": 2} for i in (3, 4, 5)]
    expected = dict(
        data="data.jsonl",
        input_identity={"data_sha256": "fixture"},
        implementation={"source": "fixture"},
        seq_length=3,
        activation_mode="checkpoint",
        chunk_size=8192,
        available_examples=3,
        requested_examples=3,
    )
    cache = dict(enabled=True, limit_bytes=1024, hits=9, misses=3)
    model = SimpleNamespace(training=True, report=lambda: {"prefix_cache": dict(cache)})

    def evaluate(model, selected, **kwargs):
        assert model.training is True
        assert kwargs["preserve_mode"] is True and kwargs["include_token_metrics"] is True
        kwargs["check"]()
        cache["hits"] += len(selected)
        return scores(selected)

    metadata = {
        **{k: v for k, v in expected.items() if k != "available_examples"},
        "checkpoint": str(checkpoint),
        "step": 12,
        "checkpoint_files_sha256": {
            name: file_sha256(checkpoint / name)
            for name in ("adapter.safetensors", "adapter_config.json", "trainer_state.json")
        },
    }
    probe = run_final_fit_probe(
        model, rows, count=3, evaluate=evaluate, check=lambda: None, metadata=metadata
    )
    fresh = {**scores(rows), "prefix_cache": {"enabled": False, "limit_bytes": 0}}
    return dict(
        probe=probe, fresh=fresh, selected_rows=rows, checkpoint=checkpoint, expected=expected
    )


def test_three_row_warm_probe_requires_exact_fresh_logits(pair):
    result = compare_final_fit_probe(**pair)
    assert result["rows_verified"] == 3 and result["per_row_logits_exact"]
    assert result["real_model_warm_cache_parity_verified"] is True
    assert result["quality_certified"] is False


@pytest.mark.parametrize("change", ["disabled", "miss", "incomplete_hits"])
def test_same_scores_do_not_claim_unmeasured_cache_reuse(pair, change):
    before, after = pair["probe"]["prefix_cache_before"], pair["probe"]["prefix_cache_after"]
    if change == "disabled":
        before["enabled"] = after["enabled"] = False
    elif change == "miss":
        after["misses"] += 1
    else:
        after["hits"] -= 1
    result = compare_final_fit_probe(**pair)
    assert result["status"] == "exact"
    assert result["real_model_warm_cache_parity_verified"] is False


@pytest.mark.parametrize(
    "change",
    ["logits", "targets", "loss", "selection", "source", "checkpoint", "mode", "cold_cache"],
)
def test_bad_probe_comparison_is_rejected(pair, change):
    probe = pair["probe"]
    if change == "logits":
        pair["fresh"]["rows"][1]["logits_sha256"] = "f" * 64
    elif change == "targets":
        pair["fresh"]["rows"][0]["target_ids"] = [9, 1]
    elif change == "loss":
        pair["fresh"]["rows"][1]["loss"] = float("nan")
    elif change == "selection":
        probe["selected_examples_sha256"] = "f" * 64
    elif change == "source":
        probe["implementation"] = {"source": "different"}
    elif change == "checkpoint":
        (pair["checkpoint"] / "adapter.safetensors").write_text("changed")
    elif change == "mode":
        probe["training_mode"] = False
    else:
        pair["fresh"]["prefix_cache"]["enabled"] = True
    with pytest.raises(ValueError, match="Training fit probe"):
        compare_final_fit_probe(**pair)


def test_probe_never_switches_an_eval_model_into_training(pair):
    model = SimpleNamespace(training=False)
    with pytest.raises(ValueError, match="requires the current training mode"):
        run_final_fit_probe(
            model,
            pair["selected_rows"],
            count=3,
            evaluate=lambda *a, **k: pytest.fail("Must fail before scoring"),
            check=lambda: None,
            metadata={},
        )


def test_sft_forwards_probe_and_uses_uncached_fresh_fit(project, monkeypatch):
    project.args.final_fit_examples = 1
    project.args.prefix_cache_mib = 128
    project.args.max_steps = 1
    manifest = _read(project.root / "project.json")
    manifest["config"] = dataclasses.asdict(project.args)
    _write(project.root / "project.json", manifest)
    seen = []

    def child(command, *, label, directory):
        seen.append(command)
        if command[0] == "train":
            assert _arg(command, "--final-fit-examples") == 1
            assert _arg(command, "--prefix-cache-mib") == 128
            output = Path(_arg(command, "--output"))
            _checkpoint(project, 1, path=output / "step-000001")
            _write(output / "training.json", {"status": "completed", "completed_step": 1})
        elif command[0] == "evaluate":
            assert _arg(command, "--prefix-cache-mib") == 0
            _write(Path(_arg(command, "--report")), _evaluation_result(project, command))

    monkeypatch.setattr(project, "_child", child)
    project.train(compare=False)
    result = project.verify_training_fit()
    assert result["training_mode_probe_parity"]["status"] == "unavailable"
    assert result["training_mode_probe_parity"]["real_model_warm_cache_parity_verified"] is False
    assert len([c for c in seen if c[0] == "evaluate"]) == 2


def test_sft_probe_comparison_binds_selected_checkpoint(project, pair):
    # Full model-specific numeric identities are tested above; this tests routing
    # of the selected checkpoint and absence/mismatch statuses.
    selected = _checkpoint(project, 1)
    path = selected.parent / "training.json"
    _write(path, {"status": "completed", "training_fit_probe": {"checkpoint": "other"}})
    result = project._final_fit_parity(selected, {}, [], [], {})
    assert result["status"] == "unavailable" and "different checkpoint" in result["reason"]


def test_legacy_project_missing_probe_field_defaults_off(project):
    manifest = _read(project.root / "project.json")
    manifest["config"].pop("final_fit_examples")
    _write(project.root / "project.json", manifest)
    assert sft.SFTTrainer.from_project(project.root).args.final_fit_examples == 0


@pytest.mark.parametrize(
    "count,stopped,mutate",
    [
        (0, False, None),
        (3, False, None),
        (3, True, None),
        (3, False, "load"),
        (3, False, "probe"),
    ],
)
def test_actual_workflow_hook_runs_after_checkpoint_and_skips_disabled_or_stopped(
    tmp_path, monkeypatch, count, stopped, mutate
):
    """Execute the real control-flow function with CPU fakes for engine math."""
    from exl3_mlx.resources import resources_from_args

    source = Path(__file__).parents[1] / "src/exl3_mlx/workflow.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "train_run"
    )
    rows = [{"input_ids": [0, 2, i, 1], "loss_start": 2} for i in (3, 4, 5)]
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "run"
    stop_file = tmp_path / "stop"
    if stopped:
        stop_file.touch()
    args = workflow_namespace()["parse_args"](
        [
            "train",
            "--model",
            str(tmp_path / "model"),
            "--data",
            str(data),
            "--output",
            str(output),
            "--steps",
            "1",
            "--final-fit-examples",
            str(count),
            "--stop-file",
            str(stop_file),
        ]
    )
    events = []
    identity_state = {"changed": False}
    arrays = {"lora_b": np.zeros((2,), np.float32)}
    cache = dict(enabled=True, hits=9, misses=3)
    model = SimpleNamespace(
        training=True,
        adapter_config={"activation_mode": "checkpoint"},
        trainable_parameters=lambda: arrays,
        report=lambda: {"prefix_cache": dict(cache)},
    )

    def update(model, optimizer, examples, **kwargs):
        events.append("update")
        arrays["lora_b"] += 1
        return {"loss": 0.25, "supervised_tokens": 2, "microbatches": 1}

    def save(model, optimizer, checkpoint, state):
        events.append("save")
        checkpoint.mkdir()
        _write(checkpoint / "trainer_state.json", state)
        for name in ("adapter.safetensors", "adapter_config.json"):
            (checkpoint / name).write_text("fixture")

    def evaluate(model, selected, **kwargs):
        events.append("probe")
        assert events == ["update", "save", "probe"]
        assert kwargs["preserve_mode"] and kwargs["include_token_metrics"]
        assert np.array_equal(arrays["lora_b"], np.ones((2,), np.float32))
        cache["hits"] += len(selected)
        if mutate == "probe":
            identity_state["changed"] = True
        return scores(selected)

    def load(*args, **kwargs):
        if mutate == "load":
            identity_state["changed"] = True
        return 3, [rows]

    monkeypatch.setattr(
        sft, "implementation_identity", lambda: {"changed": identity_state["changed"]}
    )
    namespace = dict(
        __package__=project_tests._PACKAGE,
        Path=Path,
        json=json,
        math=__import__("math"),
        time=time,
        shutil=shutil,
        importlib=importlib,
        ENGINE_SEMANTICS="fixture",
        validate_final_fit_examples=validate_final_fit_examples,
        file_sha256=file_sha256,
        run_final_fit_probe=run_final_fit_probe,
        training_data_identity=lambda path, root: {"data_sha256": file_sha256(path)},
        load_resolved_datasets=load,
        validation_identity=lambda *a, **k: None,
        mx=SimpleNamespace(
            random=SimpleNamespace(seed=lambda _: None),
            get_peak_memory=lambda: 0,
            __version__="fixture",
            device_info=lambda: {"device_name": "CPU-fake"},
        ),
        optim=SimpleNamespace(Adam=lambda **k: object()),
        make_model=lambda *a: model,
        atomic_text=lambda path, text: Path(path).write_text(text),
        sample_indices=lambda *a: [0],
        optimizer_step=update,
        save_checkpoint=save,
        prune_checkpoints=lambda *a, **k: None,
        tree_flatten=lambda values: list(values.items()),
        evaluate=evaluate,
        resources_from_args=resources_from_args,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    budget = SimpleNamespace(check=lambda: None, limit=2**30, usage=lambda: 0)
    if mutate:
        with pytest.raises(ValueError, match="implementation changed"):
            namespace["train_run"](args, budget, lambda _: None)
        assert not (output / "training.json").exists()
        assert not (output / "training-fit-probe.json").exists()
        if mutate == "load":
            assert events == []
        else:
            assert (output / "step-000001/trainer_state.json").exists()
        return
    report = namespace["train_run"](args, budget, lambda _: None)
    if count and not stopped:
        assert events == ["update", "save", "probe"]
        probe = report["training_fit_probe"]
        assert probe["scores"]["examples"] == 3
        assert probe["selected_examples_sha256"] == selection_sha256(rows)
        assert (output / "training-fit-probe.json").exists()
    else:
        assert "probe" not in events
        assert report["training_fit_probe"] is None
        assert not (output / "training-fit-probe.json").exists()


@pytest.mark.parametrize("change", ["smaller_request", "prediction", "balanced_token_loss"])
def test_probe_cannot_reduce_declared_scope_or_forge_token_metrics(pair, change):
    if change == "smaller_request":
        pair["probe"]["requested_examples"] = 1
    elif change == "prediction":
        pair["fresh"]["rows"][0]["predicted_ids"][0] = 9
    else:
        pair["fresh"]["rows"][0]["token_nll"] = [0.125, 0.375]
    with pytest.raises(ValueError, match="Training fit probe"):
        compare_final_fit_probe(**pair)
