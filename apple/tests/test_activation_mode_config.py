"""CPU-only activation-mode serialization, CLI, project and checkpoint boundaries."""

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_sft_project as project_tests
from safetensors.numpy import load_file, save_file
from test_resource_profiles import workflow_namespace
from test_sft_project import _arg, _checkpoint, _read, _write, cli, sft

from exl3_mlx.activation_config import (
    activation_description,
    normalize_adapter_config,
)

project = project_tests.project


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "fp32", "BF16", ""])
def test_invalid_activation_modes_fail_early(tmp_path, value):
    with pytest.raises(ValueError, match="activation_mode"):
        sft.SFTConfig(output_dir=str(tmp_path), activation_mode=value)
    with pytest.raises(ValueError, match="activation_mode"):
        normalize_adapter_config({"activation_mode": value})


def test_default_and_legacy_missing_mode_are_checkpoint(tmp_path):
    assert sft.SFTConfig(output_dir=str(tmp_path)).activation_mode == "checkpoint"
    assert normalize_adapter_config({}) == {"activation_mode": "checkpoint"}
    assert "FrozenIndexer" in activation_description("bf16")
    assert "packed weights" in activation_description("bf16")
    assert "identity STE" in activation_description("checkpoint")


def train_namespace():
    """Execute actual I/O functions with NumPy safetensors in place of MLX arrays."""
    source = Path(__file__).parents[1] / "src/exl3_mlx/train.py"
    names = {
        "checkpoint_identity",
        "_file_sha256",
        "_verify_adapter_integrity",
        "save_adapter",
        "load_adapter_config",
        "load_adapter",
    }
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    fake_mx = SimpleNamespace(
        save_safetensors=lambda path, arrays: save_file(arrays, path),
        load=load_file,
        eval=lambda *values: None,
        array_equal=lambda a, b: SimpleNamespace(item=lambda: np.array_equal(a, b)),
        all=np.all,
        isfinite=np.isfinite,
    )
    namespace = dict(
        Path=Path,
        hashlib=hashlib,
        json=json,
        mx=fake_mx,
        tree_flatten=lambda value: list(value.items()),
        normalize_adapter_config=normalize_adapter_config,
        activation_description=activation_description,
        MAX_CONTEXT_TOKENS=32768,
        ENGINE_SEMANTICS="cpu-fixture",
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.fixture
def adapter_io(tmp_path):
    namespace = train_namespace()
    root = tmp_path / "model"
    for name in (
        "config.json",
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
    arrays = {"a": np.ones((2, 2), dtype=np.float32)}
    model = SimpleNamespace(
        _root=root,
        adapter_config=dict(rank=4, alpha=8.0, adapter_layers=[39], expert_adapter_layers=[]),
        trainable_parameters=lambda: arrays,
        load_weights=lambda entries, strict: arrays.update(entries),
    )
    return namespace, model, arrays, tmp_path / "adapter"


@pytest.mark.parametrize("mode", ["checkpoint", "bf16"])
def test_saved_mode_and_honest_description_survive_roundtrip(adapter_io, mode):
    namespace, model, arrays, directory = adapter_io
    model.adapter_config["activation_mode"] = mode
    namespace["save_adapter"](model, directory, {})
    config = namespace["load_adapter_config"](directory)
    assert config["activation_mode"] == mode
    assert config["activation_quantization"] == activation_description(mode)
    arrays["a"] = np.zeros((2, 2), dtype=np.float32)
    namespace["load_adapter"](model, directory)
    assert np.array_equal(arrays["a"], np.ones((2, 2), dtype=np.float32))


@pytest.mark.parametrize("saved", ["missing", "checkpoint", "bf16"])
def test_cross_mode_adapter_load_is_rejected_before_parameters_change(adapter_io, saved):
    namespace, model, arrays, directory = adapter_io
    model.adapter_config["activation_mode"] = "bf16" if saved == "bf16" else "checkpoint"
    namespace["save_adapter"](model, directory, {})
    if saved == "missing":
        config = _read(directory / "adapter_config.json")
        config.pop("activation_mode")
        _write(directory / "adapter_config.json", config)
        assert namespace["load_adapter_config"](directory)["activation_mode"] == "checkpoint"
    model.adapter_config["activation_mode"] = "checkpoint" if saved == "bf16" else "bf16"
    before = arrays["a"]
    with pytest.raises(ValueError, match="activation_mode"):
        namespace["load_adapter"](model, directory)
    assert arrays["a"] is before


def test_legacy_model_and_adapter_both_use_checkpoint_without_mutation(adapter_io):
    namespace, model, arrays, directory = adapter_io
    namespace["save_adapter"](model, directory, {})
    config = _read(directory / "adapter_config.json")
    config.pop("activation_mode")
    _write(directory / "adapter_config.json", config)
    namespace["load_adapter"](model, directory)
    assert "activation_mode" not in model.adapter_config
    assert "activation_mode" not in _read(directory / "adapter_config.json")


@pytest.mark.parametrize("kind", ["adapter", "resume", "resume_adapter"])
def test_low_level_mode_inheritance_and_explicit_mismatch_fail_before_allocation(adapter_io, kind):
    io, model, _, directory = adapter_io
    model.adapter_config["activation_mode"] = "bf16"
    io["save_adapter"](model, directory, {})
    workflow = workflow_namespace()
    workflow["load_adapter_config"] = io["load_adapter_config"]
    assert (
        workflow["model_settings"](SimpleNamespace(**{kind: directory}))["activation_mode"]
        == "bf16"
    )
    with pytest.raises(ValueError, match="activation_mode differs"):
        workflow["model_settings"](
            SimpleNamespace(**{kind: directory, "activation_mode": "checkpoint"})
        )
    assert workflow["model_settings"](SimpleNamespace())["activation_mode"] == "checkpoint"


def test_low_level_cli_and_json_config_reject_unknown_mode(tmp_path):
    workflow = workflow_namespace()
    args = workflow["parse_args"](
        ["evaluate", "--model", "M", "--data", "D", "--activation-mode", "bf16"]
    )
    assert workflow["model_settings"](args)["activation_mode"] == "bf16"
    with pytest.raises(SystemExit):
        workflow["parse_args"](["doctor", "--model", "M", "--activation-mode", "fp32"])
    config = tmp_path / "config.json"
    _write(config, dict(model="M", data="D", output="O", activation_mode="fp32"))
    # argparse does not validate choice defaults supplied by set_defaults. Model
    # settings must still fail before any model allocation.
    with pytest.raises((ValueError, SystemExit), match="activation_mode|2"):
        workflow["model_settings"](workflow["parse_args"](["train", "--config", str(config)]))


def test_standard_cli_selection_and_resume_has_no_mode_override(tmp_path):
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
            "--activation-mode",
            "bf16",
        ]
    )
    assert cli.start_trainer(args, parser).args.activation_mode == "bf16"
    with pytest.raises(SystemExit):
        parser.parse_args(["resume", str(tmp_path), "--steps", "2", "--activation-mode", "bf16"])


def test_mode_is_immutable_project_identity_and_every_child_uses_it(project, monkeypatch, tmp_path):
    project.args.activation_mode = "bf16"
    project.args.max_steps = 1
    manifest = _read(project.root / "project.json")
    manifest["config"] = dataclasses.asdict(project.args)
    _write(project.root / "project.json", manifest)
    calls = []

    def child(command, *, label, directory):
        calls.append(command)
        assert _arg(command, "--activation-mode") == "bf16"
        if command[0] == "train":
            output = Path(_arg(command, "--output"))
            _checkpoint(project, 1, path=output / "step-000001")
            _write(
                output / "training.json",
                {"status": "completed", "completed_step": 1, "activation_mode": "bf16"},
            )
        elif command[0] == "evaluate":
            _write(
                Path(_arg(command, "--report")),
                project_tests._evaluation_result(project, command, logits_sha256="fixed"),
            )

    monkeypatch.setattr(project, "_child", child)
    assert project.train(compare=False)["activation_mode"] == "bf16"
    # Base and adapted fit both use the saved mode through _engine_options.
    project.verify_training_fit()
    assert len([c for c in calls if c[0] == "evaluate"]) == 2
    reopened = sft.SFTTrainer.from_project(project.root)
    assert reopened.args.activation_mode == "bf16"
    exported = Path(reopened.save_model(tmp_path / "export"))
    assert _read(exported / "adapter_config.json")["activation_mode"] == "bf16"
    with pytest.raises(ValueError, match="Only resource"):
        sft.SFTTrainer.from_project(project.root, resources={"activation_mode": "checkpoint"})
    reopened.args.activation_mode = "checkpoint"
    with pytest.raises(ValueError, match="Project settings changed"):
        reopened.train(resume_from_checkpoint=True, compare=False)
    with pytest.raises(ValueError, match="adapter settings differ"):
        reopened._checkpoints()


def test_legacy_project_config_defaults_mode_but_still_checks_source_identity(project):
    manifest = _read(project.root / "project.json")
    manifest["config"].pop("activation_mode")
    _write(project.root / "project.json", manifest)
    checkpoint = _checkpoint(project, 1)
    config = _read(checkpoint / "adapter_config.json")
    config.pop("activation_mode")
    _write(checkpoint / "adapter_config.json", config)
    state = _read(checkpoint / "trainer_state.json")
    state["files"]["adapter_config.json"] = sft.digest(checkpoint / "adapter_config.json")
    _write(checkpoint / "trainer_state.json", state)
    assert sft.SFTTrainer.from_project(project.root).status()["activation_mode"] == "checkpoint"
    manifest["implementation"]["source"]["sft.py"] = "changed-code"
    _write(project.root / "project.json", manifest)
    with pytest.raises(ValueError, match="implementation/runtime changed"):
        sft.SFTTrainer.from_project(project.root)


@pytest.mark.parametrize(
    "receipt", [{}, {"activation_mode": "checkpoint"}, {"activation_mode": None}]
)
def test_bf16_project_rejects_missing_or_wrong_child_mode_receipt(project, receipt):
    project.args.activation_mode = "bf16"
    with pytest.raises(ValueError, match="activation_mode"):
        project._verify_activation_receipt(receipt)
    project._verify_activation_receipt({"activation_mode": "bf16"})
