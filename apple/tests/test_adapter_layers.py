"""Selected adapter scope contracts; CPU-only fake child processes."""

import dataclasses
from pathlib import Path

import pytest
import test_sft_project as _project_tests
from test_sft_project import _arg, _checkpoint, _read, _write, cli, sft

project = _project_tests.project


@pytest.mark.parametrize(
    "layers", [[], [True], [-1], [40], [39, 39], [1.0], ["39"], "39", 39, {39}, [None]]
)
def test_invalid_layer_selection_is_rejected(tmp_path, layers):
    with pytest.raises(ValueError, match="adapter_layers"):
        sft.SFTConfig(output_dir=str(tmp_path), adapter_layers=layers)


def test_defaults_and_explicit_layers_have_canonical_serializable_identity(tmp_path):
    assert sft.SFTConfig(output_dir=str(tmp_path)).adapter_layers == list(range(40))
    supplied = [39, 37]
    config = sft.SFTConfig(output_dir=str(tmp_path), adapter_layers=supplied)
    supplied.append(0)
    assert config.adapter_layers == [37, 39]
    assert dataclasses.asdict(config)["adapter_layers"] == [37, 39]
    assert sft.SFTConfig(output_dir=str(tmp_path), adapter_layers=(39,)).adapter_layers == [39]


def test_cli_explicit_layers_override_config_and_resume_disallows_override(tmp_path):
    path = tmp_path / "config.json"
    _write(path, {"output_dir": str(tmp_path / "out"), "adapter_layers": [37, 38, 39]})
    parser = cli.build_parser()
    args = parser.parse_args(
        ["finetune", "--model", "M", "--data", "D", "--config", str(path), "--adapter-layers", "39"]
    )
    assert cli.start_trainer(args, parser).args.adapter_layers == [39]
    with pytest.raises(SystemExit):
        parser.parse_args(["finetune", "--adapter-layers"])
    with pytest.raises(SystemExit):
        parser.parse_args(["resume", str(tmp_path), "--steps", "2", "--adapter-layers", "39"])


def test_selected_scope_reaches_training_resume_checkpoint_and_export(
    project, monkeypatch, tmp_path
):
    project.args.adapter_layers = [38, 39]
    project.args.max_steps = 2
    manifest = _read(project.root / "project.json")
    manifest["config"] = dataclasses.asdict(project.args)
    _write(project.root / "project.json", manifest)
    first = _checkpoint(project, 1)
    calls = []

    def child(command, *, label, directory):
        calls.append(command)
        if command[0] == "train":
            start = command.index("--adapter-layers") + 1
            assert command[start : start + 2] == [38, 39]
            assert command[start + 2] == "--learning-rate"
            assert _arg(command, "--resume") == first
            output = Path(_arg(command, "--output"))
            _checkpoint(project, 2, path=output / "step-000002")
            _write(output / "training.json", {"status": "completed", "completed_step": 2})

    monkeypatch.setattr(project, "_child", child)
    result = project.train(resume_from_checkpoint=True, compare=False)
    assert result["adapter_layers"] == [38, 39]
    assert result["completed_steps"] == 2
    assert any(command[0] == "train" for command in calls)
    reopened = sft.SFTTrainer.from_project(project.root)
    assert reopened.args.adapter_layers == [38, 39]
    exported = Path(reopened.save_model(tmp_path / "export"))
    assert _read(exported / "adapter_config.json")["adapter_layers"] == [38, 39]
    with pytest.raises(ValueError, match="Only resource"):
        sft.SFTTrainer.from_project(project.root, resources={"adapter_layers": [39]})
    project.args.adapter_layers = [39]
    with pytest.raises(ValueError, match="Project settings changed"):
        project.train(resume_from_checkpoint=True, compare=False)
    with pytest.raises(ValueError, match="adapter settings differ"):
        project._checkpoints()


def test_resigned_checkpoint_with_foreign_layer_scope_fails(project, monkeypatch):
    _checkpoint(project, 1, config_change={"adapter_layers": [39]})
    monkeypatch.setattr(project, "_child", lambda *a, **kw: pytest.fail("No child may run"))
    with pytest.raises(ValueError, match="adapter settings differ"):
        project.train(resume_from_checkpoint=True, compare=False)


@pytest.mark.parametrize("stored", ["omitted", None])
def test_legacy_all_layer_config_keeps_scope_and_implementation_check(project, stored):
    manifest = _read(project.root / "project.json")
    if stored == "omitted":
        manifest["config"].pop("adapter_layers")
    else:
        manifest["config"]["adapter_layers"] = stored
    _write(project.root / "project.json", manifest)
    _checkpoint(project, 1)
    reopened = sft.SFTTrainer.from_project(project.root)
    assert reopened.args.adapter_layers == list(range(40))
    assert reopened.status()["adapter_layers"] == list(range(40))
    manifest["implementation"]["source"]["sft.py"] = "old-source-must-still-fail"
    _write(project.root / "project.json", manifest)
    with pytest.raises(ValueError, match="implementation/runtime changed"):
        sft.SFTTrainer.from_project(project.root)
