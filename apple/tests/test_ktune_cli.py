"""CLI contracts use the same importer and standard SFT project implementation."""

import json
from pathlib import Path

import pytest

from exl3_mlx import ktune_cli, ktune_data


def test_prepare_forwards_complete_context_controls(tmp_path, monkeypatch, capsys):
    observed = {}

    def import_data(source, model, output, **kwargs):
        observed.update(source=source, model=model, output=str(output), **kwargs)
        return {"ready_for_sft": True, "max_length": 6000}

    monkeypatch.setattr(ktune_data, "import_ktune", import_data)
    assert (
        ktune_cli.main(
            [
                "prepare",
                "--source",
                "archive",
                "--model",
                "model",
                "--output",
                str(tmp_path / "data"),
                "--max-length",
                "8192",
                "--max-examples",
                "2",
                "--context-mode",
                "parent_files",
                "--parent-repo",
                "parents",
                "--tier",
                "official_promoted",
            ]
        )
        == 0
    )
    assert observed["max_length"] == 8192
    assert observed["context_mode"] == "parent_files"
    assert observed["max_examples"] == 2
    assert observed["tiers"] == ["official_promoted"]
    assert json.loads(capsys.readouterr().out)["max_length"] == 6000


@pytest.mark.parametrize("status", ["completed", "stopped"])
@pytest.mark.parametrize("chunk_size", [512, 8192])
def test_finetune_prepares_standard_project_and_keeps_training_scope(
    tmp_path, monkeypatch, capsys, status, chunk_size
):
    observed = {}
    output = tmp_path / "project"

    def import_data(source, model, path, **kwargs):
        return {"ready_for_sft": True, "max_length": 6100}

    class Trainer:
        def __init__(self, *, model, args, train_dataset):
            observed.update(model=model, config=args, dataset=train_dataset)

        def prepare(self, *, provenance_required=False):
            assert provenance_required
            output.mkdir()
            return {"data": {"max_length": 6100}}

        def bind_provenance(self, dataset, report):
            (output / "ktune-source.json").write_text(json.dumps(report))

        def train(self, **kwargs):
            observed["train"] = kwargs
            return {"status": status}

        def verify_training_fit(self):
            assert status == "completed"
            return {"evaluation_scope": "training_only", "loss_delta": -0.1}

    monkeypatch.setattr(ktune_data, "import_ktune", import_data)
    monkeypatch.setattr(ktune_cli, "SFTTrainer", Trainer)
    assert (
        ktune_cli.main(
            [
                "finetune",
                "--source",
                "archive",
                "--model",
                "M",
                "--output",
                str(output),
                "--fit-examples",
                "2",
                "--verify-fit",
                "--chunk-size",
                str(chunk_size),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["quality_certified"] is False
    assert result["status"] == status
    assert observed["config"].max_length == 6100
    assert observed["config"].memory_gib == 32
    assert observed["config"].chunk_size == chunk_size
    assert observed["config"].comparison_examples == 2
    assert observed["train"] == {"compare": False}
    assert observed["dataset"] == Path(str(output) + "-ktune-data")
    assert (output / "ktune-source.json").is_file()
