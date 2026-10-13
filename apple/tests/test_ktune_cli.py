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


@pytest.mark.parametrize("activation_mode", ["checkpoint", "bf16"])
@pytest.mark.parametrize("status", ["completed", "stopped"])
@pytest.mark.parametrize("chunk_size", [512, 8192])
@pytest.mark.parametrize("adapter_layers", [None, [38, 39]])
def test_finetune_prepares_standard_project_and_keeps_training_scope(
    tmp_path, monkeypatch, capsys, status, chunk_size, adapter_layers, activation_mode
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
                "--final-fit-examples",
                "2",
                "--verify-fit",
                "--chunk-size",
                str(chunk_size),
                "--activation-mode",
                activation_mode,
                "--prefix-cache-mib",
                "2048",
                *(["--adapter-layers", *map(str, adapter_layers)] if adapter_layers else []),
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
    assert observed["config"].prefix_cache_mib == 2048
    assert observed["config"].activation_mode == activation_mode
    assert observed["config"].adapter_layers == (adapter_layers or list(range(40)))
    assert observed["config"].comparison_examples == 2
    assert observed["config"].final_fit_examples == 2
    assert observed["train"] == {"compare": False}
    assert observed["dataset"] == Path(str(output) + "-ktune-data")
    assert (output / "ktune-source.json").is_file()


def test_prepare_upstream_uses_separate_safe_importer(tmp_path, monkeypatch, capsys):
    from exl3_mlx import upstream_data

    observed = {}

    def importer(source, model, output, **kwargs):
        observed.update(source=source, model=model, output=output, **kwargs)
        return {"ready_for_sft": False, "execution_verified_examples": 0}

    monkeypatch.setattr(upstream_data, "import_upstream", importer)
    assert (
        ktune_cli.main(
            [
                "prepare-upstream",
                "--source",
                "manifest.json",
                "--model",
                "model",
                "--output",
                str(tmp_path / "data"),
                "--max-length",
                "8192",
                "--max-examples",
                "100",
                "--minimum-verification",
                "execution_verified",
            ]
        )
        == 0
    )
    assert observed["max_length"] == 8192 and observed["max_examples"] == 100
    assert observed["minimum_verification"] == "execution_verified"
    assert json.loads(capsys.readouterr().out)["execution_verified_examples"] == 0


def test_compile_metal_dispatches_to_separate_compiler_gate(monkeypatch):
    import sys
    import types

    observed = {}
    fake = types.ModuleType("exl3_mlx.metal_compile")

    def main(argv):
        observed["argv"] = argv
        return 0

    fake.main = main
    monkeypatch.setitem(sys.modules, "exl3_mlx.metal_compile", fake)
    assert (
        ktune_cli.main(
            [
                "compile-metal",
                "--before",
                "a.metal",
                "--after",
                "b.metal",
                "--output",
                "receipt",
                "--standard",
                "metal4.0",
            ]
        )
        == 0
    )
    assert observed["argv"] == [
        "--before",
        "a.metal",
        "--after",
        "b.metal",
        "--output",
        "receipt",
        "--standard",
        "metal4.0",
    ]


@pytest.mark.parametrize(
    "status,prepare_only", [("completed", False), ("stopped", False), ("completed", True)]
)
def test_finetune_upstream_reuses_standard_engine_and_requires_bound_provenance(
    tmp_path, monkeypatch, capsys, status, prepare_only
):
    from exl3_mlx import upstream_data

    output = tmp_path / "project"
    events, observed = [], {}
    source_report = {"ready_for_sft": True, "max_length": 3394, "execution_verified_examples": 0}

    def importer(source, model, dataset, **kwargs):
        events.append("import")
        observed.update(source=source, import_args=kwargs, dataset=dataset)
        return source_report

    class Trainer:
        def __init__(self, *, model, args, train_dataset):
            observed.update(model=model, config=args, train_dataset=train_dataset)

        def prepare(self, *, provenance_required=False):
            assert provenance_required is True
            events.append("prepare")
            return {"data": {"max_length": 3394}}

        def bind_provenance(self, dataset, report):
            assert report is source_report and dataset == observed["dataset"]
            events.append("bind")

        def train(self, **kwargs):
            assert events == ["import", "prepare", "bind"]
            assert kwargs == {"compare": False}
            events.append("train")
            return {"status": status}

        def verify_training_fit(self):
            assert status == "completed"
            events.append("fit")
            return {"evaluation_scope": "training_only", "quality_certified": False}

    monkeypatch.setattr(upstream_data, "import_upstream", importer)
    monkeypatch.setattr(ktune_cli, "SFTTrainer", Trainer)
    argv = [
        "finetune-upstream",
        "--source",
        "manifest.json",
        "--model",
        "model",
        "--output",
        str(output),
        "--steps",
        "3",
        "--max-examples",
        "10",
        "--adapter-layers",
        "39",
        "--learning-rate",
        "0.0001",
        "--prefix-cache-mib",
        "2048",
        "--fit-examples",
        "2",
        "--final-fit-examples",
        "2",
        "--verify-fit",
    ]
    if prepare_only:
        argv.append("--prepare-only")
    assert ktune_cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert events == [
        "import",
        "prepare",
        "bind",
        *([] if prepare_only else ["train"]),
        *(["fit"] if not prepare_only and status == "completed" else []),
    ]
    assert observed["dataset"] == Path(str(output) + "-upstream-data")
    assert observed["train_dataset"] == observed["dataset"]
    assert observed["config"].adapter_layers == [39]
    assert observed["config"].max_steps == 3
    assert observed["config"].prefix_cache_mib == 2048
    assert observed["config"].final_fit_examples == 2
    assert observed["config"].max_length == 3394
    assert observed["import_args"] == {
        "max_length": None,
        "max_examples": 10,
        "minimum_verification": "source_reconstructed",
    }
    assert result["status"] == ("prepared" if prepare_only else status)
    assert result["quality_certified"] is False and result["evaluation_scope"] == "training_only"


@pytest.mark.parametrize("ready", [False, True])
def test_finetune_upstream_no_update_before_successful_provenance_binding(
    tmp_path, monkeypatch, capsys, ready
):
    from exl3_mlx import upstream_data

    observed = []
    monkeypatch.setattr(
        upstream_data,
        "import_upstream",
        lambda *a, **k: {"ready_for_sft": ready, "max_length": 3394},
    )

    class Trainer:
        def __init__(self, **kwargs):
            assert ready

        def prepare(self, **kwargs):
            return {"data": {"max_length": 3394}}

        def bind_provenance(self, *args):
            raise ValueError("Evidence hash mismatch")

        def train(self, **kwargs):
            observed.append("unexpected update")

    monkeypatch.setattr(ktune_cli, "SFTTrainer", Trainer)
    assert (
        ktune_cli.main(
            [
                "finetune-upstream",
                "--source",
                "manifest.json",
                "--model",
                "model",
                "--output",
                str(tmp_path / "project"),
                "--minimum-verification",
                "execution_verified",
            ]
        )
        == 1
    )
    assert not observed
    assert "Evidence hash mismatch" in capsys.readouterr().err if ready else True
