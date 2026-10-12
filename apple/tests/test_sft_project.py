"""CPU project/API contracts; fake child reports never load a model or Metal."""

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src/exl3_mlx"
_PACKAGE = "exl3_sft_project_cpu_tests"
_namespace = types.ModuleType(_PACKAGE)
_namespace.__path__ = [str(_ROOT)]
sys.modules[_PACKAGE] = _namespace


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.{name}", _ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sft = _module("sft")
cli = _module("sft_cli")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _read(path):
    return json.loads(path.read_text())


@pytest.fixture
def project(tmp_path, monkeypatch):
    identity = {
        "source": {"sft.py": "fixed-test-code"},
        "dependency_lock": "fixed-test-lock",
        "runtime": {"mlx": "0.32.3", "mlx-lm": "0.32.0"},
        "python": "3.12",
    }
    monkeypatch.setattr(sft, "implementation_identity", lambda: copy.deepcopy(identity))
    model = tmp_path / "model"
    _write(
        model / "config.json",
        {
            "model_type": "deepseek_v41",
            "quantization": {"mode": "mxfp8"},
            "text_config": {"num_hidden_layers": 40, "index_topk": 512},
        },
    )
    for name in (
        "expert-manifest-v2.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "engram/engram-residents.safetensors",
    ):
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    root = tmp_path / "project"
    data = root / "data"
    data.mkdir(parents=True)
    for index, split in enumerate(("train", "valid", "test")):
        _write(data / f"{split}.jsonl", {"input_ids": [0, 2, 5 + index, 3, 6, 1], "loss_start": 4})
        # JSONL, unlike the generic pretty-JSON helper above, needs one record per line.
        (data / f"{split}.jsonl").write_text(
            json.dumps({"input_ids": [0, 2, 5 + index, 3, 6, 1], "loss_start": 4}) + "\n"
        )
        (data / f"raw-{split}.jsonl").write_text(
            json.dumps({"prompt": f"Question {split}", "completion": "blue"}) + "\n"
        )
    _write(data / "dataset.json", {"format": "exl3-mlx-dataset-v1"})
    args = sft.SFTConfig(output_dir=str(root), max_steps=10, max_new_tokens=4)
    trainer = sft.SFTTrainer(model=model, args=args, progress=lambda _: None)
    manifest = {
        "format": "exl3-sft-project-v1",
        "model": str(model.resolve()),
        "config": dataclasses.asdict(args),
        "capabilities": sft.capabilities(model),
        "implementation": identity,
        "data": {"seq_length": 5, "max_length": 6, "counts": {"train": 1, "valid": 1, "test": 1}},
        "data_files": {path.name: sft.digest(path) for path in data.iterdir()},
        "status": "prepared",
    }
    _write(root / "project.json", manifest)
    return trainer


def _checkpoint(trainer, step, *, path=None, contract_change=None, config_change=None):
    manifest = _read(trainer.root / "project.json")
    args = trainer.args
    path = path or trainer.root / "attempts" / "initial" / "train" / f"step-{step:06d}"
    path.mkdir(parents=True, exist_ok=True)
    expected_base = {
        k: manifest["capabilities"]["base_metadata"][k]
        for k in (
            "config.json",
            "expert-manifest-v2.json",
            "engram/engram-manifest.json",
            "engram-token-map.u32.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        )
    }
    config = {
        "format": "exl3-mlx-deepseek-short-text-v1",
        "base_identity": expected_base,
        "rank": args.rank,
        "alpha": args.lora_alpha,
        "adapter_layers": list(range(40)),
        "expert_adapter_layers": [],
        **(config_change or {}),
    }
    _write(path / "adapter_config.json", config)
    for name in ("adapter.safetensors", "optimizer.safetensors", "random.safetensors"):
        (path / name).write_bytes(f"fake {name} at step {step}".encode())
    contract = {
        "learning_rate": args.learning_rate,
        "grad_accumulation": args.gradient_accumulation_steps,
        "clip_grad": args.max_grad_norm,
        "seed": args.seed,
        "seq_length": manifest["data"]["seq_length"],
        "truncate": False,
        "chunk_size": args.chunk_size,
        "no_checkpoint": False,
        "data_sha256": manifest["data_files"]["train.jsonl"],
        "mlx": manifest["implementation"]["runtime"]["mlx"],
        "mlx_lm": manifest["implementation"]["runtime"]["mlx-lm"],
        "engine_semantics": sft.ENGINE_SEMANTICS,
        **(contract_change or {}),
    }
    validation = None
    if manifest["data"]["counts"]["valid"]:
        rows = [
            json.loads(line)
            for line in (trainer.root / "data/valid.jsonl").read_text().splitlines()
            if line.strip()
        ]
        validation = {
            "data_sha256": manifest["data_files"]["valid.jsonl"],
            "selection": "first examples in immutable dataset order",
            "requested_examples": len(rows),
            "examples": len(rows),
            "selected_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    state = {
        "format": "exl3-mlx-training-state-v1",
        "step": step,
        "cursor": step * args.gradient_accumulation_steps,
        "contract": contract,
        "selection": {"validation_identity": validation, "best": None},
        "files": {
            name: sft.digest(path / name)
            for name in (
                "adapter.safetensors",
                "adapter_config.json",
                "optimizer.safetensors",
                "random.safetensors",
            )
        },
    }
    _write(path / "trainer_state.json", state)
    return path


def _arg(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize(
    "change",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"gradient_accumulation_steps": 1.5},
        {"learning_rate": float("nan")},
        {"learning_rate": 0},
        {"max_grad_norm": float("inf")},
        {"per_device_train_batch_size": 2},
        {"per_device_train_batch_size": True},
        {"packing": True},
        {"loss_scope": "all_assistant"},
        {"max_length": 32769},
        {"max_length": True},
        {"max_new_tokens": 32769},
        {"seed": True},
        {"demo": 1},
        {"prompt_column": "question"},
    ],
)
def test_config_rejects_unsupported_or_invalid_settings(tmp_path, change):
    with pytest.raises((ValueError, TypeError)):
        sft.SFTConfig(output_dir=str(tmp_path), **change)


@pytest.mark.parametrize("memory", [0.5, 81])
def test_config_rejects_budget_outside_backend_supported_range(tmp_path, memory):
    with pytest.raises(ValueError):
        sft.SFTConfig(output_dir=str(tmp_path), memory_gib=memory)


def test_unknown_config_arguments_are_not_silently_accepted(tmp_path):
    with pytest.raises(TypeError):
        sft.SFTConfig(output_dir=str(tmp_path), warmup_steps=5)


def test_child_process_boundary_inherits_lock_and_persists_status(project, monkeypatch):
    captured = {}

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            captured.update(command=command, kwargs=kwargs)
            os.fstat(kwargs["pass_fds"][0])

        def wait(self):
            assert _read(project.root / "active.json")["running"] is True
            return 0

    monkeypatch.setattr(sft.subprocess, "Popen", Child)
    with sft.project_lock(project.root) as fd:
        project._lock_fd = fd
        project._child(
            ["doctor", "--model", Path(project.model)],
            label="preflight",
            directory=project.root / "attempts/test",
        )
        assert captured["kwargs"]["pass_fds"] == (fd,)
    assert captured["command"] == [
        sys.executable,
        "-m",
        "exl3_mlx.workflow",
        "doctor",
        "--model",
        project.model,
    ]
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stderr"] == subprocess.STDOUT
    assert _read(project.root / "active.json")["running"] is False


def test_cpu_child_retains_project_lock_after_parent_descriptor_closes(project):
    child = None
    try:
        with sft.project_lock(project.root) as fd:
            inode = (project.root / ".run.lock").stat().st_ino
            child = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.read()"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(fd,),
            )
        assert child.poll() is None
        with pytest.raises(RuntimeError, match="already running"):
            with sft.project_lock(project.root):
                pass
        child.communicate(b"finish", timeout=5)
        assert child.returncode == 0
        with sft.project_lock(project.root):
            assert (project.root / ".run.lock").stat().st_ino == inode
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


@pytest.mark.parametrize(
    "command,verbose_count",
    [
        (["train"], 1),
        (["evaluate"], 1),
        (["train", "--verbose"], 1),
        (["doctor"], 0),
        (["generate"], 0),
    ],
)
def test_child_records_training_and_evaluation_progress(
    project, monkeypatch, command, verbose_count
):
    captured = []
    original = list(command)

    class Child:
        pid = 12345

        def __init__(self, arguments, **kwargs):
            captured.extend(arguments)

        def wait(self):
            return 0

    monkeypatch.setattr(sft.subprocess, "Popen", Child)
    project._child(command, label="progress", directory=project.root / "logs")
    assert captured.count("--verbose") == verbose_count
    assert command == original


def test_total_step_resume_dispatches_only_remaining_updates(project, monkeypatch):
    first = _checkpoint(project, 7)
    commands = []

    def child(command, *, label, directory):
        commands.append(command)
        os.fstat(project._lock_fd)
        if command[0] == "train":
            assert _arg(command, "--resume") == first
            target = Path(_arg(command, "--output"))
            _checkpoint(project, 10, path=target / "step-000010")
            _write(target / "training.json", {"status": "completed", "completed_step": 10})

    monkeypatch.setattr(project, "_child", child)
    result = project.train(resume_from_checkpoint=True, compare=False)
    train = next(command for command in commands if command[0] == "train")
    assert _arg(train, "--steps") == 3
    assert _arg(train, "--seq-length") == 5
    assert result["completed_steps"] == 10
    assert result["status"] == "completed"


def test_same_target_resume_still_compares_after_prior_interruption(project, monkeypatch):
    _checkpoint(project, 10)
    manifest = _read(project.root / "project.json")
    manifest["status"] = "interrupted"
    _write(project.root / "project.json", manifest)
    calls = []
    monkeypatch.setattr(project, "_child", lambda *a, **k: pytest.fail("No new GPU work is needed"))
    monkeypatch.setattr(project, "evaluate", lambda: calls.append("compare"))
    result = project.train(resume_from_checkpoint=True)
    assert calls == ["compare"]
    assert result["completed_steps"] == 10


def test_lower_total_step_target_rejected_without_child(project, monkeypatch):
    _checkpoint(project, 11)
    monkeypatch.setattr(project, "_child", lambda *a, **k: pytest.fail("Must reject before child"))
    with pytest.raises(ValueError, match="total steps"):
        project.train(resume_from_checkpoint=True)


@pytest.mark.parametrize("change", [{"data_sha256": "foreign"}, {"learning_rate": 0.5}])
def test_copied_foreign_checkpoint_with_valid_file_hashes_is_rejected(project, change):
    path = _checkpoint(project, 3, contract_change=change)
    assert sft.checkpoint_state(path)["step"] == 3  # Integrity alone is insufficient.
    with pytest.raises(ValueError, match="does not belong"):
        project._recover()


def test_resigned_wrong_base_or_adapter_config_rejected(project):
    _checkpoint(project, 3, config_change={"rank": 64})
    with pytest.raises(ValueError, match="adapter settings differ"):
        project._recover()


def test_test_exposure_receipt_precedes_even_failed_evaluation(project, monkeypatch):
    _checkpoint(project, 1)
    seen = []

    def child(*args, **kwargs):
        seen.append(_read(project.root / "project.json")["test_exposure_count"])
        raise RuntimeError("simulated child failure")

    monkeypatch.setattr(project, "_child", child)
    for expected in (1, 2):
        with pytest.raises(RuntimeError, match="simulated"):
            project.evaluate()
        assert _read(project.root / "project.json")["test_exposure_count"] == expected
    assert seen == [1, 2]
    assert not (project.root / "comparison.json").exists()


@pytest.mark.parametrize("mutation", ["data", "model", "implementation", "config"])
def test_project_mutation_blocked_before_child(project, monkeypatch, mutation):
    if mutation == "data":
        (project.root / "data/raw-train.jsonl").write_text("changed")
    elif mutation == "model":
        (Path(project.model) / "chat_template.jinja").write_text("changed")
    elif mutation == "implementation":
        monkeypatch.setattr(sft, "implementation_identity", lambda: {"different": True})
    else:
        project.args.learning_rate = 0.01
    monkeypatch.setattr(project, "_child", lambda *a, **k: pytest.fail("Must reject before child"))
    with pytest.raises(ValueError, match="changed"):
        project.generate("Hello", base=True)


def test_export_is_private_inference_bundle_without_resume_receipt(project):
    source = _checkpoint(project, 1)
    before = sft.digest(source / "adapter_config.json")
    output = Path(project.save_model(project.root.parent / "export"))
    assert {path.name for path in output.iterdir()} == {
        "adapter.safetensors",
        "adapter_config.json",
        "bundle.json",
    }
    config = _read(output / "adapter_config.json")
    assert config["format"] == "exl3-mlx-deepseek-short-text-v1"
    assert config["adapter_sha256"] == sft.digest(output / "adapter.safetensors")
    receipt = _read(output / "bundle.json")
    assert receipt["format"] == "exl3-mlx-adapter-bundle-v1"
    assert receipt["external_compatibility"] is False
    assert receipt["source_checkpoint"] == str(source)
    assert sft.digest(source / "adapter_config.json") == before
    with pytest.raises(FileExistsError):
        project.save_model(output)


@pytest.fixture
def preview(project, monkeypatch):
    class Tokenizer:
        eos_token_id = 1

        @staticmethod
        def apply_chat_template(*args, **kwargs):
            return [0, 2, 5, 3]

        @staticmethod
        def decode(ids, *, skip_special_tokens=False):
            pieces = {1: "" if skip_special_tokens else "<eos>", 6: "blue", 7: " ", 8: " ", 9: " "}
            return "".join(pieces[token] for token in ids)

    data_module = types.ModuleType(f"{_PACKAGE}.data")
    data_module.load_tokenizer = lambda _: Tokenizer()
    monkeypatch.setitem(sys.modules, data_module.__name__, data_module)
    directory = project.root / "comparison"
    directory.mkdir()
    return directory


def _generation_result():
    return {
        "text": "blue",
        "stop_reason": "eos",
        "prompt_ids": [0, 2, 5, 3],
        "prompt_tokens": 4,
        "eos_token_id": 1,
        "max_new_tokens": 4,
        "generated_ids": [6, 1],
    }


def test_exact_preview_requires_eos_not_just_equal_text(project, preview, monkeypatch):
    def child(command, *, label, directory):
        result = _generation_result()
        if "--adapter" not in command:
            result.update(text="blue   ", stop_reason="max_tokens", generated_ids=[6, 7, 8, 9])
        _write(Path(_arg(command, "--report")), result)

    monkeypatch.setattr(project, "_child", child)
    result = project._paired_preview(
        {"prompt": "Q", "completion": "blue"}, Path("adapter"), preview
    )
    assert result["base"]["exact_answer_with_eos"] is False
    assert result["adapter"]["exact_answer_with_eos"] is True


@pytest.mark.parametrize(
    "invalid",
    [
        {"prompt_ids": [0, 2, 8, 3]},
        {"max_new_tokens": 5},
        {"eos_token_id": 2},
        {"generated_ids": [6, 7], "stop_reason": "eos"},
        {"generated_ids": [6, 1, 1], "stop_reason": "eos"},
        {"text": "red"},
    ],
)
def test_preview_rejects_same_wrong_inputs_or_false_eos_from_both_children(
    project, preview, monkeypatch, invalid
):
    def child(command, *, label, directory):
        _write(Path(_arg(command, "--report")), {**_generation_result(), **invalid})

    monkeypatch.setattr(project, "_child", child)
    with pytest.raises(ValueError):
        project._paired_preview({"prompt": "Q", "completion": "blue"}, Path("adapter"), preview)


def test_cli_configuration_flags_override_json_without_initializing_model(tmp_path):
    config = tmp_path / "config.json"
    _write(config, {"output_dir": str(tmp_path / "out"), "max_steps": 2, "learning_rate": 0.001})
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "finetune",
            "--model",
            "local-model",
            "--data",
            "data.csv",
            "--config",
            str(config),
            "--steps",
            "7",
        ]
    )
    trainer = cli.start_trainer(args, parser)
    assert trainer.args.max_steps == 7
    assert trainer.args.learning_rate == 0.001
    assert trainer.datasets[0] == "data.csv"


def test_cli_resume_preserves_total_steps_and_compare_choice(monkeypatch, capsys):
    observed = {}
    monkeypatch.setattr(cli.time, "time_ns", lambda: 123456789)

    class Trainer:
        @staticmethod
        def from_project(project, **kwargs):
            observed.update(project=project, **kwargs)
            return Trainer()

        @staticmethod
        def train(**kwargs):
            observed.update(kwargs)
            return {"status": "completed"}

    monkeypatch.setattr(cli, "SFTTrainer", Trainer)
    assert cli.main(["resume", "my-project", "--steps", "20", "--no-compare"]) == 0
    assert observed == {
        "project": "my-project",
        "max_steps": 20,
        "resume_from_checkpoint": True,
        "compare": False,
        "request_start_ns": 123456789,
    }
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_comparison_keeps_metrics_and_checkpoint_identity_distinct(project, monkeypatch):
    selected = _checkpoint(project, 1)

    def child(command, *, label, directory):
        result = {"loss": 1.0 if label == "adapter" else 2.0, "examples": 1}
        if label == "adapter":
            result["adapter"] = str(selected)
        _write(Path(_arg(command, "--report")), result)

    monkeypatch.setattr(project, "_child", child)
    monkeypatch.setattr(project, "_paired_preview", lambda *args: {"status": "fixture"})
    result = project.evaluate()
    assert result["selected_checkpoint"] == str(selected)
    assert result["adapter"]["loss"] == 1.0
    assert project.status()["comparison_is_current"] is True
    del result["selected_checkpoint"]
    _write(project.root / "comparison.json", result)
    assert project.status()["comparison_is_current"] is True  # Earlier receipt remains readable.
    _checkpoint(project, 2)
    assert project.status()["comparison_is_current"] is False


def test_training_fit_is_explicit_and_has_no_generation_or_quality_claim(project, monkeypatch):
    selected = _checkpoint(project, 1)
    seen = []

    def child(command, *, label, directory):
        seen.append(command)
        assert command[0] == "evaluate"
        assert Path(_arg(command, "--data")).name == "train.jsonl"
        _write(
            Path(_arg(command, "--report")),
            {"loss": 2.0 if label == "fit-base" else 1.8, "examples": 1, "supervised_tokens": 2},
        )

    monkeypatch.setattr(project, "_child", child)
    result = project.verify_training_fit()
    assert len(seen) == 2
    assert "--adapter" not in seen[0]
    assert _arg(seen[1], "--adapter") == selected
    assert result["quality_certified"] is False
    assert result["evaluation_scope"] == "training_only"
    assert result["loss_delta"] == pytest.approx(-0.2)
    assert result["training_data_sha256"] == sft.digest(project.root / "data/train.jsonl")
    assert (project.root / "fit.json").is_file()
    assert "test_exposure_count" not in _read(project.root / "project.json")


def test_external_evidence_is_bound_before_training_and_checked_on_resume(project, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ktune-import.json").write_text('{"scope":"training_only"}')
    (evidence / "ledger.jsonl").write_text('{"outcome":"failed"}\n')
    report = {
        "artifacts_sha256": {"ledger.jsonl": sft.digest(evidence / "ledger.jsonl")},
        "manifest_sha256": sft.digest(evidence / "ktune-import.json"),
    }
    project.bind_provenance(evidence, report)
    assert len(project._manifest()["provenance_files"]) == 3
    (project.root / "ktune-evidence/ledger.jsonl").write_text('{"outcome":"passed"}\n')
    with pytest.raises(ValueError, match="provenance changed"):
        sft.SFTTrainer.from_project(project.root)


def test_evidence_cannot_be_attached_after_checkpoint(project, tmp_path):
    _checkpoint(project, 1)
    with pytest.raises(ValueError, match="before training"):
        project.bind_provenance(tmp_path, {})


def _evidence(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "ktune-import.json").write_text('{"scope":"training_only"}')
    (evidence / "ledger.jsonl").write_text('{"outcome":"failed"}\n')
    report = {
        "artifacts_sha256": {"ledger.jsonl": sft.digest(evidence / "ledger.jsonl")},
        "manifest_sha256": sft.digest(evidence / "ktune-import.json"),
    }
    return evidence, report


def test_interrupted_evidence_copy_is_durable_blocks_training_and_recovers(
    project, tmp_path, monkeypatch
):
    evidence, report = _evidence(tmp_path)
    original_copy = sft.shutil.copyfile
    copied = []

    def interrupted_copy(source, destination):
        saved = _read(project.root / "project.json")
        assert saved["provenance_binding"]["state"] == "pending"
        assert saved["status"] == "provenance_pending"
        copied.append(source)
        if len(copied) == 2:
            Path(destination).write_text("partial copy")
            raise KeyboardInterrupt
        return original_copy(source, destination)

    monkeypatch.setattr(sft.shutil, "copyfile", interrupted_copy)
    with pytest.raises(KeyboardInterrupt):
        project.bind_provenance(evidence, report)
    assert len(copied) == 2
    with pytest.raises(ValueError, match="binding is pending"):
        sft.SFTTrainer.from_project(project.root)
    repair = sft.SFTTrainer.from_project(project.root, allow_pending_provenance=True)
    monkeypatch.setattr(repair, "_child", lambda *a, **kw: pytest.fail("Must not launch a child"))
    with pytest.raises(ValueError, match="binding is pending"):
        repair.train(compare=False)
    monkeypatch.setattr(sft.shutil, "copyfile", original_copy)
    repair.bind_provenance(evidence, report)
    manifest = repair._manifest()
    assert manifest["provenance_binding"]["state"] == "complete"
    assert manifest["status"] == "prepared"
    assert len(manifest["provenance_files"]) == 3
    assert not list(project.root.glob(".ktune-evidence-*"))


def test_failed_evidence_hash_pins_exact_report_for_recovery(project, tmp_path):
    evidence, report = _evidence(tmp_path)
    original = (evidence / "ledger.jsonl").read_bytes()
    (evidence / "ledger.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Import changed before"):
        project.bind_provenance(evidence, report)
    changed = copy.deepcopy(report)
    changed["artifacts_sha256"]["ledger.jsonl"] = sft.digest(evidence / "ledger.jsonl")
    with pytest.raises(ValueError, match="same import report"):
        project.bind_provenance(evidence, changed)
    with pytest.raises(ValueError, match="same import report"):
        project.bind_provenance(evidence, {**report, "scope": "changed"})
    (evidence / "ledger.jsonl").write_bytes(original)
    project.bind_provenance(evidence, report)
    assert project._manifest()["status"] == "prepared"


@pytest.mark.parametrize("tamper", [False, True])
def test_published_evidence_recovers_without_recopy_but_rejects_tamper(
    project, tmp_path, monkeypatch, tamper
):
    evidence, report = _evidence(tmp_path)
    original_write = sft.write_json

    def interrupted_write(path, value):
        if (
            Path(path).name == "project.json"
            and value.get("provenance_binding", {}).get("state") == "complete"
        ):
            raise OSError("simulated final manifest failure")
        return original_write(path, value)

    monkeypatch.setattr(sft, "write_json", interrupted_write)
    with pytest.raises(OSError, match="simulated"):
        project.bind_provenance(evidence, report)
    assert (project.root / "ktune-evidence").is_dir()
    assert _read(project.root / "project.json")["provenance_binding"]["state"] == "pending"
    monkeypatch.setattr(sft, "write_json", original_write)
    monkeypatch.setattr(
        sft.shutil, "copyfile", lambda *a, **kw: pytest.fail("Must verify published files")
    )
    if tamper:
        (project.root / "ktune-evidence/ledger.jsonl").write_text("changed")
        with pytest.raises(ValueError, match="Published provenance changed"):
            project.bind_provenance(evidence, report)
        with pytest.raises(ValueError, match="binding is pending"):
            project._manifest()
    else:
        # The already published, hash-verified evidence is enough to finish recovery.
        sft.shutil.rmtree(evidence)
        project.bind_provenance(evidence, report)
        assert project._manifest()["provenance_binding"]["state"] == "complete"


@pytest.mark.parametrize("stage_kind", ["symlink", "escape"])
def test_pending_evidence_never_deletes_unowned_paths(project, tmp_path, stage_kind):
    evidence, report = _evidence(tmp_path)
    (evidence / "ledger.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Import changed before"):
        project.bind_provenance(evidence, report)
    saved = _read(project.root / "project.json")
    stage = project.root / saved["provenance_binding"]["stage"]
    stage.rmdir()
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "keep").write_text("keep me")
    if stage_kind == "symlink":
        stage.symlink_to(outside, target_is_directory=True)
        error = "must not follow symlinks"
    else:
        saved["provenance_binding"]["stage"] = "../unrelated"
        _write(project.root / "project.json", saved)
        error = "Invalid provenance staging identity"
    with pytest.raises(ValueError, match=error):
        project.bind_provenance(evidence, report)
    assert (outside / "keep").read_text() == "keep me"


def test_required_provenance_blocks_gap_before_binder_starts(project, tmp_path, monkeypatch):
    manifest = _read(project.root / "project.json")
    manifest["provenance_required"] = True
    _write(project.root / "project.json", manifest)
    with pytest.raises(ValueError, match="requires provenance binding"):
        sft.SFTTrainer.from_project(project.root)
    repair = sft.SFTTrainer.from_project(project.root, allow_pending_provenance=True)
    monkeypatch.setattr(repair, "_child", lambda *a, **kw: pytest.fail("Must not launch a child"))
    with pytest.raises(ValueError, match="requires provenance binding"):
        repair.train(compare=False)
    evidence, report = _evidence(tmp_path)
    repair.bind_provenance(evidence, report)
    assert sft.SFTTrainer.from_project(project.root)._manifest()["provenance_required"] is True


def test_prepare_publishes_required_provenance_in_initial_manifest(project, tmp_path, monkeypatch):
    data_module = _module("sft_data")

    def prepare_data(source, model, output, **kwargs):
        sft.shutil.copytree(project.root / "data", output)
        report = {
            "source": str(source),
            "source_files_sha256": {str(source): "fixture"},
            "preview": [],
            "counts": {"train": 1, "valid": 1, "test": 1},
            "seq_length": 5,
            "max_length": 6,
        }
        _write(output / "dataset.json", report)
        (output / "provenance.jsonl").write_text(
            json.dumps({"origin": str(source), "source_file": str(source)}) + "\n"
        )
        return report

    monkeypatch.setattr(data_module, "prepare_sft_data", prepare_data)
    fresh = sft.SFTTrainer(
        model=project.model,
        args=sft.SFTConfig(output_dir=str(tmp_path / "required-project"), chunk_size=8192),
        train_dataset=project.root / "data/train.jsonl",
        progress=lambda _: None,
    )
    manifest = fresh.prepare(provenance_required=True)
    assert manifest["provenance_required"] is True
    assert _read(fresh.root / "project.json")["provenance_required"] is True
    assert manifest["config"]["chunk_size"] == 8192
    assert "provenance_binding" not in manifest
    with pytest.raises(ValueError, match="requires provenance binding"):
        sft.SFTTrainer.from_project(fresh.root)


def test_resource_profile_is_explicit_and_all_limits_reach_children(tmp_path):
    low = sft.SFTConfig(output_dir=str(tmp_path / "low"))
    assert (low.memory_gib, low.decoded_cache_mib, low.packed_cache_mib) == (16, 0, 128)
    high = sft.SFTConfig(output_dir=str(tmp_path / "high"), profile="development")
    assert (high.memory_gib, high.reserve_gib) == (80, 32)
    assert (high.packed_cache_mib, high.decoded_cache_mib, high.allocator_cache_mib) == (
        8192,
        0,
        256,
    )
    trainer = sft.SFTTrainer(model="unused", args=high)
    flags = trainer._resources()
    for key in sft.RESOURCE_FIELDS:
        assert _arg(flags, "--" + key.replace("_", "-")) == getattr(high, key)


def test_resource_only_resume_preserves_math_contract_and_records_attempt(project, monkeypatch):
    first = _checkpoint(project, 7)
    original_contract = _read(first / "trainer_state.json")["contract"]
    trainer = sft.SFTTrainer.from_project(
        project.root, max_steps=8, resources={"profile": "development", "decoded_cache_mib": 16384}
    )
    assert trainer.args.memory_gib == 80
    assert trainer.args.decoded_cache_mib == 16384

    def child(command, *, label, directory):
        assert _arg(command, "--memory-gib") == 80
        assert _arg(command, "--decoded-cache-mib") == 16384
        recorded = _read(Path(directory) / "resources.json")
        assert recorded["profile"] == "development"
        if label == "training":
            target = Path(_arg(command, "--output"))
            _checkpoint(trainer, 8, path=target / "step-000008")
            _write(target / "training.json", {"status": "completed", "completed_step": 8})

    monkeypatch.setattr(trainer, "_child", child)
    trainer.train(resume_from_checkpoint=True, compare=False)
    saved = sft.SFTTrainer.from_project(project.root)
    assert saved.args.memory_gib == 80
    assert saved.args.decoded_cache_mib == 16384
    assert _read(saved._recover()[1] / "trainer_state.json")["contract"] == original_contract
    with pytest.raises(ValueError, match="Only resource"):
        sft.SFTTrainer.from_project(project.root, resources={"learning_rate": 1e-3})


def test_cli_development_profile_allows_explicit_cache_override(tmp_path):
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
            "--profile",
            "development",
            "--decoded-cache-mib",
            "1024",
        ]
    )
    trainer = cli.start_trainer(args, parser)
    assert trainer.args.memory_gib == 80
    assert trainer.args.decoded_cache_mib == 1024

    config = tmp_path / "low-config.json"
    _write(config, dataclasses.asdict(sft.SFTConfig(output_dir=str(tmp_path / "new-project"))))
    args = parser.parse_args(
        [
            "finetune",
            "--model",
            "M",
            "--data",
            "D",
            "--config",
            str(config),
            "--profile",
            "development",
            "--decoded-cache-mib",
            "1024",
        ]
    )
    trainer = cli.start_trainer(args, parser)
    assert trainer.args.memory_gib == 80
    assert trainer.args.packed_cache_mib == 8192
    assert trainer.args.decoded_cache_mib == 1024


@pytest.mark.parametrize("chunk", [0, 127, 129, 32769, True, 512.0, float("nan")])
def test_invalid_project_chunk_size_is_rejected(tmp_path, chunk):
    with pytest.raises(ValueError, match="chunk_size"):
        sft.SFTConfig(output_dir=str(tmp_path), chunk_size=chunk)


def test_chunk_choice_is_independent_of_resource_profile_and_cli_explicit(tmp_path):
    for profile in ("low-memory", "development"):
        assert sft.SFTConfig(output_dir=str(tmp_path), profile=profile).chunk_size == 512
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
            "--profile",
            "development",
            "--chunk-size",
            "8192",
        ]
    )
    assert cli.start_trainer(args, parser).args.chunk_size == 8192
    with pytest.raises(SystemExit):
        parser.parse_args(["resume", str(tmp_path), "--steps", "2", "--chunk-size", "8192"])


def test_saved_chunk_reaches_all_children_and_cannot_change_on_resume(
    project, preview, monkeypatch, tmp_path
):
    project.args.chunk_size = 8192
    project.args.max_steps = 1
    manifest = _read(project.root / "project.json")
    manifest["config"] = dataclasses.asdict(project.args)
    _write(project.root / "project.json", manifest)
    seen = set()

    def child(command, *, label, directory):
        seen.add(command[0])
        assert _arg(command, "--chunk-size") == 8192
        if command[0] == "train":
            output = Path(_arg(command, "--output"))
            _checkpoint(project, 1, path=output / "step-000001")
            _write(output / "training.json", {"status": "completed", "completed_step": 1})
        elif command[0] == "evaluate":
            _write(
                Path(_arg(command, "--report")),
                {"loss": 1.0, "examples": 1, "supervised_tokens": 2},
            )
        elif command[0] == "generate":
            _write(Path(_arg(command, "--report")), _generation_result())

    monkeypatch.setattr(project, "_child", child)
    project.train(compare=False)
    project.verify_training_fit()
    project.evaluate()
    project.generate("Q")
    assert seen == {"doctor", "train", "evaluate", "generate"}
    assert project._checkpoints()[0][2]["contract"]["chunk_size"] == 8192
    exported = Path(project.save_model(tmp_path / "chunk-export"))
    assert _read(exported / "bundle.json")["engine_settings"]["chunk_size"] == 8192
    with pytest.raises(ValueError, match="Only resource"):
        sft.SFTTrainer.from_project(project.root, resources={"chunk_size": 512})
    project.args.chunk_size = 512
    with pytest.raises(ValueError, match="Project settings changed"):
        project.train(resume_from_checkpoint=True, compare=False)
    with pytest.raises(ValueError, match="Checkpoint does not belong"):
        project._checkpoints()
