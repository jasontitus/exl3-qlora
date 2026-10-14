"""CPU-only snapshot, preparation and phase gate tests; no model execution."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_agent_corpus_run.py"
spec = importlib.util.spec_from_file_location("prepare_agent_experiment", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def snapshot(tmp_path):
    root = tmp_path / "runtime"
    for name in ["src/exl3_mlx/sft.py", "src/exl3_mlx/context.py", "uv.lock", "pyproject.toml"]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# pinned\n")
    files = {str(p.relative_to(root)): helper.digest(p) for p in root.rglob("*") if p.is_file()}
    dump(root / "snapshot.json", files)
    return root


def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    rows = {"train": [{"text": "one"}, {"text": "two"}], "valid": [], "test": [{"text": "test"}]}
    for name, values in rows.items():
        (root / (name + ".jsonl")).write_text("".join(json.dumps(row) + "\n" for row in values))
    report = {
        "format": "exl3-agent-kernel-corpus-v1",
        "truncated": 0,
        "counts": {k: len(v) for k, v in rows.items()},
        "max_length": 8,
        "artifacts_sha256": {p.name: helper.digest(p) for p in root.iterdir()},
    }
    dump(root / "ktune-import.json", report)
    return root


def test_runtime_exact_snapshot_and_new_source_guard(tmp_path):
    root = snapshot(tmp_path)
    assert helper.verify_runtime(root)[0] == root
    (root / "src/exl3_mlx/new.py").write_text("# unbound")
    with pytest.raises(ValueError, match="omits"):
        helper.verify_runtime(root)


@pytest.mark.parametrize("mutation", ["bytes", "escape", "symlink"])
def test_runtime_tamper_rejected(tmp_path, mutation):
    root = snapshot(tmp_path)
    p = root / "src/exl3_mlx/sft.py"
    if mutation == "bytes":
        p.write_text("changed")
    elif mutation == "escape":
        data = helper.read_json(root / "snapshot.json")
        data["../outside"] = "0" * 64
        dump(root / "snapshot.json", data)
    else:
        p.unlink()
        outside = tmp_path / "outside.py"
        outside.write_text("# pinned\n")
        p.symlink_to(outside)
    with pytest.raises(ValueError):
        helper.verify_runtime(root)


@pytest.mark.parametrize("mutation", ["bytes", "validation", "length", "count", "truncated"])
def test_corpus_integrity_admission_guards(tmp_path, mutation):
    root = corpus(tmp_path)
    assert helper.corpus_inputs(root, 8)[2] == {"train": 2, "valid": 0, "test": 1}
    report = helper.read_json(root / "ktune-import.json")
    if mutation == "bytes":
        (root / "train.jsonl").write_text('{"changed":true}\n')
    elif mutation == "validation":
        (root / "valid.jsonl").write_text('{"text":"leak"}\n')
        report["counts"]["valid"] = 1
        report["artifacts_sha256"]["valid.jsonl"] = helper.digest(root / "valid.jsonl")
    elif mutation == "length":
        report["max_length"] = 9
    elif mutation == "count":
        report["counts"]["train"] = 3
    else:
        report["truncated"] = 1
    dump(root / "ktune-import.json", report)
    with pytest.raises(ValueError):
        helper.corpus_inputs(root, 8)


def prepare_fixture(tmp_path, monkeypatch, *, changed_count=False):
    from exl3_mlx.sft import RESOURCE_FIELDS, SFTConfig, SFTTrainer

    runtime = snapshot(tmp_path)
    source = corpus(tmp_path)
    events = []

    class Trainer:
        _engine_options = SFTTrainer._engine_options
        _resources = SFTTrainer._resources

        def __init__(self, *, model, args, train_dataset):
            self.model, self.args, self.root = (
                str(Path(model).resolve()),
                args,
                Path(args.output_dir),
            )

        def prepare(self, *, provenance_required):
            events.append(("prepare", provenance_required))
            (self.root / "data").mkdir(parents=True)
            self.manifest = {
                "config": dataclasses.asdict(self.args),
                "data": {
                    "counts": {"train": 1 if changed_count else 2, "valid": 0, "test": 1},
                    "duplicates_removed": int(changed_count),
                    "seq_length": 7,
                },
                "implementation": {"source": {}, "runtime": {"mlx": "a", "mlx-lm": "b"}},
            }
            for name, lengths in [("train", [5, 8]), ("valid", []), ("test", [6])]:
                (self.root / "data" / (name + ".jsonl")).write_text(
                    "".join(
                        json.dumps({"input_ids": list(range(n)), "loss_start": 2}) + "\n"
                        for n in lengths
                    )
                )
            dump(self.root / "project.json", self.manifest)
            return self.manifest

        def bind_provenance(self, source, report):
            events.append(("bind", report["manifest_sha256"]))

        def _manifest(self):
            return self.manifest

    fake = SimpleNamespace(
        SFTConfig=SFTConfig,
        SFTTrainer=Trainer,
        RESOURCE_FIELDS=RESOURCE_FIELDS,
        implementation_identity=lambda: {"source": {}, "runtime": {"mlx": "a", "mlx-lm": "b"}},
    )
    monkeypatch.setattr(
        helper, "activate", lambda r: (runtime, helper.digest(runtime / "snapshot.json"), fake)
    )
    args = SimpleNamespace(
        runtime=str(runtime),
        corpus=str(source),
        model=str(tmp_path / "model"),
        project=str(tmp_path / "project"),
        plan=str(tmp_path / "plan.json"),
        prefix_disk_cache=str(tmp_path / "cache"),
        max_length=8,
        memory_gib=86,
    )
    return args, events


def test_prepare_bound_fixed_recipe_and_commands_without_execution(tmp_path, monkeypatch):
    args, events = prepare_fixture(tmp_path, monkeypatch)
    result = helper.prepare(args)
    plan = helper.read_json(args.plan)
    assert result["counts"]["train"] == 2
    assert events[0] == ("prepare", True) and events[1][0] == "bind"
    assert plan["config"]["max_steps"] == 4 and plan["config"]["comparison_examples"] == 1
    assert plan["config"]["adapter_layers"] == [39]
    assert plan["config"]["memory_gib"] == 86
    assert plan["config"]["chunk_size"] == 2048
    assert plan["longest"]["train"]["index"] == 1
    assert plan["order"] == ["preflight-train", "preflight-test", "baseline", "train", "adapter"]
    assert "--no-compare" in plan["commands"]["train"]
    for phase in ["baseline", "adapter", "preflight-test"]:
        assert "--token-metrics" in plan["commands"][phase]
    assert plan["commands"]["adapter"][-1] == "<FIXED_FINAL_CHECKPOINT>"
    assert not (Path(args.project) / "attempts").exists()


def test_changed_counts_fail_before_provenance_binding(tmp_path, monkeypatch):
    args, events = prepare_fixture(tmp_path, monkeypatch, changed_count=True)
    with pytest.raises(ValueError, match="changed corpus counts"):
        helper.prepare(args)
    assert events == [("prepare", True)]
    assert not Path(args.plan).exists()


def test_existing_plan_not_overwritten(tmp_path, monkeypatch):
    args, events = prepare_fixture(tmp_path, monkeypatch)
    Path(args.plan).write_text("sentinel")
    with pytest.raises(FileExistsError):
        helper.prepare(args)
    assert Path(args.plan).read_text() == "sentinel" and not events


def gate_fixture(tmp_path, monkeypatch):
    from exl3_mlx.context import ENGINE_SEMANTICS

    root = tmp_path / "project"
    folder = root / "experiment"
    checkpoint = folder / "preflight-train/step-000001"
    selected = root / "attempts/attempt-1/train/step-000004"
    config = {"learning_rate": 5e-5, "max_steps": 4, "memory_gib": 86}
    base = {
        name: "1" * 64
        for name in (
            "config.json",
            "expert-manifest-v2.json",
            "engram/engram-manifest.json",
            "engram-token-map.u32.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        )
    }
    resources = {"memory_gib": 86}
    contract = {
        "learning_rate": 5e-5,
        "grad_accumulation": 1,
        "clip_grad": 1.0,
        "seed": 8,
        "seq_length": 7,
        "truncate": False,
        "chunk_size": 2048,
        "no_checkpoint": False,
        "data_sha256": "a" * 64,
        "mlx": "a",
        "mlx_lm": "b",
        "engine_semantics": ENGINE_SEMANTICS,
    }
    state = {"step": 1, "cursor": 1, "contract": contract}
    report = {
        "status": "completed",
        "completed_step": 1,
        "requested_steps": 1,
        "history": [{}],
        "contract": contract,
        "activation_mode": "checkpoint",
        "resources": resources,
        "checkpoint": str(checkpoint),
    }
    dump(folder / "preflight-train/training.json", report)
    dump(
        checkpoint / "adapter_config.json",
        {
            "base_identity": base,
            "rank": 4,
            "alpha": 8.0,
            "adapter_layers": [39],
            "expert_adapter_layers": [],
            "activation_mode": "checkpoint",
        },
    )
    plan_path = tmp_path / "plan.json"
    dump(plan_path, {"fixed": True})
    baseline = folder / "baseline.json"
    dump(baseline, {"evidence": True})
    plan = {
        "project": str(root),
        "counts": {"train": 2},
        "config": config,
        "resources": resources,
        "implementation": {"runtime": {"mlx": "a", "mlx-lm": "b"}},
        "longest": {"train": {"sha256": "a" * 64}},
        "commands": {
            name: ["python", "-m", "module", name]
            for name in ["preflight-train", "preflight-test", "baseline", "train", "adapter"]
        },
        "cwd": str(tmp_path),
        "environment": {"PYTHONPATH": "runtime/src"},
    }
    observations = []
    monkeypatch.setattr(
        helper,
        "verify_evaluation",
        lambda plan, trainer, manifest, phase: (
            observations.append(phase) or baseline,
            {"evaluation_context": {"process_id": 12345}},
        ),
    )
    trainer = SimpleNamespace(
        root=root, model="model", _checkpoints=lambda: [], selected_checkpoint=lambda: selected
    )
    sft = SimpleNamespace(
        checkpoint_state=lambda path: {"step": 4, "cursor": 4} if path == selected else state
    )
    manifest = {
        "data": {"seq_length": 7},
        "status": "prepared",
        "capabilities": {"base_metadata": base},
    }
    args = SimpleNamespace(plan=str(plan_path), phase="train")
    return args, plan, trainer, manifest, sft, state, observations


def test_train_command_records_validated_baseline_before_any_update(tmp_path, monkeypatch):
    args, plan, trainer, manifest, sft, _, seen = gate_fixture(tmp_path, monkeypatch)
    result = helper._emit_locked(args, plan, trainer, manifest, sft)
    assert result["phase"] == "train"
    assert seen == ["preflight-test", "baseline"]
    guard = helper.read_json(trainer.root / "experiment/baseline-before-training.json")
    assert guard["process_id"] == 12345 and guard["plan_sha256"] == helper.digest(args.plan)


@pytest.mark.parametrize("mutation", ["step", "cursor", "learning_rate", "resources", "mode"])
def test_preflight_mismatch_blocks_training(tmp_path, monkeypatch, mutation):
    args, plan, trainer, manifest, sft, state, seen = gate_fixture(tmp_path, monkeypatch)
    if mutation == "step":
        state["step"] = 0
    elif mutation == "cursor":
        state["cursor"] = 0
    elif mutation == "learning_rate":
        state["contract"]["learning_rate"] = 1.0
    else:
        path = trainer.root / "experiment/preflight-train/training.json"
        report = helper.read_json(path)
        if mutation == "resources":
            report["resources"]["memory_gib"] = 90
        else:
            report["activation_mode"] = "bf16"
        dump(path, report)
    with pytest.raises(ValueError):
        helper._emit_locked(args, plan, trainer, manifest, sft)
    assert not seen


def test_checkpoint_before_baseline_authorization_rejected(tmp_path, monkeypatch):
    args, plan, trainer, manifest, sft, _, _ = gate_fixture(tmp_path, monkeypatch)
    trainer._checkpoints = lambda: [(1, trainer.root / "step", {})]
    with pytest.raises(ValueError, match="Missing baseline binding"):
        helper._emit_locked(args, plan, trainer, manifest, sft)
    args.phase = "baseline"
    with pytest.raises(ValueError, match="must precede"):
        helper._emit_locked(args, plan, trainer, manifest, sft)


def test_post_only_fixed_final_completed_checkpoint(tmp_path, monkeypatch):
    args, plan, trainer, manifest, sft, _, _ = gate_fixture(tmp_path, monkeypatch)
    helper._emit_locked(args, plan, trainer, manifest, sft)
    args.phase = "adapter"
    with pytest.raises(ValueError, match="fixed final"):
        helper._emit_locked(args, plan, trainer, manifest, sft)
    manifest["status"] = "completed"
    result = helper._emit_locked(args, plan, trainer, manifest, sft)
    assert result["argv"][-1] == str(trainer.selected_checkpoint())
    dump(trainer.root / "experiment/baseline.json", {"modified": True})
    with pytest.raises(ValueError, match="baseline binding changed"):
        helper._emit_locked(args, plan, trainer, manifest, sft)


def test_terminal_adapter_verify_requires_report_validation(tmp_path, monkeypatch):
    import contextlib

    args, plan, trainer, manifest, sft, _, _ = gate_fixture(tmp_path, monkeypatch)
    args.phase = "adapter"
    target = trainer.selected_checkpoint()
    target.mkdir(parents=True)
    dump(target / "trainer_state.json", {"step": 4, "cursor": 4})
    dump(trainer.root / "project.json", {"status": "completed"})
    dump(trainer.root / "experiment/adapter.json", {"result": "validated below"})
    trainer._manifest = lambda: manifest
    sft.project_lock = lambda root: contextlib.nullcontext()
    monkeypatch.setattr(helper, "load_plan", lambda path: (plan, trainer, manifest, sft))
    concrete = ["python", "-m", "exl3_mlx.workflow", "evaluate", "--adapter", str(target)]
    monkeypatch.setattr(helper, "_emit_locked", lambda *args: {"argv": concrete})
    seen = []

    def verify(*args, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(helper, "verify_evaluation", verify)
    report = helper.verify_phase(args)
    assert report["verified_complete"] is True
    assert seen == [{"command": concrete, "adapter": target}]
    assert str(trainer.root / "experiment/adapter.json") in report["artifacts_sha256"]

    def malformed(*args, **kwargs):
        raise ValueError("bad terminal evidence")

    monkeypatch.setattr(helper, "verify_evaluation", malformed)
    with pytest.raises(ValueError, match="bad terminal evidence"):
        helper.verify_phase(args)


def test_explicit_cache_is_bound_to_all_direct_phases(tmp_path, monkeypatch):
    args, _ = prepare_fixture(tmp_path, monkeypatch)
    args.packed_cache_mib = 6144
    helper.prepare(args)
    plan = helper.read_json(args.plan)
    assert plan['config']['packed_cache_mib'] == plan['resources']['packed_cache_mib'] == 6144
    assert plan['config']['memory_gib'] == 86
    assert plan['config']['reserve_gib'] == 8
    for phase in ('preflight-train', 'preflight-test', 'baseline', 'adapter'):
        command = plan['commands'][phase]
        assert command.count('--packed-cache-mib') == 1
        assert command[command.index('--packed-cache-mib') + 1] == '6144'
    assert plan['config']['decoded_cache_mib'] == 0


@pytest.mark.parametrize('quota', [-1, 1.5, True, 1000000])
def test_invalid_cache_does_not_create_project(tmp_path, monkeypatch, quota):
    args, events = prepare_fixture(tmp_path, monkeypatch)
    args.packed_cache_mib = quota
    with pytest.raises(ValueError):
        helper.prepare(args)
    assert not Path(args.plan).exists()
    assert not Path(args.project).exists()
    assert not events
