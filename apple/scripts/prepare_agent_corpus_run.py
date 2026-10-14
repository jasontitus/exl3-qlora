"""CPU-only preparation and gated command emission for the native edit experiment.

No command is executed here. A supplied runtime must already be frozen. Emit and
run phases serially: preflight-train, preflight-test, baseline, train, adapter.
The disposable one-update preflight is never a warm start for the final project.
"""

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Nonfinite JSON: {value}")

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def verify_runtime(runtime):
    root = Path(runtime).expanduser().resolve(strict=True)
    snapshot = root / "snapshot.json"
    files = read_json(snapshot)
    if not isinstance(files, dict) or not files:
        raise ValueError("Expected a frozen runtime snapshot mapping")
    required = {"pyproject.toml", "uv.lock"} | {
        str(p.relative_to(root))
        for p in (root / "src/exl3_mlx").iterdir()
        if p.suffix in {".py", ".metal", ".html"}
    }
    if not required <= files.keys() or "src/exl3_mlx/sft.py" not in required:
        raise ValueError("Runtime snapshot omits implementation files")
    for name, checksum in files.items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise ValueError("Invalid runtime snapshot entry")
        if not path.resolve().is_relative_to(root) or any(
            p.is_symlink() for p in [path, *path.parents] if p != root.parent
        ):
            raise ValueError("Runtime snapshot must contain regular pinned files")
        if not path.is_file() or digest(path) != checksum:
            raise ValueError("Runtime snapshot changed: " + name)
    return root, digest(snapshot)


def activate(runtime):
    root, checksum = verify_runtime(runtime)
    if any(name == "exl3_mlx" or name.startswith("exl3_mlx.") for name in sys.modules):
        raise ValueError("Start a fresh Python process before selecting the pinned runtime")
    sys.path.insert(0, str(root / "src"))
    import exl3_mlx.sft as sft

    if Path(sft.__file__).resolve() != root / "src/exl3_mlx/sft.py":
        raise ValueError("Wrong runtime imported")
    return root, checksum, sft


def corpus_inputs(corpus, max_length):
    if type(max_length) is not int or not 3 <= max_length <= 32768:
        raise ValueError("Choose a complete-context ceiling in3..32768 before scoring")
    root = Path(corpus).expanduser().resolve(strict=True)
    report = read_json(root / "ktune-import.json")
    if report.get("format") != "exl3-agent-kernel-corpus-v1" or report.get("truncated") != 0:
        raise ValueError("Expected a nontruncated native edit corpus")
    hashes = report.get("artifacts_sha256", {})
    if not {"train.jsonl", "valid.jsonl", "test.jsonl"} <= hashes.keys():
        raise ValueError("Corpus must bind all explicit split files")
    for name, checksum in hashes.items():
        if (
            Path(name).name != name
            or name in {".", ".."}
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise ValueError("Invalid corpus artifact identity")
        if (root / name).is_symlink() or digest(root / name) != checksum:
            raise ValueError("Corpus artifact changed: " + name)
    rows = {
        name: [
            json.loads(line)
            for line in (root / (name + ".jsonl")).read_text().splitlines()
            if line.strip()
        ]
        for name in ("train", "valid", "test")
    }
    counts = {name: len(value) for name, value in rows.items()}
    if (
        counts != report.get("counts")
        or not counts["train"]
        or not counts["test"]
        or counts["valid"]
    ):
        raise ValueError("Require nonempty train/test and empty validation with exact bound counts")
    if type(report.get("max_length")) is not int or report["max_length"] > max_length:
        raise ValueError("Corpus exceeds the nominated qualified context ceiling; never truncate")
    return root, {**report, "manifest_sha256": digest(root / "ktune-import.json")}, counts


def argv(python, module, values):
    return [str(python), "-m", module, *map(str, values)]


def commands_for(trainer, manifest, selections, python):
    project, config = trainer.root, trainer.args
    folder = project / "experiment"
    counts = manifest["data"]["counts"]
    common = [
        "--model",
        trainer.model,
        "--seq-length",
        manifest["data"]["seq_length"],
        *trainer._engine_options(),
        "--verbose",
    ]
    baseline = [
        "evaluate",
        *common,
        "--data",
        project / "data/test.jsonl",
        "--max-examples",
        counts["test"],
        "--token-metrics",
        "--report",
        folder / "baseline.json",
    ]
    pretest = [
        "evaluate",
        *common,
        "--data",
        selections["test"]["path"],
        "--max-examples",
        1,
        "--token-metrics",
        "--report",
        folder / "preflight-test.json",
    ]
    # Training accepts the same explicit placement, but has no --report flag.
    pretrain = [
        "train",
        *common,
        "--data",
        selections["train"]["path"],
        "--output",
        folder / "preflight-train",
        "--steps",
        1,
        "--rank",
        4,
        "--alpha",
        8,
        "--learning-rate",
        5e-5,
        "--grad-accumulation",
        1,
        "--clip-grad",
        1,
        "--save-every",
        1,
        "--seed",
        8,
    ]
    commands = {
        "preflight-train": argv(python, "exl3_mlx.workflow", pretrain),
        "preflight-test": argv(python, "exl3_mlx.workflow", pretest),
        "baseline": argv(python, "exl3_mlx.workflow", baseline),
        "train": argv(
            python,
            "exl3_mlx.sft_cli",
            ["resume", project, "--steps", config.max_steps, "--no-compare"],
        ),
        "adapter": argv(
            python,
            "exl3_mlx.workflow",
            [*baseline[:-1], folder / "adapter.json", "--adapter", "<FIXED_FINAL_CHECKPOINT>"],
        ),
    }
    return commands


def prepare(args):
    plan_path = Path(args.plan).expanduser().resolve()
    project = Path(args.project).expanduser().resolve()
    if plan_path.exists() or project.exists():
        raise FileExistsError("Use fresh plan and project paths")
    runtime, snapshot, sft = activate(args.runtime)
    corpus, report, counts = corpus_inputs(args.corpus, args.max_length)
    config = sft.SFTConfig(
        output_dir=str(project),
        max_steps=2 * counts["train"],
        max_length=args.max_length,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        rank=4,
        lora_alpha=8.0,
        adapter_layers=[39],
        activation_mode="checkpoint",
        seed=8,
        chunk_size=2048,
        profile="development",
        memory_gib=args.memory_gib,
        reserve_gib=8,
        packed_cache_mib=getattr(args, "packed_cache_mib", 2048),
        decoded_cache_mib=0,
        prefix_cache_mib=1024,
        prefix_disk_cache=str(Path(args.prefix_disk_cache).expanduser().resolve()),
        prefix_disk_cache_gib=64,
        allocator_cache_mib=256,
        save_steps=1,
        eval_steps=2 * counts["train"] + 1,
        save_total_limit=2,
        comparison_examples=counts["test"],
        final_fit_examples=0,
    )
    trainer = sft.SFTTrainer(model=args.model, args=config, train_dataset=str(corpus))
    manifest = trainer.prepare(provenance_required=True)
    if manifest["data"]["counts"] != counts or manifest["data"].get("duplicates_removed"):
        raise ValueError(
            "SFT import changed corpus counts; provenance remains pending, do not train"
        )
    trainer.bind_provenance(corpus, report)
    manifest = trainer._manifest()
    folder = project / "experiment"
    folder.mkdir(exist_ok=False)
    prepared = {
        name: [
            json.loads(line)
            for line in (project / "data" / (name + ".jsonl")).read_text().splitlines()
            if line.strip()
        ]
        for name in ("train", "test")
    }
    selections = {}
    for name in prepared:
        index = max(range(len(prepared[name])), key=lambda i: len(prepared[name][i]["input_ids"]))
        path = folder / f"longest-{name}.jsonl"
        with path.open("x") as f:
            f.write(json.dumps(prepared[name][index], separators=(",", ":")) + "\n")
        selections[name] = {
            "index": index,
            "total_tokens": len(prepared[name][index]["input_ids"]),
            "path": str(path),
            "sha256": digest(path),
        }
    python = str(Path(sys.executable).absolute())
    commands = commands_for(trainer, manifest, selections, python)
    plan = {
        "format": "native-agent-kernel-run-plan-v1",
        "runtime": str(runtime),
        "snapshot_sha256": snapshot,
        "helper": str(Path(__file__).resolve()),
        "helper_sha256": digest(__file__),
        "python": python,
        "environment": {"PYTHONPATH": str(runtime / "src")},
        "cwd": str(runtime),
        "project": str(project),
        "initial_project_sha256": digest(project / "project.json"),
        "corpus_manifest_sha256": report["manifest_sha256"],
        "counts": counts,
        "config": dataclasses.asdict(config),
        "resources": {key: getattr(config, key) for key in sft.RESOURCE_FIELDS},
        "implementation": manifest["implementation"],
        "longest": selections,
        "commands": commands,
        "order": ["preflight-train", "preflight-test", "baseline", "train", "adapter"],
        "qualification": "Context ceiling is nominated before scores, not automatically proven. Longest full-row update and held-out forward must finish before baseline. Preflight update is disposable, never a warm start.",
        "selection": "Empty validation; fixed final checkpoint after2N updates with accumulation1; no held-out result selects training settings.",
        "limits": "Command emitter validates bound receipts but is not a launcher or proof of process execution chronology. Preserve actual invocation/start/end/exit/log receipts separately. No quality pass is implied.",
    }
    if (
        verify_runtime(runtime)[1] != snapshot
        or sft.implementation_identity() != plan["implementation"]
    ):
        raise ValueError("Runtime changed during preparation")
    write_new(plan_path, plan)
    return {
        "plan": str(plan_path),
        "plan_sha256": digest(plan_path),
        "counts": counts,
        "next": "Emit preflight-train command; no GPU operation was launched",
    }


def load_plan(path):
    plan = read_json(path)
    if plan.get("format") != "native-agent-kernel-run-plan-v1" or digest(
        plan["helper"]
    ) != plan.get("helper_sha256"):
        raise ValueError("Plan/helper identity changed")
    runtime, snapshot, sft = activate(plan["runtime"])
    if (
        snapshot != plan["snapshot_sha256"]
        or str(Path(sys.executable).absolute()) != plan["python"]
    ):
        raise ValueError("Use the pinned runtime and Python environment")
    trainer = sft.SFTTrainer.from_project(plan["project"])
    manifest = trainer._manifest()
    if (
        dataclasses.asdict(trainer.args) != plan["config"]
        or manifest["implementation"] != plan["implementation"]
        or manifest["data"]["counts"] != plan["counts"]
    ):
        raise ValueError("Project differs from fixed experiment plan")
    for value in plan["longest"].values():
        if digest(value["path"]) != value["sha256"]:
            raise ValueError("Preflight selection changed")
    if plan["commands"] != commands_for(trainer, manifest, plan["longest"], plan["python"]):
        raise ValueError("Plan command arguments differ from immutable configuration")
    if (
        not (trainer.root / "attempts").exists()
        and digest(trainer.root / "project.json") != plan["initial_project_sha256"]
    ):
        raise ValueError("Initial project changed before any training attempt")
    return plan, trainer, manifest, sft


def verify_evaluation(plan, trainer, manifest, phase, *, command=None, adapter=None):
    command = plan["commands"][phase][3:] if command is None else command[3:]
    data = Path(command[command.index("--data") + 1])
    report_path = Path(command[command.index("--report") + 1])
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    expected = {
        "data": str(data),
        "input_identity": {"data_sha256": digest(data)},
        "selected_examples_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "seq_length": manifest["data"]["seq_length"],
        "requested_max_examples": len(rows),
        "implementation": plan["implementation"],
    }
    receipt = read_json(report_path)
    trainer._verify_evaluation_receipt(receipt, expected, rows, command, adapter)
    from exl3_mlx.fit_probe import validate_scores

    validate_scores(receipt, rows)
    if adapter is not None:
        baseline = read_json(Path(plan["project"]) / "experiment/baseline.json")
        if (
            receipt["evaluation_context"]["process_id"]
            == baseline["evaluation_context"]["process_id"]
        ):
            raise ValueError("Baseline and adapter must have distinct evaluation processes")
    return report_path, receipt


def _emit_locked(args, plan, trainer, manifest, sft):
    phase = args.phase
    checkpoints = trainer._checkpoints()
    if phase in {"preflight-train", "preflight-test", "baseline"} and checkpoints:
        raise ValueError("Baseline and preflights must precede final-project gradients")
    folder = Path(plan["project"]) / "experiment"
    if phase in {"preflight-test", "baseline", "train", "adapter"}:
        report = read_json(folder / "preflight-train/training.json")
        state = sft.checkpoint_state(folder / "preflight-train/step-000001")
        if (
            report.get("status") != "completed"
            or report.get("completed_step") != 1
            or report.get("requested_steps") != 1
            or len(report.get("history", [])) != 1
            or state.get("step") != 1
            or state.get("cursor") != 1
        ):
            raise ValueError("Disposable longest-row full update is not complete")
        from exl3_mlx.context import ENGINE_SEMANTICS

        expected_contract = {
            "learning_rate": plan["config"]["learning_rate"],
            "grad_accumulation": 1,
            "clip_grad": 1.0,
            "seed": 8,
            "seq_length": manifest["data"]["seq_length"],
            "truncate": False,
            "chunk_size": 2048,
            "no_checkpoint": False,
            "data_sha256": plan["longest"]["train"]["sha256"],
            "mlx": plan["implementation"]["runtime"]["mlx"],
            "mlx_lm": plan["implementation"]["runtime"]["mlx-lm"],
            "engine_semantics": ENGINE_SEMANTICS,
        }
        adapter_config = read_json(folder / "preflight-train/step-000001/adapter_config.json")
        expected_adapter = {
            "base_identity": {
                key: manifest["capabilities"]["base_metadata"][key]
                for key in (
                    "config.json",
                    "expert-manifest-v2.json",
                    "engram/engram-manifest.json",
                    "engram-token-map.u32.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                )
            },
            "rank": 4,
            "alpha": 8.0,
            "adapter_layers": [39],
            "expert_adapter_layers": [],
            "activation_mode": "checkpoint",
        }
        if (
            report.get("contract") != expected_contract
            or state.get("contract") != expected_contract
            or report.get("activation_mode") != "checkpoint"
            or report.get("resources") != plan["resources"]
            or any(adapter_config.get(k) != v for k, v in expected_adapter.items())
            or report.get("checkpoint") != str(folder / "preflight-train/step-000001")
        ):
            raise ValueError("Preflight training receipt differs from declared data/mode/resources")
    if phase in {"baseline", "train", "adapter"}:
        verify_evaluation(plan, trainer, manifest, "preflight-test")
    if phase in {"train", "adapter"}:
        baseline_path, baseline = verify_evaluation(plan, trainer, manifest, "baseline")
        guard_path = folder / "baseline-before-training.json"
        binding = {
            "baseline": str(baseline_path),
            "sha256": digest(baseline_path),
            "process_id": baseline["evaluation_context"]["process_id"],
            "plan_sha256": digest(args.plan),
        }
        if guard_path.exists():
            guard = read_json(guard_path)
            if any(guard.get(k) != v for k, v in binding.items()):
                raise ValueError("Pre-training baseline binding changed")
        elif phase == "train" and not checkpoints:
            write_new(
                guard_path, {**binding, "verified_before_any_project_checkpoint_ns": time.time_ns()}
            )
        else:
            raise ValueError("Missing baseline binding from before final-project gradients")
    command = list(plan["commands"][phase])
    if phase == "adapter":
        selected = trainer.selected_checkpoint()
        state = sft.checkpoint_state(selected)
        if (
            manifest.get("status") != "completed"
            or state.get("step") != plan["config"]["max_steps"]
            or state.get("cursor") != plan["config"]["max_steps"]
            or plan["config"]["max_steps"] != 2 * plan["counts"]["train"]
        ):
            raise ValueError("Only the fixed final completed checkpoint can be evaluated")
        command[-1] = str(selected)
    return {
        "phase": phase,
        "argv": command,
        "cwd": plan["cwd"],
        "environment": plan["environment"],
        "plan": str(Path(args.plan).resolve()),
        "plan_sha256": digest(args.plan),
        "scope": "CPU-validated command only; caller must record and execute it serially",
    }


def emit_command(args):
    plan, trainer, manifest, sft = load_plan(args.plan)
    with sft.project_lock(trainer.root):
        manifest = trainer._manifest()
        return _emit_locked(args, plan, trainer, manifest, sft)


def verify_phase(args):
    plan, trainer, manifest, sft = load_plan(args.plan)
    folder = trainer.root / "experiment"
    with sft.project_lock(trainer.root):
        manifest = trainer._manifest()
        next_phase = {
            "preflight-train": "preflight-test",
            "preflight-test": "baseline",
            "baseline": "train",
            "train": "adapter",
            "adapter": "adapter",
        }[args.phase]
        emitted = _emit_locked(
            SimpleNamespace(plan=args.plan, phase=next_phase), plan, trainer, manifest, sft
        )
        if args.phase == "adapter":
            verify_evaluation(
                plan,
                trainer,
                manifest,
                "adapter",
                command=emitted["argv"],
                adapter=Path(emitted["argv"][-1]),
            )
        if args.phase == "preflight-train":
            paths = [
                folder / "preflight-train/training.json",
                *(folder / "preflight-train/step-000001").iterdir(),
            ]
        elif args.phase == "preflight-test":
            paths = [folder / "preflight-test.json"]
        elif args.phase == "baseline":
            paths = [folder / "baseline.json", folder / "baseline-before-training.json"]
        else:
            paths = [*trainer.selected_checkpoint().iterdir(), trainer.root / "project.json"]
            if args.phase == "train":
                paths.append(Path(manifest["last_training_report"]))
            else:
                paths.extend([folder / "baseline.json", folder / "adapter.json"])
        artifacts = {str(path): digest(path) for path in paths if path.is_file()}
    return {
        "phase": args.phase,
        "verified_complete": True,
        "artifacts_sha256": artifacts,
        "plan_sha256": digest(args.plan),
        "scope": "Bound receipt/checkpoint consistency, not proof of model quality or independently authenticated process execution",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    create = sub.add_parser("prepare")
    for name in ("runtime", "corpus", "model", "project", "plan", "prefix-disk-cache"):
        create.add_argument("--" + name, required=True)
    create.add_argument("--max-length", type=int, required=True)
    create.add_argument("--memory-gib", type=float, default=90)
    create.add_argument("--packed-cache-mib", type=int, default=2048)
    for action in ("command", "verify"):
        command = sub.add_parser(action)
        command.add_argument("--plan", required=True)
        command.add_argument(
            "--phase",
            required=True,
            choices=("preflight-train", "preflight-test", "baseline", "train", "adapter"),
        )
    args = p.parse_args()
    result = {"prepare": prepare, "command": emit_command, "verify": verify_phase}[args.action](
        args
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
