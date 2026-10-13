"""CPU-only recovery of a completed base plus resource-amended fresh adapter score.

Imports the checksummed pinned SFT validators, never invokes their child runner,
and never modifies project metadata, checkpoints, or source receipts.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def check_hash(path, expected):
    require(digest(path) == expected, f"Changed recovery input: {path}")


def publish_new(path, result):
    """Publish a fully written new report atomically; never replace any file."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_token_details(fit, receipt, selected):
    fit.validate_scores(receipt, selected)
    for row in receipt["rows"]:
        correct = sum(a == b for a, b in zip(row["target_ids"], row["predicted_ids"], strict=True))
        require(
            type(row.get("correct_tokens")) is int and row["correct_tokens"] == correct,
            "Invalid correct-token count",
        )


def load_pinned_sft(runtime, launch):
    runtime = Path(runtime).resolve()
    require(Path(launch["runtime"]).resolve() == runtime, "Runtime launch path differs")
    check_hash(runtime / "snapshot.json", launch["snapshot_sha256"])
    snapshot = read(runtime / "snapshot.json")
    for name, expected in snapshot.items():
        check_hash(runtime / name, expected)
    package = "_exl3_fit_recovery_pinned"
    require(
        not any(name == package or name.startswith(package + ".") for name in sys.modules),
        "Pinned recovery package already loaded; use a fresh CPU process",
    )
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(runtime / "src/exl3_mlx")]
    sys.modules[package] = namespace
    spec = importlib.util.spec_from_file_location(package + ".sft", runtime / "src/exl3_mlx/sft.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def argument(command, flag):
    positions = [
        i for i, value in enumerate(command) if value == flag or value.startswith(flag + "=")
    ]
    require(len(positions) == 1, f"Expected exactly one {flag}")
    index = positions[0]
    require(
        command[index] == flag and index + 1 < len(command), f"Use explicit flag/value for {flag}"
    )
    return index, command[index + 1]


def validate_retry_command(base_command, amendment, adapter):
    command = amendment["argv"]
    require(
        isinstance(command, list) and all(isinstance(x, str) for x in command), "Invalid retry argv"
    )
    require(command[1:4] == ["-u", "-m", "exl3_mlx.workflow"], "Unexpected retry launcher")
    retry = command[4:]
    require(retry and retry[0] == "evaluate", "Retry is not evaluation")
    require(
        amendment["resource_only_changes"]
        == {"base_memory_gib": 80, "adapter_memory_gib": 72, "reserve_gib": 8},
        "Unapproved resource change",
    )
    require(float(argument(base_command, "--memory-gib")[1]) == 80, "Base memory budget differs")
    require(float(argument(base_command, "--reserve-gib")[1]) == 8, "Base reserve differs")
    expected = list(base_command)
    expected[argument(expected, "--memory-gib")[0] + 1] = "72"
    expected[argument(expected, "--report")[0] + 1] = argument(retry, "--report")[1]
    expected += ["--adapter", str(adapter)]
    require(retry == expected, "Retry changed more than memory budget/report path/adapter")
    return retry, Path(argument(retry, "--report")[1]).resolve()


def base_command(trainer, manifest, path):
    options = trainer._engine_options()
    options[options.index("--prefix-cache-mib") + 1] = 0
    return list(
        map(
            str,
            [
                "evaluate",
                "--model",
                trainer.model,
                "--data",
                trainer.root / "data/train.jsonl",
                "--token-metrics",
                "--seq-length",
                manifest["data"]["seq_length"],
                "--max-examples",
                trainer.args.comparison_examples,
                "--report",
                path,
                *options,
                "--verbose",
            ],
        )
    )


def assemble(
    amendment_path,
    runtime_launch_path,
    training_audit_path,
    *,
    validate_base_only=False,
    output_path=None,
):
    amendment_path, runtime_launch_path = (
        Path(amendment_path).resolve(),
        Path(runtime_launch_path).resolve(),
    )
    amendment, launch = read(amendment_path), read(runtime_launch_path)
    if output_path is not None:
        output_path = Path(output_path).resolve()
        require(not validate_base_only, "Base-only validation cannot publish fit")
        require(
            output_path
            in {
                Path(amendment["project"]).resolve() / "fit.json",
                amendment_path.parent / "fit.json",
            },
            "Output must be a new project fit.json or recovery-directory fit.json",
        )
        require(not output_path.exists() and not output_path.is_symlink(), "Output already exists")
    require(
        amendment["schema"] == "standard-fit-resource-recovery-v1", "Unknown recovery amendment"
    )
    require(
        amendment["status"] == "predeclared_before_adapter_retry", "Expected predeclared amendment"
    )
    inputs = {
        str(amendment_path): digest(amendment_path),
        str(runtime_launch_path): digest(runtime_launch_path),
        str(Path(training_audit_path).resolve()): digest(training_audit_path),
        str(Path(__file__).resolve()): digest(__file__),
        amendment["base_report"]: amendment["base_report_sha256"],
        amendment["failed_adapter_log"]: amendment["failed_adapter_log_sha256"],
        amendment["prior_protocol"]: amendment["prior_protocol_sha256"],
    }
    for path, expected in inputs.items():
        check_hash(path, expected)
    typo_path, actual_path = (
        amendment_path.parent / "typo-failure.json",
        amendment_path.parent / "actual-launch.json",
    )
    typo, actual = read(typo_path), read(actual_path)
    require(
        type(typo.get("exit_code")) is int and typo["exit_code"] != 0,
        "Missing failed typo invocation",
    )
    check_hash(typo["log"], typo["sha256"])
    inputs.update(
        {
            str(typo_path): digest(typo_path),
            typo["log"]: typo["sha256"],
            str(actual_path): digest(actual_path),
        }
    )
    require(actual["schema"] == "actual-recovery-launch-v1", "Unknown actual retry launch")
    require(
        Path(actual["prelaunch_receipt"]).resolve() == amendment_path
        and actual["prelaunch_sha256"] == digest(amendment_path),
        "Actual retry is not bound to amendment",
    )
    require(
        actual["argv"] == amendment["argv"] and actual["environment"] == amendment["environment"],
        "Actual retry invocation differs",
    )
    training_audit_path = Path(training_audit_path).resolve()
    training_audit = read(training_audit_path)
    require(
        training_audit.get("training_completion_verified") is True
        and training_audit.get("checkpoints_verified") == 12,
        "Incomplete independent training audit",
    )
    audit_root = training_audit_path.parent.parent
    for relative, checksum in training_audit["input_sha256"].items():
        source = (audit_root / relative).resolve()
        require(source.is_relative_to(audit_root), "Training audit path escapes evidence root")
        inputs[str(source)] = checksum
        check_hash(source, checksum)
    require(launch["protocol_sha256"] == amendment["prior_protocol_sha256"], "Protocol pin differs")
    runtime = Path(amendment["runtime"]).resolve()
    require(
        amendment["environment"] == {"PYTHONPATH": str(runtime / "src")},
        "Retry runtime environment differs",
    )
    require(
        os.path.abspath(amendment["argv"][0]) == os.path.abspath(sys.executable),
        "Use the planned Python executable",
    )
    sft = load_pinned_sft(runtime, launch)
    trainer = sft.SFTTrainer.from_project(amendment["project"], progress=lambda _: None)
    # Reuse the existing lock inode without rewriting its diagnostic contents.
    with (trainer.root / ".run.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = trainer._manifest()
        adapter = trainer.selected_checkpoint().resolve()
        require(
            adapter == Path(amendment["selected_checkpoint"]).resolve(),
            "Selected checkpoint changed",
        )
        step_evidence = training_audit["steps"][-1]
        require(step_evidence["step"] == 12, "Independent audit does not select step12")
        for name, key in (
            ("adapter.safetensors", "adapter_sha256"),
            ("optimizer.safetensors", "optimizer_sha256"),
            ("trainer_state.json", "trainer_state_sha256"),
        ):
            check_hash(adapter / name, step_evidence[key])
        rows, selected, expected = trainer._evaluation_rows(manifest, "train")
        path = Path(amendment["base_report"]).resolve()
        require(
            path.is_relative_to(trainer.root / "fit-checks"),
            "Original baseline is outside project fit checks",
        )
        base = read(path)
        base_argv = base_command(trainer, manifest, path)
        trainer._verify_evaluation_receipt(base, expected, selected, base_argv, None)
        fit = sys.modules[sft.__package__ + ".fit_probe"]
        verify_token_details(fit, base, selected)
        retry_argv, retry_path = validate_retry_command(base_argv, amendment, adapter)
        require(
            retry_path.parent == amendment_path.parent and retry_path.name == "adapter.json",
            "Retry report path is outside amendment directory",
        )
        require(retry_path != path, "Retry must not overwrite baseline")
        base_resources = {key: getattr(trainer.args, key) for key in sft.RESOURCE_FIELDS}
        base_resources["prefix_cache_mib"] = 0
        require(base["resources"] == base_resources, "Base resolved resources differ")
        require(
            base["prefix_cache"]["enabled"] is False and base["prefix_cache"]["limit_bytes"] == 0,
            "Base used prefix cache",
        )
        if validate_base_only:
            return {
                "status": "base_validated_only",
                "base_sha256": digest(path),
                "adapter_pending": True,
                "quality_certified": False,
            }
        require(
            actual.get("exit_code") == 0
            and type(actual.get("exit_code")) is int
            and bool(actual.get("finished_utc")),
            "Retry process has not completed successfully",
        )
        retry_log = amendment_path.parent / "fit-adapter.log"
        check_hash(retry_log, actual["log_sha256"])
        inputs[str(retry_log)] = actual["log_sha256"]
        fresh = read(retry_path)
        inputs[str(retry_path)] = digest(retry_path)
        trainer._verify_evaluation_receipt(fresh, expected, selected, retry_argv, adapter)
        verify_token_details(fit, fresh, selected)
        require(
            fresh["evaluation_context"]["process_id"] == actual["pid"],
            "Retry receipt PID differs from actual launch",
        )
        require(
            len(
                {
                    base["evaluation_context"]["process_id"],
                    fresh["evaluation_context"]["process_id"],
                    os.getpid(),
                }
            )
            == 3,
            "Base, retry, and assembler must be distinct processes",
        )
        require(
            fresh["resources"] == {**base_resources, "memory_gib": 72},
            "Retry resolved resources differ",
        )
        require(
            base["expert_manifest_sha256"] == fresh["expert_manifest_sha256"],
            "Expert manifest differs",
        )
        parity = trainer._final_fit_parity(adapter, fresh, rows, selected, expected)
        require(
            parity.get("status") == "exact"
            and parity.get("real_model_warm_cache_parity_verified") is True,
            "Required full warm-probe parity was not verified",
        )
        # Retain the checkpoint and original training receipt bindings separately.
        for name in (
            "adapter.safetensors",
            "adapter_config.json",
            "optimizer.safetensors",
            "random.safetensors",
            "trainer_state.json",
        ):
            inputs[str(adapter / name)] = digest(adapter / name)
        inputs[str(adapter.parent / "training.json")] = digest(adapter.parent / "training.json")
        trainer._manifest()
        require(
            trainer.selected_checkpoint().resolve() == adapter,
            "Checkpoint selection changed during assembly",
        )
        for source, checksum in inputs.items():
            check_hash(source, checksum)
        check_hash(runtime / "snapshot.json", launch["snapshot_sha256"])
        result = {
            "available_examples": len(rows),
            "evaluated_examples": len(selected),
            "supervised_tokens": sum(len(row["input_ids"]) - row["loss_start"] for row in selected),
            "selection": "first immutable training rows; no held-out interpretation",
            "selected_examples_sha256": expected["selected_examples_sha256"],
            "format": "exl3-sft-training-fit-v1",
            "evaluation_scope": "training_only",
            "quality_certified": False,
            "note": "Training likelihood is a wiring/fit check, not generalization or kernel speedup.",
            "selected_checkpoint": str(adapter),
            "training_data_sha256": manifest["data_files"]["train.jsonl"],
            "loss_delta": fresh["loss"] - base["loss"],
            "training_mode_probe_parity": parity,
            "base": base,
            "adapter": fresh,
            "recovery_provenance": {
                "format": "exl3-resource-amended-fit-recovery-v1",
                "amendment": str(amendment_path),
                "original_base_report": str(path),
                "independent_training_audit": str(training_audit_path),
                "runtime_launch": str(runtime_launch_path),
                "retry_adapter_report": str(retry_path),
                "failed_adapter_log": amendment["failed_adapter_log"],
                "failed_invocations": [
                    {
                        "kind": "original_memory_preflight",
                        "stage": "startup_before_model_load",
                        "log": amendment["failed_adapter_log"],
                        "sha256": amendment["failed_adapter_log_sha256"],
                    },
                    {
                        "kind": "retry_path_typo_before_load",
                        "stage": "data_loading_before_model_load",
                        "receipt": str(typo_path),
                        "receipt_sha256": digest(typo_path),
                        "log": typo["log"],
                        "sha256": typo["sha256"],
                        "exit_code": typo["exit_code"],
                    },
                ],
                "actual_retry_launch": str(actual_path),
                "reason": amendment["reason"],
                "resource_only_changes": amendment["resource_only_changes"],
                "original_receipts_preserved": True,
                "baseline_rerun": False,
                "receipt_argv_preserved": True,
                "pinned_validators_runtime": str(runtime),
                "runtime_snapshot_sha256": launch["snapshot_sha256"],
                "input_sha256": inputs,
                "assembly_script_sha256": digest(__file__),
                "execution_limit": "Original and retry evaluation_context plus external launch evidence establish process provenance; this CPU assembly does not rerun inference.",
            },
        }
        if output_path is not None:
            publish_new(output_path, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--runtime-launch", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-base-only", action="store_true")
    args = parser.parse_args()
    if args.validate_base_only:
        require(args.output is None, "Base-only validation does not write a fit report")
    else:
        require(args.output is not None, "Provide a new output file")
        require(not args.output.exists() and not args.output.is_symlink(), "Output already exists")
    result = assemble(
        args.amendment,
        args.runtime_launch,
        args.training_audit,
        validate_base_only=args.validate_base_only,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "loss_delta", "training_mode_probe_parity")
                if key in result
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
