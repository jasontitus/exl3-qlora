#!/usr/bin/env python3
"""CPU-only audit of a predeclared standard SFT project and its native fit receipt.

This reads real project/fit.json and child receipts; it never fabricates diagnostic
history or imports MLX. Run after training AND the separate base/adapter scores.
It measures teacher-forced training fit, not generated-kernel quality.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

_HELPER_PATH = Path(__file__).with_name("summarize_fit_tokens.py")
_SPEC = importlib.util.spec_from_file_location("_fit_score_validation", _HELPER_PATH)
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)
ATOL = _HELPER.LOSS_MEAN_ATOL
AUDITOR_VERSION = "standard-sft-three-change-auditor-v3-recovery"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def checked_path(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    require(
        not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
        f"Evidence path escapes project: {path}",
    )
    require(path.is_file(), f"Missing evidence file: {path}")
    return path.resolve()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def summarize_pair(base, trained, labels, row, *, decoder=None):
    """Reuse the diagnostic's strict score validation, with native SFT row inputs."""
    _HELPER.validate_score(base)
    _HELPER.validate_score(trained)
    ids, start = row["input_ids"], row["loss_start"]
    require(type(start) is int and 1 <= start < len(ids), "Invalid complete answer mask")
    require(all(type(x) is int and x >= 0 for x in ids), "Invalid serialized IDs")
    targets = ids[start:]
    require(
        base["target_ids"] == trained["target_ids"] == labels["supervised_token_ids"] == targets,
        "Scores/categories differ from exact prepared targets",
    )
    require(
        labels["serialized_tokens"] == len(ids)
        and labels["loss_start"] == start
        and labels["supervised_tokens"] == len(targets),
        "Category mask/length differs",
    )
    classes = labels["supervised_token_categories"]
    require(
        isinstance(classes, list)
        and len(classes) == len(targets)
        and all(isinstance(c, str) and c and c != "all" for c in classes),
        "Invalid target category array",
    )
    for score in (base, trained):
        require(valid_digest(score["logits_sha256"]), "Invalid supervised-logit digest")
        require(all(v >= 0 for v in score["token_nll"]), "Negative token NLL")
    groups = {}
    for category in ["all", *sorted(set(classes))]:
        indices = [i for i, c in enumerate(classes) if category == "all" or c == category]
        group = {"count": len(indices)}
        for name, score in (("base", base), ("trained", trained)):
            group[name] = {
                "loss": math.fsum(score["token_nll"][i] for i in indices) / len(indices),
                "correct_tokens": sum(score["predicted_ids"][i] == targets[i] for i in indices),
            }
        group["loss_delta"] = group["trained"]["loss"] - group["base"]["loss"]
        groups[category] = group
    changes = {"corrected": [], "regressed": []}
    for i, (gold, category) in enumerate(zip(targets, classes)):
        if category != "added_code":
            continue
        before, after = base["predicted_ids"][i], trained["predicted_ids"][i]
        kind = (
            "corrected"
            if before != gold == after
            else "regressed"
            if before == gold != after
            else None
        )
        if kind is None:
            continue
        change = {
            "answer_token_index": i,
            "target_id": gold,
            "base_prediction_id": before,
            "trained_prediction_id": after,
        }
        if decoder is not None:
            lo, hi = max(0, i - 5), min(len(targets), i + 6)
            change.update(
                context_token_range=[lo, hi],
                target_context=decoder(targets[lo:hi]),
                target_text=decoder([gold]),
                base_prediction_text=decoder([before]),
                trained_prediction_text=decoder([after]),
            )
        changes[kind].append(change)
    return {
        "categories": groups,
        "serialized_tokens": len(ids),
        "loss_start": start,
        "input_ids_sha256": canonical_sha(ids),
        "target_ids_sha256": canonical_sha(targets),
        "row_sha256": canonical_sha(row),
        "logits_changed": base["logits_sha256"] != trained["logits_sha256"],
        "added_code_predictions_corrected": changes["corrected"],
        "added_code_predictions_regressed": changes["regressed"],
    }


def validate_native_fit(fit, manifest, rows, project, checkpoint):
    require(
        fit.get("format") == "exl3-sft-training-fit-v1"
        and fit.get("evaluation_scope") == "training_only"
        and fit.get("quality_certified") is False,
        "Expected native training-fit receipt",
    )
    require(
        fit.get("training_data_sha256") == manifest["data_files"]["train.jsonl"]
        and fit.get("selected_examples_sha256") == canonical_sha(rows),
        "Fit data identity mismatch",
    )
    require(fit.get("selected_checkpoint") == str(checkpoint), "Fit selected checkpoint mismatch")
    total = sum(len(r["input_ids"]) - r["loss_start"] for r in rows)
    require(
        fit.get("available_examples") == fit.get("evaluated_examples") == len(rows)
        and fit.get("supervised_tokens") == total,
        "Fit omits declared training rows",
    )
    pids = []
    for name in ("base", "adapter"):
        value = fit[name]
        ctx = value.get("evaluation_context", {})
        expected = {
            "data": str(project / "data/train.jsonl"),
            "input_identity": {"data_sha256": manifest["data_files"]["train.jsonl"]},
            "selected_examples_sha256": canonical_sha(rows),
            "seq_length": manifest["data"]["seq_length"],
            "requested_max_examples": manifest["config"]["comparison_examples"],
            "implementation": manifest["implementation"],
        }
        require(
            all(ctx.get(k) == v for k, v in expected.items()),
            "Child evaluation provenance mismatch",
        )
        require(
            type(ctx.get("process_id")) is int and ctx["process_id"] > 0,
            "Missing child process identity",
        )
        pids.append(ctx["process_id"])
        require(
            value.get("activation_mode", "checkpoint")
            == manifest["config"].get("activation_mode", "checkpoint")
            and value.get("checkpoint") == manifest["model"]
            and value.get("adapter") == (str(checkpoint) if name == "adapter" else None),
            "Child model/mode/adapter identity mismatch",
        )
        require(
            value.get("examples") == len(rows)
            and value.get("supervised_tokens") == total
            and len(value.get("rows", [])) == len(rows),
            "Missing complete per-row token scores",
        )
        require(
            value.get("logits_scope") == "supervised_rows"
            and valid_digest(value.get("logits_sha256")),
            "Expected supervised-row logit digest",
        )
        for score in value["rows"]:
            _HELPER.validate_score(score)
        aggregate = math.fsum(s["loss"] * s["supervised_tokens"] for s in value["rows"]) / total
        require(
            type(value["loss"]) in (float, int)
            and math.isfinite(value["loss"])
            and abs(value["loss"] - aggregate) <= ATOL,
            "Aggregate/per-row loss mismatch",
        )
        argv = ctx.get("argv", [])
        require(
            isinstance(argv, list) and argv and argv[0] == "evaluate" and "--token-metrics" in argv,
            "Child argv does not describe token scoring",
        )
        for flag, expected_value in (
            ("--data", str(project / "data/train.jsonl")),
            ("--seq-length", str(manifest["data"]["seq_length"])),
            ("--model", manifest["model"]),
            ("--chunk-size", str(manifest["config"]["chunk_size"])),
        ):
            require(
                argv.count(flag) == 1
                and argv.index(flag) + 1 < len(argv)
                and argv[argv.index(flag) + 1] == expected_value,
                f"Child argv differs: {flag}",
            )
        require(
            value.get("resources", {}).get("prefix_cache_mib") == 0,
            "Fresh comparison must explicitly disable prefix cache",
        )
    require(pids[0] != pids[1], "Base/adapter scores must use separate fresh processes")
    require(
        abs(fit["loss_delta"] - (fit["adapter"]["loss"] - fit["base"]["loss"])) <= 1e-15,
        "Fit paired delta mismatch",
    )


def validate_recovery(fit, manifest, project, checkpoint, protocol_path, runtime, read):
    """Recheck the exceptional resource-only pairing and disclose its failed starts."""
    provenance = fit.get("recovery_provenance")
    base_resources, fresh_resources = (fit[name]["resources"] for name in ("base", "adapter"))
    if provenance is None:
        require(
            base_resources == fresh_resources, "Different resources require recovery provenance"
        )
        return {"used": False}
    require(
        isinstance(provenance, dict)
        and provenance.get("format") == "exl3-resource-amended-fit-recovery-v1"
        and provenance.get("original_receipts_preserved") is True
        and provenance.get("baseline_rerun") is False
        and provenance.get("receipt_argv_preserved") is True,
        "Invalid recovery preservation contract",
    )
    protocol_path = Path(protocol_path).resolve()
    allowed = (
        project,
        runtime.parent,
        protocol_path.parent,
        protocol_path.parent.parent / "scripts",
        Path(__file__).resolve().parent,
    )
    inputs = provenance.get("input_sha256")
    require(isinstance(inputs, dict) and inputs, "Recovery lacks bound inputs")
    for name, sha in inputs.items():
        path = Path(name)
        require(
            path.is_absolute()
            and not path.is_symlink()
            and any(path.resolve().is_relative_to(root) for root in allowed)
            and valid_digest(sha)
            and digest(path) == sha,
            f"Changed or unscoped recovery evidence: {name}",
        )
        read(path, binary=True)

    def bound(path):
        path = Path(path).resolve()
        require(str(path) in inputs, f"Recovery input is not bound: {path}")
        require(inputs.get(str(path)) == digest(path), f"Recovery input is not bound: {path}")
        return read(path)

    amendment_path = checked_path(project, provenance["amendment"])
    amendment = bound(amendment_path)
    require(
        amendment.get("schema") == "standard-fit-resource-recovery-v1"
        and amendment.get("status") == "predeclared_before_adapter_retry"
        and Path(amendment["project"]).resolve() == project
        and Path(amendment["runtime"]).resolve() == runtime
        and Path(amendment["selected_checkpoint"]).resolve() == checkpoint
        and Path(amendment["prior_protocol"]).resolve() == protocol_path
        and amendment["prior_protocol_sha256"] == digest(protocol_path)
        and provenance["pinned_validators_runtime"] == str(runtime)
        and provenance["reason"] == amendment["reason"],
        "Recovery amendment identity differs",
    )
    changes = {"base_memory_gib": 80, "adapter_memory_gib": 72, "reserve_gib": 8}
    require(
        amendment["resource_only_changes"] == provenance["resource_only_changes"] == changes
        and base_resources["memory_gib"] == 80
        and base_resources["reserve_gib"] == 8
        and fresh_resources == {**base_resources, "memory_gib": 72}
        and all(
            value == (0 if key == "prefix_cache_mib" else manifest["config"][key])
            for key, value in base_resources.items()
        ),
        "Recovery resource exception differs",
    )
    helper_path = Path(__file__).with_name("recover_standard_fit.py").resolve()
    require(
        provenance["assembly_script_sha256"] == digest(helper_path)
        and any(
            Path(name).name == helper_path.name and sha == digest(helper_path)
            for name, sha in inputs.items()
        ),
        "Recovery assembly script differs",
    )
    read(helper_path, binary=True)
    spec = importlib.util.spec_from_file_location("_checked_fit_recovery", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)  # CPU-only helpers; never call assemble/child/model functions.
    base_argv = fit["base"]["evaluation_context"]["argv"]
    fresh_argv = fit["adapter"]["evaluation_context"]["argv"]
    retry_argv, retry_path = helper.validate_retry_command(base_argv, amendment, checkpoint)
    require(retry_argv == fresh_argv, "Actual score argv differs from recovery amendment")
    base_path = checked_path(project, helper.argument(base_argv, "--report")[1])
    retry_path = checked_path(project, retry_path)
    require(
        str(base_path) == provenance["original_base_report"] == amendment["base_report"]
        and digest(base_path) == amendment["base_report_sha256"]
        and str(retry_path) == provenance["retry_adapter_report"]
        and retry_path.parent == amendment_path.parent
        and retry_path.name == "adapter.json"
        and base_path != retry_path,
        "Recovery raw receipt paths differ",
    )
    require(
        bound(base_path) == fit["base"] and bound(retry_path) == fit["adapter"],
        "Recovery raw scores differ",
    )
    launch = bound(provenance["runtime_launch"])
    snapshot = runtime / "snapshot.json"
    require(
        Path(launch["runtime"]).resolve() == runtime
        and launch["snapshot_sha256"] == provenance["runtime_snapshot_sha256"] == digest(snapshot)
        and launch["protocol_sha256"] == digest(protocol_path),
        "Recovery runtime launch differs",
    )
    for relative, sha in read(snapshot).items():
        path = checked_path(runtime, runtime / relative)
        require(digest(path) == sha, f"Pinned recovery runtime changed: {relative}")
        read(path, binary=True)
    audit = bound(provenance["independent_training_audit"])
    require(
        audit.get("training_completion_verified") is True
        and audit.get("checkpoints_verified") == 12,
        "Recovery training audit is incomplete",
    )
    audit_root = Path(provenance["independent_training_audit"]).resolve().parent.parent
    for relative, sha in audit["input_sha256"].items():
        require(
            inputs.get(str((audit_root / relative).resolve())) == sha,
            "Recovery audit input binding differs",
        )
    actual_path = checked_path(project, provenance["actual_retry_launch"])
    actual = bound(actual_path)
    require(
        actual_path == amendment_path.parent / "actual-launch.json"
        and actual.get("schema") == "actual-recovery-launch-v1"
        and Path(actual["prelaunch_receipt"]).resolve() == amendment_path
        and actual["prelaunch_sha256"] == digest(amendment_path)
        and actual["argv"] == amendment["argv"]
        and actual["environment"]
        == amendment["environment"]
        == {"PYTHONPATH": str(runtime / "src")}
        and type(actual.get("exit_code")) is int
        and actual["exit_code"] == 0
        and isinstance(actual.get("finished_utc"), str)
        and bool(actual["finished_utc"])
        and type(actual.get("pid")) is int
        and actual["pid"] == fit["adapter"]["evaluation_context"]["process_id"]
        and inputs.get(str(actual_path.parent / "fit-adapter.log")) == actual["log_sha256"],
        "Recovery retry completion/launch differs",
    )
    failures = provenance.get("failed_invocations")
    require(
        isinstance(failures, list)
        and len(failures) == 2
        and [row.get("kind") for row in failures]
        == ["original_memory_preflight", "retry_path_typo_before_load"],
        "Recovery must preserve both failed invocations",
    )
    original, typo = failures
    require(
        original.get("stage") == "startup_before_model_load"
        and original["log"] == provenance["failed_adapter_log"] == amendment["failed_adapter_log"]
        and original["sha256"]
        == amendment["failed_adapter_log_sha256"]
        == inputs.get(original["log"]),
        "Original failed startup binding differs",
    )
    typo_receipt = bound(checked_path(project, typo["receipt"]))
    require(
        Path(typo["receipt"]).resolve() == amendment_path.parent / "typo-failure.json"
        and typo.get("stage") == "data_loading_before_model_load"
        and inputs.get(typo["receipt"]) == typo["receipt_sha256"]
        and typo_receipt["log"] == typo["log"]
        and typo_receipt["sha256"] == typo["sha256"] == inputs.get(typo["log"])
        and type(typo.get("exit_code")) is int
        and typo["exit_code"] != 0
        and typo_receipt["exit_code"] == typo["exit_code"],
        "Pre-model typo failure binding differs",
    )
    return {
        "used": True,
        "base_process_memory_gib": 80,
        "adapter_process_memory_gib": 72,
        "reserve_gib": 8,
        "baseline_reused_without_rerun": True,
        "math_data_checkpoint_and_cache_settings_unchanged": True,
        "amendment": str(amendment_path),
        "actual_retry_launch": str(actual_path),
        "raw_base_report": str(base_path),
        "raw_adapter_report": str(retry_path),
        "failed_invocations": failures,
        "reason": provenance["reason"],
        "verification_scope": "Checksummed local launch/receipt evidence, not independent process attestation; failed starts add no training updates.",
    }


def expected_samples(cursor, size, seed):
    # Same documented deterministic epoch shuffle; no import of GPU training module.
    order, result = [], []
    for position in range(cursor):
        epoch, offset = divmod(position, size)
        if offset == 0:
            order = list(range(size))
            random.Random(seed + epoch).shuffle(order)
        result.append(order[offset])
    return result


def validate_case_provenance(project, manifest, protocol, rows, read):
    """Bind labels to retained prepared rows and the original raw source receipt."""
    required = {"dataset.json", "provenance.jsonl", "train.jsonl"}
    require(required <= set(manifest["data_files"]), "Missing hash-bound dataset/provenance files")
    dataset = read(project / "data/dataset.json")
    sources = dataset.get("source_files_sha256")
    require(
        isinstance(sources, dict)
        and len(sources) == 1
        and sources == manifest["data"].get("source_files_sha256")
        and list(sources.values()) == [protocol.get("raw_sha256")]
        and valid_digest(protocol.get("raw_sha256")),
        "Protocol raw SHA differs from the prepared dataset source receipt",
    )
    raw_path = Path(next(iter(sources))).expanduser().resolve()
    require(digest(raw_path) == protocol["raw_sha256"], "Original raw source changed")
    read(raw_path, binary=True)
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    provenance_path = project / "data/provenance.jsonl"
    provenance = [
        json.loads(line) for line in provenance_path.read_text().splitlines() if line.strip()
    ]
    require(
        len(raw) == len(provenance) == len(rows) == 3,
        "Expected exactly three retained source records",
    )
    bound = []
    for i, (entry, row, raw_row, case_id) in enumerate(
        zip(provenance, rows, raw, protocol["case_ids"])
    ):
        require(
            entry.get("retained") is True
            and entry.get("split") == "train"
            and entry.get("duplicate_of") is None
            and type(entry.get("source_row")) is int
            and entry["source_row"] == i + 1
            and Path(entry["source_file"]).resolve() == raw_path
            and entry.get("tokens_sha256") == canonical_sha(row),
            "Retained provenance does not bind the exact prepared row/mask",
        )
        metadata = entry.get("metadata", {})
        require(
            metadata.get("case_id") == raw_row.get("case_id") == case_id,
            "Protocol case ID differs from the bound raw/prepared row mapping",
        )
        require(
            metadata.get("source_id") == raw_row.get("source_id")
            and isinstance(metadata.get("source_id"), str)
            and bool(metadata["source_id"]),
            "Source lineage metadata differs",
        )
        bound.append(
            {
                "row_index": i,
                "case_id": case_id,
                "source_row": i + 1,
                "source_id": metadata["source_id"],
                "tokens_sha256": entry["tokens_sha256"],
            }
        )
    require(len({item["source_id"] for item in bound}) == 1, "Expected the declared single lineage")
    return {
        "raw_sha256": protocol["raw_sha256"],
        "raw_source": str(raw_path),
        "provenance_sha256": digest(provenance_path),
        "rows": bound,
    }


def audit_training(project, manifest, fit, protocol, read):
    import numpy as np
    from safetensors.numpy import load_file

    steps = protocol["settings"]["steps_total"]
    config = manifest["config"]
    require(
        manifest.get("status") == "completed" and config["max_steps"] == steps,
        "Project must complete the predeclared total step budget",
    )
    require(
        manifest["data"]["counts"] == {"train": 3, "valid": 0, "test": 0},
        "This protocol requires exactly three training-only changes",
    )
    for key, value in protocol["settings"].items():
        mapped = {"steps_total": "max_steps", "alpha": "lora_alpha"}.get(key, key)
        if mapped in config:
            require(config[mapped] == value, f"Project differs from protocol: {mapped}")
    checkpoint = Path(fit["selected_checkpoint"]).resolve()
    state = read(checked_path(project, checkpoint / "trainer_state.json"))
    require(state["step"] == state["cursor"] == steps, "Final step/cursor differs from protocol")
    paths = sorted(project.glob("attempts/*/train/step-*/trainer_state.json"))
    require(len(paths) == steps, "All twelve retained checkpoints are required for this audit")
    sample_order = expected_samples(steps, 3, config["seed"])
    require(
        sample_order == protocol["sampler"]["expected_row_indices"], "Sampler differs from protocol"
    )
    expected_keys = {
        f"blocks.39.attn.linears.{target}.lora_{part}"
        for target in ("wq_a", "wo_b")
        for part in ("a", "b")
    }
    expected_shapes = {
        "blocks.39.attn.linears.wq_a.lora_a": (5120, 4),
        "blocks.39.attn.linears.wq_a.lora_b": (4, 1280),
        "blocks.39.attn.linears.wo_b.lora_a": (8192, 4),
        "blocks.39.attn.linears.wo_b.lora_b": (4, 5120),
    }
    observed, tensors = {}, []
    for path in paths:
        saved = read(checked_path(project, path))
        require(saved.get("format") == "exl3-mlx-training-state-v1", "Unknown checkpoint format")
        step = saved["step"]
        require(
            type(step) is int and 1 <= step <= steps and step not in observed,
            "Duplicate/invalid checkpoint step",
        )
        require(
            saved["cursor"] == step
            and saved["metrics"]["step"] == step
            and saved["metrics"]["cursor"] == step,
            "Checkpoint sample cursor mismatch",
        )
        require(
            saved["metrics"]["microbatches"] == 1
            and saved["metrics"]["supervised_tokens"]
            == protocol["supervised_tokens"][sample_order[step - 1]],
            "Recorded update target count differs from deterministic selected row",
        )
        require(saved.get("contract") == state["contract"], "Checkpoint optimizer contract changed")
        required = {
            "adapter.safetensors",
            "adapter_config.json",
            "optimizer.safetensors",
            "random.safetensors",
        }
        require(set(saved["files"]) == required, "Incomplete checkpoint file manifest")
        for name, sha in saved["files"].items():
            p = checked_path(project, path.parent / name)
            require(digest(p) == sha, f"Checkpoint file checksum mismatch: {p}")
            read(p, binary=True)
        adapter_config = read(path.parent / "adapter_config.json")
        require(
            adapter_config["adapter_layers"] == [39]
            and adapter_config["expert_adapter_layers"] == []
            and adapter_config["rank"] == config["rank"]
            and adapter_config["alpha"] == config["lora_alpha"]
            and adapter_config.get("activation_mode", "checkpoint") == config["activation_mode"],
            "Checkpoint adapter scope mismatch",
        )
        require(
            adapter_config.get("engine_semantics") == state["contract"].get("engine_semantics")
            and isinstance(adapter_config.get("base_identity"), dict)
            and set(adapter_config["base_identity"])
            == {
                "config.json",
                "expert-manifest-v2.json",
                "engram/engram-manifest.json",
                "engram-token-map.u32.json",
                "model.safetensors.index.json",
                "tokenizer.json",
            }
            and all(
                manifest["capabilities"]["base_metadata"].get(name) == value
                for name, value in adapter_config["base_identity"].items()
            ),
            "Checkpoint base model or numerical semantics mismatch",
        )
        adapters = load_file(str(path.parent / "adapter.safetensors"))
        optimizer = load_file(str(path.parent / "optimizer.safetensors"))
        require(set(adapters) == expected_keys, "Checkpoint adapter tensor keys differ")
        require(
            all(value.shape == expected_shapes[name] for name, value in adapters.items()),
            "Checkpoint adapter shapes differ",
        )
        require(
            all(x.dtype == np.float32 and np.isfinite(x).all() for x in adapters.values()),
            "Expected finite FP32 adapters",
        )
        require(
            optimizer and all(np.isfinite(x).all() for x in optimizer.values()),
            "Nonfinite/empty optimizer state",
        )
        require(
            set(optimizer)
            == {"step", "learning_rate"}
            | {f"{name}.{moment}" for name in expected_keys for moment in ("m", "v")}
            and optimizer["step"].size == 1
            and optimizer["step"].item() == step
            and optimizer["learning_rate"].size == 1
            and float(optimizer["learning_rate"].item())
            == float(np.float32(config["learning_rate"]))
            and all(
                optimizer[f"{name}.{moment}"].shape == shape
                for name, shape in expected_shapes.items()
                for moment in ("m", "v")
            ),
            "Saved Adam state/step/learning rate differs",
        )
        count = sum(x.size for x in adapters.values())
        require(count == 78848, "Unexpected selected-block parameter count")
        observed[step] = {
            "cursor": saved["cursor"],
            "derived_row": sample_order[step - 1],
            "recorded_supervised_tokens": saved["metrics"]["supervised_tokens"],
        }
        tensors.append(
            {
                "step": step,
                "adapter_tensors": len(adapters),
                "adapter_parameters": count,
                "optimizer_tensors": len(optimizer),
                "finite": True,
                "nonzero_B_elements": sum(
                    int(np.count_nonzero(value))
                    for name, value in adapters.items()
                    if name.endswith("lora_b")
                ),
            }
        )
    require(set(observed) == set(range(1, steps + 1)), "Incomplete completed-update sequence")
    report_path = checked_path(project, manifest["last_training_report"])
    training = read(report_path)
    require(
        training["status"] == "completed"
        and training["completed_step"] == steps
        and Path(training["checkpoint"]).resolve() == checkpoint,
        "Training report/final checkpoint mismatch",
    )
    require(
        training.get("additional_steps") == steps
        and [record.get("step") for record in training.get("history", [])]
        == list(range(1, steps + 1)),
        "Expected all predeclared fresh-run updates in native training history",
    )
    launch = read(checked_path(project, report_path.with_name("run_config.json")))
    require(
        launch.get("resume") is None
        and launch.get("resume_adapter") is None
        and launch.get("steps") == steps
        and launch.get("seed") == config["seed"]
        and launch.get("adapter_layers") == [39]
        and launch.get("activation_mode") == config["activation_mode"],
        "Expected fresh predeclared initialization rather than adapter continuation",
    )
    contract = state["contract"]
    expected = {
        "data_sha256": manifest["data_files"]["train.jsonl"],
        "learning_rate": config["learning_rate"],
        "grad_accumulation": 1,
        "clip_grad": config["max_grad_norm"],
        "seed": config["seed"],
        "seq_length": manifest["data"]["seq_length"],
        "truncate": False,
        "chunk_size": config["chunk_size"],
        "no_checkpoint": False,
        "mlx": manifest["implementation"]["runtime"]["mlx"],
        "mlx_lm": manifest["implementation"]["runtime"]["mlx-lm"],
    }
    require(
        all(contract.get(k) == v for k, v in expected.items()),
        "Saved optimizer/data contract mismatch",
    )
    return checkpoint, {
        "optimizer_updates": steps,
        "sample_cursor": steps,
        "samples": [observed[k] for k in sorted(observed)],
        "exposures_per_row": dict(Counter(sample_order)),
        "sample_identity_scope": "Verified cursor+seed+bound sampler source, independently reconstructed order; per-step target counts also match",
        "checkpoints": tensors,
    }


def build_report(project, protocol_path, category_paths, runtime, *, decode=False):
    project, runtime = Path(project).resolve(), Path(runtime).resolve()
    hashes = {}

    def read(path, *, binary=False):
        path = Path(path).resolve()
        hashes[str(path)] = digest(path)
        return None if binary else json.loads(path.read_text())

    protocol = read(protocol_path)
    require(
        isinstance(protocol.get("case_ids"), list)
        and all(isinstance(value, str) and value for value in protocol["case_ids"])
        and len(protocol["case_ids"]) == len(set(protocol["case_ids"])) == 3,
        "Protocol must identify three distinct historical changes",
    )
    manifest, fit = read(project / "project.json"), read(project / "fit.json")
    require(manifest["format"] == "exl3-sft-project-v1", "Expected native SFT project")
    for name, sha in manifest["data_files"].items():
        path = checked_path(project, project / "data" / name)
        require(digest(path) == sha, f"Project data changed: {name}")
        read(path, binary=True)
    for name, sha in manifest.get("provenance_files", {}).items():
        path = checked_path(project, name)
        require(digest(path) == sha, f"Project provenance changed: {name}")
        read(path, binary=True)
    require(
        not manifest.get("provenance_required")
        or manifest.get("provenance_binding", {}).get("state") == "complete",
        "Required provenance is not completely bound",
    )
    for name, sha in manifest["implementation"]["source"].items():
        path = runtime / "src/exl3_mlx" / name
        require(digest(path) == sha, f"Bound runtime source differs: {name}")
        read(path, binary=True)
    require(
        digest(runtime / "uv.lock") == manifest["implementation"]["dependency_lock"],
        "Runtime lock differs",
    )
    read(runtime / "uv.lock", binary=True)
    data = project / "data/train.jsonl"
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    require(
        len(rows) == 3 and digest(data) == protocol["expected_prepared_sha256"],
        "Prepared protocol data differs",
    )
    case_binding = validate_case_provenance(project, manifest, protocol, rows, read)
    checkpoint, update_audit = audit_training(project, manifest, fit, protocol, read)
    validate_native_fit(fit, manifest, rows, project, checkpoint)
    recovery = validate_recovery(fit, manifest, project, checkpoint, protocol_path, runtime, read)
    training = read(checked_path(project, manifest["last_training_report"]))
    probe = training.get("training_fit_probe")
    require(
        isinstance(probe, dict) and probe.get("requested_examples") == 3,
        "Expected predeclared complete three-row training-mode probe",
    )
    require(
        probe == read(checked_path(project, checkpoint.parent / "training-fit-probe.json")),
        "Embedded warm probe differs from original probe receipt",
    )
    spec = importlib.util.spec_from_file_location(
        "_bound_fit_probe", runtime / "src/exl3_mlx/fit_probe.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    expected = {
        "data": str(data),
        "input_identity": {"data_sha256": digest(data)},
        "implementation": manifest["implementation"],
        "seq_length": manifest["data"]["seq_length"],
        "activation_mode": manifest["config"]["activation_mode"],
        "chunk_size": manifest["config"]["chunk_size"],
        "available_examples": 3,
        "requested_examples": 3,
    }
    cache_parity = helper.compare_final_fit_probe(
        probe, fit["adapter"], rows, checkpoint=checkpoint, expected=expected
    )
    recorded_parity = fit.get("training_mode_probe_parity", {})
    require(
        all(recorded_parity.get(k) == v for k, v in cache_parity.items()),
        "Recorded warm/fresh comparison differs from independently rechecked receipts",
    )
    require(
        recorded_parity.get("training_report_sha256") == digest(manifest["last_training_report"]),
        "Warm/fresh comparison training receipt hash differs",
    )
    # Read the original separately written child receipts, not only embedded copies.
    for name in ("base", "adapter"):
        argv = fit[name]["evaluation_context"]["argv"]
        require(argv.count("--report") == 1, "Child argv lacks unique report path")
        raw = read(checked_path(project, argv[argv.index("--report") + 1]))
        require(raw == fit[name], "Embedded score differs from its original child receipt")
    decoder = None
    tokenizer = Path(manifest["model"]) / "tokenizer.json"
    tokenizer_sha = manifest["capabilities"]["base_metadata"]["tokenizer.json"]
    require(digest(tokenizer) == tokenizer_sha, "Tokenizer differs from model receipt")
    read(tokenizer, binary=True)
    if decode:
        from tokenizers import Tokenizer

        cpu_tokenizer = Tokenizer.from_file(str(tokenizer))

        def decoder(ids):
            return cpu_tokenizer.decode(ids, skip_special_tokens=False)

    categories = {}
    for path in category_paths:
        labels = read(path)
        i = labels["row_index"]
        require(
            type(i) is int and i in range(3) and i not in categories,
            "Duplicate/invalid category row",
        )
        require(
            labels["source_sha256"]["prepared"] == digest(data)
            and labels["source_sha256"]["tokenizer"] == tokenizer_sha,
            "Category provenance mismatch",
        )
        categories[i] = labels
    require(set(categories) == {0, 1, 2}, "Provide category evidence for all three rows")
    results = {
        str(i): summarize_pair(
            fit["base"]["rows"][i], fit["adapter"]["rows"][i], categories[i], row, decoder=decoder
        )
        for i, row in enumerate(rows)
    }
    for i, case_id in enumerate(protocol["case_ids"]):
        results[str(i)]["case_id"] = case_id
    total = sum(len(r["input_ids"]) - r["loss_start"] for r in rows)
    weighted, macro = {}, {}
    for name, condition in (("base_loss", "base"), ("trained_loss", "adapter")):
        weighted[name] = (
            math.fsum(s["loss"] * s["supervised_tokens"] for s in fit[condition]["rows"]) / total
        )
        macro[name] = math.fsum(s["loss"] for s in fit[condition]["rows"]) / len(rows)
    for value in (weighted, macro):
        value["loss_delta"] = value["trained_loss"] - value["base_loss"]
    read(Path(__file__), binary=True)
    read(_HELPER_PATH, binary=True)
    return {
        "schema": "standard-sft-three-change-fit-audit-v1",
        "auditor": {
            "version": AUDITOR_VERSION,
            "sha256": digest(__file__),
            "scope": "Post-run CPU auditor; not asserted to be the frozen launch script",
        },
        "status": "completed",
        "scope": "All three predeclared training rows; teacher forcing supplies preceding answer tokens",
        "quality_certified": False,
        "held_out_rows": 0,
        "independent_lineages": 1,
        "distinct_changes": 3,
        "case_binding": case_binding,
        "supervised_tokens": total,
        "selected_checkpoint": str(checkpoint),
        "completed_updates": update_audit,
        "resource_recovery": recovery,
        "weighted": weighted,
        "macro": macro,
        "every_row_improved": all(
            r["categories"]["all"]["loss_delta"] < -1e-6 for r in results.values()
        ),
        "aggregate_fit_improved": weighted["loss_delta"] < -1e-6,
        "rows": results,
        "warm_cache_verification": cache_parity,
        "tool_verification_complete": cache_parity["real_model_warm_cache_parity_verified"],
        "tool_verification_scope": "This three-case saved-optimizer, adapter-reload and warm/fresh-cache path only; not full-model oracle correctness or useful kernel quality",
        "source_hashes": hashes,
        "limitations": [
            "No held-out/generalization or generated-patch evaluation.",
            "Diff token categories are coarse lexical labels, not semantic correctness.",
            "Code top-1 predictions use correct preceding tokens; they are not autonomous generation.",
            "Checksum/source receipts bind local evidence, not independent attestation or full weight-byte authentication.",
            "Report every regressing row/category; aggregate gain is not all-case success.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--categories", type=Path, nargs=3, required=True)
    parser.add_argument(
        "--runtime",
        type=Path,
        required=True,
        help="Frozen source root with src/exl3_mlx and uv.lock",
    )
    parser.add_argument(
        "--decode-context",
        action="store_true",
        help="Use CPU tokenizers only for changed added-code token context",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output must be a new file; preserve earlier reports")
    result = build_report(
        args.project, args.protocol, args.categories, args.runtime, decode=args.decode_context
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("rows", "source_hashes", "completed_updates")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
