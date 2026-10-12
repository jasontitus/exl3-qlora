"""Read-only CPU audit of this pilot's saved tensors and fresh fit receipts.

This is an evidence-collection script, not a model or kernel-quality evaluator.
It imports no MLX and performs no inference. Use after a completed update.
"""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

if not __debug__:
    raise RuntimeError("Run this audit without Python -O: its evidence checks must stay enabled")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tensors(path):
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    assert 0 < length <= 1 << 20
    header = json.loads(data[8 : 8 + length])
    payload = memoryview(data)[8 + length :]
    dtypes = {"F32": "<f4", "F16": "<f2", "U64": "<u8", "U32": "<u4", "I32": "<i4"}
    values = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        assert 0 <= start <= end <= len(payload)
        value = np.frombuffer(payload[start:end], dtype=dtypes[info["dtype"]])
        assert value.size == math.prod(info["shape"])
        assert np.isfinite(value).all(), f"Nonfinite saved tensor: {name}"
        values[name] = value.reshape(info["shape"])
    assert values
    return values


def audit(project, checkpoint_only=False):
    manifest = json.loads((project / "project.json").read_text())
    report_path = Path(manifest["last_training_report"])
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed" and report["completed_step"] == 1
    assert len(report["history"]) == 1
    assert manifest["data"]["counts"] == {"train": 1, "valid": 0, "test": 0}
    record = report["history"][-1]
    assert record["microbatches"] == 1
    assert math.isfinite(record["loss"])
    assert math.isfinite(record["gradient_norm"]) and record["gradient_norm"] > 0
    checkpoint = Path(report["checkpoint"])
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    names = {
        "adapter.safetensors",
        "adapter_config.json",
        "optimizer.safetensors",
        "random.safetensors",
    }
    assert set(state["files"]) == names
    assert all(digest(checkpoint / name) == checksum for name, checksum in state["files"].items())
    adapter = read_tensors(checkpoint / "adapter.safetensors")
    optimizer = read_tensors(checkpoint / "optimizer.safetensors")
    read_tensors(checkpoint / "random.safetensors")
    assert state["step"] == int(optimizer["step"].item()) == report["completed_step"]
    assert all(name.endswith((".lora_a", ".lora_b")) for name in adapter)
    assert sum(value.size for value in adapter.values()) == report["trainable_parameters"]
    b_values = {name: value for name, value in adapter.items() if name.endswith(".lora_b")}
    changed_b = [name for name, value in b_values.items() if np.any(value != 0)]
    assert changed_b
    run = json.loads((report_path.parent / "run_config.json").read_text())
    assert not run.get("resume") and not run.get("resume_adapter")
    data_path = project / "data/train.jsonl"
    assert digest(data_path) == manifest["data_files"]["train.jsonl"]
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert len(rows[0]["input_ids"]) - rows[0]["loss_start"] == record["supervised_tokens"]
    result = {
        "scope": "One fresh update artifact audit; no kernel-quality or generalization claim",
        "project": str(project),
        "implementation": manifest["implementation"],
        "resources": report["resources"],
        "training": report,
        "checkpoint_tensor_audit": {
            "checksums_match": True,
            "all_saved_tensors_finite": True,
            "only_adapter_parameters_exported": True,
            "adapter_tensor_count": len(adapter),
            "adapter_parameter_count": sum(value.size for value in adapter.values()),
            "optimizer_tensor_count": len(optimizer),
            "optimizer_step": int(optimizer["step"].item()),
            "lora_b_tensor_count": len(b_values),
            "nonzero_lora_b_count": len(changed_b),
            "zero_b_initialization": "Fresh run config; B is zero-initialized by the recorded implementation",
        },
        "receipts_sha256": {
            "project.json": digest(project / "project.json"),
            "training.json": digest(report_path),
            "run_config.json": digest(report_path.parent / "run_config.json"),
            "data/train.jsonl": digest(data_path),
            "trainer_state.json": digest(checkpoint / "trainer_state.json"),
            **state["files"],
        },
        "quality_certified": False,
    }
    if not checkpoint_only:
        fit_path = project / "fit.json"
        fit = json.loads(fit_path.read_text())
        assert fit["evaluation_scope"] == "training_only" and fit["quality_certified"] is False
        assert fit["available_examples"] == fit["evaluated_examples"] == 1
        assert fit["base"]["examples"] == fit["adapter"]["examples"] == 1
        assert (
            fit["selected_examples_sha256"]
            == hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        assert fit["selected_checkpoint"] == str(checkpoint)
        assert (
            fit["base"]["supervised_tokens"]
            == fit["adapter"]["supervised_tokens"]
            == record["supervised_tokens"]
        )
        assert fit["training_data_sha256"] == manifest["data_files"]["train.jsonl"]
        assert all(math.isfinite(fit[k]["loss"]) for k in ("base", "adapter"))
        difference = abs(fit["base"]["loss"] - record["loss"])
        assert difference < 1e-5, f"Training/eval base-loss mismatch: {difference}"
        result["fit"] = fit
        result["fit_audit"] = {
            "fresh_base_matches_training_loss_absolute_error": difference,
            "fresh_adapter_logits_changed": fit["base"]["logits_sha256"]
            != fit["adapter"]["logits_sha256"],
            "training_target_nll_improved": fit["adapter"]["loss"] < fit["base"]["loss"],
        }
        result["receipts_sha256"]["fit.json"] = digest(fit_path)
    result["audit_script_sha256"] = digest(Path(__file__))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(args.project.resolve()):
        parser.error("Write the audit outside the project so its evidence stays unchanged")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; choose a new receipt path")
    result = audit(args.project.resolve(), args.checkpoint_only)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("fit_audit", result["checkpoint_tensor_audit"]), indent=2))
