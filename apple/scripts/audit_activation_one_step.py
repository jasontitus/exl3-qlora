"""CPU-only audit of the predeclared one-step activation-mode experiment.

This validates saved evidence, not execution authenticity, upstream derivatives,
or Metal-kernel quality. It imports no MLX and never edits input artifacts.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

LOSS_TOLERANCE = 1e-6
NORM_RTOL = 1e-5
UPDATE_RTOL = 3e-6
UPDATE_ATOL = 1e-10
CHECKPOINT_FILES = {
    "adapter.safetensors",
    "adapter_config.json",
    "optimizer.safetensors",
    "random.safetensors",
}
EXPECTED_KEYS = {
    f"blocks.{layer}.attn.linears.{projection}.lora_{factor}"
    for layer in range(40)
    for projection in ("wq_a", "wo_b")
    for factor in ("a", "b")
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def finite(value, label, *, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), f"Nonfinite/invalid {label}")
    require(not nonnegative or value >= 0, f"Negative {label}")
    return float(value)


def integer(value, expected, label):
    require(type(value) is int and value == expected, f"Invalid {label}; expected {expected}")


def same_path(actual, expected, label):
    require(isinstance(actual, str), f"Missing {label}")
    require(Path(actual).expanduser().resolve() == Path(expected).resolve(), f"Mismatched {label}")


def mode(config):
    value = config.get("activation_mode", "checkpoint")
    require(value in ("checkpoint", "bf16"), "Invalid saved activation mode")
    return value


def row_file(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(rows) == 1, "Expected exactly one training row")
    row = rows[0]
    ids = row.get("input_ids")
    require(isinstance(ids, list) and len(ids) == 5809, "Expected 5809 serialized token IDs")
    require(all(type(x) is int and x >= 0 for x in ids), "Invalid token IDs")
    integer(row.get("loss_start"), 4998, "loss_start")
    return path, row


def read_checkpoint(directory):
    directory = Path(directory).resolve()
    state = read_json(directory / "trainer_state.json")
    require(state.get("format") == "exl3-mlx-training-state-v1", "Wrong checkpoint format")
    integer(state.get("step"), 1, "checkpoint step")
    integer(state.get("cursor"), 1, "checkpoint cursor")
    require(set(state.get("files", {})) == CHECKPOINT_FILES, "Incomplete checkpoint manifest")
    for name, expected in state["files"].items():
        require(digest(directory / name) == expected, f"Checkpoint checksum mismatch: {name}")
    config = read_json(directory / "adapter_config.json")
    require(
        config.get("adapter_sha256") == state["files"]["adapter.safetensors"],
        "Adapter's internal hash differs",
    )
    require(
        config.get("training") == {k: v for k, v in state.items() if k not in {"format", "files"}},
        "Adapter/trainer state disagreement",
    )
    arrays = load_file(str(directory / "adapter.safetensors"))
    optimizer = load_file(str(directory / "optimizer.safetensors"))
    rng = load_file(str(directory / "random.safetensors"))
    require(set(arrays) == EXPECTED_KEYS, "Expected only all40 attention A/B adapter tensors")
    expected_optimizer = {f"{key}.{moment}" for key in arrays for moment in ("m", "v")}
    require(
        set(optimizer) == expected_optimizer | {"step", "learning_rate"},
        "Unexpected optimizer state structure",
    )
    require(bool(rng), "Missing RNG tensors")
    for group in (arrays, optimizer, rng):
        require(all(np.isfinite(value).all() for value in group.values()), "Nonfinite saved tensor")
    require(
        optimizer["step"].shape == ()
        and optimizer["step"].dtype.kind in "iu"
        and optimizer["step"].item() == 1,
        "Optimizer must be at step1",
    )
    require(
        optimizer["learning_rate"].shape == ()
        and np.isclose(optimizer["learning_rate"].item(), 1e-5, rtol=1e-7, atol=0),
        "Unexpected optimizer learning rate",
    )
    for key, value in arrays.items():
        require(value.dtype == np.float32 and value.ndim == 2, f"Expected FP32 adapter: {key}")
        for moment in ("m", "v"):
            saved = optimizer[f"{key}.{moment}"]
            require(
                saved.shape == value.shape and saved.dtype == value.dtype,
                f"Moment shape/dtype mismatch: {key}",
            )
        require(np.all(optimizer[f"{key}.v"] >= 0), f"Negative second moment: {key}")
        if key.endswith(".lora_a"):
            require(
                not np.any(optimizer[f"{key}.m"]) and not np.any(optimizer[f"{key}.v"]),
                f"First-step A moments must be zero: {key}",
            )
    require(
        config.get("adapter_layers") == list(range(40))
        and config.get("expert_adapter_layers") == [],
        "Wrong adapter layer scope",
    )
    integer(config.get("rank"), 4, "rank")
    require(finite(config.get("alpha"), "alpha") == 8.0, "Wrong alpha")
    return directory, state, config, arrays, optimizer


def audit_moments(arrays, optimizer, norm):
    norm = finite(norm, "gradient_norm", nonnegative=True)
    require(norm > 0, "Expected nonzero gradient norm")
    clip_scale = min(1.0, 1.0 / (norm + 1e-6))
    squared_norm = 0.0
    prediction = 0.0
    tensor_rows = []
    for key in sorted(arrays):
        if key.endswith(".lora_a"):
            continue
        value = arrays[key].astype(np.float64)
        require(np.any(value != 0), f"Unchanged B tensor: {key}")
        m = optimizer[key + ".m"].astype(np.float64)
        v = optimizer[key + ".v"].astype(np.float64)
        clipped = m / 0.1
        raw = clipped / clip_scale
        require(
            np.allclose(v, 0.001 * clipped**2, rtol=3e-6, atol=1e-20),
            f"Moments are not a fresh first Adam step: {key}",
        )
        expected_update = -float(optimizer["learning_rate"]) * m / (np.sqrt(v) + 1e-8)
        require(
            np.allclose(value, expected_update, rtol=UPDATE_RTOL, atol=UPDATE_ATOL),
            f"Saved B does not match uncorrected first Adam update: {key}",
        )
        contribution = float(np.sum(raw * value, dtype=np.float64))
        squared_norm += float(np.sum(raw**2, dtype=np.float64))
        prediction += contribution
        tensor_rows.append(
            {
                "key": key,
                "shape": list(value.shape),
                "raw_gradient_norm": float(np.linalg.norm(raw)),
                "update_norm": float(np.linalg.norm(value)),
                "g_dot_delta": contribution,
            }
        )
    reconstructed_norm = math.sqrt(squared_norm)
    require(
        math.isclose(reconstructed_norm, norm, rel_tol=NORM_RTOL, abs_tol=1e-12),
        "Moment-reconstructed gradient norm differs from recorded pre-clip norm",
    )
    require(prediction < 0, "Expected negative first Adam surrogate direction")
    return {
        "changed_b_tensors": len(tensor_rows),
        "clip_scale": clip_scale,
        "reported_raw_gradient_norm": norm,
        "reconstructed_raw_gradient_norm": reconstructed_norm,
        "g_dot_actual_delta": prediction,
        "tensors": tensor_rows,
    }


def check_post(post, expected_targets, checkpoint, model, expected_mode):
    integer(post.get("examples"), 1, "post-evaluation examples")
    integer(post.get("supervised_tokens"), 811, "post-evaluation targets")
    require(mode(post) == expected_mode, "Post-evaluation used the wrong activation mode")
    same_path(post.get("adapter"), checkpoint, "post-evaluation adapter")
    same_path(post.get("checkpoint"), model, "post-evaluation model")
    require(post.get("logits_scope") == "supervised_rows", "Wrong post logits scope")
    value = post.get("logits_sha256")
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "Invalid post logits digest",
    )
    rows = post.get("rows")
    require(isinstance(rows, list) and len(rows) == 1, "Post evaluation requires --token-metrics")
    row = rows[0]
    integer(row.get("supervised_tokens"), 811, "post row targets")
    require(
        row.get("target_ids") == expected_targets
        and all(type(v) is int for v in row["target_ids"]),
        "Post target IDs differ",
    )
    predicted = row.get("predicted_ids")
    require(
        isinstance(predicted, list)
        and len(predicted) == 811
        and all(type(v) is int and v >= 0 for v in predicted),
        "Invalid predicted IDs",
    )
    token_nll = row.get("token_nll")
    require(isinstance(token_nll, list) and len(token_nll) == 811, "Incomplete token NLL")
    values = [finite(v, "token NLL", nonnegative=True) for v in token_nll]
    loss = finite(post.get("loss"), "post loss", nonnegative=True)
    row_loss = finite(row.get("loss"), "post row loss", nonnegative=True)
    mean = math.fsum(values) / len(values)
    require(
        abs(loss - mean) <= LOSS_TOLERANCE and abs(row_loss - mean) <= LOSS_TOLERANCE,
        "Post scalar loss does not match complete token NLL mean",
    )
    integer(
        row.get("correct_tokens"),
        sum(a == b for a, b in zip(predicted, expected_targets)),
        "correct-token count",
    )
    return loss


def check_invocation(receipt, post_path, data_path, checkpoint, model):
    argv = receipt.get("argv")
    require(
        isinstance(argv, list) and all(isinstance(x, str) for x in argv),
        "Invocation needs argv list",
    )
    require("evaluate" in argv, "Expected evaluation invocation")

    def argument(flag):
        occurrences = [
            i for i, value in enumerate(argv) if value == flag or value.startswith(flag + "=")
        ]
        require(len(occurrences) == 1, f"Invocation must contain exactly one {flag}")
        index = occurrences[0]
        if "=" in argv[index]:
            return argv[index].split("=", 1)[1]
        require(index + 1 < len(argv), f"Missing invocation value: {flag}")
        return argv[index + 1]

    invoked_data, _ = row_file(argument("--data"))
    require(digest(invoked_data) == digest(data_path), "Evaluation invocation data differs")
    same_path(argument("--adapter"), checkpoint, "invocation adapter")
    same_path(argument("--model"), model, "invocation model")
    same_path(argument("--report"), post_path, "invocation report")
    require(argument("--activation-mode") == "bf16", "Invocation mode differs")
    require(argument("--chunk-size") == "8192", "Invocation chunk size differs")
    require(argument("--seq-length") == "5808", "Invocation sequence length differs")
    require(argument("--prefix-cache-mib") == "0", "Invocation must disable prefix cache")
    require("--token-metrics" in argv, "Invocation needs --token-metrics")
    if "report_sha256" in receipt:
        require(receipt["report_sha256"] == digest(post_path), "Invocation report digest differs")
    if "data_sha256" in receipt:
        require(receipt["data_sha256"] == digest(data_path), "Invocation data digest differs")
    return {
        "provided": True,
        "argv_data_and_settings_match": True,
        "execution_authenticity": "Recorded argv is evidence of intended invocation, not proof that a fresh process executed it.",
    }


def check_context(post, post_path, data, row, checkpoint, model, contract):
    context = post.get("evaluation_context")
    require(isinstance(context, dict), "Post evaluation lacks evaluation_context")
    same_path(context.get("data"), data, "evaluation context data")
    require(
        context.get("input_identity") == {"data_sha256": digest(data)},
        "Evaluation context data hash differs",
    )
    selected = hashlib.sha256(
        json.dumps([row], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        context.get("selected_examples_sha256") == selected,
        "Loaded evaluation context IDs or mask differ",
    )
    integer(context.get("seq_length"), 5808, "evaluation context length")
    require(
        type(context.get("requested_max_examples")) is int
        and context["requested_max_examples"] in (0, 1),
        "Unexpected evaluation subset",
    )
    require(
        type(context.get("process_id")) is int and context["process_id"] > 0,
        "Missing evaluation process ID",
    )
    implementation = context.get("implementation")
    require(
        isinstance(implementation, dict) and bool(implementation.get("source")),
        "Missing evaluation implementation identity",
    )
    for package, key in (("mlx", "mlx"), ("mlx-lm", "mlx_lm")):
        require(
            implementation.get("runtime", {}).get(package) == contract[key],
            f"Evaluation dependency differs: {package}",
        )
    checked = check_invocation(context, post_path, data, checkpoint, model)
    return {
        **checked,
        "selected_context_and_target_ids_match": True,
        "data_hash_matches": True,
        "process_id": context["process_id"],
        "implementation": implementation,
        "fresh_process_limit": "A process ID alone does not prove independence from training; inspect the saved launch logs.",
    }


def audit(
    original_checkpoint, training_report, post_evaluation, config, protocol, post_invocation=None
):
    original, old_state, old_config, old_arrays, old_optimizer = read_checkpoint(
        original_checkpoint
    )
    training_report, post_evaluation, config, protocol = map(
        Path, (training_report, post_evaluation, config, protocol)
    )
    report, planned, plan = map(read_json, (training_report, config, protocol))
    require(plan.get("schema") == "exl3-activation-causal-plan-v2", "Wrong predeclared protocol")
    require(
        plan["predeclared_scoring"]["absolute_nll_guard"] == LOSS_TOLERANCE,
        "Protocol tolerance differs",
    )
    same_path(plan["original_control"]["checkpoint"], original, "protocol original checkpoint")
    for path, expected in plan["original_control"]["files_sha256"].items():
        require(digest(path) == expected, f"Original protocol artifact changed: {path}")
    run_path = training_report.parent / "run_config.json"
    run = read_json(run_path)
    require(report.get("status") == "completed", "Training is incomplete")
    for key in ("completed_step", "additional_steps", "requested_steps"):
        integer(report.get(key), 1, key)
    history = report.get("history")
    require(
        isinstance(history, list) and len(history) == 1, "Expected exactly one optimizer update"
    )
    record = history[0]
    for key, expected in (
        ("step", 1),
        ("cursor", 1),
        ("microbatches", 1),
        ("supervised_tokens", 811),
    ):
        integer(record.get(key), expected, key)
    baseline = finite(record.get("loss"), "pre-update loss", nonnegative=True)
    current, state, saved, arrays, optimizer = read_checkpoint(report["checkpoint"])
    require(
        state["metrics"] == record and state["contract"] == report.get("contract"),
        "Training report/checkpoint state disagreement",
    )
    require(
        mode(old_config) == "checkpoint" and mode(saved) == mode(run) == mode(report) == "bf16",
        "Expected original checkpoint-mode and treatment bf16-mode",
    )
    require(
        saved.get("base_identity") == old_config.get("base_identity"), "Base model metadata differs"
    )
    require(
        state["contract"] == old_state["contract"],
        "Original optimizer/data/arithmetic contract differs",
    )
    for key, expected in planned.items():
        if key in {"model", "data", "output"}:
            same_path(run.get(key), expected, f"planned {key}")
        else:
            require(
                run.get(key) == expected, f"Resolved run differs from predeclared config: {key}"
            )
    expected_run = dict(
        steps=1,
        seq_length=5808,
        rank=4,
        alpha=8.0,
        learning_rate=1e-5,
        grad_accumulation=1,
        clip_grad=1.0,
        seed=8,
        chunk_size=8192,
        prefix_cache_mib=0,
        adapter_layers=list(range(40)),
        expert_adapter_layers=[],
        activation_mode="bf16",
    )
    for key, expected in expected_run.items():
        require(run.get(key) == expected, f"Wrong experiment setting: {key}")
    require(
        not run.get("resume")
        and not run.get("resume_adapter")
        and not run.get("truncate")
        and not run.get("no_checkpoint"),
        "Expected fresh untruncated recomputed run",
    )
    same_path(run["output"], training_report.parent, "training output")
    require(
        current.parent == training_report.parent.resolve(),
        "Checkpoint is outside this training run",
    )
    require(report.get("block_recomputation") is True, "Recomputation report differs")
    original_run = read_json(original.parent / "run_config.json")
    if "model" in original_run:
        same_path(run["model"], original_run["model"], "original model")
    if "model" in plan["matched_settings"]:
        same_path(run["model"], plan["matched_settings"]["model"], "protocol model")
    original_data, original_row = row_file(original_run["data"])
    data, row = row_file(run["data"])
    require(
        row["input_ids"] == original_row["input_ids"]
        and row["loss_start"] == original_row["loss_start"],
        "Original/new context IDs or mask differ",
    )
    require(
        digest(data)
        == digest(original_data)
        == state["contract"]["data_sha256"]
        == plan["matched_settings"]["data_sha256"],
        "Prepared data hash mismatch",
    )
    a_keys = sorted(key for key in arrays if key.endswith(".lora_a"))
    require(
        all(
            arrays[key].shape == old_arrays[key].shape
            and arrays[key].dtype == old_arrays[key].dtype
            and arrays[key].tobytes() == old_arrays[key].tobytes()
            for key in a_keys
        ),
        "A initialization differs from original step1 A",
    )
    require(
        all(arrays[key].shape == old_arrays[key].shape for key in arrays), "Adapter shapes differ"
    )
    integer(
        report.get("trainable_parameters"), sum(a.size for a in arrays.values()), "parameter count"
    )
    moments = audit_moments(arrays, optimizer, record["gradient_norm"])
    old_moments = audit_moments(old_arrays, old_optimizer, old_state["metrics"]["gradient_norm"])
    post = read_json(post_evaluation)
    post_loss = check_post(
        post, row["input_ids"][row["loss_start"] :], current, run["model"], "bf16"
    )
    invocation = check_context(
        post, post_evaluation, data, row, current, run["model"], state["contract"]
    )
    if post_invocation:
        invocation["separate_launch_receipt"] = check_invocation(
            read_json(post_invocation), post_evaluation, data, current, run["model"]
        )
    delta = post_loss - baseline
    outcome = (
        "descent"
        if delta < -LOSS_TOLERANCE
        else "regression"
        if delta > LOSS_TOLERANCE
        else "unresolved"
    )
    inputs = [
        training_report,
        run_path,
        post_evaluation,
        config,
        protocol,
        data,
        original_data,
        original.parent / "run_config.json",
    ]
    for directory in (original, current):
        inputs.extend(directory / name for name in CHECKPOINT_FILES | {"trainer_state.json"})
    if post_invocation:
        inputs.append(Path(post_invocation))
    return {
        "schema": "exl3-activation-one-step-audit-v1",
        "audit_completed": True,
        "scope": "One training row, one fresh first Adam update; no quality certification",
        "quality_certified": False,
        "matched_causal_claim": False,
        "causal_limit": "Historical control only unless separately replayed under the new source; this audit does not establish replay or source equivalence.",
        "own_baseline_kind": "Complete uncached zero-B pre-update training objective",
        "evidence_checks": {
            "complete_training": True,
            "checkpoint_checksums": True,
            "finite_saved_tensors": True,
            "all80_A_bitwise_equal_original": True,
            "all_A_first_moments_zero": True,
            "all80_B_changed_from_zero": True,
            "exact_original_context_ids_mask": True,
            "post_target_ids_and_scalar_mean": True,
        },
        "zero_B_initialization_evidence": "Fresh run_config plus recorded initializer; no pre-update B tensor snapshot is inferred from a post checkpoint.",
        "baseline": baseline,
        "fresh_post_loss": post_loss,
        "actual_loss_delta": delta,
        "loss_change_tolerance": LOSS_TOLERANCE,
        "training_fit_outcome": outcome,
        "surrogate_prediction": moments["g_dot_actual_delta"],
        "actual_minus_surrogate_prediction": delta - moments["g_dot_actual_delta"],
        "moments": moments,
        "original_moments": old_moments,
        "gradient_reconstruction": "CPUfloat64 m/.1 then undo recorded clipping; retains FP32 moment rounding and is not an independent gradient oracle.",
        "tolerances": {
            "loss_abs": LOSS_TOLERANCE,
            "norm_relative": NORM_RTOL,
            "update_relative": UPDATE_RTOL,
            "update_abs": UPDATE_ATOL,
        },
        "evaluation_invocation": invocation,
        "resources": report.get("resources"),
        "original_checkpoint": str(original),
        "new_checkpoint": str(current),
        "input_sha256": {str(p.resolve()): digest(p) for p in inputs},
        "audit_script_sha256": digest(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "original-checkpoint",
        "training-report",
        "post-evaluation",
        "config",
        "protocol",
        "output",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--post-invocation", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    roots = (args.original_checkpoint.resolve().parent, args.training_report.resolve().parent)
    if any(output.is_relative_to(root) for root in roots):
        parser.error("Output must be outside both training directories")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; choose a new receipt path")
    try:
        result = audit(
            args.original_checkpoint,
            args.training_report,
            args.post_evaluation,
            args.config,
            args.protocol,
            args.post_invocation,
        )
        with output.open("x") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "training_fit_outcome",
                    "baseline",
                    "fresh_post_loss",
                    "actual_loss_delta",
                    "surrogate_prediction",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
