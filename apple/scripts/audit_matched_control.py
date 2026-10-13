"""CPU-only audit of the predeclared fresh checkpoint-mode control.

Uses the checksummed runtime-v2 evidence reader, never imports MLX, and retains
negative/non-reproducing outcomes. This is a single-case training-fit audit.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from safetensors.numpy import load_file


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def outcome(delta):
    require(math.isfinite(delta), "Nonfinite NLL delta")
    return "descent" if delta < -1e-6 else "regression" if delta > 1e-6 else "unresolved"


def compare_tensors(current, historical):
    require(set(current) == set(historical), "Historical tensor keys differ")
    different = []
    for key in sorted(current):
        a, b = current[key], historical[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            different.append(key)
    return {"exact": not different, "count": len(current), "different": different}


def bound_path(plan, name):
    matches = [Path(p) for p in plan["bound_inputs_sha256"] if Path(p).name == name]
    require(len(matches) == 1, f"Expected exactly one bound {name}")
    return matches[0].resolve()


def verify_treatment(plan, helper):
    """Expand prior receipt hashes and independently rerun its pinned checks."""
    prior_path = bound_path(plan, "bf16-all-layers-one-step-audit.json")
    protocol_path = bound_path(plan, "activation-causal-plan-v2.json")
    post_path = bound_path(plan, "bf16-all-layers-post.json")
    prior = read(prior_path)
    require(prior.get("audit_completed") is True, "Treatment audit was not complete")
    checkpoint = Path(plan["treatment"]["checkpoint"]).resolve()
    original = Path(plan["historical_control"]["checkpoint"]).resolve()
    require(
        Path(prior["new_checkpoint"]).resolve() == checkpoint,
        "Prior treatment checkpoint path differs",
    )
    require(
        Path(prior["original_checkpoint"]).resolve() == original,
        "Prior original checkpoint path differs",
    )
    config_path = Path(plan["treatment_config"]).resolve()
    config = read(config_path)
    report_path, run_path = (
        checkpoint.parent / "training.json",
        checkpoint.parent / "run_config.json",
    )
    old_run_path = original.parent / "run_config.json"
    expected_paths = {
        report_path,
        run_path,
        post_path,
        config_path,
        protocol_path,
        helper.row_file(config["data"])[0],
        old_run_path,
        helper.row_file(read(old_run_path)["data"])[0],
        *(
            directory / name
            for directory in (checkpoint, original)
            for name in (
                "adapter.safetensors",
                "adapter_config.json",
                "optimizer.safetensors",
                "random.safetensors",
                "trainer_state.json",
            )
        ),
    }
    bindings = prior.get("input_sha256", {})
    require(isinstance(bindings, dict) and bool(bindings), "Missing treatment audit input bindings")
    canonical = {Path(path).resolve(): expected for path, expected in bindings.items()}
    require(len(canonical) == len(bindings), "Duplicate canonical treatment input binding")
    require(
        set(canonical) == expected_paths,
        "Treatment audit input paths differ from planned artifacts",
    )
    for path, expected in canonical.items():
        require(digest(path) == expected, f"Changed treatment audit input: {path}")
    report, run, post, saved = (
        read(report_path),
        read(run_path),
        read(post_path),
        read(checkpoint / "adapter_config.json"),
    )
    for label, value in (
        ("config", config),
        ("run", run),
        ("report", report),
        ("post", post),
        ("checkpoint", saved),
    ):
        require(value.get("activation_mode") == "bf16", f"Treatment {label} mode must be bf16")
    helper.same_path(config["output"], checkpoint.parent, "treatment config output")
    helper.same_path(run["output"], checkpoint.parent, "treatment run output")
    helper.same_path(run["config"], config_path, "treatment run config")
    helper.same_path(report["checkpoint"], checkpoint, "treatment report checkpoint")
    helper.same_path(post["adapter"], checkpoint, "treatment post checkpoint")
    result = helper.audit(original, report_path, post_path, config_path, protocol_path)
    require(result == prior, "Pinned treatment audit rerun differs from bound prior audit")
    for key, expected in (
        ("baseline", plan["treatment"]["pre_update_loss"]),
        ("fresh_post_loss", plan["treatment"]["fresh_post_loss"]),
        ("actual_loss_delta", plan["treatment"]["delta"]),
    ):
        require(result[key] == expected, f"Planned treatment {key} differs from audited evidence")
    return {
        "prior_audit": str(prior_path),
        "prior_audit_sha256": digest(prior_path),
        "input_bindings_verified": len(canonical),
        "pinned_audit_rerun_exact": True,
    }


def resource_expectations(plan):
    if plan["schema"] == "exl3-matched-checkpoint-control-plan-v2":
        return plan["expected_training_resources"], plan["expected_evaluation_resources"]
    return plan["expected_resources"], plan["expected_resources"]


def verify_resource_amendment(plan):
    """Permit only the predeclared cap/path amendment to a fully verified v1 plan."""
    amendment = plan["resource_amendment"]
    original_path = Path(amendment["original_plan"]).resolve()
    require(digest(original_path) == amendment["original_plan_sha256"], "Original plan changed")
    original = read(original_path)
    require(original["schema"] == "exl3-matched-checkpoint-control-plan-v1", "Wrong original plan")
    verify_plan(original)
    require(
        amendment.get("original_config") == original["config"]
        and amendment.get("original_config_sha256") == digest(original["config"]),
        "Original config declaration differs",
    )
    require(
        amendment.get("changes")
        == {
            "treatment_train_gib": 80.0,
            "control_train_gib": 76.0,
            "treatment_eval_gib": 80.0,
            "control_eval_gib": 72.0,
            "reserve_gib_all": 8.0,
        },
        "Resource changes declaration differs",
    )
    allowed = {
        "schema",
        "created_utc",
        "status",
        "config",
        "config_sha256",
        "launch",
        "bound_inputs_sha256",
        "only_planned_config_differences",
        "prelaunch_gates",
        "resource_amendment",
        "expected_training_resources",
        "expected_evaluation_resources",
    }
    require(
        all(
            plan.get(k) == original.get(k)
            for k in plan.keys() | original.keys()
            if k not in allowed
        ),
        "Resource amendment changes protected plan settings",
    )
    config, old_config = read(plan["config"]), read(original["config"])
    require(
        config == {**old_config, "memory_gib": 76.0, "output": config["output"]},
        "Resource amendment changes mathematical/config settings",
    )
    require(
        config["output"] not in (old_config["output"], read(plan["treatment_config"])["output"]),
        "Amendment requires fresh training output",
    )
    require(plan["config"] != original["config"], "Amendment must preserve original config")
    require(plan["config_sha256"] == digest(plan["config"]), "Amended config hash differs")
    original_resources = original["expected_resources"]
    require(
        original_resources["memory_gib"] == 80.0 and original_resources["reserve_gib"] == 8.0,
        "Unexpected original resource budgets",
    )
    for phase, cap in (("training", 76.0), ("evaluation", 72.0)):
        require(
            plan[f"expected_{phase}_resources"] == {**original_resources, "memory_gib": cap},
            f"Unexpected amended {phase} resources",
        )
    expected_launch = copy.deepcopy(original["launch"])
    expected_launch["train_argv"][-1] = plan["config"]
    argv = expected_launch["post_argv"]
    require("--memory-gib" not in argv, "Unexpected original explicit evaluation cap")
    argv[argv.index("--adapter") + 1] = str(Path(config["output"]) / "step-000001")
    new_post = plan["launch"]["post_report"]
    require(new_post != original["launch"]["post_report"], "Amendment requires fresh post output")
    argv[argv.index("--report") + 1] = new_post
    argv.extend(["--memory-gib", "72"])
    expected_launch["post_report"] = new_post
    require(plan["launch"] == expected_launch, "Amendment changes launch math/runtime settings")
    motivation = amendment["motivation_inputs_sha256"]
    require(isinstance(motivation, dict) and len(motivation) == 4, "Missing amendment motivation")
    paths = amendment["motivation_paths"]
    require(
        set(paths)
        == {
            "standard_failure_log",
            "recovery_declaration",
            "treatment_training_report",
            "historical_training_report",
        },
        "Wrong motivation roles",
    )
    require(set(motivation) == set(paths.values()), "Motivation paths are not all bound")
    for path, expected in motivation.items():
        require(digest(path) == expected, f"Changed amendment motivation: {path}")
    recovery = read(paths["recovery_declaration"])
    require(
        recovery["failed_adapter_log"] == paths["standard_failure_log"]
        and recovery["failed_adapter_log_sha256"] == digest(paths["standard_failure_log"]),
        "Startup failure binding differs",
    )
    require(
        "Insufficient free/reclaimable system memory"
        in Path(paths["standard_failure_log"]).read_text(),
        "Missing observed standard-fit startup failure",
    )
    require(
        Path(paths["treatment_training_report"]).resolve()
        == Path(plan["treatment"]["checkpoint"]).resolve().parent / "training.json",
        "Wrong treatment memory evidence",
    )
    require(
        Path(paths["historical_training_report"]).resolve()
        == Path(plan["historical_control"]["checkpoint"]).resolve().parent / "training.json",
        "Wrong historical memory evidence",
    )
    observed = {
        role: {
            "peak_mlx_gib": read(path)["peak_mlx_gib"],
            "end_observed_memory_gib": read(path)["observed_memory_gib"],
        }
        for role, path in paths.items()
        if role.endswith("training_report")
    }
    require(amendment.get("memory_observations") == observed, "Memory observations differ")
    require(
        amendment.get("historical_mlx_peak_headroom_gib")
        == 76.0 - observed["historical_training_report"]["peak_mlx_gib"],
        "Historical peak margin differs",
    )
    expected_bindings = {
        **original["bound_inputs_sha256"],
        str(original_path): digest(original_path),
        plan["config"]: digest(plan["config"]),
        **motivation,
    }
    require(plan["bound_inputs_sha256"] == expected_bindings, "Amendment input bindings differ")
    require(amendment.get("exact_resource_match") is False, "Resource exception must be explicit")
    require(amendment.get("control_previously_launched") is False, "Not an observed control retry")


def verify_plan(plan):
    require(
        plan["schema"]
        in ("exl3-matched-checkpoint-control-plan-v1", "exl3-matched-checkpoint-control-plan-v2"),
        "Wrong plan",
    )
    amended = plan["schema"] == "exl3-matched-checkpoint-control-plan-v2"
    if amended:
        verify_resource_amendment(plan)
    for path, expected in plan["bound_inputs_sha256"].items():
        require(digest(path) == expected, f"Changed plan input: {path}")
    runtime = Path(plan["runtime"])
    require(
        digest(runtime / "snapshot.json") == plan["runtime_snapshot_sha256"], "Runtime pin changed"
    )
    snapshot = read(runtime / "snapshot.json")
    for name, expected in snapshot.items():
        require(digest(runtime / name) == expected, f"Pinned runtime changed: {name}")
    config, treatment = read(plan["config"]), read(plan["treatment_config"])
    changed = {
        key for key in config.keys() | treatment.keys() if config.get(key) != treatment.get(key)
    }
    require(
        changed == {"activation_mode", "output"} | ({"memory_gib"} if amended else set()),
        "Unexpected treatment/control config difference",
    )
    require(
        config["activation_mode"] == "checkpoint" and treatment["activation_mode"] == "bf16",
        "Wrong modes",
    )
    helper = runtime / "scripts/audit_activation_one_step.py"
    spec = importlib.util.spec_from_file_location("pinned_activation_audit", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verify_treatment(plan, module)
    return module, config


def audit(plan_path, training_path, post_path):
    plan = read(plan_path)
    helper, config = verify_plan(plan)
    report, post = read(training_path), read(post_path)
    output = Path(config["output"]).resolve()
    require(
        Path(training_path).resolve() == output / "training.json",
        "Training report outside planned run",
    )
    run = read(output / "run_config.json")
    treatment_run = read(Path(plan["treatment"]["checkpoint"]).parent / "run_config.json")
    allowed = {"activation_mode", "output", "config"}
    if plan["schema"] == "exl3-matched-checkpoint-control-plan-v2":
        allowed.add("memory_gib")
    require(
        all(
            run.get(k) == treatment_run.get(k)
            for k in run.keys() | treatment_run.keys()
            if k not in allowed
        ),
        "Resolved control/treatment settings differ",
    )
    require(
        all(run.get(k) == value for k, value in config.items()), "Resolved config differs from plan"
    )
    require(Path(run["config"]).resolve() == Path(plan["config"]).resolve(), "Wrong config file")
    require(report.get("status") == "completed", "Control training incomplete")
    for key in ("completed_step", "additional_steps", "requested_steps"):
        helper.integer(report.get(key), 1, key)
    require(len(report["history"]) == 1, "Expected one history row")
    record = report["history"][0]
    for key, value in (("step", 1), ("cursor", 1), ("microbatches", 1), ("supervised_tokens", 811)):
        helper.integer(record.get(key), value, key)
    baseline = helper.finite(record["loss"], "own pre-update loss", nonnegative=True)
    current, state, saved, arrays, optimizer = helper.read_checkpoint(report["checkpoint"])
    historical = plan["historical_control"]
    old, old_state, old_saved, old_arrays, old_optimizer = helper.read_checkpoint(
        historical["checkpoint"]
    )
    _, treatment_state, treatment_saved, treatment_arrays, _ = helper.read_checkpoint(
        plan["treatment"]["checkpoint"]
    )
    require(current == output / "step-000001", "Wrong new control checkpoint")
    require(state["metrics"] == record, "Checkpoint/report metrics differ")
    require(
        state["contract"]
        == report["contract"]
        == old_state["contract"]
        == treatment_state["contract"],
        "Contracts differ",
    )
    require(
        saved["base_identity"] == old_saved["base_identity"] == treatment_saved["base_identity"],
        "Base identities differ",
    )
    require(
        helper.mode(saved) == helper.mode(report) == "checkpoint",
        "Wrong saved/reported control mode",
    )
    require(report.get("block_recomputation") is True, "Control did not recompute blocks")
    training_resources, evaluation_resources = resource_expectations(plan)
    require(report.get("resources") == training_resources, "Control resources differ")
    helper.integer(
        report.get("trainable_parameters"), sum(a.size for a in arrays.values()), "parameter count"
    )
    for reference in (old_arrays, treatment_arrays):
        a_comparison = compare_tensors(
            {k: v for k, v in arrays.items() if k.endswith(".lora_a")},
            {k: v for k, v in reference.items() if k.endswith(".lora_a")},
        )
        require(a_comparison["exact"], "A initialization differs")
    for path, expected in historical["files_sha256"].items():
        require(digest(path) == expected, f"Historical reference changed: {path}")
    moments = helper.audit_moments(arrays, optimizer, record["gradient_norm"])
    data, row = helper.row_file(config["data"])
    require(digest(data) == state["contract"]["data_sha256"], "Training data differs")
    post_loss = helper.check_post(
        post, row["input_ids"][row["loss_start"] :], current, config["model"], "checkpoint"
    )
    require(
        Path(post_path).resolve() == Path(plan["launch"]["post_report"]).resolve(),
        "Unplanned post report",
    )
    context = post.get("evaluation_context", {})
    require(
        context.get("argv") == plan["launch"]["post_argv"][3:], "Evaluation argv differs from plan"
    )
    helper.same_path(context.get("data"), data, "context data")
    require(
        context.get("input_identity") == {"data_sha256": digest(data)}, "Post data hash differs"
    )
    selected = hashlib.sha256(
        json.dumps([row], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(context.get("selected_examples_sha256") == selected, "Post selected IDs/mask differ")
    helper.integer(context.get("seq_length"), 5808, "post sequence length")
    helper.integer(context.get("requested_max_examples"), 0, "post subset")
    require(
        type(context.get("process_id")) is int and context["process_id"] > 0,
        "Missing post process ID",
    )
    treatment_post = read(bound_path(plan, "bf16-all-layers-post.json"))
    require(
        context.get("implementation") == treatment_post["evaluation_context"]["implementation"],
        "Control/treatment evaluation implementation differs",
    )
    snapshot = read(Path(plan["runtime"]) / "snapshot.json")
    for name, expected in context["implementation"]["source"].items():
        require(
            snapshot.get("src/exl3_mlx/" + name) == expected,
            f"Evaluation source is not pinned: {name}",
        )
    require(
        context["implementation"]["dependency_lock"] == snapshot["uv.lock"],
        "Evaluation lock is not pinned",
    )
    require(post["resources"] == evaluation_resources, "Post resources differ")
    require(
        post["rows"][0]["logits_sha256"] == post["logits_sha256"],
        "Post row/aggregate digest differs",
    )
    require(
        post["expert_manifest_sha256"] == treatment_post["expert_manifest_sha256"],
        "Expert manifest differs",
    )
    comparisons = {
        "adapters": compare_tensors(arrays, old_arrays),
        "optimizer": compare_tensors(optimizer, old_optimizer),
        "rng": compare_tensors(
            load_file(str(current / "random.safetensors")),
            load_file(str(old / "random.safetensors")),
        ),
    }
    delta = post_loss - baseline
    baseline_reproduced = abs(baseline - historical["pre_update_train_loss"]) <= 1e-6
    post_reproduced = (
        abs(post_loss - historical["fresh_post_nll"]) <= 1e-6
        and post["logits_sha256"] == historical["fresh_post_logits_sha256"]
    )
    historical_reproduced = (
        baseline_reproduced and post_reproduced and all(v["exact"] for v in comparisons.values())
    )
    treatment_baseline = helper.finite(
        treatment_state["metrics"]["loss"], "treatment baseline", nonnegative=True
    )
    treatment_delta = (
        helper.finite(treatment_post["loss"], "treatment post", nonnegative=True)
        - treatment_baseline
    )
    require(
        plan["treatment"]["delta"] == treatment_delta,
        "Planned treatment delta differs from evidence",
    )
    require(
        plan["treatment"]["pre_update_loss"] == treatment_baseline,
        "Planned treatment baseline differs",
    )
    require(
        plan["treatment"]["fresh_post_loss"] == treatment_post["loss"],
        "Planned treatment post differs",
    )
    return {
        "schema": "exl3-matched-checkpoint-control-audit-v1",
        "audit_completed": True,
        "scope": "Single previously inspected training row, one fresh update per mode; prospective control after treatment result known",
        "quality_certified": False,
        "unique_root_cause_proven": False,
        "matched_evaluation_runtime_and_mathematical_settings": True,
        "exact_resources_matched_to_treatment": plan["schema"]
        == "exl3-matched-checkpoint-control-plan-v1",
        "resource_comparison": {
            "treatment": plan["expected_resources"],
            "control_training": training_resources,
            "control_evaluation": evaluation_resources,
            "amendment": plan.get("resource_amendment"),
        },
        "training_runtime_provenance": "Requires separately inspected launch receipts; evaluation_context alone does not attest the training process runtime.",
        "treatment_prior_audit_revalidated": True,
        "historical_update_reproduced": historical_reproduced,
        "historical_baseline_reproduced": baseline_reproduced,
        "historical_post_reproduced": post_reproduced,
        "historical_tensor_comparisons": comparisons,
        "control": {
            "own_baseline": baseline,
            "fresh_post": post_loss,
            "delta": delta,
            "outcome": outcome(delta),
            "moments": moments,
            "prediction_residual": delta - moments["g_dot_actual_delta"],
        },
        "treatment": {**plan["treatment"], "outcome": outcome(treatment_delta)},
        "delta_bf16_minus_delta_checkpoint": treatment_delta - delta,
        "observed_regression_to_descent_contrast": outcome(delta) == "regression"
        and outcome(treatment_delta) == "descent",
        "limits": [
            "Different forward objectives; no absolute-quality claim",
            "Bundled quantizer intervention, routing can differ",
            "No held-out or generated-code evaluation",
            "Receipts/checksums are consistency evidence, not independent execution attestation",
            "Frozen base binding is metadata/manifest, not every payload byte",
        ],
        "evidence_sha256": {
            str(Path(p).resolve()): digest(p)
            for p in (
                plan_path,
                training_path,
                post_path,
                output / "run_config.json",
                current / "trainer_state.json",
                current / "adapter.safetensors",
                current / "optimizer.safetensors",
                current / "random.safetensors",
            )
        },
        "audit_script_sha256": digest(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "training-report", "post-evaluation", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = read(args.plan)
        protected = [Path(plan["config"]).parent, Path(plan["runtime"])]
        require(
            not any(args.output.resolve().is_relative_to(p.resolve()) for p in protected),
            "Output must be outside artifacts/runtime",
        )
        require(not args.output.exists() and not args.output.is_symlink(), "Output already exists")
        result = audit(args.plan, args.training_report, args.post_evaluation)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "historical_update_reproduced",
                    "observed_regression_to_descent_contrast",
                    "delta_bf16_minus_delta_checkpoint",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
