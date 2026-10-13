"""CPU-only receipt tamper tests; no model imports or GPU activity."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ROOT = Path(__file__).parents[1]
m = module(ROOT / "scripts/audit_matched_control.py", "matched_control_audit")
fixture = module(
    Path(__file__).with_name("test_activation_one_step_audit.py"), "activation_fixtures"
)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


@pytest.fixture
def evidence(tmp_path):
    e = fixture.evidence.__wrapped__(tmp_path)
    treatment_report = m.read(e["training_report"])
    treatment_report.update(peak_mlx_gib=73.31, observed_memory_gib=23.22)
    write(e["training_report"], treatment_report)
    treatment_config = m.read(e["config"])
    runtime = tmp_path / "runtime"
    helper = runtime / "scripts/audit_activation_one_step.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((ROOT / "scripts/audit_activation_one_step.py").read_bytes())
    source = runtime / "src/exl3_mlx/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Fixture only\n")
    lock = runtime / "uv.lock"
    lock.write_text("fixture\n")
    snapshot = {str(p.relative_to(runtime)): m.digest(p) for p in (helper, source, lock)}
    write(runtime / "snapshot.json", snapshot)
    treatment_post = m.read(e["post_evaluation"])
    treatment_post["evaluation_context"]["implementation"]["source"] = {
        "fixture.py": m.digest(source)
    }
    treatment_post["evaluation_context"]["implementation"]["dependency_lock"] = m.digest(lock)
    bf_post = tmp_path / "bf16-all-layers-post.json"
    treatment_argv = treatment_post["evaluation_context"]["argv"]
    treatment_argv[treatment_argv.index("--report") + 1] = str(bf_post)
    write(bf_post, treatment_post)
    out = tmp_path / "control"
    configpath = tmp_path / "control-config.json"
    config = {**treatment_config, "output": str(out), "activation_mode": "checkpoint"}
    write(configpath, config)
    treatment_run = {**treatment_config, "config": str(e["config"])}
    write(e["training_report"].parent / "run_config.json", treatment_run)
    state, count = fixture.checkpoint(
        out / "step-000001", "checkpoint", treatment_report["contract"], 0.001
    )
    write(out / "run_config.json", {**config, "config": str(configpath)})
    report = {
        **treatment_report,
        "history": [state["metrics"]],
        "checkpoint": str(out / "step-000001"),
        "activation_mode": "checkpoint",
        "trainable_parameters": count,
        "resources": {},
    }
    training = out / "training.json"
    write(training, report)
    post = {
        **copy.deepcopy(treatment_post),
        "adapter": str(out / "step-000001"),
        "activation_mode": "checkpoint",
        "resources": {},
        "expert_manifest_sha256": "fixture",
    }
    post["rows"][0]["logits_sha256"] = post["logits_sha256"]
    postpath = tmp_path / "control-post.json"
    argv = list(post["evaluation_context"]["argv"])
    for flag, value in (
        ("--adapter", post["adapter"]),
        ("--activation-mode", "checkpoint"),
        ("--report", str(postpath)),
    ):
        argv[argv.index(flag) + 1] = value
    post["evaluation_context"]["argv"] = argv
    write(postpath, post)
    treatment_post["expert_manifest_sha256"] = "fixture"
    write(bf_post, treatment_post)
    protocol = tmp_path / "activation-causal-plan-v2.json"
    write(protocol, m.read(e["protocol"]))
    helper_module = module(helper, "pinned_fixture_reader")
    prior = helper_module.audit(
        e["original_checkpoint"], e["training_report"], bf_post, e["config"], protocol
    )
    prior_path = tmp_path / "bf16-all-layers-one-step-audit.json"
    write(prior_path, prior)
    historical = m.read(e["protocol"])["original_control"]
    historical.update(
        pre_update_train_loss=0.5, fresh_post_nll=0.4, fresh_post_logits_sha256="a" * 64
    )
    plan = dict(
        schema="exl3-matched-checkpoint-control-plan-v1",
        runtime=str(runtime),
        runtime_snapshot_sha256=m.digest(runtime / "snapshot.json"),
        config=str(configpath),
        treatment_config=str(e["config"]),
        bound_inputs_sha256={
            str(p): m.digest(p)
            for p in (
                configpath,
                e["config"],
                bf_post,
                protocol,
                prior_path,
                e["training_report"],
                e["training_report"].parent / "run_config.json",
                Path(treatment_config["data"]),
                runtime / "snapshot.json",
            )
        },
        historical_control=historical,
        expected_resources={},
        treatment=dict(
            checkpoint=treatment_report["checkpoint"],
            pre_update_loss=0.5,
            fresh_post_loss=0.4,
            delta=0.4 - 0.5,
        ),
        launch=dict(
            post_report=str(postpath), post_argv=["python", "-m", "exl3_mlx.workflow", *argv]
        ),
    )
    planpath = tmp_path / "plan.json"
    write(planpath, plan)
    return dict(plan_path=planpath, training_path=training, post_path=postpath)


def test_full_synthetic_audit_and_negative_causal_conclusion(evidence):
    result = m.audit(**evidence)
    assert result["historical_update_reproduced"] is True
    assert result["control"]["outcome"] == "descent"
    assert result["observed_regression_to_descent_contrast"] is False
    assert result["quality_certified"] is False


@pytest.mark.parametrize("loss,label", [(0.6, "regression"), (0.5000002, "unresolved")])
def test_preserves_nonhistorical_and_unresolved_outcomes(evidence, loss, label):
    post = m.read(evidence["post_path"])
    post["loss"] = post["rows"][0]["loss"] = loss
    post["rows"][0]["token_nll"] = [loss] * 811
    write(evidence["post_path"], post)
    result = m.audit(**evidence)
    assert result["control"]["outcome"] == label
    assert result["historical_update_reproduced"] is False
    assert result["observed_regression_to_descent_contrast"] == (label == "regression")


@pytest.mark.parametrize(
    "path,mutation,match",
    [
        ("training_path", lambda d: d.update(status="stopped"), "incomplete"),
        ("training_path", lambda d: d.update(activation_mode="bf16"), "control mode"),
        ("training_path", lambda d: d.update(resources={"prefix_cache_mib": 100}), "resources"),
        ("post_path", lambda d: d["rows"][0]["target_ids"].__setitem__(0, 999), "target IDs"),
        (
            "post_path",
            lambda d: d["rows"][0]["token_nll"].__setitem__(0, float("nan")),
            "Nonfinite",
        ),
        (
            "post_path",
            lambda d: d["evaluation_context"].update(selected_examples_sha256="bad"),
            "IDs/mask",
        ),
        (
            "post_path",
            lambda d: d["evaluation_context"]["argv"].extend(["--activation-mode", "bf16"]),
            "argv",
        ),
        (
            "post_path",
            lambda d: d["evaluation_context"]["implementation"].update(dependency_lock="bad"),
            "implementation",
        ),
        ("post_path", lambda d: d["rows"][0].update(logits_sha256="b" * 64), "row/aggregate"),
        ("plan_path", lambda d: d["treatment"].update(delta=-0.2), "delta differs"),
    ],
)
def test_tampered_receipts_fail_closed(evidence, path, mutation, match):
    data = m.read(evidence[path])
    mutation(data)
    write(evidence[path], data)
    with pytest.raises(ValueError, match=match):
        m.audit(**evidence)


def test_runtime_tamper_rejected_before_helper_import(evidence):
    runtime = Path(m.read(evidence["plan_path"])["runtime"])
    (runtime / "scripts/audit_activation_one_step.py").write_text(
        "raise RuntimeError('must not execute')\n"
    )
    with pytest.raises(ValueError, match="Pinned runtime changed"):
        m.audit(**evidence)


def treatment_files(evidence):
    plan = m.read(evidence["plan_path"])
    checkpoint = Path(plan["treatment"]["checkpoint"])
    return plan, {
        "checkpoint": checkpoint / "adapter_config.json",
        "run": checkpoint.parent / "run_config.json",
        "report": checkpoint.parent / "training.json",
        "post": m.bound_path(plan, "bf16-all-layers-post.json"),
    }


def refresh_prior_binding(evidence, path):
    """Test defense in depth after repairing outer hashes; not used on real data."""
    plan = m.read(evidence["plan_path"])
    prior_path = m.bound_path(plan, "bf16-all-layers-one-step-audit.json")
    prior = m.read(prior_path)
    prior["input_sha256"][str(path)] = m.digest(path)
    write(prior_path, prior)
    if str(path) in plan["bound_inputs_sha256"]:
        plan["bound_inputs_sha256"][str(path)] = m.digest(path)
    plan["bound_inputs_sha256"][str(prior_path)] = m.digest(prior_path)
    write(evidence["plan_path"], plan)


def test_repaired_local_checksum_cannot_relabel_prior_treatment(evidence):
    _, files = treatment_files(evidence)
    config = m.read(files["checkpoint"])
    config["activation_mode"] = "checkpoint"
    write(files["checkpoint"], config)
    state_path = files["checkpoint"].parent / "trainer_state.json"
    state = m.read(state_path)
    state["files"]["adapter_config.json"] = m.digest(files["checkpoint"])
    write(state_path, state)
    with pytest.raises(ValueError, match="Changed treatment audit input"):
        m.audit(**evidence)


@pytest.mark.parametrize("which", ["checkpoint", "run", "report", "post"])
def test_explicit_bf16_mode_required_even_with_refreshed_receipt_hash(evidence, which):
    _, files = treatment_files(evidence)
    data = m.read(files[which])
    data["activation_mode"] = "checkpoint"
    write(files[which], data)
    refresh_prior_binding(evidence, files[which])
    with pytest.raises(ValueError, match=f"Treatment {which} mode must be bf16"):
        m.audit(**evidence)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda p: p["input_sha256"].pop(
                next(k for k in p["input_sha256"] if k.endswith("optimizer.safetensors"))
            ),
            "input paths",
        ),
        (lambda p: p.update(new_checkpoint="/tmp/unrelated-checkpoint"), "checkpoint path"),
        (lambda p: p.update(audit_completed=False), "not complete"),
        (lambda p: p.update(baseline=0.6), "audit rerun differs"),
    ],
)
def test_prior_audit_structure_and_outcome_are_bound(evidence, mutation, match):
    plan = m.read(evidence["plan_path"])
    prior_path = m.bound_path(plan, "bf16-all-layers-one-step-audit.json")
    prior = m.read(prior_path)
    mutation(prior)
    write(prior_path, prior)
    plan["bound_inputs_sha256"][str(prior_path)] = m.digest(prior_path)
    write(evidence["plan_path"], plan)
    with pytest.raises(ValueError, match=match):
        m.audit(**evidence)


def test_treatment_report_state_disagreement_rejected_with_refreshed_hash(evidence):
    _, files = treatment_files(evidence)
    report = m.read(files["report"])
    report["history"][0]["loss"] = 0.6
    write(files["report"], report)
    refresh_prior_binding(evidence, files["report"])
    with pytest.raises(ValueError, match="report/checkpoint state disagreement"):
        m.audit(**evidence)


@pytest.fixture
def amended_evidence(evidence):
    """A real receipt fixture amended prospectively; no model or child execution."""
    oldpath = evidence["plan_path"]
    old = m.read(oldpath)
    resources = dict(
        profile="development",
        memory_gib=80.0,
        reserve_gib=8.0,
        packed_cache_mib=8192,
        decoded_cache_mib=0,
        prefix_cache_mib=0,
        allocator_cache_mib=256,
    )
    old["expected_resources"] = resources
    old["launch"]["train_argv"] = [
        "python",
        "-m",
        "exl3_mlx.workflow",
        "train",
        "--config",
        old["config"],
    ]
    write(oldpath, old)
    plan = copy.deepcopy(old)
    oldconfig = m.read(old["config"])
    out = oldpath.parent / "control-76g"
    out.mkdir()
    # Relocate fixture receipts/checkpoint to the new run without changing tensors.
    import shutil

    shutil.copytree(Path(oldconfig["output"]) / "step-000001", out / "step-000001")
    configpath = oldpath.parent / "control-76g-config.json"
    config = {**oldconfig, "output": str(out), "memory_gib": 76.0}
    write(configpath, config)
    write(out / "run_config.json", {**config, "config": str(configpath)})
    report = m.read(evidence["training_path"])
    report.update(checkpoint=str(out / "step-000001"), resources={**resources, "memory_gib": 76.0})
    training = out / "training.json"
    write(training, report)
    postpath = oldpath.parent / "control-76g-post.json"
    post = m.read(evidence["post_path"])
    post.update(adapter=str(out / "step-000001"), resources={**resources, "memory_gib": 72.0})
    argv = post["evaluation_context"]["argv"]
    argv[argv.index("--adapter") + 1] = post["adapter"]
    argv[argv.index("--report") + 1] = str(postpath)
    argv.extend(["--memory-gib", "72"])
    write(postpath, post)
    plan.update(
        schema="exl3-matched-checkpoint-control-plan-v2",
        config=str(configpath),
        config_sha256=m.digest(configpath),
        expected_training_resources=report["resources"],
        expected_evaluation_resources=post["resources"],
    )
    plan["launch"]["train_argv"][-1] = str(configpath)
    plan["launch"]["post_report"] = str(postpath)
    plan["launch"]["post_argv"] = [*old["launch"]["post_argv"][:3], *argv]
    failed = oldpath.parent / "failure.log"
    failed.write_text("MemoryError: Insufficient free/reclaimable system memory\n")
    declaration = oldpath.parent / "recovery.json"
    write(
        declaration,
        {"failed_adapter_log": str(failed), "failed_adapter_log_sha256": m.digest(failed)},
    )
    historical_report = Path(old["historical_control"]["checkpoint"]).parent / "training.json"
    if not historical_report.exists():
        write(historical_report, {"peak_mlx_gib": 73.95, "observed_memory_gib": 27.65})
    roles = dict(
        standard_failure_log=str(failed),
        recovery_declaration=str(declaration),
        treatment_training_report=str(
            Path(old["treatment"]["checkpoint"]).parent / "training.json"
        ),
        historical_training_report=str(historical_report),
    )
    motivation = {p: m.digest(p) for p in roles.values()}
    plan["resource_amendment"] = dict(
        original_plan=str(oldpath),
        original_plan_sha256=m.digest(oldpath),
        original_config=old["config"],
        original_config_sha256=m.digest(old["config"]),
        changes=dict(
            treatment_train_gib=80.0,
            control_train_gib=76.0,
            treatment_eval_gib=80.0,
            control_eval_gib=72.0,
            reserve_gib_all=8.0,
        ),
        memory_observations={
            role: dict(
                peak_mlx_gib=m.read(path)["peak_mlx_gib"],
                end_observed_memory_gib=m.read(path)["observed_memory_gib"],
            )
            for role, path in roles.items()
            if role.endswith("training_report")
        },
        historical_mlx_peak_headroom_gib=76.0 - m.read(historical_report)["peak_mlx_gib"],
        exact_resource_match=False,
        control_previously_launched=False,
        motivation_paths=roles,
        motivation_inputs_sha256=motivation,
    )
    plan["bound_inputs_sha256"].update(
        {str(oldpath): m.digest(oldpath), str(configpath): m.digest(configpath), **motivation}
    )
    planpath = oldpath.parent / "amended-plan.json"
    write(planpath, plan)
    return dict(plan_path=planpath, training_path=training, post_path=postpath)


def test_resource_amended_phase_caps_and_qualified_claim(amended_evidence):
    result = m.audit(**amended_evidence)
    assert result["matched_evaluation_runtime_and_mathematical_settings"] is True
    assert result["exact_resources_matched_to_treatment"] is False
    assert result["resource_comparison"]["control_training"]["memory_gib"] == 76.0
    assert result["resource_comparison"]["control_evaluation"]["memory_gib"] == 72.0
    assert result["historical_update_reproduced"] is True


@pytest.mark.parametrize(
    "phase,field,value",
    [
        ("training", "reserve_gib", 4.0),
        ("evaluation", "memory_gib", 76.0),
        ("training", "packed_cache_mib", 0),
        ("evaluation", "decoded_cache_mib", 32),
    ],
)
def test_amendment_rejects_undeclared_phase_resources(amended_evidence, phase, field, value):
    plan = m.read(amended_evidence["plan_path"])
    plan[f"expected_{phase}_resources"][field] = value
    write(amended_evidence["plan_path"], plan)
    with pytest.raises(ValueError, match="amended .* resources"):
        m.audit(**amended_evidence)


@pytest.mark.parametrize(
    "field,value", [("chunk_size", 512), ("reserve_gib", 4), ("learning_rate", 0.01)]
)
def test_amendment_cannot_change_math_or_reserve_with_repaired_hash(amended_evidence, field, value):
    plan = m.read(amended_evidence["plan_path"])
    path = Path(plan["config"])
    config = m.read(path)
    config[field] = value
    write(path, config)
    plan["config_sha256"] = plan["bound_inputs_sha256"][str(path)] = m.digest(path)
    write(amended_evidence["plan_path"], plan)
    with pytest.raises(ValueError, match="mathematical/config"):
        m.audit(**amended_evidence)


@pytest.mark.parametrize("path,cap", [("training_path", 80.0), ("post_path", 76.0)])
def test_actual_phase_receipts_must_respect_separate_caps(amended_evidence, path, cap):
    receipt = m.read(amended_evidence[path])
    receipt["resources"]["memory_gib"] = cap
    write(amended_evidence[path], receipt)
    with pytest.raises(ValueError, match="resources differ"):
        m.audit(**amended_evidence)


def test_amendment_original_plan_remains_bound(amended_evidence):
    plan = m.read(amended_evidence["plan_path"])
    old = Path(plan["resource_amendment"]["original_plan"])
    old.write_text(old.read_text() + " ")
    with pytest.raises(ValueError, match="Original plan changed"):
        m.audit(**amended_evidence)


def test_amendment_post_command_cannot_change_cache(amended_evidence):
    plan = m.read(amended_evidence["plan_path"])
    plan["launch"]["post_argv"].extend(["--decoded-cache-mib", "4096"])
    write(amended_evidence["plan_path"], plan)
    with pytest.raises(ValueError, match="launch math/runtime"):
        m.audit(**amended_evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_config_sha256", "a" * 64),
        ("original_config", "/unrelated/config.json"),
        ("changes", {"control_train_gib": 80}),
        ("memory_observations", {}),
        ("historical_mlx_peak_headroom_gib", 9),
    ],
)
def test_amendment_descriptive_resource_claims_are_verified(amended_evidence, field, value):
    plan = m.read(amended_evidence["plan_path"])
    plan["resource_amendment"][field] = value
    write(amended_evidence["plan_path"], plan)
    with pytest.raises(ValueError, match="declaration differs|observations differ|margin differs"):
        m.audit(**amended_evidence)
