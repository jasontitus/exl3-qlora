"""CPU-only negative cases for the one-step activation evidence auditor."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

SCRIPT = Path(__file__).parents[1] / "scripts/audit_activation_one_step.py"
spec = importlib.util.spec_from_file_location("activation_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def checkpoint(directory, mode, contract, gradient):
    directory.mkdir(parents=True)
    arrays, optimizer = (
        {},
        {"step": np.array(1, np.uint64), "learning_rate": np.array(1e-5, np.float32)},
    )
    squared_norm = 0.0
    for key in sorted(audit_module.EXPECTED_KEYS):
        shape = (2, 4) if key.endswith(".lora_a") else (4, 2)
        g = (
            np.zeros(shape, np.float32)
            if key.endswith(".lora_a")
            else np.full(shape, gradient, np.float32)
        )
        m, v = np.float32(0.1) * g, np.float32(0.001) * g**2
        optimizer[key + ".m"], optimizer[key + ".v"] = m, v
        arrays[key] = (
            np.full(shape, 0.123, np.float32)
            if key.endswith(".lora_a")
            else -np.float32(1e-5) * m / (np.sqrt(v) + np.float32(1e-8))
        )
        squared_norm += float(np.sum(g.astype(np.float64) ** 2))
    record = dict(
        step=1,
        cursor=1,
        microbatches=1,
        supervised_tokens=811,
        loss=0.5,
        gradient_norm=squared_norm**0.5,
    )
    state = dict(
        step=1,
        cursor=1,
        contract=contract,
        metrics=record,
        selection={"validation_identity": None, "best": None},
    )
    config = dict(
        format="exl3-mlx-deepseek-short-text-v1",
        rank=4,
        alpha=8.0,
        adapter_layers=list(range(40)),
        expert_adapter_layers=[],
        base_identity={"config.json": "fixture"},
        training=state,
    )
    if mode is not None:
        config["activation_mode"] = mode
    save_file(arrays, str(directory / "adapter.safetensors"))
    config["adapter_sha256"] = audit_module.digest(directory / "adapter.safetensors")
    write(directory / "adapter_config.json", config)
    save_file(optimizer, str(directory / "optimizer.safetensors"))
    save_file({"0": np.array([1, 2], np.uint32)}, str(directory / "random.safetensors"))
    saved = dict(
        format="exl3-mlx-training-state-v1",
        **state,
        files={
            name: audit_module.digest(directory / name) for name in audit_module.CHECKPOINT_FILES
        },
    )
    write(directory / "trainer_state.json", saved)
    return saved, sum(v.size for v in arrays.values())


@pytest.fixture
def evidence(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old_data, data = old / "data.jsonl", tmp_path / "data.jsonl"
    row = {"input_ids": list(range(5809)), "loss_start": 4998}
    write(old_data, row)
    write(data, row)
    contract = dict(
        data_sha256=audit_module.digest(data),
        learning_rate=1e-5,
        grad_accumulation=1,
        clip_grad=1.0,
        seed=8,
        seq_length=5808,
        truncate=False,
        chunk_size=8192,
        no_checkpoint=False,
        engine_semantics="fixture",
        mlx="fixture",
        mlx_lm="fixture",
    )
    original_checkpoint = old / "step-000001"
    checkpoint(original_checkpoint, None, contract, 0.001)
    current = new / "step-000001"
    state, count = checkpoint(current, "bf16", contract, 0.002)
    write(old / "run_config.json", {"data": str(old_data)})
    run = dict(
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
        model=str(tmp_path / "model"),
        data=str(data),
        output=str(new),
    )
    write(new / "run_config.json", run)
    config = tmp_path / "config.json"
    write(config, run)
    report = dict(
        status="completed",
        completed_step=1,
        additional_steps=1,
        requested_steps=1,
        history=[state["metrics"]],
        checkpoint=str(current),
        contract=contract,
        activation_mode="bf16",
        block_recomputation=True,
        trainable_parameters=count,
    )
    report_path = new / "training.json"
    write(report_path, report)
    post_path = tmp_path / "post.json"
    argv = [
        "evaluate",
        "--model",
        run["model"],
        "--data",
        str(data),
        "--adapter",
        str(current),
        "--report",
        str(post_path),
        "--activation-mode",
        "bf16",
        "--chunk-size",
        "8192",
        "--seq-length",
        "5808",
        "--prefix-cache-mib",
        "0",
        "--token-metrics",
    ]
    selected = audit_module.hashlib.sha256(
        json.dumps([row], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    post = dict(
        examples=1,
        supervised_tokens=811,
        activation_mode="bf16",
        adapter=str(current),
        checkpoint=run["model"],
        logits_scope="supervised_rows",
        logits_sha256="a" * 64,
        loss=0.4,
        rows=[
            dict(
                supervised_tokens=811,
                target_ids=row["input_ids"][4998:],
                predicted_ids=row["input_ids"][4998:],
                token_nll=[0.4] * 811,
                loss=0.4,
                correct_tokens=811,
            )
        ],
        evaluation_context=dict(
            argv=argv,
            process_id=123,
            data=str(data),
            input_identity={"data_sha256": audit_module.digest(data)},
            selected_examples_sha256=selected,
            seq_length=5808,
            requested_max_examples=0,
            implementation={
                "source": {"fixture.py": "a" * 64},
                "runtime": {"mlx": "fixture", "mlx-lm": "fixture"},
            },
        ),
    )
    write(post_path, post)
    protocol = tmp_path / "protocol.json"
    write(
        protocol,
        dict(
            schema="exl3-activation-causal-plan-v2",
            predeclared_scoring={"absolute_nll_guard": 1e-6},
            matched_settings={"data_sha256": audit_module.digest(data)},
            original_control={
                "checkpoint": str(original_checkpoint),
                "files_sha256": {str(old_data): audit_module.digest(old_data)},
            },
        ),
    )
    return dict(
        original_checkpoint=original_checkpoint,
        training_report=report_path,
        post_evaluation=post_path,
        config=config,
        protocol=protocol,
    )


def change_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


def test_complete_fresh_evidence_passes_and_is_not_quality_certification(evidence):
    result = audit_module.audit(**evidence)
    assert result["training_fit_outcome"] == "descent"
    assert result["moments"]["changed_b_tensors"] == 80
    assert result["evidence_checks"]["all80_A_bitwise_equal_original"]
    assert result["quality_certified"] is False and result["matched_causal_claim"] is False
    assert result["surrogate_prediction"] < 0


@pytest.mark.parametrize("post_loss,outcome", [(0.6, "regression"), (0.5000005, "unresolved")])
def test_negative_or_unresolved_outcomes_are_retained(evidence, post_loss, outcome):
    def mutate(p):
        p["loss"] = p["rows"][0]["loss"] = post_loss
        p["rows"][0]["token_nll"] = [post_loss] * 811

    change_json(evidence["post_evaluation"], mutate)
    assert audit_module.audit(**evidence)["training_fit_outcome"] == outcome


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(status="stopped"), "incomplete"),
        (lambda p: p.update(history=[]), "one optimizer"),
        (lambda p: p["history"][0].update(loss=float("nan")), "Nonfinite"),
        (lambda p: p.update(activation_mode="checkpoint"), "bf16-mode"),
    ],
)
def test_invalid_training_receipts_fail_closed(evidence, mutation, match):
    change_json(evidence["training_report"], mutation)
    with pytest.raises(ValueError, match=match):
        audit_module.audit(**evidence)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p["rows"][0]["target_ids"].__setitem__(0, 99), "target IDs"),
        (lambda p: p["rows"][0]["token_nll"].__setitem__(0, float("inf")), "Nonfinite"),
        (lambda p: p.update(loss=0.3), "scalar loss"),
        (
            lambda p: p["evaluation_context"].update(selected_examples_sha256="b" * 64),
            "context IDs",
        ),
        (
            lambda p: p["evaluation_context"].update(input_identity={"data_sha256": "b" * 64}),
            "data hash",
        ),
        (lambda p: p.pop("evaluation_context"), "lacks evaluation_context"),
        (
            lambda p: p["evaluation_context"]["argv"].extend(["--activation-mode", "checkpoint"]),
            "exactly one",
        ),
    ],
)
def test_post_score_provenance_and_grading_tamper_rejected(evidence, mutation, match):
    change_json(evidence["post_evaluation"], mutation)
    with pytest.raises(ValueError, match=match):
        audit_module.audit(**evidence)


def repair_adapter_checksums(directory):
    change_json(
        directory / "adapter_config.json",
        lambda c: c.update(adapter_sha256=audit_module.digest(directory / "adapter.safetensors")),
    )
    change_json(
        directory / "trainer_state.json",
        lambda s: s.update(
            files={
                name: audit_module.digest(directory / name)
                for name in audit_module.CHECKPOINT_FILES
            }
        ),
    )


@pytest.mark.parametrize(
    "change,match",
    [("a", "A initialization"), ("zero_b", "Unchanged B"), ("bias", "uncorrected first Adam")],
)
def test_semantic_tensor_tamper_rejected_even_with_repaired_checksums(evidence, change, match):
    directory = evidence["training_report"].parent / "step-000001"
    arrays = load_file(str(directory / "adapter.safetensors"))
    if change == "a":
        arrays["blocks.0.attn.linears.wq_a.lora_a"][0, 0] += 0.1
    else:
        key = "blocks.0.attn.linears.wq_a.lora_b"
        arrays[key] *= 0 if change == "zero_b" else 0.316
    save_file(arrays, str(directory / "adapter.safetensors"))
    repair_adapter_checksums(directory)
    with pytest.raises(ValueError, match=match):
        audit_module.audit(**evidence)


def test_checksum_tamper_rejected(evidence):
    path = evidence["training_report"].parent / "step-000001/adapter.safetensors"
    with path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        audit_module.audit(**evidence)


def test_active_clipping_reconstructs_raw_direction(evidence):
    directory = evidence["training_report"].parent / "step-000001"
    arrays = load_file(str(directory / "adapter.safetensors"))
    optimizer = load_file(str(directory / "optimizer.safetensors"))
    b_keys = [key for key in arrays if key.endswith(".lora_b")]
    elements = sum(arrays[key].size for key in b_keys)
    raw_norm = 2.0
    raw_element = raw_norm / elements**0.5
    scale = 1.0 / (raw_norm + 1e-6)
    for key in b_keys:
        g = np.full(arrays[key].shape, raw_element * scale, np.float32)
        optimizer[key + ".m"] = np.float32(0.1) * g
        optimizer[key + ".v"] = np.float32(0.001) * g**2
        arrays[key] = (
            -np.float32(1e-5)
            * optimizer[key + ".m"]
            / (np.sqrt(optimizer[key + ".v"]) + np.float32(1e-8))
        )
    result = audit_module.audit_moments(arrays, optimizer, raw_norm)
    expected_dot = sum(
        float(np.sum(arrays[key].astype(np.float64))) * raw_element for key in b_keys
    )
    assert result["clip_scale"] == scale
    assert result["reconstructed_raw_gradient_norm"] == pytest.approx(raw_norm, rel=1e-6)
    assert result["g_dot_actual_delta"] == pytest.approx(expected_dot, rel=1e-6)
