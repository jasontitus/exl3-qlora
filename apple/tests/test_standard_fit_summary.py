"""CPU-only native SFT receipt and teacher-forced metric audit tests."""

import importlib.util
import json
from pathlib import Path

import pytest
import test_fit_recovery as recovery_tests

evidence = recovery_tests.evidence
project = recovery_tests.project

SPEC = importlib.util.spec_from_file_location(
    "standard_fit_summary", Path(__file__).parents[1] / "scripts/summarize_standard_fit.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.fixture
def recovered(evidence):
    """Use native recovery-shaped receipts and the CPU-only pinned-validator fixture."""
    helper = recovery_tests.m
    fit = helper.assemble(**evidence)
    amendment = helper.read(evidence["amendment_path"])
    project = Path(amendment["project"])
    manifest = helper.read(project / "project.json")

    def read(path, *, binary=False):
        return None if binary else helper.read(path)

    return (
        fit,
        manifest,
        project,
        Path(fit["selected_checkpoint"]),
        Path(amendment["prior_protocol"]),
        Path(amendment["runtime"]),
        read,
    )


def test_recovery_summary_preserves_both_failures_and_resource_exception(recovered):
    result = module.validate_recovery(*recovered)
    assert result["used"] is True
    assert result["base_process_memory_gib"] == 80
    assert result["adapter_process_memory_gib"] == 72
    assert result["reserve_gib"] == 8
    assert result["baseline_reused_without_rerun"] is True
    assert [value["kind"] for value in result["failed_invocations"]] == [
        "original_memory_preflight",
        "retry_path_typo_before_load",
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "resource",
        "math",
        "pid",
        "missing_failure",
        "changed_log",
        "missing_input",
        "baseline_rerun",
        "snapshot",
        "incomplete_retry",
        "wrong_amendment",
        "unrecorded",
    ],
)
def test_recovery_summary_rejects_missing_or_changed_provenance(recovered, kind):
    fit = recovered[0]
    p = fit["recovery_provenance"]
    if kind == "resource":
        fit["adapter"]["resources"]["reserve_gib"] = 0
    elif kind == "math":
        argv = fit["adapter"]["evaluation_context"]["argv"]
        argv[argv.index("--chunk-size") + 1] = "128"
    elif kind == "pid":
        fit["adapter"]["evaluation_context"]["process_id"] += 1
    elif kind == "missing_failure":
        p["failed_invocations"].pop()
    elif kind == "changed_log":
        Path(p["failed_invocations"][1]["log"]).write_text("Changed failure log")
    elif kind == "missing_input":
        del p["input_sha256"][p["original_base_report"]]
    elif kind == "baseline_rerun":
        p["baseline_rerun"] = True
    elif kind == "snapshot":
        (recovered[5] / "snapshot.json").write_text('{"changed": true}')
    elif kind in {"incomplete_retry", "wrong_amendment"}:
        path = Path(p["actual_retry_launch"] if kind == "incomplete_retry" else p["amendment"])
        value = json.loads(path.read_text())
        if kind == "incomplete_retry":
            value["exit_code"] = 1
        else:
            value["selected_checkpoint"] = str(recovered[3].with_name("step-000011"))
        path.write_text(json.dumps(value))
        p["input_sha256"][str(path)] = module.digest(path)
    else:
        fit.pop("recovery_provenance")
    with pytest.raises(ValueError):
        module.validate_recovery(*recovered)


def score(losses, predictions, digest="a"):
    return {
        "loss": sum(losses) / len(losses),
        "supervised_tokens": len(losses),
        "target_ids": [8, 9, 1],
        "predicted_ids": predictions,
        "token_nll": losses,
        "logits_sha256": digest * 64,
    }


@pytest.fixture
def pair():
    base = score([0.8, 0.2, 0.2], [7, 9, 1])
    trained = score([0.1, 0.4, 0.1], [8, 7, 1], "b")
    row = {"input_ids": [2, 3, 8, 9, 1], "loss_start": 2}
    labels = {
        "supervised_token_ids": [8, 9, 1],
        "serialized_tokens": 5,
        "loss_start": 2,
        "supervised_tokens": 3,
        "supervised_token_categories": ["added_code", "added_code", "eos"],
    }
    return base, trained, labels, row


def test_native_pair_keeps_corrected_and_regressed_code_tokens(pair):
    result = module.summarize_pair(*pair, decoder=lambda ids: str(ids))
    assert result["categories"]["all"]["loss_delta"] < 0
    assert result["categories"]["added_code"]["base"]["correct_tokens"] == 1
    assert result["categories"]["added_code"]["trained"]["correct_tokens"] == 1
    assert result["added_code_predictions_corrected"][0]["answer_token_index"] == 0
    assert result["added_code_predictions_regressed"][0]["answer_token_index"] == 1
    assert result["added_code_predictions_corrected"][0]["target_context"] == "[8, 9, 1]"


@pytest.mark.parametrize(
    "kind", ["mask", "targets", "classes", "nan", "aggregate", "digest", "negative"]
)
def test_pair_rejects_partial_or_forged_scores(pair, kind):
    base, trained, labels, row = pair
    if kind == "mask":
        row["loss_start"] = 1
    elif kind == "targets":
        trained["target_ids"][0] = 7
    elif kind == "classes":
        labels["supervised_token_categories"].pop()
    elif kind == "nan":
        base["token_nll"][0] = float("nan")
    elif kind == "aggregate":
        base["loss"] += 0.1
    elif kind == "digest":
        base["logits_sha256"] = "not-a-digest"
    else:
        base["token_nll"] = [-1, 1, 1]
        base["loss"] = 1 / 3
    with pytest.raises(ValueError):
        module.summarize_pair(base, trained, labels, row)


@pytest.fixture
def native(tmp_path, pair):
    project = tmp_path / "project"
    checkpoint = project / "attempts/1/train/step-000012"
    base, trained, labels, row = pair
    manifest = {
        "model": str(tmp_path / "model"),
        "data_files": {"train.jsonl": "c" * 64},
        "data": {"seq_length": 4},
        "implementation": {"source": {"training.py": "d" * 64}},
        "config": {"comparison_examples": 1, "chunk_size": 8192, "activation_mode": "checkpoint"},
    }
    fit = {
        "format": "exl3-sft-training-fit-v1",
        "evaluation_scope": "training_only",
        "quality_certified": False,
        "training_data_sha256": "c" * 64,
        "selected_examples_sha256": module.canonical_sha([row]),
        "selected_checkpoint": str(checkpoint),
        "available_examples": 1,
        "evaluated_examples": 1,
        "supervised_tokens": 3,
        "loss_delta": trained["loss"] - base["loss"],
    }
    for i, (name, values) in enumerate((("base", base), ("adapter", trained))):
        fit[name] = {
            "loss": values["loss"],
            "supervised_tokens": 3,
            "examples": 1,
            "rows": [values],
            "checkpoint": manifest["model"],
            "adapter": str(checkpoint) if name == "adapter" else None,
            "logits_scope": "supervised_rows",
            "logits_sha256": ("a" if i == 0 else "b") * 64,
            "resources": {"prefix_cache_mib": 0},
            "evaluation_context": {
                "process_id": 100 + i,
                "data": str(project / "data/train.jsonl"),
                "input_identity": {"data_sha256": "c" * 64},
                "selected_examples_sha256": module.canonical_sha([row]),
                "seq_length": 4,
                "requested_max_examples": 1,
                "implementation": manifest["implementation"],
                "argv": [
                    "evaluate",
                    "--token-metrics",
                    "--data",
                    str(project / "data/train.jsonl"),
                    "--seq-length",
                    "4",
                    "--model",
                    manifest["model"],
                    "--chunk-size",
                    "8192",
                ],
            },
        }
    return fit, manifest, [row], project, checkpoint


def test_native_receipt_passes_without_diagnostic_history(native):
    module.validate_native_fit(*native)
    assert "history" not in native[0]


@pytest.mark.parametrize(
    "kind",
    [
        "pid",
        "source",
        "selection",
        "data",
        "rows",
        "adapter",
        "mode",
        "cache",
        "argv",
        "delta",
        "aggregate",
    ],
)
def test_native_pairing_rejects_mismatches(native, kind):
    fit, manifest, rows, project, checkpoint = native
    value = fit["adapter"]
    if kind == "pid":
        value["evaluation_context"]["process_id"] = fit["base"]["evaluation_context"]["process_id"]
    elif kind == "source":
        value["evaluation_context"]["implementation"] = {}
    elif kind == "selection":
        value["evaluation_context"]["selected_examples_sha256"] = "e" * 64
    elif kind == "data":
        value["evaluation_context"]["input_identity"] = {}
    elif kind == "rows":
        value["rows"] = []
    elif kind == "adapter":
        value["adapter"] = None
    elif kind == "mode":
        value["activation_mode"] = "bf16"
    elif kind == "cache":
        value["resources"]["prefix_cache_mib"] = 2048
    elif kind == "argv":
        value["evaluation_context"]["argv"][-1] = "512"
    elif kind == "delta":
        fit["loss_delta"] += 0.1
    else:
        value["loss"] += 0.1
    with pytest.raises(ValueError):
        module.validate_native_fit(fit, manifest, rows, project, checkpoint)


def test_predeclared_epoch_sampler_is_four_exposures_each():
    order = module.expected_samples(12, 3, 8)
    assert order == [2, 1, 0, 0, 2, 1, 1, 0, 2, 0, 2, 1]
    assert [order.count(i) for i in range(3)] == [4, 4, 4]


def test_manifest_paths_cannot_escape_project(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    secret = tmp_path / "outside"
    secret.write_text("x")
    with pytest.raises(ValueError, match="escapes"):
        module.checked_path(project, "../outside")
    (project / "symlink").symlink_to(secret)
    with pytest.raises(ValueError, match="escapes"):
        module.checked_path(project, "symlink")


def test_summary_reuses_existing_score_validator():
    assert module.ATOL == 1e-6
    assert module._HELPER.validate_score.__module__ == "_fit_score_validation"


def test_standard_summary_has_no_gpu_imports():
    import ast

    tree = ast.parse(Path(module.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert all(not name.startswith(("mlx", "exl3_mlx")) for name in imports)


def test_partial_training_rejected_before_reading_tensors(tmp_path):
    manifest = {"status": "training", "config": {"max_steps": 12}}
    with pytest.raises(ValueError, match="complete"):
        module.audit_training(
            tmp_path, manifest, {}, {"settings": {"steps_total": 12}}, lambda p: {}
        )


@pytest.fixture
def completed_training(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    root = tmp_path / "project"
    output = root / "attempts/1/train"
    output.mkdir(parents=True)
    config = {
        "max_steps": 12,
        "seed": 8,
        "rank": 4,
        "lora_alpha": 8,
        "activation_mode": "checkpoint",
        "chunk_size": 8192,
        "learning_rate": 1e-4,
        "max_grad_norm": 1,
    }
    metadata = {
        name: "d" * 64
        for name in (
            "config.json",
            "expert-manifest-v2.json",
            "engram/engram-manifest.json",
            "engram-token-map.u32.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        )
    }
    manifest = {
        "status": "completed",
        "config": config,
        "data": {"counts": {"train": 3, "valid": 0, "test": 0}, "seq_length": 5821},
        "data_files": {"train.jsonl": "e" * 64},
        "capabilities": {"base_metadata": metadata},
        "implementation": {"runtime": {"mlx": "test", "mlx-lm": "test"}},
        "last_training_report": str(output / "training.json"),
    }
    contract = {
        "data_sha256": "e" * 64,
        "learning_rate": 1e-4,
        "grad_accumulation": 1,
        "clip_grad": 1,
        "seed": 8,
        "seq_length": 5821,
        "truncate": False,
        "chunk_size": 8192,
        "no_checkpoint": False,
        "mlx": "test",
        "mlx_lm": "test",
        "engine_semantics": "test",
    }
    order = module.expected_samples(12, 3, 8)
    protocol = {
        "settings": {"steps_total": 12},
        "supervised_tokens": [204, 811, 1877],
        "sampler": {"expected_row_indices": order},
    }
    shapes = {
        "wq_a.lora_a": (5120, 4),
        "wq_a.lora_b": (4, 1280),
        "wo_b.lora_a": (8192, 4),
        "wo_b.lora_b": (4, 5120),
    }
    arrays = {
        "blocks.39.attn.linears." + name: np.ones(shape, dtype=np.float32)
        for name, shape in shapes.items()
    }
    history = []
    for step in range(1, 13):
        path = output / f"step-{step:06d}"
        path.mkdir()
        save_file(arrays, str(path / "adapter.safetensors"))
        optimizer = {
            f"{name}.{part}": array for name, array in arrays.items() for part in ("m", "v")
        }
        optimizer.update(
            step=np.array(step, dtype=np.uint64), learning_rate=np.array(1e-4, dtype=np.float32)
        )
        save_file(optimizer, str(path / "optimizer.safetensors"))
        save_file({"state": np.array([1], dtype=np.uint64)}, str(path / "random.safetensors"))
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "adapter_layers": [39],
                    "expert_adapter_layers": [],
                    "rank": 4,
                    "alpha": 8,
                    "activation_mode": "checkpoint",
                    "engine_semantics": "test",
                    "base_identity": metadata,
                }
            )
        )
        record = {
            "step": step,
            "cursor": step,
            "microbatches": 1,
            "supervised_tokens": protocol["supervised_tokens"][order[step - 1]],
        }
        history.append(record)
        saved = {
            "format": "exl3-mlx-training-state-v1",
            "step": step,
            "cursor": step,
            "contract": contract,
            "metrics": record,
            "files": {p.name: module.digest(p) for p in path.iterdir()},
        }
        (path / "trainer_state.json").write_text(json.dumps(saved))
    (output / "training.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "completed_step": 12,
                "checkpoint": str(path),
                "additional_steps": 12,
                "history": history,
            }
        )
    )
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "steps": 12,
                "seed": 8,
                "resume": None,
                "resume_adapter": None,
                "adapter_layers": [39],
                "activation_mode": "checkpoint",
            }
        )
    )

    def read(path, *, binary=False):
        return None if binary else json.loads(Path(path).read_text())

    return root, manifest, {"selected_checkpoint": str(path)}, protocol, read


def test_all_twelve_real_format_checkpoints_and_sampler_audited(completed_training):
    _, audit = module.audit_training(*completed_training)
    assert audit["optimizer_updates"] == 12 and audit["sample_cursor"] == 12
    assert audit["exposures_per_row"] == {0: 4, 1: 4, 2: 4}
    assert len(audit["checkpoints"]) == 12
    assert all(
        x["adapter_parameters"] == 78848 and x["optimizer_tensors"] == 10
        for x in audit["checkpoints"]
    )


@pytest.mark.parametrize(
    "kind", ["checksum", "cursor", "row_count", "missing_checkpoint", "continued"]
)
def test_native_checkpoint_audit_rejects_partial_and_mismatched_evidence(completed_training, kind):
    root, manifest, fit, protocol, read = completed_training
    step = root / "attempts/1/train/step-000003"
    if kind == "checksum":
        (step / "adapter.safetensors").write_bytes(b"broken")
    elif kind == "missing_checkpoint":
        (step / "trainer_state.json").unlink()
    elif kind == "continued":
        path = step.parent / "run_config.json"
        value = read(path)
        value["resume_adapter"] = "foreign"
        path.write_text(json.dumps(value))
    else:
        path = step / "trainer_state.json"
        value = read(path)
        if kind == "cursor":
            value["cursor"] = 2
        else:
            value["metrics"]["supervised_tokens"] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.audit_training(root, manifest, fit, protocol, read)


@pytest.fixture
def native_project(completed_training):
    """Real-format project, safetensors checkpoints and separate bound child JSON."""
    import shutil

    root, manifest, fit, protocol, read = completed_training
    runtime, model = root.parent / "runtime", root.parent / "model"
    (runtime / "src/exl3_mlx").mkdir(parents=True)
    source = Path(__file__).parents[1] / "src/exl3_mlx/fit_probe.py"
    shutil.copyfile(source, runtime / "src/exl3_mlx/fit_probe.py")
    (runtime / "uv.lock").write_text("CPU fixture lock")
    model.mkdir()
    (model / "tokenizer.json").write_text("CPU fixture; decoding deliberately disabled")
    tokenizer_sha = module.digest(model / "tokenizer.json")
    metadata = manifest["capabilities"]["base_metadata"]
    metadata["tokenizer.json"] = tokenizer_sha
    for path in root.glob("attempts/*/train/step-*/adapter_config.json"):
        value = read(path)
        value["base_identity"] = metadata
        path.write_text(json.dumps(value))
        state_path = path.with_name("trainer_state.json")
        state = read(state_path)
        state["files"][path.name] = module.digest(path)
        state_path.write_text(json.dumps(state))
    manifest["model"] = str(model)
    manifest["format"] = "exl3-sft-project-v1"
    manifest["config"].update(comparison_examples=3, final_fit_examples=3)
    manifest["implementation"].update(
        source={"fit_probe.py": module.digest(source)},
        dependency_lock=module.digest(runtime / "uv.lock"),
    )
    data_dir = root / "data"
    data_dir.mkdir()
    rows = [
        {"input_ids": [2, 3] + [8] * (count - 1) + [1], "loss_start": 2}
        for count in protocol["supervised_tokens"]
    ]
    data = data_dir / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    data_sha = module.digest(data)
    manifest["data_files"]["train.jsonl"] = data_sha
    protocol["expected_prepared_sha256"] = data_sha
    protocol["case_ids"] = ["fixture-change-0", "fixture-change-1", "fixture-change-2"]
    raw_path = root.parent / "raw.jsonl"
    raw_path.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": case_id,
                    "source_id": "fixture-lineage",
                    "prompt": "p",
                    "completion": str(i),
                }
            )
            + "\n"
            for i, case_id in enumerate(protocol["case_ids"])
        )
    )
    protocol["raw_sha256"] = module.digest(raw_path)
    manifest["data"]["source_files_sha256"] = {str(raw_path): protocol["raw_sha256"]}
    (data_dir / "dataset.json").write_text(
        json.dumps({"source_files_sha256": manifest["data"]["source_files_sha256"]})
    )
    (data_dir / "provenance.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "retained": True,
                    "split": "train",
                    "duplicate_of": None,
                    "source_row": i + 1,
                    "source_file": str(raw_path),
                    "tokens_sha256": module.canonical_sha(row),
                    "metadata": {
                        "case_id": protocol["case_ids"][i],
                        "source_id": "fixture-lineage",
                    },
                }
            )
            + "\n"
            for i, row in enumerate(rows)
        )
    )
    manifest["data_files"].update(
        {name: module.digest(data_dir / name) for name in ("dataset.json", "provenance.jsonl")}
    )
    for path in root.glob("attempts/*/train/step-*/trainer_state.json"):
        state = read(path)
        state["contract"]["data_sha256"] = data_sha
        path.write_text(json.dumps(state))
    (root / "project.json").write_text(json.dumps(manifest))
    protocol_path = root.parent / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    categories = []
    for i, row in enumerate(rows):
        count = len(row["input_ids"]) - row["loss_start"]
        path = root.parent / f"categories-{i}.json"
        path.write_text(
            json.dumps(
                {
                    "row_index": i,
                    "source_sha256": {"prepared": data_sha, "tokenizer": tokenizer_sha},
                    "serialized_tokens": len(row["input_ids"]),
                    "loss_start": 2,
                    "supervised_tokens": count,
                    "supervised_token_ids": row["input_ids"][2:],
                    "supervised_token_categories": ["added_code"]
                    + ["unchanged_context"] * (count - 2)
                    + ["eos"],
                }
            )
        )
        categories.append(path)
    checkpoint = Path(fit["selected_checkpoint"])
    fit_dir = root / "fit-checks/1"
    fit_dir.mkdir(parents=True)
    fit.update(
        format="exl3-sft-training-fit-v1",
        evaluation_scope="training_only",
        quality_certified=False,
        training_data_sha256=data_sha,
        selected_examples_sha256=module.canonical_sha(rows),
        available_examples=3,
        evaluated_examples=3,
        supervised_tokens=sum(protocol["supervised_tokens"]),
    )
    for condition_index, name in enumerate(("base", "adapter")):
        scores = []
        for i, row in enumerate(rows):
            targets = row["input_ids"][2:]
            loss = 1.0 if name == "base" else [2.0, 0.9, 0.7][i]
            scores.append(
                {
                    "target_ids": targets,
                    "predicted_ids": targets,
                    "token_nll": [loss] * len(targets),
                    "loss": loss,
                    "supervised_tokens": len(targets),
                    "logits_sha256": str(i + condition_index) * 64,
                }
            )
        total = sum(s["supervised_tokens"] for s in scores)
        argv = [
            "evaluate",
            "--token-metrics",
            "--data",
            str(data),
            "--seq-length",
            "5821",
            "--model",
            str(model),
            "--chunk-size",
            "8192",
            "--report",
            str(fit_dir / f"{name}.json"),
        ]
        fit[name] = {
            "examples": 3,
            "supervised_tokens": total,
            "rows": scores,
            "loss": sum(s["loss"] * s["supervised_tokens"] for s in scores) / total,
            "logits_scope": "supervised_rows",
            "logits_sha256": ("a" if name == "base" else "b") * 64,
            "activation_mode": "checkpoint",
            "checkpoint": str(model),
            "adapter": str(checkpoint) if name == "adapter" else None,
            "resources": {"prefix_cache_mib": 0},
            "prefix_cache": {"enabled": False, "limit_bytes": 0},
            "evaluation_context": {
                "data": str(data),
                "input_identity": {"data_sha256": data_sha},
                "selected_examples_sha256": module.canonical_sha(rows),
                "seq_length": 5821,
                "requested_max_examples": 3,
                "implementation": manifest["implementation"],
                "process_id": 100 + condition_index,
                "argv": argv,
            },
        }
        (fit_dir / f"{name}.json").write_text(json.dumps(fit[name]))
    fit["loss_delta"] = fit["adapter"]["loss"] - fit["base"]["loss"]
    expected = {
        "data": str(data),
        "input_identity": {"data_sha256": data_sha},
        "implementation": manifest["implementation"],
        "seq_length": 5821,
        "activation_mode": "checkpoint",
        "chunk_size": 8192,
        "available_examples": 3,
        "requested_examples": 3,
    }
    probe = {
        **expected,
        "format": "exl3-final-training-fit-probe-v1",
        "step": 12,
        "checkpoint": str(checkpoint),
        "training_mode": True,
        "selected_examples_sha256": module.canonical_sha(rows),
        "scores": fit["adapter"],
        "prefix_cache_before": {"enabled": True, "hits": 9, "misses": 3},
        "prefix_cache_after": {"enabled": True, "hits": 12, "misses": 3},
        "checkpoint_files_sha256": {
            name: module.digest(checkpoint / name)
            for name in ("adapter.safetensors", "adapter_config.json", "trainer_state.json")
        },
    }
    spec = importlib.util.spec_from_file_location("fixture_fit_probe", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    parity = helper.compare_final_fit_probe(
        probe, fit["adapter"], rows, checkpoint=checkpoint, expected=expected
    )
    path = Path(manifest["last_training_report"])
    report = read(path)
    report["training_fit_probe"] = probe
    path.write_text(json.dumps(report))
    path.with_name("training-fit-probe.json").write_text(json.dumps(probe))
    parity["training_report_sha256"] = module.digest(path)
    fit["training_mode_probe_parity"] = parity
    (root / "fit.json").write_text(json.dumps(fit))
    return root, protocol_path, categories, runtime


def test_full_native_project_audit_discloses_mixed_fit(native_project):
    report = module.build_report(*native_project)
    assert report["schema"] == "standard-sft-three-change-fit-audit-v1"
    assert report["status"] == "completed" and report["aggregate_fit_improved"]
    assert not report["every_row_improved"] and report["macro"]["loss_delta"] > 0
    assert report["rows"]["0"]["categories"]["all"]["loss_delta"] > 0
    assert report["tool_verification_complete"]
    assert not report["quality_certified"] and report["held_out_rows"] == 0
    assert report["auditor"]["version"] == module.AUDITOR_VERSION
    assert report["auditor"]["sha256"] == module.digest(module.__file__)
    assert len(report["case_binding"]["rows"]) == 3
    assert "this three-case" in report["tool_verification_scope"].lower()


@pytest.mark.parametrize("kind", ["raw_receipt", "runtime", "labels", "probe"])
def test_full_native_project_rejects_broken_external_receipts(native_project, kind):
    root, protocol, categories, runtime = native_project
    if kind == "raw_receipt":
        path = root / "fit-checks/1/adapter.json"
        value = json.loads(path.read_text())
        value["loss"] += 1
        path.write_text(json.dumps(value))
    elif kind == "runtime":
        (runtime / "src/exl3_mlx/fit_probe.py").write_text("changed")
    elif kind == "labels":
        path = categories[0]
        value = json.loads(path.read_text())
        value["supervised_token_ids"][0] += 1
        path.write_text(json.dumps(value))
    else:
        path = next(root.glob("attempts/*/train/training-fit-probe.json"))
        value = json.loads(path.read_text())
        value["scores"]["loss"] += 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module.build_report(root, protocol, categories, runtime)


@pytest.mark.parametrize(
    "change", ["swapped_ids", "foreign_ids", "raw_sha", "raw_bytes", "retained", "tokens"]
)
def test_case_mapping_and_raw_source_are_bound(native_project, change):
    root, protocol_path, categories, runtime = native_project
    protocol = json.loads(protocol_path.read_text())
    if change == "swapped_ids":
        protocol["case_ids"][0], protocol["case_ids"][1] = (
            protocol["case_ids"][1],
            protocol["case_ids"][0],
        )
    elif change == "foreign_ids":
        protocol["case_ids"] = ["wrong-one", "wrong-two", "wrong-three"]
    elif change == "raw_sha":
        protocol["raw_sha256"] = "f" * 64
    elif change == "raw_bytes":
        with (root.parent / "raw.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        path = root / "data/provenance.jsonl"
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "retained":
            entries[0]["retained"] = False
        else:
            entries[0]["tokens_sha256"] = "e" * 64
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        # Even an internally rechecksummed project cannot relabel the original targets.
        manifest_path = root / "project.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["data_files"][path.name] = module.digest(path)
        manifest_path.write_text(json.dumps(manifest))
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        module.build_report(root, protocol_path, categories, runtime)
