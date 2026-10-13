"""Synthetic CPU receipt recovery through the real SFT validators, no children."""

import copy
import dataclasses
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import test_sft_project as fixtures

spec = importlib.util.spec_from_file_location(
    "fit_recovery", Path(__file__).parents[1] / "scripts/recover_standard_fit.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
project = fixtures.project


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def evidence(project, tmp_path, monkeypatch):
    sft = fixtures.sft
    project.args = dataclasses.replace(
        project.args,
        max_steps=12,
        final_fit_examples=1,
        comparison_examples=1,
        profile="development",
        memory_gib=80,
        reserve_gib=8,
        packed_cache_mib=8192,
        decoded_cache_mib=0,
        prefix_cache_mib=2048,
        allocator_cache_mib=256,
    )
    manifest = m.read(project.root / "project.json")
    manifest["config"] = dataclasses.asdict(project.args)
    write(project.root / "project.json", manifest)
    (project.root / ".run.lock").touch()
    adapter = fixtures._checkpoint(project, 12)
    rows, selected, expected = project._evaluation_rows(manifest, "train")
    base_path = project.root / "fit-checks/original/base.json"
    retry_dir = project.root / "fit-checks/recovery"
    retry_path = retry_dir / "adapter.json"
    base_argv = m.base_command(project, manifest, base_path)
    retry_argv = list(base_argv)
    retry_argv[retry_argv.index("--report") + 1] = str(retry_path)
    retry_argv[retry_argv.index("--memory-gib") + 1] = "72"
    retry_argv += ["--adapter", str(adapter)]
    resources = {k: getattr(project.args, k) for k in sft.RESOURCE_FIELDS}
    resources["prefix_cache_mib"] = 0

    def score(command, loss, pid, memory):
        targets = selected[0]["input_ids"][selected[0]["loss_start"] :]
        row = dict(
            target_ids=targets,
            predicted_ids=targets,
            token_nll=[loss] * len(targets),
            loss=loss,
            supervised_tokens=len(targets),
            correct_tokens=len(targets),
            logits_sha256="a" * 64,
        )
        value = fixtures._evaluation_result(
            project,
            command,
            loss=loss,
            rows=[row],
            logits_sha256="a" * 64,
            logits_scope="supervised_rows",
            expert_manifest_sha256="fixture",
            resources={**resources, "memory_gib": memory},
            prefix_cache={"enabled": False, "limit_bytes": 0},
        )
        value["evaluation_context"]["process_id"] = pid
        return value

    base, fresh = score(base_argv, 1.0, 123, 80), score(retry_argv, 0.5, 124, 72)
    write(base_path, base)
    write(retry_path, fresh)
    before = {"enabled": True, "hits": 0, "misses": 1}
    probe = {
        "format": "exl3-final-training-fit-probe-v1",
        "checkpoint": str(adapter),
        "training_mode": True,
        "requested_examples": 1,
        "selected_examples_sha256": expected["selected_examples_sha256"],
        **{k: expected[k] for k in ("data", "input_identity", "implementation", "seq_length")},
        "activation_mode": "checkpoint",
        "chunk_size": project.args.chunk_size,
        "step": 12,
        "checkpoint_files_sha256": {
            n: m.digest(adapter / n)
            for n in ("adapter.safetensors", "adapter_config.json", "trainer_state.json")
        },
        "prefix_cache_before": before,
        "prefix_cache_after": {**before, "hits": 1},
        "scores": copy.deepcopy(fresh),
    }
    training_path = adapter.parent / "training.json"
    write(training_path, {"status": "completed", "training_fit_probe": probe})
    runtime = tmp_path / "runtime"
    write(runtime / "snapshot.json", {})
    protocol = tmp_path / "protocol.json"
    write(protocol, {"fixture": True})
    runtime_launch = tmp_path / "runtime-launch.json"
    write(
        runtime_launch,
        {
            "runtime": str(runtime),
            "snapshot_sha256": m.digest(runtime / "snapshot.json"),
            "protocol_sha256": m.digest(protocol),
        },
    )
    failed = base_path.parent / "fit-adapter.log"
    failed.write_text("Memory preflight failed\n")
    amendment = retry_dir / "recovery-launch.json"
    write(
        amendment,
        {
            "schema": "standard-fit-resource-recovery-v1",
            "status": "predeclared_before_adapter_retry",
            "project": str(project.root),
            "runtime": str(runtime),
            "base_report": str(base_path),
            "base_report_sha256": m.digest(base_path),
            "failed_adapter_log": str(failed),
            "failed_adapter_log_sha256": m.digest(failed),
            "prior_protocol": str(protocol),
            "prior_protocol_sha256": m.digest(protocol),
            "selected_checkpoint": str(adapter),
            "resource_only_changes": {
                "base_memory_gib": 80,
                "adapter_memory_gib": 72,
                "reserve_gib": 8,
            },
            "reason": "fixture memory preflight",
            "argv": [sys.executable, "-u", "-m", "exl3_mlx.workflow", *retry_argv],
            "environment": {"PYTHONPATH": str(runtime / "src")},
        },
    )
    typo = retry_dir / "typo.log"
    typo.write_text("Bad path before model loading\n")
    write(
        retry_dir / "typo-failure.json",
        {"exit_code": 1, "log": str(typo), "sha256": m.digest(typo)},
    )
    a = m.read(amendment)
    (retry_dir / "fit-adapter.log").write_text("Completed fake evaluation\n")
    write(
        retry_dir / "actual-launch.json",
        {
            "schema": "actual-recovery-launch-v1",
            "prelaunch_receipt": str(amendment),
            "prelaunch_sha256": m.digest(amendment),
            "argv": a["argv"],
            "environment": a["environment"],
            "pid": 124,
            "exit_code": 0,
            "finished_utc": "fixture",
            "log_sha256": m.digest(retry_dir / "fit-adapter.log"),
        },
    )
    audit = tmp_path / "reports/training-audit.json"
    write(
        audit,
        {
            "training_completion_verified": True,
            "checkpoints_verified": 12,
            "input_sha256": {
                str(x.relative_to(tmp_path)): m.digest(x)
                for x in (training_path, project.root / "project.json")
            },
            "steps": [
                {
                    "step": 12,
                    "adapter_sha256": m.digest(adapter / "adapter.safetensors"),
                    "optimizer_sha256": m.digest(adapter / "optimizer.safetensors"),
                    "trainer_state_sha256": m.digest(adapter / "trainer_state.json"),
                }
            ],
        },
    )
    monkeypatch.setattr(m, "load_pinned_sft", lambda *_: sft)
    monkeypatch.setattr(
        sft.SFTTrainer, "from_project", classmethod(lambda *_args, **_kwargs: project)
    )
    monkeypatch.setattr(project, "_child", lambda *_a, **_k: pytest.fail("No GPU child may run"))
    return dict(
        amendment_path=amendment, runtime_launch_path=runtime_launch, training_audit_path=audit
    )


def test_full_recovery_preserves_raw_receipts_and_writes_while_locked(evidence, monkeypatch):
    amendment = m.read(evidence["amendment_path"])
    project = Path(amendment["project"])
    output = project / "fit.json"
    original = m.publish_new

    def locked_publish(path, result):
        with (project / ".run.lock").open() as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original(path, result)

    monkeypatch.setattr(m, "publish_new", locked_publish)
    result = m.assemble(**evidence, output_path=output)
    assert result["loss_delta"] == -0.5 and result["quality_certified"] is False
    assert result["training_mode_probe_parity"]["real_model_warm_cache_parity_verified"] is True
    assert result["base"] == m.read(amendment["base_report"])
    assert result["adapter"] == m.read(evidence["amendment_path"].parent / "adapter.json")
    assert m.read(output) == result


@pytest.mark.parametrize(
    "case,match",
    [
        ("base_tamper", "Changed recovery input"),
        ("pending", "completed successfully"),
        ("pid", "distinct processes"),
        ("source", "provenance differs"),
        ("tokens", "target IDs"),
        ("logits", "logits mismatch"),
        ("resources", "resolved resources"),
        ("correct", "correct-token count"),
        ("no_warm_hits", "full warm-probe parity"),
    ],
)
def test_invalid_recovery_does_not_publish(evidence, case, match):
    amendment = m.read(evidence["amendment_path"])
    directory = evidence["amendment_path"].parent
    fresh = m.read(directory / "adapter.json")
    if case == "base_tamper":
        Path(amendment["base_report"]).write_text("{}\n")
    elif case == "pending":
        a = m.read(directory / "actual-launch.json")
        a.pop("exit_code")
        write(directory / "actual-launch.json", a)
    elif case == "pid":
        fresh["evaluation_context"]["process_id"] = 123
        a = m.read(directory / "actual-launch.json")
        a["pid"] = 123
        write(directory / "actual-launch.json", a)
    elif case == "source":
        fresh["evaluation_context"]["implementation"] = {}
    elif case == "tokens":
        fresh["rows"][0]["target_ids"][0] += 1
    elif case == "logits":
        fresh["rows"][0]["logits_sha256"] = "b" * 64
    elif case == "resources":
        fresh["resources"]["prefix_cache_mib"] = 2048
    elif case == "correct":
        fresh["rows"][0]["correct_tokens"] = 0
    elif case == "no_warm_hits":
        training_path = Path(amendment["selected_checkpoint"]).parent / "training.json"
        training = m.read(training_path)
        training["training_fit_probe"]["prefix_cache_after"]["hits"] = 0
        write(training_path, training)
        audit = m.read(evidence["training_audit_path"])
        audit["input_sha256"][
            str(training_path.relative_to(evidence["training_audit_path"].parent.parent))
        ] = m.digest(training_path)
        write(evidence["training_audit_path"], audit)
    write(directory / "adapter.json", fresh)
    output = Path(amendment["project"]) / "fit.json"
    with pytest.raises(ValueError, match=match):
        m.assemble(**evidence, output_path=output)
    assert not output.exists()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--reserve-gib", "0"),
        ("--chunk-size", "128"),
        ("--prefix-cache-mib", "2048"),
        ("--activation-mode", "bf16"),
    ],
)
def test_retry_cannot_change_math_or_other_resources(evidence, flag, value):
    a = m.read(evidence["amendment_path"])
    retry = a["argv"][4:]
    base = list(retry[:-2])
    base[base.index("--memory-gib") + 1] = "80.0"
    a["argv"][a["argv"].index(flag) + 1] = value
    with pytest.raises(ValueError, match="Retry changed"):
        m.validate_retry_command(base, a, a["selected_checkpoint"])


def test_atomic_publication_cannot_replace_existing_file(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text("sentinel")
    with pytest.raises(FileExistsError):
        m.publish_new(path, {"new": True})
    assert path.read_text() == "sentinel" and list(tmp_path.iterdir()) == [path]
