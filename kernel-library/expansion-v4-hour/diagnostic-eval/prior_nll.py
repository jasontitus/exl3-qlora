"""CPU-seal then explicitly run one prior-adapter held-out NLL comparison.

Consumes a sealed diagnostic campaign for runtime/full-base/adapter bindings and
an already completed M5 five-phase training launch. No remote execution, retry,
resumption, model selection or generation is performed implicitly.
"""

import argparse
import hashlib
import importlib.util
import json
import signal
from pathlib import Path
from types import SimpleNamespace


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launcher(request):
    # Check bytes before importing executable source; local pinned helper only.
    import hashlib

    path = Path(request["launcher"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != request["launcher_sha256"]:
        raise ValueError("Diagnostic launcher changed")
    return load(path, "prior_nll_diagnostic_bindings")


def request_read(path):
    # Bootstrap strict JSON without executing any request-provided command.
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key")
            result[k] = v
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


def check_completed_launch(d, launch, plan, plan_path, runner, bindings):
    d.require(
        launch.get("status") == "completed_pending_quality_audit",
        "Training launch not complete",
    )
    d.require(
        launch.get("plan") == str(plan_path)
        and launch.get("plan_sha256") == d.digest(plan_path),
        "Wrong completed training plan",
    )
    d.require(
        launch.get("numerical_environment") == d.ENV,
        "Training numerical environment differs",
    )
    for p, checksum in launch["identity_sha256"].items():
        d.bind_file(p, checksum, bindings)
    for phase in runner.PHASES:
        record = launch["processes"][phase]
        d.require(
            record.get("status") == "completed"
            and record.get("exit_code") == 0
            and {
                k: v
                for k, v in record.get("environment", {}).items()
                if k.startswith("MLX_")
            }
            == d.ENV,
            "Incomplete or wrong-mode phase",
        )
        command = record.get("argv")
        expected = plan["commands"][phase]
        d.require(
            command == expected
            if phase != "adapter"
            else command[:-1] == expected[:-1],
            "Wrong phase command",
        )
        for key, checksum in record["log_sha256"].items():
            d.bind_file(record[key], checksum, bindings)
        verified = launch["verified_phases"][phase]
        runner.verify_completion(
            verified, phase, plan, d.digest(plan_path), record["process_id"]
        )
        for p, checksum in verified["artifacts_sha256"].items():
            d.bind_file(p, checksum, bindings)


def build_command(base_command, adapter, output, resources):
    command = list(base_command)
    for key, value in resources.items():
        flag = "--" + key.replace("_", "-")
        # Runtime's alias for packed-cache is normalized to emitted official flag.
        if flag in command:
            i = command.index(flag)
            del command[i : i + 2]
        if value is not None:
            command.extend([flag, str(value)])
    i = command.index("--report")
    command[i + 1] = str(output)
    command.extend(["--adapter", str(adapter)])
    return command


def context(d, request, diagnostic, bindings):
    plan_path = d.bind_file(
        request["training_plan"], request["training_plan_sha256"], bindings
    )
    launch_path = d.bind_file(
        request["training_launch"], request["training_launch_sha256"], bindings
    )
    training = d.read(plan_path)
    r = diagnostic["request"]
    d.require(
        training["runtime"] == r["runtime"]
        and training["snapshot_sha256"] == r["snapshot_sha256"]
        and training["python"] == r["python"]
        and training["helper"] == r["helper"]
        and training["helper_sha256"] == r["helper_sha256"],
        "Training/diagnostic implementation differs",
    )
    helper = d.load_module(r["helper"], "prior_nll_frozen_helper")
    plan, trainer, manifest, sft = helper.load_plan(plan_path)
    runner = d.load_module(r["runner"], "prior_nll_frozen_runner")
    completed = d.read(launch_path)
    check_completed_launch(d, completed, plan, plan_path, runner, bindings)
    d.require(
        completed["payload_verification"]
        == {"path": r["payload_receipt"], "sha256": r["payload_sha256"]},
        "Training base payload differs",
    )
    with sft.project_lock(trainer.root):
        selected = helper._emit_locked(
            SimpleNamespace(plan=str(plan_path), phase="adapter"),
            plan,
            trainer,
            manifest,
            sft,
        )["argv"][-1]
        final = next(c for c in r["conditions"] if c["name"] == "final")
        d.require(
            selected == final["adapter"],
            "Diagnostic final is not the fixed completed training checkpoint",
        )
        _, baseline = helper.verify_evaluation(plan, trainer, manifest, "baseline")
        helper.verify_evaluation(
            plan,
            trainer,
            manifest,
            "adapter",
            command=completed["processes"]["adapter"]["argv"],
            adapter=Path(selected),
        )
        d.require(
            plan["counts"]["test"] == 16, "Require exact fixed sixteen held-out rows"
        )
        data = Path(
            plan["commands"]["baseline"][
                plan["commands"]["baseline"].index("--data") + 1
            ]
        )
        d.bind_file(data, manifest["data_files"]["test.jsonl"], bindings)
        rows = [json.loads(v) for v in data.read_text().splitlines() if v.strip()]
        d.require(
            len(rows) == 16
            and plan["commands"]["baseline"][
                plan["commands"]["baseline"].index("--max-examples") + 1
            ]
            == "16",
            "Baseline must cover all sixteen rows",
        )
        d.require(
            plan["config"]["chunk_size"] == r["inference"]["chunk_size"]
            and plan["config"]["activation_mode"] == r["inference"]["activation_mode"],
            "Mathematical settings differ",
        )
        d.require(str(trainer.model) == r["model"], "Wrong model root")
        d.bind_file(
            trainer.root / "project.json",
            d.digest(trainer.root / "project.json"),
            bindings,
        )
    return helper, plan, trainer, manifest, sft, runner, baseline, rows


def prepare(request_path, checksum, output):
    if hashlib.sha256(Path(request_path).read_bytes()).hexdigest() != checksum:
        raise ValueError("Wrong prior request hash")
    request = request_read(request_path)
    d = launcher(request)
    d.require(request.get("schema") == "trusted-prior-nll-request-v1", "Wrong request")
    bindings = {}
    d.bind_file(request_path, checksum, bindings)
    d.bind_file(request["launcher"], request["launcher_sha256"], bindings)
    d.bind_file(__file__, d.digest(__file__), bindings)
    diagnostic_path = d.bind_file(
        request["diagnostic_sealed_plan"], request["diagnostic_sealed_sha256"], bindings
    )
    diagnostic = d.read(diagnostic_path)
    d.guard(diagnostic)
    runtime = Path(diagnostic["request"]["runtime"])
    d.validate_resources(request["resources"], runtime)
    _helper, plan, _trainer, _manifest, _sft, _runner, baseline, rows = context(
        d, request, diagnostic, bindings
    )
    root = Path(request["output"]).absolute()
    d.require(
        root.parent.is_dir()
        and root.parent.resolve() == root.parent
        and not root.exists()
        and not root.is_symlink(),
        "Require fresh canonical output",
    )
    d.require(
        not root.is_relative_to(runtime)
        and not root.is_relative_to(Path(diagnostic["request"]["model"])),
        "Output inside model/runtime",
    )
    prior = next(c for c in diagnostic["request"]["conditions"] if c["name"] == "prior")
    command = build_command(
        plan["commands"]["baseline"],
        prior["adapter"],
        root / "prior.json",
        request["resources"],
    )
    result = {
        "schema": "trusted-prior-nll-plan-v1",
        "created_utc": d.now(),
        "request": request,
        "inputs_sha256": bindings,
        "command": command,
        "baseline_loss": baseline["loss"],
        "baseline_process_id": baseline["evaluation_context"]["process_id"],
        "examples": len(rows),
        "selected_examples_sha256": baseline["evaluation_context"][
            "selected_examples_sha256"
        ],
        "qualification": "Same exact prepared sixteen rows and loss masks as completed M5 baseline/final. "
        "Prior prefix/decoded caches disabled, explicitly reported resource difference; no kernel-speed claim.",
    }
    guard(d, result, diagnostic)
    d.write_new(output, result)
    return result


def guard(d, sealed, diagnostic):
    d.guard(diagnostic)
    for p, checksum in sealed["inputs_sha256"].items():
        d.require(d.digest(p) == checksum, "Prior evaluation input drift: " + p)


def verify_result(
    d, helper, plan, trainer, manifest, command, adapter, pid, baseline, rows
):
    from exl3_mlx.fit_probe import validate_scores

    path, result = helper.verify_evaluation(
        plan, trainer, manifest, "baseline", command=command, adapter=Path(adapter)
    )
    d.require(
        result["evaluation_context"]["process_id"] == pid
        and pid != baseline["evaluation_context"]["process_id"],
        "Prior PID does not match observed child",
    )
    d.require(
        result["evaluation_context"]["selected_examples_sha256"]
        == baseline["evaluation_context"]["selected_examples_sha256"],
        "Selected rows differ",
    )
    validate_scores(result, rows)
    d.require(
        [r["target_ids"] for r in result["rows"]]
        == [r["target_ids"] for r in baseline["rows"]],
        "Loss targets differ from baseline",
    )
    return path, result


def run(plan_path, checksum, verify_only=False):
    if hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() != checksum:
        raise ValueError("Wrong prior plan hash")
    sealed = request_read(plan_path)
    request = sealed["request"]
    d = launcher(request)
    d.require(
        d.digest(plan_path) == checksum
        and sealed["schema"] == "trusted-prior-nll-plan-v1",
        "Wrong prior plan",
    )
    diagnostic = d.read(request["diagnostic_sealed_plan"])
    guard(d, sealed, diagnostic)
    # Fresh process imports pinned CPU project verifier once.
    helper, plan, trainer, manifest, _sft, runner, baseline, rows = context(
        d, request, diagnostic, {}
    )
    prior = next(c for c in diagnostic["request"]["conditions"] if c["name"] == "prior")
    out = Path(request["output"])
    expected = build_command(
        plan["commands"]["baseline"],
        prior["adapter"],
        out / "prior.json",
        request["resources"],
    )
    d.require(sealed["command"] == expected, "Prior command drift")
    d.require(
        not out.exists() and not out.is_symlink(), "Existing/uncertain prior output"
    )
    if verify_only:
        return {"status": "verified_only", "no_subprocesses_started": True}
    out.mkdir()
    receipt_path = out / "launch.json"
    receipt = {
        "schema": "trusted-prior-nll-launch-v1",
        "status": "prepared",
        "created_utc": d.now(),
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": checksum,
        "processes": {},
        "numerical_environment": d.ENV,
        "quality_certified": False,
    }
    d.write_new(receipt_path, receipt)
    handlers = {
        s: signal.signal(s, runner.handle_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    r = diagnostic["request"]
    child_plan = {
        "cwd": r["runtime"],
        "environment": {
            "PYTHONPATH": str(Path(r["runtime"]) / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "project": plan["project"],
    }
    try:
        with d.controlled_environment(r["runtime"]) as env:
            receipt["actual_child_environment"] = dict(env)
            guard(d, sealed, diagnostic)
            record = runner.run_child(
                "prior-nll",
                expected,
                child_plan,
                receipt,
                receipt_path,
                model_phase=True,
            )
            guard(d, sealed, diagnostic)
            d.require(
                d.digest(plan_path) == checksum, "Prior plan changed during execution"
            )
            path, result = verify_result(
                d,
                helper,
                plan,
                trainer,
                manifest,
                expected,
                prior["adapter"],
                record["process_id"],
                baseline,
                rows,
            )
            d.require(
                result.get("resources") == request["resources"],
                "Wrong prior resource report",
            )
            receipt.update(
                status="completed_pending_independent_audit",
                report=str(path),
                report_sha256=d.digest(path),
                examples=16,
                baseline_loss=baseline["loss"],
                prior_loss=result["loss"],
                prior_minus_base_loss=result["loss"] - baseline["loss"],
            )
            for name, check in record["log_sha256"].items():
                d.require(d.digest(record[name]) == check, "Prior execution log drift")
            guard(d, sealed, diagnostic)
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        receipt["finished_utc"] = d.now()
        runner.persist(receipt_path, receipt)
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--request", required=True)
    q.add_argument("--sha256", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("run")
    q.add_argument("--plan", required=True)
    q.add_argument("--sha256", required=True)
    q.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    r = (
        prepare(a.request, a.sha256, a.output)
        if a.action == "prepare"
        else run(a.plan, a.sha256, a.verify_only)
    )
    print(
        json.dumps({"status": r.get("status", "sealed"), "gpu_during_prepare": False})
    )


if __name__ == "__main__":
    main()
