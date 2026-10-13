"""Launch the already-predeclared control train/evaluate serially.

The launcher and its preflight are CPU-only. Running without --verify-only starts
the explicit GPU subprocesses in the plan. Never retries or changes settings.
"""

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def persist(path, value):
    """Replace only the receipt this invocation has already exclusively created."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_auditor():
    path = Path(__file__).with_name("audit_matched_control.py")
    spec = importlib.util.spec_from_file_location("matched_control_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LaunchInterrupted(BaseException):
    """Translate handled process signals into child cleanup and a receipt."""


def handle_stop(signum, _frame):
    # A second ordinary stop signal must not interrupt cleanup/receipt writing.
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(stop_signal, signal.SIG_IGN)
    raise LaunchInterrupted(f"Received {signal.Signals(signum).name}")


def verify_runtime_versions(auditor, plan):
    expected = auditor.read(auditor.bound_path(plan, "bf16-all-layers-post.json"))[
        "evaluation_context"
    ]["implementation"]
    for phase in ("train_argv", "post_argv"):
        auditor.require(
            os.path.abspath(plan["launch"][phase][0]) == os.path.abspath(sys.executable),
            "Run this launcher with the exact planned Python executable",
        )
    actual = {
        "python": platform.python_version(),
        "runtime": {name: importlib.metadata.version(name) for name in expected["runtime"]},
        "executable": sys.executable,
    }
    auditor.require(
        actual["python"] == expected["python"], "Installed Python version differs from treatment"
    )
    auditor.require(
        actual["runtime"] == expected["runtime"],
        "Installed dependency versions differ from treatment",
    )
    return actual


def guard(auditor, plan_path, identity):
    for path, expected in identity.items():
        auditor.require(auditor.digest(path) == expected, f"Launch input changed: {path}")
    plan = auditor.read(plan_path)
    auditor.verify_plan(plan)
    return {
        "checked_at_utc": now(),
        "identity_sha256": identity,
        "all_pinned_files_verified": True,
        "installed_runtime": verify_runtime_versions(auditor, plan),
    }


def run_phase(name, argv, cwd, environment, receipt, receipt_path, stdout, stderr):
    record = dict(
        argv=argv,
        cwd=str(cwd),
        environment={"PYTHONPATH": environment["PYTHONPATH"]},
        stdout=str(stdout),
        stderr=str(stderr),
        started_utc=now(),
        status="starting",
    )
    receipt["phases"][name] = record
    persist(receipt_path, receipt)
    process = None
    try:
        with stdout.open("x") as out, stderr.open("x") as err:
            process = subprocess.Popen(
                argv, cwd=cwd, env=environment, stdout=out, stderr=err, start_new_session=True
            )
            record.update(process_id=process.pid, status="running")
            persist(receipt_path, receipt)
            code = process.wait()
        record.update(exit_code=code, status="completed" if code == 0 else "failed")
        if code:
            raise RuntimeError(f"{name} exited with status {code}; no later phase was launched")
    except BaseException as error:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        if process is not None:
            record["exit_code"] = process.returncode
        raise
    finally:
        record["finished_utc"] = now()
        persist(receipt_path, receipt)


def launch(plan_path, receipt_path, *, verify_only=False):
    auditor = load_auditor()
    plan_path, receipt_path = Path(plan_path).resolve(), Path(receipt_path).resolve()
    plan = auditor.read(plan_path)
    auditor.verify_plan(plan)
    config = auditor.read(plan["config"])
    output, post = Path(config["output"]), Path(plan["launch"]["post_report"])
    logs = {
        phase: tuple(
            receipt_path.with_name(receipt_path.stem + f"-{phase}.{suffix}")
            for suffix in ("stdout.log", "stderr.log")
        )
        for phase in ("train", "evaluate")
    }
    for path in (receipt_path, output, post, *(p for pair in logs.values() for p in pair)):
        auditor.require(
            not path.exists() and not path.is_symlink(), f"Refusing existing output: {path}"
        )
    auditor.require(receipt_path.parent.is_dir(), "Receipt parent does not exist")
    identity = {
        str(path): auditor.digest(path)
        for path in (
            plan_path,
            Path(plan["config"]),
            Path(plan["runtime"]) / "snapshot.json",
            Path(__file__).resolve(),
            Path(auditor.__file__).resolve(),
        )
    }
    initial = guard(auditor, plan_path, identity)
    if verify_only:
        return {"status": "verified_only", "no_subprocesses_started": True, **initial}
    receipt = dict(
        schema="exl3-matched-control-launch-v1",
        status="prepared",
        created_utc=now(),
        plan=str(plan_path),
        identity_sha256=identity,
        before_train=initial,
        phases={},
        quality_certified=False,
    )
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    environment = {**os.environ, **plan["launch"]["environment"]}
    cwd = Path(plan["launch"]["cwd"])
    previous_handlers = {
        stop_signal: signal.signal(stop_signal, handle_stop)
        for stop_signal in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        run_phase(
            "train",
            plan["launch"]["train_argv"],
            cwd,
            environment,
            receipt,
            receipt_path,
            *logs["train"],
        )
        receipt["after_train"] = guard(auditor, plan_path, identity)
        training = auditor.read(output / "training.json")
        auditor.require(
            training.get("status") == "completed" and training.get("completed_step") == 1,
            "Training did not complete planned step",
        )
        auditor.require(
            Path(training["checkpoint"]).resolve() == output.resolve() / "step-000001",
            "Training checkpoint differs",
        )
        helper, _ = auditor.verify_plan(plan)
        helper.read_checkpoint(training["checkpoint"])
        receipt["training_report_sha256"] = auditor.digest(output / "training.json")
        persist(receipt_path, receipt)
        run_phase(
            "evaluate",
            plan["launch"]["post_argv"],
            cwd,
            environment,
            receipt,
            receipt_path,
            *logs["evaluate"],
        )
        receipt["after_evaluate"] = guard(auditor, plan_path, identity)
        receipt["post_report_sha256"] = auditor.digest(post)
        receipt["status"] = "completed_pending_cpu_audit"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            receipt["finished_utc"] = now()
            for phase in receipt["phases"].values():
                phase["log_sha256"] = {
                    key: auditor.digest(phase[key])
                    for key in ("stdout", "stderr")
                    if Path(phase[key]).is_file()
                }
            persist(receipt_path, receipt)
        finally:
            for stop_signal, handler in previous_handlers.items():
                signal.signal(stop_signal, handler)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="CPU preflight only; no files or subprocesses created",
    )
    args = parser.parse_args()
    result = launch(args.plan, args.receipt, verify_only=args.verify_only)
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
