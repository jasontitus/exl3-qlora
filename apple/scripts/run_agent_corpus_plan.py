"""Run a pinned native-edit experiment once, serially, without retries.

The helper emits and verifies commands in fresh CPU processes. Only the five
explicit phase commands use the model. Existing/uncertain outputs block launch;
an interrupted run needs separate operator review, never automatic continuation.
"""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PHASES = ("preflight-train", "preflight-test", "baseline", "train", "adapter")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def persist(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_helper(plan):
    path = Path(plan["helper"])
    require(digest(path) == plan["helper_sha256"], "Pinned helper changed")
    spec = importlib.util.spec_from_file_location("agent_plan_validation", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def guard(plan_path, identity, helper):
    for path, checksum in identity.items():
        require(digest(path) == checksum, "Pinned launch input changed: " + path)
    plan = read(plan_path)
    runtime, snapshot = helper.verify_runtime(plan["runtime"])
    require(snapshot == plan["snapshot_sha256"], "Pinned runtime snapshot changed")
    require(str(Path(sys.executable).absolute()) == plan["python"], "Use the planned Python")
    require(platform.python_version() == plan["implementation"]["python"], "Python version changed")
    versions = {
        name: importlib.metadata.version(name) for name in plan["implementation"]["runtime"]
    }
    require(versions == plan["implementation"]["runtime"], "Installed dependency versions changed")
    require(plan["order"] == list(PHASES), "Unexpected phase order")
    require(plan["cwd"] == str(runtime), "Unexpected working directory")
    require(plan["environment"] == {"PYTHONPATH": str(runtime / "src")}, "Unexpected environment")
    return {"checked_utc": now(), "snapshot_sha256": snapshot, "runtime_versions": versions}


def initial_paths(plan):
    project = Path(plan["project"])
    folder = project / "experiment"
    return [
        project / ".agent-corpus-launch.json",
        project / "attempts",
        project / "active.json",
        project / "stop.request",
        folder / "preflight-train",
        folder / "preflight-test.json",
        folder / "baseline.json",
        folder / "baseline-before-training.json",
        folder / "adapter.json",
    ]


def process_table():
    """Read macOS process ancestry/birth/command without an added dependency."""
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,lstart=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    table = {}
    for line in result.stdout.splitlines():
        values = line.split(maxsplit=7)
        if len(values) == 8:
            table[int(values[0])] = {
                "parent": int(values[1]),
                "identity": " ".join(values[2:]),
            }
    return table


def observed_children(pid):
    """Capture only descendants of the process started by this launcher."""
    table, found, parents = process_table(), {}, {pid}
    while parents:
        children = {p for p, v in table.items() if v["parent"] in parents and p not in found}
        found.update({p: table[p]["identity"] for p in children})
        parents = children
    return found


def cleanup(process, children):
    children.update(observed_children(process.pid))
    table = process_table()
    for pid, identity in children.items():
        if table.get(pid, {}).get("identity") == identity:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while children:
        table = process_table()
        children = {
            p: identity
            for p, identity in children.items()
            if table.get(p, {}).get("identity") == identity
        }
        if not children or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    for pid in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


class LaunchInterrupted(BaseException):
    pass


def handle_stop(signum, _frame):
    for item in (signal.SIGINT, signal.SIGTERM):
        signal.signal(item, signal.SIG_IGN)
    raise LaunchInterrupted("Received " + signal.Signals(signum).name)


@contextlib.contextmanager
def direct_phase_lock(project, enabled):
    # SFT train owns this same lock itself. Direct workflow phases otherwise
    # need exclusion from an independently launched project operation.
    if not enabled:
        yield ()
        return
    fd = os.open(Path(project) / ".run.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield (fd,)
    finally:
        os.close(fd)


def run_child(key, argv, plan, receipt, receipt_path, *, model_phase=False):
    record = {
        "argv": argv,
        "cwd": plan["cwd"],
        "environment": plan["environment"],
        "started_utc": now(),
        "started_ns": time.time_ns(),
        "status": "starting",
        "stdout": str(receipt_path.with_name(receipt_path.stem + "-" + key + ".stdout.log")),
        "stderr": str(receipt_path.with_name(receipt_path.stem + "-" + key + ".stderr.log")),
    }
    receipt["processes"][key] = record
    persist(receipt_path, receipt)
    process, children = None, {}
    try:
        with direct_phase_lock(plan["project"], model_phase and key != "train") as fds:
            with Path(record["stdout"]).open("x") as out, Path(record["stderr"]).open("x") as err:
                process = subprocess.Popen(
                    argv,
                    cwd=plan["cwd"],
                    env={**os.environ, **plan["environment"]},
                    stdout=out,
                    stderr=err,
                    shell=False,
                    start_new_session=True,
                    pass_fds=fds,
                )
                record.update(process_id=process.pid, status="running")
                persist(receipt_path, receipt)
                while True:
                    children.update(observed_children(process.pid))
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            record.update(exit_code=code, status="completed" if code == 0 else "failed")
            require(code == 0, f"{key} exited {code}; no later phase will run")
    except BaseException as error:
        if process is not None:
            cleanup(process, children)
            record["exit_code"] = process.returncode
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(finished_utc=now(), finished_ns=time.time_ns(), observed_descendants=children)
        record["log_sha256"] = {
            name: digest(record[name])
            for name in ("stdout", "stderr")
            if Path(record[name]).is_file()
        }
        persist(receipt_path, receipt)
    return record


def emitted_command(value, phase, plan, plan_path, checksum):
    require(value.get("phase") == phase, "Emitter returned wrong phase")
    require(
        value.get("plan") == str(plan_path) and value.get("plan_sha256") == checksum,
        "Emitter returned a different plan",
    )
    require(
        value.get("cwd") == plan["cwd"] and value.get("environment") == plan["environment"],
        "Emitter changed execution environment",
    )
    command = value.get("argv")
    require(isinstance(command, list) and all(isinstance(x, str) for x in command), "Invalid argv")
    expected = plan["commands"][phase]
    if phase == "adapter":
        require(command[:-1] == expected[:-1], "Emitter changed adapter arguments")
        selected = Path(command[-1]).resolve(strict=True)
        require(
            selected.is_relative_to(Path(plan["project"]) / "attempts"),
            "Checkpoint escapes project",
        )
        require(
            selected.name == f"step-{plan['config']['max_steps']:06d}", "Wrong final checkpoint"
        )
    else:
        require(command == expected, "Emitter changed planned argv")
    require(command[0] == plan["python"] and command[1] == "-m", "Unexpected executable")
    return command


def verify_completion(value, phase, plan, checksum, process_id):
    require(
        value.get("verified_complete") is True
        and value.get("phase") == phase
        and value.get("plan_sha256") == checksum,
        "Helper did not verify completed phase for this plan",
    )
    artifacts = value.get("artifacts_sha256")
    require(isinstance(artifacts, dict) and bool(artifacts), "Missing completed artifact bindings")
    for path, expected in artifacts.items():
        require(digest(path) == expected, "Verified artifact changed: " + path)
    if phase in {"preflight-test", "baseline", "adapter"}:
        path = Path(plan["project"]) / "experiment" / (phase + ".json")
        require(str(path) in artifacts, "Evaluation report missing from verified artifacts")
        report = read(path)
        require(
            report["evaluation_context"]["process_id"] == process_id,
            "Evaluation receipt PID differs from launched process",
        )


def launch(plan_path, receipt_path, *, expected_sha256, verify_only=False):
    plan_path, receipt_path = Path(plan_path).resolve(strict=True), Path(receipt_path).absolute()
    require(digest(plan_path) == expected_sha256, "Plan differs from explicitly pinned SHA256")
    plan = read(plan_path)
    require(plan.get("format") == "native-agent-kernel-run-plan-v1", "Unsupported plan")
    helper = load_helper(plan)
    identity = {
        str(path): digest(path)
        for path in (
            plan_path,
            Path(plan["helper"]),
            Path(plan["runtime"]) / "snapshot.json",
            Path(__file__).resolve(),
        )
    }
    initial = guard(plan_path, identity, helper)
    require(
        digest(Path(plan["project"]) / "project.json") == plan["initial_project_sha256"],
        "Project changed since preparation; no automatic resume",
    )
    for path in [receipt_path, *initial_paths(plan)]:
        require(
            not path.exists() and not path.is_symlink(), "Existing/uncertain output: " + str(path)
        )
    for phase in PHASES:
        for key in (phase, "emit-" + phase, "verify-" + phase):
            for suffix in ("stdout.log", "stderr.log"):
                path = receipt_path.with_name(receipt_path.stem + "-" + key + "." + suffix)
                require(not path.exists() and not path.is_symlink(), "Existing log: " + str(path))
    require(receipt_path.parent.is_dir(), "Receipt parent must exist")
    if verify_only:
        return {"status": "verified_only", "no_subprocesses_started": True, "guard": initial}
    receipt = {
        "format": "native-agent-corpus-launch-v1",
        "status": "prepared",
        "created_utc": now(),
        "plan": str(plan_path),
        "plan_sha256": expected_sha256,
        "identity_sha256": identity,
        "initial_guard": initial,
        "processes": {},
        "verified_phases": {},
        "quality_certified": False,
        "retry_policy": "No retries or automatic resume. Missing/failed receipts require operator review.",
    }
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    claim = Path(plan["project"]) / ".agent-corpus-launch.json"
    with claim.open("x") as stream:
        json.dump(
            {"receipt": str(receipt_path), "plan_sha256": expected_sha256, "pid": os.getpid()},
            stream,
        )
    previous = {item: signal.signal(item, handle_stop) for item in (signal.SIGINT, signal.SIGTERM)}
    try:
        for phase in PHASES:
            guard(plan_path, identity, helper)
            record = run_child(
                "emit-" + phase,
                [
                    plan["python"],
                    plan["helper"],
                    "command",
                    "--plan",
                    str(plan_path),
                    "--phase",
                    phase,
                ],
                plan,
                receipt,
                receipt_path,
            )
            guard(plan_path, identity, helper)
            command = emitted_command(
                read(record["stdout"]), phase, plan, plan_path, expected_sha256
            )
            if phase == "train":
                require(
                    not (Path(plan["project"]) / "attempts").exists(),
                    "Unexpected prior training attempt",
                )
                require(
                    digest(Path(plan["project"]) / "project.json")
                    == plan["initial_project_sha256"],
                    "Project changed before training",
                )
            model_record = run_child(phase, command, plan, receipt, receipt_path, model_phase=True)
            guard(plan_path, identity, helper)
            record = run_child(
                "verify-" + phase,
                [
                    plan["python"],
                    plan["helper"],
                    "verify",
                    "--plan",
                    str(plan_path),
                    "--phase",
                    phase,
                ],
                plan,
                receipt,
                receipt_path,
            )
            verified = read(record["stdout"])
            verify_completion(verified, phase, plan, expected_sha256, model_record["process_id"])
            receipt["verified_phases"][phase] = verified
            receipt["last_guard"] = guard(plan_path, identity, helper)
            persist(receipt_path, receipt)
        receipt["status"] = "completed_pending_quality_audit"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            receipt["finished_utc"] = now()
            persist(receipt_path, receipt)
        finally:
            for item, handler in previous.items():
                signal.signal(item, handler)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = launch(
        args.plan, args.receipt, expected_sha256=args.plan_sha256, verify_only=args.verify_only
    )
    print(json.dumps({"status": result["status"], "receipt": str(args.receipt)}))


if __name__ == "__main__":
    main()
