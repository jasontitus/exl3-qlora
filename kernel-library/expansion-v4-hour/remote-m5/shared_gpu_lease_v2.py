"""Fail-fast Sushi-compatible GPU lease for an explicitly reviewed local command.

Never breaks/prunes foreign leases or tickets and never signals a process.
No retry: a busy/unknown owner blocks before command launch. On interrupted or
uncertain cleanup the owned lease remains with a recovery receipt.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

LOCK = Path("/private/tmp/sushi-gpu.lock.d")
PROTOCOL_SHA256 = "b50468be43cda9970f0d0919af1e0fa8657d26fbac66da368eb109c756abee41"
PROTOCOL_PATH = Path("/Users/jasontitus/experiments/sushi/scripts/gpu-lock.sh")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def read(path):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, "Duplicate JSON key")
            d[k] = v
        return d

    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def save(path, value, *, fresh=False):
    path = Path(path)
    if fresh:
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        return
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def exists(path):
    return Path(path).exists() or Path(path).is_symlink()


def identity(path):
    s = Path(path).lstat()
    return {"dev": s.st_dev, "ino": s.st_ino, "mode": s.st_mode, "uid": s.st_uid}


def owner_status(lock):
    """Observation is deliberately not proof that a absent-PID lease is stale."""
    lock = Path(lock)
    if not exists(lock):
        return {"status": "free"}
    result = {"status": "occupied_or_uncertain", "lock_identity": identity(lock)}
    if lock.is_symlink():
        result["claim"] = os.readlink(lock)
    try:
        info = (lock / "owner").stat()
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= 1024, "Invalid owner file"
        )
        result["owner_record"] = (lock / "owner").read_text()
    except (OSError, ValueError, UnicodeError) as error:
        result["uncertain"] = str(error)
    return result


def queue_entries(lock):
    q = Path(str(lock) + ".queue")
    if not exists(q):
        return []
    require(not q.is_symlink() and q.is_dir(), "Uncertain queue type")
    entries = []
    for p in q.iterdir():
        require(p.name.isdecimal() and p.is_symlink(), "Uncertain queue entry")
        value = os.readlink(p)
        require(
            value == "done" or re.fullmatch(r"[1-9][0-9]* [A-Za-z0-9_-]+", value),
            "Uncertain queue ticket",
        )
        entries.append((int(p.name), p, value))
    return sorted(entries)


def recovery_clear(lock, own=None):
    for p in lock.parent.glob(lock.name + ".ktune-engine-*.json"):
        if p == own:
            continue
        require(
            not p.is_symlink() and p.stat().st_size <= 65536,
            "Uncertain recovery receipt",
        )
        require(
            read(p).get("recovery_required") is False,
            "Unresolved GPU recovery: " + str(p),
        )


def validate_state(state, lock):
    require(state["lock_dir"] == str(lock), "Wrong shared lock")
    owner = state["owner"]
    require(
        re.fullmatch(r"ktune-engine-[1-9][0-9]*-[a-f0-9]{32}", owner),
        "Invalid own identity",
    )
    require(
        state["controller_pid"] > 0
        and owner.startswith("ktune-engine-" + str(state["controller_pid"]) + "-"),
        "Wrong own PID",
    )
    claim = Path(state["claim"])
    require(
        claim.parent == lock.parent
        and re.fullmatch(re.escape(lock.name) + r"\.holder\.[a-f0-9]{32}", claim.name),
        "Invalid own claim path",
    )
    recovery = Path(state["recovery_receipt"])
    require(
        recovery == lock.with_name(lock.name + "." + owner + ".json"),
        "Invalid own recovery path",
    )
    return claim, recovery


def claim_locked(state, *, lock=LOCK):
    """Called only by this file under the same /usr/bin/lockf mutation lock."""
    claim, recovery = validate_state(state, lock)
    require(
        owner_status(lock)["status"] == "free",
        "GPU occupied or uncertain; no acquisition",
    )
    recovery_clear(lock)
    require(
        all(v == "done" for _, _, v in queue_entries(lock)),
        "Existing waiter; do not jump queue",
    )
    q = Path(str(lock) + ".queue")
    q.mkdir(exist_ok=True)
    # Register with the existing FIFO namespace. Never delete another ticket.
    ticket_value = str(state["controller_pid"]) + " " + state["owner"]
    ticket = None
    published = False
    created_claim = False
    require(
        not exists(claim) and not exists(recovery), "Own destination already exists"
    )
    try:
        n = max([n for n, _, _ in queue_entries(lock)], default=0) + 1
        for _ in range(100):
            ticket = q / str(n)
            try:
                os.symlink(ticket_value, ticket)
                break
            except FileExistsError:
                n += 1
        else:
            raise ValueError("Concurrent queue registration uncertain")
        require(
            all(i >= n or v == "done" for i, _, v in queue_entries(lock)),
            "Earlier waiter appeared",
        )
        require(
            owner_status(lock)["status"] == "free", "Owner appeared before atomic claim"
        )
        claim.mkdir(mode=0o700)
        created_claim = True
        owner_bytes = (
            state["owner"]
            + " "
            + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + "\n"
        ).encode()
        with (claim / "owner").open("xb") as f:
            f.write(owner_bytes)
            f.flush()
            os.fsync(f.fileno())
        state.update(
            ownership="unresolved",
            released=False,
            recovery_required=True,
            created_utc=now(),
        )
        save(recovery, state, fresh=True)
        os.symlink(str(claim), lock)
        published = True
        state.update(
            ownership="owned",
            lock_identity=identity(lock),
            claim_identity=identity(claim),
            owner_sha256=sha(claim / "owner"),
        )
        save(recovery, state)
        # Only replace our own exact ticket. Done marker is compatible with Sushi.
        require(
            ticket.is_symlink() and os.readlink(ticket) == ticket_value,
            "Own queue ticket drift",
        )
        done = q / (".done." + uuid.uuid4().hex)
        os.symlink("done", done)
        os.replace(done, ticket)
        return state
    finally:
        if not published:
            if (
                ticket is not None
                and ticket.is_symlink()
                and os.readlink(ticket) == ticket_value
            ):
                ticket.unlink()
            if (
                created_claim
                and claim.is_dir()
                and not claim.is_symlink()
                and (claim / "owner").is_file()
                and (claim / "owner").read_text().split()[:1] == [state["owner"]]
            ):
                (claim / "owner").unlink()
                claim.rmdir()
            if recovery.exists():
                state.update(
                    ownership="not_acquired", released=False, recovery_required=False
                )
                save(recovery, state)


def release_locked(state, *, lock=LOCK):
    """Delete only exact owned objects; retain any changed/uncertain ownership."""
    claim, recovery = validate_state(state, lock)
    require(
        state.get("cleanup_verified") is True, "Cleanup not proven; retain own lease"
    )
    require(
        lock.is_symlink() and os.readlink(lock) == str(claim),
        "Foreign/uncertain lease; no deletion",
    )
    require(
        identity(lock) == state["lock_identity"]
        and identity(claim) == state["claim_identity"],
        "Lease identity drift",
    )
    require(
        not claim.is_symlink() and {p.name for p in claim.iterdir()} == {"owner"},
        "Claim contents changed",
    )
    require(
        (claim / "owner").is_file()
        and not (claim / "owner").is_symlink()
        and sha(claim / "owner") == state["owner_sha256"],
        "Owner changed",
    )
    require((claim / "owner").read_text().split()[0] == state["owner"], "Foreign owner")
    lock.unlink()
    (claim / "owner").unlink()
    claim.rmdir()
    state.update(
        ownership="released", released=True, recovery_required=False, released_utc=now()
    )
    save(recovery, state)
    return state


def capture_command(argv):
    # Unlike subprocess.run exception cleanup, this never signals a child.
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def mutate(action, state_path):
    cmd = [
        "/usr/bin/lockf",
        "-k",
        "-t",
        "0",
        str(LOCK) + ".mutation",
        sys.executable,
        str(Path(__file__).resolve()),
        "_mutate",
        action,
        str(state_path),
        sha(state_path),
    ]
    code, _stdout, stderr = capture_command(cmd)
    require(code == 0, "Shared mutation busy/failed; no GPU command: " + stderr[-500:])
    return read(state_path)


def processes():
    code, stdout, _stderr = capture_command(
        ["/bin/ps", "-axo", "pid=,ppid=,pgid=,lstart="]
    )
    require(code == 0, "Cannot verify process cleanup")
    result = {}
    for line in stdout.splitlines():
        x = line.split()
        require(
            len(x) == 8 and all(s.isdecimal() for s in x[:3]),
            "Uncertain process snapshot",
        )
        result[int(x[0])] = {
            "ppid": int(x[1]),
            "pgid": int(x[2]),
            "start": " ".join(x[3:]),
        }
    return result


def update_descendants(table, tracked, leader, generations=None):
    """Track owned process generations; a reused leader is uncertain, never adopted.

    A reused descendant PID is admitted only through a currently live owned
    parent or the still-live original leader's process group. An unrelated new
    process never inherits an expired PID's ownership. Birth identity is the
    macOS ps start string, so this is cooperative cleanup evidence, not OS
    attestation against unobserved daemonization or same-second PID reuse.
    """
    require(type(leader) is int and leader > 0, "Positive owned leader required")
    if generations is not None:
        for pid, start in tracked.items():
            history = generations.setdefault(pid, [])
            if start not in history:
                history.append(start)
    for pid, row in table.items():
        require(
            type(pid) is int and pid >= 0 and isinstance(row, dict)
            and type(row.get("ppid")) is int and row["ppid"] >= 0
            and type(row.get("pgid")) is int and row["pgid"] >= 0
            and isinstance(row.get("start"), str) and bool(row["start"].strip()),
            "Uncertain process generation metadata",
        )
    if leader in tracked and leader in table:
        require(
            tracked[leader] == table[leader]["start"],
            "Leader PID generation changed; retain lease for ownership review",
        )
    changed = True
    while changed:
        changed = False
        live = {
            pid for pid, start in tracked.items()
            if pid in table and table[pid]["start"] == start
        }
        for pid, row in table.items():
            belongs = pid > 0 and (row["ppid"] in live or (leader in live and row["pgid"] == leader))
            if tracked.get(pid) != row["start"] and belongs:
                tracked[pid] = row["start"]
                changed = True
    if leader not in table:
        require(
            not any(
                row["pgid"] == leader and tracked.get(pid) != row["start"]
                for pid, row in table.items()
            ),
            "Uncertain late leader-group process; retain lease for cleanup review",
        )
    if generations is not None:
        for pid, start in tracked.items():
            history = generations.setdefault(pid, [])
            if start not in history:
                history.append(start)
    return {
        pid: row for pid, row in table.items()
        if pid in tracked and row["start"] == tracked[pid]
    }


def verify_inputs(request):
    pins = request.get("inputs_sha256")
    require(isinstance(pins, dict) and pins, "Explicit command/input hashes required")
    for name, checksum in pins.items():
        require(
            isinstance(name, str)
            and isinstance(checksum, str)
            and re.fullmatch(r"[a-f0-9]{64}", checksum),
            "Invalid input identity",
        )
        path = Path(name)
        require(
            path.is_absolute()
            and str(path) == name
            and path.parent.resolve() == path.parent
            and path.is_file(),
            "Noncanonical/missing input: " + name,
        )
        if path.is_symlink():
            require(
                pins.get(str(path.resolve())) == checksum,
                "Symlink executable/input target must also be pinned",
            )
        require(sha(path) == checksum, "Command/input drift: " + name)
    require(request["argv"][0] in pins, "Executable must be hash pinned")
    for arg in request["argv"][1:]:
        if (
            arg != request.get("completion_receipt")
            and arg.startswith("/")
            and Path(arg).is_file()
        ):
            require(arg in pins, "Existing file argument must be pinned: " + arg)
    if "observed_harness_sha256" in request:
        require(
            request["observed_harness_sha256"] == pins[request["argv"][0]],
            "Harness expectation differs from pinned executable",
        )
    return pins.copy()


def validate_request(request):
    require(request.get("schema") == "shared-gpu-lease-command-v1", "Wrong request")
    require(
        sys.platform == "darwin" and socket.gethostname() == request["hostname"],
        "Wrong execution host",
    )
    require(sha(PROTOCOL_PATH) == PROTOCOL_SHA256, "Sushi lease protocol changed")
    require(
        os.environ.get("GPU_LOCK_DIR", str(LOCK))
        in (str(LOCK), "/tmp/sushi-gpu.lock.d"),
        "Conflicting GPU_LOCK_DIR",
    )
    cmd = request["argv"]
    require(
        isinstance(cmd, list)
        and cmd
        and all(isinstance(s, str) and "\x00" not in s for s in cmd),
        "Invalid command argv",
    )
    require(
        Path(cmd[0]).is_absolute() and Path(cmd[0]).is_file(),
        "Absolute executable required",
    )
    cwd = Path(request["cwd"])
    require(
        cwd.is_absolute() and cwd.resolve() == cwd and cwd.is_dir(),
        "Canonical existing cwd required",
    )
    require(
        isinstance(request["expected_exit_codes"], list)
        and request["expected_exit_codes"]
        and all(type(i) is int for i in request["expected_exit_codes"]),
        "Expected exit codes required",
    )
    receipt = Path(request["completion_receipt"])
    require(
        receipt.is_absolute()
        and receipt.parent.resolve() == receipt.parent
        and not exists(receipt),
        "Fresh canonical completion receipt required",
    )
    require(
        isinstance(request["completion_expect"], dict) and request["completion_expect"],
        "Explicit completion criteria required",
    )
    require(
        request.get("no_detached_background_children") is True,
        "Command lifecycle must be reviewed",
    )
    verify_inputs(request)


def run(request_path, checksum, receipt_path):
    require(sha(request_path) == checksum, "Request hash mismatch")
    request = read(request_path)
    validate_request(request)
    receipt_path = Path(receipt_path)
    require(
        receipt_path.is_absolute()
        and receipt_path.parent.resolve() == receipt_path.parent
        and receipt_path.parent.is_dir()
        and not exists(receipt_path),
        "Fresh canonical wrapper receipt required",
    )
    observation = owner_status(LOCK)
    receipt = {
        "schema": "shared-gpu-lease-command-receipt-v1",
        "request": str(Path(request_path).resolve()),
        "request_sha256": checksum,
        "wrapper_sha256": sha(__file__),
        "protocol_sha256": PROTOCOL_SHA256,
        "started_utc": now(),
        "status": "preflight",
        "initial_owner": observation,
        "processes_signalled": False,
        "foreign_objects_removed": False,
        "inputs_sha256": request.get("inputs_sha256", {}).copy(),
    }
    save(receipt_path, receipt, fresh=True)
    state_path = receipt_path.with_name(receipt_path.name + ".lease.json")
    owner = "ktune-engine-" + str(os.getpid()) + "-" + uuid.uuid4().hex
    state = {
        "owner": owner,
        "controller_pid": os.getpid(),
        "lock_dir": str(LOCK),
        "claim": str(LOCK.with_name(LOCK.name + ".holder." + uuid.uuid4().hex)),
        "recovery_receipt": str(LOCK.with_name(LOCK.name + "." + owner + ".json")),
        "wrapper_sha256": receipt["wrapper_sha256"],
    }
    child = None
    tracked = {}
    generations = {}
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(
            "Wrapper interrupted; no process signalled; retain lease pending cleanup review"
        )

    try:
        require(
            observation["status"] == "free",
            "GPU occupied/uncertain; existing owner preserved",
        )
        recovery_clear(LOCK)
        require(
            all(v == "done" for _, _, v in queue_entries(LOCK)),
            "GPU waiter exists; no queue jumping",
        )
        save(state_path, state, fresh=True)
        state = mutate("claim", state_path)
        receipt.update(status="owned", lease=state)
        save(receipt_path, receipt)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            handlers[sig] = signal.signal(sig, interrupted)
        env = os.environ.copy()
        env["GPU_LOCK_DIR"] = str(LOCK)
        logs = {
            k: receipt_path.with_name(receipt_path.name + "." + k + ".log")
            for k in ("stdout", "stderr")
        }
        with logs["stdout"].open("x") as stdout, logs["stderr"].open("x") as stderr:
            verify_inputs(request)
            require(sha(request_path) == checksum, "Request changed before child start")
            child = subprocess.Popen(
                request["argv"],
                cwd=request["cwd"],
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            receipt.update(
                status="running",
                child_pid=child.pid,
                argv=request["argv"],
                cwd=request["cwd"],
            )
            save(receipt_path, receipt)
            while True:
                table = processes()
                if child.pid in table and child.pid not in tracked:
                    tracked[child.pid] = table[child.pid]["start"]
                update_descendants(table, tracked, child.pid, generations)
                if child.poll() is not None:
                    break
                time.sleep(0.2)
        remaining = update_descendants(processes(), tracked, child.pid, generations)
        require(
            not remaining, "Owned descendants remain; own lease deliberately retained"
        )
        require(
            child.returncode in request["expected_exit_codes"],
            "Unexpected child exit; own lease retained for review",
        )
        complete = read(request["completion_receipt"])
        require("error" not in complete, "Completion receipt contains error")
        require(
            all(complete.get(k) == v for k, v in request["completion_expect"].items()),
            "Completion criteria not met; lease retained",
        )
        verify_inputs(request)
        require(sha(request_path) == checksum, "Request changed during child execution")
        require(
            sha(__file__) == receipt["wrapper_sha256"]
            and sha(PROTOCOL_PATH) == PROTOCOL_SHA256,
            "Lease implementation drift; retain own lease",
        )
        receipt.update(
            exit_code=child.returncode,
            completion_sha256=sha(request["completion_receipt"]),
            logs_sha256={str(v): sha(v) for v in logs.values()},
            observed_descendants=tracked,
            observed_descendant_generations=generations,
        )
        state["cleanup_verified"] = True
        save(state_path, state)
        state = mutate("release", state_path)
        receipt.update(status="completed", lease=state)
    except BaseException as error:
        receipt.update(
            status="blocked" if child is None else "failed_lease_retained",
            error=type(error).__name__ + ": " + str(error),
            observed_descendants=tracked,
            observed_descendant_generations=generations,
        )
        # Acquisition can have published its lease before a controller interruption.
        if exists(Path(state["recovery_receipt"])):
            receipt["lease"] = read(state["recovery_receipt"])
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        receipt["finished_utc"] = now()
        save(receipt_path, receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("status")
    a = sub.add_parser("run")
    a.add_argument("--request", required=True)
    a.add_argument("--sha256", required=True)
    a.add_argument("--receipt", required=True)
    a = sub.add_parser("_mutate")
    a.add_argument("operation", choices=("claim", "release"))
    a.add_argument("state")
    a.add_argument("sha256")
    args = p.parse_args()
    if args.action == "status":
        print(json.dumps(owner_status(LOCK), indent=2))
        return
    if args.action == "_mutate":
        require(
            sys.platform == "darwin" and sha(PROTOCOL_PATH) == PROTOCOL_SHA256,
            "Wrong host/protocol",
        )
        require(sha(args.state) == args.sha256, "State drift")
        value = read(args.state)
        require(value["wrapper_sha256"] == sha(__file__), "Wrapper source drift")
        value = (claim_locked if args.operation == "claim" else release_locked)(value)
        save(args.state, value)
        return
    print(json.dumps(run(args.request, args.sha256, args.receipt)))


if __name__ == "__main__":
    main()
