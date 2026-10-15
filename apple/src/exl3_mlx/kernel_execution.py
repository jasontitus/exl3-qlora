"""Run explicitly trusted evaluation backends, never model-generated commands.

Backend manifests bind argv, code and semantic JSON result predicates. The
default only validates. Device backends must be existing owned controllers;
this bridge neither grants a lease nor replaces their resource/cleanup checks.
"""

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .evaluation_receipts import hash_file
from .kernel_eval_suite import STAGES, Reader, require, validate_result


def equal(left, right):
    """JSON equality without bool/int or int/float coercion."""
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def field(value, path):
    require(
        isinstance(path, list) and path and all(isinstance(x, str) for x in path),
        "Invalid field path",
    )
    for part in path:
        require(
            isinstance(value, dict) and part in value, "Missing result field: " + ".".join(path)
        )
        value = value[part]
    return value


def matches(value, predicates):
    require(isinstance(predicates, list) and predicates, "A semantic predicate is required")
    for item in predicates:
        require(isinstance(item, dict) and set(item) == {"path", "equals"}, "Invalid predicate")
        if not equal(field(value, item["path"]), item["equals"]):
            return False
    return True


def substitute(text, values):
    require(isinstance(text, str) and "\0" not in text, "Invalid argv/path string")
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    require("{" not in text and "}" not in text, "Unknown template placeholder")
    return text


def expand(value, values):
    if isinstance(value, str):
        return substitute(value, values)
    if isinstance(value, list):
        return [expand(x, values) for x in value]
    if isinstance(value, dict):
        return {k: expand(v, values) for k, v in value.items()}
    return value


def ref_path(ref, base):
    path = Path(ref["path"])
    return path.absolute() if path.is_absolute() else (base / path).absolute()


def file_ref(path):
    return {"path": str(path), "sha256": hash_file(path, max_bytes=128 << 20)}


def validate_row(row, reader):
    task = {
        "sha256": row["task_sha256"],
        "prompt_sha256": row["prompt_sha256"],
        "protocol_sha256": row["protocol_sha256"],
    }
    return validate_result(
        row,
        {row["task_id"]: task},
        {row["adapter_id"]: {"sha256": row["adapter_sha256"]}},
        row["budget"],
        reader,
    )


def validate_backend(backend, reader):
    require(
        isinstance(backend, dict)
        and set(backend)
        == {
            "format",
            "kind",
            "executable",
            "code",
            "argv",
            "cwd",
            "env",
            "timeout_seconds",
            "result",
        },
        "Invalid backend manifest",
    )
    require(backend["format"] == "kernel-eval-backend-v1", "Invalid backend format")
    require(backend["kind"] in ("cpu", "owned_device_controller"), "Unsupported backend kind")
    executable = ref_path(backend["executable"], reader.base)
    require(
        set(backend["executable"]) == {"path", "sha256"}
        and hash_file(executable, max_bytes=128 << 20) == backend["executable"]["sha256"],
        "Executable changed",
    )
    require(
        isinstance(backend["code"], list) and 0 < len(backend["code"]) <= 128,
        "Missing trusted code pins",
    )
    for ref in backend["code"]:
        reader.read(ref)
    require(
        isinstance(backend["argv"], list)
        and backend["argv"]
        and all(isinstance(x, str) and "\0" not in x for x in backend["argv"]),
        "Invalid argv",
    )
    require(
        isinstance(backend["env"], dict)
        and all(
            isinstance(k, str) and isinstance(v, str) and "=" not in k and "\0" not in k + v
            for k, v in backend["env"].items()
        ),
        "Invalid environment",
    )
    timeout = backend["timeout_seconds"]
    if backend["kind"] == "owned_device_controller":
        require(timeout is None, "Device controller must own timeout and descendant cleanup")
    else:
        require(type(timeout) in (int, float) and 0 < timeout <= 3600, "Invalid CPU timeout")
    result = backend["result"]
    require(
        isinstance(result, dict) and set(result) == {"path", "bindings", "stages"},
        "Invalid result contract",
    )
    require(
        isinstance(result["stages"], dict)
        and result["stages"]
        and set(result["stages"]) <= set(STAGES[1:]),
        "Invalid backend result stages",
    )
    require(
        "timing" not in result["stages"] or set(result["stages"]) == {"timing"},
        "Timing must be a separate stage after numerical acceptance",
    )
    # A backend may produce raw_format+replay together, or source+compile, etc.
    # Predicate values and exit codes are explicit; exit zero is never enough.
    require(isinstance(result["bindings"], list) and result["bindings"], "Missing result bindings")
    require(
        any(
            isinstance(p, dict) and p.get("equals") == "{input_sha256}" for p in result["bindings"]
        ),
        "Result must bind the exact input SHA256",
    )
    for name, outcomes in result["stages"].items():
        require(isinstance(outcomes, list) and 0 < len(outcomes) <= 8, "Invalid outcomes")
        for outcome in outcomes:
            require(
                isinstance(outcome, dict)
                and set(outcome)
                <= {"when", "exit_codes", "status", "outcome_class", "reason", "measurements_path"}
                and {"when", "exit_codes", "status", "outcome_class"} <= set(outcome),
                "Invalid outcome",
            )
            require(
                outcome["status"] in ("passed", "failed", "refused", "not_run", "pending"),
                "Invalid status",
            )
            require(
                outcome["outcome_class"] in ("model_output", "infrastructure", "unspecified"),
                "Invalid outcome class",
            )
            require(
                isinstance(outcome["exit_codes"], list)
                and outcome["exit_codes"]
                and all(type(x) is int for x in outcome["exit_codes"]),
                "Invalid exit codes",
            )
            require(
                isinstance(outcome["when"], list) and outcome["when"],
                "Exit code alone cannot establish a result",
            )
            if name == "timing" and outcome["status"] == "passed":
                require("measurements_path" in outcome, "Timing requires actual measurements")
    return executable


def execute(plan_path, checksum, output, *, run=False, allow_owned_device=False):
    plan_path, output = Path(plan_path).absolute(), Path(output).absolute()
    reader = Reader(plan_path.parent)
    plan = reader.json({"path": str(plan_path), "sha256": checksum})
    require(
        isinstance(plan, dict) and set(plan) == {"format", "initial_result", "track", "steps"},
        "Invalid execution plan",
    )
    require(
        plan["format"] == "kernel-eval-execution-v1" and plan["track"] in ("strict", "normalized"),
        "Invalid protocol/track",
    )
    row = copy.deepcopy(reader.json(plan["initial_result"]))
    validate_row(row, reader)
    # Make inherited evidence portable when the output is in another directory.
    for track in row["tracks"].values():
        for stage in track.values():
            for ref in stage.get("evidence", []):
                ref["path"] = str(ref_path(ref, reader.base))
    require(isinstance(plan["steps"], list) and 0 < len(plan["steps"]) <= 8, "Invalid step count")
    prepared = []
    for step in plan["steps"]:
        require(isinstance(step, dict) and set(step) == {"backend", "input"}, "Invalid step")
        backend = reader.json(step["backend"])
        executable = validate_backend(backend, reader)
        reader.read(step["input"])
        prepared.append((step, backend, executable))
    reader.recheck()
    if not run:
        return {
            "format": "kernel-eval-execution-validation-v1",
            "validated_steps": len(prepared),
            "executed": False,
            "device_backend_requested": any(b["kind"] != "cpu" for _, b, _ in prepared),
        }
    require(
        allow_owned_device or all(b["kind"] == "cpu" for _, b, _ in prepared),
        "Device execution requires explicit --allow-owned-device and the backend own admission",
    )
    require(output.parent.resolve(strict=True) == output.parent, "Noncanonical output parent")
    output.mkdir()
    stages = row["tracks"].setdefault(plan["track"], {})
    runs = []
    for index, (step, backend, executable) in enumerate(prepared):
        names = sorted(backend["result"]["stages"], key=STAGES.index)
        first = names[0]
        prior = STAGES[STAGES.index(first) - 1]
        ready = (
            row["tracks"]["strict"].get("generation", {}).get("status") == "passed"
            if plan["track"] == "normalized" and first == "replay"
            else stages.get(prior, {}).get("status") == "passed"
        )
        if not ready:
            break  # Includes the numerical-pass prerequisite for timing.
        require(
            all(
                stages.get(name, {}).get("status", "not_run") in ("pending", "not_run")
                for name in names
            ),
            "Cannot rerun or overwrite a completed stage",
        )
        directory = output / f"{index:02d}-{first}"
        directory.mkdir()
        values = {
            "input": ref_path(step["input"], reader.base),
            "input_sha256": step["input"]["sha256"],
            "output": directory / "work",
            "backend_dir": ref_path(step["backend"], reader.base).parent,
        }
        argv = [str(executable)] + [substitute(x, values) for x in backend["argv"]]
        cwd = Path(substitute(backend["cwd"], values)).absolute()
        require(cwd.resolve(strict=True) == cwd, "Noncanonical backend cwd")
        env = {k: substitute(v, values) for k, v in backend["env"].items()}
        result_path = Path(substitute(backend["result"]["path"], values)).absolute()
        # Existing qualified controllers can constrain their output parent.
        # That exact path may be outside bridge logs, but must still be fresh.
        require(
            result_path.resolve(strict=False) == result_path
            and not result_path.exists()
            and not result_path.is_symlink(),
            "Backend result must be fresh and canonical",
        )
        reader.recheck()
        require(
            hash_file(executable, max_bytes=128 << 20) == backend["executable"]["sha256"],
            "Executable changed",
        )
        started = time.time()
        run_receipt = {
            "argv": argv,
            "cwd": str(cwd),
            "kind": backend["kind"],
            "started_unix": started,
            "input": step["input"],
            "backend": step["backend"],
            "exit_code": None,
            "timed_out": False,
        }
        semantic = {}
        try:
            with (
                (directory / "stdout.log").open("xb") as stdout,
                (directory / "stderr.log").open("xb") as stderr,
            ):
                child = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=backend["kind"] == "cpu",
                )
                try:
                    code = child.wait(timeout=backend["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    # Only our newly created CPU process group. Device ownership
                    # and cleanup are exclusively the qualified controller's job.
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    run_receipt["timed_out"] = True
                    raise ValueError("CPU backend timed out") from None
                run_receipt["exit_code"] = code
            native_ref = file_ref(result_path)
            native = reader.json(native_ref)
            bindings = expand(backend["result"]["bindings"], values)
            require(matches(native, bindings), "Backend result binding mismatch")
            for name, outcomes in backend["result"]["stages"].items():
                selected = [
                    x for x in outcomes if code in x["exit_codes"] and matches(native, x["when"])
                ]
                require(len(selected) == 1, "Missing/ambiguous semantic outcome: " + name)
                choice = selected[0]
                value = {k: choice[k] for k in ("status", "outcome_class")}
                value["evidence"] = [native_ref]
                if "reason" in choice:
                    value["reason"] = choice["reason"]
                if "measurements_path" in choice:
                    value["measurements"] = field(native, choice["measurements_path"])
                semantic[name] = value
            candidate = copy.deepcopy(row)
            candidate["tracks"][plan["track"]].update(semantic)
            validate_row(candidate, reader)
            reader.recheck()
            require(
                hash_file(executable, max_bytes=128 << 20) == backend["executable"]["sha256"],
                "Late executable change",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            semantic = {
                first: {"status": "refused", "outcome_class": "infrastructure", "reason": str(exc)}
            }
        run_receipt["finished_unix"] = time.time()
        run_receipt["stage_outcomes"] = semantic
        receipt_path = directory / "execution.json"
        receipt_path.write_text(json.dumps(run_receipt, indent=2, allow_nan=False) + "\n")
        receipt_ref = file_ref(receipt_path)
        for value in semantic.values():
            value.setdefault("evidence", []).append(receipt_ref)
        stages.update(semantic)
        runs.append(receipt_ref)
        if any(x["status"] in ("failed", "refused") for x in semantic.values()):
            break
    row["execution"] = {
        "plan_sha256": checksum,
        "runs": runs,
        "trust": "Backend assertions; not independent certification. No generated shell is executed.",
    }
    validate_row(row, reader)
    reader.recheck()
    with (output / "result.json").open("x") as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return row


def import_replay(request_path, checksum, output, *, run=False):
    """Convenience adapter for an existing capture and a task's baseline bundle.

    The capture's generation stage is supplied evidence, not attested here.
    All later replay stage decisions come from the actual bounded parser.
    """
    from . import evaluation_audit, evaluation_receipts, kernel_replay

    request_path, output = Path(request_path).absolute(), Path(output).absolute()
    reader = Reader(request_path.parent)
    request = reader.json({"path": str(request_path), "sha256": checksum})
    require(
        isinstance(request, dict)
        and set(request) == {"format", "initial_result", "replay"}
        and request["format"] == "kernel-eval-replay-request-v1",
        "Invalid replay request",
    )
    row = reader.json(request["initial_result"])
    validate_row(row, reader)
    for stages in row["tracks"].values():
        for value in stages.values():
            for ref in value.get("evidence", []):
                ref["path"] = str(ref_path(ref, reader.base))
    replay = copy.deepcopy(request["replay"])
    require(set(replay) == {"generation", "baseline", "required_paths"}, "Invalid replay bundle")
    for ref in [replay["generation"], *replay["baseline"].values()]:
        reader.read(ref)
        ref["path"] = str(ref_path(ref, reader.base))
    reader.recheck()
    require(output.parent.resolve(strict=True) == output.parent, "Noncanonical output parent")
    output.mkdir()

    def write(name, data):
        path = output / name
        with path.open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return file_ref(path)

    initial_ref = write("initial-result.json", row)
    input_ref = write("replay-input.json", replay)
    outcomes = {
        stage: [
            {
                "when": [{"path": ["stages", stage, "status"], "equals": status}],
                "exit_codes": [0, 1],
                "status": status,
                "outcome_class": "model_output" if status == "failed" else "unspecified",
            }
            for status in ("passed", "failed", "not_run")
        ]
        for stage in ("raw_format", "replay")
    }
    backend = {
        "format": "kernel-eval-backend-v1",
        "kind": "cpu",
        "executable": file_ref(Path(sys.executable).resolve()),
        "code": [
            file_ref(Path(module.__file__).resolve())
            for module in (kernel_replay, evaluation_audit, evaluation_receipts)
        ],
        "argv": [
            "-m",
            "exl3_mlx.kernel_replay",
            "--input",
            "{input}",
            "--sha256",
            "{input_sha256}",
            "--output",
            "{output}",
        ],
        "cwd": str(output),
        "env": {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "timeout_seconds": 30,
        "result": {
            "path": "{output}/replay.json",
            "bindings": [
                {"path": ["format"], "equals": "kernel-eval-replay-v1"},
                {"path": ["input_sha256"], "equals": "{input_sha256}"},
            ],
            "stages": outcomes,
        },
    }
    backend_ref = write("replay-backend.json", backend)
    plan = {
        "format": "kernel-eval-execution-v1",
        "initial_result": initial_ref,
        "track": "strict",
        "steps": [{"backend": backend_ref, "input": input_ref}],
    }
    plan_ref = write("execution-plan.json", plan)
    reader.recheck()
    return execute(plan_ref["path"], plan_ref["sha256"], output / "execution", run=run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path)
    source.add_argument(
        "--import-replay",
        type=Path,
        help="Prepare a replay plan from a captured response/task bundle",
    )
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-owned-device", action="store_true")
    args = parser.parse_args(argv)
    if args.import_replay:
        require(not args.allow_owned_device, "Replay import never uses a device")
        result = import_replay(args.import_replay, args.sha256, args.output, run=args.execute)
    else:
        result = execute(
            args.plan,
            args.sha256,
            args.output,
            run=args.execute,
            allow_owned_device=args.allow_owned_device,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
