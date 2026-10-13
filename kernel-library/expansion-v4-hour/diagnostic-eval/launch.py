"""Seal and run one fixed, serial base/prior/final diagnostic campaign.

CPU preparation does not import MLX. Execution is explicit, has no retry/resume,
and retains failures. This records trusted local execution, not adversarial OS
attestation. Full checkpoint verification must precede preparation; subsequent
stat guards assume the checkpoint remains quiescent (no hostile stat spoofing).
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import socket
import sys
from pathlib import Path

ENV = {"MLX_ENABLE_TF32": "0"}
ROLES = ("base", "prior", "final")
TASKS = (
    "conv_interval",
    "conv_gap",
    "broadcast_index",
    "broadcast_discriminator",
    "sum_missing_collective",
    "sum_rounding_contract",
    "timing_scope",
    "timing_attribution",
)
BASE_NAMES = (
    "config.json",
    "expert-manifest-v2.json",
    "engram/engram-manifest.json",
    "engram-token-map.u32.json",
    "model.safetensors.index.json",
    "tokenizer.json",
)
SETTINGS = (
    "rank",
    "alpha",
    "adapter_layers",
    "expert_adapter_layers",
    "activation_mode",
)
PACKAGES = {
    "mlx": "0.32.3",
    "mlx-metal": "0.32.3",
    "mlx-lm": "0.32.0",
    "numpy": "2.5.4",
    "transformers": "5.19.0",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 << 20), b""):
            h.update(data)
    return h.hexdigest()


def read(path):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, "Duplicate JSON key: " + k)
            d[k] = v
        return d

    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def regular(path):
    p = Path(path).absolute()
    require(
        p.is_file() and not any(q.is_symlink() for q in (p, *p.parents)),
        "Expected regular non-symlink file: " + str(p),
    )
    return p


def bind_file(path, expected, bindings):
    p = regular(path)
    require(digest(p) == expected, "Input checksum mismatch: " + str(p))
    require(
        str(p) not in bindings or bindings[str(p)] == expected,
        "Conflicting input identity: " + str(p),
    )
    bindings[str(p)] = expected
    return p


def stat_id(path):
    s = regular(path).stat()
    return {
        k: getattr(s, "st_" + k) for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")
    }


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_environment(request):
    require(
        {k: v for k, v in os.environ.items() if k.startswith("MLX_")} == ENV,
        "Require only explicit MLX_ENABLE_TF32=0",
    )
    require(
        not any(
            k == "PYTHONHOME" or k.startswith(("DYLD_", "LD_")) for k in os.environ
        ),
        "Unreviewed interpreter/native library environment",
    )
    require(
        str(Path(sys.executable).absolute()) == request["python"],
        "Wrong Python executable",
    )
    require(platform.python_version() == "3.12.13", "Wrong Python version")
    require(socket.gethostname() == request["hostname"], "Wrong host")
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(versions == PACKAGES, "Dependency version drift")
    return versions


def package_bindings():
    """Bind distribution payloads including native MLX/Metal, without importing them."""
    result = {}
    for name in PACKAGES:
        dist = importlib.metadata.distribution(name)
        require(bool(dist.files), "Missing installed package inventory: " + name)
        for relative in dist.files:
            if "__pycache__" in relative.parts or relative.suffix == ".pyc":
                continue
            p = Path(dist.locate_file(relative)).absolute()
            # Entry-point symlinks are not executable dependencies of this launcher.
            if p.is_file():
                result[str(p.resolve())] = digest(p)
    executable = Path(sys.executable).resolve(strict=True)
    result[str(executable)] = digest(executable)
    venv = Path(sys.executable).absolute().parent.parent
    for p in [venv / "pyvenv.cfg", *venv.glob("lib/python*/site-packages/*.pth")]:
        if p.is_file():
            result[str(p)] = digest(p)
    return result


def payload_state(request, runner, bindings):
    receipt_path = bind_file(
        request["payload_receipt"], request["payload_sha256"], bindings
    )
    source = bind_file(
        request["payload_source_manifest"], request["payload_source_sha256"], bindings
    )
    receipt, original = read(receipt_path), read(source)
    runner.validate_payload_receipt(receipt)
    require(
        receipt["verified_manifest_sha256"] == digest(source),
        "Payload source not bound",
    )
    require(
        original.get("schema") == "checkpoint-full-payload-sha256-v1"
        and receipt["files"] == original.get("files")
        and receipt["total_bytes"] == original.get("total_bytes"),
        "Full payload inventory differs",
    )
    model = Path(request["model"]).resolve(strict=True)
    require(str(model) == receipt["root"], "Payload receipt points to another model")
    start = datetime.datetime.fromisoformat(receipt["started_utc"]).timestamp() * 1e9
    require(
        datetime.datetime.fromisoformat(receipt["completed_utc"])
        <= datetime.datetime.now(datetime.UTC),
        "Future payload verification timestamp",
    )
    actual = {str(p.relative_to(model)): p for p in model.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in model.rglob("*")), "Model contains symlink")
    require(
        set(actual) == set(receipt["files"]), "Model inventory changed after full hash"
    )
    states = {}
    for name, p in actual.items():
        s = stat_id(p)
        require(s["size"] == receipt["files"][name]["size"], "Model size changed")
        require(
            max(s["mtime_ns"], s["ctime_ns"]) <= start,
            "Model changed since full verification began: " + name,
        )
        states[str(p)] = s
    return states, receipt


def validate_spec(request, bindings):
    plan_path = bind_file(
        request["diagnostic_plan"], request["diagnostic_plan_sha256"], bindings
    )
    review_path = bind_file(request["review"], request["review_sha256"], bindings)
    scorer = bind_file(request["scorer"], request["scorer_sha256"], bindings)
    plan, review = read(plan_path), read(review_path)
    require(
        plan.get("schema") == "metal-diagnostic-decision-eval-v1",
        "Wrong diagnostic plan",
    )
    require(
        review.get("status") == "approved_for_staged_path_only_relocation",
        "Missing independent review",
    )
    require(
        digest(plan_path) in review["inputs_sha256"].values()
        and digest(scorer) in review["inputs_sha256"].values(),
        "Review does not bind plan/scorer",
    )
    require(
        tuple(t["id"] for t in plan["tasks"]) == TASKS,
        "Require fixed ordered eight tasks",
    )
    require(
        plan["decoding"]["method"] == "greedy"
        and plan["decoding"]["max_new_tokens"] == 8,
        "Unreviewed decoding contract",
    )
    for field in ("source_bindings", "tokenizer_bindings"):
        for p, checksum in plan[field].items():
            bind_file(p, checksum, bindings)
    model = Path(request["model"]).resolve()
    require(
        set(plan["tokenizer_bindings"])
        == {
            str(model / n)
            for n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "Wrong tokenizer root",
    )
    for task in plan["tasks"]:
        p = regular(plan_path.parent / (task["id"] + ".messages.json"))
        require(read(p) == task["messages"], "Messages do not match fixed task")
        require(
            digest(p) in review["inputs_sha256"].values(),
            "Messages not independently reviewed",
        )
        bindings[str(p)] = digest(p)
    return plan


def adapter_conditions(request, payload, bindings):
    require(
        [c["name"] for c in request["conditions"]] == list(ROLES),
        "Declare base/prior/final in order",
    )
    contracts = []
    common = request["inference"]
    require(
        common["activation_mode"] == "checkpoint",
        "Only reviewed checkpoint activation mode",
    )
    require(
        type(common["chunk_size"]) is int
        and common["chunk_size"] >= 128
        and common["chunk_size"] % 128 == 0,
        "Invalid chunk size",
    )
    for condition in request["conditions"]:
        if condition["name"] == "base":
            require(condition["adapter"] is None, "Base must not use an adapter")
            continue
        directory = Path(condition["adapter"]).resolve(strict=True)
        config_path, weights = (
            directory / "adapter_config.json",
            directory / "adapter.safetensors",
        )
        bind_file(config_path, digest(config_path), bindings)
        bind_file(weights, digest(weights), bindings)
        config = read(config_path)
        require(
            config.get("format") == "exl3-mlx-deepseek-short-text-v1",
            "Wrong adapter format",
        )
        require(
            config.get("engine_semantics") == common["engine_semantics"],
            "Wrong adapter arithmetic",
        )
        require(
            config.get("activation_mode") == common["activation_mode"],
            "Wrong adapter activation",
        )
        require(
            config.get("adapter_sha256") == digest(weights),
            "Wrong adapter weight checksum",
        )
        require(
            config["base_identity"]
            == {name: payload["files"][name]["sha256"] for name in BASE_NAMES},
            "Adapter base differs from verified model",
        )
        settings = {k: config[k] for k in SETTINGS}
        require(
            settings == condition["settings"],
            "Adapter settings differ from declaration",
        )
        contracts.append(settings)
        for p in (config_path, weights):
            bind_file(p, digest(p), bindings)
        state_path = directory / "trainer_state.json"
        require(
            state_path.is_file(), "Require full training checkpoint integrity state"
        )
        bind_file(state_path, digest(state_path), bindings)
        state = read(state_path)
        require(
            state.get("format") == "exl3-mlx-training-state-v1",
            "Wrong trainer-state format",
        )
        for p in (config_path, weights):
            require(
                state["files"].get(p.name) == digest(p), "Trainer checksum mismatch"
            )
        require(
            state.get("step") == condition["step"], "Wrong declared checkpoint step"
        )
        bind_file(state_path, digest(state_path), bindings)
    require(
        contracts[0] == contracts[1],
        "Prior/final adapters require identical inference settings",
    )
    require(
        request["conditions"][1]["adapter"] != request["conditions"][2]["adapter"],
        "Duplicate adapter conditions",
    )
    return contracts[0]


def validate_resources(values, runtime):
    module = load_module(runtime / "src/exl3_mlx/resources.py", "diagnostic_resources")
    resolved = module.resolve_resources(**values)
    require(resolved == values, "Declare every resolved resource field explicitly")
    require(
        values["decoded_cache_mib"] == values["prefix_cache_mib"] == 0
        and values["prefix_disk_cache"] is None
        and values["prefix_disk_cache_gib"] == 0,
        "Diagnostic decoded/prefix caches must be disabled",
    )


def validate_request_paths(request):
    """Seal canonical paths, before loading any supplied executable helper."""
    for name in (
        "model",
        "runtime",
        "diagnostic_plan",
        "scorer",
        "helper",
        "runner",
        "output",
    ):
        value = request[name]
        require(isinstance(value, str), "Path must be a string: " + name)
        path = Path(value)
        require(
            path.is_absolute() and str(path) == value and path.resolve() == path,
            "Require canonical absolute path: " + name,
        )
        if name != "output":
            require(path.exists(), "Missing input path: " + name)
    adapters = []
    require(
        [c.get("name") for c in request["conditions"]] == list(ROLES),
        "Require ordered base/prior/final conditions",
    )
    require(request["conditions"][0].get("adapter") is None, "Base has an adapter")
    for condition in request["conditions"][1:]:
        value = condition["adapter"]
        require(isinstance(value, str), "Adapter path must be a string")
        path = Path(value)
        require(
            path.is_absolute()
            and str(path) == value
            and path.resolve() == path
            and path.is_dir(),
            "Require canonical absolute adapter directory",
        )
        adapters.append(path)
    require(
        len(set(adapters)) == 2, "Prior/final adapters must be distinct directories"
    )


def prepare(request_path, expected_sha256, output):
    bindings = {}
    request_path = bind_file(request_path, expected_sha256, bindings)
    request = read(request_path)
    require(
        request.get("schema") == "trusted-diagnostic-generation-request-v1",
        "Wrong request",
    )
    require(not Path(output).exists(), "Existing sealed plan")
    validate_request_paths(request)
    check_environment(request)
    runner_path = bind_file(request["runner"], request["runner_sha256"], bindings)
    helper_path = bind_file(request["helper"], request["helper_sha256"], bindings)
    runner = load_module(runner_path, "trusted_diagnostic_process_wrapper")
    helper = load_module(helper_path, "trusted_diagnostic_runtime_verifier")
    runtime, snapshot = helper.verify_runtime(request["runtime"])
    require(snapshot == request["snapshot_sha256"], "Wrong frozen runtime")
    # Read the pinned constant without importing the MLX-dependent package.
    context_text = (runtime / "src/exl3_mlx/context.py").read_text()
    require(
        "ENGINE_SEMANTICS = " + json.dumps(request["inference"]["engine_semantics"])
        in context_text,
        "Runtime does not declare requested arithmetic version",
    )
    validate_resources(request["resources"], runtime)
    bindings[str(runtime / "snapshot.json")] = snapshot
    bindings[str(Path(__file__).resolve())] = digest(__file__)
    bindings.update(package_bindings())
    states, payload = payload_state(request, runner, bindings)
    plan = validate_spec(request, bindings)
    adapter_settings = adapter_conditions(request, payload, bindings)
    out = Path(request["output"]).absolute()
    require(
        out.parent.is_dir() and out.parent.resolve() == out.parent,
        "Output parent must be canonical and existing",
    )
    require(
        not out.is_relative_to(runtime)
        and not out.is_relative_to(Path(request["model"])),
        "Campaign output must be outside runtime/model",
    )
    require(
        not out.exists() and not out.is_symlink(),
        "Output must be a fresh campaign directory",
    )
    result = {
        "schema": "trusted-diagnostic-generation-plan-v1",
        "created_utc": now(),
        "request": request,
        "inputs_sha256": bindings,
        "model_file_stats": states,
        "adapter_settings": adapter_settings,
        "task_ids": [t["id"] for t in plan["tasks"]],
        "qualification": "Trusted local serial execution; no malicious OS/stat-spoofing attestation. "
        "Fixed eight choices from five inspected families; no generated-kernel quality claim.",
        "comparison_policy": "Base, prior and fixed final adapter declared before any output. "
        "No retries, checkpoint selection, prompt tuning or automatic resume from results.",
    }
    guard(result)
    write_new(output, result)
    return result


def guard(sealed):
    request = sealed["request"]
    check_environment(request)
    for p, checksum in sealed["inputs_sha256"].items():
        require(digest(p) == checksum, "Launch input drift: " + p)
    helper = load_module(request["helper"], "trusted_diagnostic_runtime_guard")
    require(
        helper.verify_runtime(request["runtime"])[1] == request["snapshot_sha256"],
        "Runtime drift",
    )
    model = Path(request["model"])
    inventory = {str(p) for p in model.rglob("*") if p.is_file()}
    require(inventory == set(sealed["model_file_stats"]), "Model inventory drift")
    for p, state in sealed["model_file_stats"].items():
        require(stat_id(p) == state, "Model changed since preparation: " + p)
    return {
        "checked_utc": now(),
        "host": socket.gethostname(),
        "numerical_environment": ENV,
    }


def generate_argv(sealed, role, task, output):
    r = sealed["request"]
    args = [
        r["python"],
        "-m",
        "exl3_mlx.workflow",
        "generate",
        "--model",
        r["model"],
        "--messages-file",
        str(Path(r["diagnostic_plan"]).parent / (task + ".messages.json")),
        "--max-tokens",
        "8",
        "--report",
        str(output),
        "--chunk-size",
        str(r["inference"]["chunk_size"]),
        "--activation-mode",
        r["inference"]["activation_mode"],
    ]
    for key, value in r["resources"].items():
        if value is not None:
            args.extend(["--" + key.replace("_", "-"), str(value)])
    condition = next(c for c in r["conditions"] if c["name"] == role)
    if condition["adapter"]:
        args.extend(["--adapter", condition["adapter"]])
    else:
        # The frozen CLI uses rank16/alpha32 for a zero-B base allocation;
        # these allocation defaults do not claim trained-adapter configuration.
        args.extend(
            [
                "--adapter-layers",
                *map(str, sealed["adapter_settings"]["adapter_layers"]),
                "--expert-adapter-layers",
                *map(str, sealed["adapter_settings"]["expert_adapter_layers"]),
            ]
        )
    return args


def verify_generation(sealed, role, task, path):
    r = sealed["request"]
    result = read(path)
    t = next(x for x in read(r["diagnostic_plan"])["tasks"] if x["id"] == task)
    c = next(x for x in r["conditions"] if x["name"] == role)
    require(
        result.get("prompt_ids") == t["prompt_ids"]
        and result.get("prompt_messages") == t["messages"],
        "Wrong generated prompt",
    )
    require(
        result.get("checkpoint") == r["model"]
        and result.get("adapter") == c["adapter"],
        "Wrong generation model/adapter",
    )
    require(
        result.get("activation_mode") == r["inference"]["activation_mode"]
        and result.get("resources") == r["resources"],
        "Wrong reported generation contract",
    )
    require(
        result.get("method") == "greedy full-prefix recomputation"
        and result.get("max_new_tokens") == 8
        and result.get("tools") is None,
        "Wrong generation method",
    )
    return digest(path)


@contextlib.contextmanager
def controlled_environment(runtime):
    old = dict(os.environ)
    keep = {
        k: v
        for k, v in old.items()
        if k in ("HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LANG", "LC_ALL")
    }
    keep.update(
        ENV,
        PYTHONPATH=str(Path(runtime) / "src"),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    os.environ.clear()
    os.environ.update(keep)
    try:
        yield keep
    finally:
        os.environ.clear()
        os.environ.update(old)


def run(plan_path, expected_sha256, verify_only=False):
    p = regular(plan_path)
    require(digest(p) == expected_sha256, "Wrong sealed plan hash")
    sealed = read(p)
    require(
        sealed.get("schema") == "trusted-diagnostic-generation-plan-v1",
        "Wrong sealed plan",
    )
    r = sealed["request"]
    output = Path(r["output"])
    require(
        not output.exists() and not output.is_symlink(),
        "Existing/uncertain campaign output",
    )
    initial = guard(sealed)
    require(sealed["task_ids"] == list(TASKS), "Incomplete fixed task order")
    runner = load_module(r["runner"], "trusted_diagnostic_run_wrapper")
    if verify_only:
        return {
            "status": "verified_only",
            "no_subprocesses_started": True,
            "guard": initial,
        }
    output.mkdir()  # Exclusive single-use campaign claim, before any child.
    receipt_path = output / "launch.json"
    receipt = {
        "schema": "trusted-diagnostic-generation-launch-v1",
        "status": "prepared",
        "plan": str(p),
        "plan_sha256": expected_sha256,
        "created_utc": now(),
        "host": socket.gethostname(),
        "initial_guard": initial,
        "processes": {},
        "scores": {},
        "conditions": r["conditions"],
        "numerical_environment": ENV,
        "quality_certified": False,
    }
    write_new(receipt_path, receipt)
    previous = {
        s: signal.signal(s, runner.handle_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    child_plan = {
        "cwd": r["runtime"],
        "environment": {
            "PYTHONPATH": str(Path(r["runtime"]) / "src"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "project": str(output),
    }
    try:
        with controlled_environment(r["runtime"]) as env:
            receipt["actual_child_environment"] = dict(env)
            for role in ROLES:
                reports = []
                for task in TASKS:
                    require(digest(p) == expected_sha256, "Sealed plan changed")
                    guard(sealed)
                    report = output / (role + "-" + task + ".json")
                    require(not report.exists(), "Preexisting task report")
                    key = role + "-" + task
                    record = runner.run_child(
                        key,
                        generate_argv(sealed, role, task, report),
                        child_plan,
                        receipt,
                        receipt_path,
                    )
                    guard(sealed)
                    require(
                        record["exit_code"] == 0
                        and record["status"] == "completed"
                        and type(record["process_id"]) is int,
                        "Unsuccessful generation child",
                    )
                    record["generation_report"] = str(report)
                    record["generation_sha256"] = verify_generation(
                        sealed, role, task, report
                    )
                    reports.append((task, report))
                    runner.persist(receipt_path, receipt)
                # Score only this complete, fixed condition after rechecking every report/log.
                for task, report in reports:
                    record = receipt["processes"][role + "-" + task]
                    require(
                        digest(report) == record["generation_sha256"],
                        "Generated report drift",
                    )
                    for logname, checksum in record["log_sha256"].items():
                        require(digest(record[logname]) == checksum, "Child log drift")
                guard(sealed)
                score_path = output / (role + "-score.json")
                argv = [
                    r["python"],
                    r["scorer"],
                    "--plan",
                    r["diagnostic_plan"],
                    "--model",
                    r["model"],
                    "--output",
                    str(score_path),
                ]
                for task, report in reports:
                    argv.extend(["--report", task + "=" + str(report)])
                adapter = next(
                    c["adapter"] for c in r["conditions"] if c["name"] == role
                )
                if adapter:
                    argv.extend(["--adapter", adapter])
                runner.run_child(
                    role + "-score", argv, child_plan, receipt, receipt_path
                )
                score = read(score_path)
                require(
                    score.get("total") == 8
                    and [x["id"] for x in score["results"]] == list(TASKS)
                    and score.get("plan_sha256") == r["diagnostic_plan_sha256"]
                    and score.get("adapter") == adapter,
                    "Incomplete or mismatched scoring",
                )
                receipt["scores"][role] = {
                    "path": str(score_path),
                    "sha256": digest(score_path),
                }
                guard(sealed)
            # A final all-condition rehash closes cross-condition output mutation.
            for record in receipt["processes"].values():
                for name, checksum in record["log_sha256"].items():
                    require(digest(record[name]) == checksum, "Final child-log drift")
                if "generation_report" in record:
                    require(
                        digest(record["generation_report"])
                        == record["generation_sha256"],
                        "Final generation drift",
                    )
            for item in receipt["scores"].values():
                require(digest(item["path"]) == item["sha256"], "Final score drift")
            require(digest(p) == expected_sha256, "Final sealed-plan drift")
            receipt["status"] = "completed_pending_independent_audit"
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        receipt["finished_utc"] = now()
        runner.persist(receipt_path, receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--request", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        prepare(args.request, args.sha256, args.output)
        if args.action == "prepare"
        else run(args.plan, args.sha256, args.verify_only)
    )
    print(
        json.dumps(
            {
                "status": result.get("status", "sealed"),
                "no_gpu_during_prepare": args.action == "prepare",
            }
        )
    )


if __name__ == "__main__":
    main()
