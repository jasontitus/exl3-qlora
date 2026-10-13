"""Explicit CPU compilation after trusted per-probe generation admission.

No generation, repair, retry or GPU dispatch. The original compiler/Swift builder
is loaded by its sealed hash. Its per-probe parser receives a rejection for an
inadmissible response; other probes retain their original expression and compile.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out

    def invalid(_):
        raise ValueError("Nonfinite JSON")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


def checked(item, pins):
    path = Path(item["path"])
    require(
        path.is_absolute() and path.resolve() == path and path.is_file(),
        "Noncanonical input",
    )
    require(digest(path) == item["sha256"], "Input hash drift: " + str(path))
    require(
        str(path) not in pins or pins[str(path)] == item["sha256"], "Conflicting pin"
    )
    pins[str(path)] = item["sha256"]
    return path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def validate_processes(e, d, sealed, diagnostic, plan, launch):
    """Match observed records to all nine fixed commands, not only return codes."""
    request = sealed["request"]
    out = Path(request["output"])
    expected = {}
    for role in d.ROLES:
        for probe in plan["probes"]:
            key = role + "-" + probe["id"]
            expected[key] = e.command(d, diagnostic, probe, role, out / (key + ".json"))
        expected[role + "-collect"] = e.collection_command(diagnostic, request, role)
    require(set(launch["processes"]) == set(expected), "Wrong fixed process keys")
    env = e.child_environment(d, diagnostic)
    require(
        launch.get("numerical_environment") == d.ENV,
        "Wrong campaign numerical environment",
    )
    actual_env = launch.get("actual_child_environment", {})
    require(
        isinstance(actual_env, dict)
        and {k: actual_env.get(k) for k in env} == env
        and set(actual_env)
        <= set(env) | {"HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LANG", "LC_ALL"}
        and all(isinstance(v, str) for v in actual_env.values()),
        "Wrong controlled child environment",
    )
    last_finish = 0
    for key, argv in expected.items():
        proc = launch["processes"][key]
        require(proc.get("argv") == argv, "Wrong actual child argv: " + key)
        require(
            proc.get("cwd") == diagnostic["request"]["runtime"]
            and proc.get("environment") == env,
            "Wrong actual child environment: " + key,
        )
        require(
            proc.get("status") == "completed"
            and proc.get("exit_code") == 0
            and "error" not in proc,
            "Failed campaign child",
        )
        require(
            type(proc.get("process_id")) is int and proc["process_id"] > 0,
            "Missing actual child PID",
        )
        start, end = proc.get("started_ns"), proc.get("finished_ns")
        require(
            type(start) is int
            and type(end) is int
            and 0 < start <= end
            and start >= last_finish,
            "Missing/unordered serial child timestamps",
        )
        last_finish = end
        require(
            set(proc.get("log_sha256", {})) == {"stdout", "stderr"},
            "Missing actual child logs",
        )
        for name in ("stdout", "stderr"):
            require(
                proc.get(name) == str(out / ("launch-" + key + "." + name + ".log")),
                "Wrong actual child log path",
            )
    for key, result in launch["results"].items():
        require(
            result["report"] == str(out / (key + ".json")),
            "Wrong generation output path",
        )


def context(admission_path, checksum):
    pins = {}
    admission_path = checked(
        {"path": str(Path(admission_path).absolute()), "sha256": checksum}, pins
    )
    admission = read(admission_path)
    require(
        admission["schema"] == "trusted-expression-build-admission-v1",
        "Wrong admission schema",
    )
    require(
        checked(admission["build_wrapper"], pins) == Path(__file__).resolve(),
        "Wrong build wrapper",
    )
    e = load(
        checked(admission["expression_launcher"], pins), "trusted_expression_generation"
    )
    sealed = read(checked(admission["sealed_plan"], pins))
    d = e.load_launcher(sealed["request"])
    diagnostic = d.read(sealed["request"]["diagnostic_sealed_plan"])
    e.guard(d, sealed, diagnostic)
    plan = d.read(sealed["request"]["expression_plan"])
    require(admission["condition"] in d.ROLES, "Unknown condition")
    responses = read(checked(admission["responses"], pins))
    require(
        responses["origin"] == "model_generation"
        and responses["plan_sha256"] == sealed["request"]["expression_plan_sha256"]
        and set(responses["responses"]) == set(admission["results"]) == set(e.PROBES),
        "Wrong response set",
    )
    launch_path = Path(admission["launch_receipt"])
    launch = read(
        checked({"path": str(launch_path), "sha256": digest(launch_path)}, pins)
    )
    require(
        launch.get("status") == "completed_pending_independent_audit"
        and launch["plan_sha256"] == admission["sealed_plan"]["sha256"]
        and launch.get("plan") == admission["sealed_plan"]["path"]
        and set(launch["results"]) == {r + "-" + p for r in d.ROLES for p in e.PROBES}
        and len(launch["processes"]) == 9,
        "Incomplete fixed campaign",
    )
    require(
        launch_path == Path(sealed["request"]["output"]) / "launch.json",
        "Wrong campaign receipt path",
    )
    validate_processes(e, d, sealed, diagnostic, plan, launch)
    collection = launch["collections"][admission["condition"]]
    require(
        collection["admission"] == {"path": str(admission_path), "sha256": checksum}
        and {k: collection[k] for k in ("path", "sha256")} == admission["responses"],
        "Unbound build admission",
    )
    for proc in launch["processes"].values():
        require(
            proc.get("status") == "completed" and proc.get("exit_code") == 0,
            "Failed campaign child",
        )
        for name, checksum0 in proc["log_sha256"].items():
            checked({"path": proc[name], "sha256": checksum0}, pins)
    for key, result in launch["results"].items():
        checked({"path": result["report"], "sha256": result["sha256"]}, pins)
        require(
            result["process_id"] == launch["processes"][key]["process_id"],
            "Generation PID differs from actual child",
        )
    builder = d.load_module(
        sealed["request"]["builder"], "unchanged_expression_builder"
    )
    tok = e.tokenizer(diagnostic)
    e.verify_prompt_ids(d, plan, tok)
    for probe in plan["probes"]:
        key = probe["id"]
        result = admission["results"][key]
        require(
            result == launch["results"][admission["condition"] + "-" + key],
            "Admission result drift",
        )
        response = responses["responses"][key]
        require(
            response["generation_receipt"]
            == {"path": result["report"], "sha256": result["sha256"]},
            "Wrong generation identity",
        )
        generation = read(checked(response["generation_receipt"], pins))
        require(response["text"] == generation["text"], "Changed generated text")
        actual = e.validate_generation(
            d, diagnostic, probe, admission["condition"], generation, tok, builder
        )
        require(
            actual
            == {
                k: v
                for k, v in result.items()
                if k not in ("report", "sha256", "process_id")
            },
            "Generation revalidation disagrees",
        )
    return admission, sealed, diagnostic, plan, pins, e, d, builder


def install_gate(builder, plan, admission):
    """No text substitution: rejection uses the original builder's exception path."""
    original = builder.parse_expression
    probes = {p["operation"]: p for p in plan["probes"]}
    require(len(probes) == len(plan["probes"]), "Nonunique operations")

    def gate(text, operation, cases):
        require(operation in probes, "Unknown operation")
        probe = probes[operation]
        require(cases == probe["cases"], "Changed cases")
        result = admission["results"][probe["id"]]
        if result["status"] != "accepted_for_compile":
            raise ValueError("Trusted generation rejection: " + result["category"])
        require(text == result["expression"], "Candidate expression changed")
        return original(text, operation, cases)

    builder.parse_expression = gate
    return original


def build(admission_path, checksum, output, verify_only=False):
    a, sealed, diagnostic, plan, pins, e, d, builder = context(admission_path, checksum)
    output = Path(output).absolute()
    require(
        output.parent.is_dir()
        and output.parent.resolve() == output.parent
        and not output.exists()
        and not output.is_symlink(),
        "Require fresh canonical build output",
    )
    r = diagnostic["request"]
    require(
        not output.is_relative_to(Path(r["model"]))
        and not output.is_relative_to(Path(r["runtime"])),
        "Build inside model/runtime",
    )
    if verify_only:
        return {
            "status": "verified_only",
            "compiler_executed": False,
            "gpu_executed": False,
        }
    receipt = {
        "schema": "trusted-expression-build-receipt-v1",
        "status": "started",
        "inputs_sha256": pins,
        "condition": a["condition"],
        "gpu_executed": False,
        "scope": "Per-probe admission and compile preparation; no numerical or speed result.",
    }
    original = install_gate(builder, plan, a)
    try:
        builder.build(
            Path(sealed["request"]["expression_plan"]),
            Path(a["responses"]["path"]),
            output,
        )
        e.guard(d, sealed, diagnostic)
        for path, checksum0 in pins.items():
            require(digest(path) == checksum0, "Build input drift")
        job = read(output / "job.json")
        require(
            [p["id"] for p in job["probes"]] == list(e.PROBES), "Wrong built probes"
        )
        for probe in job["probes"]:
            expected = a["results"][probe["id"]]
            if expected["status"] == "rejected":
                require(
                    probe["candidate_status"] == "rejected"
                    and "candidate" not in probe,
                    "Rejected probe was compiled",
                )
            else:
                require(
                    probe["candidate_status"] in ("compiled", "compile_failed")
                    and probe["expression"] == expected["expression"],
                    "Admitted probe not built as emitted",
                )
        receipt.update(
            status="built_pending_numerical_evaluation",
            job={
                "path": str(output / "job.json"),
                "sha256": digest(output / "job.json"),
            },
            probes=[
                {
                    "id": p["id"],
                    "admission": a["results"][p["id"]]["status"],
                    "candidate_status": p["candidate_status"],
                }
                for p in job["probes"]
            ],
        )
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        builder.parse_expression = original
        # Original builder creates the directory only after its own input guards.
        if output.is_dir():
            d.write_new(output / "build-admission-receipt.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--admission", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    result = build(a.admission, a.sha256, a.output, a.verify_only)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
