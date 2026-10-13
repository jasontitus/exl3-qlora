"""Explicit fixed two-expression generation; collection only, never compile/dispatch.

Uses a sealed diagnostic campaign for exact runtime/base/adapter identities.
Malformed/truncated candidates are preserved failures and never repaired.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import signal
from pathlib import Path

PROBES = ("sum_simd_collective", "broadcast_mul_index")


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def invalid(_):
        raise ValueError("Nonfinite JSON")

    return json.loads(
        Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_launcher(request):
    if digest(request["launcher"]) != request["launcher_sha256"]:
        raise ValueError("Launcher drift")
    spec = importlib.util.spec_from_file_location(
        "expression_trusted_bindings", request["launcher"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_tree(d, value, bindings):
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            d.bind_file(value["path"], value["sha256"], bindings)
        for child in value.values():
            bind_tree(d, child, bindings)
    elif isinstance(value, list):
        for child in value:
            bind_tree(d, child, bindings)


def validate_controls(d, request, plan, bindings):
    d.require(
        set(request["controls"]) == {"golden", "negative"},
        "Both fresh M5 controls required",
    )
    expected_plan = request["expression_plan_sha256"]
    for key in ("golden", "negative"):
        item = request["controls"][key]
        report = d.read(d.bind_file(item["receipt"], item["sha256"], bindings))
        job_path = d.bind_file(item["job"], item["job_sha256"], bindings)
        job = d.read(job_path)
        d.require(
            report.get("job_sha256") == item["job_sha256"]
            and report.get("plan_sha256") == expected_plan,
            "Control report does not bind relocated job/plan",
        )
        d.require(
            report.get("device") == "Apple M5 Max"
            and report.get("gpu_executed") is True
            and report.get("bindings_verified") is True
            and report.get("reference_passed") is True
            and report.get("numerical") == "completed_fixed_cases"
            and "error" not in report,
            "Require fresh M5 control reference/dispatch success",
        )
        start, end = report.get("started_unix"), report.get("finished_unix")
        d.require(
            type(start) in (int, float)
            and type(end) in (int, float)
            and math.isfinite(start)
            and math.isfinite(end)
            and 0 < start <= end,
            "Control completion timestamps missing or unordered",
        )
        origin = (
            "source_reference_not_model"
            if key == "golden"
            else "predeclared_negative_control_not_model"
        )
        d.require(
            report.get("origin") == job.get("origin") == origin, "Wrong control origin"
        )
        d.require(
            job.get("plan_sha256") == expected_plan
            and job["plan"]["sha256"] == expected_plan
            and report.get("response_sha256") == job["responses"]["sha256"],
            "Wrong control plan",
        )
        bind_tree(d, job, bindings)
        d.require(
            job["harness_source"]["sha256"] == request["swift_source_sha256"]
            and job["builder"]["sha256"] == request["builder_sha256"],
            "Unreviewed control implementation",
        )
        cases = report.get("cases", [])
        expected = {
            (p["id"], i, variant)
            for p in plan["probes"]
            for i in range(len(p["cases"]))
            for variant in ("reference", "candidate")
        }
        actual = {(c["probe"], c["case_index"], c["variant"]) for c in cases}
        d.require(
            actual == expected and len(cases) == len(expected) == 30,
            "Incomplete control case matrix",
        )
        failures = {p["id"]: 0 for p in plan["probes"]}
        for c in cases:
            planned = next(p for p in plan["probes"] if p["id"] == c["probe"])["cases"][
                c["case_index"]
            ]
            d.require(
                c["case"] == planned
                and c["canary_mutations"] == 0
                and c["input_mutation"] is False
                and c["nonfinite_values"] == 0,
                "Control shape/guard failure",
            )
            if c["variant"] == "reference" or key == "golden":
                d.require(
                    c["passed"] is True and c["oracle_bit_mismatches"] == 0,
                    "Control numerical failure",
                )
            elif not c["passed"]:
                d.require(
                    c["oracle_bit_mismatches"] > 0,
                    "Negative control lacks numerical rejection",
                )
                failures[c["probe"]] += 1
        d.require(
            report.get("passed") is (key == "golden")
            and report.get("candidate_passed") is (key == "golden"),
            "Unexpected control outcome",
        )
        if key == "negative":
            d.require(
                all(failures.values()),
                "Both negative expressions must be discriminated",
            )


def validate_expression_plan(d, request, diagnostic, bindings):
    plan_path = d.bind_file(
        request["expression_plan"], request["expression_plan_sha256"], bindings
    )
    review = d.read(d.bind_file(request["review"], request["review_sha256"], bindings))
    plan = d.read(plan_path)
    d.require(
        review.get("status") == "approved_path_only_relocation_execution_gates_pending",
        "Missing relocation review",
    )
    d.require(
        request["expression_plan_sha256"] in review["inputs_sha256"].values(),
        "Review does not bind plan",
    )
    for field in ("builder", "swift_source"):
        d.bind_file(request[field], request[field + "_sha256"], bindings)
        d.require(
            request[field + "_sha256"] in review["inputs_sha256"].values(),
            "Unreviewed expression implementation",
        )
    d.require(
        plan.get("format") == "heldout-expression-probe-plan-v1"
        and tuple(p["id"] for p in plan["probes"]) == PROBES,
        "Require exact fixed two probes",
    )
    d.require(
        plan["model_tokenizer"] == diagnostic["request"]["model"],
        "Wrong expression tokenizer",
    )
    d.require(
        plan["generation"]["max_tokens"] == 8
        and plan["generation"]["temperature"] == 0,
        "Unreviewed decoding",
    )
    bind_tree(d, plan["tokenizer_files"], bindings)
    for p in plan["probes"]:
        for field in (
            "source_record",
            "golden_source",
            "template",
            "messages",
            "prompt_token_ids",
        ):
            d.bind_file(p[field]["path"], p[field]["sha256"], bindings)
        d.require(p["max_new_tokens"] == 8, "Unexpected probe budget")
        template = Path(p["template"]["path"]).read_bytes()
        d.require(
            template.count(b"__HOLE__") == 1
            and template.replace(b"__HOLE__", p["golden"].encode())
            == Path(p["golden_source"]["path"]).read_bytes(),
            "Template/source mismatch",
        )
    return plan


def tokenizer(diagnostic):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(
        diagnostic["request"]["model"], local_files_only=True
    )


def verify_prompt_ids(d, plan, tok):
    for p in plan["probes"]:
        messages = d.read(p["messages"]["path"])
        ids = d.read(p["prompt_token_ids"]["path"])
        d.require(
            list(
                tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_dict=False
                )
            )
            == ids,
            "Actual expression prompt tokenization differs",
        )
        d.require(len(ids) == p["prompt_tokens"] <= 1024, "Unexpected prompt length")


def prepare(request_path, checksum, output):
    if digest(request_path) != checksum:
        raise ValueError("Wrong expression request hash")
    request = read(request_path)
    d = load_launcher(request)
    bindings = {}
    d.require(
        request["schema"] == "trusted-expression-generation-request-v1", "Wrong request"
    )
    d.bind_file(request_path, checksum, bindings)
    d.bind_file(request["launcher"], request["launcher_sha256"], bindings)
    d.bind_file(__file__, digest(__file__), bindings)
    d.bind_file(request["build_wrapper"], request["build_wrapper_sha256"], bindings)
    dp = d.bind_file(
        request["diagnostic_sealed_plan"], request["diagnostic_sealed_sha256"], bindings
    )
    diagnostic = d.read(dp)
    d.guard(diagnostic)
    plan = validate_expression_plan(d, request, diagnostic, bindings)
    verify_prompt_ids(d, plan, tokenizer(diagnostic))
    validate_controls(d, request, plan, bindings)
    out = Path(request["output"]).absolute()
    d.require(
        out.parent.is_dir()
        and out.parent.resolve() == out.parent
        and not out.exists()
        and not out.is_symlink(),
        "Require new canonical output",
    )
    d.require(
        not out.is_relative_to(Path(diagnostic["request"]["runtime"]))
        and not out.is_relative_to(Path(diagnostic["request"]["model"])),
        "Output inside model/runtime",
    )
    result = {
        "schema": "trusted-expression-generation-plan-v1",
        "created_utc": d.now(),
        "request": request,
        "inputs_sha256": bindings,
        "order": [
            {"condition": role, "probe": probe} for role in d.ROLES for probe in PROBES
        ],
        "rejection_policy": "Unexpected special IDs, missing final EOS, empty or disallowed expressions are per-probe failures; keep original responses and continue all six generations once. Inconsistent provenance/token/text/stop metadata or child failures abort the campaign. No repair/retry. Each accepted probe remains eligible independently for compilation through the bound admission wrapper.",
        "execution_scope": "Six model generations; collect all original outputs and record per-probe admission. Emit gated build commands; never execute compiler or Swift GPU harness automatically. No full patch/speed/generalization claim.",
    }
    guard(d, result, diagnostic)
    d.write_new(output, result)
    return result


def guard(d, sealed, diagnostic):
    d.guard(diagnostic)
    for p, expected in sealed["inputs_sha256"].items():
        d.require(digest(p) == expected, "Expression input drift: " + p)


def command(d, diagnostic, probe, role, output):
    argv = d.generate_argv(diagnostic, role, probe["id"], output)
    argv[argv.index("--messages-file") + 1] = probe["messages"]["path"]
    return argv


def collection_command(diagnostic, request, role):
    out = Path(request["output"])
    argv = [
        diagnostic["request"]["python"],
        request["builder"],
        "collect",
        "--plan",
        request["expression_plan"],
        "--output",
        str(out / (role + "-responses.json")),
    ]
    for probe in PROBES:
        argv.extend(
            ["--receipt", probe + "=" + str(out / (role + "-" + probe + ".json"))]
        )
    return argv


def child_environment(d, diagnostic):
    return {
        "PYTHONPATH": str(Path(diagnostic["request"]["runtime"]) / "src"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        **d.ENV,
    }


def validate_generation(d, diagnostic, probe, role, result, tok, builder):
    r = diagnostic["request"]
    adapter = next(c["adapter"] for c in r["conditions"] if c["name"] == role)
    d.require(
        result.get("prompt_ids") == d.read(probe["prompt_token_ids"]["path"])
        and result.get("prompt_messages") == d.read(probe["messages"]["path"]),
        "Wrong expression prompt",
    )
    d.require(
        result.get("checkpoint") == r["model"]
        and result.get("adapter") == adapter
        and result.get("activation_mode") == r["inference"]["activation_mode"]
        and result.get("resources") == r["resources"],
        "Wrong generation condition",
    )
    d.require(
        result.get("method") == "greedy full-prefix recomputation"
        and result.get("max_new_tokens") == 8
        and result.get("tools") is None,
        "Wrong generation method",
    )
    ids = result.get("generated_ids")
    d.require(
        isinstance(ids, list)
        and 1 <= len(ids) <= 8
        and all(type(i) is int and 0 <= i < len(tok) for i in ids),
        "Invalid generated IDs",
    )
    eos = tok.eos_token_id
    stop = "eos" if ids[-1] == eos else "max_tokens"
    d.require(
        result.get("eos_token_id") == eos
        and result.get("stop_reason") == stop
        and eos not in ids[:-1]
        and (stop == "eos" or len(ids) == 8),
        "Inconsistent EOS/stop metadata",
    )
    d.require(
        result.get("text") == tok.decode(ids, skip_special_tokens=True)
        and result.get("raw_text") == tok.decode(ids, skip_special_tokens=False),
        "Decoded text differs from IDs",
    )
    if any(i in tok.all_special_ids for i in ids[:-1]) or (
        ids[-1] in tok.all_special_ids and ids[-1] != eos
    ):
        return {"status": "rejected", "category": "unexpected_special_tokens"}
    if stop != "eos":
        return {"status": "rejected", "category": "truncated"}
    expression = result["text"].strip()
    try:
        builder.parse_expression(expression, probe["operation"], probe["cases"])
    except (ValueError, SyntaxError, TypeError, KeyError) as error:
        return {
            "status": "rejected",
            "category": "malformed_expression",
            "reason": str(error),
        }
    return {
        "status": "accepted_for_compile",
        "expression": expression,
        "category": "not_yet_compiled_or_executed",
    }


def run(plan_path, checksum, verify_only=False):
    if digest(plan_path) != checksum:
        raise ValueError("Wrong expression plan hash")
    sealed = read(plan_path)
    request = sealed["request"]
    d = load_launcher(request)
    d.require(
        sealed["schema"] == "trusted-expression-generation-plan-v1", "Wrong sealed plan"
    )
    diagnostic = d.read(request["diagnostic_sealed_plan"])
    guard(d, sealed, diagnostic)
    plan = d.read(request["expression_plan"])
    tok = tokenizer(diagnostic)
    verify_prompt_ids(d, plan, tok)
    d.require(
        sealed["order"]
        == [{"condition": r, "probe": p} for r in d.ROLES for p in PROBES],
        "Wrong fixed execution order",
    )
    out = Path(request["output"])
    d.require(not out.exists() and not out.is_symlink(), "Existing/uncertain output")
    builder = d.load_module(request["builder"], "pinned_expression_grammar")
    runner = d.load_module(
        diagnostic["request"]["runner"], "expression_process_wrapper"
    )
    if verify_only:
        return {"status": "verified_only", "no_subprocesses_started": True}
    out.mkdir()
    rp = out / "launch.json"
    receipt = {
        "schema": "trusted-expression-generation-launch-v1",
        "status": "prepared",
        "created_utc": d.now(),
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": checksum,
        "processes": {},
        "results": {},
        "collections": {},
        "numerical_environment": d.ENV,
        "quality_certified": False,
    }
    d.write_new(rp, receipt)
    handlers = {
        s: signal.signal(s, runner.handle_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    r = diagnostic["request"]
    child = {
        "cwd": r["runtime"],
        "environment": child_environment(d, diagnostic),
        "project": str(out),
    }
    try:
        with d.controlled_environment(r["runtime"]) as env:
            receipt["actual_child_environment"] = dict(env)
            for role in d.ROLES:
                reports = []
                for probe in plan["probes"]:
                    guard(d, sealed, diagnostic)
                    d.require(digest(plan_path) == checksum, "Sealed plan drift")
                    key = role + "-" + probe["id"]
                    report = out / (key + ".json")
                    d.require(not report.exists(), "Existing generation report")
                    record = runner.run_child(
                        key,
                        command(d, diagnostic, probe, role, report),
                        child,
                        receipt,
                        rp,
                    )
                    guard(d, sealed, diagnostic)
                    d.require(
                        record.get("exit_code") == 0
                        and record.get("status") == "completed",
                        "Generation child failed",
                    )
                    result = validate_generation(
                        d, diagnostic, probe, role, d.read(report), tok, builder
                    )
                    result.update(
                        report=str(report),
                        sha256=digest(report),
                        process_id=record["process_id"],
                    )
                    receipt["results"][key] = result
                    reports.append((probe["id"], report))
                    runner.persist(rp, receipt)
                response = out / (role + "-responses.json")
                argv = collection_command(diagnostic, request, role)
                guard(d, sealed, diagnostic)
                record = runner.run_child(role + "-collect", argv, child, receipt, rp)
                guard(d, sealed, diagnostic)
                d.require(
                    record.get("exit_code") == 0
                    and record.get("status") == "completed",
                    "Collection child failed",
                )
                value = d.read(response)
                d.require(
                    value.get("origin") == "model_generation"
                    and value.get("plan_sha256") == request["expression_plan_sha256"]
                    and set(value.get("responses", {})) == set(PROBES),
                    "Wrong collection",
                )
                for probe, report in reports:
                    answer = value["responses"][probe]
                    d.require(
                        answer["generation_receipt"]
                        == {
                            "path": str(report.resolve()),
                            "sha256": receipt["results"][role + "-" + probe]["sha256"],
                        }
                        and answer["text"] == d.read(report)["text"],
                        "Collection changed generated text/receipt",
                    )
                admission = out / (role + "-admission.json")
                d.write_new(
                    admission,
                    {
                        "schema": "trusted-expression-build-admission-v1",
                        "expression_launcher": {
                            "path": str(Path(__file__).resolve()),
                            "sha256": digest(__file__),
                        },
                        "build_wrapper": {
                            "path": request["build_wrapper"],
                            "sha256": request["build_wrapper_sha256"],
                        },
                        "sealed_plan": {
                            "path": str(Path(plan_path).resolve()),
                            "sha256": checksum,
                        },
                        "responses": {
                            "path": str(response),
                            "sha256": digest(response),
                        },
                        "condition": role,
                        "launch_receipt": str(rp),
                        "results": {
                            p: receipt["results"][role + "-" + p] for p in PROBES
                        },
                        "scope": "Per-probe admission only; neither compilation nor numerical execution has happened.",
                    },
                )
                receipt["collections"][role] = {
                    "path": str(response),
                    "sha256": digest(response),
                    "admission": {"path": str(admission), "sha256": digest(admission)},
                    "build_command_not_executed": [
                        r["python"],
                        request["build_wrapper"],
                        "--admission",
                        str(admission),
                        "--sha256",
                        digest(admission),
                        "--output",
                        str(out / (role + "-build")),
                    ],
                }
                runner.persist(rp, receipt)
            guard(d, sealed, diagnostic)
            d.require(digest(plan_path) == checksum, "Final sealed plan drift")
            for record in receipt["processes"].values():
                for name, checksum0 in record["log_sha256"].items():
                    d.require(digest(record[name]) == checksum0, "Child log drift")
            for result in receipt["results"].values():
                d.require(
                    digest(result["report"]) == result["sha256"],
                    "Generated report drift",
                )
            for result in receipt["collections"].values():
                d.require(
                    digest(result["path"]) == result["sha256"],
                    "Collected response drift",
                )
                d.require(
                    digest(result["admission"]["path"])
                    == result["admission"]["sha256"],
                    "Build admission drift",
                )
            d.require(
                len(receipt["results"]) == 6, "Incomplete fixed generation matrix"
            )
            receipt.update(
                status="completed_pending_independent_audit",
                accepted_responses=sum(
                    x["status"] == "accepted_for_compile"
                    for x in receipt["results"].values()
                ),
                rejected_responses=sum(
                    x["status"] == "rejected" for x in receipt["results"].values()
                ),
                compiled=False,
                kernel_gpu_executed=False,
            )
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        receipt["finished_utc"] = d.now()
        runner.persist(rp, receipt)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
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
    result = (
        prepare(a.request, a.sha256, a.output)
        if a.action == "prepare"
        else run(a.plan, a.sha256, a.verify_only)
    )
    print(json.dumps({"status": result.get("status", "sealed")}))


if __name__ == "__main__":
    main()
