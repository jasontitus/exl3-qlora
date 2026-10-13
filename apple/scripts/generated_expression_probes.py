#!/usr/bin/env python3
"""CPU-only preparation/compilation for bounded held-out expression probes.

This script never evaluates model-generated code on the CPU or starts a Metal
GPU workload. The separate Swift verifier executes the resulting libraries.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
CORPUS = STAGE / "artifacts/loss-fix/kernel-corpus-v1"
MODEL = Path("/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw")
SUM_CASES = [
    {"n": n, "threads": t}
    for n, t in [
        (0, 32),
        (1, 32),
        (17, 32),
        (32, 32),
        (33, 64),
        (65, 64),
        (511, 128),
        (1025, 256),
        (4097, 128),
    ]
]
ROW_CASES = [
    {"vectors_per_row": n, "rows": m} for n, m in [(1, 1), (1, 7), (3, 2), (7, 5), (17, 3), (31, 2)]
]
SPECS = [
    {
        "id": "sum_simd_collective",
        "case_prefix": "f4ce81c45e7b",
        "function": "kernel_op_sum_f32",
        "anchor": "sumf = simd_sum(sumf);",
        "golden": "simd_sum(sumf)",
        "negative_control": "sumf",
        "operation": "sum",
        "instruction": "Complete the missing floating-point expression for a global sum. Each thread has accumulated its strided input into sumf; lane zero of each SIMD group then writes its group partial to shared memory. The rest of the kernel is fixed. Use a pure scalar expression with sumf and, if needed, simd_sum, simd_min, or simd_max. Return only the expression replacing __HOLE__, without a semicolon, Markdown, or explanation.",
        "cases": SUM_CASES,
    },
    {
        "id": "broadcast_mul_index",
        "case_prefix": "13268c533177",
        "function": "kernel_mul_row",
        "anchor": "src1[tpig % nb]",
        "golden": "tpig % nb",
        "negative_control": "0",
        "operation": "broadcast_mul",
        "instruction": "Complete the missing index expression for float4 row-broadcast multiplication. src0 and dst contain rows of nb float4 values; src1 contains one row reused for every row. tpig is the global float4 index and is less than rows*nb; nb is positive. Return only the expression replacing __HOLE__, using tpig, nb, nonnegative integer constants, parentheses, +, *, or %, without a semicolon, Markdown, or explanation.",
        "cases": ROW_CASES,
    },
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(x):
    return json.dumps(
        x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def binding(p):
    return {"path": str(Path(p).resolve()), "sha256": sha(Path(p).read_bytes())}


def checked(b):
    p = Path(b["path"])
    raw = p.read_bytes()
    if sha(raw) != b["sha256"]:
        raise ValueError(f"Binding changed: {p}")
    return raw


def tokenizer(root):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(str(root), local_files_only=True)


def tokenizer_bindings(root):
    return {
        name: binding(root / name)
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    }


def parse_expression(text, operation, cases):
    """Reject side effects and prove every broadcast index before compilation."""
    if (
        not isinstance(text, str)
        or not text
        or len(text) > 128
        or text != text.strip()
        or "\n" in text
        or "\r" in text
    ):
        raise ValueError("Expected one stripped expression <=128 characters")
    if not re.fullmatch(r"[A-Za-z0-9_+*%()., /-]+", text):
        raise ValueError("Expression contains unsupported syntax")
    parsed = ast.parse(text, mode="eval")
    if len(list(ast.walk(parsed))) > 40:
        raise ValueError("Expression too complex")

    def walk(node, values):
        if isinstance(node, ast.Expression):
            return walk(node.body, values)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or not 0 <= node.value <= 1000000:
                raise ValueError("Numeric literal out of range")
            if operation == "broadcast_mul" and type(node.value) is not int:
                raise ValueError("Index needs integers")
            return node.value
        allowed = (
            (ast.Add, ast.Mult, ast.Mod)
            if operation == "broadcast_mul"
            else (ast.Add, ast.Sub, ast.Mult, ast.Div)
        )
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed):
            a, b = walk(node.left, values), walk(node.right, values)
            if isinstance(node.op, (ast.Div, ast.Mod)) and b == 0:
                raise ValueError("Zero divisor")
            value = (
                a + b
                if isinstance(node.op, ast.Add)
                else a - b
                if isinstance(node.op, ast.Sub)
                else a * b
                if isinstance(node.op, ast.Mult)
                else a % b
                if isinstance(node.op, ast.Mod)
                else a / b
            )
            if operation == "broadcast_mul" and not 0 <= value < 2**31:
                raise ValueError("Index intermediate exceeds nonnegative signed32 range")
            return value
        if (
            operation == "sum"
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("simd_sum", "simd_min", "simd_max")
            and len(node.args) == 1
            and not node.keywords
        ):
            # Static grammar validation only; the Metal collective runs on GPU.
            walk(node.args[0], values)
            return 1.0
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    if operation == "broadcast_mul":
        for c in cases:
            n = c["vectors_per_row"]
            for i in range(n * c["rows"]):
                value = walk(parsed, {"tpig": i, "nb": n})
                if not 0 <= value < n:
                    raise ValueError(f"Index outside source row at tpig={i}, nb={n}")
    else:
        walk(parsed, {"sumf": 1.0})
    return parsed


def prepare(output, model):
    output.mkdir(parents=True, exist_ok=False)
    tok = tokenizer(model)
    probes = []
    for spec in SPECS:
        original = next(
            (CORPUS / "llama/reviewed-records").glob("*" + spec["case_prefix"] + "*.json")
        )
        snapshot = output / (spec["id"] + "-source-record.json")
        shutil.copyfile(original, snapshot)
        r = read_json(snapshot)["source_record"]
        if not r["protected"]:
            raise ValueError("Probe source must be held out")
        allunits = [
            (path, u) for path, units in r["reviewed_context"]["units"].items() for u in units
        ]
        path, kernel = next(
            (p, u)
            for p, u in allunits
            if re.search(r"kernel void " + spec["function"] + r"\(", u["after_content"])
        )
        pieces = []
        for literal in ["#include <metal_stdlib>", "using namespace metal;"]:
            content = r["files"][path]["after"]["content"]
            start = content.index(literal)
            pieces.append(
                {
                    "path": path,
                    "start": start,
                    "end": start + len(literal),
                    "sha256": sha(literal.encode()),
                    "content": literal,
                }
            )
        if spec["operation"] == "sum":
            p, u = next(
                (p, u)
                for p, u in allunits
                if u["kind"] == "complete_type" and "ggml_metal_kargs_sum;" in u["after_content"]
            )
            pieces.append({"path": p, **u["after"], "content": u["after_content"]})
        pieces.append({"path": path, **kernel["after"], "content": kernel["after_content"]})
        for piece in pieces:
            raw = r["files"][piece["path"]]["after"]["content"].encode()
            b = raw[piece["start"] : piece["end"]]
            if b != piece["content"].encode() or sha(b) != piece["sha256"]:
                raise ValueError("Exact source composition mismatch")
        source = "\n\n".join(x["content"] for x in pieces) + "\n"
        if source.count(spec["anchor"]) != 1:
            raise ValueError("Hole anchor must be unique")
        offset = source.index(spec["anchor"]) + spec["anchor"].index(spec["golden"])
        template = source[:offset] + "__HOLE__" + source[offset + len(spec["golden"]) :]
        if template.replace("__HOLE__", spec["golden"]) != source:
            raise ValueError("Hole must reconstruct original source")
        folder = output / spec["id"]
        folder.mkdir()
        (folder / "golden.metal").write_text(source)
        (folder / "template.metal").write_text(template)
        prompt = (
            spec["instruction"]
            + "\n\nComplete standalone Metal kernel:\n```metal\n"
            + template
            + "```"
        )
        messages = [{"role": "user", "content": prompt}]
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=False))
        if len(ids) > 1024 or len(tok.encode(spec["golden"], add_special_tokens=False)) > 8:
            raise ValueError("Probe exceeds declared token budget")
        write_json(folder / "messages.json", messages)
        write_json(folder / "prompt-token-ids.json", ids)
        probes.append(
            {
                **spec,
                "source_record": binding(snapshot),
                "commit": r["head"]["commit"],
                "parent": r["parent"]["commit"],
                "repository": r["repository"],
                "original_case_id": r["case_id"],
                "source_composition": pieces,
                "source_files_sha256": {
                    p: r["files"][p]["after"]["sha256"] for p in {x["path"] for x in pieces}
                },
                "golden_source": binding(folder / "golden.metal"),
                "template": binding(folder / "template.metal"),
                "messages": binding(folder / "messages.json"),
                "prompt_token_ids": binding(folder / "prompt-token-ids.json"),
                "prompt_tokens": len(ids),
                "golden_expression_tokens": len(
                    tok.encode(spec["golden"], add_special_tokens=False)
                ),
                "exact_hole_utf8_offset": len(source[:offset].encode()),
                "max_new_tokens": 8,
            }
        )
    plan = {
        "format": "heldout-expression-probe-plan-v1",
        "declared_before_model_scores": True,
        "model_tokenizer": str(model),
        "tokenizer_files": tokenizer_bindings(model),
        "selection": "Two original held-out source operations selected before generation; no optimization/timing criterion based on model output. Flawed normalization and short-row vector-softmax sources excluded after source audit.",
        "generation": {
            "temperature": 0,
            "max_tokens": 8,
            "messages_template": "default checkpoint chat template with add_generation_prompt=True, no tools",
            "policy": "One deterministic completion per probe per model; no retries or best-of-N. Store exact token IDs and raw decoded text; only surrounding whitespace may be stripped for expression parsing.",
        },
        "probes": probes,
        "numerical": {
            "sum": "exact FP32 bit comparison to independent integer/dyadic CPU sum for fixed cases; zero elements leave destination unchanged",
            "broadcast_mul": "exact FP32 bit comparison to independent dyadic scalar CPU broadcast multiplication",
            "guards": "exact prefix/tail canary bits and immutable input buffers",
        },
        "limitations": [
            "Correctness of a single generated expression in an otherwise provided optimized kernel; not optimization discovery or full edit generation.",
            "No timing/speed gate or full upstream repository build.",
            "Two tasks are not statistically powered generalization evidence.",
            "Known public source may have appeared in base-model pretraining.",
            "Scope is the predeclared finite shapes/values and compiler flags, not all possible inputs.",
        ],
    }
    write_json(output / "plan.json", plan)
    write_json(
        output / "golden-responses.json",
        {
            "format": "heldout-expression-responses-v1",
            "origin": "source_reference_not_model",
            "plan_sha256": sha((output / "plan.json").read_bytes()),
            "responses": {p["id"]: {"text": p["golden"]} for p in probes},
        },
    )
    write_json(
        output / "negative-control-responses.json",
        {
            "format": "heldout-expression-responses-v1",
            "origin": "predeclared_negative_control_not_model",
            "plan_sha256": sha((output / "plan.json").read_bytes()),
            "responses": {p["id"]: {"text": p["negative_control"]} for p in probes},
        },
    )
    print(
        json.dumps(
            {
                "plan": str(output / "plan.json"),
                "sha256": sha((output / "plan.json").read_bytes()),
                "probes": [
                    {k: p[k] for k in ["id", "prompt_tokens", "golden_expression_tokens"]}
                    for p in probes
                ],
            },
            indent=2,
        )
    )


def compile_source(source, folder):
    folder.mkdir()
    inp = folder / "source.metal"
    inp.write_bytes(source)
    air = folder / "source.air"
    lib = folder / "source.metallib"
    commands = []
    for argv in [
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            "-std=metal3.1",
            "-fmodules-cache-path=/tmp/exl3-expression-metal-cache",
            "-O3",
            "-fno-fast-math",
            "-c",
            str(inp),
            "-o",
            str(air),
        ],
        ["xcrun", "--sdk", "macosx", "metal", str(air), "-o", str(lib)],
    ]:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        commands.append(
            {
                "argv": argv,
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        if proc.returncode:
            break
    passed = len(commands) == 2 and commands[-1]["exit_code"] == 0
    rec = {"source": binding(inp), "commands": commands, "passed": passed}
    if air.exists():
        rec["air"] = binding(air)
    if passed:
        rec["library"] = binding(lib)
    write_json(folder / "compile.json", rec)
    return rec


def build(plan_path, responses_path, output):
    raw_plan = plan_path.read_bytes()
    plan = json.loads(raw_plan)
    response_raw = responses_path.read_bytes()
    responses = json.loads(response_raw)
    if (
        plan["format"] != "heldout-expression-probe-plan-v1"
        or responses["format"] != "heldout-expression-responses-v1"
        or responses["plan_sha256"] != sha(raw_plan)
    ):
        raise ValueError("Plan/response identity mismatch")
    if set(responses["responses"]) != {p["id"] for p in plan["probes"]}:
        raise ValueError("Every planned probe needs exactly one response")
    for b in plan["tokenizer_files"].values():
        checked(b)
    if responses.get("origin") not in {
        "source_reference_not_model",
        "predeclared_negative_control_not_model",
        "model_generation",
    }:
        raise ValueError("Unknown response origin")
    tok = tokenizer(Path(plan["model_tokenizer"]))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, output / "plan.json")
    shutil.copyfile(responses_path, output / "responses.json")
    result = []
    for p in plan["probes"]:
        checked(p["source_record"])
        checked(p["messages"])
        checked(p["prompt_token_ids"])
        template = checked(p["template"]).decode()
        golden = checked(p["golden_source"])
        if (
            template.count("__HOLE__") != 1
            or template.replace("__HOLE__", p["golden"]).encode() != golden
        ):
            raise ValueError("Golden/template mismatch")
        folder = output / p["id"]
        folder.mkdir()
        reference = compile_source(golden, folder / "reference")
        if not reference["passed"]:
            raise ValueError("Reference compilation failed")
        entry = {
            "id": p["id"],
            "operation": p["operation"],
            "function": p["function"],
            "cases": p["cases"],
            "reference": reference,
            "candidate_status": "rejected",
            "source_reference_not_a_model_result": responses["origin"] != "model_generation",
        }
        try:
            answer = responses["responses"][p["id"]]
            text = answer["text"]
            expression = text.strip()
            if responses["origin"] == "model_generation":
                generation_raw = checked(answer["generation_receipt"])
                generation = json.loads(generation_raw)
                ids = generation["generated_ids"]
                if (
                    not isinstance(ids, list)
                    or not 1 <= len(ids) <= 8
                    or any(type(i) is not int or i < 0 or i >= len(tok) for i in ids)
                ):
                    raise ValueError("Invalid emitted token IDs")
                if (
                    generation.get("max_new_tokens") != 8
                    or generation.get("method") != "greedy full-prefix recomputation"
                    or generation.get("tools") is not None
                ):
                    raise ValueError("Generation settings differ from protocol")
                if generation.get("prompt_ids") != json.loads(
                    checked(p["prompt_token_ids"])
                ) or generation.get("prompt_messages") != json.loads(checked(p["messages"])):
                    raise ValueError("Generation prompt differs from protocol")
                if (
                    text != generation["text"]
                    or text != tok.decode(ids, skip_special_tokens=True)
                    or generation["raw_text"] != tok.decode(ids, skip_special_tokens=False)
                ):
                    raise ValueError("Expression differs from exact generated tokens")
                snapshot = folder / "generation.json"
                snapshot.write_bytes(generation_raw)
                entry["generation_receipt"] = binding(snapshot)
                entry["generated_ids"] = ids
            else:
                expected = (
                    p["golden"]
                    if responses["origin"] == "source_reference_not_model"
                    else p["negative_control"]
                )
                if text != expected:
                    raise ValueError("Nonmodel control differs from the predeclared expression")
                if len(tok.encode(text, add_special_tokens=False)) > 8:
                    raise ValueError("Control exceeds eight-token budget")
            parse_expression(expression, p["operation"], p["cases"])
            candidate = compile_source(
                template.replace("__HOLE__", expression).encode(), folder / "candidate"
            )
            entry.update(
                expression=expression,
                raw_text=text,
                candidate=candidate,
                candidate_status="compiled" if candidate["passed"] else "compile_failed",
            )
        except (ValueError, SyntaxError, TypeError, KeyError) as exc:
            entry["error"] = str(exc)
        result.append(entry)
    compiler = subprocess.run(
        ["xcrun", "--sdk", "macosx", "metal", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    harness = output / "verify_expression_probes.swift"
    shutil.copyfile(STAGE / "scripts/verify_expression_probes.swift", harness)
    binary = output / "verify_expression_probes"
    args = [
        "xcrun",
        "swiftc",
        "-module-cache-path",
        "/tmp/exl3-expression-swift-module-cache",
        str(harness),
        "-o",
        str(binary),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    write_json(
        output / "harness-compile.json",
        {"argv": args, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
    )
    if proc.returncode:
        raise ValueError("Swift verifier compilation failed")
    builder_copy = output / "builder.py"
    shutil.copyfile(Path(__file__), builder_copy)
    job = {
        "format": "heldout-expression-job-v1",
        "plan": binding(output / "plan.json"),
        "responses": binding(output / "responses.json"),
        "plan_sha256": sha(raw_plan),
        "origin": responses["origin"],
        "probes": result,
        "compiler": compiler,
        "harness_source": binding(harness),
        "harness_binary": binding(binary),
        "builder": binding(builder_copy),
        "scope": "Compile-only preparation; no GPU result yet.",
    }
    write_json(output / "job.json", job)
    print(
        json.dumps(
            {
                "job": str(output / "job.json"),
                "sha256": sha((output / "job.json").read_bytes()),
                "compiled_candidates": sum(p["candidate_status"] == "compiled" for p in result),
            },
            indent=2,
        )
    )


def self_test():
    for spec in SPECS:
        parse_expression(spec["golden"], spec["operation"], spec["cases"])
        parse_expression(spec["negative_control"], spec["operation"], spec["cases"])
    bad = [
        "tpig",
        "nb",
        "tpig-1",
        "tpig/nb",
        "1%0",
        "tpig + 2147483647",
        "src0[0]",
        "foo(tpig)",
        "tpig; return",
        "tpig\n%nb",
        "__import__(1)",
    ]
    for expression in bad:
        try:
            parse_expression(expression, "broadcast_mul", ROW_CASES)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError(expression)
    for expression in [
        "simd_sum(sumf);",
        "barrier(sumf)",
        "simd_sum(src0[0])",
        "sumf = 0",
        "simd_sum(sumf, 2)",
    ]:
        try:
            parse_expression(expression, "sum", SUM_CASES)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError(expression)
    print("20 expression parser/range checks passed; no GPU")


def collect(plan_path, receipt_items, output):
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    responses = {}
    for item in receipt_items:
        key, separator, value = item.partition("=")
        if not separator or key in responses:
            raise ValueError("Use one unique PROBE_ID=GENERATION_REPORT.json per probe")
        path = Path(value).expanduser().resolve(strict=True)
        generation = read_json(path)
        responses[key] = {"text": generation["text"], "generation_receipt": binding(path)}
    if set(responses) != {p["id"] for p in plan["probes"]}:
        raise ValueError("Need exactly the predeclared probe IDs")
    result = {
        "format": "heldout-expression-responses-v1",
        "origin": "model_generation",
        "plan_sha256": sha(plan_raw),
        "responses": responses,
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print("Collected receipts; build performs full prompt/token/source validation")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="command", required=True)
    p = sp.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=MODEL)
    p = sp.add_parser("build")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sp.add_parser("collect")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--receipt", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    sp.add_parser("self-test")
    a = ap.parse_args()
    if a.command == "prepare":
        prepare(a.output.resolve(), a.model.resolve())
    elif a.command == "build":
        build(a.plan.resolve(), a.responses.resolve(), a.output.resolve())
    elif a.command == "collect":
        collect(a.plan.resolve(), a.receipt, a.output.resolve())
    else:
        self_test()
    assert "mlx" not in sys.modules


if __name__ == "__main__":
    main()
