"""Prepare complete historical edit tasks; no upstream execution or GPU imports."""

import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from curation import DECISIONS

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / "loss-fix/apple/src"))
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity  # noqa: E402
from exl3_mlx.upstream_data import FORMAT, _evidence, reconstruct_candidate  # noqa: E402


def imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


units = imp("reviewed_units", WORK / "kernel-library/expansion-v2/group-b/refined_units.py")
agent_example = imp(
    "hour_exact_renderer", WORK / "kernel-library/expansion-v3-hour/tools/exact_edit_renderer.py"
).agent_example


def H(b):
    return hashlib.sha256(b).hexdigest()


def D(x):
    return json.dumps(
        x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")


def fullunit(path, pair):
    b = pair["before"]["content"].encode() if pair["before"] else b""
    a = pair["after"]["content"].encode() if pair["after"] else b""
    return [
        {
            "path": path,
            "kind": "complete_declaration",
            "before": units.bind(b, 0, len(b)),
            "after": units.bind(a, 0, len(a)),
        }
    ]


C_SELECT = {
    ("789be76e", "mlx/backend/metal/scaled_dot_product_attention.cpp"): [
        ("void sdpa_vector_2pass(", False)
    ],
    ("789be76e", "mlx/backend/metal/kernels/sdpa_vector.h"): [("void sdpa_vector_2pass_1(", True)],
    ("a1316145", "mlx/backend/metal/kernels/rms_norm.metal"): [("void rms_single_row(", True)],
    ("494c8a6c", "mlx/backend/metal/kernels/fp_quantized.h"): [
        (n, True)
        for n in [
            "void fp_qmm_t_impl(",
            "void fp_qmm_t(",
            "void fp_gather_qmm_t(",
            "void fp_qmm_t_splitk(",
        ]
    ],
    ("494c8a6c", "mlx/backend/metal/kernels/quantized.h"): [
        (n, True)
        for n in [
            "void qmm_t_impl(",
            "void affine_qmm_t(",
            "void affine_gather_qmm_t(",
            "void affine_qmm_t_splitk(",
        ]
    ],
    ("494c8a6c", "mlx/backend/metal/quantized.cpp"): [("void qmm_splitk(", False)],
    ("02adf7b2", "mlx/ops.cpp"): [("std::vector<array> fp_quantize(", False)],
    ("02adf7b2", "mlx/backend/cpu/quantized.cpp"): [("void fp_quantize_dequantize(", True)],
    ("02adf7b2", "mlx/backend/metal/kernels/fp_quantized.h"): [
        ("void fp_quantize(", True),
        ("void fp_quantize_dequantize(", True),
    ],
}


def c_unit(text, sig, template):
    a, b = units.function_range(text, sig)
    if template:
        raw = text.encode()
        code, _, _ = units._source_units.lex(raw, "c")
        start = code.rfind("template <", 0, a)
        if start < 0:
            raise ValueError("Missing reviewed template header")
        gap = code[start:a]
        if "}" in gap or ";" in gap:
            raise ValueError("Intervening declaration before template function")
        a = code.rfind("\n", 0, start) + 1
    return a, b


def main():
    inventory = json.loads((ROOT / "inventory.json").read_text())["records"]
    reserved_path = WORK / "kernel-library/integration-v1/group-a/evidence/reserved-metadata.json"
    reserved = json.loads(reserved_path.read_text())["review"]
    ids = {v[k] for v in reserved["coverage"] for k in ("head", "base") if v.get(k)}
    reserved_prs = {v["number"] for v in reserved["reserved_pr_lineage"] if v["repo"] == "mlx"} | {
        n
        for v in reserved["reserved_pr_lineage"]
        if v["repo"] == "mlx"
        for n in v["referenced_prs"]
    }
    ev = {}
    for row in inventory:
        pr = row["pr"]
        key = f"{row['repo']}-pr-{pr['number']}"
        ev[key] = {
            "path": pr["body_path"],
            "sha256": pr["body_sha256"],
            "phase": "post_change",
            "url": pr["url"],
        }
    for row in inventory:
        ev[row["repo"] + "-" + row["commit"][:8] + "-diff"] = {
            "path": row["patch"]["path"],
            "sha256": row["patch"]["sha256"],
            "phase": "post_change",
            "url": row["url"],
        }
    repos = {
        r: {"path": str(ROOT / "mirrors" / f"{r}.git"), "url": "https://github.com/" + u}
        for r, u in {
            "mlx": "ml-explore/mlx",
            "mlx-lm": "ml-explore/mlx-lm",
            "mlx-vlm": "Blaizzy/mlx-vlm",
        }.items()
    }
    sources, _ = _evidence(ROOT, ev)
    tkpath = "/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
    tk = load_tokenizer(tkpath)
    candidates = []
    cases = []
    docs = {}
    out = ROOT / "reviewed"
    for row in inventory:
        key = row["commit"][:8]
        outcome, kind, operation, rationale = DECISIONS[key]
        pr = row["pr"]
        ek = f"{row['repo']}-pr-{pr['number']}"
        body = (ROOT / pr["body_path"]).read_bytes().decode()
        hit = [x for x in (row["commit"], row["parent"]) if x in ids]
        ref_hit = pr["number"] in reserved_prs if row["repo"] == "mlx" else False
        if hit or ref_hit:
            outcome = "protected"
        case = {
            "id": f"{row['repo']}-{row['commit'][:12]}-hour-v3",
            "repo": repos[row["repo"]]["url"],
            "commit": row["commit"],
            "parent": row["parent"],
            "lineage": row["repo"] + ":" + row["commit"],
            "source_group": "mlx-derived-framework-training",
            "task_kind": kind,
            "outcome_class": outcome,
            "licenses": row["licenses"],
            "exact_reserved_matches": hit,
            "reserved_pr_match": ref_hit,
            "source_evidence": ek,
        }
        if outcome in {"pending", "protected"}:
            case.update(status="excluded_protection", reason=rationale)
            cases.append(case)
            continue
        reason = (
            "Reviewed exact head/parent IDs and reserved PR/reference IDs against the pinned ledger, then inspected actual changed operation. This case is "
            + operation.lower()
            + ". Finite public-source/metadata review distinguishes the selected changed operation from reserved tasks; shared framework primitives and motifs remain. The bound finite-operation report describes the quantization, RMSNorm and SDPA cases explicitly. Related MLX framework and GDN/attention motifs remain shared training lineage; no independent-family efficacy or global semantic nonoverlap is claimed."
        )

        def ann(text):
            return {
                "text": text,
                "basis": "source",
                "evidence": [{"source": ek, "quote": body if body.strip() else row["message"]}],
            }

        # Minimal PRs use exact commit message as their source annotation.
        if not body.strip():
            ek = "commit_message"
        c = {
            "id": case["id"],
            "repository": row["repo"],
            "commit": row["commit"],
            "parent": row["parent"],
            "outcome": "reported_improvement" if outcome == "positive" else "source_only",
            "protected": False,
            "source_group": case["source_group"],
            "task_conditioning": "historical_source",
            "task_kind": {
                "value": kind,
                "reviewed_by": "adversarial_code source review",
                "evidence": [{"source": ek, "quote": body if body.strip() else row["message"]}],
            },
            "annotations": {"operation": ann(operation), "architecture": ann(rationale)},
            "historical_protection_review": {
                "status": "cleared",
                "reviewed_by": "adversarial_code",
                "reason": reason,
                "evidence_sha256": {
                    str(reserved_path.relative_to(ROOT))
                    if reserved_path.is_relative_to(ROOT)
                    else str(reserved_path): H(reserved_path.read_bytes()),
                    str(ROOT / "evidence/finite-operation-review.json"): H(
                        (ROOT / "evidence/finite-operation-review.json").read_bytes()
                    ),
                },
            },
        }
        c["annotations"]["architecture"]["evidence"].append(
            {
                "source": row["repo"] + "-" + key + "-diff",
                "quote": (ROOT / row["patch"]["path"]).read_bytes().decode(),
            }
        )
        try:
            rec = reconstruct_candidate(c, repos, sources)
            selected = []
            for path, pair in rec["files"].items():
                if key == "83b976ea" and path in {
                    "mlx/backend/metal/kernels/CMakeLists.txt",
                    "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h",
                    "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal",
                }:
                    continue
                sigs = {
                    ("83b976ea", "mlx/backend/metal/scaled_dot_product_attention.cpp"): [
                        "void sdpa_full_self_attention_metal(",
                        "void ScaledDotProductAttention::eval_gpu(",
                        "void ScaledDotProductAttentionVJP::eval_gpu(",
                    ],
                    ("0fdcdc0e", "mlx/fast.cpp"): [
                        "array layer_norm(",
                        "array scaled_dot_product_attention(",
                    ],
                    ("f5cffe94", "mlx/transforms.cpp"): ["array eval_impl("],
                }.get((key, path))
                special = C_SELECT.get((key, path))
                if special:
                    for sig, templ in special:
                        b = pair["before"]["content"]
                        a = pair["after"]["content"]
                        b0, b1 = c_unit(b, sig, templ)
                        a0, a1 = c_unit(a, sig, templ)
                        selected.append(
                            {
                                "path": path,
                                "kind": "complete_function",
                                "before": units.bind(b.encode(), b0, b1),
                                "after": units.bind(a.encode(), a0, a1),
                            }
                        )
                elif sigs:
                    for sig in sigs:
                        b = pair["before"]["content"]
                        a = pair["after"]["content"]
                        b0, b1 = units.function_range(b, sig)
                        a0, a1 = units.function_range(a, sig)
                        selected.append(
                            {
                                "path": path,
                                "kind": "complete_function",
                                "before": units.bind(b.encode(), b0, b1),
                                "after": units.bind(a.encode(), a0, a1),
                            }
                        )
                elif path.endswith(".py") and pair["before"] and pair["after"]:
                    try:
                        selected.extend(units.python_units(path, pair))
                    except (SyntaxError, ValueError, IndexError):
                        selected.extend(fullunit(path, pair))
                else:
                    selected.extend(fullunit(path, pair))
            c["context_policy"] = "reviewed_units"
            c["reviewed_units"] = {
                "reviewed_by": "adversarial_code",
                "dependency_review": "Complete changed Python methods/declarations or complete changed C++/Metal/build files. Every source, host, test and build edit remains in the exact full non-documentation patch target. Unchanged includes and repository dependencies are not asserted to be standalone-compilable. CPU replay only.",
                "units": selected,
            }
            if key == "83b976ea":
                c["context_policy"] = "reviewed_operation"
                c["operation_scope"] = {
                    "description": "Tiled SDPA backward integration with complete classic-attention and selected host functions as prompt context. All changed NAX sibling, host, benchmark and build code remains in the full exact non-documentation patch target and baseline edit anchors. No standalone dependency-closure claim.",
                    "omitted_path_reasons": {
                        p: "Prompt selects complete classic-attention and VJP host units; sibling NAX/build/other host changes remain in the full exact patch target and baseline anchors rather than duplicate complete baseline files."
                        for p in [
                            "mlx/backend/metal/kernels/CMakeLists.txt",
                            "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h",
                            "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.metal",
                            "mlx/backend/metal/scaled_dot_product_attention.cpp",
                        ]
                    },
                }
            if key == "02adf7b2":
                c["context_policy"] = "reviewed_operation"
                c["operation_scope"] = {
                    "description": "MX scale rounding with complete quantization functions as prompt context. All changed CPU/Metal helpers and source headers remain in the exact full target patch and baseline anchors.",
                    "omitted_path_reasons": {
                        p: "The complete quantization function is shown; newly inserted scale helper or copyright metadata remains in the exact target patch and baseline anchors."
                        for p in ["mlx/backend/cpu/quantized.cpp", "mlx/ops.cpp"]
                    },
                }
            rec = reconstruct_candidate(c, repos, sources)
            docs[c["id"]] = {
                p: {
                    "before_sha256": v["before"]["sha256"] if v["before"] else None,
                    "after_sha256": v["after"]["sha256"] if v["after"] else None,
                    "reason": "Human-facing documentation omitted from target; full source diff remains captured. Source, tests, host and build edits are unchanged.",
                }
                for p, v in rec["files"].items()
                if Path(p).suffix.lower() in {".md", ".rst"}
            }
            raw, replay = agent_example(rec, docs[c["id"]])
            # Repair tasks must not masquerade as generic speed optimization.
            msg = next(m for m in raw["messages"] if m["role"] == "user")
            msg["content"] = (
                "Task: "
                + operation
                + ".\nMake the necessary complete source, host, build and test changes with exec_command and git apply. Preserve supported shapes, dtype semantics and the host/kernel ABI except for the requested repair. This is a historical repository-edit task; do not invent profiling, compiler or runtime results.\n\n"
                + msg["content"].split("\n\n", 1)[1]
            )
            raw = json.loads(D(raw))
            tokens = tokenize_row(raw, tk)
            for suffix, obj in [
                ("source", rec),
                ("raw", raw),
                ("replay", replay),
                ("prepared", tokens),
            ]:
                save(out / f"{c['id']}.{suffix}.json", obj)
            n = len(tokens["input_ids"])
            case.update(
                stable_patch_id=rec["stable_patch_id"],
                exact_edit_eligible=key != "b08ec310",
                exact_edit_exclusion_reason="Historical std::async timeout can still block at future destruction; reasoning review only."
                if key == "b08ec310"
                else None,
                status="reviewed_exact_edit",
                total_serialized_ids=n,
                supervised_tokens=n - tokens["loss_start"],
                fits_12k=n <= 12288,
                fits_32k=n <= 32768,
                exact_replay=True,
                protection_review=c["historical_protection_review"],
                paths={
                    s: str(out / f"{c['id']}.{s}.json")
                    for s in ["source", "raw", "replay", "prepared"]
                },
                files_sha256={
                    s: H((out / f"{c['id']}.{s}.json").read_bytes())
                    for s in ["source", "raw", "replay", "prepared"]
                },
            )
            candidates.append(c)
        except Exception as exc:
            case.update(status="source_or_replay_error", error=repr(exc))
        cases.append(case)
        save(ROOT / "progress.json", cases)
        print(
            case["id"],
            case["status"],
            case.get("total_serialized_ids", case.get("error")),
            flush=True,
        )
    manifest = {
        "format": FORMAT,
        "repositories": repos,
        "evidence": ev,
        "protected_keys": ["commit:" + r["commit"] for r in inventory if r["commit"] in ids],
        "candidates": candidates,
    }
    manifest["manifest_sha256"] = H(D(manifest))
    save(ROOT / "manifest-reviewed.json", manifest)
    save(ROOT / "documentation-omissions.json", docs)
    save(
        ROOT / "case-index.json",
        {
            "cases": cases,
            "scope": "CPU exact source reconstruction, complete non-documentation patch replay and canonical tokenizer serialization. No compiler, numerical kernel or performance executions.",
            "tokenizer": tokenizer_identity(tkpath),
            "counts": dict(collections.Counter(x["status"] for x in cases)),
            "outcomes": dict(collections.Counter(x["outcome_class"] for x in cases)),
            "reserved_ledger_sha256": H(reserved_path.read_bytes()),
            "source_files_sha256": {
                str(p): H(p.read_bytes())
                for p in [
                    Path(__file__),
                    ROOT / "curation.py",
                    ROOT / "inventory.json",
                    WORK / "kernel-library/expansion-v3-hour/tools/exact_edit_renderer.py",
                    WORK / "kernel-library/expansion-v2/group-b/refined_units.py",
                    WORK / "kernel-library/expansion-v2/tools/source_units.py",
                    *[
                        WORK / "loss-fix/apple/src/exl3_mlx" / x
                        for x in ["agent_edits.py", "upstream_data.py", "data.py", "tool_data.py"]
                    ],
                ]
            },
        },
    )


if __name__ == "__main__":
    main()
