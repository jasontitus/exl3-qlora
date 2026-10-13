"""Exact historical patch reconstruction using complete, reviewed source units."""

import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from curation import DECISIONS

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / "loss-fix/apple/src"))
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity
from exl3_mlx.upstream_data import FORMAT, _evidence, reconstruct_candidate


def imp(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


units = imp(
    "hour4_units", WORK / "kernel-library/expansion-v2/group-b/refined_units.py"
)
agent_example = imp(
    "hour4_renderer",
    WORK / "kernel-library/expansion-v3-hour/tools/exact_edit_renderer.py",
).agent_example
H = lambda b: hashlib.sha256(b).hexdigest()
D = lambda x: json.dumps(
    x, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
).encode()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def fullunit(path, pair):
    b = (pair["before"] or {"content": ""})["content"].encode()
    a = (pair["after"] or {"content": ""})["content"].encode()
    return [
        {
            "path": path,
            "kind": "whole_file",
            "before": units.bind(b, 0, len(b)),
            "after": units.bind(a, 0, len(a)),
        }
    ]


# Prefix-to-path maps are explicitly reviewed declarations, never arbitrary hunks.
SELECT = {
    "0512aa6": {"mlx/backend/metal/quantized.cpp": ["void qvm_split_k("]},
    "95f8533": {"mlx/backend/metal/kernels/sdpa_vector.h": ["void sdpa_vector("]},
    "ad00ea2": {
        "mlx/backend/metal/kernels/reduction/reduce_col.h": [
            "void col_reduce_longcolumn("
        ]
    },
    "dfe17ba": {
        "mlx/backend/metal/kernels/fp_quantized.h": [
            "struct QuantizedBlockLoader",
            "void fp_qmm_t_impl(",
            "void fp_qmm_n_impl(",
        ],
        "mlx/backend/metal/kernels/quantized.h": ["struct QuantizedBlockLoader"],
    },
    "685dba9": {
        "mlx/backend/metal/kernels/sdpa_vector.h": ["void sdpa_vector_2pass_1_gqa("]
    },
    "d73eb75": {
        "mlx/backend/metal/kernels/quantized_nax.h": ["void affine_gather_qmm_rhs_nax("]
    },
    "a9eed5a": {"mlx/backend/metal/quantized.cpp": ["void quantize_impl("]},
    "a1e0e0b": {
        "mlx/backend/metal/kernels/reduction/reduce_row.h": ["void row_reduce_looped("]
    },
    "052d428": {
        "mlx/backend/metal/kernels/sort.h": [
            "void block_sort_nc(",
            "void mb_block_sort(",
        ]
    },
    "2414e5d": {
        "mlx/backend/metal/kernels/steel/gemm/mma.h": ["struct BaseMMAFrag<T, 8, 8>"]
    },
    "241f1bf": {
        "mlx/backend/metal/kernels/fp_quantized.h": [
            "void fp_qvm_impl(",
            "void fp_qvm(",
            "void fp_qvm_split_k(",
            "void fp_gather_qvm(",
        ],
        "mlx/backend/metal/kernels/quantized.h": [
            "void qvm_impl(",
            "void affine_qvm(",
            "void affine_qvm_split_k(",
            "void affine_gather_qvm(",
        ],
    },
    "fa7b5b8": {
        "mlx/backend/metal/conv.cpp": [
            "void implicit_gemm_conv_2D_gpu(",
            "void dispatch_conv_2D_gpu(",
        ],
        "mlx/backend/metal/kernels/steel/conv/loaders/loader_channel_n.h": [
            "struct ChannelHelper {",
            "struct ChannelHelper<1>",
            "struct ChannelHelper<2>",
            "struct ChannelHelper<3>",
            "struct ChannelHelper<4>",
            "struct Conv2DInputBlockLoaderSmallChannels",
        ],
    },
    "f8aaf49": {
        "mlx/backend/metal/kernels/fp_quantized.h": [
            "void fp_qmv_fast_impl(",
            "void fp_qmv_fast(",
            "void fp_qmv(",
        ],
        "mlx/backend/metal/kernels/quantized.h": [
            "inline U load_vector(",
            "inline U load_vector_safe(",
            "void qmv_fast_impl(",
            "void affine_qmv_fast(",
            "void affine_qmv(",
        ],
        "mlx/backend/metal/kernels/quantized_nax.h": [
            "inline U load_vector(",
            "inline U load_vector_safe(",
        ],
        "mlx/backend/metal/quantized.cpp": ["void qmv("],
    },
    "d56ed24": {
        "mlx/backend/metal/kernels/fft.h": [
            "void radix_butterfly(",
            "void radix_n_steps(",
            "void perform_fft(",
            "void bluestein_fft(",
        ],
        "mlx/backend/metal/fft.cpp": [
            "std::pair<array, array> compute_bluestein_constants(",
        ],
        "mlx/backend/metal/jit_kernels.cpp": [
            "MTL::ComputePipelineState* get_fft_kernel("
        ],
        "mlx/backend/metal/nojit_kernels.cpp": [
            "MTL::ComputePipelineState* get_fft_kernel("
        ],
    },
    "2d27ab0": {
        "mlx/backend/metal/matmul.cpp": [
            "void steel_matmul_regular_axpby_nax(",
            "void steel_gemm_splitk_axpby_nax(",
        ]
    },
    "c7ff35d": {
        "mlx/backend/metal/kernels/fp_quantized_nax.h": ["void fp_gather_qmm_rhs_nax("],
        "mlx/backend/metal/kernels/quantized_nax.h": [
            "void affine_gather_qmm_rhs_nax("
        ],
    },
    "9008944": {
        "mlx/backend/metal/kernels/fp_quantized.h": ["void fp_gather_qmm_rhs("],
        "mlx/backend/metal/kernels/fp_quantized_nax.h": ["void fp_gather_qmm_rhs_nax("],
        "mlx/backend/metal/kernels/quantized.h": ["void affine_gather_qmm_rhs("],
        "mlx/backend/metal/kernels/quantized_nax.h": [
            "void affine_gather_qmm_rhs_nax("
        ],
        "mlx/backend/metal/quantized.cpp": [
            "void gather_qmm_rhs_nax(",
            "void gather_qmm_rhs(",
        ],
    },
    "a2a09fd": {
        "mlx/backend/metal/matmul.cpp": [
            "void gather_mm_rhs(",
            "void gather_mm_rhs_nax(",
        ]
    },
}


def declaration_range(text, sig):
    raw = text.encode()
    code, pairs, _directives = units._source_units.lex(raw, "c")
    if sig.startswith("struct "):
        if code.count(sig) != 1:
            raise ValueError("Ambiguous struct " + sig)
        pos = code.index(sig)
        start = code.rfind("\n", 0, pos) + 1
        br = code.index("{", pos)
        end = pairs[br] + 1
        if raw[end : end + 1] == b";":
            end += 1
        if raw[end : end + 1] == b"\n":
            end += 1
    else:
        start, end = units.function_range(text, sig)
    # Preserve adjacent whole template prefix, including spaced/unspaced forms.
    previous = code[:start].rstrip()
    m = list(re.finditer(r"\btemplate\s*<", previous))
    if previous.endswith(">") and m:
        candidate = m[-1].start()
        if re.fullmatch(r"template\s*<[^;{}]*>\s*", code[candidate:start]):
            start = code.rfind("\n", 0, candidate) + 1
    return start, end


def omitted_paths(rec, selected):
    out = {}
    for path, pair in rec["files"].items():
        b = (pair["before"] or {"content": ""})["content"].encode()
        a = (pair["after"] or {"content": ""})["content"].encode()
        bc = ac = 0
        for u in sorted(
            [u for u in selected if u["path"] == path],
            key=lambda u: u["before"]["start"],
        ):
            bs, as_ = u["before"]["start"], u["after"]["start"]
            if b[bc:bs] != a[ac:as_]:
                out[path] = (
                    "Prompt context selects complete reviewed declarations; other changed bytes remain fully present in baseline anchors and exact non-documentation target patch."
                )
            bc, ac = u["before"]["end"], u["after"]["end"]
        if b[bc:] != a[ac:]:
            out[path] = (
                "Prompt context selects complete reviewed declarations; other changed bytes remain fully present in baseline anchors and exact non-documentation target patch."
            )
    return out


def main():
    inventory = json.loads((ROOT / "inventory-all.json").read_text())["records"]
    prior = json.loads((ROOT / "evidence/prior-identities.json").read_text())
    used = {x["commit"] for x in prior}
    oldpatch = {x["stable_patch_id"] for x in prior if x.get("stable_patch_id")}
    reserved_path = (
        WORK / "kernel-library/integration-v1/group-a/evidence/reserved-metadata.json"
    )
    reserved = json.loads(reserved_path.read_text())["review"]
    ids = {v[k] for v in reserved["coverage"] for k in ("head", "base") if v.get(k)}
    reserved_prs = {
        v["number"] for v in reserved["reserved_pr_lineage"] if v["repo"] == "mlx"
    } | {
        n
        for v in reserved["reserved_pr_lineage"]
        if v["repo"] == "mlx"
        for n in v["referenced_prs"]
    }
    ev = {}
    repos = {
        r: {
            "path": str(ROOT / "mirrors" / f"{r}.git"),
            "url": "https://github.com/" + u,
        }
        for r, u in {
            "mlx": "ml-explore/mlx",
            "mlx-lm": "ml-explore/mlx-lm",
            "mlx-vlm": "Blaizzy/mlx-vlm",
        }.items()
    }
    for row in inventory:
        pr = row["pr"]
        ev[f"{row['repo']}-pr-{pr['number']}"] = {
            "path": pr["body_path"],
            "sha256": pr["body_sha256"],
            "phase": "post_change",
            "url": pr["url"],
        }
        ev[row["commit"] + "-diff"] = {
            "path": row["patch"]["path"],
            "sha256": row["patch"]["sha256"],
            "phase": "post_change",
            "url": row["url"],
        }
    sources, _ = _evidence(ROOT, ev)
    tkpath = "/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
    tk = load_tokenizer(tkpath)
    candidates = []
    cases = []
    docs = {}
    seen = set()
    for row in inventory:
        key = row["commit"][:7]
        decision = next(v for k, v in DECISIONS.items() if k.startswith(key))
        outcome, kind, operation, rationale = decision
        pr = row["pr"]
        ek = f"{row['repo']}-pr-{pr['number']}"
        body = (ROOT / pr["body_path"]).read_bytes().decode("utf-8")
        case = {
            "id": f"{row['repo']}-{row['commit'][:12]}-hour-v4",
            "repo": repos[row["repo"]]["url"],
            "commit": row["commit"],
            "parent": row["parent"],
            "lineage": row["repo"] + ":" + row["commit"],
            "source_group": "mlx-derived-framework-training",
            "task_kind": kind,
            "outcome_class": outcome,
            "licenses": row["licenses"],
        }
        matches = [x for x in (row["commit"], row["parent"]) if x in ids]
        prhit = row["repo"] == "mlx" and pr["number"] in reserved_prs
        if matches or prhit or row["commit"] in used or outcome == "pending":
            case.update(
                status="excluded_protection_or_prior",
                reason=rationale,
                exact_reserved_matches=matches,
                reserved_pr_match=prhit,
                prior_identity_match=row["commit"] in used,
            )
            cases.append(case)
            continue

        def ann(text, ek=ek, body=body, row=row):
            return {
                "text": text,
                "basis": "source",
                "evidence": [
                    {
                        "source": ek if body.strip() else "commit_message",
                        "quote": body if body.strip() else row["message"],
                    },
                    {
                        "source": row["commit"] + "-diff",
                        "quote": (ROOT / row["patch"]["path"])
                        .read_bytes()
                        .decode("utf-8"),
                    },
                ],
            }

        protection = {
            "status": "cleared",
            "reviewed_by": "adversarial_code",
            "reason": "Exact head/parent and reserved PR identities are absent from the captured protected ledger. The bound finite-operation report reviews the actual changed numerical, layout or dispatch mechanism; shared MLX primitive families remain related training lineage, with no independent-family or global semantic nonoverlap claim.",
            "evidence_sha256": {
                str(reserved_path): H(reserved_path.read_bytes()),
                str(ROOT / "evidence/finite-operation-review.json"): H(
                    (ROOT / "evidence/finite-operation-review.json").read_bytes()
                ),
            },
        }
        c = {
            "id": case["id"],
            "repository": row["repo"],
            "commit": row["commit"],
            "parent": row["parent"],
            "outcome": "reported_improvement"
            if outcome == "positive"
            else "source_only",
            "protected": False,
            "source_group": case["source_group"],
            "task_conditioning": "historical_source",
            "task_kind": {
                "value": kind,
                "reviewed_by": "adversarial_code",
                "evidence": [
                    {
                        "source": ek if body.strip() else "commit_message",
                        "quote": body if body.strip() else row["message"],
                    }
                ],
            },
            "annotations": {
                "operation": ann(operation),
                "architecture": ann(rationale),
            },
            "historical_protection_review": protection,
        }
        try:
            rec = reconstruct_candidate(c, repos, sources)
            selected = []
            selection = []
            if rec["stable_patch_id"] in oldpatch or rec["stable_patch_id"] in seen:
                raise ValueError("Prior or within-batch stable-patch duplicate")
            for path, pair in rec["files"].items():
                if (
                    path == ".gitignore"
                    or (key == "fa7b5b8" and path == "python/tests/test_conv.py")
                    or (key == "d56ed24" and path == "mlx/backend/metal/kernels.h")
                ):
                    continue  # Full target and baseline anchors retain this non-kernel metadata.
                sigs = SELECT.get(key, {}).get(path)
                if sigs:
                    for sig in sigs:
                        b = pair["before"]["content"]
                        a = pair["after"]["content"]
                        bs = declaration_range(b, sig)
                        as_ = declaration_range(a, sig)
                        selected.append(
                            {
                                "path": path,
                                "kind": "complete_type"
                                if sig.startswith("struct ")
                                else "complete_function",
                                "before": units.bind(b.encode(), *bs),
                                "after": units.bind(a.encode(), *as_),
                            }
                        )
                        selection.append({"path": path, "signature": sig})
                elif path.endswith(".py") and pair["before"] and pair["after"]:
                    try:
                        selected.extend(units.python_units(path, pair))
                    except (SyntaxError, ValueError, IndexError):
                        selected.extend(fullunit(path, pair))
                else:
                    selected.extend(fullunit(path, pair))
            c["context_policy"] = "reviewed_operation"
            c["reviewed_units"] = {
                "reviewed_by": "adversarial_code",
                "dependency_review": "Complete changed Python AST declarations, explicit reviewed C++/Metal functions/types, or whole changed files. Full non-documentation target includes ALL source, host, build and test edits plus baseline anchors. External includes/runtime dependencies remain repository context; no standalone compiler or execution claim.",
                "units": selected,
            }
            c["operation_scope"] = {
                "description": operation
                + ". Complete non-documentation target; reviewed prompt declarations only.",
                "omitted_path_reasons": omitted_paths(rec, selected),
            }
            rec = reconstruct_candidate(c, repos, sources)
            docs[c["id"]] = {
                p: {
                    "before_sha256": v["before"]["sha256"] if v["before"] else None,
                    "after_sha256": v["after"]["sha256"] if v["after"] else None,
                    "reason": "Human-facing documentation omitted; source/build/tests/host remain unchanged in complete target.",
                }
                for p, v in rec["files"].items()
                if Path(p).suffix.lower() in {".md", ".rst"}
            }
            raw, replay = agent_example(rec, docs[c["id"]])
            msg = next(m for m in raw["messages"] if m["role"] == "user")
            msg["content"] = (
                "Task: "
                + operation
                + ".\nMake the complete required source, host, build and test changes with exec_command and git apply. Preserve unsupported-path fallbacks and the numerical contract outside this requested change. This is historical source reconstruction, not evidence of local profiling, compiler or runtime execution.\n\n"
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
                save(ROOT / "reviewed" / f"{c['id']}.{suffix}.json", obj)
            n = len(tokens["input_ids"])
            eligible = n <= 32768
            seen.add(rec["stable_patch_id"])
            tier = {
                "positive": "optimization",
                "correctness": "correctness",
                "neutral": "source_grounded_unmeasured",
            }[outcome]
            case.update(
                status="reviewed_exact_edit",
                stable_patch_id=rec["stable_patch_id"],
                lineage=(
                    "mlx-expert-boundary-scheduler-a2a09fd5"
                    if key in {"9008944", "a2a09fd"}
                    else "patch:" + rec["stable_patch_id"]
                ),
                curriculum_tier=tier,
                recommended_core=outcome != "neutral",
                exact_edit_eligible=eligible,
                total_serialized_ids=n,
                supervised_tokens=n - tokens["loss_start"],
                fits_12k=n <= 12288,
                fits_32k=n <= 32768,
                exact_replay=True,
                protection_review=protection,
                selected_declarations=selection,
                paths={
                    s: str(ROOT / "reviewed" / f"{c['id']}.{s}.json")
                    for s in ["source", "raw", "prepared", "replay"]
                },
                files_sha256={
                    s: H((ROOT / "reviewed" / f"{c['id']}.{s}.json").read_bytes())
                    for s in ["source", "raw", "prepared", "replay"]
                },
                review={
                    "status": "pending",
                    "reason": "Independent source/payload review requested.",
                },
            )
            candidates.append(c)
        except Exception as e:  # noqa: BLE001 -- per-case exclusion ledger must retain failures
            case.update(status="source_or_replay_error", error=repr(e))
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
        "protected_keys": [],
        "candidates": candidates,
    }
    manifest["manifest_sha256"] = H(D(manifest))
    save(ROOT / "manifest-reviewed.json", manifest)
    save(ROOT / "documentation-omissions.json", docs)
    save(
        ROOT / "case-index.json",
        {
            "format": "historical-kernel-curation-index-v1",
            "status": "awaiting_independent_review",
            "cases": cases,
            "scope": "Complete exact non-doc edit replay and canonical tokenizer audit; no GPU/compiler/performance qualification. Unmeasured tier opt-in only.",
            "tokenizer": tokenizer_identity(tkpath),
            "counts": dict(collections.Counter(x["status"] for x in cases)),
            "source_files_sha256": {
                str(p): H(p.read_bytes())
                for p in [
                    Path(__file__),
                    ROOT / "curation.py",
                    ROOT / "inventory-all.json",
                ]
            },
        },
    )


if __name__ == "__main__":
    main()
