"""CPU-only Group B source replay/token drafts. No upstream code is executed."""

import ast
import difflib
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / "loss-fix/apple/src"))
from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import (
    load_tokenizer,
    tokenize_row,
    tokenizer_identity,
    validate_example,
)
from exl3_mlx.upstream_data import FORMAT, _evidence, reconstruct_candidate

H = lambda b: hashlib.sha256(b).hexdigest()
D = lambda v: json.dumps(
    v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
).encode()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


URLS = {
    "mlx": "https://github.com/ml-explore/mlx",
    "mlx-lm": "https://github.com/ml-explore/mlx-lm",
    "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
    "mtplx": "https://github.com/youssofal/MTPLX",
}
META = {
    "0e3ff364": (
        4596,
        "kernel",
        "generic two-pass single-token SDPA key-loop scheduling",
        "Apple M1 Pro",
        "Group four independent K/V loads before their score and online-softmax updates, leaving the scalar tail and GQA-specialized kernel unchanged.",
    ),
    "f8aaf49d": (
        4516,
        "kernel",
        "aligned-input quantized matrix-vector projection with a partial output tile",
        "Apple M5 Max",
        "Reuse the last valid weight row for out-of-range reads and mask stores in the final SIMD group while keeping the reduction loop unchanged.",
    ),
    "15a72f2f": (
        1824,
        "host_dispatch",
        "BatchKVCache left-padding extraction and filtering",
        "Apple M4 Max",
        "Read left padding using tolist instead of launching a reduction to recover values supplied by the host.",
    ),
    "e9308d7b": (
        1559,
        "kernel",
        "scalar-gated unmasked Dk=128 gated-delta prefill",
        "Apple M5 Max",
        "Pack eight independent value rows per SIMD-group, with four lanes per row and 32 state elements per lane, preserving the explicit ascending butterfly reduction contract.",
    ),
    "4f4634bb": (
        2443,
        "host_dispatch",
        "RF-DETR DINOv2 backbone attention",
        "Apple M3 Pro",
        "Replace the manual attention matmul-softmax-matmul chain with fused scaled_dot_product_attention.",
    ),
    "933558a6": (
        2468,
        "host_dispatch",
        "RF-DETR segmentation-head bilinear resize",
        "Apple M3 Pro",
        "Reuse resize_bilinear_nhwc with align_corners=True in place of the gather-based helper and preserve its supported resize behavior.",
    ),
    "76284023": (
        None,
        "kernel",
        "verify-width hyperconnection projection and gated stream reduction",
        "Apple M5 Max",
        "Use the installed MLX GemvWide projection template with its accumulation type and no bias branch; retain the stock chain when the required template is unavailable.",
    ),
}


def binding(raw, a, b):
    return {"start": a, "end": b, "sha256": H(raw[a:b])}


def nodes(raw):
    text = raw.decode()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    off = [0]
    for line in lines:
        off.append(off[-1] + len(line.encode()))
    out = []
    prev = 0
    for n in tree.body:
        end = off[n.end_lineno]
        kind = (
            "complete_type"
            if isinstance(n, ast.ClassDef)
            else "complete_function"
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "complete_declaration"
        )
        if hasattr(n, "name"):
            key = type(n).__name__ + ":" + n.name
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            key = "assign:" + ast.dump(
                n.targets[0] if isinstance(n, ast.Assign) else n.target
            )
        elif (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        ):
            key = "module_docstring"
        else:
            key = type(n).__name__ + ":" + ast.dump(n, include_attributes=False)
        out.append((key, prev, end, kind))
        prev = end
    if out:
        out[-1] = (out[-1][0], out[-1][1], len(raw), out[-1][3])
    return out


def units(path, pair):
    b = (pair["before"] or {"content": ""})["content"].encode()
    a = (pair["after"] or {"content": ""})["content"].encode()
    if path == "mlx/backend/metal/kernels/sdpa_vector.h":
        spans = []
        for raw in (b, a):
            at = raw.index(b"[[kernel]] void sdpa_vector_2pass_1(")
            start = raw.rindex(b"template <", 0, at)
            end = raw.index(b"\ntemplate <", at)
            spans.append(binding(raw, start, end))
        return [
            {
                "path": path,
                "kind": "complete_kernel",
                "before": spans[0],
                "after": spans[1],
            }
        ]
    if path.endswith(".py") and b and a:
        bn, an = nodes(b), nodes(a)
        out = []
        for tag, i, j, k, l in difflib.SequenceMatcher(
            None, [x[0] for x in bn], [x[0] for x in an], autojunk=False
        ).get_opcodes():
            spans = (
                [(ii, ii + 1, kk, kk + 1) for ii, kk in zip(range(i, j), range(k, l))]
                if tag == "equal"
                else [(i, j, k, l)]
            )
            for i, j, k, l in spans:
                b0 = bn[i][1] if i < len(bn) else len(b)
                b1 = bn[j - 1][2] if j > i else b0
                a0 = an[k][1] if k < len(an) else len(a)
                a1 = an[l - 1][2] if l > k else a0
                if b[b0:b1] == a[a0:a1]:
                    continue
                kind = bn[i][3] if j - i == l - k == 1 else "complete_declaration"
                out.append(
                    {
                        "path": path,
                        "kind": kind,
                        "before": binding(b, b0, b1),
                        "after": binding(a, a0, a1),
                    }
                )
        return out
    return [
        {
            "path": path,
            "kind": "whole_file",
            "before": binding(b, 0, len(b)),
            "after": binding(a, 0, len(a)),
        }
    ]


def main():
    evid = {}
    bodies = {}
    for p in (ROOT / "evidence").glob("*-pr-*.json"):
        d = json.loads(p.read_text())
        name = p.stem
        body = d["body"]
        target = p.with_suffix(".txt")
        target.write_text(body)
        bodies[name] = body
        evid[name] = {
            "path": str(target.relative_to(ROOT)),
            "sha256": H(target.read_bytes()),
            "phase": "post_change",
            "url": d["html_url"],
        }
    protection = (
        WORK
        / "loss-fix/apple/artifacts/loss-fix/upstream-inventory/ktune_protection_metadata.json"
    )
    shutil.copyfile(protection, ROOT / "evidence/protection.json")
    protsha = H(protection.read_bytes())
    draft = json.loads((ROOT / "selection-draft.json").read_text())
    repos = {
        r: {"path": str(ROOT / "mirrors" / f"{r}.git"), "url": u}
        for r, u in URLS.items()
    }
    sources, _ = _evidence(ROOT, evid)
    candidates = []
    records = []
    docs = {}
    report = []
    for r in draft:
        key = r["sha"][:8]
        if key not in META:
            continue
        pr, kind, op, hardware, arch = META[key]
        ev = f"{r['repo']}-pr-{pr}" if pr else "commit_message"
        body = bodies[ev] if pr else r["message"]

        def ann(text, ev=ev, body=body):
            return {
                "text": text,
                "basis": "source",
                "evidence": [{"source": ev, "quote": body}],
            }

        c = {
            "id": r["repo"] + "-" + r["sha"][:12],
            "repository": r["repo"],
            "commit": r["sha"],
            "parent": r["parents"][0],
            "outcome": "reported_improvement",
            "protected": False,
            "source_group": r["repo"]
            + "-kernel-family-"
            + ("rfdetr" if r["repo"] == "mlx-vlm" else key),
            "task_conditioning": "historical_source",
            "task_kind": {
                "value": kind,
                "reviewed_by": "adversarial_process; source-only curation",
                "evidence": [{"source": ev, "quote": body}],
            },
            "annotations": {
                "operation": ann(op),
                "hardware": ann(hardware),
                "architecture": ann(arch),
            },
            "historical_protection_review": {
                "status": "cleared",
                "reviewed_by": "adversarial_process",
                "reason": "Reviewed bound ktune protection metadata, including breadth commit IDs. These pinned commits are not any listed historical commit; edited operations are neither protected ds4 fp8_rounding/indexer_radix/moe_reduce nor Gemma/Laguna tasks nor llama.cpp. Generic shared optimization motifs are not certified independent. Repository and related-kernel grouping retained; no universal semantic no-leakage claim.",
                "evidence_sha256": {"evidence/protection.json": protsha},
            },
        }
        record = reconstruct_candidate(c, repos, sources)
        c["context_policy"] = "reviewed_units"
        c["reviewed_units"] = {
            "reviewed_by": "adversarial_process",
            "dependency_review": "Complete changed top-level Python functions/types/declarations selected and full changed-file replay enforced. Unchanged dependencies remain in the pinned parent repository; this is an in-repository edit task, not a standalone compilable kernel or complete dependency closure.",
            "units": [
                u for path, pair in record["files"].items() for u in units(path, pair)
            ],
        }
        record = reconstruct_candidate(c, repos, sources)
        omissions = {
            p: {
                "before_sha256": pair["before"]["sha256"] if pair["before"] else None,
                "after_sha256": pair["after"]["sha256"],
                "reason": "Historical human-facing documentation claims are preserved in source provenance but omitted from the target patch; all executable, host, tests and build edits remain.",
            }
            for p, pair in record["files"].items()
            if pathlib.Path(p).suffix.lower() in (".md", ".rst")
        }
        docs[c["id"]] = omissions
        records.append(record)
        candidates.append(c)
        print(
            c["id"],
            [
                (
                    p,
                    len((v["before"] or {"content": ""})["content"]),
                    len((v["after"] or {"content": ""})["content"]),
                )
                for p, v in record["files"].items()
            ],
            flush=True,
        )
    manifest = {
        "format": FORMAT,
        "repositories": repos,
        "evidence": evid,
        "protected_keys": [],
        "candidates": candidates,
    }
    manifest["manifest_sha256"] = H(D(manifest))
    save(ROOT / "manifest-reviewed.json", manifest)
    save(ROOT / "documentation-omissions.json", docs)
    out = ROOT / "drafts"
    out.mkdir(exist_ok=True)
    tokroot = (
        "/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
    )
    tok = load_tokenizer(tokroot)
    raws = []
    preps = []
    for rec in records:
        case = rec["case_id"]
        save(out / f"{case}.source.json", rec)
        try:
            raw, receipt = agent_example(rec, docs[case])
            tokens = tokenize_row(raw, tok)
            n = len(tokens["input_ids"]) - 1
            save(out / f"{case}.raw.json", raw)
            save(out / f"{case}.replay.json", receipt)
            save(out / f"{case}.prepared.json", tokens)
            status = (
                "fits_12288"
                if n <= 12288
                else "fits_32768_only"
                if n <= 32768
                else "context_excluded"
            )
            if n <= 32768:
                validate_example(tokens, seq_length=32768, vocab_size=len(tok))
                raws.append(raw)
                preps.append(tokens)
            report.append(
                {
                    "case_id": case,
                    "repository": rec["repository"],
                    "targets": n,
                    "supervised_tokens": len(tokens["input_ids"])
                    - tokens["loss_start"],
                    "status": status,
                    "source_sha256": H((out / f"{case}.source.json").read_bytes()),
                    "raw_sha256": H((out / f"{case}.raw.json").read_bytes()),
                    "replay_sha256": H((out / f"{case}.replay.json").read_bytes()),
                    "prepared_sha256": H((out / f"{case}.prepared.json").read_bytes()),
                }
            )
            print(case, status, n, flush=True)
        except (ValueError, OSError, RuntimeError) as e:
            report.append({"case_id": case, "error": str(e), "status": "excluded"})
            print(case, "ERROR", repr(e), flush=True)
    (ROOT / "train.raw.jsonl").write_bytes(b"".join(D(r) + b"\n" for r in raws))
    (ROOT / "train.prepared.jsonl").write_bytes(b"".join(D(r) + b"\n" for r in preps))
    save(
        ROOT / "integration-report.json",
        {
            "scope": "CPU source reconstruction, exact Git patch replay and local native-tool tokenization only; no compilation, runtime or training claim",
            "manifest_sha256": manifest["manifest_sha256"],
            "tokenizer": tokenizer_identity(tokroot),
            "cases": report,
            "prepared_rows": len(preps),
            "all_reported_performance": "source_reported_not_independently_verified",
            "limitations": [
                "Historical benchmark revisions may differ from final merged commit; exact PR metadata retained.",
                "Not a standalone dependency closure; model sees complete selected declarations and baseline patch anchors.",
                "No source-family-independent evaluation assigned to these training candidates.",
            ],
        },
    )


if __name__ == "__main__":
    main()
