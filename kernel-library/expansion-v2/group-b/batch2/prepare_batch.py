"""Prepare complete public historical edits; CPU only, no upstream execution.

Metadata screening does not admit rows: this file records explicit curator decisions
and retains every rejection. Native rows are canonicalized BEFORE tokenization.
"""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from refined_units import units as refined_units

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[3]
sys.path.insert(0, str(WORK / "loss-fix/apple/src"))
from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity, validate_example
from exl3_mlx.upstream_data import FORMAT, _evidence, reconstruct_candidate

spec = importlib.util.spec_from_file_location("prior_units", WORK / "kernel-library/integration-v1/group-b/prepare_group_b.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
spec = importlib.util.spec_from_file_location("fetch_batch", ROOT / "fetch_batch.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)

H = lambda b: hashlib.sha256(b).hexdigest()
D = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()

# Curator classifications after reading complete captured primary PR descriptions.
# Titles alone never establish positive measured evidence.
DECISIONS = {
 "39a182fb": ("pending_protection", "kernel", "pre-M5 D512 attention working set", "Apple M4 Pro and M3 Ultra", "Use a memory-bounded D512 attention path on pre-M5 GPUs. Primary measurements use protected Gemma model families; finite operation protection review remains pending."),
 "a87691d2": ("positive", "host_dispatch", "thin-M dense GEMM", "Apple M5 Pro", "Route thin-M products through the thin-GEMM path, complementing the earlier thin-N kernel. Source benchmarks report 1.33–1.58x gains on measured BF16 layer shapes, not a full-model speedup."),
 "bbf8a7a5": ("positive", "host_dispatch", "long-context GQA factors 12 and 16", "Apple M5 Pro 24GB", "Reuse KV across GQA heads for factors12 and16, choosing four and two heads per threadgroup to stay within threadgroup memory. Source full-size model results are unavailable; truncated-model observations and extrapolations are not full-model measurements."),
 "a082cb91": ("positive", "host_dispatch", "single-block 32-row quantized NAX matmul", "Apple M5 10-core GPU", "Use a 32-row block only when it covers all of M; larger M retains the parent path. The source reports1.21–1.22x on two FP16 affine shapes, a13% regression if forced at M128, and4.02% larger metallib."),
 "7ace4ae0": ("source_only", "host_dispatch", "top-p mask permutation", None, "Scatter the thresholded top-p keep mask instead of gathering cumulative probabilities. Exact permutation equivalence is source-reported; isolated measured performance is pending."),
 "489f84e5": ("positive", "kernel", "multi-pass hyperconnection verification", "Apple M5 Max", "Split verify rows using the library wide-GEMV pass plan, reducing register pressure while retaining each row reduction. Source reports gains against stock on a97-read compiled BF16 chain, including a previous8-row regression; Finite source review confirms a related typed-HC task; no HC-family-independent efficacy is claimed."),
 "3eb822ca": ("positive", "kernel", "global-scale transposed quantized matrix multiplication", "Apple M5 Max and M3 Ultra", "Load global scales in qmm_t kernels; preserve the separately reported combined effect with the related gather-kernel change."),
 "76284023": ("positive", "kernel", "hyperconnection verification projections", "Apple M5 Max", "Use the installed MLX GemvWide projections and dtype-specific sigmoid lookup tables in the three-dispatch verify-width HC read. This is related HC training with a distinct typed-intermediate/GemvWide ABI; it does not certify HC-family-independent efficacy."),
}



def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    evidence = {}
    for p in sorted((ROOT / "evidence").glob("*-pr-*.txt")):
        repo, number = p.stem.rsplit("-pr-", 1)
        evidence[p.stem] = {"path": str(p.relative_to(ROOT)), "sha256": H(p.read_bytes()), "phase": "post_change", "url": fetch.URLS[repo] + "/pull/" + number}
    protection = WORK / "kernel-library/integration-v1/group-a/evidence/protection.json"
    shutil.copyfile(protection, ROOT / "evidence/protection.json")
    reserved = WORK / "kernel-library/integration-v1/group-a/evidence/reserved-task-contracts.json"
    shutil.copyfile(reserved, ROOT / "evidence/reserved-task-contracts.json")
    binding = {"evidence/protection.json": H(protection.read_bytes()), "evidence/reserved-task-contracts.json": H(reserved.read_bytes())}
    finite_path = ROOT / "evidence/finite-operation-review.json"
    finite_reviews = {x["commit"]: x for x in json.loads(finite_path.read_text())["cases"]}
    sources, _ = _evidence(ROOT, evidence)
    repos = {r: {"path": str(ROOT / "mirrors" / (r + ".git")), "url": u} for r, u in fetch.URLS.items()}
    rows = json.loads((ROOT / "selection.json").read_text())
    out = ROOT / "drafts"
    out.mkdir(exist_ok=True)
    tokenizer_path = "/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
    tokenizer = load_tokenizer(tokenizer_path)
    candidates, cases, docs, train12, train32, prepared12, prepared32 = [], [], {}, [], [], [], []
    for row in rows:
        key = row["sha"][:8]
        tier, kind, operation, hardware, rationale = DECISIONS[key]
        pr = re.search(r"\(#(\d+)\)", row["message"].splitlines()[0])
        ev = f"{row['repo']}-pr-{pr[1]}" if pr else "commit_message"
        body = (ROOT / evidence[ev]["path"]).read_text() if ev in evidence else row["message"]
        ann = lambda text: {"text": text, "basis": "source", "evidence": [{"source": ev, "quote": body}]}
        c = {
            "id": f"{row['repo']}-{row['sha'][:12]}-expansion-v2" + ("-refined" if key == "3eb822ca" else ""), "repository": row["repo"], "commit": row["sha"], "parent": row["parents"][0],
            "outcome": "reported_improvement" if tier == "positive" else "source_only", "protected": False, "source_group": "mlx-derived-optimization-training-lineage", "task_conditioning": "historical_source",
            "task_kind": {"value": kind, "reviewed_by": "adversarial_process source-only curation", "evidence": [{"source": ev, "quote": body}]},
            "annotations": {"operation": ann(operation), "architecture": ann(rationale)},
            "historical_protection_review": {"status": "pending" if tier == "pending_protection" else "cleared", "reviewed_by": "adversarial_process", "reason": ("Pending named family review: " + rationale) if tier == "pending_protection" else "Reviewed exact prior commit/protection metadata and reserved HC/Q8 contracts. This operation is outside reserved DeepSeek FP8 packing/index radix/expert-down reduction, HC gate/mix, and Gemma/Laguna model task families. Same repository/framework lineage and common motifs remain; no family-independent efficacy or universal semantic exclusion is claimed.", "evidence_sha256": binding},
        }
        if hardware:
            c["annotations"]["hardware"] = ann(hardware)
        if row["sha"] in finite_reviews:
            c["historical_protection_review"]["reason"] = finite_reviews[row["sha"]]["reason"]
            c["historical_protection_review"]["evidence_sha256"] = {**binding, "evidence/finite-operation-review.json": H(finite_path.read_bytes())}
        if key == "76284023":
            c["related_commits"] = ["489f84e5a51e25fc8da3d390f1656d7d7fc50a09"]
        if key == "489f84e5":
            c["related_commits"] = ["7628402337f8a23e95c22a0ef67ee473d2cdafe1"]
        case = {"case_id": c["id"], "repository": row["repo"], "commit": row["sha"], "curation_tier": tier, "source_evidence": ev, "rationale": rationale, "execution_verified": False}
        try:
            rec = reconstruct_candidate(c, repos, sources)
            c["context_policy"] = "reviewed_units"
            c["reviewed_units"] = {"reviewed_by": "adversarial_process", "dependency_review": "Complete changed Python functions/methods/types/declarations, explicitly selected complete C++ functions, or complete changed files. Every executable/host/test/build change is retained and exact whole-file replay enforced. Unchanged class headers and dependency closure remain in the pinned repository; this is not a standalone kernel task.", "units": [u for p, pair in rec["files"].items() for u in refined_units(key, p, pair, prior.units)]}
            rec = reconstruct_candidate(c, repos, sources)
            docs[c["id"]] = {p: {"before_sha256": v["before"]["sha256"] if v["before"] else None, "after_sha256": v["after"]["sha256"] if v["after"] else None, "reason": "Human-facing documentation claims omitted from target; source provenance retains full commit; executable/host/tests/build edits unchanged."} for p, v in rec["files"].items() if Path(p).suffix.lower() in {".md", ".rst"}}
            save(out / f"{c['id']}.source.json", rec)
            # Capture license text at the same exact source revision; never execute it.
            license_rows = []
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
                command = ["git", "-c", "core.hooksPath=/dev/null", "-C", repos[row["repo"]]["path"], "show", f"{row['sha']}:{name}"]
                value = subprocess.run(command, capture_output=True)
                if value.returncode == 0:
                    lp = ROOT / "evidence/licenses" / (c["id"] + "-" + name)
                    lp.parent.mkdir(exist_ok=True)
                    lp.write_bytes(value.stdout)
                    license_rows.append({"path": str(lp.relative_to(ROOT)), "sha256": H(value.stdout), "upstream_path": name, "commit": row["sha"]})
            case["licenses"] = license_rows
            raw, replay = agent_example(rec, docs[c["id"]])
            raw = json.loads(D(raw))
            tokens = tokenize_row(raw, tokenizer)
            n = len(tokens["input_ids"]) - 1
            for suffix, obj in (("raw", raw), ("replay", replay), ("prepared", tokens)):
                save(out / f"{c['id']}.{suffix}.json", obj)
            case.update(sequence_positions=n, total_serialized_ids=n + 1, supervised_tokens=n + 1 - tokens["loss_start"], exact_replay=True, practical_12k=n + 1 <= 12288, archival_32k=n + 1 <= 32768, files=list(rec["files"]), source_sha256=H((out / f"{c['id']}.source.json").read_bytes()), raw_sha256=H((out / f"{c['id']}.raw.json").read_bytes()), prepared_sha256=H((out / f"{c['id']}.prepared.json").read_bytes()), replay_sha256=H((out / f"{c['id']}.replay.json").read_bytes()))
            if tier == "positive" and n + 1 <= 32768 and license_rows:
                validate_example(tokens, seq_length=32768, vocab_size=len(tokenizer))
                train32.append(raw); prepared32.append(tokens)
                if n + 1 <= 12288:
                    train12.append(raw); prepared12.append(tokens)
            case["status"] = "admitted_12k" if tier == "positive" and n + 1 <= 12288 and license_rows else "admitted_32k_only" if tier == "positive" and n + 1 <= 32768 and license_rows else "context_excluded" if tier == "positive" and n + 1 > 32768 else "not_positive_admitted"
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            case.update(status="source_or_replay_excluded", error=str(exc))
        candidates.append(c); cases.append(case)
        save(ROOT / "progress.json", cases)
        print(c["id"], tier, case["status"], case.get("sequence_positions", case.get("error")), flush=True)
    manifest = {"format": FORMAT, "repositories": repos, "evidence": evidence, "protected_keys": ["commit:c793734eb715dbcfdb1ced58e348ec53c2d7ed85", "stable_patch:a64f05b89099f719065fd2f432583df50af6d489"], "candidates": candidates}
    manifest["manifest_sha256"] = H(D(manifest))
    save(ROOT / "manifest-reviewed.json", manifest)
    save(ROOT / "documentation-omissions.json", docs)
    for name, values in [("train-12k.raw.jsonl", train12), ("train-12k.prepared.jsonl", prepared12), ("train-32k.raw.jsonl", train32), ("train-32k.prepared.jsonl", prepared32)]:
        (ROOT / name).write_bytes(b"".join(D(x) + b"\n" for x in values))
    save(ROOT / "integration-report.json", {"format": "upstream-expansion-source-report-v1", "scope": "Historical source-reported optimization edits; CPU exact source reconstruction/patch replay/tokenization only. No compile, benchmark, model quality or training result.", "manifest_sha256": manifest["manifest_sha256"], "tokenizer": tokenizer_identity(tokenizer_path), "cases": cases, "attrition": dict(collections.Counter(x["status"] for x in cases)), "curation_tiers": dict(collections.Counter(x["curation_tier"] for x in cases)), "prepared_12k": len(train12), "prepared_32k": len(train32), "negative_outcome_policy": "Mixed source gains/regressions and numerical differences retained in primary evidence and rationale. Correctness-only, copied ports, pending protection and unmeasured changes are excluded from positive rows.", "source_files_sha256": {str(p.relative_to(WORK)): H(p.read_bytes()) for p in [Path(__file__), ROOT / "fetch_batch.py", ROOT / "refined_units.py", WORK / "kernel-library/expansion-v2/tools/source_units.py", WORK / "kernel-library/integration-v1/group-b/prepare_group_b.py", *[WORK / "loss-fix/apple/src/exl3_mlx" / x for x in ("agent_edits.py", "upstream_data.py", "data.py", "tool_data.py")]]}})


if __name__ == "__main__":
    main()
