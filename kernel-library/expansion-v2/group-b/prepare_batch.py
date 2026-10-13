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
WORK = ROOT.parents[2]
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
    "3c40e8f9": ("positive", "kernel", "thin-N dense GEMM", "Apple M5 Pro", "Use 32x32 SIMD-group output blocks and split K for small-N GEMMs instead of wasting NAX tile columns."),
    "0b37b0c7": ("correctness_only", "host_dispatch", "sort dispatch for transposed views", "Apple M3 Pro", "Collapse nonsorted axes and require a single ordered run for contiguous-kernel selection; incorrect shapes can become slower."),
    "ea7e3973": ("positive", "host_dispatch", "six-row large affine quantized matmul", "Apple M1 Max", "Keep applegpu_g13s large affine inputs on the QMV path through six rows; other architecture thresholds remain unchanged."),
    "02ce1fb6": ("positive", "kernel", "cross entropy forward and backward", "Apple M3 Pro", "Fuse rowwise FP32 cross entropy and reconstruct the backward softmax without a second reduction or one-hot allocation. Negative-infinity target behavior differs from the fallback."),
    "09e67c68": ("positive", "host_dispatch", "linear interpolation upsampling", None, "Route linear interpolation without antialiasing through separable interpolation for the measured upscaling cases."),
    "d9add9d1": ("positive", "host_dispatch", "global-scale gather quantized matrix multiplication", "Apple M5 Max and M3 Ultra", "Use matrix gather kernels for gather_qqmm with global scale."),
    "3eb822ca": ("positive", "kernel", "global-scale transposed quantized matrix multiplication", "Apple M5 Max and M3 Ultra", "Load global scales in qmm_t kernels; preserve the separately reported combined effect with the related gather-kernel change."),
    "f99e916b": ("pending_protection", "host_dispatch", "head-dimension-256 array-masked NAX attention", "Apple M5", "Allow the existing fused NAX path for array-masked head-dimension-256 prefill; includes Gemma-family evidence requiring protection review."),
    "c793734e": ("correctness_only", "kernel", "compiled float-to-bool casts", "Apple M3 Max", "Use cast_to for fused AsType so representable subnormals remain true; no speedup is claimed."),
    "f79dba78": ("positive", "host_dispatch", "minimum probability sampling", "Apple Silicon; exact chip not stated", "Use max and threshold for min-p sampling, with argpartition only when multiple minimum tokens are retained, avoiding full vocabulary sort and temporary arrays."),
    "d13bd3cc": ("duplicate_port", "kernel", "packed gated delta prefill", "Apple M5 Max", "This ports the already admitted mlx-lm packed GDN implementation; preserve evidence but do not count copied kernels as additional distinct examples."),
    "125ca681": ("positive", "host_dispatch", "large MoE route permutation inversion", "Apple M1 Max", "Invert large route permutations with a scatter instead of a second sort, keeping smaller routes and expert arithmetic unchanged."),
    "952f2b15": ("source_only", "host_dispatch", "YaRN fused attention dispatch", None, "Keep fused attention rows under YaRN; quantitative isolated evidence has not been established for this commit."),
    "c79c6aaa": ("positive", "kernel", "Q8 checkpoint-layout W8A8 prefill", "Apple M5 Max", "Read group scales and biases in checkpoint layout, swapping tensor operands to remove transposed metadata copies while preserving INT32 accumulation and FP32 correction order."),
    "ce70139b": ("positive", "kernel", "Q8 group-size-64 W8A8 prefill", "Apple M5 Max", "Route eligible Q8 affine prefill projections through W8A8 on M5; this changes activation arithmetic and retains source quality and rounding caveats."),
    "5ec8ef32": ("pending_protection", "kernel", "batched QSA decode and verification", "Apple M5 Ultra", "Reuse row index banks and gather selected attention for multi-request batches; indexer-family protection and short-context regressions require separate review."),
    "f7270625": ("positive", "host_dispatch", "NAX sparse attention for verification blocks", "Apple M5 Ultra", "Reuse NAX sparse attention for verification blocks. Source outputs differ and speculative acceptance changes, so the reported serving gain is not a bitwise-equivalence result."),
    "b3445232": ("positive", "host_dispatch", "deferred evaluation and compiled FFN verification", "Apple M5 Ultra", "Extend deferred-view handling, evaluation scheduling and compiled FFNs to verification blocks while preserving the source-tested greedy outputs; this does not edit HC kernel arithmetic."),
    "f0d4cb53": ("pending_protection", "kernel", "DeepSeek FP8 packing and index radix kernels", "Apple M3 Ultra and M5 Max", "Contains FP8 packing, Sinkhorn, sparse attention and radix selection; overlaps protected DeepSeek operation families and is not admitted."),
    "00decffe": ("positive", "host_dispatch", "fused GLM decode architecture dispatch", "Apple M3 Ultra", "Enable attempts at existing fused decode kernels on gen15 and newer GPUs. Reported benchmarks used an earlier opt-in variant; final automatic guard timing has not been independently reproduced."),
    "57fda05d": ("pending_protection", "kernel", "FP16 and top-eight fused MoE decode", None, "Fused decode reduction-family overlap remains pending; no positive admission by title alone."),
    "7ab0fd9c": ("positive", "kernel", "Flash-Next routed expert gate/up A8 prefill", "Apple M5", "Route expert gate/up prefill through A8 on M5; source numerical and quality caveats remain part of the evidence."),
    "1bead140": ("positive", "host_dispatch", "two-row SIMD-group MoE down launch", "Apple M5 Ultra", "Use two rows per SIMD group for one-token MoE down launches while verification keeps four; kernel arithmetic is unchanged, and source reports flat MTP results alongside a small non-MTP gain."),
    "fb663154": ("positive", "host_dispatch", "multi-request grouped-query attention decode", "Apple M5 Max, M5 Pro and M3", "Select P256/P512 grouped-query kernels from actual KV lengths, GPU occupancy and bounded scratch; retain negative controls and fallback routes."),
    "88cd4e2b": ("source_only", "host_dispatch", "TurboQuant admission threshold memoization", None, "Reduce repeated threshold function calls; no runtime speed or memory measurement established."),
    "f552da27": ("pending_protection", "host_dispatch", "selective multimodal logits projection", "Apple M2", "Project only sampled multimodal logits; Gemma-family protection remains pending."),
    "d2149cda": ("positive", "host_dispatch", "absorbed MLA query-head folding", "Apple M5 Pro", "Fold query heads into one SDPA head so shared latent KV is read by one GEMM; batched rounding and natural generation caveats remain explicit."),
    "2eab838b": ("positive", "host_dispatch", "P256/P512 grouped-query attention dispatch", "Apple M5 Max, M5 Pro and M3", "Choose a supported grouped-query partition using complete partitions and detected GPU cores, retaining unsupported-input fallback routes."),
    "d8cb131f": ("positive", "kernel", "P256/P512 grouped-query decode kernels", "Apple M5 Max, M5 Pro and M3", "Share KV loads across grouped query heads while retaining split-KV reduction; this kernel-only commit does not itself enable serving dispatch."),
    "93397f88": ("source_only", "host_dispatch", "eager float32 attention sink initialization", None, "Evaluate the widened sink tensor at patch time; no independent measured optimization evidence in this description."),
    "501fbb68": ("source_only", "host_dispatch", "float32 attention sink warm cache", None, "Construct the widened sink at patch time; a follow-up observes this remains lazy, so this is not an independently measured positive example."),
    "0ff8b4c4": ("positive", "kernel", "Q6_K batch matrix-vector and GEMM paths", "Apple M1 Ultra", "Decode weights once for batched qmv and match the activation dtype for GEMM; preserve reported FP16 regressions at some batch5-8 shapes."),
    "1a4aada2": ("positive", "host_dispatch", "short-context absorbed MLA decode batching", None, "Batch short-context decode rows into a padded gather and SDPA under explicit row/context/volume gates; retain long-context negative controls."),
    "cb8e46c6": ("pending_protection", "kernel", "sliding-window decode block-range skipping", "Apple M5 Pro and M3 Ultra", "Skip blocks before the window using neutral partition outputs; Gemma-family protection remains pending."),
    "aa68928d": ("positive", "kernel", "routed gate/up row-map fused SwiGLU prefill", "Apple M5 Max", "Read routed token rows through their map and fuse SiLU into gate/up matmul using installed MLX templates; keep down projection on its existing path, where the proposed alternative was slower."),
    "ae2746a5": ("positive", "host_dispatch", "version-dependent routed gate/up threshold", "Apple M5 Max", "Raise the routed-row threshold on MLX0.32.3 because the existing sorted gather is faster on short chunks; preserve older-version routing."),
    "07261055": ("positive", "kernel", "split-weight SwitchGLU row-map SwiGLU", "Apple M5 Max", "Extend the row-map gate/up kernel to separate quantized gate and up weights with the module's own activation and first-use fallback; down projection stays unchanged."),
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
    rows = json.loads((ROOT / "batch1-selection.json").read_text())
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
            "id": f"{row['repo']}-{row['sha'][:12]}-expansion-v2", "repository": row["repo"], "commit": row["sha"], "parent": row["parents"][0],
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
    manifest = {"format": FORMAT, "repositories": repos, "evidence": evidence, "protected_keys": [], "candidates": candidates}
    manifest["manifest_sha256"] = H(D(manifest))
    save(ROOT / "manifest-reviewed.json", manifest)
    save(ROOT / "documentation-omissions.json", docs)
    for name, values in [("train-12k.raw.jsonl", train12), ("train-12k.prepared.jsonl", prepared12), ("train-32k.raw.jsonl", train32), ("train-32k.prepared.jsonl", prepared32)]:
        (ROOT / name).write_bytes(b"".join(D(x) + b"\n" for x in values))
    save(ROOT / "integration-report.json", {"format": "upstream-expansion-source-report-v1", "scope": "Historical source-reported optimization edits; CPU exact source reconstruction/patch replay/tokenization only. No compile, benchmark, model quality or training result.", "manifest_sha256": manifest["manifest_sha256"], "tokenizer": tokenizer_identity(tokenizer_path), "cases": cases, "attrition": dict(collections.Counter(x["status"] for x in cases)), "curation_tiers": dict(collections.Counter(x["curation_tier"] for x in cases)), "prepared_12k": len(train12), "prepared_32k": len(train32), "negative_outcome_policy": "Mixed source gains/regressions and numerical differences retained in primary evidence and rationale. Correctness-only, copied ports, pending protection and unmeasured changes are excluded from positive rows.", "source_files_sha256": {str(p.relative_to(WORK)): H(p.read_bytes()) for p in [Path(__file__), ROOT / "fetch_batch.py", ROOT / "refined_units.py", WORK / "kernel-library/expansion-v2/tools/source_units.py", WORK / "kernel-library/integration-v1/group-b/prepare_group_b.py", *[WORK / "loss-fix/apple/src/exl3_mlx" / x for x in ("agent_edits.py", "upstream_data.py", "data.py", "tool_data.py")]]}})


if __name__ == "__main__":
    main()
