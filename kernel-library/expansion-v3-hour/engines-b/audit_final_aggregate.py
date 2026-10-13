"""Read-only CPU audit of final exact/reasoning/preference release files."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
sys.path[:0] = [str(WORK / "loss-fix/apple/src"), str(ROOT / "tools")]
from exl3_mlx.data import load_dataset
from reasoning_data import reserved_ids, validate
from pack_exact import checks

MODEL = Path(
    "/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
)
SEEN = {}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if str(path) in SEEN:
        require(SEEN[str(path)] == sha(raw), "Input drift: " + str(path))
    SEEN[str(path)] = sha(raw)
    return raw


def obj(path):
    return json.loads(read(path))


def rows(path):
    return [
        json.loads(line) for line in read(path).decode().splitlines() if line.strip()
    ]


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def check_receipt(folder):
    report = obj(folder / "audit.json")
    for name, digest in report.get("inputs_sha256", {}).items():
        require(sha(read(name)) == digest, "Input receipt mismatch: " + name)
    for name, digest in report.get("artifacts_sha256", {}).items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder), "Artifact path escape")
        require(sha(read(path)) == digest, "Artifact receipt mismatch: " + str(path))
    return report


def norm(repo):
    return repo.lower().rstrip("/")


def main():
    frozen = (
        WORK
        / "loss-fix/apple/artifacts/loss-fix/kernel-corpus-v1/native-code-training-12k/data/test.jsonl"
    )
    require(
        sha(read(frozen))
        == "0720ead3d696e6e9215e99cbf18dd26e532def8c309add06bbbe21efe1841ad0",
        "Historical test changed",
    )
    frozen_rows = rows(frozen)
    protected = reserved_ids()
    exact = {}
    exact_report = {}
    positive_answers = []
    all_sources = {}
    for capacity in (12288, 32768):
        folder = ROOT / f"corpus-{capacity}-v1"
        audit = check_receipt(folder)
        metadata = rows(folder / "provenance.jsonl")
        require(
            len({m["id"] for m in metadata}) == len(metadata), "Duplicate case metadata"
        )
        raw = {}
        prepared = {}
        sources = []
        indexed = {}
        for split in ("train", "test"):
            raw[split] = rows(folder / (split + ".raw.jsonl"))
            prepared[split] = rows(folder / (split + ".jsonl"))
            require(
                all(set(r) == {"tools", "messages"} for r in raw[split]),
                "Non-native raw keys",
            )
            # Real standard loader; no MLX/model loading.
            loaded_raw = load_dataset(
                folder / (split + ".raw.jsonl"), MODEL, seq_length=capacity - 1
            )
            loaded_prepared = load_dataset(
                folder / (split + ".jsonl"), MODEL, seq_length=capacity - 1
            )
            require(
                loaded_raw == loaded_prepared == prepared[split],
                "Raw/prepared IDs or masks differ",
            )
            declared = [m for m in metadata if m["split"] == split]
            require(
                len(declared) == len(raw[split]) == audit[split + "_examples"],
                "Split count mismatch",
            )
            for i, (m, r, p) in enumerate(zip(declared, raw[split], prepared[split])):
                require(m["row"] == i, "Metadata order changed")
                require(
                    m["raw_sha256"] == sha(canonical(r)), "Raw metadata SHA mismatch"
                )
                require(
                    m["prepared_sha256"] == sha(canonical(p)),
                    "Prepared metadata SHA mismatch",
                )
                require(
                    m["serialized_tokens"] == len(p["input_ids"]),
                    "Serialized count mismatch",
                )
                require(
                    m["supervised_tokens"] == len(p["input_ids"]) - p["loss_start"],
                    "Mask count mismatch",
                )
                require(
                    str(Path(m["provenance_file"]).resolve()) in audit["inputs_sha256"],
                    "Unbound source provenance",
                )
                d = obj(m["provenance_file"])
                s = d["source"]
                sources.append(d)
                require(
                    d["id"] == m["id"] and d["split"] == split,
                    "Source row identity mismatch",
                )
                require(
                    (norm(s["repository"]), s["head"]["commit"], s["parent"]["commit"])
                    == (norm(m["repo"]), m["commit"], m["parent"]),
                    "Source commit mismatch",
                )
                if split == "train":
                    require(
                        m["curriculum_tier"]
                        not in {
                            "source_grounded_unmeasured",
                            "rejected",
                            "neutral",
                            "exploratory",
                        },
                        "Excluded tier entered core",
                    )
                    require(
                        s["outcome"] not in {"negative", "rejected", "contradictory"},
                        "Negative source entered positive edit corpus",
                    )
                    require(
                        not s["protected"]
                        and not (
                            (set(s["commits"]) | {m["commit"], m["parent"]}) & protected
                        ),
                        "Protected identity entered train",
                    )
                    indexed[m["id"]] = {
                        "raw": r,
                        "prepared": p,
                        "repo": norm(m["repo"]),
                        "commit": m["commit"],
                        "patch": s["stable_patch_id"],
                        "tier": m["curriculum_tier"],
                    }
                    positive_answers.extend(
                        msg.get("content", "") or ""
                        for msg in r["messages"]
                        if msg["role"] == "assistant"
                    )
                    all_sources[(norm(m["repo"]), m["commit"])] = s
            if split == "test":
                require(
                    prepared[split] == frozen_rows,
                    "Historical heldout IDs/masks differ",
                )
        require(
            (folder / "valid.jsonl").read_bytes() == b"", "Unexpected validation rows"
        )
        require(
            len({(v["repo"], v["commit"]) for v in indexed.values()})
            == len(indexed)
            == audit["distinct_training_commits"],
            "Distinct change count",
        )
        require(
            len({v["patch"] for v in indexed.values()}) == len(indexed),
            "Stable patch duplicate",
        )
        overlap = checks.cross_split_code_report(sources)
        require(
            not overlap["automatic_blocker"] and not overlap["near_duplicate_pairs"],
            "Cross-split source overlap",
        )
        require(
            overlap == obj(folder / "cross-split-code-overlap.json"),
            "Stored overlap differs from recomputation",
        )
        require(
            sum(m["supervised_tokens"] for m in metadata if m["split"] == "train")
            == audit["supervised_tokens"],
            "Supervised total",
        )
        exact[capacity] = indexed
        exact_report[str(capacity)] = {
            "train": len(indexed),
            "test": 16,
            "distinct_changes_normalized_repo": len(indexed),
            "by_tier": dict(Counter(v["tier"] for v in indexed.values())),
            "native_loader_exact": True,
            "historical_heldout_ids_masks_exact": True,
            "source_and_output_hashes_valid": True,
            "cross_split_code_blockers": 0,
            "near_duplicate_pairs": 0,
        }
    require(set(exact[12288]) <= set(exact[32768]), "12K case subset failed")
    require(
        all(value == exact[32768][cid] for cid, value in exact[12288].items()),
        "12K same case content differs at32K",
    )
    reasoning = ROOT / "reasoning-library-v1"
    ra = check_receipt(reasoning)
    pr = check_receipt(reasoning / "prepared-12288")
    rr = rows(reasoning / "provenance.jsonl")
    rawr = rows(reasoning / "train.messages.jsonl")
    pcr = rows(reasoning / "train.prompt-completion.jsonl")
    prepared_reasoning = load_dataset(
        reasoning / "prepared-12288/train.jsonl", MODEL, seq_length=12287
    )
    require(
        load_dataset(reasoning / "train.messages.jsonl", MODEL, seq_length=12287)
        == prepared_reasoning,
        "Reasoning loader roundtrip",
    )
    require(
        len(rr)
        == len(rawr)
        == len(pcr)
        == len(prepared_reasoning)
        == ra["records"]
        == pr["accepted"],
        "Reasoning counts",
    )
    require(
        not pr["excluded"] and pr["truncated"] == 0, "Unexpected reasoning truncation"
    )
    reasoning_by_id = {}
    for r, raw, pc in zip(rr, rawr, pcr):
        for p, h in validate(r, protected).items():
            require(sha(read(p)) == h, "Reasoning evidence hash")
        require(
            raw
            == {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["completion"]},
                ]
            },
            "Reasoning positive projection",
        )
        require(
            pc == {"prompt": r["prompt"], "completion": r["completion"]},
            "Reasoning alternate projection",
        )
        require(r["id"] not in reasoning_by_id, "Duplicate reasoning ID")
        reasoning_by_id[r["id"]] = r
        positive_answers.append(r["completion"])
    normalized_reasoning = {(norm(r["repo"]), r["commit"]) for r in rr}
    require(
        len(normalized_reasoning) == ra["distinct_commits"],
        "Reasoning normalized change count",
    )
    pf = ROOT / "preference-library-v1"
    pa = check_receipt(pf)
    pairs = rows(pf / "preferences.jsonl")
    pm = rows(pf / "provenance.jsonl")
    require(len(pairs) == len(pm) == pa["pairs"], "Preference count")
    require(
        sha(read(pf / "preferences.jsonl")) == pa["pair_file_sha256"],
        "Pair file digest",
    )
    pair_records = {}
    for path in pa["inputs_sha256"]:
        if Path(path).suffix != ".json":
            continue
        try:
            v = json.loads(read(path))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(v, list):
            for row in v:
                if isinstance(row, dict) and "verifier_negative" in row:
                    pair_records[row["id"]] = row
    for i, (pair, meta) in enumerate(zip(pairs, pm)):
        r = pair_records[meta["id"]]
        n = r["verifier_negative"]
        require(meta["row"] == i, "Preference order")
        require(
            pair
            == {
                "prompt": [{"role": "user", "content": r["prompt"]}],
                "chosen": [{"role": "assistant", "content": r["completion"]}],
                "rejected": [{"role": "assistant", "content": n["text"]}],
            },
            "Pair record projection",
        )
        require(
            r["review"].get("negative_rubric_review")
            and n["origin"]
            == "curator_authored_counterexample_not_observed_model_output"
            and n["training_role"] == "verifier_or_preference_only_never_positive_SFT",
            "Synthetic negative review/provenance",
        )
        require(meta["rubric"] == r["rubric"], "Preference rubric changed")
        require(
            meta["id"] in reasoning_by_id
            and reasoning_by_id[meta["id"]]["completion"] == r["completion"],
            "Chosen record differs from positive SFT",
        )
        require(
            not any(n["text"] in answer for answer in positive_answers),
            "Rejected answer leaked into positive SFT",
        )
    normalized_pairs = {(norm(m["repo"]), m["commit"]) for m in pm}
    require(
        len(normalized_pairs) == pa["distinct_changes"], "Preference normalized changes"
    )
    for path, digest in list(SEEN.items()):
        require(sha(Path(path).read_bytes()) == digest, "End-of-audit input drift")
    report = {
        "schema": "final-aggregate-independent-review-v1",
        "reviewer": "adversarial_process",
        "status": "passed_scoped_preparation_audit",
        "exact": exact_report,
        "twelve_k_subset_of_thirtytwo_k_exact_rows": True,
        "reasoning": {
            "records": len(rr),
            "distinct_changes_normalized_repo": len(normalized_reasoning),
            "raw_prepared_standard_loader_exact": True,
            "supervised_tokens": sum(
                len(r["input_ids"]) - r["loss_start"] for r in prepared_reasoning
            ),
        },
        "preferences": {
            "pairs": len(pairs),
            "distinct_changes_normalized_repo": len(normalized_pairs),
            "reviewed_synthetic_negatives": True,
            "rejected_answers_absent_from_positive_SFT": True,
        },
        "exact32k_and_reasoning_union_distinct_changes": len(
            set(all_sources) | normalized_reasoning
        ),
        "blockers": [],
        "limitations": [
            "This verifies released-file integrity, provenance, formatting and review bindings; it does not independently re-establish every source claim.",
            "Source-family dependence remains; distinct commits/views are not independent optimizations.",
            "No training or GPU dispatch was run. Kernel numerical correctness, runtime speed and model learning are not certified.",
            "32K is prepared context capacity, not hardware-qualified training.",
            "Historical16 held-out token IDs/masks are exact. Prior69/97 training raw, tokenizer and serializer were pinned, but original all-training IDs were not stored.",
            "Preference rejected answers are curator-authored, not observed failed rollouts; rubrics are not calibrated automatic graders.",
        ],
        "inputs_sha256": SEEN,
    }
    report["review_script_sha256"] = sha(Path(__file__).read_bytes())
    output = ROOT / "final-aggregate-independent-review.json"
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "inputs_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
