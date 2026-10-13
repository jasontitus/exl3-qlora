"""Bounded metadata publication; never changes source or prepared payloads."""

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]


def H(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = ROOT / "case-index.json"
    r = json.loads(p.read_text())
    previous = []
    for q in (WORK / "kernel-library/expansion-v2/corpus-32768-v3").glob(
        "*.provenance.json"
    ):
        v = json.loads(q.read_text())
        previous.append(
            (
                v["id"],
                v["split"],
                v["source"].get("commit") or v["source"]["head"]["commit"],
                v["source"].get("stable_patch_id"),
            )
        )
    seen = {}
    dups = []
    for c in r["cases"]:
        if c["status"] != "reviewed_exact_edit":
            continue
        for k, v in c["paths"].items():
            if H(Path(v)) != c["files_sha256"][k]:
                raise ValueError("Changed source/payload")
        matches = [
            x for x in previous if x[2] == c["commit"] or x[3] == c["stable_patch_id"]
        ]
        if c["stable_patch_id"] in seen:
            matches.append(("new_batch_duplicate", seen[c["stable_patch_id"]]))
        seen[c["stable_patch_id"]] = c["id"]
        c["prior_duplicate_matches"] = matches
        if matches:
            dups.append({"id": c["id"], "matches": matches})
        if c["outcome_class"] == "neutral":
            c.update(
                exact_edit_eligible=False,
                exact_edit_exclusion_reason="Analytical source-traffic reduction without a runtime measurement; reasoning/archive only.",
            )
        if not c["fits_32k"]:
            c.update(
                exact_edit_eligible=False,
                exact_edit_exclusion_reason="Complete exact edit exceeds32768 serialized IDs; preserve untruncated archive only.",
            )
        if matches:
            c.update(
                exact_edit_eligible=False,
                exact_edit_exclusion_reason="Exact commit/stable-patch duplicate; do not count as a new edit.",
            )
        c["curriculum_tier"] = {
            "positive": "optimization",
            "correctness": "correctness",
            "tradeoff": "correctness",
            "neutral": "neutral",
        }[c["outcome_class"]]
        c["lineage"] = "patch:" + c["stable_patch_id"]
        c["source_group"] = "mlx-derived-framework-training"
        c["review"] = {
            "status": "pending",
            "reason": "Independent review requested; source/raw/prepared/replay are held.",
        }
    r["status"] = "awaiting_independent_review"
    r["format"] = "historical-kernel-curation-index-v1"
    r["held_payloads"] = True
    r["prior_comparison"] = {
        "rows": len(previous),
        "duplicates": dups,
        "source": "released expansion-v2 corpus32768v3, including all16 frozen heldout payloads",
    }
    eligible = [c for c in r["cases"] if c.get("exact_edit_eligible")]
    r["eligible_counts"] = {
        "32k": len(eligible),
        "12k": sum(c["fits_12k"] for c in eligible),
        "outcome": dict(collections.Counter(c["outcome_class"] for c in eligible)),
    }
    p.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["eligible_counts"]))


if __name__ == "__main__":
    main()
