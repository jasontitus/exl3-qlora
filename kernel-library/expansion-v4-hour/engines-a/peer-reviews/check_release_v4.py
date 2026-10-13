"""Rehash the sealed v4 release and independently count its immutable data."""

import collections
import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT.parent / "expansion-v3-hour"
CACHE = {}


def sha(path):
    path = Path(path)
    if path not in CACHE:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        CACHE[path] = digest
    return CACHE[path]


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def identity(row):
    return row["repo"].lower().rstrip("/"), row["commit"]


def check():
    release = read(ROOT / "release.json")
    assert sha(ROOT / "release.json") == "50decf9a6f13be38cf14affad6738f701455422ace04b99a6b83bd2e27edb113"
    assert sha(PRIOR / "release.json") == release["prior_release_sha256"]
    for name, expected in release["files_sha256"].items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha(ROOT / name) == expected, name
    audit_paths = [ROOT / name for name in [
        "corpus-12288-v1/audit.json", "corpus-32768-v1/audit.json",
        "reasoning-library-v2/audit.json", "training-mix-12288-v3/ktune-import.json",
    ]]
    evidence = {}
    for path in audit_paths:
        doc = read(path)
        for name, expected in doc["inputs_sha256"].items():
            assert sha(name) == expected, name
            evidence[name] = expected
        for name, expected in doc.get("artifacts_sha256", {}).items():
            assert sha(path.parent / name) == expected, name
    protected = "0720ead3d696e6e9215e99cbf18dd26e532def8c309add06bbbe21efe1841ad0"
    exact = {}
    current_sets = {}
    for capacity in (12288, 32768):
        directory = ROOT / f"corpus-{capacity}-v1"
        audit = read(directory / "audit.json")
        train = rows(directory / "train.jsonl")
        raw = rows(directory / "train.raw.jsonl")
        test = rows(directory / "test.jsonl")
        provenance = [r for r in rows(directory / "provenance.jsonl") if r["split"] == "train"]
        old = [r for r in rows(PRIOR / f"corpus-{capacity}-v1/provenance.jsonl") if r["split"] == "train"]
        assert len(train) == len(raw) == len(provenance) == audit["train_examples"]
        assert len(test) == 16 and sha(directory / "test.jsonl") == protected
        assert not rows(directory / "valid.jsonl")
        assert max(len(r["input_ids"]) for r in train) <= capacity
        assert all(type(r["loss_start"]) is int and 1 <= r["loss_start"] < len(r["input_ids"])
                   for r in train + test)
        current = {identity(r) for r in provenance}
        previous = {identity(r) for r in old}
        assert len(current) == len(provenance) and previous <= current
        old_rows = rows(PRIOR / f"corpus-{capacity}-v1/train.jsonl")
        new_by_identity = {identity(p): row for p, row in zip(provenance, train, strict=True)}
        assert all(new_by_identity[identity(p)] == row for p, row in zip(old, old_rows, strict=True))
        supervised = sum(len(r["input_ids"]) - r["loss_start"] for r in train)
        assert supervised == audit["supervised_tokens"]
        exact[str(capacity)] = {"rows": len(train), "new_commits": len(current - previous),
                                "retained_prior_rows_exact": len(previous),
                                "max_serialized_ids": max(len(r["input_ids"]) for r in train),
                                "supervised_tokens": supervised}
        current_sets[capacity] = new_by_identity
    assert all(current_sets[32768][key] == row for key, row in current_sets[12288].items())
    assert exact["12288"]["rows"] == 159 and exact["12288"]["new_commits"] == 48
    assert exact["32768"]["rows"] == 226 and exact["32768"]["new_commits"] == 72
    reasoning = ROOT / "reasoning-library-v2"
    assert len(rows(reasoning / "train.messages.jsonl")) == len(rows(reasoning / "train.prompt-completion.jsonl")) == 76
    mix = ROOT / "training-mix-12288-v3"
    manifest = read(mix / "ktune-import.json")
    mixed = rows(mix / "train.prepared.jsonl")
    provenance = [r for r in rows(mix / "provenance.jsonl") if r["split"] == "train"]
    assert len(mixed) == len(rows(mix / "train.jsonl")) == len(provenance) == 198
    assert mixed[:159] == rows(ROOT / "corpus-12288-v1/train.jsonl")
    assert sha(mix / "test.prepared.jsonl") == protected and len(rows(mix / "test.jsonl")) == 16
    kinds = collections.Counter(r["kind"] for r in provenance)
    assert kinds == {"exact_edit": 159, "diagnostic": 39}
    assert 39 <= 159 // 4 and 2 * len(mixed) == release["planned_updates"] == 396
    aliases_path = ROOT / "mlx/peer-reviews/known-family-aliases-v2.json"
    aliases = {}
    for group in read(aliases_path)["groups"]:
        for member in group["members"]:
            aliases[identity(member)] = group["id"]
    seen = collections.Counter(aliases.get(identity(r), r["lineage"]) for r in provenance if r["kind"] == "diagnostic")
    assert all(count == 1 for count in seen.values())
    assert manifest["inputs_sha256"][str(aliases_path)] == sha(aliases_path)
    assert release["source_history_complete"] is False
    return {
        "schema": "metal-library-v4-independent-seal-review-v1",
        "status": "passed_scoped_release_integrity_and_count_audit",
        "reviewer": "quality_research",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "release_sha256": sha(ROOT / "release.json"),
        "checks": {"all_release_files_rehashed": len(release["files_sha256"]),
                   "unique_current_audit_input_bindings_rehashed": len(evidence),
                   "all_current_artifact_maps_verified": True,
                   "heldout16_prepared_bytes_exact": True,
                   "prior_exact_training_rows_preserved": True,
                   "12k_exact_rows_identical_subset_of32k": True,
                   "complete_rows_fit_declared_caps_no_truncation": True,
                   "mix_exact_rows_match_reviewed_exact_corpus": True,
                   "no_duplicate_known_diagnostic_exposure_groups": True},
        "exact": exact,
        "new_reasoning_views": 76,
        "mix": {"exact": 159, "diagnostic": 39, "total": 198, "updates": 396,
                "diagnostic_update_fraction": 39 / 198},
        "qualification": release["qualification"],
        "limitations": [
            "This audits sealed data, counts, current evidence hashes, subset preservation and exposure grouping. It does not redo every source semantic review or execute patches.",
            "Historical superseded reports inside evidence retain their historical scopes; this audit does not recursively reinterpret old receipts as current attestations.",
            "No source-history completeness, dependency-closure, universal kernel correctness/performance, training capacity, or model-quality improvement claim.",
            "The source-reviewed excerpts and diagnostic views are historical reconstructions, not observed multi-turn agent trajectories. No model/GPU run occurred here.",
        ],
        "inputs_sha256": {str(ROOT / "release.json"): sha(ROOT / "release.json"),
                          str(PRIOR / "release.json"): sha(PRIOR / "release.json"),
                          str(Path(__file__)): sha(__file__)},
    }


if __name__ == "__main__":
    result = check()
    output = Path(__file__).with_name("release-v4-independent-review.json")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(output), "sha256": sha(output), "checks": result["checks"]}))
