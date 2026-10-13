"""CPU-only peer audit; never executes upstream source or arbitrary command text."""

import collections
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
ROOT = WORK / "kernel-library/expansion-v4-hour"
sys.path[:0] = [str(WORK / "kernel-library/expansion-v3-hour/tools"), str(WORK / "loss-fix/apple/src")]
import pack_exact as pk  # noqa: E402
import reasoning_data as rd  # noqa: E402


def H(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


index = ROOT / "engines-a/case-index.json"
d = json.loads(index.read_text())
reserved = rd.reserved_ids()
tok = pk.load_tokenizer(
    "/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"
)
cases = []
seen = set()
hashes = {}
manifest_path = ROOT / "engines-a/manifest-reviewed.json"
repos = json.loads(manifest_path.read_text())["repositories"]
hashes[str(manifest_path)] = H(manifest_path)


def git(repo, *args):
    return subprocess.run(
        ["git", "--git-dir", repo, *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
    ).stdout


def verify_tree(record, case):
    repo = next(v["path"] for v in repos.values() if v["url"].lower()==record["repository"].lower())
    parents = (
        git(repo, "show", "-s", "--format=%P", record["head"]["commit"])
        .decode()
        .split()
    )
    assert record["parent"]["commit"] in parents
    for side, ref in [
        ("before", record["parent"]["commit"]),
        ("after", record["head"]["commit"]),
    ]:
        entries = git(repo, "ls-tree", "-rz", ref, "--", *record["files"])
        tree = {}
        for line in entries.split(b"\0"):
            if not line:
                continue
            meta, path = line.split(b"\t", 1)
            mode, kind, blob = meta.decode().split()
            tree[path.decode()] = (mode, kind, blob)
        for path, pair in record["files"].items():
            value = pair[side]
            if value is None:
                assert path not in tree
            else:
                assert tree[path] == (value["mode"], "blob", value["blob"])


for c in d["cases"]:
    if not c.get("exact_edit_eligible"):
        continue
    vals = {k: json.loads(Path(p).read_text()) for k, p in c["paths"].items()}
    binds = {str(Path(p).resolve()): H(Path(p)) for p in c["paths"].values()}
    for k, p in c["paths"].items():
        assert H(Path(p)) == c["files_sha256"][k]
    s = vals["source"]
    assert not s["protected"]
    assert not (set(s["commits"]) & reserved)
    assert s["input_metadata"]["historical_protection_review"]["status"] == "cleared"
    assert s["stable_patch_id"] not in seen
    seen.add(s["stable_patch_id"])
    pk.verify_source(s)
    verify_tree(s, c)
    r = pk.verify_edit(s, vals["raw"], vals["replay"])
    t = pk.tokenize_row(vals["raw"], tok)
    assert t == vals["prepared"]
    assert len(t["input_ids"]) == c["total_serialized_ids"]
    assert len(t["input_ids"]) <= 32768
    assert len(t["input_ids"]) - t["loss_start"] == c["supervised_tokens"]
    assert vals["raw"]["messages"][-1]["role"] == "assistant"
    cases.append(
        {
            "id": c["id"],
            "inputs_sha256": binds,
            "serialized_ids": len(t["input_ids"]),
            "tier": c["curriculum_tier"],
            "replay": r["exact_scoped_replay"],
        }
    )
    hashes.update(binds)
    print(c["id"], len(t["input_ids"]), flush=True)
rp = ROOT / "engines-a/reasoning-records.json"
recs = json.loads(rp.read_text())["records"]
rh = []
for r in recs:
    rd.validate(r, reserved)
    rh.append(
        {"id": r["id"], "record_sha256": hashlib.sha256(pk.canonical(r)).hexdigest()}
    )
hashes[str(rp)] = H(rp)
# A separate source quote audit does not certify every upstream claim as experimentally true.
report = {
    "schema": "exact-edit-independent-review-v1",
    "status": "mechanical_pass_pending_semantic_signoff",
    "reviewer": "adversarial_code",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cases": cases,
    "counts": {
        "exact_32k": len(cases),
        "exact_12k": sum(c["serialized_ids"] <= 12288 for c in cases),
        "tiers": dict(collections.Counter(c["tier"] for c in cases)),
    },
    "reasoning_review": {
        "status": "mechanical_quote_checks_pass",
        "records": rh,
        "path": str(rp),
        "sha256": H(rp),
        "semantic_scope": "Quote hashes and schema mechanically checked; semantic review recorded separately.",
    },
    "findings": [],
    "index_snapshot_sha256": H(index),
    "review_script_sha256": H(Path(__file__).resolve()),
    "scope": ["CPU checks of every admitted payload hash, Git tree modes/blob identities, exact full selected edit replay, actual tokenizer IDs/mask, source/evidence checksums and reserved identities. Semantic signoff is separate."],
    "limitations": [
        "Source semantic review and exact reconstruction are not compilation, GPU numerical validation or speed qualification. No local GPU/model/compiler execution was performed.",
        "Historical target-selected source contexts and single synthetic edit actions are imitation data, not recorded autonomous agent trajectories or optimization-discovery evidence.",
        "Finite operation review does not establish globally independent families; existing heldout-family and stable-patch gates still apply at final packaging.",
    ],
    "inputs_sha256": hashes,
}
# Index review metadata may be updated by its owner after this receipt; immutable payloads are case-bound.
(HERE / "engines-a-mechanical-review.json").write_text(json.dumps(report, indent=2) + "\n")
print("PASS", report["counts"], len(recs), flush=True)
