"""Package reviewed exact-edit curricula; no training, shell text or upstream code execution.

All code/build/test edits are replayed; only explicitly reviewed documentation
omissions are accepted. Correctness is never relabeled as a speed improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
sys.path[:0] = [str(WORK / "loss-fix/apple/src"), str(ROOT / "tools")]
from exact_edit_renderer import full_commit_images, parse_command, replay
from exl3_mlx.data import (
    load_tokenizer,
    tokenize_row,
    tokenizer_identity,
    validate_example,
)
from reasoning_data import ALLOWED_REPOS, reserved_ids

spec = importlib.util.spec_from_file_location(
    "corpus_checks", WORK / "kernel-library/integration-v1/build_next_corpus.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()


def bound(path, hashes):
    p = Path(path).resolve()
    if not p.is_relative_to(WORK / "kernel-library"):
        raise ValueError("Input outside library")
    b = p.read_bytes()
    digest = sha(b)
    if str(p) in hashes and hashes[str(p)] != digest:
        raise ValueError("Input changed during binding: " + str(p))
    hashes[str(p)] = digest
    return json.loads(b)


def verify_review(entry, hashes):
    """A status label cannot authorize arbitrary or subsequently changed examples."""
    review = entry.get("review", {})
    path = Path(review.get("receipt", "")).resolve()
    if not path.is_relative_to(WORK / "kernel-library"):
        raise ValueError("Review outside library")
    raw = path.read_bytes()
    if sha(raw) != review.get("sha256"):
        raise ValueError("Review identity mismatch")
    receipt = json.loads(raw)
    if (
        receipt.get("schema") != "exact-edit-independent-review-v1"
        or receipt.get("status") != "approved"
        or not isinstance(receipt.get("reviewer"), str)
        or not receipt["reviewer"].strip()
    ):
        raise ValueError("Invalid independent review receipt")
    cases = receipt.get("cases")
    if not isinstance(cases, list) or any(not isinstance(c, dict) for c in cases):
        raise ValueError("Review case authorization missing")
    ids = [c.get("id") for c in cases]
    if any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate or invalid review case identity")
    selected = [c for c in cases if c["id"] == entry["id"]]
    if len(selected) != 1:
        raise ValueError("Case not authorized by review")
    bindings = selected[0].get("inputs_sha256")
    if not isinstance(bindings, dict):
        raise ValueError("Review input bindings missing")  # noqa: TRY004
    if set(entry.get("paths", {})) != {"source", "raw", "prepared", "replay"}:
        raise ValueError("Require four exact reviewed inputs")
    for name, value in entry["paths"].items():
        inp = Path(value).resolve()
        if not inp.is_relative_to(WORK / "kernel-library"):
            raise ValueError("Reviewed input outside library")
        digest = sha(inp.read_bytes())
        if bindings.get(str(inp)) != digest:
            raise ValueError("Review input identity mismatch: " + name)
        hashes[str(inp)] = digest
    hashes[str(path)] = sha(raw)
    return receipt


def implementation_inputs():
    paths = [
        Path(__file__).resolve(),
        ROOT / "tools/exact_edit_renderer.py",
        ROOT / "tools/reasoning_data.py",
        Path(checks.__file__).resolve(),
        WORK / "kernel-library/sources.json",
        WORK / "kernel-library/integration-v1/group-a/evidence/reserved-metadata.json",
    ]
    paths.extend((WORK / "loss-fix/apple/src/exl3_mlx").glob("*.py"))
    return {str(p): sha(p.read_bytes()) for p in paths}


def verify_source(record):
    """Check captured byte content against both SHA256 and actual Git blob IDs."""
    for path, pair in record["files"].items():
        for side in ("before", "after"):
            value = pair[side]
            if value is None:
                continue
            raw = value["content"].encode()
            oid = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            if sha(raw) != value["sha256"] or oid != value["blob"]:
                raise ValueError("Captured Git blob mismatch: " + path)


def verify_edit(record, raw, prior_receipt):
    before, after = full_commit_images(record)
    for path, entry in prior_receipt.get("omitted_documentation", {}).items():
        if Path(path).suffix.lower() not in {".md", ".rst"}:
            raise ValueError("Non-documentation omission")
        if path not in before or not entry.get("reason"):
            raise ValueError("Unreviewed documentation omission")
        for phase, values in [("before", before), ("after", after)]:
            actual = sha(values[path]) if values[path] is not None else None
            if actual != entry[phase + "_sha256"]:
                raise ValueError("Documentation omission mismatch")
        del before[path], after[path]
    calls = raw["messages"][-1].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("Require one exact edit call")
    function = calls[0]["function"]
    args = function["arguments"]
    args = json.loads(args) if isinstance(args, str) else args
    if function["name"] != "exec_command" or set(args) != {"cmd"}:
        raise ValueError("Unexpected tool contract")
    parse_command(args["cmd"])
    current = replay(args["cmd"], before, after)
    if not current["exact_scoped_replay"]:
        raise ValueError("Replay failed")
    if current["patch_sha256"] != prior_receipt["patch_sha256"]:
        raise ValueError("Changed replay payload")
    return current


def core_tier(tier, outcome, include_unmeasured=False):
    if type(include_unmeasured) is not bool:
        raise ValueError("include_unmeasured must be boolean")
    if tier in {"optimization", "measured_tradeoff"}:
        if outcome != "reported_improvement":
            raise ValueError("Measured curriculum without positive source evidence")
        return True
    if tier == "correctness":
        return True
    if tier == "source_grounded_unmeasured":
        if outcome not in {"source_only", "reported_improvement"}:
            raise ValueError("Unmeasured curriculum cannot override a negative outcome")
        return include_unmeasured
    raise ValueError("Unknown or non-training curriculum tier")


def build(index_paths, model, output, max_length=12288, *, include_unmeasured=False):
    if type(max_length) is not int or not 3 <= max_length <= 32768:
        raise ValueError("Unsupported context")
    out = Path(output)
    if out.exists():
        raise ValueError("Refuse existing destination")
    hashes = implementation_inputs()
    reserved = reserved_ids()
    tokenizer_hashes = tokenizer_identity(model)
    tokenizer = load_tokenizer(model)
    config_path = Path(model).expanduser().resolve() / "config.json"
    config_raw = config_path.read_bytes()
    hashes[str(config_path)] = sha(config_raw)
    config = json.loads(config_raw)
    vocab = config.get("text_config", config)["vocab_size"]
    tests = []
    protected_patches = set()
    base = WORK / "kernel-library/expansion-v2/corpus-12288-v3"
    for p in base.glob("*.provenance.json"):
        d = bound(p, hashes)
        if d["split"] == "test":
            tests.append(d)
            protected_patches.add(d["source"]["stable_patch_id"])
    accepted = []
    excluded = []
    seen_ids = set()
    seen_commits = set()
    seen_patches = set()
    for path in index_paths:
        index = bound(path, hashes)
        for entry in index["cases"]:
            cid = entry["id"]
            if cid in seen_ids:
                raise ValueError("Duplicate ID " + cid)
            seen_ids.add(cid)
            if entry.get("review", {}).get("status") != "approved":
                excluded.append({"id": cid, "reason": "independent_review_pending"})
                continue
            verify_review(entry, hashes)
            source = bound(entry["paths"]["source"], hashes)
            raw = bound(entry["paths"]["raw"], hashes)
            prepared = bound(entry["paths"]["prepared"], hashes)
            replay_before = bound(entry["paths"]["replay"], hashes)
            if source["case_id"] != cid:
                raise ValueError("Case identity mismatch")
            repo = source["repository"].lower().rstrip("/")
            commit = source["head"]["commit"]
            if repo not in ALLOWED_REPOS or source["protected"]:
                raise ValueError("Protected repository/source")
            if set(source["commits"]) & reserved:
                raise ValueError("Reserved exact identity")
            if (
                source["input_metadata"]["historical_protection_review"]["status"]
                != "cleared"
            ):
                raise ValueError("Uncleared semantic protection")
            stable = source["stable_patch_id"]
            if stable in protected_patches:
                raise ValueError("Protected stable patch")
            if (repo, commit) in seen_commits or stable in seen_patches:
                excluded.append({"id": cid, "reason": "duplicate_commit_or_patch"})
                continue
            tier = entry["curriculum_tier"]
            if not core_tier(tier, source["outcome"], include_unmeasured):
                excluded.append(
                    {"id": cid, "reason": "unmeasured_requires_explicit_opt_in"}
                )
                continue
            seen_commits.add((repo, commit))
            seen_patches.add(stable)
            verify_source(source)
            verified_replay = verify_edit(source, raw, replay_before)
            # Exact published key order/template, no silent canonical rewrite here.
            tokens = tokenize_row(raw, tokenizer)
            if tokens != prepared:
                raise ValueError(
                    "Prepared tokens differ from raw serialization: " + cid
                )
            if len(tokens["input_ids"]) > max_length:
                excluded.append(
                    {
                        "id": cid,
                        "reason": "complete_item_over_context",
                        "tokens": len(tokens["input_ids"]),
                    }
                )
                continue
            validate_example(
                tokens, seq_length=max_length - 1, vocab_size=vocab, truncate=False
            )
            # Native tool rows must remain exactly tools/messages; provenance carries IDs.
            row = raw
            accepted.append(
                {
                    "id": cid,
                    "split": "train",
                    "source_id": entry.get("lineage", repo + ":" + commit),
                    "source": source,
                    "raw": row,
                    "prepared": tokens,
                    "replay": verified_replay,
                    "curriculum_tier": tier,
                    "tokens": len(tokens["input_ids"]),
                    "target_tokens": len(tokens["input_ids"]) - tokens["loss_start"],
                    "review": entry["review"],
                }
            )
    overlap = checks.cross_split_code_report([*accepted, *tests])
    if overlap["automatic_blocker"]:
        raise ValueError("Exact heldout code overlap")
    if overlap["near_duplicate_pairs"]:
        raise ValueError("Near overlap requires review")
    for p, digest in hashes.items():
        if sha(Path(p).read_bytes()) != digest:
            raise ValueError("Input drift: " + p)
    if tokenizer_identity(model) != tokenizer_hashes:
        raise ValueError("Tokenizer drift")
    out.mkdir(parents=True)
    for name, items in [
        ("train.jsonl", [r["raw"] for r in accepted]),
        ("train.prepared.jsonl", [r["prepared"] for r in accepted]),
    ]:
        (out / name).write_text(
            "".join(
                json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in items
            )
        )
    for r in accepted:
        (out / (r["id"] + ".provenance.json")).write_text(
            json.dumps(
                {k: v for k, v in r.items() if k not in {"raw", "prepared"}}, indent=2
            )
            + "\n"
        )
    report = {
        "count": len(accepted),
        "by_tier": dict(Counter(r["curriculum_tier"] for r in accepted)),
        "distinct_commits": len(
            {
                (r["source"]["repository"], r["source"]["head"]["commit"])
                for r in accepted
            }
        ),
        "supervised_tokens": sum(r["target_tokens"] for r in accepted),
        "max_length": max_length,
        "include_unmeasured": include_unmeasured,
        "exclusions": excluded,
        "tokenizer": tokenizer_hashes,
        "inputs_sha256": hashes,
        "source_replay": "Complete implementation/build/test changes; explicitly reviewed documentation omissions only",
        "qualification": "Historical reconstructed tool edits, not observed trajectories or locally verified kernel performance",
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "cross-split-code-overlap.json").write_text(
        json.dumps(overlap, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", action="append", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument(
        "--include-unmeasured",
        action="store_true",
        help="Explicit optional source-only curriculum; never certified performance",
    )
    a = p.parse_args()
    r = build(
        a.index,
        a.model,
        a.output,
        a.max_length,
        include_unmeasured=a.include_unmeasured,
    )
    print(json.dumps({k: v for k, v in r.items() if k != "inputs_sha256"}, indent=2))
