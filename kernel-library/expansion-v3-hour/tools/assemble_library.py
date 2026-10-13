"""Combine sealed exact-edit releases into native training data without changing them."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[1]
W = R.parents[1]
sys.path[:0] = [str(W / "loss-fix/apple/src"), str(R / "tools")]
from exact_edit_renderer import parse_command
from exl3_mlx.data import (
    load_tokenizer,
    tokenize_row,
    tokenizer_identity,
    validate_example,
)
from pack_exact import (
    checks,
    core_tier,
    implementation_inputs,
    reserved_ids,
    verify_edit,
    verify_review,
    verify_source,
)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canon(x):
    return json.dumps(
        x, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def load(p, hashes):
    p = Path(p).resolve()
    if not p.is_relative_to(W / "kernel-library"):
        raise ValueError("Input outside library")
    raw = p.read_bytes()
    h = sha(raw)
    if str(p) in hashes and hashes[str(p)] != h:
        raise ValueError("Input changed")
    hashes[str(p)] = h
    return raw


# The previous release is an explicit immutable trust anchor, not any 16-row test file.
BASE_RELEASES = {
    12288: "694df64ca629a3568422e438b52495db89c009731987abf3e5dc1e338f98ed94",
    32768: "4fe2a875647d94dddf7eb857e236ea1995b8d2c14189efba3bda4da19d890b4f",
}


def verify_base(base, capacity, hashes):
    raw = load(base / "ktune-import.json", hashes)
    if sha(raw) != BASE_RELEASES[capacity]:
        raise ValueError("Frozen base release identity mismatch")
    report = json.loads(raw)
    for name, digest in report["artifacts_sha256"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Unsafe base artifact path")
        if sha(load(path, hashes)) != digest:
            raise ValueError("Base artifact checksum mismatch: " + name)
    return report


FROZEN_TEST_PREPARED = (
    W
    / "loss-fix/apple/artifacts/loss-fix/kernel-corpus-v1/native-code-training-12k/data/test.jsonl"
)
FROZEN_TEST_PREPARED_SHA = (
    "0720ead3d696e6e9215e99cbf18dd26e532def8c309add06bbbe21efe1841ad0"
)


def frozen_test_reference(base_report, hashes):
    # The historical v2 release stored raw rows and counts. The original project
    # additionally preserved these16 held-out IDs/masks, before this assembly.
    for name, digest in base_report["helper_source_sha256"].items():
        if Path(name).name not in {"data.py", "tool_data.py"}:
            continue
        if sha(Path(name).read_bytes()) != digest:
            raise ValueError("Original serializer implementation drift")
        if (
            str(Path(name).resolve()) in hashes
            and hashes[str(Path(name).resolve())] != digest
        ):
            raise ValueError("Conflicting serializer identity")
        hashes[str(Path(name).resolve())] = digest
    raw = FROZEN_TEST_PREPARED.read_bytes()
    if sha(raw) != FROZEN_TEST_PREPARED_SHA:
        raise ValueError("Historical heldout prepared identity mismatch")
    hashes[str(FROZEN_TEST_PREPARED)] = FROZEN_TEST_PREPARED_SHA
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 16:
        raise ValueError("Historical heldout prepared count mismatch")
    return rows


def require_exact_prepared(actual, expected):
    if actual != expected:
        raise ValueError("Historical heldout token IDs or loss mask changed")


def addition_entries(audit, hashes):
    entries = {}
    for name, digest in audit["inputs_sha256"].items():
        p = Path(name).resolve()
        # Pack implementation and model-config inputs legitimately live outside the library.
        raw = p.read_bytes()
        if sha(raw) != digest or (str(p) in hashes and hashes[str(p)] != digest):
            raise ValueError("New pack input drift")
        hashes[str(p)] = digest
        if p.suffix != ".json" or not p.is_relative_to(W / "kernel-library"):
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("cases"), list):
            continue
        for e in obj["cases"]:
            if not isinstance(e, dict) or set(e.get("paths", {})) != {
                "source",
                "raw",
                "prepared",
                "replay",
            }:
                continue
            if e["id"] in entries and entries[e["id"]] != e:
                raise ValueError("Ambiguous source index authorization")
            entries[e["id"]] = e
    return entries


def verify_new_row(d, raw, prepared, entries, hashes):
    if d["id"] not in entries:
        raise ValueError("New row missing signed index authorization")
    e = entries[d["id"]]
    verify_review(e, hashes)
    bound = {k: json.loads(load(v, hashes)) for k, v in e["paths"].items()}
    if (
        d["source"] != bound["source"]
        or raw != bound["raw"]
        or prepared != bound["prepared"]
    ):
        raise ValueError("New row differs from independently reviewed payload")
    if d["review"] != e["review"] or d["curriculum_tier"] != e["curriculum_tier"]:
        raise ValueError("New row review/tier mismatch")
    if not core_tier(e["curriculum_tier"], d["source"]["outcome"]):
        raise ValueError("Optional/unmeasured row in core")
    verify_source(d["source"])
    if verify_edit(d["source"], raw, bound["replay"]) != d["replay"]:
        raise ValueError("New row replay receipt mismatch")


def assemble(base, addition, model, output, capacity):
    out = Path(output)
    if out.exists():
        raise ValueError("Refuse overwrite")
    if capacity not in (12288, 32768):
        raise ValueError("Capacity must match release")
    base = Path(base).resolve()
    addition = Path(addition).resolve()
    hashes = implementation_inputs()
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).read_bytes())
    tokenizer = tokenizer_identity(model)
    tok = load_tokenizer(model)
    config_path = Path(model).expanduser().resolve() / "config.json"
    config_raw = config_path.read_bytes()
    hashes[str(config_path)] = sha(config_raw)
    config = json.loads(config_raw)
    vocab = config.get("text_config", config)["vocab_size"]
    reserved = reserved_ids()
    sources = []
    splits = {"train": [], "test": []}
    metadata = []
    seen_commits = set()
    seen_patches = set()
    seen_payloads = set()
    duplicates = []
    base_report = verify_base(base, capacity, hashes)
    frozen_test = frozen_test_reference(base_report, hashes)
    if base_report["tokenizer"] != tokenizer:
        raise ValueError("Base tokenizer mismatch")
    # New pack must have passed all source/replay/review bindings. Recheck its immutable inputs.
    audit = json.loads(load(addition / "audit.json", hashes))
    if audit["max_length"] != capacity or audit["include_unmeasured"]:
        raise ValueError(
            "Recommended core requires matching capacity and measured/correctness admission"
        )
    entries = addition_entries(audit, hashes)
    if (addition / "test.jsonl").exists():
        raise ValueError("Addition must not redefine the frozen test split")
    if audit["tokenizer"] != tokenizer:
        raise ValueError("Tokenizer identity mismatch")
    for folder, origin in [
        (base, "prior-reviewed-v2"),
        (addition, "new-independent-reviewed-v3"),
    ]:
        by_id = {}
        by_patch = {}
        for f in folder.glob("*.provenance.json"):
            d = json.loads(load(f, hashes))
            if d["id"] in by_id:
                raise ValueError("Duplicate provenance ID")
            if f.name != d["id"] + ".provenance.json":
                raise ValueError("Provenance filename identity mismatch")
            by_id[d["id"]] = d
            ps = d["replay"]["patch_sha256"]
            if ps in by_patch:
                raise ValueError("Ambiguous patch provenance")
            by_patch[ps] = d
        for split in ["train", "test"]:
            p = folder / (split + ".jsonl")
            if not p.exists():
                continue
            rawrows = [
                json.loads(l)
                for l in load(p, hashes).decode().splitlines()
                if l.strip()
            ]
            if (
                origin == "prior-reviewed-v2"
                and len(rawrows) != base_report["counts"][split]
            ):
                raise ValueError("Base row count mismatch")
            if (
                origin == "new-independent-reviewed-v3"
                and split == "train"
                and len(rawrows) != audit["count"]
            ):
                raise ValueError("New pack count mismatch")
            for row in rawrows:
                raw = {
                    k: v for k, v in row.items() if k not in {"case_id", "source_id"}
                }
                if set(raw) != {"tools", "messages"}:
                    raise ValueError("Unexpected exact row schema")
                call = raw["messages"][-1]["tool_calls"][0]["function"]
                args = call["arguments"]
                args = json.loads(args) if isinstance(args, str) else args
                patch = sha(parse_command(args["cmd"]).encode())
                d = by_id[row["case_id"]] if "case_id" in row else by_patch[patch]
                if d["split"] != split or d["replay"]["patch_sha256"] != patch:
                    raise ValueError("Raw/provenance mismatch")
                s = d["source"]
                key = (s["repository"].lower().rstrip("/"), s["head"]["commit"])
                stable = s["stable_patch_id"]
                if split == "train":
                    if (
                        set(s["commits"]) | {s["head"]["commit"], s["parent"]["commit"]}
                    ) & reserved or s["protected"]:
                        raise ValueError("Protected training identity")
                    if key in seen_commits or stable in seen_patches:
                        duplicates.append(
                            {"id": d["id"], "reason": "same_commit_or_stable_patch"}
                        )
                        continue
                    seen_commits.add(key)
                    seen_patches.add(stable)
                prepared = tokenize_row(raw, tok)
                if origin == "prior-reviewed-v2" and split == "test":
                    require_exact_prepared(prepared, frozen_test[len(splits["test"])])
                if origin == "new-independent-reviewed-v3":
                    verify_new_row(d, raw, prepared, entries, hashes)
                validate_example(
                    prepared, seq_length=capacity - 1, vocab_size=vocab, truncate=False
                )
                if (
                    len(prepared["input_ids"]) != d.get("tokens", d.get("total_tokens"))
                    or len(prepared["input_ids"]) - prepared["loss_start"]
                    != d["target_tokens"]
                ):
                    raise ValueError("Prior token receipt mismatch")
                identity = sha(canon(prepared))
                if identity in seen_payloads:
                    raise ValueError("Duplicate tokenized row")
                seen_payloads.add(identity)
                splits[split].append((raw, prepared))
                sources.append(d)
                metadata.append(
                    {
                        "split": split,
                        "row": len(splits[split]) - 1,
                        "id": d["id"],
                        "repo": s["repository"],
                        "commit": s["head"]["commit"],
                        "parent": s["parent"]["commit"],
                        "change_lineage": key[0] + ":" + key[1],
                        "split_family": s["input_metadata"].get("source_group"),
                        "curriculum_tier": d.get(
                            "curriculum_tier", "source_reported_optimization"
                        ),
                        "origin": origin,
                        "serialized_tokens": len(prepared["input_ids"]),
                        "supervised_tokens": d["target_tokens"],
                        "raw_sha256": sha(canon(raw)),
                        "prepared_sha256": identity,
                        "provenance_file": str(folder / (d["id"] + ".provenance.json")),
                    }
                )
    protected_patches = {
        d["source"]["stable_patch_id"] for d in sources if d["split"] == "test"
    }
    if seen_patches & protected_patches:
        raise ValueError("Protected stable patch in training")
    if len(splits["test"]) != 16:
        raise ValueError("Frozen heldout count changed")
    overlap = checks.cross_split_code_report(sources)
    if overlap["automatic_blocker"] or overlap["near_duplicate_pairs"]:
        raise ValueError("Heldout code overlap requires review")
    for p, h in hashes.items():
        if sha(Path(p).read_bytes()) != h:
            raise ValueError("Input drift")
    if tokenizer_identity(model) != tokenizer:
        raise ValueError("Tokenizer drift")
    out.mkdir(parents=True)
    for split, pairs in splits.items():
        for suffix, index in [(".raw.jsonl", 0), (".jsonl", 1)]:
            (out / (split + suffix)).write_text(
                "".join(
                    json.dumps(pair[index], ensure_ascii=False) + "\n" for pair in pairs
                )
            )
    (out / "valid.jsonl").write_text("")
    (out / "provenance.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in metadata)
    )
    training = [x for x in metadata if x["split"] == "train"]
    report = {
        "capacity": capacity,
        "train_examples": len(training),
        "test_examples": len(splits["test"]),
        "distinct_training_commits": len(seen_commits),
        "by_repository": dict(Counter(x["repo"] for x in training)),
        "by_curriculum_tier": dict(Counter(x["curriculum_tier"] for x in training)),
        "by_origin": dict(Counter(x["origin"] for x in training)),
        "supervised_tokens": sum(x["supervised_tokens"] for x in training),
        "max_serialized_tokens": max(x["serialized_tokens"] for x in training),
        "duplicates_excluded": duplicates,
        "truncated": 0,
        "inputs_sha256": hashes,
        "tokenizer": tokenizer,
        "qualification": "Preparation, evidence and exact-edit replay qualified; runtime numerical/performance validation tracked separately. Training not launched. 32K preparation does not imply hardware-qualified training. Existing heldout tasks unchanged; raw transport metadata omitted only, tokenization preserved.",
    }
    (out / "cross-split-code-overlap.json").write_text(
        json.dumps(overlap, indent=2) + "\n"
    )
    report["artifacts_sha256"] = {
        f.name: sha(f.read_bytes()) for f in out.iterdir() if f.is_file()
    }
    report["base_release_sha256"] = BASE_RELEASES[capacity]
    report["heldout_prepared_exact"] = True
    report["heldout_prepared_reference_sha256"] = FROZEN_TEST_PREPARED_SHA
    report["prior_training_token_identity_scope"] = (
        "Original raw and tokenizer files plus original data.py/tool_data.py hashes preserved; re-tokenized with exact receipt counts. Historical all-training-row IDs were not stored in the v2 release."
    )
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--addition", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--capacity", type=int, required=True)
    a = p.parse_args()
    r = assemble(a.base, a.addition, a.model, a.output, a.capacity)
    print(json.dumps({k: v for k, v in r.items() if k != "inputs_sha256"}, indent=2))
