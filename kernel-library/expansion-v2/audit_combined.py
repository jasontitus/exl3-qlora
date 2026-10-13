"""CPU-only audit of immutable, already published corpus artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent
OLD = ROOT.parent / "integration-v1/corpus-12288-v3"
MODEL = Path("/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw")
sys.path.insert(0, str(WORK / "loss-fix/apple/src"))
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


inputs = {}


def bind(path, expected=None):
    path = Path(path).resolve()
    observed = digest(path)
    if expected is not None:
        assert observed == expected, (str(path), observed, expected)
    inputs[str(path)] = observed
    return observed


def path_ok(name):
    p = PurePosixPath(name)
    assert not p.is_absolute() and str(p) == name
    assert not ({"..", ".git"} & set(p.parts))
    assert re.fullmatch(r"[A-Za-z0-9_.\-/]+", name)


# Do not execute supplied commands. Parse exactly the one allowed patch grammar.
def replay_independent(row, provenance):
    source = provenance["source"]
    before, after = {}, {}
    omitted = provenance["replay"]["omitted_documentation"]
    for name, pair in source["files"].items():
        path_ok(name)
        for version in ("before", "after"):
            value = pair[version]
            if value is None:
                continue
            raw = value["content"].encode()
            assert sha(raw) == value["sha256"]
            assert (
                hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
                == value["blob"]
            )
        if pair["role"] != "changed":
            continue
        if name in omitted:
            assert Path(name).suffix.lower() in {".md", ".rst"}
            assert omitted[name]["after_sha256"] == pair["after"]["sha256"]
            assert omitted[name]["before_sha256"] == (
                pair["before"]["sha256"] if pair["before"] else None
            )
            continue
        assert pair["after"] is not None
        assert pair["after"]["mode"] == (pair["before"]["mode"] if pair["before"] else "100644")
        a = pair["after"]["content"].encode()
        b = pair["before"]["content"].encode() if pair["before"] else None
        if a != b:
            before[name], after[name] = b, a
    calls = row["messages"][-1]["tool_calls"]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "exec_command"
    args = calls[0]["function"]["arguments"]
    assert set(args) == {"cmd"}
    m = re.fullmatch(
        r"git apply --whitespace=nowarn - <<'(KERNEL_PATCH(?:_[1-9][0-9]*)?)'\n([\s\S]+\n)\1",
        args["cmd"],
    )
    assert m and m[1] not in m[2].splitlines()
    patch = m[2].encode()
    assert sha(patch) == provenance["replay"]["patch_sha256"]
    for name in re.findall(r"^(?:--- a/|\+\+\+ b/)(.*)$", m[2], re.M):
        path_ok(name)
        assert name in before
    with tempfile.TemporaryDirectory(prefix="independent-corpus-") as directory:
        tmp = Path(directory)
        for name, payload in before.items():
            if payload is not None:
                dest = tmp / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
            GIT_TERMINAL_PROMPT="0",
        )
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "apply", "--whitespace=nowarn", "-"],
            cwd=tmp,
            input=patch,
            env=env,
            check=True,
            capture_output=True,
            timeout=30,
        )
        actual = {str(p.relative_to(tmp)): p.read_bytes() for p in tmp.rglob("*") if p.is_file()}
        assert actual == after
    assert source["verification"]["execution_verified"] is False
    assert all(
        provenance["replay"][k] == "not_run" for k in ("compile", "numerical", "performance")
    )
    assert source["parent"]["commit"] in source["head"]["parents"]
    assert sha(source["diff"].encode()) == source["diff_sha256"]
    for evidence in source["evidence"].values():
        assert sha(evidence["text"].encode()) == evidence["sha256"]
    for annotation in source["annotations"].values():
        for quote in annotation.get("evidence", []):
            assert quote["quote"] in source["evidence"][quote["source"]]["text"]
    return {
        "files_replayed": len(after),
        "source_images_hashed": sum(
            p[v] is not None for p in source["files"].values() for v in ("before", "after")
        ),
        "patch_sha256": sha(patch),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--version", default="v1")
args = parser.parse_args()
bind(__file__)
bind(ROOT / "build_combined.py")
bind(ROOT / "summarize_corpus.py")
old_rows = {s: rows(OLD / f"{s}.jsonl") for s in ("train", "test")}
assert len(old_rows["train"]) == 24 and len(old_rows["test"]) == 16
for s in old_rows:
    bind(OLD / f"{s}.jsonl")
old_map = {s: {r["case_id"]: r for r in rs} for s, rs in old_rows.items()}
reserved_file = ROOT.parent / "integration-v1/group-a/evidence/reserved-metadata.json"
bind(reserved_file)
reserved = read(reserved_file)["review"]
reserved_commits = {
    r[k]: r["id"] for r in reserved["coverage"] for k in ("head", "base") if r.get(k)
}
# Fail exact protected identities before the relatively expensive tokenizer and
# replay pass. Curator operation-level clearance cannot override this gate.
for ceiling in (32768, 12288):
    root = ROOT / f"corpus-{ceiling}-{args.version}"
    assert not (root / "REJECTED.json").exists()
    assert not (root / "SUPERSEDED.json").exists()
    for row in rows(root / "train.jsonl"):
        source = read(root / (row["case_id"] + ".provenance.json"))["source"]
        assert (
            not {source["head"]["commit"], source["parent"]["commit"]} & reserved_commits.keys()
        ), row["case_id"]
tokenizer = load_tokenizer(MODEL)
tokenizer_hashes = tokenizer_identity(MODEL)
for name, expected in tokenizer_hashes.items():
    bind(MODEL / name, expected)
results, seen, git_checks, git_seen = [], {}, [], set()
for ceiling in (32768, 12288):
    root = ROOT / f"corpus-{ceiling}-{args.version}"
    report = read(root / "ktune-import.json")
    bind(root / "ktune-import.json")
    preflight = read(root / "reserved-identity-preflight.json")
    bind(root / "reserved-identity-preflight.json")
    assert preflight["matches"] == []
    assert Path(preflight["metadata"]).resolve() == reserved_file.resolve()
    bind(preflight["metadata"], preflight["metadata_sha256"])
    bind(preflight["helper_path"], preflight["helper_sha256"])
    for name, expected in preflight["manifests_sha256"].items():
        assert report["source_manifests"][name] == expected
        bind(name, expected)
    assert report["context_ceiling"] == ceiling and report["truncated"] == 0
    assert report["tokenizer"] == tokenizer_hashes
    for name, expected in report["artifacts_sha256"].items():
        bind(root / name, expected)
    for group in ("source_manifests", "source_evidence", "helper_source_sha256"):
        for name, expected in report[group].items():
            bind(name, expected)
    bind(ROOT.parent / "integration-v1/build_next_corpus.py", report["builder_sha256"])
    policy = read(root / "source-policy.json")
    assert "repository:https://github.com/ggml-org/llama.cpp" in policy["explicit_protected_keys"]
    manifest_cases = {}
    for name in report["source_manifests"]:
        manifest = read(name)
        clone = dict(manifest)
        expected = clone.pop("manifest_sha256")
        assert sha(canonical(clone)) == expected, name
        for c in manifest["candidates"]:
            assert c["id"] not in manifest_cases
            manifest_cases[c["id"]] = (manifest, c, Path(name))
    audit = {a["id"]: a for a in read(root / "audit.json")}
    collected = {s: rows(root / f"{s}.jsonl") for s in ("train", "valid", "test")}
    assert {s: len(v) for s, v in collected.items()} == report["counts"]
    assert not collected["valid"]
    assert (root / "test.jsonl").read_bytes() == (OLD / "test.jsonl").read_bytes()
    per_source = {s: Counter() for s in collected}
    per_kind = Counter()
    id_sets, row_details, patch_keys = {}, [], {}
    for split, records in collected.items():
        id_sets[split] = set()
        for row in records:
            case = row["case_id"]
            assert case not in id_sets[split]
            id_sets[split].add(case)
            p = read(root / (case + ".provenance.json"))
            assert p["split"] == split and p["id"] == case and p["source_id"] == row["source_id"]
            source = p["source"]
            repo = source["repository"].lower().removesuffix(".git").rstrip("/")
            assert (repo == "https://github.com/ggml-org/llama.cpp") == (split == "test")
            per_source[split][repo] += 1
            if split == "train":
                per_kind[source["task_kind"]["value"]] += 1
                assert source["task_kind"]["value"] in {"kernel", "host_dispatch"}
                assert source["outcome"] in {"reported_improvement", "verified_improvement"}
                assert not source["protected"]
                assert set(source["lineage_keys"]).isdisjoint(policy["explicit_protected_keys"]), (
                    case
                )
                assert (
                    "stable_patch:" + source["stable_patch_id"]
                    not in policy["explicit_protected_keys"]
                ), case
            assert {k: p[k] for k in audit[case]} == audit[case]
            assert policy["candidate_reviews"][case]["status"] != "pending"
            manifest, candidate, manifest_path = manifest_cases[case]
            if split == "train":
                assert not {candidate["commit"], candidate["parent"]} & reserved_commits.keys(), (
                    case
                )
            key = source["stable_patch_id"]
            assert key not in patch_keys, (case, patch_keys.get(key))
            patch_keys[key] = case
            if case not in seen:
                tokenized = tokenize_row({k: row[k] for k in ("messages", "tools")}, tokenizer)
                assert len(tokenized["input_ids"]) == p["total_tokens"], case
                assert len(tokenized["input_ids"]) - tokenized["loss_start"] == p["target_tokens"]
                replay = replay_independent(row, p)
                seen[case] = {
                    "row_sha256": sha(canonical(row)),
                    "tokenized_sha256": sha(canonical(tokenized)),
                    "replay": replay,
                    "published_total_tokens": len(tokenized["input_ids"]),
                }
                print("verified", case, len(tokenized["input_ids"]), flush=True)
            else:
                assert sha(canonical(row)) == seen[case]["row_sha256"]
            assert p["total_tokens"] <= ceiling
            row_details.append(
                {
                    "id": case,
                    "repository": repo,
                    "split": split,
                    "total_tokens": p["total_tokens"],
                    "target_tokens": p["target_tokens"],
                    **seen[case],
                }
            )
            if case not in git_seen:
                git_repo = Path(manifest["repositories"][candidate["repository"]]["path"])
                env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
                env.update(
                    GIT_CONFIG_NOSYSTEM="1",
                    GIT_CONFIG_GLOBAL=os.devnull,
                    GIT_NO_LAZY_FETCH="1",
                    GIT_TERMINAL_PROMPT="0",
                )
                checked = 0
                for version, commit in (
                    ("before", source["parent"]["commit"]),
                    ("after", source["head"]["commit"]),
                ):
                    raw = subprocess.run(
                        ["git", "--git-dir", str(git_repo), "cat-file", "commit", commit],
                        env=env,
                        check=True,
                        capture_output=True,
                        timeout=30,
                    ).stdout
                    assert (
                        sha(raw)
                        == source["parent" if version == "before" else "head"]["raw_commit_sha256"]
                    )
                    for name, pair in source["files"].items():
                        image = pair[version]
                        if image is not None:
                            raw = subprocess.run(
                                [
                                    "git",
                                    "--git-dir",
                                    str(git_repo),
                                    "cat-file",
                                    "blob",
                                    image["blob"],
                                ],
                                env=env,
                                check=True,
                                capture_output=True,
                                timeout=30,
                            ).stdout
                            assert raw == image["content"].encode()
                            checked += 1
                git_checks.append(
                    {"id": case, "local_git_commit_objects": 2, "local_git_blob_objects": checked}
                )
                git_seen.add(case)
    assert id_sets["train"].isdisjoint(id_sets["test"])
    train_groups = {r["source_id"] for r in collected["train"]}
    test_groups = {r["source_id"] for r in collected["test"]}
    assert train_groups.isdisjoint(test_groups)
    preserved = {}
    for split in ("train", "test"):
        new = {r["case_id"]: r for r in collected[split]}
        assert set(old_map[split]).issubset(new)
        matches = []
        for case, old in old_map[split].items():
            a = {k: v for k, v in old.items() if k != "source_id"}
            b = {k: v for k, v in new[case].items() if k != "source_id"}
            assert canonical(a) == canonical(b), case
            matches.append(
                {
                    "id": case,
                    "unchanged_payload_sha256": sha(canonical(a)),
                    "metadata_change_allowed": "source_id only",
                }
            )
        if split == "test":
            assert set(old_map[split]) == set(new)
        preserved[split] = matches
    exclusions = read(root / "exclusions.json")
    assert len({e["id"] for e in exclusions}) == len(exclusions)
    assert set(manifest_cases) == set(audit) | {e["id"] for e in exclusions}
    assert set(audit).isdisjoint({e["id"] for e in exclusions})
    assert max(x["total_tokens"] for x in row_details) == report["max_length"]
    results.append(
        {
            "corpus": str(root),
            "report_sha256": digest(root / "ktune-import.json"),
            "counts": report["counts"],
            "repositories_by_split": {s: dict(v) for s, v in per_source.items()},
            "train_task_kinds": dict(per_kind),
            "total_manifest_candidates": len(manifest_cases),
            "new_training_examples_over_prior24": len(collected["train"]) - 24,
            "max_length": report["max_length"],
            "truncated": 0,
            "preserved": preserved,
            "test_jsonl_byte_identical": True,
            "exclusions": exclusions,
            "rows": row_details,
            "unique_train_source_groups": len(train_groups),
            "unique_test_source_groups": len(test_groups),
            "artifact_bindings_checked": len(report["artifacts_sha256"]),
            "source_evidence_bindings_checked": len(report["source_evidence"]),
        }
    )
assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)
for path, expected in inputs.items():
    assert digest(path) == expected, ("input drift", path)
summary = {
    "format": "expansion-corpus-independent-aggregate-audit-v1",
    "status": "passed_scoped_audit",
    "scope": "CPU-only independent tokenizer/mask counts of published rows, parsed fixed-grammar complete non-documentation Git edit replay, exact local Git source objects, manifest/artifact/evidence pins, finite protected identity/admission checks, preserved24 train payloads and byte-identical16 llama heldout rows.",
    "corpora": results,
    "unique_admitted_rows_replayed_and_retokenized": len(seen),
    "local_git_checks": git_checks,
    "inputs_sha256": inputs,
    "limitations": [
        "Source-grounded retrospective synthetic edit imitation, not a real tool trajectory or optimization-discovery evaluation.",
        "Source reports and exact patch replay are not current compilation, numerical correctness or measured performance qualification.",
        "Stable patch dedup does not establish independent optimization families, transitive ancestry or absence from pretraining. All llama rows remain one related evaluation family.",
        "All published rows are within their declared tokenizer ceiling; 32K preparation does not qualify runtime training at32K.",
        "Private protected implementation files were not fetched. Finite captured metadata and public-operation distinctions do not prove universal semantic independence.",
    ],
}
out = ROOT / f"aggregate-independent-audit-{args.version}.json"
with out.open("x") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
print(
    json.dumps(
        {
            "status": summary["status"],
            "unique_rows": len(seen),
            "counts": [r["counts"] for r in results],
        }
    )
)
