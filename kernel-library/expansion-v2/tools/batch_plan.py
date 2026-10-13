"""CPU-only candidate planning and publication attrition; never grants admission.

select consumes a local source config: {sources:[{repository:HTTPS GitHub URL,
history:relative JSONL path}]}. History rows require commit/sha, parents, message
and paths/changed_files. Only exact message lines are quoted, never paraphrased.

reconcile consumes sealed upstream manifests and already published corpus dirs.
It can re-tokenize published messages with --model (local tokenizer only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "candidate-batch-plan-v1"
MAX_FILE = 64 << 20
MAX_TOTAL = 256 << 20
MAX_ROWS = 100_000
COMMIT = re.compile(r"[a-f0-9]{40}\Z")
CASE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")
PERF = re.compile(
    r"\b(perf(?:ormance)?|optimi[sz]\w*|faster|speed\w*|latency|throughput|bandwidth|fusion|fused|vectori[sz]\w*|simd\w*|cache|bottleneck)\b",
    re.I,
)
METAL = re.compile(r"\b(metal|mlx|apple|m[1-9](?:\s+(?:ultra|max|pro))?|simdgroup|nax|msl)\b", re.I)
LLAMA = "https://github.com/ggml-org/llama.cpp"


def encode(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("Duplicate JSON key")
            out[key] = value
        return out

    def constant(_):
        raise ValueError("Nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def repository(value):
    if not isinstance(value, str):
        raise ValueError("Repository must be an HTTPS GitHub URL")
    match = re.fullmatch(
        r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?", value
    )
    if not match or any(p in {".", ".."} for p in match.groups()):
        raise ValueError("Repository must be an HTTPS GitHub URL")
    return "https://github.com/" + "/".join(match.groups()).lower()


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.total = 0

    def read(self, path, expected=None):
        path = Path(path).expanduser().resolve()
        if path.stat().st_size > MAX_FILE:
            raise ValueError("Input exceeds byte bound")
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE:
            raise ValueError("Input exceeds byte bound")
        value = sha(raw)
        if expected is not None and value != expected:
            raise ValueError("Input checksum mismatch: " + str(path))
        old = self.hashes.get(str(path))
        if old is not None and old != value:
            raise ValueError("Input changed while reading")
        if old is None:
            self.total += len(raw)
            if self.total > MAX_TOTAL:
                raise ValueError("Total input exceeds byte bound")
        self.hashes[str(path)] = value
        return raw

    def json(self, path, expected=None):
        return strict(self.read(path, expected))

    def verify(self):
        for path, expected in tuple(self.hashes.items()):
            self.read(path, expected)


def manifest(inputs, path):
    path = Path(path).resolve()
    value = inputs.json(path)
    if not isinstance(value, dict) or value.get("format") != "exl3-mlx-upstream-commits-v1":
        raise ValueError("Unsupported upstream manifest")
    bare = dict(value)
    claimed = bare.pop("manifest_sha256", None)
    if claimed != sha(encode(bare)):
        raise ValueError("Manifest canonical checksum mismatch")
    repos = {key: repository(row["url"]) for key, row in value["repositories"].items()}
    if not isinstance(value.get("candidates"), list):
        raise ValueError("Manifest requires candidate list")
    for candidate in value["candidates"]:
        if (
            not CASE.fullmatch(candidate.get("id", ""))
            or candidate.get("repository") not in repos
            or not COMMIT.fullmatch(candidate.get("commit", ""))
        ):
            raise ValueError("Invalid candidate identity")
    return value, repos


def prior_inputs(inputs, paths):
    commits, protected_repos, protected_commits = set(), {LLAMA}, set()
    for path in paths:
        value, repos = manifest(inputs, path)
        commits.update(c["commit"] for c in value["candidates"])
        for key in value.get("protected_keys", []):
            if not isinstance(key, str):
                raise ValueError("Invalid protected key")
            if key.startswith("repository:"):
                repo = repository(key[11:])
                if key != "repository:" + repo:
                    raise ValueError("Protected repository key is not canonical")
                protected_repos.add(repo)
            elif key.startswith("commit:"):
                commit = key[7:]
                if not COMMIT.fullmatch(commit):
                    raise ValueError("Protected commit key is not canonical")
                protected_commits.add(commit)
            elif not (
                re.fullmatch(r"stable_patch:[a-f0-9]{40}", key)
                or re.fullmatch(r"declared_group:[!-~][ -~]{0,255}", key)
            ):
                raise ValueError("Unknown protected key")
        for c in value["candidates"]:
            if c.get("protected") is True:
                protected_commits.add(c["commit"])
    return commits, protected_repos, protected_commits


def exact_lines(message, pattern):
    spans, offset = [], 0
    for line in message.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        if pattern.search(text):
            spans.append(
                {"text": text, "start_character": offset, "end_character": offset + len(text)}
            )
        offset += len(line)
    return spans


def select(config_path, prior=(), *, per_source=20, total=100, seed="expansion-v2"):
    if (
        type(per_source) is not int
        or not 1 <= per_source <= 100
        or type(total) is not int
        or not 1 <= total <= 500
    ):
        raise ValueError("Quota must be1..100 per source and1..500 total")
    if not isinstance(seed, str) or not seed or len(seed) > 200:
        raise ValueError("Require a bounded nonempty selection seed")
    inputs = Inputs()
    config_path = Path(config_path).resolve()
    config = inputs.json(config_path)
    sources = config.get("sources") if isinstance(config, dict) else None
    if not isinstance(sources, list) or not 1 <= len(sources) <= 25:
        raise ValueError("Require1..25 source histories")
    prior_commits, protected_repos, protected_commits = prior_inputs(inputs, prior)
    ledger, pools, known_sources, count = [], defaultdict(list), set(), 0
    for source in sources:
        repo = repository(source["repository"])
        if repo in known_sources:
            raise ValueError("Duplicate repository source")
        known_sources.add(repo)
        name = source["history"]
        if not isinstance(name, str) or not name:
            raise ValueError("Require local history path")
        path = (config_path.parent / name).resolve()
        raw = inputs.read(path)
        seen = set()
        for number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            count += 1
            if count > MAX_ROWS:
                raise ValueError("History row bound exceeded")
            row = strict(line)
            if not isinstance(row, dict):
                raise ValueError("History row must be an object")
            commit = row.get("commit", row.get("sha"))
            if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
                raise ValueError("History commit must be full lowercase SHA")
            if row.get("sha", commit) != commit:
                raise ValueError("Conflicting history commit identity")
            declared = row.get("source_url")
            if declared is not None and repository(declared) != repo:
                raise ValueError("History source differs from configured repository")
            parents, message = row.get("parents"), row.get("message")
            if not isinstance(parents, list) or any(
                not isinstance(x, str) or not COMMIT.fullmatch(x) for x in parents
            ):
                raise ValueError("History parents must be exact commit SHAs")
            if not isinstance(message, str) or len(message.encode()) > 65536:
                raise ValueError("History requires bounded complete message")
            paths = row.get("paths")
            if paths is None and isinstance(row.get("changed_files"), list):
                paths = [p.get("filename") for p in row["changed_files"]]
            if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
                raise ValueError("History requires path inventory")
            record = {
                "repository": repo,
                "commit": commit,
                "parents": parents,
                "history": str(path),
                "line": number,
                "row_sha256": sha(line),
                "message_sha256": sha(message.encode()),
                "message": message,
                "paths": paths,
                "source_claim_lines": exact_lines(message, PERF),
                "training_eligible": False,
                "qualification": "discovery_only_requires_source_and_operation_review",
            }
            reason = None
            if (
                repo in protected_repos
                or commit in protected_commits
                or set(parents) & protected_commits
            ):
                reason = "protected_identity"
            elif commit in prior_commits:
                reason = "already_in_prior_manifest"
            elif commit in seen:
                reason = "duplicate_history_commit"
            elif len(parents) != 1:
                reason = "non_single_parent_requires_review"
            elif not PERF.search(message) or not (
                METAL.search(message)
                or any(
                    p.lower().endswith((".metal", ".metal.h")) or "metal" in p.lower().split("/")
                    for p in paths
                )
            ):
                reason = "no_combined_performance_and_metal_signal"
            seen.add(commit)
            record["reason"] = reason
            ledger.append(record)
            if reason is None:
                record["selection_rank"] = sha(encode([seed, repo, commit]))
                pools[repo].append(record)
    # Fixed source order plus one candidate per round prevents a large repository
    # from consuming the total budget. Ranking is source-order independent.
    for pool in pools.values():
        pool.sort(key=lambda r: (r["selection_rank"], r["commit"]))
    order = sorted(pools, key=lambda r: (sha(encode([seed, r])), r))
    accepted, chosen, per_repo = [], set(), Counter()
    positions = {repo: 0 for repo in order}
    while len(accepted) < total:
        progress = False
        for repo in order:
            if per_repo[repo] >= per_source or len(accepted) >= total:
                continue
            pool = pools[repo]
            while positions[repo] < len(pool):
                row = pool[positions[repo]]
                positions[repo] += 1
                if row["commit"] in chosen:
                    row["reason"] = "shared_commit_alias"
                    continue
                row["reason"] = "selected_for_review"
                chosen.add(row["commit"])
                accepted.append(row)
                per_repo[repo] += 1
                progress = True
                break
        if not progress:
            break
    for row in ledger:
        if row["reason"] is None:
            row["reason"] = (
                "shared_commit_alias" if row["commit"] in chosen else "quota_not_selected"
            )
    inputs.verify()
    return {
        "format": VERSION,
        "mode": "select",
        "seed": seed,
        "per_source_quota": per_source,
        "total_quota": total,
        "input_sha256": inputs.hashes,
        "selected": accepted,
        "ledger": ledger,
        "counts": {
            "history_rows": len(ledger),
            "selected_unique_commits": len(chosen),
            "reasons": dict(Counter(r["reason"] for r in ledger)),
            "by_source": {
                repo: {
                    "history_rows": sum(r["repository"] == repo for r in ledger),
                    "selected": per_repo[repo],
                }
                for repo in sorted(known_sources)
            },
        },
        "scope": "Deterministic finite discovery shortlist, not training admission. No source reconstruction, patch equivalence, license, benchmark, semantic-family or execution clearance. Nonidentity protected keys remain downstream review gates. No code context is cropped or generated.",
    }


def reconcile(manifests, corpora, *, model=None):
    inputs, candidates = Inputs(), {}
    for path in manifests:
        value, repos = manifest(inputs, path)
        for c in value["candidates"]:
            if c["id"] in candidates:
                raise ValueError("Duplicate candidate ID across manifests")
            candidates[c["id"]] = {
                "repository": repos[c["repository"]],
                "commit": c["commit"],
                "declared_outcome": c.get("outcome"),
                "historical_status": c.get("historical_protection_review", {}).get(
                    "status", "not_declared"
                ),
            }
    if not candidates or not corpora:
        raise ValueError("Require manifests and published corpora")
    tokenizer = None
    if model:
        from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity

        tokenizer = load_tokenizer(model)
        token_hashes = tokenizer_identity(model)
        for name, expected in token_hashes.items():
            inputs.read(Path(model) / name, expected)
    summaries = []
    for root in corpora:
        root = Path(root).resolve()
        report = inputs.json(root / "ktune-import.json")
        if report.get("format") != "exl3-agent-kernel-corpus-v1" or report.get("truncated") != 0:
            raise ValueError("Expected nontruncated native agent corpus")
        for name, expected in report["artifacts_sha256"].items():
            if Path(name).name != name:
                raise ValueError("Artifact must be a local filename")
            inputs.read(root / name, expected)
        for group in ("source_manifests", "source_evidence", "helper_source_sha256"):
            for name, expected in report[group].items():
                inputs.read(name, expected)
        for path in manifests:
            if str(Path(path).resolve()) not in report["source_manifests"]:
                raise ValueError("Requested manifest is not bound by corpus")
        if tokenizer and report["tokenizer"] != token_hashes:
            raise ValueError("Corpus tokenizer identity mismatch")
        audits = inputs.json(root / "audit.json")
        exclusion_list = inputs.json(root / "exclusions.json")
        audit = {row["id"]: row for row in audits}
        exclusions = {row["id"]: row for row in exclusion_list}
        if (
            len(audit) != len(audits)
            or len(exclusions) != len(exclusion_list)
            or audit.keys() & exclusions.keys()
        ):
            raise ValueError("Duplicate or contradictory row disposition")
        accepted = {}
        split_counts = {}
        for split in ("train", "valid", "test"):
            values = [
                strict(x) for x in inputs.read(root / f"{split}.jsonl").splitlines() if x.strip()
            ]
            split_counts[split] = len(values)
            for row in values:
                identity = row["case_id"]
                if (
                    identity in accepted
                    or identity not in audit
                    or audit[identity]["split"] != split
                ):
                    raise ValueError("Published rows disagree with audit")
                accepted[identity] = split
                if tokenizer:
                    ids = tokenize_row({k: row[k] for k in ("messages", "tools")}, tokenizer)
                    if (
                        len(ids["input_ids"]) != audit[identity]["total_tokens"]
                        or len(ids["input_ids"]) - ids["loss_start"]
                        != audit[identity]["target_tokens"]
                        or len(ids["input_ids"]) > report["context_ceiling"]
                    ):
                        raise ValueError("Published token counts or context ceiling mismatch")
        if split_counts != report["counts"] or accepted.keys() != audit.keys():
            raise ValueError("Published split counts disagree with report")
        dispositions, per_source = [], defaultdict(Counter)
        for identity, c in candidates.items():
            if identity in accepted:
                row = {
                    "id": identity,
                    **c,
                    "status": "published_" + accepted[identity],
                    "total_tokens": audit[identity]["total_tokens"],
                    "target_tokens": audit[identity]["target_tokens"],
                }
                if (
                    c["historical_status"] in {"pending", "protected"}
                    and accepted[identity] == "train"
                ):
                    raise ValueError("Pending/protected candidate published as train")
            elif identity in exclusions:
                row = {
                    "id": identity,
                    **c,
                    "status": "excluded",
                    "reason": exclusions[identity]["reason"],
                }
            else:
                raise ValueError("Candidate disappeared without a disposition: " + identity)
            dispositions.append(row)
            per_source[c["repository"]][row["status"]] += 1
        summaries.append(
            {
                "corpus": str(root),
                "context_ceiling": report["context_ceiling"],
                "published_counts": split_counts,
                "requested_candidates": len(candidates),
                "actual_published_tokens_verified": tokenizer is not None,
                "counts_by_source": {k: dict(v) for k, v in sorted(per_source.items())},
                "dispositions": dispositions,
            }
        )
    inputs.verify()
    return {
        "format": VERSION,
        "mode": "reconcile",
        "input_sha256": inputs.hashes,
        "corpora": summaries,
        "scope": "Immutable input and attrition accounting. Model tokenization when requested uses local tokenizer only. This does not reconstruct source or independently replay patches, certify protection judgments, or run compilation/numerical/performance/training gates.",
    }


def publish(value, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encode(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    plan = sub.add_parser("select")
    plan.add_argument("--config", required=True)
    plan.add_argument("--prior-manifest", action="append", default=[])
    plan.add_argument("--per-source", type=int, default=20)
    plan.add_argument("--total", type=int, default=100)
    plan.add_argument("--seed", default="expansion-v2")
    plan.add_argument("--output", required=True)
    audit = sub.add_parser("reconcile")
    audit.add_argument("--manifest", action="append", required=True)
    audit.add_argument("--corpus", action="append", required=True)
    audit.add_argument("--model")
    audit.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = (
        select(
            args.config,
            args.prior_manifest,
            per_source=args.per_source,
            total=args.total,
            seed=args.seed,
        )
        if args.mode == "select"
        else reconcile(args.manifest, args.corpus, model=args.model)
    )
    publish(result, args.output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "mode": args.mode,
                "sha256": sha(encode(result) + b"\n"),
            }
        )
    )


if __name__ == "__main__":
    main()
