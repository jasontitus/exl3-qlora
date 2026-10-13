"""Bounded, stdlib-only GitHub discovery. Never executes or promotes upstream code."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

VERSION = "metal-candidate-collector-v1"
MAX_RESPONSE = 8 << 20
SHA = re.compile(r"[a-f0-9]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
PERFORMANCE = re.compile(
    r"\b(perf(?:ormance)?|optimi[sz]\w*|faster|speed\w*|latency|throughput|bandwidth|fusion|fused|vectori[sz]\w*|simd\w*|cache|memory|bottleneck)\b",
    re.IGNORECASE,
)
METAL = re.compile(
    r"\b(metal|mlx|apple|m[1-9](?:\s+(?:ultra|max|pro))?|simdgroup|nax|msl)\b",
    re.IGNORECASE,
)
METRIC = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|×|x\b|ms\b|us\b|µs\b|ns\b|tok(?:en)?s?/s\b|GB/s\b|TFLOPS\b)",
    re.IGNORECASE,
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    ).encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def exclusive(path, raw):
    """Complete bytes become visible atomically, never replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_sources(path):
    raw = path.read_bytes()
    value = strict_json(raw)
    repositories = value.get("repositories") if isinstance(value, dict) else None
    if not isinstance(repositories, list) or not 1 <= len(repositories) <= 25:
        raise ValueError("Sources must contain 1..25 repositories")
    seen, ids = set(), set()
    for row in repositories:
        if not isinstance(row, dict) or not re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{0,79}", row.get("id", "")
        ):
            raise ValueError("Invalid repository ID")
        repo = row.get("repository", "")
        if not REPOSITORY.fullmatch(repo) or any(
            x in {".", ".."} for x in repo.split("/")
        ):
            raise ValueError("Invalid GitHub owner/repository")
        if row.get("url") != "https://github.com/" + repo:
            raise ValueError("Repository URL does not match exact GitHub identity")
        if repo.casefold() in seen or row["id"] in ids:
            raise ValueError("Duplicate source identity")
        seen.add(repo.casefold())
        ids.add(row["id"])
    return repositories, raw


def classify(message, paths=()):
    performance, metal = bool(PERFORMANCE.search(message)), bool(METAL.search(message))
    touched = [
        p
        for p in paths
        if p.lower().endswith((".metal", ".metal.h")) or "metal" in p.lower().split("/")
    ]
    return {
        "candidate": bool((performance and metal) or touched),
        "performance_message_match": performance,
        "metal_message_match": metal,
        "metal_paths": touched,
        "classification": "heuristic_discovery_not_quality_verdict",
    }


def normalize_commit(repository, row, *, detail=False):
    sha = row.get("sha") if isinstance(row, dict) else None
    commit = row.get("commit") if isinstance(row, dict) else None
    parents = row.get("parents") if isinstance(row, dict) else None
    if (
        not isinstance(sha, str)
        or not SHA.fullmatch(sha)
        or not isinstance(commit, dict)
    ):
        raise ValueError("Invalid commit identity")
    message = commit.get("message")
    if (
        not isinstance(message, str)
        or not isinstance(parents, list)
        or any(
            not isinstance(p, dict)
            or not isinstance(p.get("sha"), str)
            or not SHA.fullmatch(p["sha"])
            for p in parents
        )
    ):
        raise ValueError("Missing complete message or parent identities")
    files = row.get("files") if detail else None
    if detail and not isinstance(files, list):
        raise ValueError("Commit detail lacks files list")
    changes = []
    for f in files or []:
        if not isinstance(f, dict) or not isinstance(f.get("filename"), str):
            raise TypeError("Invalid changed-file entry")
        changes.append(
            {
                key: f.get(key)
                for key in (
                    "filename",
                    "previous_filename",
                    "status",
                    "sha",
                    "additions",
                    "deletions",
                    "changes",
                    "patch",
                    "raw_url",
                    "blob_url",
                )
            }
        )
    return {
        "repository": repository,
        "sha": sha,
        "parents": [p["sha"] for p in parents],
        "message": message,
        "author": commit.get("author"),
        "committer": commit.get("committer"),
        "html_url": row.get("html_url"),
        "changed_files": changes,
        "classification": classify(message, [f["filename"] for f in changes]),
        "source_reported_metric_quotes": [
            line for line in message.splitlines() if METRIC.search(line)
        ],
        "reported_metrics_reproduced": False,
        "training_eligible": False,
        "validation_status": "unreviewed",
        "detail_fetched": detail,
        "protected_overlap_check_pending": True,
        "protected_final_heldout": repository.casefold() == "ggml-org/llama.cpp",
        "cross_repository_deduplication": "not_performed",
        "missing_or_unverified": {
            "full_before_after_source": True,
            "license_review": True,
            "fork_lineage_deduplication": True,
            "protected_evaluation_overlap": True,
            "patch_completeness_verified": False,
            "missing_patch_paths": [
                f["filename"] for f in changes if f["patch"] is None
            ],
            "patches_may_be_truncated": bool(detail),
            "compile_correctness_benchmarks": "not_run",
        },
    }


def transport(url, timeout):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": VERSION,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, dict(response.headers), response.read(MAX_RESPONSE + 1)


class Requests:
    def __init__(self, run, *, budget, timeout, max_seconds, requester):
        self.run, self.budget, self.timeout = run, budget, timeout
        self.deadline, self.requester = time.monotonic() + max_seconds, requester
        self.receipts, self.rate_limited = [], False

    @property
    def available(self):
        return (
            len(self.receipts) < self.budget
            and not self.rate_limited
            and time.monotonic() < self.deadline
        )

    def fetch(self, repository, kind, endpoint):
        if not self.available:
            return None, {
                "ok": False,
                "error": "request_budget_rate_limit_or_time_limit",
                "request_made": False,
            }
        url = "https://api.github.com/repos/" + repository + endpoint
        number = len(self.receipts) + 1
        receipt = {
            "number": number,
            "repository": repository,
            "kind": kind,
            "url": url,
            "request_made": True,
            "ok": False,
            "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        }
        self.receipts.append(receipt)
        try:
            status, headers, raw = self.requester(url, self.timeout)
            headers = {k.lower(): v for k, v in headers.items()}
            overflow = len(raw) > MAX_RESPONSE
            raw = raw[:MAX_RESPONSE]
            body = self.run / "responses" / f"{number:03d}.body"
            exclusive(body, raw)
            receipt.update(
                status=status,
                response_headers={
                    k: v
                    for k, v in headers.items()
                    if k
                    in {
                        "etag",
                        "link",
                        "date",
                        "x-ratelimit-remaining",
                        "x-ratelimit-reset",
                        "x-github-request-id",
                        "retry-after",
                    }
                },
                raw_path=str(body.resolve()),
                raw_sha256=digest(raw),
                raw_bytes=len(raw),
                response_truncated=overflow,
            )
            if headers.get("x-ratelimit-remaining") == "0" or status == 429:
                self.rate_limited = True
            if status != 200 or overflow:
                raise ValueError(
                    f"HTTP {status}"
                    if status != 200
                    else "Response exceeded bounded read; saved prefix only"
                )
            value = strict_json(raw)
            receipt["ok"] = True
            return value, receipt
        except Exception as error:  # noqa: BLE001 -- every API/row failure must enter the receipt ledger.
            receipt["error"] = f"{type(error).__name__}: {error}"
            return None, receipt
        finally:
            exclusive(self.run / "responses" / f"{number:03d}.json", encode(receipt))


def repo_key(repository):
    return (
        repository.casefold().replace("/", "--")
        + "-"
        + digest(repository.casefold().encode())[:12]
    )


def prior_progress(output):
    progress = {}
    for file in sorted((output / "runs").glob("*/run.json")):
        for row in strict_json(file.read_bytes()).get("repositories", []):
            progress[row["repository"].casefold()] = row
    return progress


def collect(
    sources,
    output,
    *,
    request_budget=50,
    max_commits=100,
    max_details=5,
    timeout=20,
    max_seconds=240,
    requester=transport,
):
    if type(request_budget) is not int or not 1 <= request_budget <= 50:
        raise ValueError("Request budget must be 1..50")
    if (
        type(max_commits) is not int
        or not 1 <= max_commits <= 100
        or type(max_details) is not int
        or not 0 <= max_details <= 5
    ):
        raise ValueError("max_commits must be 1..100 and max_details 0..5")
    if not 1 <= timeout <= 60 or not 1 <= max_seconds <= 3600:
        raise ValueError("Invalid bounded timeout")
    repositories, source_raw = load_sources(sources)
    output = output.resolve()
    if output.is_relative_to(sources.resolve()):
        raise ValueError("Output cannot replace sources")
    previous = prior_progress(output)
    run_id = (
        dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8]
    )
    run = output / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    exclusive(run / "sources.json", source_raw)
    exclusive(run / "collector.py", Path(__file__).read_bytes())
    requests = Requests(
        run,
        budget=request_budget,
        timeout=timeout,
        max_seconds=max_seconds,
        requester=requester,
    )
    statuses, added = [], {"summaries": 0, "details": 0, "candidate_records": 0}

    def store(row, receipt, kind):
        folder = output / "commits" / repo_key(row["repository"]) / row["sha"]
        row = {
            **row,
            "evidence": {
                "response_receipt": str(
                    (run / "responses" / f"{receipt['number']:03d}.json").resolve()
                ),
                "raw_sha256": receipt["raw_sha256"],
            },
        }
        target = folder / (kind + ".json")
        try:
            exclusive(target, encode(row))
            added["summaries" if kind == "summary" else "details"] += 1
        except FileExistsError:
            old = strict_json(target.read_bytes())
            if (
                old["sha"] != row["sha"]
                or old["repository"].casefold() != row["repository"].casefold()
            ):
                raise ValueError("Existing commit storage identity mismatch")
        if row["classification"]["candidate"]:
            try:
                exclusive(
                    output
                    / "candidates"
                    / repo_key(row["repository"])
                    / row["sha"]
                    / (kind + ".json"),
                    encode(row),
                )
                added["candidate_records"] += 1
            except FileExistsError:
                pass

    def ingest(status, values, receipt):
        if not isinstance(values, list):
            raise TypeError("Commit listing is not an array")
        normalized = [normalize_commit(status["repository"], row) for row in values]
        for row in normalized:
            store(row, receipt, "summary")
        return normalized

    # Inventory EVERY repository before spending requests on detail enrichment.
    for source in repositories:
        repo = source["repository"]
        old = previous.get(repo.casefold(), {})
        status = {
            "id": source["id"],
            "repository": repo,
            "status": "pending",
            "errors": [],
            "history": old.get("history"),
            "latest_head": old.get("latest_head"),
            "unresolved_latest_gap": bool(old.get("unresolved_latest_gap", False)),
            "latest_gaps": list(old.get("latest_gaps", [])),
            "details_fetched": 0,
            "inventory_attempted": True,
        }
        statuses.append(status)
        metadata, receipt = requests.fetch(repo, "metadata", "")
        status["inventory_attempted"] = receipt["request_made"]
        if not receipt["ok"]:
            status.update(status="inventory_failed", errors=[receipt])
            continue
        branch = metadata.get("default_branch") if isinstance(metadata, dict) else None
        if not isinstance(branch, str) or not branch:
            status.update(status="inventory_failed", errors=["Missing default branch"])
            continue
        status["default_branch"] = branch
        status["metadata_evidence"] = receipt
        values, receipt = requests.fetch(
            repo,
            "latest_commits",
            "/commits?"
            + urllib.parse.urlencode(
                {"sha": branch, "per_page": max_commits, "page": 1}
            ),
        )
        if not receipt["ok"]:
            status.update(status="inventory_failed", errors=[receipt])
            continue
        try:
            commits = ingest(status, values, receipt)
            head = commits[0]["sha"] if commits else None
            status.update(
                status="inventoried",
                latest_count=len(commits),
                latest_head=head,
                latest_evidence=receipt,
            )
            if (
                old.get("latest_head")
                and old["latest_head"] not in {c["sha"] for c in commits}
                and head != old["latest_head"]
            ):
                status["unresolved_latest_gap"] = True
                status["latest_gaps"].append(
                    {
                        "pinned_new_head": head,
                        "previous_head": old["latest_head"],
                        "next_page": 2,
                        "per_page": max_commits,
                        "status": "backfill_pending",
                    }
                )
            if not status["history"] and head:
                status["history"] = {
                    "pinned_head": head,
                    "next_page": 2,
                    "per_page": max_commits,
                    "complete": 'rel="next"'
                    not in receipt.get("response_headers", {}).get("link", ""),
                }
        except Exception as error:  # noqa: BLE001 -- every API/row failure must enter the receipt ledger.
            status.update(status="inventory_failed")
            status["errors"].append(str(error))

    # Subsequent runs advance stable, pinned-head pages rather than repeat page1 forever.
    for status in statuses:
        history = status["history"]
        old = previous.get(status["repository"].casefold(), {})
        if (
            status["status"] != "inventoried"
            or not history
            or history["complete"]
            or not old.get("history")
        ):
            continue
        endpoint = "/commits?" + urllib.parse.urlencode(
            {
                "sha": history["pinned_head"],
                "per_page": history["per_page"],
                "page": history["next_page"],
            }
        )
        values, receipt = requests.fetch(
            status["repository"], "historical_commits", endpoint
        )
        if not receipt["ok"]:
            status["errors"].append({"historical_backfill": receipt})
            continue
        try:
            status["history_rows_this_run"] = len(ingest(status, values, receipt))
            status["history"] = {
                **history,
                "next_page": history["next_page"] + 1,
                "complete": 'rel="next"'
                not in receipt.get("response_headers", {}).get("link", ""),
            }
        except Exception as error:  # noqa: BLE001 -- every API/row failure must enter the receipt ledger.
            status["errors"].append({"historical_backfill": str(error)})

    queues = {}
    for status in statuses:
        folder = output / "commits" / repo_key(status["repository"])
        pending = [
            p
            for p in folder.glob("*/summary.json")
            if not (p.parent / "detail.json").exists()
        ]

        def priority(path):
            row = strict_json(path.read_bytes())
            flags = row["classification"]
            return (
                -int(flags["candidate"]),
                -int(flags["performance_message_match"]),
                path.parent.name,
            )

        queues[status["repository"]] = sorted(pending, key=priority)
        status["pending_detail_count_before"] = len(pending)
    for turn in range(max_details):
        for status in statuses:
            queue = queues[status["repository"]]
            if turn >= len(queue) or not requests.available:
                continue
            sha = queue[turn].parent.name
            value, receipt = requests.fetch(
                status["repository"],
                "commit_detail",
                f"/commits/{sha}?per_page=100&page=1",
            )
            if not receipt["ok"]:
                status["errors"].append({"commit": sha, "detail": receipt})
                continue
            try:
                row = normalize_commit(status["repository"], value, detail=True)
                if row["sha"] != sha:
                    raise ValueError("Detail SHA differs from requested commit")
                row["missing_or_unverified"]["changed_files_pagination_incomplete"] = (
                    'rel="next"' in receipt.get("response_headers", {}).get("link", "")
                )
                store(row, receipt, "detail")
                status["details_fetched"] += 1
            except Exception as error:  # noqa: BLE001 -- every API/row failure must enter the receipt ledger.
                status["errors"].append({"commit": sha, "detail": str(error)})
    source_changed = sources.read_bytes() != source_raw
    result = {
        "format": VERSION,
        "run_id": run_id,
        "sources_sha256": digest(source_raw),
        "sources_changed_during_run": source_changed,
        "request_budget": request_budget,
        "requests_made": len(requests.receipts),
        "rate_limit_observed": requests.rate_limited,
        "added": added,
        "repositories": statuses,
        "training_eligible": False,
        "unique_candidate_commits_in_store": len(
            {str(p.parent) for p in (output / "candidates").glob("*/*/*.json")}
        ),
        "coverage": "Default-branch latest page plus one pinned-head historical page on subsequent runs. Not exhaustive; commit-detail file pagination/patch truncation, newer-history gaps and pending details remain explicit. Newer-history gaps retain exact anchors for future/manual backfill; only the original pinned history advances automatically. Inventory order is fixed, so repeated small budgets can starve later sources. PR/release/changelog fetching is future enrichment, not performed here.",
        "requests": requests.receipts,
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    exclusive(run / "run.json", encode(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sources.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "store"
    )
    parser.add_argument("--request-budget", type=int, default=50)
    parser.add_argument("--max-commits", type=int, default=100)
    parser.add_argument("--max-details", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=240)
    parser.add_argument("--offline-tests", action="store_true")
    args = parser.parse_args()
    if args.offline_tests:
        suite = unittest.defaultTestLoader.discover(
            str(Path(__file__).resolve().parents[1] / "tests"),
            pattern="test_collect_candidates.py",
        )
        if not suite.countTestCases():
            raise SystemExit("No offline tests found")
        raise SystemExit(
            not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        )
    report = collect(
        args.sources,
        args.output,
        request_budget=args.request_budget,
        max_commits=args.max_commits,
        max_details=args.max_details,
        timeout=args.timeout,
        max_seconds=args.max_seconds,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "run_id",
                    "requests_made",
                    "added",
                    "sources_changed_during_run",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
