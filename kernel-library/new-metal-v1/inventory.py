"""Bounded, offline discovery of newer-Metal evidence; never qualifies training rows."""

import datetime
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "metal4": r"\bmetal[ -]?4\b",
    "m5": r"\bm5\b",
    "tensor_primitives": r"tensorops|tensor_ops|MetalPerformancePrimitives|\bMPP\b",
    "neural_accelerator": r"neural accelerator|\bNAX\b",
    "cooperative_tensor": r"cooperative.{0,30}tensor|tensor.{0,30}cooperative",
}


def git(repo, *args):
    return subprocess.check_output(
        ["git", "--git-dir", str(repo), *args],
        text=True,
        timeout=90,
        env={**__import__("os").environ, "GIT_NO_LAZY_FETCH": "1"},
    )


def metadata_roots():
    # Captured revision IDs also cover partial mirrors whose commits lack refs.
    values = set()
    paths = sorted((ROOT / "expansion-v4-hour").glob("*/case-index*.json"))
    paths += sorted((ROOT / "expansion-v4-hour").glob("*/inventory*.json"))
    paths += sorted((ROOT / "expansion-v4-hour").glob("*/selection.json"))
    paths += sorted((ROOT / "expansion-v4-hour").glob("*/metadata-summaries.json"))

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and re.fullmatch("[0-9a-f]{40}", value):
            values.add(value)

    for path in paths:
        if path.stat().st_size <= 16 * 1024 * 1024:
            visit(json.loads(path.read_text()))
    return sorted(values)


def root_metadata(repo, commit):
    """Read one verified local commit object without consulting its parents."""
    if int(git(repo, "cat-file", "-s", commit)) > 1024 * 1024:
        raise ValueError("commit metadata exceeds 1 MiB bound")
    raw = git(repo, "cat-file", "-p", commit)
    header, sep, body = raw.partition("\n\n")
    if not sep:
        raise ValueError("invalid commit object")
    parents = []
    committer = None
    for line in header.splitlines():
        if line.startswith("parent "):
            parent = line[7:]
            if not re.fullmatch("[0-9a-f]{40}", parent):
                raise ValueError("invalid commit parent")
            parents.append(parent)
        elif line.startswith("committer "):
            committer = line.rsplit(" ", 2)[-2:]
    if committer is None or not re.fullmatch("[+-][0-9]{4}", committer[1]):
        raise ValueError("missing/invalid commit time")
    stamp, offset = committer
    minutes = int(offset[1:3]) * 60 + int(offset[3:])
    if offset[0] == "-":
        minutes = -minutes
    tz = datetime.timezone(datetime.timedelta(minutes=minutes))
    date = datetime.datetime.fromtimestamp(int(stamp), tz).isoformat()
    return commit, parents, date, body.strip()


def scan_records(repo, roots):
    fmt = "--format=%H%x1f%P%x1f%cI%x1f%B%x1e"
    walked = subprocess.run(
        ["git", "--git-dir", str(repo), "log", "--all", *roots, "-5000", fmt],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        env={**__import__("os").environ, "GIT_NO_LAZY_FETCH": "1"},
    )
    records = {}
    for record in walked.stdout.split("\x1e"):
        fields = record.strip().split("\x1f", 3)
        if len(fields) == 4:
            sha, parents, date, body = fields
            records[sha] = (sha, parents.split(), date, body)
    if walked.returncode:
        # log/show can both parse unavailable parents. cat-file reads only the
        # exact verified root object, so incomplete mirrors retain these seeds.
        for root in roots:
            records[root] = root_metadata(repo, root)
    return list(records.values()), walked


def main():
    rows, coverage = [], []
    candidates = metadata_roots()
    for repo in sorted((ROOT / "expansion-v4-hour").glob("*/mirrors/*.git")):
        if "metadata-only" in repo.name:
            continue
        try:
            checked = subprocess.run(
                [
                    "git",
                    "--git-dir",
                    str(repo),
                    "cat-file",
                    "--batch-check=%(objectname) %(objecttype)",
                ],
                input="\n".join(candidates) + "\n",
                capture_output=True,
                text=True,
                timeout=90,
                check=True,
                env={**__import__("os").environ, "GIT_NO_LAZY_FETCH": "1"},
            )
            roots = [
                line.split()[0]
                for line in checked.stdout.splitlines()
                if line.endswith(" commit")
            ]
            records, walked = scan_records(repo, roots)
            scanned = len(records)
            for sha, parents, date, body in records:
                tags = [
                    tag
                    for tag, pattern in PATTERNS.items()
                    if re.search(pattern, body, re.IGNORECASE)
                ]
                if tags:
                    # Git history formatting suppresses parents at shallow
                    # boundaries; the raw commit object retains actual ancestry.
                    sha, parents, date, body = root_metadata(repo, sha)
                    rows.append(
                        {
                            "repository_mirror": str(repo),
                            "commit": sha,
                            "parents": parents,
                            "date": date,
                            "message": body,
                            "discovery_tags": tags,
                            "status": "unreviewed_candidate",
                            "minimum_metal_version": None,
                            "minimum_macos_version": None,
                            "supported_gpu_families": None,
                            "validation": {
                                "compile": "not_run",
                                "numerical": "not_run",
                                "performance": "not_run",
                            },
                        }
                    )
            coverage.append(
                {
                    "mirror": str(repo),
                    "commits_scanned": scanned,
                    "limit": 5000,
                    "fresh_fetch": False,
                    "metadata_commit_roots": len(roots),
                    "history_complete": False,
                    "traversal_exit_code": walked.returncode,
                    "traversal_error": walked.stderr.strip() or None,
                    "direct_root_recovery": bool(walked.returncode and roots),
                }
            )
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            coverage.append(
                {"mirror": str(repo), "error": str(exc), "history_complete": False}
            )
    result = {
        "schema": "new-metal-discovery-v1",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scope": "Local history message discovery only; feature mentions do not establish hardware support or optimization benefit.",
        "coverage": coverage,
        "candidates": rows,
        "next_gates": [
            "Reconstruct complete source and host changes at exact parent/revision",
            "Cite primary API and availability evidence; do not infer requirements from commit wording",
            "Check protected-set and lineage overlap before any split",
            "Review rationale and exact agent edit replay",
            "Qualify compiler, numerical and performance independently",
        ],
        "coverage_gaps": [
            "llama.cpp is outside this scan root.",
            "Message search misses source-only feature additions.",
            "Local history is not proof of complete current upstream coverage.",
            "Deduplication and protected-split review are pending.",
        ],
        "active_dataset_modified": False,
    }
    target = Path(__file__).with_name("inventory.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"candidates": len(rows), "mirrors": len(coverage), "output": str(target)}
        )
    )


if __name__ == "__main__":
    main()
