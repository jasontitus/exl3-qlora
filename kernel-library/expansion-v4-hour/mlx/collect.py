"""Bounded public history/source capture only; never execute upstream code."""

import datetime
import hashlib
import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
REPOS = {
    "mlx": "ml-explore/mlx",
    "mlx-lm": "ml-explore/mlx-lm",
    "mlx-vlm": "Blaizzy/mlx-vlm",
}
SELECTION = {
    "mlx": [
        "0512aa6",
        "95f8533",
        "4089fd5",
        "49028fb",
        "ad00ea2",
        "dfe17ba",
        "685dba9",
        "a124ac0",
        "d73eb75",
        "a9eed5a",
        "a1e0e0b",
        "052d428",
        "2414e5d",
        "241f1bf",
        "fa7b5b8",
        "f8aaf49",
    ],
    "mlx-lm": ["11a6ce7", "ee19be4", "9b0063e", "f4f3b57", "e03e45e"],
    "mlx-vlm": ["768ad7d", "5a97f54", "035d6ab"],
}


def H(b):
    return hashlib.sha256(b).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def git(repo, *args):
    cmd = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "gc.auto=0",
        "-c",
        "maintenance.auto=false",
        "-C",
        str(ROOT / "mirrors" / f"{repo}.git"),
        *args,
    ]
    r = subprocess.run(cmd, check=False, capture_output=True, timeout=90)
    if r.returncode:
        raise RuntimeError(r.stderr.decode())
    return r.stdout


def fetch(url, path):
    if path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "source-grounded-metal-training-library",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as f:
        raw = f.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("Public response exceeds4MB bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return json.loads(raw)


def one_repo(repo):
    result = []
    # Content-path search coverage is retained, not just selected titles.
    inventory = git(
        repo, "log", "--format=%H%x00%P%x00%aI%x00%B%x00", "--name-only", "-z"
    ).decode()
    (ROOT / "evidence" / f"{repo}-bounded-history.txt").write_text(inventory)
    for short in SELECTION[repo]:
        commit = git(repo, "rev-parse", short).decode().strip()
        metadata = (
            git(repo, "show", "-s", "--format=%H%x00%P%x00%aI%x00%B", commit)
            .decode()
            .split("\x00")
        )
        parents = metadata[1].split()
        row = {
            "repo": repo,
            "commit": commit,
            "parents": parents,
            "date": metadata[2],
            "message": metadata[3],
            "url": f"https://github.com/{REPOS[repo]}/commit/{commit}",
        }
        try:
            assert len(parents) == 1
            row["parent"] = parents[0]
            case = f"{repo}-{commit[:12]}"
            out = ROOT / "sourcepacks" / case
            out.mkdir(parents=True, exist_ok=True)
            names = (
                git(
                    repo,
                    "diff-tree",
                    "--no-commit-id",
                    "--name-status",
                    "-r",
                    parents[0],
                    commit,
                )
                .decode()
                .splitlines()
            )
            files = {}
            for line in names:
                status, path = line.split("\t")
                assert status in ("M", "A", "D"), line
                assert not path.startswith("/") and ".." not in Path(path).parts
                entry = {"status": status}
                for side, ref, exists in [
                    ("before", parents[0], status != "A"),
                    ("after", commit, status != "D"),
                ]:
                    if exists:
                        blob = git(repo, "show", f"{ref}:{path}")
                        blob.decode("utf-8")
                        p = out / side / path
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(blob)
                        entry[side] = {
                            "path": str(p.relative_to(ROOT)),
                            "sha256": H(blob),
                            "bytes": len(blob),
                        }
                    else:
                        entry[side] = None
                files[path] = entry
            patch = git(
                repo,
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--full-index",
                "--binary",
                parents[0],
                commit,
            )
            (out / "commit.diff").write_bytes(patch)
            license_rows = []
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
                try:
                    body = git(repo, "show", f"{commit}:{name}")
                except RuntimeError:
                    continue
                lp = out / name
                lp.write_bytes(body)
                license_rows.append(
                    {
                        "path": str(lp.relative_to(ROOT)),
                        "sha256": H(body),
                        "upstream_path": name,
                    }
                )
            row.update(
                files=files,
                patch={
                    "path": str((out / "commit.diff").relative_to(ROOT)),
                    "sha256": H(patch),
                },
                licenses=license_rows,
            )
            pr = re.search(r"\(#(\d+)\)", metadata[3].splitlines()[0])
            number = pr[1] if pr else {"768ad7d": "2342", "5a97f54": "2404"}.get(short)
            if number:
                url = f"https://api.github.com/repos/{REPOS[repo]}/pulls/{number}"
                jp = ROOT / "evidence" / f"{repo}-pr-{number}.json"
                data = fetch(url, jp)
                body = data.get("body") or ""
                tp = jp.with_suffix(".txt")
                tp.write_text(body + "\n")
                row["pr"] = {
                    "number": int(number),
                    "url": data["html_url"],
                    "body_path": str(tp.relative_to(ROOT)),
                    "body_sha256": H(tp.read_bytes()),
                    "json_path": str(jp.relative_to(ROOT)),
                    "json_sha256": H(jp.read_bytes()),
                    "state": data["state"],
                    "merged": data["merged"],
                    "head": data["head"]["sha"],
                    "base": data["base"]["sha"],
                }
                if data.get("comments", 0):
                    cp = ROOT / "evidence" / f"{repo}-pr-{number}-comments.json"
                    comments = fetch(
                        f"https://api.github.com/repos/{REPOS[repo]}/issues/{number}/comments?per_page=100",
                        cp,
                    )
                    text = (
                        "\n\n".join(
                            f"COMMENT {c['html_url']} AUTHOR {c['user']['login']} AT {c['created_at']}\n{c.get('body') or ''}"
                            for c in comments
                        )
                        + "\n"
                    )
                    ct = cp.with_suffix(".txt")
                    ct.write_text(text)
                    row["pr"]["comments"] = {
                        "path": str(ct.relative_to(ROOT)),
                        "sha256": H(ct.read_bytes()),
                        "count": len(comments),
                        "reported_count": data["comments"],
                    }
            row["status"] = "source_materialized"
            save(out / "receipt.json", row)
        except Exception as e:  # noqa: BLE001 -- retained capture failures are data
            row.update(status="capture_failed", error=f"{type(e).__name__}: {e}")
        result.append(row)
        save(ROOT / "evidence" / f"{repo}-capture-progress.json", result)
        print(repo, short, row["status"], len(row.get("files", {})), flush=True)
    return result


if __name__ == "__main__":
    ROOT.joinpath("evidence").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = [r for group in pool.map(one_repo, REPOS) for r in group]
    save(
        ROOT / "inventory.json",
        {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "selection": SELECTION,
            "records": records,
            "scope": "Public exact sources and author reports, no locally executed source tests or benchmarks.",
        },
    )
