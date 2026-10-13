import json
from concurrent.futures import ThreadPoolExecutor

from collect import REPOS, ROOT, H, fetch, save

rows = json.loads((ROOT / "inventory.json").read_text())["records"]


def one(row):
    pr = row["pr"]
    meta = json.loads((ROOT / pr["json_path"]).read_text())
    if not meta.get("review_comments"):
        return None
    p = ROOT / "evidence" / f"{row['repo']}-pr-{pr['number']}-review-comments.json"
    data = fetch(
        f"https://api.github.com/repos/{REPOS[row['repo']]}/pulls/{pr['number']}/comments?per_page=100",
        p,
    )
    t = p.with_suffix(".txt")
    t.write_text(
        "\n\n".join(
            f"REVIEW {x['html_url']} AT {x['created_at']} COMMIT {x.get('commit_id')} FILE {x.get('path')}\n{x['body']}"
            for x in data
        )
        + "\n"
    )
    return {
        "repo": row["repo"],
        "commit": row["commit"],
        "pr": pr["number"],
        "reported_count": meta["review_comments"],
        "captured_count": len(data),
        "path": str(t),
        "sha256": H(t.read_bytes()),
        "json_path": str(p),
        "json_sha256": H(p.read_bytes()),
    }


with ThreadPoolExecutor(max_workers=4) as pool:
    out = [v for v in pool.map(one, rows) if v]
save(ROOT / "discussion-index.json", out)
