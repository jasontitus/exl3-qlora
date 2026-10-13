from collect import ROOT, H, fetch, save

out = []
for n in [2031, 4241, 4293, 4380]:
    p = ROOT / "evidence" / f"reserved-public-mlx-pr-{n}.json"
    v = fetch(f"https://api.github.com/repos/ml-explore/mlx/pulls/{n}", p)
    out.append(
        {
            "pr": n,
            "title": v["title"],
            "body": v.get("body"),
            "url": v["html_url"],
            "path": str(p),
            "sha256": H(p.read_bytes()),
            "scope": "review-only protection metadata; not rendered into training rows",
        }
    )
save(ROOT / "evidence/reserved-public-scope.json", out)
