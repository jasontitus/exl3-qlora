"""Bounded pinned Git source/PR intake. Does not checkout or execute upstream files."""

import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
URLS = {
    "mlx": "https://github.com/ml-explore/mlx",
    "mlx-lm": "https://github.com/ml-explore/mlx-lm",
    "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
    "mtplx": "https://github.com/youssofal/MTPLX",
}
PRS = {"mlx": [4516, 4596], "mlx-lm": [1559, 1824], "mlx-vlm": [2443, 2468]}
rows = json.loads((ROOT / "selection-draft.json").read_text())


def fetch(repo):
    mirror = ROOT / "mirrors" / f"{repo}.git"
    mirror.mkdir(exist_ok=True)
    base = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=never",
        "-c",
        "protocol.ext.allow=never",
        "-C",
        str(mirror),
    ]
    subprocess.run(base + ["init", "--bare"], check=True, capture_output=True)
    receipts = []
    for row in rows:
        if row["repo"] != repo:
            continue
        cmd = base + ["fetch", "--no-tags", "--depth=2", URLS[repo], row["sha"]]
        p = subprocess.run(cmd, check=False, capture_output=True, timeout=180)
        log = p.stdout + p.stderr
        name = f"{repo}-{row['sha'][:12]}-fetch.log"
        (ROOT / "evidence" / name).write_bytes(log)
        receipts.append(
            {
                "argv": cmd,
                "returncode": p.returncode,
                "log": name,
                "log_sha256": hashlib.sha256(log).hexdigest(),
            }
        )
        if p.returncode:
            raise RuntimeError(log.decode())
    for n in PRS.get(repo, []):
        url = (
            URLS[repo].replace("https://github.com/", "https://api.github.com/repos/")
            + f"/pulls/{n}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "bounded-source-intake",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError("PR metadata bound")
        out = ROOT / "evidence" / f"{repo}-pr-{n}.json"
        out.write_bytes(raw)
        receipts.append(
            {"url": url, "file": str(out), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    (ROOT / "evidence" / f"{repo}-fetch-receipt.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(repo, "fetched", flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for v in ex.map(fetch, URLS):
        pass
