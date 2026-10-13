"""Bounded public source intake; never imports or executes upstream code."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import json
import re
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
URLS = {
    "mlx": "https://github.com/ml-explore/mlx",
    "mlx-lm": "https://github.com/ml-explore/mlx-lm",
    "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
    "omlx": "https://github.com/jundot/omlx",
    "vllm-metal": "https://github.com/vllm-project/vllm-metal",
    "mtplx": "https://github.com/youssofal/MTPLX",
}


class Body(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.parts = []
        self.done = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if not self.depth and not self.done and "comment-body" in attrs.get("class", "").split():
            self.depth = 1
        elif self.depth and tag not in {"br", "hr", "img", "input", "source", "wbr", "meta", "link"}:
            self.depth += 1
        if self.depth and tag in {"p", "div", "br", "tr", "pre", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")
        if self.depth and tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if self.depth and tag not in {"br", "hr", "img", "input", "source", "wbr", "meta", "link"}:
            self.depth -= 1
            if not self.depth:
                self.done = True

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)


def fetch(repo):
    rows = [r for r in json.loads((ROOT / "batch1-selection.json").read_text()) if r["repo"] == repo]
    mirror = ROOT / "mirrors" / (repo + ".git")
    mirror.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-c", "protocol.file.allow=never", "-c", "protocol.ext.allow=never", "-C", str(mirror)]
    subprocess.run(cmd + ["init", "--bare"], check=True, capture_output=True)
    receipts = []
    for r in rows:
        argv = cmd + ["fetch", "--no-tags", "--depth=2", URLS[repo], r["sha"]]
        result = subprocess.run(argv, capture_output=True, timeout=180)
        if result.returncode and b"shallow file has changed" in result.stderr:
            result = subprocess.run(argv, capture_output=True, timeout=180)
        log = result.stdout + result.stderr
        p = ROOT / "evidence" / f"{repo}-{r['sha'][:12]}-fetch.log"
        p.write_bytes(log)
        receipts.append({"argv": argv, "returncode": result.returncode, "path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(log).hexdigest()})
        if result.returncode:
            raise RuntimeError(log.decode())
        pr = re.search(r"\(#(\d+)\)", r["message"].splitlines()[0])
        if not pr:
            continue
        n = pr[1]
        url = URLS[repo] + "/pull/" + n
        html = ROOT / "evidence" / f"{repo}-pr-{n}.html"
        if not html.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "bounded-source-intake"})
            with urllib.request.urlopen(req, timeout=40) as f:
                body = f.read(3_000_001)
            if len(body) > 3_000_000:
                raise ValueError("HTML exceeds bound")
            html.write_bytes(body)
        parser = Body()
        parser.feed(html.read_text())
        if not parser.done:
            raise ValueError("PR body missing")
        txt = "".join(parser.parts).strip() + "\n"
        out = html.with_suffix(".txt")
        out.write_text(txt)
        receipts.append({"url": url, "html": str(html.relative_to(ROOT)), "html_sha256": hashlib.sha256(html.read_bytes()).hexdigest(), "text": str(out.relative_to(ROOT)), "text_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "scope": "First PR description extracted from public HTML; no API calls; exact HTML retained."})
        print(repo, r["sha"][:12], "source + PR", flush=True)
    (ROOT / "evidence" / (repo + "-fetch-receipt.json")).write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    repos = sorted({r["repo"] for r in json.loads((ROOT / "batch1-selection.json").read_text())})
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, repos))
