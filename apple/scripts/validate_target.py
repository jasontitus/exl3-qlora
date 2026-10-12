"""Reproduce real DeepSeek matrix checks with under 10 MiB of tensor downloads.

Run from apple/: uv run python scripts/validate_target.py
Downloads only missing fixtures; verifies cached tensors against recorded hashes.
"""

import hashlib
import json
from pathlib import Path

from exl3_mlx.checkpoint import Checkpoint
from exl3_mlx.remote import fetch_linear
from exl3_mlx.validation import validate_linear

ROOT = Path(__file__).resolve().parents[1]
REPO = "jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw"
REVISION = "915e1d0de8ee0a99944c5bc88ab802a639c63380"
CASES = {
    "deepseek-v41-wkv": "layers.0.attn.wkv",
    "deepseek-v41-expert0-w1": "layers.0.ffn.experts.0.w1",
    "deepseek-v41-expert0-w2": "layers.0.ffn.experts.0.w2",
}


def main():
    for name, key in CASES.items():
        path = ROOT / "artifacts" / f"{name}.safetensors"
        if not path.exists():
            fetch_linear(REPO, REVISION, "layer-00.safetensors", key, path)
        provenance = json.loads(path.with_suffix(".provenance.json").read_text())
        if (provenance["repo"], provenance["revision"], provenance["key"]) != (REPO, REVISION, key):
            raise ValueError(f"Unexpected cached fixture provenance: {path}")
        ckpt = Checkpoint(path)
        for tensor, expected in provenance["sha256_by_tensor"].items():
            actual = hashlib.sha256(ckpt.array(tensor).tobytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Cached tensor digest mismatch: {tensor}")
        report = validate_linear(ckpt.load_linear(key))
        report.update(repo=REPO, revision=REVISION, key=key)
        out = ROOT / "reports" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
