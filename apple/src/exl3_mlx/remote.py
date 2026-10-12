"""Bounded HTTP Range extraction of one matrix, never a whole model download."""

import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def _range(url, start, length):
    end = start + length - 1
    sep = "&" if "?" in url else "?"
    req = urllib.request.Request(
        url + f"{sep}exl3_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        expected = f"bytes {start}-{end}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(
            expected
        ):
            raise RuntimeError("Server did not honor the byte range; refusing full-shard download")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("Unexpected HTTP range response length")
    return data


def fetch_linear(repo, revision, shard, key, output, *, max_bytes=32 << 20):
    """Extract .trellis, scales/signs, codebook marker, optional bias into safetensors.

    Requires an immutable 40-character Hub commit. At most max_bytes total tensor
    payload plus a header capped at 16 MiB. Does not access credentials or execute
    repository code. Fails before payload retrieval if a matrix exceeds the budget.
    """
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("revision must be an immutable 40-character commit hash")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    output = Path(output)
    provenance = output.with_suffix(".provenance.json")
    if output.exists() or provenance.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its provenance")
    url = f"https://huggingface.co/{urllib.parse.quote(repo, safe='/')}/resolve/{revision}/"
    url += urllib.parse.quote(shard, safe="/")
    prefix = _range(url, 0, 8)
    length = int.from_bytes(prefix, "little")
    if not 2 <= length <= 16 << 20:
        raise ValueError("Invalid or excessive safetensors header size")
    header = json.loads(_range(url, 8, length))
    suffixes = ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg", "bias", "scale")
    names = [f"{key}.{s}" for s in suffixes if f"{key}.{s}" in header]
    if f"{key}.trellis" not in names:
        raise KeyError(f"{key}.trellis not found in {shard}")
    dtypes = {
        "I16": np.dtype("<i2"),
        "I32": np.dtype("<i4"),
        "F16": np.dtype("<f2"),
        "F32": np.dtype("<f4"),
    }
    total = 0
    for name in names:
        spec = header[name]
        a, b = spec["data_offsets"]
        dtype = dtypes.get(spec["dtype"])
        shape = spec["shape"]
        if dtype is None or any(not isinstance(s, int) or s < 0 for s in shape):
            raise ValueError(f"Unsupported dtype or shape: {name}")
        if not 0 <= a <= b or b - a != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Invalid tensor offsets: {name}")
        total += b - a
    if total > max_bytes:
        raise ValueError(f"Matrix payload {total} exceeds budget {max_bytes}")
    tensors, hashes = {}, {}
    for name in names:
        spec = header[name]
        a, b = spec["data_offsets"]
        data = _range(url, 8 + length + a, b - a) if b > a else b""
        hashes[name] = hashlib.sha256(data).hexdigest()
        tensors[name] = np.frombuffer(data, dtype=dtypes[spec["dtype"]]).reshape(spec["shape"])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={"source_repo": repo, "revision": revision, "shard": shard, "key": key},
    )
    report = {
        "repo": repo,
        "revision": revision,
        "shard": shard,
        "key": key,
        "payload_bytes": total,
        "header_bytes": length,
        "sha256_by_tensor": hashes,
        "output": str(output),
    }
    provenance.write_text(json.dumps(report, indent=2) + "\n")
    return report
