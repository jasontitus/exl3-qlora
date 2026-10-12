"""Text-only Engram hashing and sparse row reads from mtplx MXFP8 banks."""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .storage import relative_file


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, math.isqrt(n) + 1, 2))


def hash_layout(config):
    used, primes, multipliers = set(), [], []
    for layer_id in config["engram_layer_ids"]:
        groups = []
        for _ in range(config["engram_max_ngram_size"] - 1):
            current = config["engram_vocab_size"] - 1
            for _ in range(config["engram_n_heads"]):
                current += 1
                while current in used or not is_prime(current):
                    current += 1
                used.add(current)
                groups.append(current)
        primes.append(groups)
        bound = max(1, (np.iinfo(np.int64).max // config["engram_compressed_vocab_size"]) // 2)
        multipliers.append(
            np.random.default_rng(10007 * layer_id).integers(
                0, bound, size=config["engram_max_ngram_size"], dtype=np.int64
            )
            * 2
            + 1
        )
    primes = np.asarray(primes, dtype=np.int64)
    if primes.sum(axis=1).tolist() != config["engram_num_embeddings"]:
        raise ValueError("Engram prime buckets disagree with checkpoint row counts")
    offsets = np.cumsum(np.pad(primes[:, :-1], ((0, 0), (1, 0))), axis=1)
    return primes, offsets, np.asarray(multipliers)


def hash_ids(ids, token_map, config, layout):
    ids = np.asarray(ids)
    if (
        ids.ndim != 2
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or np.any(ids >= len(token_map))
    ):
        raise ValueError("Expected valid token IDs [batch, sequence]")
    if np.any(ids == config.get("image_token_id", 129264)):
        raise ValueError("Image spans are not supported by the text trainer")
    primes, offsets, multipliers = layout
    tokens = token_map[ids].astype(np.int64)
    past = np.pad(
        tokens,
        ((0, 0), (config["engram_max_ngram_size"] - 1, 0)),
        constant_values=int(token_map[config["engram_pad_token_id"]]),
    )
    end = config["engram_max_ngram_size"] - 1
    history = np.stack(
        [past[:, end - i : end - i + tokens.shape[1]] for i in range(end + 1)], axis=-1
    )
    products = history[:, :, None, :] * multipliers
    rolling, result = products[..., 0], []
    heads = config["engram_n_heads"]
    for n in range(1, config["engram_max_ngram_size"]):
        rolling = np.bitwise_xor(rolling, products[..., n])
        result.append(rolling[..., None] % primes[:, (n - 1) * heads : n * heads])
    return np.concatenate(result, axis=-1) + offsets


class EngramBank:
    def __init__(self, root, config):
        root = Path(root).expanduser()
        self.config, self.layout = config, hash_layout(config)
        raw = (root / "engram-token-map.u32").read_bytes()
        metadata = json.loads((root / "engram-token-map.u32.json").read_text())
        if (
            hashlib.sha256(raw).hexdigest() != metadata["map_sha256"]
            or hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
            != metadata["tokenizer_sha256"]
        ):
            raise ValueError("Engram token map/tokenizer checksum mismatch")
        self.token_map = np.frombuffer(raw, dtype="<u4")
        if (
            self.token_map.size != config["vocab_size"]
            or self.token_map.max() + 1 != config["engram_compressed_vocab_size"]
        ):
            raise ValueError("Engram token map vocabulary mismatch")
        root = root / "engram"
        m = json.loads((root / "engram-manifest.json").read_text())
        if m["format"] != "mtplx-engram-manifest-v1":
            raise ValueError("Unsupported Engram bank")
        self.layers = {}
        self.bytes_read = 0
        for desc in m["layers"]:
            file = relative_file(root, desc["file"])
            stat = file.stat()
            layer = desc["layer_id"]
            idx = config["engram_layer_ids"].index(layer)
            if (
                desc["quant"] != {"bits": 8, "group_size": 32, "mode": "mxfp8", "head_dim": 256}
                or desc["record_bytes"] != 264
                or desc["rows"] != config["engram_num_embeddings"][idx]
                or stat.st_size != desc["rows"] * 264
                or desc["record_layout"]["weight"]
                != {"offset": 0, "length": 256, "dtype": "U32", "shape": [64]}
                or desc["record_layout"]["scales"]
                != {"offset": 256, "length": 8, "dtype": "U8", "shape": [8]}
            ):
                raise ValueError("Unsupported or incomplete Engram row layout")
            self.layers[layer] = file, desc["rows"], (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if set(self.layers) != set(config["engram_layer_ids"]):
            raise ValueError("Missing Engram layers")

    def hashes(self, ids):
        return hash_ids(ids, self.token_map, self.config, self.layout)

    def gather(self, layer, ids):
        file, rows, fingerprint = self.layers[layer]
        stat = file.stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != fingerprint:
            raise RuntimeError("Frozen Engram bank changed")
        unique, inverse = np.unique(ids, return_inverse=True)
        if unique.min() < 0 or unique.max() >= rows:
            raise ValueError("Engram hash outside bank")
        data = np.empty((len(unique), 264), dtype=np.uint8)
        with file.open("rb") as f:
            for i, row in enumerate(unique):
                f.seek(int(row) * 264)
                raw = f.read(264)
                if len(raw) != 264:
                    raise ValueError("Truncated Engram row")
                data[i] = np.frombuffer(raw, dtype=np.uint8)
        self.bytes_read += data.nbytes
        weight = mx.array(data[:, :256].copy().view(np.uint32))
        scale = mx.array(data[:, 256:].copy())
        value = mx.dequantize(weight, scale, group_size=32, bits=8, mode="mxfp8", dtype=mx.bfloat16)
        return value[mx.array(inverse.astype(np.int32))].reshape(*ids.shape, 256)


class Engram(nn.Module):
    def __init__(self, wkv, q_weight, k_weight, *, eps=1e-20):
        super().__init__()
        self.wkv, self.q_weight, self.k_weight = wkv, q_weight, k_weight
        self.eps = eps
        self.freeze()

    def __call__(self, streams, embedding):
        h = streams.astype(mx.float32)
        mult, dim = h.shape[-2:]
        kv = self.wkv(embedding.reshape(*embedding.shape[:-2], -1)).astype(mx.float32)
        key, value = kv[..., : mult * dim].reshape(h.shape), kv[..., mult * dim :]
        rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + self.eps) * mx.rsqrt(
            mx.mean(key * key, axis=-1) + self.eps
        )
        dot = mx.sum(h * self.q_weight * self.k_weight * key, axis=-1) * rstd / math.sqrt(dim)
        signed_sqrt = mx.where(dot < 0, -1.0, 1.0) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        return (h + mx.sigmoid(signed_sqrt)[..., None] * value[..., None, :]).astype(streams.dtype)
