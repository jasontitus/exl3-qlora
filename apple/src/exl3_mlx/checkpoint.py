"""Inspect checkpoint headers and load individual EXL3 linears without CUDA."""

import json
import math
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .linear import EXL3Linear

MARKERS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}


class Checkpoint:
    """Lazy index of local safetensors shards. Only requested tensors are read.

    Reads tensor shapes from headers, not Hugging Face's apparent parameter count
    (packed integer elements do not represent one original parameter each).
    """

    def __init__(self, path):
        path = Path(path).expanduser()
        files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors shards at {path}")
        self.path = path
        self.tensors = {}
        self.total_bytes = sum(p.stat().st_size for p in files)
        for file in files:
            with safe_open(file, framework="np") as f:
                for key in f.keys():
                    if key in self.tensors:
                        raise ValueError(f"Duplicate tensor {key} across checkpoint shards")
                    sl = f.get_slice(key)
                    self.tensors[key] = (file, sl.get_shape(), sl.get_dtype())

    def array(self, key):
        if key not in self.tensors:
            raise KeyError(f"Missing checkpoint tensor: {key}")
        file = self.tensors[key][0]
        with safe_open(file, framework="np") as f:
            return f.get_tensor(key)

    @property
    def linear_keys(self):
        return sorted(k.removesuffix(".trellis") for k in self.tensors if k.endswith(".trellis"))

    def codebook(self, key):
        names = [name for name in MARKERS if key + "." + name in self.tensors]
        if len(names) > 1:
            raise ValueError(f"{key}: ambiguous codebook markers")
        if not names:
            return "3inst"
        name = names[0]
        marker = self.array(key + "." + name)
        if marker.dtype != np.int32 or marker.shape != ():
            raise ValueError(f"{key}: {name} marker must be an int32 scalar")
        if int(marker) & 0xFFFFFFFF != MARKERS[name]:
            raise ValueError(f"{key}: unrecognized {name} codebook multiplier")
        return name

    def _scale(self, key, name, size):
        unpacked = key + "." + name + "h"
        if unpacked in self.tensors:
            value = self.array(unpacked)
            if value.dtype != np.float16 or value.shape != (size,):
                raise ValueError(f"{unpacked}: expected fp16 vector of length {size}")
        else:
            packed = self.array(key + "." + name)
            if packed.dtype != np.int16 or packed.size * 16 != size or packed.ndim != 1:
                raise ValueError(f"{key}.{name}: expected int16 packed sign vector")
            bits = (packed.view(np.uint16)[:, None] >> np.arange(16)) & 1
            value = (1 - bits.astype(np.float32) * 2).reshape(-1).astype(np.float16)
        if not np.isfinite(value).all():
            raise ValueError(f"{key}.{name}: nonfinite scale")
        return mx.array(value)

    def load_linear(self, key, *, chunk_size=512, max_packed_bytes=256 << 20):
        """Load one stored matrix, including per-expert bitrates and old sign vectors."""
        if key + ".scale" in self.tensors:
            raise ValueError("Legacy .scale checkpoints are not supported; refusing to ignore it")
        _, shape, dtype = self.tensors[key + ".trellis"]
        if len(shape) != 3 or dtype != "I16":
            raise ValueError(f"{key}: expected rank-3 int16 trellis, got {shape}, {dtype}")
        if math.prod(shape) * 2 > max_packed_bytes:
            raise ValueError(
                "Packed matrix exceeds max_packed_bytes; increase the explicit load budget"
            )
        k, n = shape[0] * 16, shape[1] * 16
        codebook = self.codebook(key)
        bias = mx.array(self.array(key + ".bias")) if key + ".bias" in self.tensors else None
        return EXL3Linear(
            mx.array(self.array(key + ".trellis")),
            self._scale(key, "su", k),
            self._scale(key, "sv", n),
            codebook=codebook,
            bias=bias,
            chunk_size=chunk_size,
        )

    def summary(self):
        rates = Counter()
        examples = []
        for key in self.linear_keys:
            _, shape, dtype = self.tensors[key + ".trellis"]
            if len(shape) == 3:
                rate = shape[2] / 16
                rates[str(rate)] += 1
                if len(examples) < 10:
                    examples.append(
                        {
                            "key": key,
                            "in": shape[0] * 16,
                            "out": shape[1] * 16,
                            "bpw": rate,
                            "dtype": dtype,
                        }
                    )
            else:
                rates["unsupported_shape"] += 1
        config = (
            self.path / "config.json" if self.path.is_dir() else self.path.parent / "config.json"
        )
        model_type = json.loads(config.read_text()).get("model_type") if config.exists() else None
        return {
            "path": str(self.path),
            "model_type": model_type,
            "full_model_training_supported": False,
            "storage_gib": self.total_bytes / 2**30,
            "tensor_count": len(self.tensors),
            "exl3_matrices": len(self.linear_keys),
            "matrices_by_bpw": dict(rates),
            "examples": examples,
        }
