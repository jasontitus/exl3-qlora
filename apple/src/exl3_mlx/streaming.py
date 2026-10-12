"""Frozen matrices loaded on demand in forward and backward, with bounded caching.

This is an eager, single-threaded training path. It intentionally synchronizes
each decoded column slice before proceeding; it is not an mx.compile function.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .linear import EXL3Linear, _base
from .materialize import materialize


@dataclass
class StreamStats:
    loads: int = 0
    hits: int = 0
    evictions: int = 0
    loaded_bytes: int = 0
    cached_bytes: int = 0
    peak_cached_bytes: int = 0
    forward_calls: int = 0
    backward_calls: int = 0


class MatrixStore:
    """An LRU for frozen packed MLX matrices, backed by a local Checkpoint.

    cache_bytes bounds retained packed tensors, not OS page cache, MLX allocator
    cache, activations, adapter gradients, or a matrix currently in use. Setting
    zero forces a disk read per call. Source shard stat fingerprints are checked
    so backward cannot silently reread different weights after a file replacement.
    """

    def __init__(
        self, checkpoint, *, cache_bytes=64 << 20, chunk_size=512, max_matrix_bytes=256 << 20
    ):
        if cache_bytes < 0 or max_matrix_bytes <= 0:
            raise ValueError("Cache budget must be nonnegative and matrix budget positive")
        if chunk_size < 128 or chunk_size % 128:
            raise ValueError("chunk_size must be a multiple of 128")
        self.checkpoint = checkpoint
        self.cache_bytes, self.chunk_size = cache_bytes, chunk_size
        self.max_matrix_bytes = max_matrix_bytes
        self.stats = StreamStats()
        self._cache = OrderedDict()
        self._fingerprints = {p: self._stat(p) for p in {v[0] for v in checkpoint.tensors.values()}}
        self._files_by_key = {}
        for name, (p, _, _) in checkpoint.tensors.items():
            self._files_by_key.setdefault(name.rsplit(".", 1)[0], set()).add(p)

    @staticmethod
    def _stat(path):
        s = path.stat()
        return s.st_ino, s.st_size, s.st_mtime_ns

    def _check(self, key):
        for path in self._files_by_key.get(key, ()):
            if self._stat(path) != self._fingerprints[path]:
                raise RuntimeError(f"Frozen checkpoint changed during training: {path}")

    def get(self, key):
        self._check(key)
        if key in self._cache:
            self.stats.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        layer = self.checkpoint.load_linear(
            key, chunk_size=self.chunk_size, max_packed_bytes=self.max_matrix_bytes
        )
        mx.eval(layer.parameters())
        size = sum(v.nbytes for _, v in tree_flatten(layer.parameters()))
        self.stats.loads += 1
        self.stats.loaded_bytes += size
        # Evict before retaining the new entry; temporary loading may overlap one matrix.
        while self._cache and self.stats.cached_bytes + size > self.cache_bytes:
            _, (_, old_size) = self._cache.popitem(last=False)
            self.stats.cached_bytes -= old_size
            self.stats.evictions += 1
        if size <= self.cache_bytes:
            self._cache[key] = layer, size
            self.stats.cached_bytes += size
            self.stats.peak_cached_bytes = max(
                self.stats.peak_cached_bytes, self.stats.cached_bytes
            )
        return layer

    def clear(self):
        self._cache.clear()
        self.stats.cached_bytes = 0

    def report(self):
        return asdict(self.stats)

    def linear(self, key):
        return StreamedEXL3Linear(self, key)


class StreamedEXL3Linear(EXL3Linear):
    """A frozen linear holding a key and provider, not weight-array references."""

    def __init__(self, store, key):
        nn.Module.__init__(self)
        _, shape, dtype = store.checkpoint.tensors[key + ".trellis"]
        if len(shape) != 3 or dtype != "I16":
            raise ValueError(f"{key}: unsupported trellis storage")
        self.in_features, self.out_features = shape[0] * 16, shape[1] * 16
        self.bpw = shape[2] / 16
        self.codebook = store.checkpoint.codebook(key)
        self.chunk_size = store.chunk_size
        # Private attributes keep the non-parameter provider out of Module traversal.
        self._store, self._key = store, key

    def __call__(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input [..., {self.in_features}]")
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("Input must be floating point")
        store, key = self._store, self._key

        @mx.custom_function
        def streamed(inp):
            # The registered VJP supplies dX. Detach the *internal* forward so
            # MLX cannot retain every decoded slice on the outer gradient tape.
            inp = mx.stop_gradient(inp)
            base = store.get(key)
            out = _base(
                inp, base.trellis, base.suh, base.svh, base.codebook, base.chunk_size, eager=True
            )
            if "bias" in base:
                out = out + base.bias.astype(inp.dtype)
            mx.eval(out)
            store.stats.forward_calls += 1
            return materialize(out)

        @streamed.vjp
        def streamed_vjp(primals, cotangent, output):
            base = store.get(key)
            dx = _base(
                cotangent,
                base.trellis,
                base.suh,
                base.svh,
                base.codebook,
                base.chunk_size,
                transpose=True,
                eager=True,
            )
            mx.eval(dx)
            store.stats.backward_calls += 1
            return materialize(dx)

        return streamed(x)
