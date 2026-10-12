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
from .trellis import reconstruct


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
    decoded_loads: int = 0
    decoded_hits: int = 0
    decoded_evictions: int = 0
    decoded_loaded_bytes: int = 0
    decoded_cached_bytes: int = 0
    decoded_peak_cached_bytes: int = 0


class MatrixStore:
    """Independent LRUs for frozen packed matrices and optional decoded tiles.

    cache_bytes bounds retained packed tensors, not OS page cache, MLX allocator
    cache, activations, adapter gradients, or a matrix currently in use. Setting
    zero forces a disk read per call. Source shard stat fingerprints are checked
    so backward cannot silently reread different weights after a file replacement.

    decoded_cache_bytes bounds retained, evaluated FP16 tiles, not the working
    FP32 matmul tile. EXL3 codebooks already round to FP16 before returning FP32,
    so this storage is lossless. A decoded hit survives packed-cache eviction.
    Both caches are owned by this store and checked against its frozen sources.
    """

    def __init__(
        self,
        checkpoint,
        *,
        cache_bytes=64 << 20,
        decoded_cache_bytes=0,
        chunk_size=512,
        max_matrix_bytes=256 << 20,
    ):
        if cache_bytes < 0 or max_matrix_bytes <= 0:
            raise ValueError("Cache budget must be nonnegative and matrix budget positive")
        if chunk_size < 128 or chunk_size % 128:
            raise ValueError("chunk_size must be a multiple of 128")
        if type(decoded_cache_bytes) is not int or decoded_cache_bytes < 0:
            raise ValueError("Decoded cache budget must be a nonnegative integer")
        self.checkpoint = checkpoint
        self.cache_bytes, self.chunk_size = cache_bytes, chunk_size
        self.max_matrix_bytes = max_matrix_bytes
        self.decoded_cache_bytes = decoded_cache_bytes
        self.stats = StreamStats()
        self._cache = OrderedDict()
        self._decoded_cache = OrderedDict()
        self._decoded_identity = object()
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
        # Private scalar metadata binds decoder callbacks to this store/key
        # without retaining the store or adding trainable array parameters.
        layer._matrix_store_identity = self._decoded_identity
        layer._matrix_store_key = key
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

    def get_decoded(self, key, base, start, width):
        """Return an FP32 working tile, optionally cached losslessly in FP16.

        ``base`` must be the frozen layer returned by get(key). Every request,
        including a decoded cache hit, rechecks the checkpoint fingerprints.
        Decoded byte counters describe FP16 payload sizes; casts are transient.
        """
        if (
            getattr(base, "_matrix_store_identity", None) is not self._decoded_identity
            or getattr(base, "_matrix_store_key", None) != key
        ):
            raise ValueError("Decoded tile base does not belong to this store and matrix key")
        self._check(key)
        if (
            type(start) is not int
            or type(width) is not int
            or start < 0
            or width <= 0
            or start % 16
            or width % 16
            or start + width > base.out_features
        ):
            raise ValueError("Decoded tile must be positive, in bounds, and aligned to 16 columns")
        cache_key = key, base.codebook, start, width
        if cache_key in self._decoded_cache:
            self.stats.decoded_hits += 1
            self._decoded_cache.move_to_end(cache_key)
            return self._decoded_cache[cache_key].astype(mx.float32)
        size = base.in_features * width * 2
        if size <= self.decoded_cache_bytes:
            # Evict BEFORE reconstructing a tile that will be retained. An
            # oversized working tile bypasses the cache and preserves useful
            # entries rather than emptying an LRU it cannot populate.
            while (
                self._decoded_cache
                and self.stats.decoded_cached_bytes + size > self.decoded_cache_bytes
            ):
                _, old = self._decoded_cache.popitem(last=False)
                self.stats.decoded_cached_bytes -= old.nbytes
                self.stats.decoded_evictions += 1
                del old
            # Cut the decode graph so cached values do not retain packed matrices
            # or an enclosing gradient trace. Preserve the exact FP16 bytes.
            tile = materialize(
                reconstruct(base.trellis, base.codebook, start=start, width=width, dtype=mx.float16)
            )
            self._check(key)
            self._decoded_cache[cache_key] = tile
            self.stats.decoded_cached_bytes += tile.nbytes
            self.stats.decoded_peak_cached_bytes = max(
                self.stats.decoded_peak_cached_bytes, self.stats.decoded_cached_bytes
            )
            result = tile.astype(mx.float32)
        else:
            # Disabled/too-small caches preserve the original decoder path.
            result = reconstruct(base.trellis, base.codebook, start=start, width=width)
        self.stats.decoded_loads += 1
        self.stats.decoded_loaded_bytes += size
        return result

    def clear(self):
        self._cache.clear()
        self._decoded_cache.clear()
        self.stats.cached_bytes = 0
        self.stats.decoded_cached_bytes = 0

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
                inp,
                base.trellis,
                base.suh,
                base.svh,
                base.codebook,
                base.chunk_size,
                eager=True,
                decoder=lambda start, width: store.get_decoded(key, base, start, width),
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
                decoder=lambda start, width: store.get_decoded(key, base, start, width),
            )
            mx.eval(dx)
            store.stats.backward_calls += 1
            return materialize(dx)

        return streamed(x)
