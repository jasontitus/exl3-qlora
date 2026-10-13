"""Exact frozen-prefix RAM LRU with an optional verified disk tier; no learned state.

Only deterministic blocks strictly before every adapter may be cached. The
owner must never mutate their weights, architecture, or behavior. Entries retain
materialized h, the next pre-mix, and all four shared attention-state leaves.
Consumers must treat those arrays as immutable; the model's blocks are functional.
The bound counts tensor and complete token-key payload bytes, not Python objects,
allocator caches, source activations, or entries still referenced by a caller.

This module stays CPU-importable. Its default materializer imports MLX lazily;
tests of admission/identity/eviction can supply a NumPy materializer instead.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class FrozenStorageIdentity:
    address: int
    shape: tuple
    strides: tuple
    dtype: str
    # Keep the original zero-copy view alive. Otherwise replacement could free
    # its allocation and allocator reuse could give different weights the same
    # address. The owner is deliberately excluded from identity comparisons.
    _owner: np.ndarray = field(compare=False, repr=False)


def frozen_structure_signature(value):
    """Detect replacement/configuration changes without hashing huge weight arrays.

    Frozen MLX arrays must never be modified in place. Evaluated storage identity
    detects replacements, not mutation of an existing array's storage. MLX creates
    temporary Python wrappers for frozen arrays inside value_and_grad, so Python
    object identity is not stable. NumPy's zero-copy buffer view gives the same
    storage address through those wrappers, without copying or hashing weights.
    Frozen weights are evaluated before the model records its initial signature.
    Private implementation caches are excluded; callers check frozen files too.
    """
    if all(hasattr(value, name) for name in ("dtype", "shape", "nbytes")):
        dtype = str(value.dtype)
        if dtype == "mlx.core.bfloat16":
            # NumPy cannot expose MLX BF16 directly. This view preserves storage.
            import mlx.core as mx

            buffer = np.asarray(value.view(mx.uint16))
        else:
            buffer = np.asarray(value)
        return FrozenStorageIdentity(
            buffer.__array_interface__["data"][0],
            tuple(value.shape),
            buffer.strides,
            dtype,
            buffer,
        )
    if isinstance(value, dict):
        return (
            type(value),
            id(value),
            tuple(
                (key, frozen_structure_signature(item))
                for key, item in value.items()
                if not isinstance(key, str) or not key.startswith("_")
            ),
        )
    if isinstance(value, (tuple, list)):
        return type(value), tuple(frozen_structure_signature(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return type(value), value
    return type(value), id(value)


@dataclass(frozen=True)
class PrefixKey:
    shape: tuple[int, int]
    training: bool
    checkpoint_blocks: bool
    tokens: bytes


@dataclass
class PrefixStats:
    hits: int = 0
    misses: int = 0
    insertions: int = 0
    evictions: int = 0
    bypasses: int = 0
    cached_bytes: int = 0
    peak_cached_bytes: int = 0


def materialize_frozen(value):
    import mlx.core as mx

    from .materialize import materialize

    return materialize(mx.stop_gradient(value))


class FrozenPrefixCache:
    """One cache per constructed model; suffix adapter updates need no invalidation.

    Training and evaluation have separate keys even though the supported frozen
    prefix currently has no dropout or other mode-dependent arithmetic. Toggling
    block recomputation is separate too. Equal complete token IDs and batch shape
    are required; this is not an incremental decoding or common-prefix cache.
    """

    def __init__(
        self, limit_bytes=0, *, frozen_blocks=0, materializer=materialize_frozen, disk=None
    ):
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError("Prefix cache byte limit must be a nonnegative integer")
        if type(frozen_blocks) is not int or frozen_blocks < 0:
            raise ValueError("Frozen prefix block count must be a nonnegative integer")
        self.limit_bytes, self.frozen_blocks = limit_bytes, frozen_blocks
        self.stats = PrefixStats()
        self._entries = OrderedDict()
        self._materializer = materializer
        self.disk = disk

    @property
    def enabled(self):
        return (self.limit_bytes > 0 or self.disk is not None) and self.frozen_blocks > 0

    @staticmethod
    def key(tokens, *, training, checkpoint_blocks):
        ids = np.asarray(tokens)
        if (
            ids.ndim != 2
            or min(ids.shape) < 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("Prefix cache needs nonnegative integer token IDs [batch, sequence]")
        if type(training) is not bool or type(checkpoint_blocks) is not bool:
            raise ValueError("Prefix cache modes must be boolean")
        # Retain the exact sequence, not just a hash. A canonical width permits
        # equal int32/int64 IDs to hit without allowing shape/batch collisions.
        return PrefixKey(tuple(ids.shape), training, checkpoint_blocks, ids.astype("<i8").tobytes())

    def get(self, key):
        if not isinstance(key, PrefixKey):
            raise TypeError("Expected a complete PrefixKey")
        if not self.enabled:
            self.stats.bypasses += 1
            return None
        if key not in self._entries:
            if self.disk is not None:
                saved = self.disk.get(key)
                if saved is not None:
                    self.stats.hits += 1
                    # The entry was fully evaluated and validated on disk load.
                    # Retain only if the RAM budget allows; disk-only is useful.
                    self._retain(key, saved)
                    return saved
            self.stats.misses += 1
            return None
        self._entries.move_to_end(key)
        self.stats.hits += 1
        return self._entries[key][0]

    def put(self, key, state):
        """Materialize an admitted state after eviction; oversized entries bypass.

        Returned state is the same private immutable tuple subsequently returned
        by get. An oversized entry returns None without discarding useful entries
        or making an unnecessary host copy. A disabled cache does likewise.
        """
        if not isinstance(key, PrefixKey):
            raise TypeError("Expected a complete PrefixKey")
        if not self.enabled:
            self.stats.bypasses += 1
            return None
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError("Prefix state must be (h, pre, four-leaf KV tuple)")
        h, pre, kv = state
        if not isinstance(kv, tuple) or len(kv) != 4:
            raise ValueError("Prefix attention state needs all four KV leaves")
        leaves = (h, pre, *kv)
        if h.shape[:2] != key.shape or pre.shape[:2] != key.shape:
            raise ValueError("Prefix state and complete token shape disagree")
        size = len(key.tokens) + sum(value.nbytes for value in leaves)
        if size > self.limit_bytes and self.disk is None:
            self.stats.bypasses += 1
            return None
        # Evict before creating a fresh host/materialized copy. Disk-only or
        # oversized-for-RAM entries remain bounded to the current caller.
        self._evict(key, size)
        frozen = tuple(self._materializer(value) for value in leaves)
        if sum(value.nbytes for value in frozen) + len(key.tokens) != size:
            raise ValueError("Prefix materializer changed tensor storage size")
        saved = frozen[0], frozen[1], frozen[2:]
        if self.disk is not None:
            self.disk.put(key, saved)
        self._retain(key, saved)
        return saved

    def _evict(self, key, size):
        if size > self.limit_bytes:
            return
        if key in self._entries:
            _, old_size = self._entries.pop(key)
            self.stats.cached_bytes -= old_size
        while self._entries and self.stats.cached_bytes + size > self.limit_bytes:
            _, (_, old_size) = self._entries.popitem(last=False)
            self.stats.cached_bytes -= old_size
            self.stats.evictions += 1

    def _retain(self, key, saved):
        size = len(key.tokens) + sum(v.nbytes for v in (saved[0], saved[1], *saved[2]))
        if size > self.limit_bytes:
            return
        self._evict(key, size)
        self._entries[key] = saved, size
        self.stats.insertions += 1
        self.stats.cached_bytes += size
        self.stats.peak_cached_bytes = max(self.stats.peak_cached_bytes, self.stats.cached_bytes)

    def clear(self, *, reset_stats=False):
        self._entries.clear()
        self.stats.cached_bytes = 0
        if reset_stats:
            self.stats = PrefixStats()

    def report(self):
        return {
            **asdict(self.stats),
            "limit_bytes": self.limit_bytes,
            "frozen_blocks": self.frozen_blocks,
            "entries": len(self._entries),
            "enabled": self.enabled,
            "scope": "exact full-token frozen prefix; separate train/eval keys",
            "disk": self.disk.report() if self.disk is not None else None,
        }
