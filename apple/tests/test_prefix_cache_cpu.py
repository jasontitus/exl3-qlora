"""CPU-only exact-key, byte-bound and invalidation-signature tests."""

import numpy as np
import pytest

from exl3_mlx.prefix_cache import FrozenPrefixCache, frozen_structure_signature


def key(ids=((1, 2, 3),), **modes):
    return FrozenPrefixCache.key(
        np.array(ids), **({"training": True, "checkpoint_blocks": True} | modes)
    )


def state(length=3):
    return (
        np.arange(length * 4 * 8, dtype=np.float32).reshape(1, length, 4, 8),
        np.ones((1, length, 4), np.float32),
        (
            np.zeros((1, 2, 8), np.float32),
            np.ones((1, 2, 4), np.float16),
            np.full((1, length, 2), -1, np.int32),
            np.full((1, length, 1), 3, np.uint8),
        ),
    )


def state_bytes(value, token_key):
    h, pre, kv = value
    return len(token_key.tokens) + sum(v.nbytes for v in (h, pre, *kv))


def test_full_token_shape_and_modes_are_part_of_exact_key():
    base = key()
    assert base == FrozenPrefixCache.key(
        np.array([[1, 2, 3]], np.int32), training=True, checkpoint_blocks=True
    )
    assert base != key(((1, 2, 4),))
    assert base != key(((1,), (2,), (3,)))
    assert base != key(training=False)
    assert base != key(checkpoint_blocks=False)
    with pytest.raises(ValueError):
        key(((True, False),))
    with pytest.raises(ValueError):
        key(((1.0, 2.0),))
    with pytest.raises(ValueError):
        key(((-1, 2),))
    with pytest.raises(ValueError):
        key(((2**63, 2),))


def test_all_attention_leaves_materialized_and_byte_counted_before_retention():
    source, token_key, copies = state(), key(), []
    size = state_bytes(source, token_key)

    def copy(value):
        copies.append(value.shape)
        return value.copy()

    cache = FrozenPrefixCache(size, frozen_blocks=2, materializer=copy)
    assert cache.get(token_key) is None
    saved = cache.put(token_key, source)
    assert len(copies) == 6
    source[0][:] = -123
    assert saved[0].flat[0] == 0
    assert cache.get(token_key) is saved
    for actual, expected in zip(saved[2], source[2], strict=True):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    report = cache.report()
    assert report["misses"] == report["hits"] == report["insertions"] == 1
    assert report["cached_bytes"] == report["peak_cached_bytes"] == size


def test_lru_evicts_before_materialization_and_oversized_bypass_preserves_entries():
    value = state()
    size = state_bytes(value, key())
    cache = FrozenPrefixCache(size * 2, frozen_blocks=2, materializer=lambda v: v.copy())
    first, second, third = key(), key(((4, 5, 6),)), key(((7, 8, 9),))
    cache.put(first, value)
    cache.put(second, value)
    cache.get(first)
    observed = []

    def copy(value):
        observed.append(cache.report()["cached_bytes"])
        return value.copy()

    cache._materializer = copy
    cache.put(third, value)
    assert set(observed) == {size}
    assert cache.get(second) is None
    assert cache.get(first) is not None and cache.get(third) is not None
    oversized = key((tuple(range(100)),))
    copies = len(observed)
    assert cache.put(oversized, state(100)) is None
    assert len(observed) == copies
    assert cache.report()["entries"] == 2
    assert cache.report()["evictions"] == 1
    assert cache.report()["bypasses"] == 1
    cache.clear()
    assert cache.report()["cached_bytes"] == cache.report()["entries"] == 0
    assert cache.report()["evictions"] == 1
    cache.clear(reset_stats=True)
    assert cache.report()["evictions"] == cache.report()["peak_cached_bytes"] == 0


@pytest.mark.parametrize("limit,boundary", [(0, 39), (1 << 20, 0)])
def test_disabled_and_all_layer_training_never_materialize(limit, boundary):
    def forbidden(value):
        raise AssertionError("Inactive cache must not materialize a tensor")

    cache = FrozenPrefixCache(limit, frozen_blocks=boundary, materializer=forbidden)
    assert not cache.enabled
    assert cache.get(key()) is None
    assert cache.put(key(), state()) is None
    assert cache.report()["entries"] == 0


def test_replacement_and_behavior_changes_alter_frozen_structure_signature():
    value = {"weight": np.ones((2, 3)), "gate_temp": 1.0, "_private_counter": 0}
    original = frozen_structure_signature(value)
    value["_private_counter"] += 1
    assert frozen_structure_signature(value) == original
    value["gate_temp"] = 2.0
    assert frozen_structure_signature(value) != original
    value["gate_temp"] = 1.0
    value["weight"] = value["weight"].copy()
    assert frozen_structure_signature(value) != original


def test_signature_retains_original_storage_owner_and_accepts_identical_alias():
    import gc
    import weakref

    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    owner = weakref.ref(weight)
    signature = frozen_structure_signature(weight)
    assert frozen_structure_signature(weight.view()) == signature
    del weight
    gc.collect()
    assert owner() is not None
    del signature
    gc.collect()
    assert owner() is None


@pytest.mark.parametrize("limit,boundary", [(-1, 2), (True, 2), (1, -1), (1, True)])
def test_invalid_limits_rejected(limit, boundary):
    with pytest.raises(ValueError):
        FrozenPrefixCache(limit, frozen_blocks=boundary)
