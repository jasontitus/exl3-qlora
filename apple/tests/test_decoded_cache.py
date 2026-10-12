"""Lossless FP16 tile storage, independent eviction, and source invalidation."""

import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

import exl3_mlx.streaming as streaming
from exl3_mlx.checkpoint import MARKERS, Checkpoint
from exl3_mlx.linear import _base
from exl3_mlx.native import LoRALinear
from exl3_mlx.reference import random_packed
from exl3_mlx.streaming import MatrixStore
from exl3_mlx.trellis import CODEBOOKS, reconstruct


def checkpoint(path):
    rng, tensors = np.random.default_rng(119), {}
    for key, k, n in (("wide", 128, 640), ("tall", 640, 128)):
        tensors[key + ".trellis"] = random_packed(k // 16 * (n // 16), 3, rng).reshape(
            k // 16, n // 16, -1
        )
        tensors[key + ".suh"] = rng.uniform(-0.2, 0.2, k).astype(np.float16)
        tensors[key + ".svh"] = rng.uniform(0.3, 0.8, n).astype(np.float16)
        tensors[key + ".mul1"] = np.array(MARKERS["mul1"], dtype=np.uint32).view(np.int32)
    save_file(tensors, path)
    return Checkpoint(path)


def bits(value):
    # Preserve native item size (including the uint64 scalar optimizer step).
    # NumPy lacks BF16 support, so reinterpret only that same-size dtype first.
    raw = np.array(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)
    return raw.reshape(-1).view(np.uint8)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("budget", [0, 1, 128 * 512 * 2, 128 * 640 * 2])
def test_forward_and_vjp_are_bitwise_equal_cold_warm_and_evicted(tmp_path, dtype, budget):
    cp = checkpoint(tmp_path / "base.safetensors")
    uncached = MatrixStore(cp, cache_bytes=0, chunk_size=512)
    cached = MatrixStore(cp, cache_bytes=0, decoded_cache_bytes=budget, chunk_size=512)
    rng = np.random.default_rng(120)
    for key, k, n in (("wide", 128, 640), ("tall", 640, 128)):
        x = mx.array(rng.normal(size=(3, k)).astype(np.float32)).astype(dtype)
        dy = mx.array(rng.normal(size=(3, n)).astype(np.float32)).astype(dtype)
        (expected,), (expected_dx,) = mx.vjp(uncached.linear(key), [x], [dy])
        # Check the new callback's disabled path against the original direct
        # decoder path too, not only cache-on against cache-off callbacks.
        resident = cp.load_linear(key, chunk_size=512)
        direct = (resident.trellis, resident.suh, resident.svh, resident.codebook, 512)
        np.testing.assert_array_equal(bits(expected), bits(_base(x, *direct, eager=True)))
        np.testing.assert_array_equal(
            bits(expected_dx), bits(_base(dy, *direct, transpose=True, eager=True))
        )
        for _ in range(2):
            (got,), (dx,) = mx.vjp(cached.linear(key), [x], [dy])
            np.testing.assert_array_equal(bits(got), bits(expected))
            np.testing.assert_array_equal(bits(dx), bits(expected_dx))
            assert cached.stats.decoded_cached_bytes <= budget
            assert cached.stats.decoded_peak_cached_bytes <= budget
            assert cached.stats.cached_bytes == 0
            assert cached.linear(key).parameters() == {}
            assert all(v.dtype == mx.float16 for v in cached._decoded_cache.values())
            assert cached.stats.decoded_cached_bytes == sum(
                v.nbytes for v in cached._decoded_cache.values()
            )
    if budget >= 128 * 640 * 2:
        assert cached.stats.decoded_hits > 0
    if budget == 128 * 512 * 2:
        assert cached.stats.decoded_evictions > 0
    cached.clear()
    assert cached.stats.cached_bytes == cached.stats.decoded_cached_bytes == 0
    assert not cached._cache and not cached._decoded_cache


def test_decoded_lru_eviction_precedes_allocation_and_packed_eviction_is_independent(
    tmp_path, monkeypatch
):
    cp = checkpoint(tmp_path / "base.safetensors")
    tile_bytes = 128 * 128 * 2
    store = MatrixStore(cp, cache_bytes=40_000, decoded_cache_bytes=tile_bytes * 2)
    base = store.get("wide")
    native_decode = streaming.reconstruct
    allocations = []

    def measured(packed, codebook, **kwargs):
        assert store.stats.decoded_cached_bytes + tile_bytes <= store.decoded_cache_bytes
        allocations.append((kwargs["start"], store.stats.decoded_cached_bytes))
        return native_decode(packed, codebook, **kwargs)

    monkeypatch.setattr(streaming, "reconstruct", measured)
    for start in (0, 128, 0, 256):
        mx.eval(store.get_decoded("wide", base, start, 128))
    assert allocations == [(0, 0), (128, tile_bytes), (256, tile_bytes)]
    assert {key[2] for key in store._decoded_cache} == {0, 256}
    assert store.stats.decoded_loads == 3 and store.stats.decoded_hits == 1
    assert store.stats.decoded_evictions == 1
    # Loading a different packed matrix evicts wide's packed entry only.
    store.get("tall")
    assert "wide" not in store._cache
    hits = store.stats.decoded_hits
    mx.eval(store.get_decoded("wide", store.get("wide"), 0, 128))
    assert store.stats.decoded_hits == hits + 1 and len(allocations) == 3


def test_decoded_hit_rechecks_source_fingerprint(tmp_path):
    path = tmp_path / "base.safetensors"
    store = MatrixStore(checkpoint(path), decoded_cache_bytes=1 << 20)
    base = store.get("wide")
    mx.eval(store.get_decoded("wide", base, 0, 128))
    before = store.stats.decoded_hits
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
    with pytest.raises(RuntimeError, match="changed"):
        store.get_decoded("wide", base, 0, 128)
    assert store.stats.decoded_hits == before


def test_foreign_matrix_or_store_cannot_poison_decoded_key(tmp_path):
    cp = checkpoint(tmp_path / "base.safetensors")
    first = MatrixStore(cp, decoded_cache_bytes=1 << 20)
    other = MatrixStore(cp, decoded_cache_bytes=1 << 20)
    for foreign in (first.get("tall"), other.get("wide")):
        with pytest.raises(ValueError, match="does not belong"):
            first.get_decoded("wide", foreign, 0, 128)
    assert not first._decoded_cache
    assert first.stats.decoded_loads == first.stats.decoded_hits == 0


@pytest.mark.parametrize("budget", [128 * 512 * 2, 128 * 640 * 2], ids=["evicted", "warm"])
def test_lora_optimizer_updates_and_checkpoint_tensors_match_without_cache(tmp_path, budget):
    from exl3_mlx.training import save_checkpoint

    cp = checkpoint(tmp_path / "base.safetensors")
    stores = [MatrixStore(cp, cache_bytes=0, decoded_cache_bytes=b) for b in (0, budget)]
    root = tmp_path / "identity"
    (root / "engram").mkdir(parents=True)
    for name in (
        "config.json",
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        (root / name).write_text("{}")

    class SmallModel(nn.Module):
        def __init__(self, store):
            super().__init__()
            self.linear = LoRALinear(store.linear("wide"), rank=2, alpha=4)
            self._root = root
            self.adapter_config = {
                "rank": 2,
                "alpha": 4,
                "adapter_layers": [],
                "expert_adapter_layers": [0],
            }

        def __call__(self, x):
            return self.linear(x)

    mx.random.seed(124)
    models = [SmallModel(store) for store in stores]
    models[0].linear.lora_b = mx.random.normal((2, 640)) * 0.01
    for name in ("lora_a", "lora_b"):
        setattr(models[1].linear, name, mx.array(np.array(getattr(models[0].linear, name))))
    optimizers = [optim.AdamW(learning_rate=1e-3) for _ in models]
    x = mx.random.normal((3, 128)).astype(mx.bfloat16)
    target = mx.random.normal((3, 640))

    def loss(model, value):
        return mx.mean((model(value).astype(mx.float32) - target) ** 2)

    def assert_trees_equal(first, second):
        a, b = dict(tree_flatten(first)), dict(tree_flatten(second))
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
            np.testing.assert_array_equal(bits(a[key]), bits(b[key]))

    for _ in range(3):
        results = []
        for model, optimizer in zip(models, optimizers, strict=True):
            value, gradients = nn.value_and_grad(model, loss)(model, x)
            dx = mx.grad(lambda inp: loss(model, inp))(x)
            gradients, norm = optim.clip_grad_norm(gradients, 1.0)
            mx.eval(value, gradients, dx, norm)
            results.append((value, gradients, dx, norm))
            optimizer.update(model, gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
        assert_trees_equal(results[0], results[1])
        assert_trees_equal(models[0].trainable_parameters(), models[1].trainable_parameters())
        assert_trees_equal(optimizers[0].state, optimizers[1].state)
    expected_keys = {"linear.lora_a", "linear.lora_b"}
    assert stores[1].stats.decoded_cached_bytes > 0
    assert stores[1].stats.decoded_peak_cached_bytes <= budget
    assert (
        stores[1].stats.decoded_hits > 0
        if budget == 128 * 640 * 2
        else stores[1].stats.decoded_evictions > 0
    )
    for index, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
        assert set(dict(tree_flatten(model.parameters()))) == expected_keys
        assert set(dict(tree_flatten(model.trainable_parameters()))) == expected_keys
        save_checkpoint(model, optimizer, tmp_path / f"step-{index}", {"step": 3})
    for filename in ("adapter.safetensors", "optimizer.safetensors"):
        first = mx.load(str(tmp_path / "step-0" / filename))
        second = mx.load(str(tmp_path / "step-1" / filename))
        assert_trees_equal(first, second)
        expected = (
            expected_keys
            if filename.startswith("adapter")
            else set(dict(tree_flatten(optimizers[1].state)))
        )
        assert set(second) == expected


def test_oversized_tile_bypasses_without_evicting_useful_entries(tmp_path):
    tile_bytes = 128 * 128 * 2
    store = MatrixStore(checkpoint(tmp_path / "base.safetensors"), decoded_cache_bytes=tile_bytes)
    base = store.get("wide")
    first = store.get_decoded("wide", base, 0, 128)
    mx.eval(first)
    keys = list(store._decoded_cache)
    oversized = store.get_decoded("wide", base, 128, 512)
    mx.eval(oversized)
    assert list(store._decoded_cache) == keys
    assert store.stats.decoded_cached_bytes == tile_bytes
    assert store.stats.decoded_evictions == 0
    np.testing.assert_array_equal(bits(store.get_decoded("wide", base, 0, 128)), bits(first))
    assert store.stats.decoded_hits == 1 and store.stats.decoded_loads == 2


@pytest.mark.parametrize("budget", [-1, 0.5, True])
def test_invalid_decoded_cache_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="Decoded cache"):
        MatrixStore(checkpoint(tmp_path / "base.safetensors"), decoded_cache_bytes=budget)


@pytest.mark.parametrize("codebook", CODEBOOKS)
def test_fp16_storage_preserves_every_codebook_value_as_fp32(codebook):
    # Existing format fixtures cover all supported bitrates. Here every possible
    # codebook entry (including tiny values) is checked across output dtypes.
    from exl3_mlx.trellis import HEADER

    kernel = mx.fast.metal_kernel(
        name="decoded_cache_codebook",
        input_names=["indices"],
        output_names=["out"],
        header=HEADER,
        source="uint i=thread_position_in_grid.x; out[i]=exl3_codebook<CB>(indices[i]);",
    )
    results = []
    for dtype in (mx.float16, mx.float32):
        results.append(
            kernel(
                inputs=[mx.arange(65536, dtype=mx.uint32)],
                template=[("CB", CODEBOOKS[codebook])],
                grid=(65536, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(65536,)],
                output_dtypes=[dtype],
            )[0].astype(mx.float32)
        )
    np.testing.assert_array_equal(bits(results[0]), bits(results[1]))


@pytest.mark.parametrize("rate", [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8])
def test_partial_decode_fp16_to_fp32_is_exact_at_every_supported_rate(rate):
    packed = mx.array(random_packed(8 * 40, rate, np.random.default_rng(122)).reshape(8, 40, -1))
    direct = reconstruct(packed, start=512, width=128)
    stored = reconstruct(packed, start=512, width=128, dtype=mx.float16).astype(mx.float32)
    np.testing.assert_array_equal(bits(direct), bits(stored))
