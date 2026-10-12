import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from exl3_mlx import EXL3QLoRALinear
from exl3_mlx.checkpoint import MARKERS, Checkpoint
from exl3_mlx.moe import DeepSeekExpert, DeepSeekGate, DeepSeekMoE
from exl3_mlx.streaming import MatrixStore
from safetensors.numpy import save_file


def make_checkpoint(path, count=6):
    rng = np.random.default_rng(32)
    arrays = {}
    for i in range(count):
        for j, rate in enumerate((2, 3, 4), 1):
            key = f"e{i}.w{j}"
            arrays[key + ".trellis"] = rng.integers(
                0, 65536, (8, 8, rate * 16), dtype=np.uint16
            ).view(np.int16)
            arrays[key + ".suh"] = rng.uniform(-0.2, 0.2, 128).astype(np.float16)
            arrays[key + ".svh"] = np.ones(128, dtype=np.float16)
            arrays[key + ".mul1"] = np.array(MARKERS["mul1"], dtype=np.uint32).view(np.int32)
    save_file(arrays, path)
    return Checkpoint(path)


def make_expert(source, i):
    fn = source.linear if isinstance(source, MatrixStore) else source.load_linear
    return DeepSeekExpert(*(fn(f"e{i}.w{j}") for j in (1, 2, 3)), swiglu_limit=0.8)


@pytest.mark.parametrize("cache_bytes", [0, 14000, 1 << 20])
def test_streamed_reload_and_adapter_gradients(tmp_path, cache_bytes):
    checkpoint = make_checkpoint(tmp_path / "base.safetensors", count=1)
    store = MatrixStore(checkpoint, cache_bytes=cache_bytes, chunk_size=128)
    resident, streamed = make_expert(checkpoint, 0), make_expert(store, 0)
    x = mx.random.normal((2, 3, 128))
    cotangent = mx.random.normal(x.shape)
    (yr,), (gr,) = mx.vjp(resident, [x], [cotangent])
    (ys,), (gs,) = mx.vjp(streamed, [x], [cotangent])
    np.testing.assert_allclose(np.array(ys), np.array(yr), atol=2e-6)
    np.testing.assert_allclose(np.array(gs), np.array(gr), atol=3e-6)
    assert store.stats.backward_calls == 3
    assert store.stats.peak_cached_bytes <= cache_bytes
    if cache_bytes == 0:
        assert store.stats.loads == 6 and store.stats.cached_bytes == 0
    elif cache_bytes == 14000:
        assert store.stats.evictions > 0
    streamed.w1 = EXL3QLoRALinear(streamed.w1, rank=4, alpha=4)
    _, grads = nn.value_and_grad(streamed, lambda m: mx.mean(m(x) ** 2))(streamed)
    assert mx.linalg.norm(grads["w1"]["lora_b"]).item() > 0
    assert store.linear("e0.w1").parameters() == {}


def test_source_replacement_is_rejected_even_on_cache_hit(tmp_path):
    path = tmp_path / "base.safetensors"
    checkpoint = make_checkpoint(path, count=1)
    store = MatrixStore(checkpoint)
    store.get("e0.w1")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    with pytest.raises(RuntimeError, match="changed"):
        store.get("e0.w1")


def numpy_route(x, w, bias, k, scale):
    scores = np.sqrt(np.logaddexp(x.astype(np.float64) @ w.astype(np.float64).T, 0))
    ids = np.argsort(-(scores + bias), axis=-1, kind="stable")[:, :k]
    weights = np.take_along_axis(scores, ids, axis=-1)
    if k > 1:
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
    return weights * scale, ids


@pytest.mark.parametrize("k", [1, 2, 4])
def test_gate_matches_unbiased_numpy_reference_and_dx(k):
    rng = np.random.default_rng(6)
    w = rng.normal(0, 0.1, (4, 128)).astype(np.float32)
    bias = np.array([1.0, -0.5, 0.8, 0.1], np.float32)
    x = rng.normal(size=(3, 128)).astype(np.float32)
    router = DeepSeekGate(mx.array(w), mx.array(bias), top_k=k)
    weights, ids = router(mx.array(x))
    expected, expected_ids = numpy_route(x, w, bias, k, 1.5)
    np.testing.assert_array_equal(np.array(ids), expected_ids)
    np.testing.assert_allclose(np.array(weights), expected, atol=3e-7)
    dx = mx.grad(lambda z: mx.sum(router(z)[0] * mx.arange(1, k + 1)))(mx.array(x))
    eps = 1e-3
    for row, col in [(0, 3), (1, 60), (2, 9)]:
        plus, minus = x.copy(), x.copy()
        plus[row, col] += eps
        minus[row, col] -= eps
        hi = (numpy_route(plus, w, bias, k, 1.5)[0] * np.arange(1, k + 1)).sum()
        lo = (numpy_route(minus, w, bias, k, 1.5)[0] * np.arange(1, k + 1)).sum()
        assert float(dx[row, col]) == pytest.approx((hi - lo) / (2 * eps), abs=5e-5)


def test_grouped_moe_matches_dense_dispatch_forward_and_vjp(tmp_path):
    cp = make_checkpoint(tmp_path / "base.safetensors")
    mx.random.seed(17)
    gate = DeepSeekGate(
        mx.random.normal((5, 128)) * 0.03, mx.array([0.6, 0.3, -10.0, 0.1, 0.2]), top_k=2
    )
    store = MatrixStore(cp, cache_bytes=0)
    model = DeepSeekMoE(gate, [make_expert(store, i) for i in range(5)], make_expert(store, 5))
    residents = [make_expert(cp, i) for i in range(6)]

    def reference(x):
        flat = x.reshape(-1, 128)
        weights, ids = gate(flat)
        out = residents[-1](flat)
        for i in range(5):
            coeff = mx.sum(mx.where(ids == i, weights, 0), axis=-1, keepdims=True)
            out = out + residents[i](flat, coeff)
        return out.reshape(x.shape)

    x = mx.random.normal((2, 3, 128))
    dy = mx.random.normal(x.shape)
    (actual,), (da,) = mx.vjp(model, [x], [dy])
    (expected,), (de,) = mx.vjp(reference, [x], [dy])
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=2e-6)
    np.testing.assert_allclose(np.array(da), np.array(de), atol=3e-6)
    assert 2 not in np.array(gate(x.reshape(-1, 128))[1])  # unused expert wasn't evaluated
    used = len(np.unique(np.array(gate(x.reshape(-1, 128))[1]))) + 1
    assert store.stats.forward_calls == used * 3
    assert store.stats.backward_calls == used * 3
    assert store.stats.cached_bytes == 0


def test_gate_negative_values_not_lower_clamped():
    class Projection(nn.Module):
        def __init__(self, factor):
            super().__init__()
            self.factor = factor
            self.in_features = self.out_features = 1

        def __call__(self, x):
            return x * self.factor

    expert = DeepSeekExpert(Projection(-20), Projection(1), Projection(1), swiglu_limit=10)
    actual = float(expert(mx.ones((1, 1))).item())
    assert actual == pytest.approx(-20 / (1 + np.exp(20)), rel=1e-5)


def test_gradient_tape_does_not_retain_decoded_matrices(tmp_path):
    """A one-matrix custom VJP must not accumulate a dense matrix per call."""
    rng = np.random.default_rng(21)
    key = "large"
    save_file(
        {
            key + ".trellis": rng.integers(0, 65536, (64, 64, 48), dtype=np.uint16).view(np.int16),
            key + ".suh": np.full(1024, 0.02, np.float16),
            key + ".svh": np.ones(1024, np.float16),
            key + ".mul1": np.array(MARKERS["mul1"], dtype=np.uint32).view(np.int32),
        },
        tmp_path / "large.safetensors",
    )
    store = MatrixStore(Checkpoint(tmp_path / "large.safetensors"), cache_bytes=0, chunk_size=128)
    layer = store.linear(key)
    old = mx.set_cache_limit(0)
    active = []

    def loss(x):
        for _ in range(24):
            x = x + layer(x) * 0.01
            mx.eval(x)
            active.append(mx.get_active_memory())
        return mx.sum(x * x)

    try:
        dx = mx.grad(loss)(mx.ones((1, 1024)))
        mx.eval(dx)
        assert mx.all(mx.isfinite(dx)).item()
        # A retained float32 1024x1024 per layer would add ~92 MiB here.
        assert active[-1] - active[0] < 8 << 20
        assert store.stats.backward_calls == 24
    finally:
        mx.set_cache_limit(old)
