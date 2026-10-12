import math

import mlx.core as mx
import numpy as np
import pytest
from exl3_mlx.attention import latent_attention, quantize_activation, rope, round_e4m3


def test_e4m3_rounding_matches_exhaustive_codebook():
    levels = np.array(
        [m * 2**-9 for m in range(8)]
        + [
            (1 + m / 8) * 2 ** (e - 7)
            for e in range(1, 16)
            for m in range(8)
            if not (e == 15 and m == 7)
        ],
        np.float32,
    )
    mids = (levels[:-1] + levels[1:]) / 2
    xs = np.concatenate([levels, mids, np.random.default_rng(1).uniform(0, 500, 1000)])
    want = []
    for x in xs:
        distances = np.abs(levels.astype(np.float64) - x)
        candidates = np.flatnonzero(distances == distances.min())
        want.append(levels[next((i for i in candidates if i % 2 == 0), candidates[0])])
    np.testing.assert_array_equal(np.array(round_e4m3(mx.array(xs, dtype=mx.float32))), want)


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
def test_quantization_value_and_explicit_straight_through_gradient(kind):
    x = mx.array(np.random.default_rng(12).normal(size=(3, 128)).astype(np.float32))
    y = quantize_activation(x, kind)
    assert mx.all(mx.isfinite(y)).item()
    dy = mx.random.normal(x.shape)
    (_,), (dx,) = mx.vjp(lambda t: quantize_activation(t, kind), [x], [dy])
    np.testing.assert_array_equal(np.array(dx), np.array(dy))
    if kind == "fp4":
        vals = np.array(
            [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -6], np.float32
        )
        want = [0, 0, 0.5, 1, 1, 1, 1.5, 2, 2, 2, 3, 4, 4, 4, 6, -6]
        np.testing.assert_array_equal(np.array(quantize_activation(mx.array(vals), "fp4")), want)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_latent_attention_sink_masks_and_gradients_against_numpy(ratio):
    rng = np.random.default_rng(4)
    q = rng.normal(size=(1, 5, 2, 8)).astype(np.float32)
    win = rng.normal(size=(1, 5, 8)).astype(np.float32)
    comp = rng.normal(size=(1, 5 // max(ratio, 1), 8)).astype(np.float32)
    sink = np.array([0.2, 1.1], np.float32)

    def oracle(q, win, comp):
        out = np.zeros_like(q, dtype=np.float64)
        for s in range(5):
            keys = win[0, max(0, s - 2) : s + 1].astype(np.float64)
            if ratio:
                keys = np.concatenate([keys, comp[0, : (s + 1) // ratio]], axis=0)
            for h in range(2):
                scores = np.append(keys @ q[0, s, h].astype(np.float64) / math.sqrt(8), sink[h])
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                out[0, s, h] = probs[:-1] @ keys
        return out

    def fn(a, b, c):
        return latent_attention(a, b, c, mx.array(sink), ratio=ratio, window_size=3)

    got = fn(mx.array(q), mx.array(win), mx.array(comp))
    np.testing.assert_allclose(np.array(got), oracle(q, win, comp), atol=3e-7)
    dq = mx.grad(lambda a: fn(a, mx.array(win), mx.array(comp)).sum())(mx.array(q))
    plus, minus = q.copy(), q.copy()
    plus[0, 3, 1, 4] += 1e-3
    minus[0, 3, 1, 4] -= 1e-3
    want = (oracle(plus, win, comp).sum() - oracle(minus, win, comp).sum()) / 2e-3
    assert dq[0, 3, 1, 4].item() == pytest.approx(want, abs=1e-5)


def test_rope_matches_adjacent_complex_reference_and_inverse():
    c = {
        "qk_rope_head_dim": 8,
        "rope_theta": 10000,
        "compress_rope_theta": 160000,
        "rope_scaling": {
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }
    x = np.random.default_rng(2).normal(size=(2, 3, 4, 16)).astype(np.float32)
    pos = np.array([0, 7, 600], np.float32)
    z = x[..., 8::2] + 1j * x[..., 9::2]
    z = z * np.exp(1j * pos[:, None] / (10000 ** (np.arange(0, 8, 2) / 8)))[None, :, None, :]
    want = x.copy()
    want[..., 8::2] = z.real
    want[..., 9::2] = z.imag
    got = rope(mx.array(x), mx.array(pos), c)
    np.testing.assert_allclose(np.array(got), want, atol=3e-6)
    for compressed in [False, True]:
        y = rope(mx.array(x), mx.array(pos), c, compressed=compressed)
        back = rope(y, mx.array(pos), c, compressed=compressed, inverse=True)
        np.testing.assert_allclose(np.array(back), x, atol=1e-6)
