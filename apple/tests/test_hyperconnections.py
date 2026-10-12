import mlx.core as mx
import numpy as np
import pytest
from exl3_mlx.hyperconnections import HyperConnection, expand_streams, hc_post, hc_pre, sinkhorn


def reference(streams, weight, scale, bias, h, iters, eps):
    x = streams.astype(np.float64).reshape(*streams.shape[:-2], -1)
    mix = (x @ weight.T) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-20)
    pre = 1 / (1 + np.exp(-(mix[..., :h] * scale[0] + bias[:h]))) + eps
    post = 2 / (1 + np.exp(-(mix[..., h : 2 * h] * scale[1] + bias[h : 2 * h])))
    cb = (mix[..., 2 * h :] * scale[2] + bias[2 * h :]).reshape(*x.shape[:-1], h, h)
    cb = np.exp(cb - cb.max(axis=-1, keepdims=True))
    cb = cb / cb.sum(axis=-1, keepdims=True) + eps
    cb = cb / (cb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        cb = cb / (cb.sum(axis=-1, keepdims=True) + eps)
        cb = cb / (cb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, cb


def test_hyperconnection_values_orientation_and_finite_difference():
    rng = np.random.default_rng(3)
    h, d = 4, 16
    x = rng.normal(size=(2, 3, h, d)).astype(np.float32)
    w = rng.normal(0, 0.1, ((2 + h) * h, h * d)).astype(np.float32)
    scale = np.array([0.4, 0.3, 0.2], np.float32)
    bias = rng.normal(0, 0.3, ((2 + h) * h,)).astype(np.float32)
    hc = HyperConnection(mx.array(w), mx.array(scale), mx.array(bias), hc_mult=h)
    expected = reference(x, w, scale, bias, h, 20, 1e-6)
    actual = hc(mx.array(x))
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.array(got), want, atol=3e-7)

    def loss(inp):
        pre, post, comb = hc(inp)
        return mx.sum(hc_post(mx.tanh(hc_pre(inp, pre)), inp, post, comb) ** 2)

    grad = np.array(mx.grad(loss)(mx.array(x)))

    def reference_loss(inp):
        pre, post, comb = reference(inp, w, scale, bias, h, 20, 1e-6)
        y = np.tanh(np.sum(inp * pre[..., None], axis=-2))
        out = y[..., None, :] * post[..., None] + np.einsum("...jk,...jd->...kd", comb, inp)
        return np.sum(out * out)

    for idx in [(0, 1, 2, 3), (1, 2, 0, 8)]:
        a, b = x.copy(), x.copy()
        a[idx] += 0.001
        b[idx] -= 0.001
        assert grad[idx] == pytest.approx(
            (reference_loss(a) - reference_loss(b)) / 0.002, abs=0.002
        )


def test_initial_mix_and_sinkhorn_stochastic_axes():
    x = mx.random.normal((2, 3, 16))
    h, pre = expand_streams(x)
    np.testing.assert_array_equal(np.array(hc_pre(h, pre)), np.array(x))
    comb = sinkhorn(mx.random.normal((3, 4, 4)))
    np.testing.assert_allclose(np.array(mx.sum(comb, axis=-2)), 1, atol=3e-6)
    np.testing.assert_allclose(np.array(mx.sum(comb, axis=-1)), 1, atol=3e-6)
