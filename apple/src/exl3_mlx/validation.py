"""Explicit, slow numerical checks; NumPy is never used on the training path."""

import time

import mlx.core as mx
import numpy as np

from .reference import had128, rotated
from .trellis import reconstruct


def validate_linear(base, *, rows=3, seed=23):
    rng = np.random.default_rng(seed)
    k, n = base.in_features, base.out_features
    # CPU oracle uses unpackbits, fp64 codebook arithmetic and dense Hadamard matrices.
    w = rotated(np.array(base.trellis), base.codebook)
    actual = np.array(reconstruct(base.trellis, base.codebook, dtype=mx.float16))
    exact = np.array_equal(actual.view(np.uint16), w.view(np.uint16))
    if not exact:
        raise AssertionError("Metal decode differs from the independent NumPy oracle")
    h = had128()
    su, sv = np.array(base.suh).astype(np.float64), np.array(base.svh).astype(np.float64)
    x = rng.normal(size=(rows, k)).astype(np.float32)
    dy = rng.normal(size=(rows, n)).astype(np.float32)

    def had(a):
        return (a.reshape(-1, a.shape[-1] // 128, 128) @ h).reshape(a.shape)

    want_y = had(had(x * su) @ w.astype(np.float64)) * sv
    if "bias" in base:
        want_y += np.array(base.bias)
    want_dx = had(had(dy * sv) @ w.astype(np.float64).T) * su
    inp, cotan = mx.array(x), mx.array(dy)
    (y,), (dx,) = mx.vjp(base, [inp], [cotan])
    y, dx = np.array(y), np.array(dx)
    np.testing.assert_allclose(y, want_y, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(dx, want_dx, rtol=3e-4, atol=3e-4)
    # Synchronize each timing: MLX arrays are lazy; construction time is not GPU time.
    mx.eval(base(inp))
    start = time.perf_counter()
    for _ in range(5):
        mx.eval(base(inp))
    elapsed = (time.perf_counter() - start) / 5
    return {
        "in_features": k,
        "out_features": n,
        "bpw": base.bpw,
        "codebook": base.codebook,
        "chunk_size": base.chunk_size,
        "decode_bit_exact": bool(exact),
        "forward_max_abs_error": float(np.max(np.abs(y - want_y))),
        "dx_max_abs_error": float(np.max(np.abs(dx - want_dx))),
        "forward_ms_warm_mean": elapsed * 1000,
        "rows": rows,
        "mlx": mx.__version__,
        "device": mx.device_info()["device_name"],
        "validation_scope": "single stored matrix against CPU format oracle; not native model parity",
    }


def validate_expert(expert, bases, *, seed=41):
    """Complete three-projection expert, against float64 NumPy forward/adjoint."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, bases[0].in_features)).astype(np.float32)
    dy = rng.normal(size=x.shape).astype(np.float32)
    scales = np.array([[0.3], [0.7]], np.float32)
    h = had128()
    weights = [rotated(np.array(b.trellis), b.codebook).astype(np.float64) for b in bases]
    su = [np.array(b.suh).astype(np.float64) for b in bases]
    sv = [np.array(b.svh).astype(np.float64) for b in bases]

    def had(a):
        return (a.reshape(-1, a.shape[-1] // 128, 128) @ h).reshape(a.shape)

    def linear(a, i, backward=False):
        left, right = (sv[i], su[i]) if backward else (su[i], sv[i])
        weight = weights[i].T if backward else weights[i]
        return had(had(a * left) @ weight) * right

    gate, up = linear(x, 0), linear(x, 2)
    limit = expert.swiglu_limit
    g, u = np.minimum(gate, limit), np.clip(up, -limit, limit)
    sigmoid = 1 / (1 + np.exp(-g))
    hidden = (g * sigmoid) * u * scales
    want = linear(hidden, 1)
    dh = linear(dy, 1, True) * scales
    dg = dh * u * sigmoid * (1 + g * (1 - sigmoid)) * (gate < limit)
    du = dh * g * sigmoid * (np.abs(up) < limit)
    want_dx = linear(dg, 0, True) + linear(du, 2, True)
    (y,), (dx,) = mx.vjp(lambda z: expert(z, mx.array(scales)), [mx.array(x)], [mx.array(dy)])
    y, dx = np.array(y), np.array(dx)
    np.testing.assert_allclose(y, want, rtol=5e-4, atol=1e-4)
    np.testing.assert_allclose(dx, want_dx, rtol=5e-4, atol=1e-4)
    return {
        "forward_max_abs_error": float(np.abs(y - want).max()),
        "dx_max_abs_error": float(np.abs(dx - want_dx).max()),
        "bits_w1_w2_w3": [b.bpw for b in bases],
        "rows": 2,
        "scope": "Complete real expert with route weights before down projection; NumPy float64 oracle",
    }
