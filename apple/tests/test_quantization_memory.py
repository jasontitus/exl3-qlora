"""Quantization chunk-boundary parity and retained-forward-scratch regression."""

import gc

import mlx.core as mx
import numpy as np
import pytest

import exl3_mlx.attention as attention
from exl3_mlx.materialize import materialize


def vectorized_reference(x, kind):
    """Independent whole-array reference using MLX's native FP8 conversion."""
    block = 16 if kind == "fp4" else 32
    values = x.astype(mx.float32).reshape(*x.shape[:-1], -1, block)
    maximum = mx.max(mx.abs(values), axis=-1, keepdims=True)
    if kind == "fp4":
        raw_scale = mx.maximum(maximum, 6 * 2**-9) / 6
        scale = mx.from_fp8(mx.to_fp8(raw_scale), mx.float32)
    else:
        denominator = 448 if kind == "fp8" else 6
        floor = 1e-4 if kind == "fp8" else 6 * 2**-126
        exponent = mx.ceil(mx.log2(mx.maximum(maximum, floor) / denominator)).astype(mx.int32)
        scale = ((exponent + 127).astype(mx.uint32) << 23).view(mx.float32)
    normalized = values / scale
    if kind == "fp8":
        rounded = mx.from_fp8(mx.to_fp8(mx.clip(normalized, -448, 448)), mx.float32)
    else:
        magnitude = mx.abs(mx.clip(normalized, -6, 6))
        index = mx.zeros(magnitude.shape, dtype=mx.int32)
        for threshold in (0.25, 1.25, 2.5, 5):
            index += (magnitude > threshold).astype(mx.int32)
        for threshold in (0.75, 1.75, 3.5):
            index += (magnitude >= threshold).astype(mx.int32)
        rounded = mx.sign(normalized) * mx.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])[index]
    return (rounded * scale).reshape(x.shape).astype(x.dtype)


def raw_values(x):
    return np.array(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x)


@pytest.mark.parametrize("kind", ["fp8", "fp4", "fp4_e8m0"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_chunked_quantization_matches_vectorized_reference_across_feature_rows(
    monkeypatch, kind, dtype
):
    # Three FP8 groups per chunk crosses the 64-feature row boundaries. The
    # four-dimensional input also checks flattening of all leading dimensions.
    monkeypatch.setattr(attention, "_QUANTIZATION_CHUNK_ELEMENTS", 96)
    values = np.random.default_rng(48).normal(size=(2, 3, 5, 64)).astype(np.float32)
    values[:, :, 0] = 0
    values[0, 0, 1, :16] = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, -6]
    x = mx.array(values).astype(dtype)
    dy = mx.array(np.random.default_rng(4).normal(size=values.shape)).astype(dtype)
    (actual,), (gradient,) = mx.vjp(
        lambda value: attention.quantize_activation(value, kind), [x], [dy]
    )
    expected = vectorized_reference(x, kind)
    np.testing.assert_array_equal(raw_values(actual), raw_values(expected))
    np.testing.assert_array_equal(raw_values(gradient), raw_values(dy))


def test_large_quantization_chain_has_bounded_scratch(record_property):
    # Realistic attention-output element count without a loaded model: 16M
    # BF16 elements. Whole-array FP32 rounding intermediates are 64 MiB each.
    mx.random.seed(49)
    x = materialize(mx.random.normal((2048, 8192)).astype(mx.bfloat16))
    gc.collect()
    old_cache = mx.set_cache_limit(0)
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()

    def fn(value):
        for _ in range(3):
            value = attention.quantize_activation(value) + value * 0.01
        return value

    try:
        (output,), (gradient,) = mx.vjp(fn, [x], [mx.ones_like(x)])
        mx.eval(output, gradient)
        peak_extra = mx.get_peak_memory() - baseline
        assert mx.all(mx.isfinite(output)).item()
        assert mx.all(mx.isfinite(gradient)).item()
        assert mx.min(gradient).item() > 1
        record_property("peak_extra_bytes", peak_extra)
        print(f"Quantization chain extra peak: {peak_extra / 2**20:.2f} MiB")
        assert peak_extra < 512 << 20
    finally:
        mx.set_cache_limit(old_cache)
