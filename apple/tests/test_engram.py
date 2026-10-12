"""Independent smooth Engram oracle; quantized projection STE is outside this test."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from exl3_mlx.engram import Engram


def engram_fixture():
    rng = np.random.default_rng(20261012)
    b, s, mult, dim, columns, embed_dim = 2, 3, 4, 8, 2, 4
    # Distinct stream magnitudes make joint-over-stream normalization detectably wrong.
    streams = rng.normal(size=(b, s, mult, dim)) * np.array([0.07, 0.7, 4.0, 19.0])[:, None]
    embedding = rng.normal(size=(b, s, columns, embed_dim))
    projection = rng.normal(0, 0.2, size=((mult + 1) * dim, columns * embed_dim))
    q_weight = rng.uniform(0.5, 1.5, size=(mult, dim))
    k_weight = rng.uniform(0.5, 1.5, size=(mult, dim))
    cotangent = rng.normal(size=streams.shape)
    return tuple(
        array.astype(np.float32)
        for array in (streams, embedding, projection, q_weight, k_weight, cotangent)
    )


def numpy_engram(streams, embedding, projection, q_weight, k_weight):
    """Scalar float64 oracle, with no MLX operations or runtime Engram helpers."""
    streams, embedding, projection, q_weight, k_weight = (
        np.asarray(array, dtype=np.float64)
        for array in (streams, embedding, projection, q_weight, k_weight)
    )
    b, s, mult, dim = streams.shape
    output = np.empty_like(streams)
    dots = np.empty((b, s, mult), dtype=np.float64)
    for batch in range(b):
        for token in range(s):
            # A separate key per stream precedes one value shared by all streams.
            projected = projection @ embedding[batch, token].ravel()
            value = projected[mult * dim :]
            for stream in range(mult):
                h = streams[batch, token, stream]
                key = projected[stream * dim : (stream + 1) * dim]
                h_rms = math.sqrt(sum(float(v) ** 2 for v in h) / dim + 1e-20)
                key_rms = math.sqrt(sum(float(v) ** 2 for v in key) / dim + 1e-20)
                weighted_dot = sum(
                    float(h[d] * q_weight[stream, d] * k_weight[stream, d] * key[d])
                    for d in range(dim)
                )
                dot = weighted_dot / (math.sqrt(dim) * h_rms * key_rms)
                dots[batch, token, stream] = dot
                signed_root = math.copysign(math.sqrt(max(abs(dot), 1e-6)), dot)
                gate = 1.0 / (1.0 + math.exp(-signed_root))
                output[batch, token, stream] = h + gate * value
    return output, dots


def test_engram_forward_and_stream_vjp_match_independent_float64_oracle():
    streams, embedding, projection, q_weight, k_weight, cotangent = engram_fixture()
    expected, dots = numpy_engram(streams, embedding, projection, q_weight, k_weight)
    # Both signed-root branches participate, safely away from the clamp and zero.
    assert np.min(np.abs(dots)) > 0.01
    assert np.any(dots > 0.1) and np.any(dots < -0.1)

    class FrozenProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array(projection)
            self.freeze()

        def __call__(self, value):
            return value @ self.weight.T

    layer = Engram(FrozenProjection(), mx.array(q_weight), mx.array(k_weight))
    embedding_mx = mx.array(embedding)
    (actual,), (dx,) = mx.vjp(
        lambda value: layer(value, embedding_mx),
        [mx.array(streams)],
        [mx.array(cotangent)],
    )
    np.testing.assert_allclose(np.array(actual), expected, rtol=3e-6, atol=3e-6)
    assert not tree_flatten(layer.trainable_parameters())

    # Differentiate a linear scalar objective at every stream coordinate. The
    # projection here is dense FP32: this is not a finite difference of FP8/FP4
    # quantization stairs and makes no claim about their intentional identity STE.
    delta = 2e-5
    source = streams.astype(np.float64)
    finite_difference = np.empty_like(source)
    for index in np.ndindex(source.shape):
        plus, minus = source.copy(), source.copy()
        plus[index] += delta
        minus[index] -= delta
        high, _ = numpy_engram(plus, embedding, projection, q_weight, k_weight)
        low, _ = numpy_engram(minus, embedding, projection, q_weight, k_weight)
        finite_difference[index] = np.sum((high - low) * cotangent) / (2 * delta)
    np.testing.assert_allclose(np.array(dx), finite_difference, rtol=2e-4, atol=2e-5)
