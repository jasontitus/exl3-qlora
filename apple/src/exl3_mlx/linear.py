"""Frozen EXL3 base, activation-side Hadamards, recomputing VJP, trainable LoRA."""

import math

import mlx.core as mx
import mlx.nn as nn

from .materialize import materialize
from .trellis import reconstruct, validate_trellis


def hadamard128(x):
    """Normalized independent 128-element blocks, never a whole-width FWHT."""
    shape = (*x.shape[:-1], x.shape[-1] // 128, 128)
    return mx.hadamard_transform(x.reshape(shape)).reshape(x.shape)


def _base(x, trellis, suh, svh, codebook, chunk_size, transpose=False, eager=False):
    # Use fp32 for transforms/accumulation (including dX); cast only at the boundary.
    n = svh.size
    left, right = (svh, suh) if transpose else (suh, svh)
    if x.size == 0:
        return mx.zeros((*x.shape[:-1], right.size), dtype=x.dtype)
    xh = hadamard128(x.astype(mx.float32) * left)
    chunks = []
    total = None
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        w = reconstruct(trellis, codebook, start=start, width=end - start)
        if transpose:
            part = xh[..., start:end] @ w.T
            total = part if total is None else total + part
            if eager:
                total = materialize(total)
        else:
            part = xh @ w
            if eager:
                part = materialize(part)
            chunks.append(part)
    inner = total if transpose else mx.concatenate(chunks, axis=-1)
    return (hadamard128(inner) * right).astype(x.dtype)


class EXL3Linear(nn.Module):
    """One resident EXL3 matrix; the separate streaming module provides disk loading.

    Shapes are storage dimensions; unsupported unpadded shapes fail explicitly.
    No dense weights are retained across calls or saved as autodiff residuals.
    chunk_size controls individual decode allocations, not total lazy-graph peak.
    """

    def __init__(self, trellis, suh, svh, *, codebook="mul1", bias=None, chunk_size=512):
        super().__init__()
        self.bpw = validate_trellis(trellis, codebook)
        k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
        if k % 128 or n % 128:
            raise ValueError("EXL3 linear storage dimensions must be multiples of 128")
        if suh.shape != (k,) or svh.shape != (n,):
            raise ValueError(f"Expected suh[{k}] and svh[{n}]")
        if not all(mx.issubdtype(s.dtype, mx.floating) for s in (suh, svh)):
            raise ValueError("suh and svh must be floating point")
        if bias is not None and (bias.shape != (n,) or not mx.issubdtype(bias.dtype, mx.floating)):
            raise ValueError(f"bias must be a floating point vector of length {n}")
        if not isinstance(chunk_size, int) or chunk_size < 128 or chunk_size % 128:
            raise ValueError("chunk_size must be a positive multiple of 128")
        self.in_features, self.out_features = k, n
        self.codebook, self.chunk_size = codebook, chunk_size
        self.trellis, self.suh, self.svh = trellis, suh, svh
        if bias is not None:
            self.bias = bias
        self.freeze()

    def __call__(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input [..., {self.in_features}], got {x.shape}")
        if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("Input must be float16, bfloat16, or float32")
        # Captured arrays are constants under MLX custom_function. Only x is a primal;
        # no fake gradients or large zero buffers are created for the frozen trellis.
        trellis, suh, svh = self.trellis, self.suh, self.svh
        codebook, chunk_size = self.codebook, self.chunk_size

        @mx.custom_function
        def frozen_linear(inp):
            return materialize(
                _base(mx.stop_gradient(inp), trellis, suh, svh, codebook, chunk_size)
            )

        @frozen_linear.vjp
        def frozen_vjp(primals, cotangent, output):
            return materialize(
                _base(cotangent, trellis, suh, svh, codebook, chunk_size, transpose=True)
            )

        out = frozen_linear(x)
        if "bias" in self:
            out = out + mx.stop_gradient(self.bias).astype(x.dtype)
        return out


class EXL3QLoRALinear(nn.Module):
    """A/rank/B adapter in the original basis over a frozen EXL3 linear."""

    def __init__(self, base, *, rank=16, alpha=32.0):
        super().__init__()
        if not isinstance(base, EXL3Linear):
            raise TypeError("base must be an EXL3Linear")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        bound = 1 / math.sqrt(base.in_features)
        self.lora_a = mx.random.uniform(-bound, bound, (base.in_features, rank))
        self.lora_b = mx.zeros((rank, base.out_features))
        self.base.freeze()

    def __call__(self, x):
        adapter = ((x.astype(mx.float32) @ self.lora_a) @ self.lora_b) * self.scaling
        return self.base(x) + adapter.astype(x.dtype)
