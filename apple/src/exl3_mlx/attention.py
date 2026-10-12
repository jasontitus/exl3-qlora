"""Differentiable V4.1 short-prefill attention (no decode cache).

For at most index_topk tokens, every causally available compressed position fits
in the indexer selection. No indexer ranking or candidate pruning is needed in
this bounded case. Longer sequences fail, rather than changing attention math.
Activation quantization uses an explicitly chosen identity straight-through VJP.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .hyperconnections import rms_norm


def pow2_exact(exponent):
    # Metal's generic pow(2, n) can be one ULP off even for integer n. That
    # changes round-to-even midpoint decisions, so construct the IEEE exponent.
    return ((exponent.astype(mx.uint32) + 127) << 23).view(mx.float32)


def round_e4m3(x):
    a = mx.minimum(mx.abs(x), 448.0)
    exponent = ((a.view(mx.uint32) >> 23) & 255).astype(mx.int32) - 127
    exponent = mx.maximum(exponent - 3, -9)
    return mx.sign(x) * mx.round(a * pow2_exact(-exponent)) * pow2_exact(exponent)


def quantize_activation(x, kind="fp8"):
    @mx.custom_function
    def fake_quant(inp):
        inp = mx.stop_gradient(inp)
        block = 32 if kind == "fp8" else 16
        v = inp.astype(mx.float32).reshape(*inp.shape[:-1], -1, block)
        amax = mx.max(mx.abs(v), axis=-1, keepdims=True)
        if kind == "fp8":
            scale = pow2_exact(mx.ceil(mx.log2(mx.maximum(amax, 1e-4) / 448.0)))
            q = round_e4m3(v / scale)
        elif kind == "fp4":
            scale = round_e4m3(mx.maximum(amax, 6 * 2**-9) / 6.0)
            z = mx.clip(v / scale, -6, 6)
            a = mx.abs(z)
            boundaries = mx.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
            # At a midpoint select the even E2M1 code (round-to-nearest-even).
            indices = mx.sum(
                (a[..., None] > boundaries)
                | ((a[..., None] == boundaries) & (mx.arange(7) % 2 == 1)),
                axis=-1,
            )
            q = mx.sign(z) * mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[indices]
        else:
            raise ValueError("Unknown activation quantization kind")
        return (q * scale).reshape(inp.shape).astype(inp.dtype)

    @fake_quant.vjp
    def vjp(primals, cotangent, output):
        return cotangent

    return fake_quant(x)


def rope(x, positions, config, *, compressed=False, inverse=False):
    d = config["qk_rope_head_dim"]
    base = config["compress_rope_theta"] if compressed else config["rope_theta"]
    freqs = 1 / (base ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
    if compressed:
        r = config["rope_scaling"]

        def corrected(rotations):
            return (
                d
                * math.log(r["original_max_position_embeddings"] / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        lo = max(math.floor(corrected(r["beta_fast"])), 0)
        hi = min(math.ceil(corrected(r["beta_slow"])), d - 1)
        ramp = mx.clip((mx.arange(d // 2) - lo) / max(hi - lo, 1e-3), 0, 1)
        freqs = freqs * (1 - ramp) + freqs / r["factor"] * ramp
    angles = positions[:, None] * freqs[None, :] * (-1 if inverse else 1)
    shape = (1, x.shape[1], *((1,) * (x.ndim - 3)), d // 2)
    c, s = mx.cos(angles).reshape(shape), mx.sin(angles).reshape(shape)
    tail = x[..., -d:].astype(mx.float32)
    a, b = tail[..., ::2], tail[..., 1::2]
    rotated = mx.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(tail.shape)
    return mx.concatenate([x[..., :-d], rotated.astype(x.dtype)], axis=-1)


def latent_attention(q, window, compressed, sink, *, ratio, window_size):
    """Shared latent is both K and V; the sink contributes only to the denominator."""
    length = q.shape[1]
    positions = mx.arange(length)
    mask = (positions[:, None] >= positions[None, :]) & (
        positions[:, None] - positions[None, :] < window_size
    )
    kv = window
    if ratio:
        if compressed is None:
            raise ValueError("Compressed attention has no source KV")
        end_positions = (mx.arange(compressed.shape[1]) + 1) * ratio - 1
        mask = mx.concatenate([mask, positions[:, None] >= end_positions[None, :]], axis=-1)
        kv = mx.concatenate([window, compressed], axis=1)
    scores = mx.einsum("bshd,btd->bhst", q.astype(mx.float32), kv.astype(mx.float32)) / math.sqrt(
        q.shape[-1]
    )
    scores = mx.where(mask[None, None, :, :], scores, -float("inf"))
    extra = mx.broadcast_to(sink[None, :, None, None], (*scores.shape[:-1], 1))
    probs = mx.softmax(mx.concatenate([scores, extra], axis=-1), axis=-1)[..., :-1]
    return mx.einsum("bhst,btd->bshd", probs, kv.astype(mx.float32)).astype(q.dtype)


class ShortAttention(nn.Module):
    def __init__(self, layer_id, config, linears, norms, sink, wo_a_groups, compressor=None):
        super().__init__()
        self.layer_id, self.config = layer_id, config
        self.linears, self.norms = linears, norms
        self.sink, self.wo_a_groups = sink, wo_a_groups
        self.compressor = compressor
        self.freeze(keys=["norms", "sink"], recurse=False)

    def __call__(self, x, shared_kv):
        c = self.config
        b, s, _ = x.shape
        if s > c["index_topk"]:
            raise ValueError("Short-prefill requires sequence length <= index_topk")
        ratio = c["compress_ratios"][self.layer_id]
        pos = mx.arange(s)

        def rot(t, p=pos, inverse=False):
            return rope(t, p, c, compressed=bool(ratio), inverse=inverse)

        qr = rms_norm(self.linears["wq_a"](x), self.norms["q"], c["rms_norm_eps"])
        q = rot(self.linears["wq_b"](qr).reshape(b, s, c["num_attention_heads"], c["head_dim"]))
        window = quantize_activation(
            rot(rms_norm(self.linears["wkv"](x), self.norms["kv"], c["rms_norm_eps"]))
        )
        if self.compressor is not None:
            comp = self.compressor
            if ratio == 1:
                latent = comp["wkv"](x)
            else:
                end = s - s % ratio
                v = comp["wkv"](x.astype(mx.float32))[:, :end].reshape(b, -1, ratio, c["head_dim"])
                g = comp["wgate"](x.astype(mx.float32))[:, :end].reshape(v.shape)
                latent = mx.sum(v * mx.softmax(g, axis=2), axis=2).astype(x.dtype)
            latent = rms_norm(latent, self.norms["compressor"], c["rms_norm_eps"])
            shared_kv = quantize_activation(rot(latent, mx.arange(latent.shape[1]) * ratio), "fp4")
        out = latent_attention(
            q, window, shared_kv, self.sink, ratio=ratio, window_size=c["sliding_window"]
        )
        out = quantize_activation(rot(out, inverse=True).reshape(b, s, c["o_groups"], -1))
        out = mx.concatenate(
            [proj(out[:, :, g]) for g, proj in enumerate(self.wo_a_groups)], axis=-1
        )
        return self.linears["wo_b"](out), shared_kv
