"""Differentiable V4.1 full-prefill attention, with frozen sparse index selection.

Activation quantization uses an explicitly chosen identity straight-through VJP.
The indexer and its candidate selection are discrete, nondifferentiable routes.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .hyperconnections import rms_norm
from .materialize import materialize
from .sparse_attention import sparse_latent_attention


def pow2_exact(exponent):
    # Metal's generic pow(2, n) can be one ULP off even for integer n. That
    # changes round-to-even midpoint decisions, so construct the IEEE exponent.
    # Float-to-unsigned conversion clamps negative exponents to zero on Metal.
    # Apply the IEEE bias in signed arithmetic before constructing the bits.
    return ((exponent.astype(mx.int32) + 127).astype(mx.uint32) << 23).view(mx.float32)


def round_e4m3(x):
    a = mx.minimum(mx.abs(x), 448.0)
    exponent = ((a.view(mx.uint32) >> 23) & 255).astype(mx.int32) - 127
    exponent = mx.maximum(exponent - 3, -9)
    return mx.sign(x) * mx.round(a * pow2_exact(-exponent)) * pow2_exact(exponent)


_QUANTIZATION_CHUNK_ELEMENTS = 1 << 20


def _quantize_blocks(v, kind):
    """Quantize independent rows, each exactly one format-specific scale group."""
    amax = mx.max(mx.abs(v), axis=-1, keepdims=True)
    if kind == "fp8":
        scale = pow2_exact(mx.ceil(mx.log2(mx.maximum(amax, 1e-4) / 448.0)))
        q = round_e4m3(v / scale)
    else:
        scale = (
            round_e4m3(mx.maximum(amax, 6 * 2**-9) / 6.0)
            if kind == "fp4"
            else pow2_exact(mx.ceil(mx.log2(mx.maximum(amax, 6 * 2**-126) / 6.0)))
        )
        z = mx.clip(v / scale, -6, 6)
        a = mx.abs(z)
        boundaries = mx.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        # At a midpoint select the even E2M1 code (round-to-nearest-even).
        indices = mx.sum(
            (a[..., None] > boundaries) | ((a[..., None] == boundaries) & (mx.arange(7) % 2 == 1)),
            axis=-1,
        )
        q = mx.sign(z) * mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[indices]
    return q * scale


def quantize_activation(x, kind="fp8"):
    """Exact format round-trip, bounded forward scratch, identity surrogate VJP.

    Complete groups are flattened without mixing last-axis feature boundaries.
    Materialization cuts the rounding graph inside an enclosing gradient trace;
    it does not detach the custom operation's declared activation derivative.
    """
    if kind not in ("fp8", "fp4", "fp4_e8m0"):
        raise ValueError("Unknown activation quantization kind")
    block = 16 if kind == "fp4" else 32
    if x.ndim < 1 or x.shape[-1] % block:
        raise ValueError(f"Activation feature dimension must be divisible by {block}")

    @mx.custom_function
    def fake_quant(inp):
        inp = mx.stop_gradient(inp)
        if not inp.size:
            return materialize(inp)
        groups = inp.reshape(-1, block)
        group_chunk = max(1, _QUANTIZATION_CHUNK_ELEMENTS // block)
        pieces = []
        for start in range(0, groups.shape[0], group_chunk):
            values = groups[start : start + group_chunk].astype(mx.float32)
            pieces.append(materialize(_quantize_blocks(values, kind).astype(inp.dtype)))
        return materialize(mx.concatenate(pieces, axis=0).reshape(inp.shape))

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
    """All-visible reference interface, retained for small numerical fixtures."""
    if ratio:
        if compressed is None:
            raise ValueError("Compressed attention has no source KV")
        selected = causal_indices(q.shape[0], q.shape[1], compressed.shape[1], ratio)
    else:
        compressed = mx.zeros((q.shape[0], 0, q.shape[-1]), dtype=window.dtype)
        selected = mx.zeros((q.shape[0], q.shape[1], 0), dtype=mx.int32)
    return sparse_latent_attention(q, window, compressed, selected, sink, window_size=window_size)


def causal_indices(batch, length, entries, ratio):
    indices = mx.broadcast_to(mx.arange(entries)[None, None], (batch, length, entries))
    visible = (mx.arange(length) + 1) // ratio
    return mx.where(indices < visible[None, :, None], indices, -1).astype(mx.int32)


def empty_attention_state(batch, length, head_dim, index_dim, dtype):
    return (
        mx.zeros((batch, 0, head_dim), dtype=dtype),
        mx.zeros((batch, 0, index_dim), dtype=dtype),
        mx.zeros((batch, length, 0), dtype=mx.int32),
        mx.zeros((batch, length, 0), dtype=mx.uint8),
    )


def pack_candidate_blocks(keep):
    pad = (-keep.shape[-1]) % 8
    if pad:
        keep = mx.pad(keep, [(0, 0), (0, 0), (0, pad)])
    bits = mx.array(1, dtype=mx.uint32) << mx.arange(8, dtype=mx.uint32)
    shape = (*keep.shape[:-1], keep.shape[-1] // 8, 8)
    return mx.sum(keep.reshape(shape).astype(mx.uint32) * bits, axis=-1).astype(mx.uint8)


def candidate_mask(packed, entries, block_size):
    # Expand only one query tile's packed blocks, never an entire [S,S] mask.
    blocks = mx.arange((entries + block_size - 1) // block_size)
    keep = ((packed[..., blocks // 8].astype(mx.uint32) >> (blocks % 8)) & 1) != 0
    return mx.repeat(keep, block_size, axis=-1)[..., :entries]


def select_index_scores(
    scores, visible, *, topk, candidate_blocks=0, block_size=8, candidates=None
):
    """Causal top-k, with an explicit low-position tie policy and sorted output.

    Candidate source layers choose their blocks from their own scores. Consumers
    first restrict scores to those blocks, then apply their own index ranking.
    """
    entries = scores.shape[-1]
    scores = mx.where(mx.arange(entries)[None, None] < visible[None, :, None], scores, -mx.inf)
    packed = None
    if candidate_blocks:
        pad = (-entries) % block_size
        padded = mx.pad(scores, [(0, 0), (0, 0), (0, pad)], constant_values=-mx.inf)
        blocks = padded.reshape(*scores.shape[:-1], -1, block_size).max(axis=-1)
        last = mx.where(visible > 0, (visible - 1) // block_size, -1)
        blocks = mx.where(mx.arange(blocks.shape[-1]) == last[None, :, None], mx.inf, blocks)
        chosen = mx.argsort(-blocks, axis=-1)[..., : min(candidate_blocks, blocks.shape[-1])]
        reachable = mx.take_along_axis(blocks, chosen, axis=-1) > -mx.inf
        keep = mx.put_along_axis(mx.zeros(blocks.shape, dtype=mx.bool_), chosen, reachable, axis=-1)
        packed = pack_candidate_blocks(keep)
    elif candidates is not None and candidates.shape[-1]:
        scores = mx.where(candidate_mask(candidates, entries, block_size), scores, -mx.inf)
    chosen = mx.argsort(-scores, axis=-1)[..., : min(topk, entries)].astype(mx.int32)
    valid = mx.take_along_axis(scores, chosen, axis=-1) > -mx.inf
    # Invalid slots sort to the end. Do not allow a candidate-masked but causal
    # position back into attention when fewer than topk candidates survive.
    chosen = mx.sort(mx.where(valid, chosen, entries), axis=-1)
    return mx.where(chosen < entries, chosen, -1).astype(mx.int32), packed


class FrozenIndexer(nn.Module):
    """Frozen routing with MLX-LM's FP32 index-score accumulation convention.

    The pinned Torch kernel can round intermediate scores in BF16; this does
    not claim bitwise selection parity with that implementation near rank ties.
    """

    def __init__(self, layer_id, config, linears, k_norm=None, *, query_chunk=16):
        super().__init__()
        self.layer_id, self.config, self.linears = layer_id, config, linears
        self.k_norm, self.query_chunk = k_norm, query_chunk
        self.freeze()

    def keys(self, latent):
        if self.k_norm is None:
            raise ValueError("Only compressed-KV sources produce index keys")
        c = self.config
        latent = mx.stop_gradient(latent)
        keys = rms_norm(self.linears["wk"](latent), self.k_norm, c["rms_norm_eps"])
        positions = mx.arange(latent.shape[1]) * c["compress_ratios"][self.layer_id]
        keys = rope(keys, positions, c, compressed=True)
        return materialize(quantize_activation(keys, "fp4_e8m0"))

    def __call__(self, x, qr, keys, candidates):
        c = self.config
        batch, length, _ = x.shape
        entries = keys.shape[1]
        if not entries:
            return mx.zeros((batch, length, 0), dtype=mx.int32), candidates
        x, qr, keys = map(mx.stop_gradient, (x, qr, keys))
        source = self.layer_id == c.get("candidate_source_layer_id", -1)
        heads, dim = c["index_n_heads"], c["index_head_dim"]
        positions = mx.arange(length)
        query = self.linears["wq_b"](qr).reshape(batch, length, heads, dim)
        query = rope(query, positions, c, compressed=True)
        query = materialize(quantize_activation(query, "fp4_e8m0"))
        weights = materialize(self.linears["weights_proj"](x).astype(mx.float32))
        selected, candidate_pieces = [], []
        for start in range(0, length, self.query_chunk):
            end = min(start + self.query_chunk, length)
            # ReLU precedes the signed head-weight sum. Both reductions use fp32.
            score = mx.einsum(
                "bqhd,bkd->bqhk", query[:, start:end].astype(mx.float32), keys.astype(mx.float32)
            )
            score = mx.maximum(score, 0.0) * dim**-0.5
            score = mx.sum(score * (weights[:, start:end, :, None] * heads**-0.5), axis=2)
            visible = (positions[start:end] + 1) // c["compress_ratios"][self.layer_id]
            indices, packed = select_index_scores(
                score,
                visible,
                topk=c["index_topk"],
                candidate_blocks=c.get("candidate_topk_blocks", 0) if source else 0,
                block_size=c.get("candidate_block_size", 8),
                candidates=None if source else candidates[:, start:end],
            )
            selected.append(materialize(indices))
            if packed is not None:
                candidate_pieces.append(materialize(packed))
        return materialize(mx.concatenate(selected, axis=1)), (
            materialize(mx.concatenate(candidate_pieces, axis=1))
            if candidate_pieces
            else candidates
        )


class ShortAttention(nn.Module):
    """Full-prefill attention; historical class name retained for compatibility."""

    def __init__(
        self, layer_id, config, linears, norms, sink, wo_a_groups, compressor=None, indexer=None
    ):
        super().__init__()
        self.layer_id, self.config = layer_id, config
        self.linears, self.norms = linears, norms
        self.sink, self.wo_a_groups = sink, wo_a_groups
        self.compressor = compressor
        self.indexer = indexer
        self.freeze(keys=["norms", "sink"], recurse=False)

    def __call__(self, x, shared_kv):
        c = self.config
        b, s, _ = x.shape
        legacy = not isinstance(shared_kv, tuple)
        if legacy:
            state = empty_attention_state(b, s, c["head_dim"], c.get("index_head_dim", 1), x.dtype)
            shared_kv, index_keys, selected, candidates = (shared_kv, *state[1:])
        else:
            shared_kv, index_keys, selected, candidates = shared_kv
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
            if self.indexer is not None:
                index_keys = self.indexer.keys(latent)
            shared_kv = quantize_activation(rot(latent, mx.arange(latent.shape[1]) * ratio), "fp4")
        if ratio:
            if self.indexer is not None:
                selected, candidates = self.indexer(x, qr, index_keys, candidates)
            elif legacy or selected.shape[-1] == 0:
                if shared_kv.shape[1] > c["index_topk"]:
                    raise ValueError("Long compressed attention requires an index-source selection")
                selected = causal_indices(b, s, shared_kv.shape[1], ratio)
            attention_kv, attention_indices = shared_kv, selected
        else:
            attention_kv = mx.zeros((b, 0, c["head_dim"]), dtype=x.dtype)
            attention_indices = mx.zeros((b, s, 0), dtype=mx.int32)
        out = sparse_latent_attention(
            q,
            window,
            attention_kv,
            attention_indices,
            self.sink,
            window_size=c["sliding_window"],
        )
        out = quantize_activation(rot(out, inverse=True).reshape(b, s, c["o_groups"], -1))
        out = mx.concatenate(
            [proj(out[:, :, g]) for g, proj in enumerate(self.wo_a_groups)], axis=-1
        )
        state = shared_kv if legacy else (shared_kv, index_keys, selected, candidates)
        return self.linears["wo_b"](out), state
