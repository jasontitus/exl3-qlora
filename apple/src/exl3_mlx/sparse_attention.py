"""Query-tiled latent attention with an explicit, bounded first-order VJP.

The shared latent supplies both keys and values. Integer selection is discrete;
gradients include both uses of each selected latent and accumulate repeated reads.
Only one query tile's scores and gathered keys are live at an autograd boundary.
"""

import math

import mlx.core as mx

from .materialize import materialize


def _gather(values, indices):
    batch, length, dim = values.shape
    offset = (mx.arange(batch, dtype=mx.int32) * length)[:, None, None]
    flat = (mx.maximum(indices, 0) + offset).reshape(-1)
    return values.reshape(-1, dim)[flat].reshape(*indices.shape, dim)


def _tile(q, window, compressed, selected, sink, start, window_size):
    batch, count, _, dim = q.shape
    positions = start + mx.arange(count)
    width = min(window_size, window.shape[1])
    wi = mx.maximum(positions[:, None] - width + 1, 0) + mx.arange(width)[None, :]
    wi = mx.where(wi <= positions[:, None], wi, -1)
    wi = mx.broadcast_to(wi[None], (batch, count, width))
    keys, valid = _gather(window, wi), wi >= 0
    if selected.shape[-1]:
        keys = mx.concatenate([keys, _gather(compressed, selected)], axis=2)
        valid = mx.concatenate([valid, selected >= 0], axis=-1)
    keys, q32 = keys.astype(mx.float32), q.astype(mx.float32)
    scores = mx.einsum("bqhd,bqkd->bqhk", q32, keys) / math.sqrt(dim)
    scores = mx.where(valid[:, :, None, :], scores, -mx.inf)
    extra = mx.broadcast_to(sink[None, None, :, None], (*scores.shape[:-1], 1))
    all_probs = mx.softmax(mx.concatenate([scores, extra], axis=-1), axis=-1)
    probs = all_probs[..., :-1]
    out = mx.einsum("bqhk,bqkd->bqhd", probs, keys)
    return out, probs, all_probs[..., -1], keys, wi, valid


def _scatter(accumulated, indices, values):
    batch, length, dim = accumulated.shape
    offset = (mx.arange(batch, dtype=mx.int32) * length)[:, None, None]
    flat = (mx.maximum(indices, 0) + offset).reshape(-1)
    values = mx.where((indices >= 0)[..., None], values, 0.0)
    result = accumulated.reshape(-1, dim).at[flat].add(values.reshape(-1, dim))
    return materialize(result.reshape(accumulated.shape))


def sparse_latent_attention(q, window, compressed, selected, sink, *, window_size, query_chunk=16):
    """Attend to the causal window and selected compressed positions, with a sink.

    q: [B,S,H,D], window: [B,S,D], compressed: [B,C,D],
    selected: [B,S,K] indices into compressed, with -1 padding. The caller must
    establish compressed-entry causality before selection. The sink adds only a
    denominator term and is differentiable for isolated numerical verification.
    """
    if type(query_chunk) is not int or query_chunk < 1 or window_size < 1:
        raise ValueError("Use positive query_chunk and window_size")
    if q.ndim != 4 or window.shape != (q.shape[0], q.shape[1], q.shape[3]):
        raise ValueError("Invalid query/window shapes")
    if compressed.ndim != 3 or (compressed.shape[0], compressed.shape[2]) != (
        q.shape[0],
        q.shape[3],
    ):
        raise ValueError("Invalid compressed latent shape")
    if selected.ndim != 3 or selected.shape[:2] != q.shape[:2]:
        raise ValueError("Invalid selected-index shape")
    if not mx.issubdtype(selected.dtype, mx.integer):
        raise ValueError("Selected indices must be integers")
    if selected.shape[-1] and mx.any((selected < -1) | (selected >= compressed.shape[1])).item():
        raise ValueError("Selected index is outside compressed latents")
    if sink.shape != (q.shape[2],):
        raise ValueError("Invalid attention sink shape")
    selected = materialize(mx.stop_gradient(selected))

    @mx.custom_function
    def attend(query, raw, pooled, sinks):
        query, raw, pooled, sinks = map(mx.stop_gradient, (query, raw, pooled, sinks))
        pieces = []
        for start in range(0, query.shape[1], query_chunk):
            end = min(start + query_chunk, query.shape[1])
            out = _tile(
                query[:, start:end],
                raw,
                pooled,
                selected[:, start:end],
                sinks,
                start,
                window_size,
            )[0]
            pieces.append(materialize(out.astype(query.dtype)))
        return materialize(mx.concatenate(pieces, axis=1))

    @attend.vjp
    def backward(primals, cotangent, output):
        query, raw, pooled, sinks = map(mx.stop_gradient, primals)
        cotangent = mx.stop_gradient(cotangent)
        raw_grad = mx.zeros(raw.shape, dtype=mx.float32)
        pooled_grad = mx.zeros(pooled.shape, dtype=mx.float32)
        sink_grad = mx.zeros(sinks.shape, dtype=mx.float32)
        query_grad = []
        scale = 1.0 / math.sqrt(query.shape[-1])
        for start in range(0, query.shape[1], query_chunk):
            end = min(start + query_chunk, query.shape[1])
            qt = query[:, start:end].astype(mx.float32)
            dy = cotangent[:, start:end].astype(mx.float32)
            indices = selected[:, start:end]
            _, p, ps, keys, wi, _ = _tile(qt, raw, pooled, indices, sinks, start, window_size)
            dp = mx.einsum("bqhd,bqkd->bqhk", dy, keys)
            expectation = mx.sum(p * dp, axis=-1, keepdims=True)
            ds = p * (dp - expectation)
            dq = mx.einsum("bqhk,bqkd->bqhd", ds, keys) * scale
            dk = mx.einsum("bqhk,bqhd->bqkd", ds, qt) * scale
            dk = dk + mx.einsum("bqhk,bqhd->bqkd", p, dy)
            query_grad.append(materialize(dq.astype(query.dtype)))
            raw_grad = _scatter(raw_grad, wi, dk[:, :, : wi.shape[-1]])
            if indices.shape[-1]:
                pooled_grad = _scatter(pooled_grad, indices, dk[:, :, wi.shape[-1] :])
            sink_grad = materialize(sink_grad - mx.sum(ps * expectation[..., 0], axis=(0, 1)))
        return (
            materialize(mx.concatenate(query_grad, axis=1)),
            materialize(raw_grad.astype(raw.dtype)),
            materialize(pooled_grad.astype(pooled.dtype)),
            materialize(sink_grad.astype(sinks.dtype)),
        )

    return attend(q, window, compressed, sink)
