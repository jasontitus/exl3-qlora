"""Bounded CPU selected-token GQA + output-gate mathematical reference.

Boundary: caller supplies projected/normalized/rotated Q/K, V, raw elementwise
gate logits, and sequence-local routes. This does not perform QKV projection,
normalization, RoPE, pooling/selection, o_proj, cache mutation, or backward.
All seven inputs must be plain NumPy ndarrays, without subclasses or masks.
Arithmetic is FP64 on the supplied values, NOT native FP16 score/probability
rounding, Torch/Metal bit parity, or a complete Qwen forward. No device imports.
"""

from dataclasses import dataclass

import numpy as np

MAX_INPUT_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 8 << 20
MAX_HISTORY = 4096
MAX_QUERIES = 64
MAX_ROUTE_WIDTH = 2080


@dataclass(frozen=True)
class AttentionResult:
    """Immutable values; allocation counts describe arrays, not process RSS."""

    context_bytes: bytes
    context_shape: tuple
    largest_selected_count: int
    selected_kv_pair_bytes: int
    score_vector_bytes: int

    def context(self):
        return np.frombuffer(self.context_bytes, dtype="<f8").reshape(self.context_shape)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _validate(q, k, v, gate, routes, positions, lengths):
    arrays = (q, k, v, gate, routes, positions, lengths)
    _require(all(type(x) is np.ndarray for x in arrays), "Plain NumPy ndarrays required; subclasses and masks are unsupported")
    _require(sum(x.nbytes for x in arrays) <= MAX_INPUT_BYTES, "Input byte bound exceeded")
    _require(q.ndim == k.ndim == v.ndim == gate.ndim == 4, "Q/K/V/gate must be rank four")
    b, n, h, d = q.shape
    _require(1 <= b <= 2 and 1 <= n <= MAX_QUERIES and 1 <= h <= 24 and 1 <= d <= 256, "Q shape exceeds fixture bounds")
    _require(k.shape == v.shape and k.shape[0] == b and k.shape[-1] == d, "K/V shape mismatch")
    t, kvh = k.shape[1:3]
    _require(1 <= t <= MAX_HISTORY and 1 <= kvh <= 2 and h % kvh == 0, "Invalid GQA/history geometry")
    _require(gate.shape == q.shape, "Elementwise gate must match Q shape")
    floats = (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))
    _require(all(x.dtype in floats for x in (q, k, v, gate)), "Floating inputs must be FP16/32/64")
    _require(positions.dtype == np.int64 and positions.shape == (b, n), "Positions must be int64[B,Q]")
    _require(lengths.dtype == np.int64 and lengths.shape == (b,), "Lengths must be int64[B]")
    _require(routes.dtype == np.int32 and routes.ndim == 3 and routes.shape[:2] == (b, n), "Routes must be int32[B,Q,L]")
    width = routes.shape[2]
    _require(32 <= width <= MAX_ROUTE_WIDTH and width % 32 == 0, "Invalid padded route width")
    _require(b * n * h * d * 8 <= MAX_OUTPUT_BYTES, "Output byte bound exceeded")
    _require(all(np.isfinite(x).all() for x in (q, k, v, gate)), "Nonfinite input")
    _require(np.all((lengths >= 1) & (lengths <= t)), "Invalid sequence lengths")
    for batch in range(b):
        row = positions[batch]
        valid = row[row >= 0]
        _require(np.all(row >= -1) and np.all(row < lengths[batch]), "Position outside sequence")
        _require(np.all(row[:len(valid)] >= 0) and (len(valid) < 2 or np.all(np.diff(valid) == 1)), "Only contiguous queries with right padding supported")
        for query, pos in enumerate(row):
            route = routes[batch, query]
            _require(np.all(route >= -1), "Only -1 route padding is supported")
            ids = route[route >= 0]
            if pos == -1:
                _require(len(ids) == 0, "Padded queries must have no routes")
                continue
            _require(len(ids) > 0, "Valid query needs a nonempty attention set")
            _require(np.all(ids <= pos) and np.all(ids < lengths[batch]), "Future/out-of-sequence route")
            _require(len(np.unique(ids)) == len(ids), "Duplicate routes change softmax multiplicity")
    return b, n, h, d, kvh


def _sigmoid(x):
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    out[~positive] = e / (1 + e)
    return out


def attend(q, k, v, gate, routes, positions, lengths):
    """Compute flattened gated context [B,Q,Hq*D] over supplied unique routes.

    GQA uses contiguous groups: kv_head = q_head // (Hq/Hkv). Scores are
    dot(Q,K)/sqrt(D), followed by stable softmax, weighted V, then an elementwise
    sigmoid(gate). Route -1 entries are ignored wherever they occur. Explicit
    right-padded queries yield zero output; this is a CPU extension, not proof of
    native padded execution. Inputs are never modified. Supplied routes are
    validated for mathematical safety, not certified as an authentic selector
    capture. No [all queries,selected K/V] or [history,history] array is made.
    """
    b, n, h, d, kvh = _validate(q, k, v, gate, routes, positions, lengths)
    out = np.zeros((b, n, h, d), dtype=np.float64)
    maximum = 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for batch in range(b):
                for query in range(n):
                    if positions[batch, query] == -1:
                        continue
                    ids = routes[batch, query]
                    ids = ids[ids >= 0]
                    maximum = max(maximum, len(ids))
                    for head in range(h):
                        kv_head = head // (h // kvh)
                        # Only one query/head's selected K/V is materialized.
                        ks = k[batch, ids, kv_head].astype(np.float64)
                        vs = v[batch, ids, kv_head].astype(np.float64)
                        scores = np.sum(ks * q[batch, query, head].astype(np.float64), axis=1) / np.sqrt(d)
                        weights = np.exp(scores - np.max(scores))
                        weights /= np.sum(weights)
                        context = np.sum(weights[:, None] * vs, axis=0)
                        out[batch, query, head] = context * _sigmoid(gate[batch, query, head].astype(np.float64))
                        del ks, vs, scores, weights, context
    except FloatingPointError as error:
        raise ValueError("Nonfinite reference arithmetic") from error
    _require(np.isfinite(out).all(), "Nonfinite reference output")
    flat = out.reshape(b, n, h * d)
    return AttentionResult(flat.astype("<f8", copy=False).tobytes(), flat.shape, maximum, 2 * maximum * d * 8, maximum * 8)
