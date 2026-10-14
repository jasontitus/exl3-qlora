"""Bounded FP64 adjoint of preprocessed selected-token attention, routes fixed.

This differentiates the approved mathematical reference, not native rounding,
QKV projection/norm/RoPE, hard route selection, caches, or a complete model.
"""

from dataclasses import dataclass

import numpy as np
import qwen_qsa_attention_reference as forward

MAX_INPUT_BYTES = 1 << 20
MAX_GRADIENT_BYTES = 1 << 20
MAX_SELECTED_COMPONENTS = 1 << 18
MAX_HISTORY = 64
MAX_QUERIES = 8
MAX_ROUTE_WIDTH = 64


@dataclass(frozen=True)
class FrozenArray:
    payload: bytes
    shape: tuple

    def array(self):
        return np.frombuffer(self.payload, dtype="<f8").reshape(self.shape)


@dataclass(frozen=True)
class AdjointResult:
    dq: FrozenArray
    dk: FrozenArray
    dv: FrozenArray
    dgate: FrozenArray
    selected_components: int
    gradient_bytes: int


def _require(value, message):
    if not value:
        raise ValueError(message)


def _validate(q, k, v, gate, routes, positions, lengths, cotangent):
    arrays = (q, k, v, gate, routes, positions, lengths, cotangent)
    _require(all(type(x) is np.ndarray for x in arrays), "Plain NumPy ndarrays required")
    _require(sum(x.nbytes for x in arrays) <= MAX_INPUT_BYTES, "Adjoint input byte bound exceeded")
    _require(q.ndim == k.ndim == v.ndim == gate.ndim == 4, "Q/K/V/gate must be rank four")
    _require(q.shape[1] <= MAX_QUERIES and k.shape[1] <= MAX_HISTORY, "Adjoint query/history bound exceeded")
    _require(routes.ndim == 3 and routes.shape[2] <= MAX_ROUTE_WIDTH, "Adjoint route width bound exceeded")
    _require(all(x.dtype == np.float64 for x in (q, k, v, gate, cotangent)), "Adjoint floating inputs must be FP64")
    gradient_bytes = sum(x.nbytes for x in (q, k, v, gate))
    _require(gradient_bytes <= MAX_GRADIENT_BYTES, "Adjoint gradient byte bound exceeded")
    # The private validator is an exact pinned copy of the approved forward v2.
    # It supplies unchanged GQA, causality, uniqueness, padding and finite checks.
    b, n, h, d, kvh = forward._validate(q, k, v, gate, routes, positions, lengths)
    _require(cotangent.shape == (b, n, h * d), "Cotangent must match flattened output [B,Q,Hq*D]")
    _require(np.isfinite(cotangent).all(), "Nonfinite cotangent")
    work = int(np.count_nonzero(routes >= 0)) * h * d
    _require(work <= MAX_SELECTED_COMPONENTS, "Adjoint selected component work bound exceeded")
    return b, n, h, d, kvh, gradient_bytes, work


def adjoint(q, k, v, gate, routes, positions, lengths, cotangent):
    """Return immutable dQ/dK/dV/dgate for a fixed-route output cotangent.

    K/V adjoints accumulate across all queries and query heads sharing a KV head.
    Unselected tokens and padded query rows receive zero derivatives. A finite
    cotangent on a padded row is ignored because that forward output is constant
    zero. No derivative is returned for routes, positions, lengths or selection.
    Inputs may be read-only/noncontiguous plain ndarrays and are never modified.
    """
    b, n, h, d, kvh, gradient_bytes, work = _validate(q, k, v, gate, routes, positions, lengths, cotangent)
    dq, dk, dv, dg = (np.zeros_like(x) for x in (q, k, v, gate))
    dy = cotangent.reshape(b, n, h, d)
    scale = 1 / np.sqrt(d)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for bi in range(b):
                for qi in range(n):
                    if positions[bi, qi] == -1:
                        continue
                    ids = routes[bi, qi]
                    ids = ids[ids >= 0]
                    for hi in range(h):
                        ki = hi // (h // kvh)
                        keys, values = k[bi, ids, ki], v[bi, ids, ki]
                        scores = np.sum(keys * q[bi, qi, hi], axis=1) / np.sqrt(d)
                        weights = np.exp(scores - np.max(scores))
                        weights /= np.sum(weights)
                        context = np.sum(weights[:, None] * values, axis=0)
                        sigmoid = forward._sigmoid(gate[bi, qi, hi])
                        dcontext = dy[bi, qi, hi] * sigmoid
                        dweights = np.sum(values * dcontext, axis=1)
                        dscores = weights * (dweights - np.sum(weights * dweights))
                        dq[bi, qi, hi] = np.sum(dscores[:, None] * keys, axis=0) * scale
                        # IDs are unique within this route; sequential additions
                        # preserve contributions from every query and GQA head.
                        dk[bi, ids, ki] += dscores[:, None] * q[bi, qi, hi] * scale
                        dv[bi, ids, ki] += weights[:, None] * dcontext
                        # This stable real-valued derivative retains the small
                        # positive sigmoid slope even if sigmoid rounds to 1.
                        exp_negative_abs = np.exp(-np.abs(gate[bi, qi, hi]))
                        dsigmoid = exp_negative_abs / (1 + exp_negative_abs) ** 2
                        dg[bi, qi, hi] = dy[bi, qi, hi] * context * dsigmoid
    except FloatingPointError as error:
        raise ValueError("Nonfinite adjoint arithmetic") from error
    _require(all(np.isfinite(x).all() for x in (dq, dk, dv, dg)), "Nonfinite adjoint output")
    frozen = tuple(FrozenArray(x.astype("<f8", copy=False).tobytes(), x.shape) for x in (dq, dk, dv, dg))
    return AdjointResult(*frozen, work, gradient_bytes)
