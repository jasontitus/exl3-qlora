"""Bounded CPU QSA routing oracle, on already projected/normalized/roped inputs.

FP32 score semantics follow QSAIndexer.block_scores/select_indices_ref. Ties use
score descending/index ascending (native selector contract), not Torch topk's
unspecified tie order. This is NOT parity with native FP16 score rounding, key
pooling/projection/RoPE, paged cache, sparse attention or backward.
Routes use compact canonical sequence-local layout: valid blocks then tail, then
-1 padding. This is NOT the slotwise Torch layout (tail after masked pool slots),
nor bitwise Torch FP32 reduction parity.
Padding extension: right-padded contiguous sequences only; no document packing.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

MAX_TOKENS = 4096
MAX_INPUT_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 16 * 1024**2
MAX_SCORE_BYTES = 1024**2


def integer(x, name, low, high):
    if type(x) is not int or not low <= x <= high:
        raise ValueError(f"{name} must be integer in [{low}, {high}]")
    return x


def digest(a):
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class Routes:
    # Immutable bytes deliberately prevent accidental mutation of captured routes.
    payload: bytes
    shape: tuple
    input_identity: str
    checksum: str
    peak_score_elements: int

    def array(self):
        return np.frombuffer(self.payload, dtype='<i4').reshape(self.shape)


def _validate(q, pooled, positions, lengths, budget, tile):
    integer(budget, 'budget', 4, 2048)
    if budget % 4:
        raise ValueError('budget must be divisible by fixed Qwen compression ratio four')
    integer(tile, 'query_tile', 1, 64)
    for a in (q, pooled, positions, lengths):
        if not isinstance(a, np.ndarray):
            raise ValueError('inputs must be NumPy arrays')
    if q.ndim != 4 or pooled.ndim != 3:
        raise ValueError('q[B,Q,H,D], pooled[B,P,D] required')
    b, n, h, d = q.shape
    if not (1 <= b <= 2 and 1 <= n <= MAX_TOKENS and 1 <= h <= 4 and 1 <= d <= 128):
        raise ValueError('reference dimensions exceed explicit bounds')
    if pooled.shape[0] != b or pooled.shape[2] != d or pooled.shape[1] > MAX_TOKENS // 4:
        raise ValueError('pooled shape mismatch')
    if q.dtype != np.float32 or pooled.dtype != np.float32:
        raise ValueError('FP32 inputs required; no silent dtype conversion')
    if positions.shape != (b, n) or lengths.shape != (b,) or positions.dtype != np.int64 or lengths.dtype != np.int64:
        raise ValueError('positions[B,Q] and lengths[B] int64 required')
    if sum(a.nbytes for a in (q, pooled, positions, lengths)) > MAX_INPUT_BYTES:
        raise ValueError('input byte bound exceeded')
    if not np.isfinite(q).all() or not np.isfinite(pooled).all():
        raise ValueError('nonfinite inputs')
    if np.any(lengths < 1) or np.any(lengths > MAX_TOKENS):
        raise ValueError('invalid sequence lengths')
    if pooled.shape[1] != int(lengths.max()) // 4:
        raise ValueError('pooled must contain exactly complete blocks of longest sequence')
    for row, length in zip(positions, lengths):
        if np.any(row < -1) or np.any(row >= length):
            raise ValueError('position outside valid sequence; -1 only padding sentinel')
        valid = row[row >= 0]
        if np.any(row[:len(valid)] < 0) or (len(valid) > 1 and np.any(np.diff(valid) != 1)):
            raise ValueError('only contiguous queries followed by right padding supported')
    width = ((budget + 3 + 31) // 32) * 32
    if b * n * width * 4 > MAX_OUTPUT_BYTES:
        raise ValueError('route output byte bound exceeded')
    if min(tile, n) * h * pooled.shape[1] * 4 > MAX_SCORE_BYTES:
        raise ValueError('score tile byte bound exceeded')
    identity = hashlib.sha256(json.dumps({'inputs': [digest(a) for a in (q, pooled, positions, lengths)], 'budget': budget, 'arithmetic': 'fp32-relu-headsum-scale-v1', 'ties': 'index-ascending'}, sort_keys=True).encode()).hexdigest()
    return width, identity


def select(q, pooled, positions, lengths, *, budget=2048, query_tile=16):
    """Return sequence-local IDs, -1 slots, no duplicate/invalid/future tokens.

    Keys must already be means of COMPLETE four-token raw-key blocks, normalized
    and rotated at block START positions. Selection never includes incomplete
    pooled keys. Full causal tail is appended separately (0..3 tokens).
    """
    width, identity = _validate(q, pooled, positions, lengths, budget, query_tile)
    b, n, h, d = q.shape
    out = np.full((b, n, width), -1, dtype='<i4')
    peak = 0
    for batch in range(b):
        for start in range(0, n, query_tile):
            end = min(start + query_tile, n)
            # No [T,T] mask, selected K/V, or all-query score array.
            with np.errstate(over='raise', invalid='raise'):
                scores = np.einsum('qhd,pd->qhp', q[batch, start:end], pooled[batch], optimize=False)
                peak = max(peak, scores.size)
                np.maximum(scores, 0, out=scores)
                scores = scores.sum(axis=1, dtype=np.float32) * np.float32(1 / np.sqrt(d))
            if not np.isfinite(scores).all():
                raise ValueError('nonfinite scores')
            for local, pos in enumerate(positions[batch, start:end]):
                if pos < 0:
                    continue
                complete = (int(pos) + 1) // 4
                ids = np.arange(complete)
                chosen = np.lexsort((ids, -scores[local, :complete]))[:budget // 4]
                tokens = np.concatenate(((chosen[:, None] * 4 + np.arange(4)).reshape(-1), np.arange(complete * 4, pos + 1)))
                out[batch, start + local, :len(tokens)] = tokens
    payload = out.tobytes()
    return Routes(payload, out.shape, identity, hashlib.sha256(payload).hexdigest(), peak)


def replay(capture, q, pooled, positions, lengths, *, budget=2048, query_tile=16):
    """Replay saved routes without recomputing top-k; exact input binding required."""
    width, identity = _validate(q, pooled, positions, lengths, budget, query_tile)
    if not isinstance(capture, Routes) or capture.input_identity != identity or capture.shape != (*q.shape[:2], width):
        raise ValueError('route capture/input mismatch')
    if len(capture.payload) != int(np.prod(capture.shape)) * 4 or hashlib.sha256(capture.payload).hexdigest() != capture.checksum:
        raise ValueError('route capture corruption')
    # This checksum is consistency, not authentication of a separately supplied capture.
    return capture.array().copy()
