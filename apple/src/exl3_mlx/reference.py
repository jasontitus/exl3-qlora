"""Independent NumPy oracle adapted from tests/testlib/trellis.py at repository root.

Upstream 068f586c00724e3eabbf4165fb66c355da9af4df; same repository MIT license.
Kept separate from runtime Metal decoder; deliberately slow bit-array unpacking.
Only the torch import and torch dense-Hadamard reference have been replaced.
"""

import numpy as np

FRAC_MASK = 0xAAAA


def frac(K) -> tuple[int, int] | None:
    """(KA, MASK) of a half-integer rate, None for integer K"""
    if float(K).is_integer():
        return None
    ka = int(K)
    assert ka + 0.5 == K
    return ka, FRAC_MASK


def step_widths(K) -> np.ndarray:
    """(256,) D(i), bits each ring position adds"""
    f = frac(K)
    if f is None:
        return np.full(256, int(K), dtype=np.int64)
    ka, mask = f
    return np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)], dtype=np.int64)


def tile_perm() -> np.ndarray:
    """(256,) row-major tile element held by ring position i"""
    perm = np.zeros(256, dtype=np.int64)
    for t in range(32):
        for j in range(8):
            r = (t % 4) * 2 + (j & 1) + (8 if j & 2 else 0)
            c = t // 4 + (8 if j & 4 else 0)
            perm[t * 8 + j] = r * 16 + c
    return perm


def _f16_bits(u: np.ndarray) -> np.ndarray:
    return u.astype(np.uint16).view(np.float16)


def decode(idx, codebook: str) -> np.ndarray:
    """fp16 codebook value of every 16-bit index (any integer array; only the low 16 bits count)"""
    x = np.asarray(idx).astype(np.int64) & 0xFFFF
    x = x.astype(np.uint64)
    M = np.uint64(0xFFFFFFFF)
    if codebook in ("3inst", "mcg"):
        if codebook == "3inst":
            x = (x * np.uint64(89226354) + np.uint64(64248484)) & M
        else:
            x = (x * np.uint64(0xCBAC1FED)) & M
        x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
        lo = _f16_bits(x & np.uint64(0xFFFF)).astype(np.float64)
        hi = _f16_bits(x >> np.uint64(16)).astype(np.float64)
        return (lo + hi).astype(np.float16)  # exact in float64, one fp16 rounding (= __hadd)
    assert codebook == "mul1", codebook
    x = (x * np.uint64(0x83DCD12D)) & M
    bsum = sum((x >> np.uint64(s)) & np.uint64(0xFF) for s in (0, 8, 16, 24)).astype(np.int64)
    h = (1024 + bsum).astype(np.float64)  # fp16 bits 0x6400 + bytesum = 1024 + bytesum exactly
    k_inv = float(_f16_bits(np.array(0x1EEE)))
    k_bias = float(_f16_bits(np.array(0xC931)))
    return (h * k_inv + k_bias).astype(np.float16)


def ring_ends(K) -> np.ndarray:
    """(256,) S(i)"""
    return np.cumsum(step_widths(K))


def _ring_bits(packed: np.ndarray) -> np.ndarray:
    """(T, words) uint16/int16 packed tiles -> (T, 16 * words) ring bits, MSB first"""
    p = np.ascontiguousarray(packed).view(np.uint16).astype(np.uint32)
    T, W = p.shape
    u32 = p[:, 0::2] | (p[:, 1::2] << 16)
    be = u32.astype(">u4").view(np.uint8).reshape(T, W * 2)
    return np.unpackbits(be, axis=1)


def unpack(packed, K, chunk: int = 2048) -> np.ndarray:
    """(..., 16 K) packed tiles -> (..., 256) uint16 windows in ring order"""
    packed = np.asarray(packed)
    flat = packed.reshape(-1, packed.shape[-1])
    S = ring_ends(K)
    R = int(S[-1])
    assert flat.shape[1] * 16 == R
    pos = (S[:, None] - 16 + np.arange(16)[None, :]) % R  # (256, 16), MSB first
    weights = (1 << np.arange(15, -1, -1)).astype(np.uint32)
    out = np.empty((flat.shape[0], 256), dtype=np.uint16)
    for i in range(0, flat.shape[0], chunk):
        bits = _ring_bits(flat[i : i + chunk])
        out[i : i + chunk] = (bits[:, pos].astype(np.uint32) * weights).sum(axis=-1)
    return out.reshape(packed.shape[:-1] + (256,))


def random_packed(tiles: int, K, gen: np.random.Generator) -> np.ndarray:
    """(tiles, 16 K) uniformly random packed tiles (every bit pattern is a valid tile)"""
    return gen.integers(0, 1 << 16, size=(tiles, int(16 * K)), dtype=np.uint16).view(np.int16)


def rotated(packed, codebook):
    tk, tn, words = packed.shape
    vals = decode(unpack(packed.reshape(-1, words), words / 16), codebook)
    tiles = np.empty_like(vals)
    tiles[:, tile_perm()] = vals
    return tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def had128():
    h = np.ones((1, 1), dtype=np.float64)
    for _ in range(7):
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(128)


def effective(packed, suh, svh, codebook):
    w = rotated(packed, codebook).astype(np.float64)
    k, n = w.shape
    h = had128()
    w = np.einsum("ij,bjn->bin", h, w.reshape(-1, 128, n)).reshape(k, n)
    w = np.einsum("kbi,ij->kbj", w.reshape(k, -1, 128), h).reshape(k, n)
    return w * suh[:, None] * svh[None, :]
