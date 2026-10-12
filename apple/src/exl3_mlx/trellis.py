"""Decode the real EXL3 ring bitstream on Metal, without a CPU weight round trip."""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}
RATES = (*range(1, 9), 1.5, 2.5, 3.5)
HEADER = Path(__file__).with_name("decode.metal").read_text()


def validate_trellis(packed, codebook):
    if packed.ndim != 3 or packed.dtype not in (mx.int16, mx.uint16):
        raise ValueError("trellis must be int16/uint16 [in/16, out/16, 16*bpw]")
    if not all(packed.shape):
        raise ValueError("trellis dimensions must be nonzero")
    rate = packed.shape[-1] / 16
    if rate not in RATES:
        raise ValueError(f"Unsupported EXL3 bitrate {rate}; supported: {RATES}")
    if codebook not in CODEBOOKS:
        raise ValueError(f"Unknown EXL3 codebook {codebook!r}")
    if not rate.is_integer() and codebook != "mul1":
        raise ValueError("Half-integer bitrates require the mul1 codebook")
    return rate


@lru_cache(maxsize=1)
def _reconstruct_kernel():
    return mx.fast.metal_kernel(
        name="exl3_reconstruct",
        input_names=["packed"],
        output_names=["out"],
        header=HEADER,
        source=r"""
        uint idx = thread_position_in_grid.x;
        uint k = packed_shape[0] * 16;
        if (idx >= k * WIDTH) return;
        uint row = idx / WIDTH;
        uint col = idx % WIDTH + OFFSET;
        uint r = row % 16, c = col % 16;
        // Inverse tensor-core fragment permutation: row/column -> ring position.
        uint i = ((c % 8) * 4 + (r % 8) / 2) * 8
                 + (r % 2) + (r / 8) * 2 + (c / 8) * 4;
        uint end = (i + 1) * KA + (HALF_RATE ? (i + 1) / 2 : 0);
        uint ring_bits = WORDS * 16;
        uint start = (end + ring_bits - 16) % ring_bits;
        uint word = start / 32, shift = start % 32;
        uint base = ((row / 16) * packed_shape[1] + col / 16) * WORDS;
        uint a = exl3_u32(packed, base + word * 2);
        uint code;
        if (shift <= 16) {
            code = a >> (16 - shift);
        } else {
            uint b = exl3_u32(packed, base + ((word + 1) % (WORDS / 2)) * 2);
            code = (a << (shift - 16)) | (b >> (48 - shift));
        }
        out[idx] = exl3_codebook<CB>(code & 65535u);
        """,
    )


def reconstruct(packed, codebook="mul1", *, start=0, width=None, dtype=mx.float32):
    """Return rotated W_hat[in, width]; scales/Hadamards are applied to activations.

    Decoding always rounds to fp16, as in EXL3, then casts to the requested dtype.
    Column slices must align with the 16-column storage tiles.
    """
    rate = validate_trellis(packed, codebook)
    k, n = packed.shape[0] * 16, packed.shape[1] * 16
    width = n - start if width is None else width
    if (
        not isinstance(start, int)
        or not isinstance(width, int)
        or start < 0
        or width <= 0
        or start % 16
        or width % 16
        or start + width > n
    ):
        raise ValueError("Slice must be positive, in bounds, and aligned to 16 columns")
    if dtype not in (mx.float16, mx.bfloat16, mx.float32):
        raise ValueError("Reconstruction dtype must be float16, bfloat16, or float32")
    if not mx.metal.is_available():
        raise RuntimeError("EXL3-MLX requires an Apple silicon Metal device")
    return _reconstruct_kernel()(
        inputs=[packed],
        template=[
            ("CB", CODEBOOKS[codebook]),
            ("KA", int(rate)),
            ("HALF_RATE", not rate.is_integer()),
            ("WORDS", packed.shape[-1]),
            ("OFFSET", start),
            ("WIDTH", width),
        ],
        grid=(k * width, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(k, width)],
        output_dtypes=[dtype],
        stream=mx.gpu,
    )[0]
