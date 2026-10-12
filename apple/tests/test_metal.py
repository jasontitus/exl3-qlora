import mlx.core as mx
import numpy as np
import pytest
from exl3_mlx.reference import decode, random_packed, rotated
from exl3_mlx.trellis import CODEBOOKS, HEADER, reconstruct

CASES = [(r, cb) for r in range(1, 9) for cb in CODEBOOKS]
CASES += [(r, "mul1") for r in (1.5, 2.5, 3.5)]


@pytest.mark.parametrize("codebook", CODEBOOKS)
def test_entire_codebook_bit_exact(codebook):
    kernel = mx.fast.metal_kernel(
        name="test_exl3_codebook",
        input_names=["indices"],
        output_names=["out"],
        header=HEADER,
        source="uint i = thread_position_in_grid.x; out[i] = exl3_codebook<CB>(indices[i]);",
    )
    indices = mx.arange(65536, dtype=mx.uint32)
    actual = kernel(
        inputs=[indices],
        template=[("CB", CODEBOOKS[codebook])],
        grid=(65536, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(65536,)],
        output_dtypes=[mx.float16],
    )[0]
    expected = decode(np.arange(65536), codebook)
    np.testing.assert_array_equal(np.array(actual).view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("rate,codebook", CASES)
def test_real_layout_bit_exact_and_column_slice(rate, codebook):
    # Non-square tiles and a partial threadgroup catch stride/permutation/bounds errors.
    p = random_packed(3 * 19, rate, np.random.default_rng(42)).reshape(3, 19, -1)
    expected = rotated(p, codebook)
    actual = np.array(reconstruct(mx.array(p), codebook, dtype=mx.float16))
    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    sliced = np.array(reconstruct(mx.array(p), codebook, start=128, width=160))
    np.testing.assert_array_equal(sliced, expected[:, 128:288].astype(np.float32))


def test_noncontiguous_trellis():
    p = random_packed(12, 3, np.random.default_rng(9)).reshape(3, 4, -1)
    view = mx.array(p).transpose(1, 0, 2)
    np.testing.assert_array_equal(
        np.array(reconstruct(view)), rotated(p.transpose(1, 0, 2), "mul1")
    )


@pytest.mark.parametrize(
    "shape,dtype,cb",
    [
        ((8, 8, 44), mx.int16, "mul1"),
        ((8, 8, 72), mx.int16, "mul1"),
        ((8, 8, 40), mx.int16, "mcg"),
        ((8, 8, 48), mx.int32, "mul1"),
        ((8, 8, 48), mx.int16, "other"),
        ((0, 8, 48), mx.int16, "mul1"),
    ],
)
def test_invalid_formats(shape, dtype, cb):
    with pytest.raises(ValueError):
        reconstruct(mx.zeros(shape, dtype=dtype), cb)


@pytest.mark.parametrize("start,width", [(-16, 128), (1, 16), (0, 129), (128, 16), (0, 0)])
def test_invalid_slices(start, width):
    with pytest.raises(ValueError):
        reconstruct(mx.zeros((8, 8, 48), dtype=mx.int16), start=start, width=width)
