"""CPU-only oracle checks; importing the diagnostic must not initialize MLX."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_native_vjp.py"
SPEC = importlib.util.spec_from_file_location("native_vjp_diagnostic", SCRIPT)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def packed_fixture():
    codes = np.tile(np.array([0x38, 0xC0, 0x30, 0x80], np.uint32), (2, 16))
    packed = np.sum(codes.reshape(2, -1, 4) << (np.arange(4, dtype=np.uint32) * 8), axis=-1).astype(
        np.uint32
    )
    scales = np.array([[127, 126], [128, 125]], np.uint8)
    values = np.tile(np.array([1, -2, 0.5, -0.0]), (2, 16))
    values *= np.repeat(np.array([[1, 0.5], [2, 0.25]]), 32, axis=1)
    return packed, scales, values


def test_bf16_rounding_midpoints_and_sign():
    x = np.array([1, 1 + 2**-8, 1 + 3 * 2**-8, -1 - 2**-8, -1 - 3 * 2**-8], np.float32)
    np.testing.assert_array_equal(D.bf16(x), [1, 1, 1 + 2**-6, -1, -1 - 2**-6])


def test_mxfp8_independent_byte_order_and_scale():
    packed, scales, values = packed_fixture()
    np.testing.assert_array_equal(D.decode_mxfp8(packed, scales), values)
    table = D.e4m3_table()
    assert table[1] == 2**-9 and table[0x7E] == 448 and table[0xFE] == -448
    assert np.isnan(table[0x7F]) and np.isnan(table[0xFF])
    scales[0, 0] = 255
    with pytest.raises(ValueError, match="Nonfinite"):
        D.decode_mxfp8(packed, scales)


def test_numpy_activation_quantizer_midpoint_rounding():
    x = np.zeros((1, 32), np.float32)
    x[0, :7] = [448, 1.0625, 1.1875, -1.0625, -1.1875, 0.06640625, 2**-10]
    expected = np.zeros_like(x)
    expected[0, :7] = [448, 1, 1.25, -1, -1.25, 0.0625, 0]
    np.testing.assert_array_equal(D.quantize_fp8_reference(x), expected)


def test_streamed_numpy_archive_rows_payload_hash_and_mutation(tmp_path):
    packed, scales, values = packed_fixture()
    path = tmp_path / "weights.safetensors"
    save_file({"projection.weight": packed, "projection.scales": scales}, path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"projection.weight": path.name, "projection.scales": path.name}})
    )
    archive = D.NumpyArchive(tmp_path)
    got = list(archive.blocks("projection", block_rows=1, rows=(1, 2), hash_payload=True))
    assert len(got) == 1 and got[0][0] == 0
    np.testing.assert_array_equal(got[0][1], values[1:])
    assert archive.payload_hashes["projection"]["rows"] == [1, 2]
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed"):
        list(archive.blocks("projection", block_rows=1))


def test_cpu_ce_oracle_gradient_and_direction_metric():
    logits = np.random.default_rng(81).normal(size=(3, 7))
    targets = np.array([1, 4, 2])
    value, gradient = D.ce_reference(logits, targets)
    for coordinate in ((0, 1), (2, 6)):
        plus, minus = logits.copy(), logits.copy()
        plus[coordinate] += 1e-5
        minus[coordinate] -= 1e-5
        finite_difference = (
            D.ce_reference(plus, targets)[0] - D.ce_reference(minus, targets)[0]
        ) / 2e-5
        assert gradient[coordinate] == pytest.approx(finite_difference, abs=1e-10)
    assert D.ce_reference(logits + 1000, targets)[0] == pytest.approx(value, abs=1e-12)
    assert D.metrics(gradient, gradient)["passed"]
    assert not D.metrics(-gradient, gradient)["passed"]


def test_default_and_full_prefill_oracle_row_selection():
    np.testing.assert_array_equal(D.oracle_row_indices(4, 8), np.arange(4))
    np.testing.assert_array_equal(D.oracle_row_indices(5808, 4), [0, 15, 16, 5807])
    np.testing.assert_array_equal(
        D.oracle_row_indices(5808, 8), [0, 1, 15, 16, 127, 128, 5806, 5807]
    )
    for length in (1, 3, 17, 128, 5808, 8192):
        for requested in (1, 2, 4, 8, 32, 128):
            sample = D.oracle_row_indices(length, requested)
            assert len(sample) == min(length, requested)
            assert sample.dtype == np.int32
            assert np.all((sample >= 0) & (sample < length))
            assert len(np.unique(sample)) == len(sample)


def test_sampled_row_vjp_matches_full_matrix_row_semantics():
    rng = np.random.default_rng(901)
    x, weights, dy = (
        rng.normal(size=(5808, 7)),
        rng.normal(size=(11, 7)),
        rng.normal(size=(5808, 11)),
    )
    sample = D.oracle_row_indices(len(x), 8)
    np.testing.assert_allclose((x @ weights.T)[sample], x[sample] @ weights.T, atol=1e-14)
    np.testing.assert_allclose((dy @ weights)[sample], dy[sample] @ weights, atol=1e-14)
