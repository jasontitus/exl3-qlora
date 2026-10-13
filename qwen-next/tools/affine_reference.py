"""Bounded CPU mathematical reference for MLX-style affine-packed matrices.

No MLX/Torch/NumPy import, network, tensor-file loader or GPU execution. Values
are Python binary64 numbers; this does not reproduce BF16/F16 rounding, fused
Metal arithmetic or device reduction order. It is not an EXL3 trellis decoder.
"""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)
SUPPORTED_GROUP_SIZES = (32, 64, 128)
MAX_PACKED_WORDS = 1 << 20
MAX_RESULT_VALUES = 1 << 20
MAX_MULTIPLY_ADDS = 1 << 22
SPEC = {
    "mlx_distribution_version": "0.32.3",
    "inspected_installed_root": "/Users/jasontitus/experiments/exl3-qlora/apple/.venv/lib/python3.12/site-packages/mlx",
    "source_sha256": {
        "core/__init__.pyi": "bfa3c09f0f1e3edb925bf92e021f3121f0b953959f43ffd91cfa13895a41e8ce",
        "include/mlx/backend/metal/kernels/quantized.h": "8e2ca65d8d8dafe67b97120d1127d54ce57032f621bdba161e54a0bc4a948b8c",
        "include/mlx/backend/common/quantized.h": "e5bfbb9a3d2459b3dab6e50c90c1d7a22b55ac84e372e1dc22ef9429f872d4a7",
    },
    "anchors": {
        "core/__init__.pyi": "quantized_matmul, quantize and dequantize (lines 3401-3558)",
        "include/mlx/backend/metal/kernels/quantized.h": "get_pack_factor/get_bytes_per_pack (17-26), dequantize (486-575)",
        "include/mlx/backend/common/quantized.h": "get_pack_factor/get_bytes_per_pack",
    },
    "packing": "Unsigned uint32 words serialized little endian; consecutive unsigned q values occupy bits from least to most significant without padding between words. 3-bit packs 8 values/3 bytes, 5-bit 8/5, 6-bit 4/3.",
    "geometry": "Two-dimensional W[out,in], in divisible by group_size, complete uint32 rows; affine metadata[out,in/group_size]. No broadcast, padding or partial group.",
    "equation": "W[o,i] = scales[o,i//group_size] * q[o,i] + group_biases[o,i//group_size]",
    "scope": "Source-inspected CPU mathematical reference only; no MLX numerical parity or hardware/training capability established.",
}


class AffineError(ValueError):
    pass


def _require(value, message):
    if not value:
        raise AffineError(message)


def _integer(value, label, *, zero=False):
    _require(
        type(value) is int and value >= (0 if zero else 1),
        f"{label} must be an integer {'>=0' if zero else '>0'}",
    )
    return value


def _number(value, label):
    _require(type(value) in (int, float), f"{label} must be a real Python number")
    try:
        result = float(value)
    except OverflowError as error:
        raise AffineError(f"{label} is not a finite binary64 value") from error
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _matrix(value, rows, columns, label, convert):
    _require(
        isinstance(value, (list, tuple)) and len(value) == rows,
        f"{label}: wrong row count",
    )
    result = []
    for row in value:
        _require(
            isinstance(row, (list, tuple)) and len(row) == columns,
            f"{label}: ragged/wrong column count",
        )
        result.append(tuple(convert(x, label) for x in row))
    return tuple(result)


def _uint32(value, label):
    _require(
        type(value) is int and 0 <= value < 2**32,
        f"{label}: packed words must be unsigned uint32 integers",
    )
    return value


def uint32_rows_from_le_bytes(
    payload, *, rows, words_per_row, max_words=MAX_PACKED_WORDS
):
    """Convert caller-supplied bounded bytes, explicitly independent of host endian."""
    _integer(rows, "rows")
    _integer(words_per_row, "words_per_row")
    _integer(max_words, "max_words")
    _require(
        isinstance(payload, (bytes, bytearray)), "payload must be bytes or bytearray"
    )
    count = rows * words_per_row
    _require(count <= max_words, "Packed-word limit exceeded")
    _require(
        len(payload) == count * 4,
        "Packed byte length disagrees with declared uint32 shape",
    )
    return tuple(
        tuple(
            struct.unpack_from("<I", payload, 4 * (r * words_per_row + c))[0]
            for c in range(words_per_row)
        )
        for r in range(rows)
    )


def source_provenance():
    """Return the fixed source-inspection record, not a runtime-parity result."""
    return copy.deepcopy(SPEC)


def verify_spec_sources(mlx_root):
    """Optional CPU-only rehash of the three small installed source files."""
    checked = {}
    for relative, expected in SPEC["source_sha256"].items():
        path = Path(mlx_root) / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"Missing regular source: {relative}",
        )
        _require(
            path.stat().st_size <= 4 << 20, f"Source exceeds bounded size: {relative}"
        )
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == expected, f"Inspected MLX source hash changed: {relative}")
        checked[relative] = actual
    return checked


@dataclass(frozen=True)
class AffineReference:
    """Immutable copied affine data with row-bounded decode and tiny dense math.

    `group_biases` belongs to quantization. An optional `linear_bias` is added
    once per output by forward(); it is never added by transpose(). Scales may
    be negative or zero; no quantizer/rounding behavior is inferred from them.
    """

    _packed: tuple
    _scales: tuple
    _group_biases: tuple
    in_features: int
    out_features: int
    bits: int
    group_size: int

    def __init__(
        self,
        packed,
        scales,
        group_biases,
        *,
        in_features,
        bits,
        group_size,
        max_packed_words=MAX_PACKED_WORDS,
    ):
        _integer(in_features, "in_features")
        _require(
            type(bits) is int and bits in SUPPORTED_BITS, "Unsupported affine bits"
        )
        _require(
            type(group_size) is int and group_size in SUPPORTED_GROUP_SIZES,
            "Unsupported affine group_size",
        )
        _integer(max_packed_words, "max_packed_words")
        _require(
            in_features % group_size == 0,
            "Input columns must contain complete affine groups",
        )
        _require(
            in_features * bits % 32 == 0,
            "Input columns must contain complete uint32 rows",
        )
        _require(
            isinstance(packed, (list, tuple)) and len(packed) > 0,
            "packed must be a nonempty 2D sequence",
        )
        out_features = len(packed)
        words = in_features * bits // 32
        _require(out_features * words <= max_packed_words, "Packed-word limit exceeded")
        groups = in_features // group_size
        fields = {
            "_packed": _matrix(packed, out_features, words, "packed", _uint32),
            "_scales": _matrix(scales, out_features, groups, "scales", _number),
            "_group_biases": _matrix(
                group_biases, out_features, groups, "group_biases", _number
            ),
            "in_features": in_features,
            "out_features": out_features,
            "bits": bits,
            "group_size": group_size,
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def _decode_row(self, row):
        words = self._packed[row]
        mask = (1 << self.bits) - 1
        values = []
        for column in range(self.in_features):
            word, shift = divmod(column * self.bits, 32)
            code = words[word] >> shift
            if shift + self.bits > 32:
                code |= words[word + 1] << (32 - shift)
            code &= mask
            group = column // self.group_size
            value = self._scales[row][group] * code + self._group_biases[row][group]
            _require(math.isfinite(value), "Affine decode overflowed binary64")
            values.append(value)
        return values

    def decode_rows(self, start=0, count=1, *, max_values=MAX_RESULT_VALUES):
        _integer(start, "start", zero=True)
        _integer(count, "count", zero=True)
        _integer(max_values, "max_values")
        _require(
            start <= self.out_features and count <= self.out_features - start,
            "Decode row bounds exceeded",
        )
        _require(count * self.in_features <= max_values, "Decoded-value limit exceeded")
        return [self._decode_row(row) for row in range(start, start + count)]

    def _inputs(self, values, columns, outputs, max_values, max_multiply_adds):
        _integer(max_values, "max_values")
        _integer(max_multiply_adds, "max_multiply_adds")
        _require(
            isinstance(values, (list, tuple)) and len(values) > 0,
            "Input must be a nonempty 2D sequence",
        )
        _require(
            max(self.in_features, len(values) * columns, len(values) * outputs)
            <= max_values,
            "Input/output/row decoded-value limit exceeded",
        )
        _require(
            len(values) * self.in_features * self.out_features <= max_multiply_adds,
            "Reference multiply-add limit exceeded",
        )
        return _matrix(values, len(values), columns, "input", _number)

    def forward(
        self,
        x,
        *,
        linear_bias=None,
        max_values=MAX_RESULT_VALUES,
        max_multiply_adds=MAX_MULTIPLY_ADDS,
    ):
        """Return x @ W.T plus optional independent output bias; stream W rows."""
        values = self._inputs(
            x, self.in_features, self.out_features, max_values, max_multiply_adds
        )
        if linear_bias is None:
            bias = (0.0,) * self.out_features
        else:
            _require(
                isinstance(linear_bias, (list, tuple))
                and len(linear_bias) == self.out_features,
                "linear_bias must have one value per output",
            )
            bias = tuple(_number(v, "linear_bias") for v in linear_bias)
        result = [[] for _ in values]
        for row in range(self.out_features):
            weights = self._decode_row(row)
            for batch, vector in enumerate(values):
                try:
                    value = (
                        math.fsum(a * b for a, b in zip(vector, weights)) + bias[row]
                    )
                except (OverflowError, ValueError) as error:
                    raise AffineError("Forward overflowed binary64") from error
                _require(math.isfinite(value), "Forward overflowed binary64")
                result[batch].append(value)
        return result

    def transpose(
        self, dy, *, max_values=MAX_RESULT_VALUES, max_multiply_adds=MAX_MULTIPLY_ADDS
    ):
        """Return dy @ W, the input VJP of forward. No linear output bias."""
        values = self._inputs(
            dy, self.out_features, self.in_features, max_values, max_multiply_adds
        )
        result = [[0.0] * self.in_features for _ in values]
        for row in range(self.out_features):
            weights = self._decode_row(row)
            for batch, vector in enumerate(values):
                for column, weight in enumerate(weights):
                    result[batch][column] += vector[row] * weight
                    _require(
                        math.isfinite(result[batch][column]),
                        "Transpose overflowed binary64",
                    )
        return result
