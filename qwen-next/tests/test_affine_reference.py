"""Hand-packed affine known answers; no MLX or tensor frameworks imported."""

import importlib.util
import math
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "affine_reference.py"
spec = importlib.util.spec_from_file_location("affine_reference", SCRIPT)
affine = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = affine
spec.loader.exec_module(affine)

# Fixed bytes derived by hand from the documented LSB-first layouts, not a
# packer implemented with the decoder's algorithm. All rows contain32 values.
# 3/5/6-bit fixtures cross byte and uint32 boundaries repeatedly.
KNOWN = {
    2: ("e4" * 8, [0, 1, 2, 3] * 8),
    3: ("88c6fa" * 4, list(range(8)) * 4),
    4: ("1032547698badcfe" * 2, list(range(16)) * 2),
    5: ("2088418af9" * 4, [0, 1, 2, 3, 4, 5, 6, 31] * 4),
    6: ("4000fe" * 8, [0, 1, 32, 63] * 8),
    8: ("00017f80ff2a40c8" * 4, [0, 1, 127, 128, 255, 42, 64, 200] * 4),
}


def fixture(bits=4, *, groups=1, scale=1.0, bias=0.0):
    raw = bytes.fromhex(KNOWN[bits][0]) * groups
    words = affine.uint32_rows_from_le_bytes(raw, rows=1, words_per_row=bits * groups)
    return affine.AffineReference(
        words,
        [[scale] * groups],
        [[bias] * groups],
        in_features=32 * groups,
        bits=bits,
        group_size=32,
    )


class AffineTests(unittest.TestCase):
    def test_hand_packed_known_answers_all_six_widths(self):
        for bits, (_, expected) in KNOWN.items():
            with self.subTest(bits=bits):
                self.assertEqual(fixture(bits).decode_rows(), [expected])
                self.assertEqual(
                    fixture(bits, scale=-0.25, bias=3.5).decode_rows(),
                    [[-0.25 * q + 3.5 for q in expected]],
                )

    def test_little_endian_words_and_cross_word_values(self):
        words = affine.uint32_rows_from_le_bytes(
            bytes.fromhex("12345678efcdab89"), rows=1, words_per_row=2
        )
        self.assertEqual(words, ((0x78563412, 0x89ABCDEF),))
        # At3bits columns10/21 cross32-bit words; at5bits6/12/19/25;
        # at6bits5/10/21/26. Expected values come only from KNOWN patterns.
        for bits in (3, 5, 6):
            result = fixture(bits).decode_rows()[0]
            cross = [i for i in range(32) if i * bits % 32 + bits > 32]
            self.assertTrue(cross)
            for column in cross:
                self.assertEqual(result[column], KNOWN[bits][1][column])

    def test_group_metadata_changes_at_exact_boundary_and_rows_do_not_alias(self):
        # Two4bit groups: allzero codes followed by all15 codes.
        raw = bytes(16) + bytes([255]) * 16
        packed = affine.uint32_rows_from_le_bytes(raw * 2, rows=2, words_per_row=8)
        ref = affine.AffineReference(
            packed,
            [[2, 0.25], [0, -2]],
            [[-3, 5], [7, 1]],
            in_features=64,
            bits=4,
            group_size=32,
        )
        self.assertEqual(
            ref.decode_rows(count=2), [[-3] * 32 + [8.75] * 32, [7] * 32 + [-29] * 32]
        )
        self.assertEqual(ref.decode_rows(start=1), [[7] * 32 + [-29] * 32])

    def test_supported_group64_and128_share_one_metadata_value(self):
        for group in (64, 128):
            raw = bytes.fromhex(KNOWN[8][0]) * (group // 32)
            packed = affine.uint32_rows_from_le_bytes(
                raw, rows=1, words_per_row=group // 4
            )
            ref = affine.AffineReference(
                packed, [[0.5]], [[-1]], in_features=group, bits=8, group_size=group
            )
            self.assertEqual(
                ref.decode_rows(), [[0.5 * q - 1 for q in KNOWN[8][1] * (group // 32)]]
            )

    def test_forward_and_transpose_against_independent_dense_answers_and_adjoint(self):
        for bits in KNOWN:
            with self.subTest(bits=bits):
                raw = bytes.fromhex(KNOWN[bits][0]) * 2
                packed = affine.uint32_rows_from_le_bytes(
                    raw, rows=2, words_per_row=bits
                )
                ref = affine.AffineReference(
                    packed,
                    [[0.5], [-0.25]],
                    [[-1], [2]],
                    in_features=32,
                    bits=bits,
                    group_size=32,
                )
                weights = [
                    [0.5 * q - 1 for q in KNOWN[bits][1]],
                    [-0.25 * q + 2 for q in KNOWN[bits][1]],
                ]
                x = [[((i % 7) - 3) / 4 for i in range(32)], [i / 8 for i in range(32)]]
                dy = [[0.5, -2], [-0.25, 3]]
                y = ref.forward(x)
                dx = ref.transpose(dy)
                self.assertEqual(
                    y,
                    [
                        [math.fsum(a * b for a, b in zip(v, w)) for w in weights]
                        for v in x
                    ],
                )
                self.assertEqual(
                    dx,
                    [
                        [v[0] * weights[0][i] + v[1] * weights[1][i] for i in range(32)]
                        for v in dy
                    ],
                )
                left = math.fsum(
                    a * b for row_y, row_dy in zip(y, dy) for a, b in zip(row_y, row_dy)
                )
                right = math.fsum(
                    a * b for row_x, row_dx in zip(x, dx) for a, b in zip(row_x, row_dx)
                )
                self.assertAlmostEqual(left, right, places=12)

    def test_linear_output_bias_distinct_from_affine_group_bias(self):
        ref = fixture(4, scale=0, bias=2)
        self.assertEqual(ref.forward([[1] * 32]), [[64]])
        self.assertEqual(ref.forward([[1] * 32], linear_bias=[7]), [[71]])
        self.assertEqual(ref.transpose([[3]]), [[6] * 32])
        with self.assertRaises(TypeError):
            ref.transpose([[3]], linear_bias=[7])

    def test_copies_input_and_freezes_geometry(self):
        packed, scales, biases = [[0] * 4], [[1]], [[2]]
        ref = affine.AffineReference(
            packed, scales, biases, in_features=32, bits=4, group_size=32
        )
        packed[0][0] = 0xFFFFFFFF
        scales[0][0] = 99
        biases[0][0] = -10
        self.assertEqual(ref.decode_rows(), [[2] * 32])
        with self.assertRaises(FrozenInstanceError):
            ref.bits = 3

    def test_row_bounds_and_bounded_allocation(self):
        ref = fixture()
        self.assertEqual(ref.decode_rows(start=1, count=0), [])
        for args in (
            {"start": -1},
            {"start": 2, "count": 0},
            {"count": 2},
            {"start": True},
            {"count": 1.0},
            {"max_values": 31},
        ):
            with self.subTest(args=args), self.assertRaises(affine.AffineError):
                ref.decode_rows(**args)
        with self.assertRaisesRegex(affine.AffineError, "multiply-add"):
            ref.forward([[1] * 32], max_multiply_adds=31)
        with self.assertRaisesRegex(affine.AffineError, "decoded-value"):
            ref.transpose([[1]], max_values=31)

    def test_geometry_shape_and_unsigned_word_validation(self):
        base = {"in_features": 32, "bits": 4, "group_size": 32}
        for change in (
            {"bits": 1},
            {"bits": True},
            {"bits": 4.0},
            {"group_size": 16},
            {"in_features": 31},
            {"in_features": 0},
            {"max_packed_words": 3},
        ):
            with self.subTest(change=change), self.assertRaises(affine.AffineError):
                affine.AffineReference([[0] * 4], [[1]], [[0]], **(base | change))
        for packed in (
            [],
            [0] * 4,
            [[0] * 3],
            [[0] * 5],
            [[-1] * 4],
            [[2**32] * 4],
            [[True] * 4],
            [[1.0] * 4],
        ):
            with self.subTest(packed=packed), self.assertRaises(affine.AffineError):
                affine.AffineReference(packed, [[1]], [[0]], **base)

    def test_metadata_not_broadcast_and_all_values_finite(self):
        base = {"in_features": 32, "bits": 4, "group_size": 32}
        for bad in (
            None,
            [1],
            [],
            [[1, 2]],
            [[float("nan")]],
            [[float("inf")]],
            [[True]],
            [["1"]],
        ):
            with self.subTest(bad=bad):
                for scales, biases in ((bad, [[0]]), ([[1]], bad)):
                    with self.assertRaises(affine.AffineError):
                        affine.AffineReference([[0] * 4], scales, biases, **base)

    def test_inputs_and_output_bias_validation(self):
        ref = fixture()
        for bad in ([], [1] * 32, [[1] * 31], [[True] * 32], [[float("nan")] * 32]):
            with self.subTest(bad=bad), self.assertRaises(affine.AffineError):
                ref.forward(bad)
        for bias in ([1, 2], [float("inf")], [True]):
            with self.subTest(bias=bias), self.assertRaises(affine.AffineError):
                ref.forward([[0] * 32], linear_bias=bias)

    def test_byte_decoder_rejects_size_type_and_bounds(self):
        for payload, kwargs in (
            (b"abc", {}),
            (b"abcdefgh", {}),
            ("abcd", {}),
            (b"abcd", {"rows": True}),
            (b"abcd", {"words_per_row": 0}),
            (b"abcdefgh", {"words_per_row": 2, "max_words": 1}),
        ):
            with (
                self.subTest(payload=payload, kwargs=kwargs),
                self.assertRaises(affine.AffineError),
            ):
                affine.uint32_rows_from_le_bytes(
                    payload, **({"rows": 1, "words_per_row": 1} | kwargs)
                )

    def test_overflow_fails_closed(self):
        with self.assertRaisesRegex(affine.AffineError, "overflow"):
            fixture(scale=1e308).decode_rows()
        with self.assertRaisesRegex(affine.AffineError, "overflow"):
            fixture(scale=1).forward([[1e308] * 32])
        with self.assertRaisesRegex(affine.AffineError, "overflow"):
            fixture(scale=1).transpose([[1e308]])

    def test_provenance_is_fixed_and_missing_or_changed_sources_rejected(self):
        report = affine.source_provenance()
        self.assertEqual(report["mlx_distribution_version"], "0.32.3")
        self.assertEqual(len(report["source_sha256"]), 3)
        report["source_sha256"].clear()
        self.assertEqual(len(affine.source_provenance()["source_sha256"]), 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(affine.AffineError, "Missing regular"):
                affine.verify_spec_sources(root)
            path = root / "core/__init__.pyi"
            path.parent.mkdir()
            path.write_text("changed")
            with self.assertRaisesRegex(affine.AffineError, "hash changed"):
                affine.verify_spec_sources(root)


if __name__ == "__main__":
    unittest.main()
