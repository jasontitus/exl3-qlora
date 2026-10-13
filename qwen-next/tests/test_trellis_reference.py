"""CPU-only decoder known answers and independently indexed ring checks."""

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import random
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/trellis_reference.py"
spec = importlib.util.spec_from_file_location("trellis_reference", SCRIPT)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
FIXTURES = ROOT / "sources/src/exl3/fixtures"
KNOWN = {
    "exl3_k2p5_mcg_w12_linear.safetensors": "870df4165f311c7651655fbe81feec85b2ab0c6c51b1c754d023423bc6b3e7ad",
    "exl3_k2p5_mcg_linear.safetensors": "5a651adc9cb67f065d2786b548b3a0a010ae482b80c4e269a6701f745840be08",
    "exl3_k3_mcg_linear.safetensors": "bda534c4387dbe7d9342ca5ebdb6460b40d831397aca89422f21e274460ca987",
}


def bit_by_bit_ring(words, n):
    bits = []
    for low, high in zip(words[::2], words[1::2]):
        word = low + (high << 16)
        bits.extend((word >> pos) & 1 for pos in range(31, -1, -1))
    decoded = []
    for position in range(256):
        end = (position + 1) * n // 16
        codeword = 0
        for bit in range(end - 16, end):
            codeword = 2 * codeword + bits[bit % len(bits)]
        decoded.append(codeword)
    return decoded


class TrellisTests(unittest.TestCase):
    def test_three_real_converter_fixtures_inner_and_public_are_bit_exact(self):
        for name, sha in KNOWN.items():
            with self.subTest(fixture=name):
                self.assertEqual(
                    hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(), sha
                )
                report = reference.verify_fixture(FIXTURES / name, public=True)
                self.assertEqual(report["inner"]["mismatches"], 0)
                self.assertEqual(report["public_matrix"]["mismatches"], 0)
                self.assertGreater(report["wrong_window_control"]["mismatches"], 0)
                self.assertFalse(report["production_metal_vjp_qualified"])
                self.assertFalse(report["serving_projection_qualified"])
                self.assertFalse(
                    report["target_qwen_window14_converter_fixture_qualified"]
                )

    def test_every_even_rate_matches_independent_bit_reader(self):
        rng = random.Random(1957)
        for n in range(16, 129, 2):
            with self.subTest(n=n):
                words = [rng.randrange(65536) for _ in range(n)]
                self.assertEqual(
                    reference.unpack_tile(words, n), bit_by_bit_ring(words, n)
                )

    def test_ring_wrap_uses_last_word_before_first_word(self):
        words = [0] * 40
        words[-2], words[-1] = 0xABCD, 0x1357
        words[0], words[1] = 0x0246, 0xC000
        first = reference.unpack_tile(words, 40)[0]
        # First codeword takes the tail14 bits and the head2 bits.
        self.assertEqual(first, ((0x1357ABCD & ((1 << 14) - 1)) << 2) | 3)
        self.assertEqual(reference.unpack_tile(words, 40)[-1], 0xABCD)

    def test_permutation_is_bijective_and_has_known_layout(self):
        p = reference.tensor_core_permutation()
        self.assertEqual(sorted(p), list(range(256)))
        self.assertEqual(p[:8], [0, 16, 128, 144, 8, 24, 136, 152])
        self.assertEqual(p[-8:], [103, 119, 231, 247, 111, 127, 239, 255])

    def test_window_mask_all_supported_widths(self):
        words = [(k * 4567 + 23123) % 65536 for k in range(40)]
        raw = bit_by_bit_ring(words, 40)
        permutation = reference.tensor_core_permutation()
        for window in range(8, 17):
            actual = reference.decode_tile(words, 40, window=window)
            for i, codeword in enumerate(raw):
                self.assertEqual(
                    actual[permutation[i]],
                    reference.decode_mcg(codeword & ((1 << window) - 1)),
                )

    def test_all_mcg_codewords_decode_finite_with_known_zero(self):
        self.assertEqual(reference.decode_mcg(0), 0x3F60)
        self.assertTrue(
            all(
                math.isfinite(reference.half_value(reference.decode_mcg(c)))
                for c in range(65536)
            )
        )

    def test_non_square_tile_matrix_placement(self):
        tiles = [
            [0] * 16,
            [65535] * 16,
            [0x1234] * 16,
            [0x5678] * 16,
            [0xABCD] * 16,
            [0xFF00] * 16,
        ]
        matrix = reference.reconstruct_inner(
            [word for tile in tiles for word in tile], 32, 48, 16, window=14
        )
        for tk in range(2):
            for tn in range(3):
                tile = reference.decode_tile(tiles[tk * 3 + tn], 16, window=14)
                for r in range(16):
                    self.assertEqual(
                        matrix[
                            (tk * 16 + r) * 48 + tn * 16 : (tk * 16 + r) * 48
                            + tn * 16
                            + 16
                        ],
                        tile[r * 16 : r * 16 + 16],
                    )

    def test_rejects_invalid_rate_window_and_words(self):
        for n in (True, 15, 17, 130, 40.0):
            with self.subTest(n=n), self.assertRaises(ValueError):
                reference.unpack_tile([0] * 40, n)
        for window in (True, 7, 17, 12.0):
            with self.subTest(window=window), self.assertRaises(ValueError):
                reference.decode_tile([0] * 40, 40, window=window)
        for words in ([0] * 39, [0] * 41, [False] * 40, [-1] * 40, [65536] * 40):
            with self.assertRaises(ValueError):
                reference.decode_tile(words, 40, window=12)
        for code in (-1, 65536, True, 1.0):
            with self.assertRaises(ValueError):
                reference.decode_mcg(code)

    def test_rejects_invalid_geometry_and_public_scales(self):
        for k, n in ((0, 128), (17, 128), (True, 128), (1 << 20, 128)):
            with self.subTest(shape=(k, n)), self.assertRaises(ValueError):
                reference.reconstruct_inner([], k, n, 40, window=12)
        with self.assertRaises(ValueError):
            reference.reconstruct_public([0] * 256, [0] * 16, [0] * 16, 16, 16)
        for scales in ([0] * 127, [0x7C00] * 128, [0x7E00] * 128):
            with self.assertRaises(ValueError):
                reference.reconstruct_public([0] * 16384, scales, [0] * 128, 128, 128)

    def test_frozen_public_matrix_linear_adjoint_and_directional_derivative(self):
        fixture = reference.read_fixture(
            FIXTURES / "exl3_k2p5_mcg_w12_linear.safetensors"
        )
        # Converter-produced public matrix: independent of our reconstruction.
        weights = fixture["tensors"]["public"]["bits"]
        x = [((i * 7) % 19 - 9) / 32 for i in range(128)]
        dy = [((i * 3) % 23 - 11) / 32 for i in range(128)]
        y = reference.frozen_matmul(x, weights, 128, 128)
        dx = reference.frozen_matmul(dy, weights, 128, 128, transpose=True)
        left, right = (
            math.fsum(a * b for a, b in zip(y, dy)),
            math.fsum(a * b for a, b in zip(x, dx)),
        )
        self.assertAlmostEqual(left, right, places=12)
        direction = [((i * 11) % 17 - 8) / 64 for i in range(128)]
        epsilon = 1e-4
        plus = reference.frozen_matmul(
            [a + epsilon * b for a, b in zip(x, direction)], weights, 128, 128
        )
        minus = reference.frozen_matmul(
            [a - epsilon * b for a, b in zip(x, direction)], weights, 128, 128
        )
        numerical = math.fsum((a - b) * g for a, b, g in zip(plus, minus, dy)) / (
            2 * epsilon
        )
        analytical = math.fsum(a * b for a, b in zip(direction, dx))
        self.assertAlmostEqual(numerical, analytical, places=9)

    def test_frozen_matmul_rejects_invalid_inputs(self):
        for values in ([0] * 15, [float("nan")] * 16, [True] * 16):
            with self.assertRaises(ValueError):
                reference.frozen_matmul(values, [0] * 256, 16, 16)

    def rewrite_fixture(self, path, *, change_header=None, change_payload=None):
        raw = (FIXTURES / "exl3_k2p5_mcg_w12_linear.safetensors").read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + size])
        payload = bytearray(raw[8 + size :])
        if change_header:
            change_header(header)
        if change_payload:
            change_payload(payload)
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    def test_fixture_wrong_metadata_or_shape_is_rejected(self):
        changes = [
            lambda h: h["__metadata__"].update(codebook="mul1"),
            lambda h: h["__metadata__"].update(k="3"),
            lambda h: h["__metadata__"].update(window="17"),
            lambda h: h["__metadata__"].pop("k"),
            lambda h: h["inner"].update(shape=[64, 256]),
            lambda h: h["trellis"].update(dtype="F16"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fixture.safetensors"
            for change in changes:
                self.rewrite_fixture(p, change_header=change)
                with self.assertRaises(ValueError):
                    reference.read_fixture(p)

    def test_fixture_wrong_valid_window_fails_known_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fixture.safetensors"
            self.rewrite_fixture(
                p, change_header=lambda h: h["__metadata__"].update(window="14")
            )
            with self.assertRaisesRegex(ValueError, "Inner known-answer mismatch"):
                reference.verify_fixture(p)

    def test_fixture_corrupted_expected_bits_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "fixture.safetensors"
            self.rewrite_fixture(p, change_payload=lambda b: b.__setitem__(0, b[0] ^ 1))
            with self.assertRaisesRegex(ValueError, "Inner known-answer mismatch"):
                reference.verify_fixture(p)

    def test_fixture_payload_limit_precedes_payload_read(self):
        with (
            mock.patch.object(reference, "MAX_FIXTURE_BYTES", 64),
            self.assertRaisesRegex(ValueError, "byte limit"),
        ):
            reference.read_fixture(FIXTURES / "exl3_k2p5_mcg_w12_linear.safetensors")

    def test_cli_nonoverwrite_and_failure_exit(self):
        fixture = FIXTURES / "exl3_k2p5_mcg_w12_linear.safetensors"
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            output = Path(tmp) / "report.json"
            self.assertEqual(reference.main([str(fixture), "--output", str(output)]), 0)
            before = output.read_bytes()
            self.assertEqual(reference.main([str(fixture), "--output", str(output)]), 2)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(reference.main([str(Path(tmp) / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
