"""Synthetic complete banks only: verified bytes -> tiny CPU public matrix."""

import copy
import hashlib
import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "expert_decode_reference", ROOT / "tools/expert_decode_reference.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def half(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


class ExpertDecodeReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.name = "synthetic.safetensors"

    def fixture(
        self,
        *,
        hidden=128,
        intermediate=128,
        projection="gate_proj",
        window=14,
        n=48,
        word_values=(0, 65535),
        suh=None,
        svh=None,
    ):
        self.prefix = "language_model.model.layers.1.mlp.switch_mlp." + projection
        inputs, outputs = (
            (intermediate, hidden)
            if projection == "down_proj"
            else (hidden, intermediate)
        )
        count = len(word_values)
        config = {
            "model_type": "qwen4_exp",
            "text_config": {
                "num_hidden_layers": 2,
                "num_experts": count,
                "hidden_size": hidden,
                "moe_intermediate_size": intermediate,
            },
            "expert_quant": {
                "format": "exl3",
                "codebook": "mcg",
                "window": window,
                "k": max(3, n / 16),
            },
        }
        self.header = {
            "__metadata__": {
                "format": "exl3",
                "codebook": "mcg",
                "window": str(window),
                "k": str(n / 16),
            }
        }
        self.raw = {}
        payload = bytearray()
        # Noncanonical bank order must be interpreted from the authenticated views.
        for kind, shape, dtype in (
            ("svh", [count, outputs], "F16"),
            ("trellis", [count, inputs // 16, outputs // 16, n], "U16"),
            ("suh", [count, inputs], "F16"),
        ):
            start = len(payload)
            length = math.prod(shape[1:])
            for expert, word in enumerate(word_values):
                if kind == "trellis":
                    bits = (
                        [word(i) for i in range(length)]
                        if callable(word)
                        else [word] * length
                    )
                else:
                    bits = (suh if kind == "suh" else svh) or [half(1)] * length
                self.assertEqual(len(bits), length)
                raw = struct.pack(f"<{length}H", *bits)
                self.raw[expert, kind] = raw
                payload.extend(raw)
            self.header[self.prefix + "." + kind] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, len(payload)],
            }
        encoded = json.dumps(self.header).encode()
        encoded += b" " * (-len(encoded) % 8)
        (self.root / self.name).write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + payload
        )
        (self.root / "config.json").write_text(json.dumps(config))
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        self.prefix + "." + kind: self.name for kind in m.rows.KINDS
                    }
                }
            )
        )
        return m.rows.bind_projection(
            self.root, shard_name=self.name, prefix=self.prefix
        )

    def decode(self, identity=None, expert=0, **limits):
        identity = self.fixture() if identity is None else identity
        return m.decode_expert(
            self.root,
            identity,
            expected_sha256=m.rows.identity_sha256(identity),
            expert=expert,
            **limits,
        )

    def assert_constant_oracle(self, decoded, codebook_value, left, right):
        # Independent dense-Hadamard identity H*constant*H: only first row/col
        # of each128 block remain. An absolute1e-4 bound permits source-ordered
        # FP32 cancellation residuals at mathematical zeros, not FP16 serving parity.
        expected = []
        for row in range(decoded.in_features):
            for col in range(decoded.out_features):
                expected.append(
                    128 * codebook_value * left[row] * right[col]
                    if row % 128 == 0 and col % 128 == 0
                    else 0.0
                )
        actual = [value(b) for b in decoded.public_bits]
        self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1e-4)

    def test_selected_rows_known_codewords_window14_and_receipt(self):
        identity = self.fixture()
        # Hand-known MCG outputs: codeword0 ->0x3f60; codeword0x3fff ->0x3d85.
        for expert, constant in ((0, 1.84375), (1, 1.3798828125)):
            with self.subTest(expert=expert):
                decoded = self.decode(identity, expert)
                self.assert_constant_oracle(decoded, constant, [1.0] * 128, [1.0] * 128)
                receipt = decoded.receipt()
                self.assertEqual(receipt["matrix_shape"], [128, 128])
                self.assertEqual(receipt["window"], 14)
                self.assertEqual(receipt["bits_per_weight"], 3)
                self.assertEqual(receipt["bounds"]["raw_payload_bytes"], 6656)
                self.assertEqual(
                    receipt["public_fp16_bits_sha256"],
                    hashlib.sha256(
                        struct.pack("<16384H", *decoded.public_bits)
                    ).hexdigest(),
                )
                self.assertTrue(
                    receipt["read_receipt"]["selected_payload_checksums_verified"]
                )
                for key in (
                    "serving_projection_qualified",
                    "target_qwen_window14_converter_fixture_qualified",
                    "gpu_parity_qualified",
                    "full_architecture_qualified",
                    "training_supported",
                ):
                    self.assertFalse(receipt[key])
                self.assertIsNone(receipt["bounds"]["measured_process_peak_bytes"])
                receipt["matrix_shape"][0] = 999
                self.assertEqual(decoded.receipt()["matrix_shape"], [128, 128])
                with self.assertRaises(FrozenInstanceError):
                    decoded.public_bits = ()

    def test_window_and_actual_fractional_rate_not_config_ceiling(self):
        identity = self.fixture(window=12, n=40, word_values=(65535,))
        decoded = self.decode(identity)
        # codeword0x0fff ->0xc076, different from the window14 all-ones fixture.
        self.assert_constant_oracle(decoded, -2.23046875, [1.0] * 128, [1.0] * 128)
        self.assertEqual(decoded.receipt()["bits_per_weight"], 2.5)
        self.assertEqual(decoded.receipt()["halfwords_per_tile"], 40)
        self.assertEqual(decoded.receipt()["window"], 12)

    def test_down_projection_axes_folded_scales_forward_transpose_and_adjoint(self):
        left = [0.5 if i == 0 else (-1 if i % 3 else 0.25) for i in range(128)]
        right = [0.25 if i == 0 else (-0.5 if i == 128 else 2) for i in range(256)]
        identity = self.fixture(
            hidden=256,
            intermediate=128,
            projection="down_proj",
            word_values=(0,),
            suh=list(map(half, left)),
            svh=list(map(half, right)),
        )
        decoded = self.decode(identity)
        self.assertEqual((decoded.in_features, decoded.out_features), (128, 256))
        self.assert_constant_oracle(decoded, 1.84375, left, right)
        self.assertEqual(value(decoded.public_bits[0]), 29.5)
        self.assertEqual(value(decoded.public_bits[128]), -59)
        dense = [value(bits) for bits in decoded.public_bits]
        x = [((i * 13) % 23 - 11) / 16 for i in range(128)]
        dy = [((i * 7) % 19 - 9) / 8 for i in range(256)]
        forward, transpose = decoded.forward(x), decoded.transpose(dy)
        self.assertEqual(
            forward,
            [
                math.fsum(x[i] * dense[i * 256 + j] for i in range(128))
                for j in range(256)
            ],
        )
        self.assertEqual(
            transpose,
            [
                math.fsum(dy[j] * dense[i * 256 + j] for j in range(256))
                for i in range(128)
            ],
        )
        self.assertAlmostEqual(
            math.fsum(a * b for a, b in zip(forward, dy)),
            math.fsum(a * b for a, b in zip(x, transpose)),
            places=12,
        )
        for invalid in ([1.0] * 127, [True] * 128, [float("nan")] * 128):
            with self.assertRaises(ValueError):
                decoded.forward(invalid)
        with self.assertRaises(ValueError):
            decoded.transpose([1.0] * 128)

    def test_nonconstant_row_composes_identically_with_existing_reference(self):
        identity = self.fixture(word_values=(lambda i: (i * 73 + 0xA173) & 65535,))
        decoded = self.decode(identity)
        words = list(struct.unpack("<3072H", self.raw[0, "trellis"]))
        inner = m.trellis.reconstruct_inner(words, 128, 128, 48, window=14)
        expected = m.trellis.reconstruct_public(
            inner, [half(1)] * 128, [half(1)] * 128, 128, 128
        )
        self.assertEqual(decoded.public_bits, tuple(expected))
        self.assertGreater(len(set(decoded.public_bits)), 100)

    def test_larger_whole_expert_refused_before_any_payload_read(self):
        identity = self.fixture(hidden=512, intermediate=256, word_values=(0,))
        with mock.patch.object(m.rows.ExpertRowReader, "read") as read:
            with self.assertRaisesRegex(ValueError, "element cap"):
                self.decode(identity)
            read.assert_not_called()

    def test_each_budget_is_checked_before_payload_read(self):
        identity = self.fixture()
        for limits, message in (
            ({"max_elements": 8192}, "element cap"),
            ({"max_estimated_python_bytes": 1}, "working-set estimate"),
            ({"max_hadamard_terms": 1}, "Hadamard work"),
        ):
            with (
                self.subTest(limits=limits),
                mock.patch.object(m.rows.ExpertRowReader, "read") as read,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    self.decode(identity, **limits)
                read.assert_not_called()

    def test_budgets_cannot_be_lifted_or_noninteger(self):
        identity = self.fixture()
        for field, maximum in (
            ("max_elements", m.MAX_ELEMENTS),
            ("max_estimated_python_bytes", m.MAX_ESTIMATED_PYTHON_BYTES),
            ("max_hadamard_terms", m.MAX_HADAMARD_TERMS),
        ):
            for invalid in (0, -1, True, 1.5, float("nan"), maximum + 1):
                with (
                    self.subTest(field=field, invalid=invalid),
                    self.assertRaises(ValueError),
                ):
                    self.decode(identity, **{field: invalid})

    def test_wrong_expert_or_identity_rejected(self):
        identity = self.fixture()
        for expert in (-1, 2, True, 0.0):
            with (
                self.subTest(expert=expert),
                self.assertRaisesRegex(ValueError, "bounds"),
            ):
                self.decode(identity, expert)
        altered = copy.deepcopy(identity)
        altered["experts"][0]["window"] = 12
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            m.decode_expert(
                self.root,
                altered,
                expert=0,
                expected_sha256=m.rows.identity_sha256(identity),
            )
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            self.decode(altered)

    def test_nonfinite_stored_scales_refused_before_reconstruction(self):
        for bits in (0x7C00, 0xFC00, 0x7E00):
            identity = self.fixture(suh=[bits] + [half(1)] * 127)
            with (
                self.subTest(bits=bits),
                mock.patch.object(m.trellis, "reconstruct_inner") as decode,
            ):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.decode(identity)
                decode.assert_not_called()

    def test_payload_replaced_after_binding_refused(self):
        identity = self.fixture()
        path = self.root / self.name
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            self.decode(identity)

    def test_wrong_architecture_and_grouped_bank_never_bind(self):
        self.fixture()
        path = self.root / "config.json"
        config = json.loads(path.read_text())
        config["model_type"] = "unsupported"
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "architecture"):
            m.rows.bind_projection(self.root, shard_name=self.name, prefix=self.prefix)
        self.fixture()
        with self.assertRaisesRegex(ValueError, "uniform routed"):
            m.rows.bind_projection(
                self.root, shard_name=self.name, prefix=self.prefix + ".groups.0"
            )

    def test_raw_halfwords_require_exact_unsigned_little_endian_bytes(self):
        self.assertEqual(m._halfwords(b"\x01\x80\x03\x02", 2, "trellis"), [32769, 515])
        for raw in (b"\x01", b"\0\0\0\0\0\0", bytearray(b"\0\0\0\0")):
            with self.assertRaises(ValueError):
                m._halfwords(raw, 2, "trellis")

    def test_mutation_during_reconstruction_rejected(self):
        identity = self.fixture()

        def changed(*_):
            path = self.root / "config.json"
            path.write_text(path.read_text() + " ")
            return [half(0)] * 16384

        # This is a failure-path test, not numerical evidence.
        with (
            mock.patch.object(m.trellis, "reconstruct_inner", return_value=[]),
            mock.patch.object(m.trellis, "reconstruct_public", side_effect=changed),
            self.assertRaisesRegex(ValueError, "changed during decode"),
        ):
            self.decode(identity)

    def test_nonfinite_public_matrix_and_overflow_fail_closed(self):
        identity = self.fixture()
        for result in ([0x7C00] * 16384, OverflowError("overflow")):
            with (
                self.subTest(result_type=type(result).__name__),
                mock.patch.object(m.trellis, "reconstruct_inner", return_value=[]),
                mock.patch.object(
                    m.trellis,
                    "reconstruct_public",
                    **(
                        {"side_effect": result}
                        if isinstance(result, Exception)
                        else {"return_value": result}
                    ),
                ),
                self.assertRaisesRegex(ValueError, "nonfinite|overflowed"),
            ):
                self.decode(identity)

    def test_source_changes_rejected_before_read(self):
        identity = self.fixture()
        with (
            mock.patch.object(m, "_sources", return_value={}),
            mock.patch.object(m.rows.ExpertRowReader, "read") as read,
        ):
            with self.assertRaisesRegex(ValueError, "sources changed"):
                self.decode(identity)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
