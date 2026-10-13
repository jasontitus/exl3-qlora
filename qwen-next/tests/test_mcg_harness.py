"""CPU fixtures/host validation only: these tests NEVER pass --dispatch."""

import copy
import importlib.util
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mcg_harness_fixtures", ROOT / "tools/make_mcg_harness_fixtures.py"
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def independent_expected(case):
    """Direct cyclic bit lookup and explicit matrix placement, not funnel shifts."""
    output = []
    for row in range(case["k"]):
        for col in range(case["start"], case["start"] + case["width"]):
            base = ((row // 16) * (case["n"] // 16) + col // 16) * case["halfwords"]
            words = case["packed"][base : base + case["halfwords"]]
            bits = []
            for low, high in zip(words[::2], words[1::2]):
                bits.extend(((low | (high << 16)) >> b) & 1 for b in range(31, -1, -1))
            # Packed permutation: thread chooses two adjacent rows in either half.
            r, c = row % 16, col % 16
            thread = (c % 8) * 4 + (r % 8) // 2
            lane = (r % 2) + (r // 8) * 2 + (c // 8) * 4
            end = ((thread * 8 + lane + 1) * case["halfwords"]) // 16
            code = 0
            for i in range(end - 16, end):
                code = 2 * code + bits[i % len(bits)]
            mixed = ((code & ((1 << case["window"]) - 1)) * 0xCBAC1FED) & 0xFFFFFFFF
            pair = (mixed & 0x8FFF8FFF) ^ 0x3B603B60
            values = [
                struct.unpack("<e", struct.pack("<H", b))[0]
                for b in (pair & 65535, pair >> 16)
            ]
            output.append(struct.unpack("<H", struct.pack("<e", sum(values)))[0])
    return output


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = fixtures.build_suite()

    def test_all_generated_values_match_direct_bit_oracle(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["case_id"]):
                self.assertEqual(independent_expected(case), case["expected_fp16_bits"])

    def test_known_zero_exact_bits(self):
        self.assertEqual(self.suite["cases"][0]["expected_fp16_bits"], [0x3F60] * 256)

    def test_coverage_bound_and_identity(self):
        cases = self.suite["cases"]
        self.assertEqual(len(cases), 94)
        self.assertEqual({c["halfwords"] for c in cases}, set(range(16, 129, 2)))
        self.assertEqual({c["window"] for c in cases}, set(range(8, 17)))
        self.assertLess(sum(len(c["expected_fp16_bits"]) for c in cases), 1 << 20)
        self.assertEqual(
            self.suite["kernel_sha256"], fixtures.digest(fixtures.KERNEL.read_bytes())
        )
        self.assertEqual(self.suite["comparison"], "exact-fp16-bits")

    def test_deterministic_and_each_payload_hash_bound(self):
        self.assertEqual(self.suite, fixtures.build_suite())
        for c in self.suite["cases"]:
            self.assertEqual(c["packed_sha256"], fixtures.word_hash(c["packed"]))
            self.assertEqual(
                c["expected_sha256"], fixtures.word_hash(c["expected_fp16_bits"])
            )


@unittest.skipUnless(
    os.environ.get("MCG_HARNESS_BIN"),
    "Set MCG_HARNESS_BIN to compiled Swift host for CPU validation",
)
class SwiftValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = str(Path(os.environ["MCG_HARNESS_BIN"]).resolve())
        cls.suite = fixtures.build_suite()
        cls.suite["cases"] = cls.suite["cases"][:1]

    def run_validation(self, mutate=None, *, mode=None, existing=False):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            suite = copy.deepcopy(self.suite)
            if mutate:
                mutate(suite)
            source = tmp / "fixtures.json"
            source.write_text(json.dumps(suite))
            output = tmp / "receipt.json"
            if existing:
                output.write_text("do not overwrite")
            command = [
                self.binary,
                *(mode if mode is not None else ["--validate-only"]),
                "--fixtures",
                str(source),
                "--kernel",
                str(fixtures.KERNEL),
                "--receipt",
                str(output),
            ]
            result = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=15
            )
            receipt = output.read_text() if output.exists() else None
            return result, receipt

    def rejects(self, mutate=None, **kwargs):
        result, receipt = self.run_validation(mutate, **kwargs)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Rejected:", result.stderr)
        self.assertIsNone(receipt)

    def test_validation_is_explicitly_not_gpu_parity(self):
        result, raw = self.run_validation()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(raw)
        self.assertTrue(report["validation_passed"])
        self.assertFalse(report["gpu_executed"])
        self.assertFalse(report["numerical_parity_qualified"])
        self.assertNotIn("device", report)
        self.assertEqual(report["validated_values"], 256)

    def test_invalid_geometry_rate_and_unsigned_payload(self):
        for key, value in (
            ("k", True),
            ("k", 17),
            ("k", 272),
            ("k", 16.5),
            ("n", 0),
            ("start", 1),
            ("start", 16),
            ("width", 32),
            ("halfwords", 17),
            ("halfwords", 130),
            ("window", 7),
            ("window", 17),
        ):
            with self.subTest(key=key, value=value):
                self.rejects(
                    lambda s, key=key, value=value: s["cases"][0].update({key: value})
                )
        for value in (-1, 65536, True, 1.5):
            with self.subTest(word=value):
                self.rejects(
                    lambda s, value=value: s["cases"][0]["packed"].__setitem__(0, value)
                )

    def test_lengths_hashes_nonfinite_and_missing_keys(self):
        mutations = [
            lambda s: s["cases"][0]["packed"].pop(),
            lambda s: s["cases"][0]["expected_fp16_bits"].pop(),
            lambda s: s["cases"][0]["expected_fp16_bits"].__setitem__(0, 0x7C00),
            lambda s: s["cases"][0]["packed"].__setitem__(0, 1),
            lambda s: s["cases"][0]["expected_fp16_bits"].__setitem__(0, 0),
            lambda s: s["cases"][0].pop("window"),
            lambda s: s["cases"][0].update(unknown="no"),
            lambda s: s.update(kernel_sha256="0" * 64),
            lambda s: s.update(cpu_reference_sha256="0" * 64),
            lambda s: s.update(generator_sha256="invalid"),
            lambda s: s.update(scope="real-model"),
            lambda s: s.update(comparison="close-enough"),
            lambda s: s.update(unknown="no"),
            lambda s: s["cases"].clear(),
            lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.rejects(mutate)

    def test_modes_required_no_overwrite(self):
        self.rejects(mode=[])
        # Does not reach a Metal device: duplicate mode rejected during argument parsing.
        self.rejects(mode=["--validate-only", "--dispatch"])
        result, receipt = self.run_validation(existing=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(receipt, "do not overwrite")

    def test_symlink_and_oversized_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "input.json"
            for oversized in (False, True):
                if oversized:
                    path.unlink()
                    with path.open("wb") as handle:
                        handle.truncate((8 << 20) + 1)
                else:
                    path.symlink_to(fixtures.KERNEL)
                result = subprocess.run(
                    [
                        self.binary,
                        "--validate-only",
                        "--fixtures",
                        str(path),
                        "--kernel",
                        str(fixtures.KERNEL),
                        "--receipt",
                        str(tmp / "out.json"),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 1)
                self.assertFalse((tmp / "out.json").exists())


if __name__ == "__main__":
    unittest.main()
