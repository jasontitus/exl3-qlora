"""CPU-only validation/scoring tests; importing the runner must not import MLX."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "quality_probe.py"
    if Path(__file__).parent.name == "tests"
    else Path(__file__).resolve().parent / "quality_probe.py"
)
spec = importlib.util.spec_from_file_location("quality_probe_under_test", SCRIPT_PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture(**overrides):
    return {
        "case_id": "case1",
        "source_family": "family",
        "category": "control",
        "prompt": "Repeat blue",
        "expected": "blue",
        "max_new_tokens": 3,
        **overrides,
    }


class ProbeTests(unittest.TestCase):
    def read(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return probe.read_cases(path)

    def test_valid_schema(self):
        self.assertEqual(self.read([fixture()]), [fixture()])

    def test_reject_duplicate_id(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read([fixture(), fixture(prompt="different")])

    def test_reject_invalid_fields(self):
        for changed in (
            {"max_new_tokens": True},
            {"max_new_tokens": 0},
            {"max_new_tokens": 32769},
            {"prompt": " "},
            {"expected": []},
            {"case_id": " case1"},
            {"surprise": 1},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.read([fixture(**changed)])
        with self.assertRaises(ValueError):
            self.read([])

    def tokenized(self, **kwargs):
        args = {
            "case": fixture(),
            "example": {"input_ids": [0, 4, 5, 1], "loss_start": 2},
            "prompt_ids": [0, 4],
            "eos_token_id": 1,
            "vocab_size": 10,
            **kwargs,
        }
        return probe.validate_tokenized_case(**args)

    def test_valid_tokenization(self):
        self.assertEqual(self.tokenized()["target_ids"], [5, 1])

    def test_reject_template_prefix_mismatch(self):
        with self.assertRaisesRegex(ValueError, "template"):
            self.tokenized(prompt_ids=[0, 3])

    def test_reject_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "context"):
            self.tokenized(
                prompt_ids=[4] * 32767,
                example={"input_ids": [4] * 32767 + [5, 1], "loss_start": 32767},
            )

    def test_reject_expectation_larger_than_generation_cap(self):
        with self.assertRaisesRegex(ValueError, "cap"):
            self.tokenized(case=fixture(max_new_tokens=1))

    def test_reject_no_eos_and_bad_id(self):
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.tokenized(example={"input_ids": [0, 4, 5, 6], "loss_start": 2})
        with self.assertRaisesRegex(ValueError, "token IDs"):
            self.tokenized(example={"input_ids": [0, 4, 15, 1], "loss_start": 2})

    def test_stripped_exact_grade_requires_eos_for_terminated_pass(self):
        result = probe.grade_answer("blue", " blue\n", [5, 1], 1, 3)
        self.assertTrue(result["exact_text_match"])
        self.assertTrue(result["exact_answer_and_eos"])
        truncated = probe.grade_answer("blue", "blue", [5], 1, 1)
        self.assertTrue(truncated["exact_text_match"])
        self.assertFalse(truncated["exact_answer_and_eos"])
        self.assertTrue(truncated["truncated"])
        self.assertFalse(probe.grade_answer("blue", "Blue", [5, 1], 1, 3)["exact_text_match"])

    def test_reject_impossible_generation(self):
        for ids in ([], [5, 1, 6], [5, 6, 7, 8], [5]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                probe.grade_answer("blue", "blue", ids, 1, 3)

    def test_generation_evidence_uses_exact_ids_cap_eos_and_stop_reason(self):
        case = self.tokenized()
        result = {
            "prompt_ids": [0, 4],
            "prompt_tokens": 2,
            "max_new_tokens": 3,
            "eos_token_id": 1,
            "stop_reason": "eos",
            "text": "blue",
            "generated_ids": [5, 1],
        }
        self.assertTrue(probe.validate_generation_evidence(case, result)["exact_answer_and_eos"])
        for wrong in (
            {"prompt_ids": [0, 3]},
            {"prompt_tokens": 3},
            {"max_new_tokens": 4},
            {"eos_token_id": 2},
            {"stop_reason": "max_tokens"},
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                probe.validate_generation_evidence(case, {**result, **wrong})

    def records(self):
        result = []
        for phase, text, ids, loss in (
            ("base", "blue", [5, 1], 2.0),
            ("adapter", "violet", [6, 1], 3.0),
        ):
            generation = {
                "text": text,
                "generated_ids": ids,
                **probe.grade_answer("blue", text, ids, 1, 3),
            }
            result.append(
                {
                    **fixture(),
                    "phase": phase,
                    "prompt_ids": [0, 4],
                    "target_ids": [5, 1],
                    "evaluation": {
                        "loss": loss,
                        "supervised_tokens": 2,
                        "nll_sum": loss * 2,
                    },
                    "generation": generation,
                }
            )
        return result

    def test_paired_regression_and_no_certification(self):
        report = probe.paired_summary([fixture()], self.records())
        self.assertEqual(report["regressions"], 1)
        self.assertEqual(report["improvements"], 0)
        self.assertTrue(report["pairs"][0]["unexpected_violet"])
        self.assertEqual(report["pairs"][0]["nll_delta_adapter_minus_base"], 1.0)
        self.assertEqual(report["totals"]["adapter"]["token_weighted_nll"], 3.0)
        self.assertFalse(report["quality_certified"])

    def test_reject_incomplete_duplicate_or_misaligned_comparison(self):
        records = self.records()
        for bad in (
            records[:1],
            records + records[:1],
            [records[0], {**records[1], "prompt_ids": [0, 3]}],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.paired_summary([fixture()], bad)

    def test_cpu_import_has_no_mlx(self):
        # The surrounding pytest process may already use MLX; check a fresh CPU process.
        code = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('probe', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "assert 'mlx.core' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
