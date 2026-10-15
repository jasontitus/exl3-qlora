import copy
import unittest

from exl3_mlx.workload_cost import _object, summarize


class WorkloadCostTests(unittest.TestCase):
    def fixture(self):
        return {"unit": "us", "mix_basis": "designed", "cases": [
            {"id": "small", "invocations": 9, "baseline": [1, 1], "candidate": [2, 2]},
            {"id": "large", "invocations": 1, "baseline": [100], "candidate": [50]},
        ]}

    def test_elapsed_cost_beats_case_count_and_mean_speedup(self):
        result = summarize(self.fixture())
        self.assertEqual(result["baseline_total_cost"], 109)
        self.assertEqual(result["candidate_total_cost"], 68)
        self.assertAlmostEqual(result["baseline_over_candidate"], 109 / 68)
        self.assertAlmostEqual(result["cases"][1]["baseline_time_share"], 100 / 109)

    def test_sample_count_is_not_invocation_weight(self):
        data = self.fixture()
        before = summarize(data)
        data["cases"][0]["baseline"] *= 17
        self.assertEqual(before, summarize(data))

    def test_average_speedup_can_reverse_the_decision(self):
        data = self.fixture()
        data["cases"] = [
            {"id": "tiny", "invocations": 1, "baseline": [1], "candidate": [0.1]},
            {"id": "large", "invocations": 1, "baseline": [100], "candidate": [200]},
        ]
        self.assertGreater((1 / 0.1 + 100 / 200) / 2, 1)
        result = summarize(data)
        self.assertLess(result["baseline_over_candidate"], 1)
        self.assertLess(result["saved_cost"], 0)

    def test_zero_weight_retained_and_baseline_only(self):
        data = self.fixture()
        data["cases"][0]["invocations"] = 0
        for row in data["cases"]:
            del row["candidate"]
        result = summarize(data)
        self.assertEqual(result["cases"][0]["baseline_time_share"], 0)
        self.assertNotIn("baseline_over_candidate", result)

    def test_missing_or_partial_fields_rejected(self):
        for field in ["id", "invocations", "baseline", "candidate"]:
            data = self.fixture()
            del data["cases"][0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(data)

    def test_invalid_numbers_rejected(self):
        for field in ["invocations", "baseline", "candidate"]:
            for value in [True, "1", float("nan"), float("inf"), -1]:
                data = self.fixture()
                data["cases"][0][field] = value if field == "invocations" else [value]
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    summarize(data)

    def test_zero_duration_or_zero_total_rejected(self):
        data = self.fixture()
        data["cases"][0]["baseline"] = [0]
        with self.assertRaises(ValueError):
            summarize(data)
        data = self.fixture()
        for row in data["cases"]:
            row["invocations"] = 0
        with self.assertRaises(ValueError):
            summarize(data)

    def test_duplicate_ids_and_mixed_units_rejected(self):
        data = self.fixture()
        data["cases"].append(copy.deepcopy(data["cases"][0]))
        with self.assertRaises(ValueError):
            summarize(data)
        data = self.fixture()
        data["cases"][0]["unit"] = "ms"
        with self.assertRaises(ValueError):
            summarize(data)
        with self.assertRaises(ValueError):
            _object([("unit", "us"), ("unit", "ms")])


if __name__ == "__main__":
    unittest.main()
