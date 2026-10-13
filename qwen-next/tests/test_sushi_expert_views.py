import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sushi_views", ROOT / "tools/sushi_expert_views.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ViewsTests(unittest.TestCase):
    def setUp(self):
        self.header = json.loads(
            (ROOT / "sources/model-exl3-L47-gate.safetensors.header.json").read_text()
        )
        self.config = json.loads((ROOT / "sources/config.json").read_text())
        self.args = {
            "prefix": "language_model.model.layers.47.mlp.switch_mlp.gate_proj",
            "expert": 0,
            "config": self.config,
            "data_start": 872,
            "file_bytes": 317850472,
        }

    def test_real_header_first_last_expert(self):
        first = m.expert_views(self.header, **self.args)
        last = m.expert_views(self.header, **{**self.args, "expert": 511})
        self.assertEqual(first["views"]["trellis"]["length_bytes"], 614400)
        self.assertEqual(first["total_read_bytes"], 620800)
        self.assertEqual(first["views"]["trellis"]["offset_bytes"], 872)
        self.assertEqual(
            last["views"]["svh"]["offset_bytes"] + last["views"]["svh"]["length_bytes"],
            self.args["file_bytes"],
        )

    def test_invalid_expert_indices(self):
        for index in [-1, 512, True, 1.0]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.expert_views(self.header, **{**self.args, "expert": index})

    def test_window_mismatch_and_invalid(self):
        for win in [16, 7, True]:
            self.config["expert_quant"]["window"] = win
            with self.subTest(win=win), self.assertRaises(ValueError):
                m.expert_views(self.header, **self.args)

    def test_unsupported_names(self):
        for suffix in [".g0", "_other"]:
            with self.assertRaises(ValueError):
                m.expert_views(
                    self.header, **{**self.args, "prefix": self.args["prefix"] + suffix}
                )

    def test_bounds_overlap_and_dtype(self):
        for change in ["bounds", "overlap", "dtype", "shape", "stamp", "missing"]:
            h = copy.deepcopy(self.header)
            key = self.args["prefix"]
            if change == "bounds":
                h[key + ".svh"]["data_offsets"][1] += 1
            elif change == "overlap":
                h[key + ".suh"]["data_offsets"] = [0, 2621440]
            elif change == "dtype":
                h[key + ".trellis"]["dtype"] = "I16"
            elif change == "shape":
                h[key + ".trellis"]["shape"][0] = 511
            elif change == "stamp":
                h["__metadata__"]["k"] = "2.5"
            else:
                del h[key + ".svh"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.expert_views(h, **self.args)


if __name__ == "__main__":
    unittest.main()
