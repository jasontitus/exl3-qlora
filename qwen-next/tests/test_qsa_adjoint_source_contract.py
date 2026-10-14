"""Portable exact dependencies and actual committed-forward composition, CPU only."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qwen-next/tools"))
import qwen_qsa_attention_adjoint as adjoint
import qwen_qsa_attention_reference as forward


class AdjointSourceContract(unittest.TestCase):
    def test_portable_source_reference_and_dependency_hashes(self):
        contract = json.loads((ROOT / "qwen-next/reference-contracts/qsa-adjoint-cpu-v1.json").read_text())
        self.assertEqual(contract["schema"], "qwen-qsa-adjoint-cpu-reference-contract-v1")
        for path, expected in contract["files_sha256"].items():
            with self.subTest(path=path):
                self.assertFalse(Path(path).is_absolute())
                self.assertNotIn("..", Path(path).parts)
                self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)
        linked = json.loads((ROOT / contract["forward_contract"]).read_text())
        for path, entry in linked["sources"].items():
            raw = (ROOT / path).read_bytes()
            with self.subTest(source=path):
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
                for span in entry["ranges"] if "ranges" in entry else []:
                    value = b"".join(raw.splitlines(keepends=True)[span["first_line"] - 1:span["last_line"]])
                    self.assertEqual(hashlib.sha256(value).hexdigest(), span["sha256"])
        self.assertTrue(contract["fixed_route_mathematical_input_adjoint"])
        for claim in ("selector_membership_gradient", "native_backward_qualified", "model_or_training_supported", "capacity_16k_qualified"):
            self.assertFalse(contract[claim])
        self.assertEqual(adjoint.MAX_INPUT_BYTES, contract["limits"]["aggregate_input_bytes"])
        self.assertEqual(adjoint.MAX_GRADIENT_BYTES, contract["limits"]["gradient_bytes"])
        self.assertEqual(adjoint.MAX_SELECTED_COMPONENTS, contract["limits"]["selected_components"])

    def test_actual_imported_forward_and_adjoint_are_the_bound_product_files(self):
        self.assertIs(adjoint.forward, forward)
        self.assertEqual(Path(adjoint.__file__).resolve(), ROOT / "qwen-next/tools/qwen_qsa_attention_adjoint.py")
        self.assertEqual(Path(forward.__file__).resolve(), ROOT / "qwen-next/tools/qwen_qsa_attention_reference.py")

    def test_committed_forward_and_adjoint_uniform_shared_kv(self):
        q = np.zeros((1, 1, 4, 2), dtype=np.float64)
        k = np.zeros((1, 8, 2, 2), dtype=np.float64)
        v = np.full_like(k, 2.0)
        gate = np.zeros_like(q)
        route = np.full((1, 1, 32), -1, dtype=np.int32)
        route[0, 0, :8] = np.arange(8)
        inputs = [q, k, v, gate, route, np.array([[7]], np.int64), np.array([8], np.int64)]
        before = [x.tobytes() for x in inputs]
        context = forward.attend(*inputs).context()
        np.testing.assert_array_equal(context, np.ones((1, 1, 8)))
        gradients = adjoint.adjoint(*inputs, np.ones_like(context))
        for observed, expected in ((gradients.dq, np.zeros_like(q)), (gradients.dk, np.zeros_like(k)), (gradients.dv, np.full_like(v, 0.125)), (gradients.dgate, np.full_like(gate, 0.5))):
            np.testing.assert_array_equal(observed.array(), expected)
            self.assertFalse(observed.array().flags.writeable)
        self.assertEqual(before, [x.tobytes() for x in inputs])


if __name__ == "__main__":
    unittest.main()
