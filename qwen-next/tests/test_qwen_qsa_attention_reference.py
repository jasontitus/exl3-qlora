"""Bounded scalar/composed CPU oracles, not native/model parity tests."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qwen_qsa_attention_reference as a
import qwen_qsa_selector_reference as selector


def scalar(q, k, v, gate, routes, positions):
    """Independent scalar loops with math.fsum and explicit contiguous GQA."""
    b, n, h, d = q.shape
    out = np.zeros((b, n, h * d), np.float64)
    for bi in range(b):
        for qi in range(n):
            if positions[bi, qi] < 0:
                continue
            ids = [int(x) for x in routes[bi, qi] if x >= 0]
            for hi in range(h):
                ki = hi // (h // k.shape[2])
                logits = [math.fsum(float(q[bi, qi, hi, j]) * float(k[bi, t, ki, j]) for j in range(d)) / math.sqrt(d) for t in ids]
                exps = [math.exp(x - max(logits)) for x in logits]
                total = math.fsum(exps)
                for j in range(d):
                    g = float(gate[bi, qi, hi, j])
                    sig = (1 / (1 + math.exp(-g))) if g >= 0 else math.exp(g) / (1 + math.exp(g))
                    out[bi, qi, hi * d + j] = math.fsum(w * float(v[bi, t, ki, j]) for t, w in zip(ids, exps)) / total * sig
    return out


def fixture(dtype=np.float64):
    rng = np.random.default_rng(71)
    q = rng.normal(size=(2, 3, 4, 3)).astype(dtype)
    k = rng.normal(size=(2, 7, 2, 3)).astype(dtype)
    v = rng.normal(size=k.shape).astype(dtype)
    gate = rng.normal(size=q.shape).astype(dtype)
    positions = np.array([[2, 3, 4], [3, 4, -1]], np.int64)
    lengths = np.array([7, 5], np.int64)
    routes = np.full((2, 3, 32), -1, np.int32)
    for b in range(2):
        for j in range(3):
            pos = positions[b, j]
            if pos >= 0:
                ids = list(range(0, pos + 1, 2))
                routes[b, j, :len(ids)] = ids
    return q, k, v, gate, routes, positions, lengths


class AttentionTests(unittest.TestCase):
    def test_scalar_oracle_three_input_dtypes(self):
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                args = fixture(dtype)
                result = a.attend(*args)
                np.testing.assert_allclose(result.context(), scalar(*args[:-1]), rtol=2e-14, atol=2e-14)
                self.assertEqual(result.context().dtype, np.float64)
                self.assertEqual(result.context_shape, (2, 3, 12))

    def test_real_head_geometry_and_elementwise_gate(self):
        q = np.zeros((1, 1, 24, 256), np.float32)
        k = np.zeros((1, 1, 2, 256), np.float32)
        v = np.empty_like(k)
        v[:, :, 0] = 2
        v[:, :, 1] = 8
        g = np.zeros_like(q)
        g[..., 0] = np.log(3)
        ids = np.full((1, 1, 32), -1, np.int32)
        ids[0, 0, 0] = 0
        out = a.attend(q, k, v, g, ids, np.array([[0]], np.int64), np.array([1], np.int64)).context().reshape(24, 256)
        np.testing.assert_array_equal(out[:12, 1:], 1)
        np.testing.assert_array_equal(out[12:, 1:], 4)
        np.testing.assert_allclose(out[:12, 0], 1.5, rtol=1e-7)
        np.testing.assert_allclose(out[12:, 0], 6, rtol=1e-7)

    def test_selector_composition_actual_2051_2052_boundary(self):
        q_idx = np.zeros((1, 2, 4, 4), np.float32)
        pooled = np.zeros((1, 513, 4), np.float32)
        positions = np.array([[2050, 2051]], np.int64)
        lengths = np.array([2052], np.int64)
        captured = selector.select(q_idx, pooled, positions, lengths)
        routes = selector.replay(captured, q_idx, pooled, positions, lengths)
        self.assertEqual([np.count_nonzero(r >= 0) for r in routes[0]], [2051, 2048])
        q = np.zeros((1, 2, 4, 2), np.float32)
        k = np.zeros((1, 2052, 2, 2), np.float32)
        v = np.broadcast_to(np.arange(2052, dtype=np.float32)[None, :, None, None], k.shape)
        result = a.attend(q, k, v, q.copy(), routes, positions, lengths)
        np.testing.assert_allclose(result.context()[0, 0], np.mean(np.arange(2051)) * 0.5)
        np.testing.assert_allclose(result.context()[0, 1], np.mean(np.arange(2048)) * 0.5)
        self.assertEqual(result.largest_selected_count, 2051)
        self.assertEqual(result.selected_kv_pair_bytes, 2 * 2051 * 2 * 8)
        self.assertEqual(result.score_vector_bytes, 2051 * 8)
        with self.assertRaises(ValueError):
            selector.replay(captured, q_idx + 1, pooled, positions, lengths)

    def test_reduced_budget_sparse_composition_scalar(self):
        rng = np.random.default_rng(81)
        positions = np.array([[8, 9, 10]], np.int64)
        lengths = np.array([11], np.int64)
        routes = selector.select(rng.normal(size=(1, 3, 2, 4)).astype(np.float32), rng.normal(size=(1, 2, 4)).astype(np.float32), positions, lengths, budget=4).array()
        q = rng.normal(size=(1, 3, 4, 3))
        k = rng.normal(size=(1, 11, 2, 3))
        v = rng.normal(size=k.shape)
        gate = rng.normal(size=q.shape)
        np.testing.assert_allclose(a.attend(q, k, v, gate, routes, positions, lengths).context(), scalar(q, k, v, gate, routes, positions), rtol=2e-14, atol=2e-14)

    def test_masked_slots_permutation_and_right_padding(self):
        args = list(fixture())
        original = a.attend(*args).context()
        args[4] = args[4][:, :, ::-1].copy()
        np.testing.assert_allclose(a.attend(*args).context(), original, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(original[1, 2], 0)

    def test_no_input_mutation_and_immutable_output(self):
        args = fixture()
        before = [x.tobytes() for x in args]
        result = a.attend(*args)
        self.assertEqual(before, [x.tobytes() for x in args])
        with self.assertRaises(ValueError):
            result.context().setflags(write=True)

    def test_noncontiguous_inputs(self):
        args = list(fixture())
        for i in range(4):
            args[i] = args[i][..., ::-1]
        np.testing.assert_allclose(a.attend(*args).context(), scalar(*args[:-1]), rtol=2e-14, atol=2e-14)

    def test_stable_extreme_gate_and_softmax(self):
        args = list(fixture())
        args[0].fill(100)
        args[1].fill(100)
        args[3][..., 0] = 1000
        args[3][..., 1] = -1000
        actual = a.attend(*args).context().reshape(args[0].shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual[..., 1], 0)
        np.testing.assert_allclose(actual.reshape(2, 3, 12), scalar(*args[:-1]), rtol=2e-14, atol=2e-14)

    def test_invalid_route_contracts(self):
        for change in ("duplicate", "future", "negative", "empty", "padded", "dtype", "width"):
            with self.subTest(change=change):
                args = list(fixture())
                if change == "duplicate": args[4][0, 0, 1] = 0
                elif change == "future": args[4][0, 0, 1] = 3
                elif change == "negative": args[4][0, 0, 1] = -2
                elif change == "empty": args[4][0, 0].fill(-1)
                elif change == "padded": args[4][1, 2, 0] = 0
                elif change == "dtype": args[4] = args[4].astype(np.int64)
                else: args[4] = args[4][:, :, :31]
                with self.assertRaises(ValueError): a.attend(*args)

    def test_invalid_shapes_positions_and_bounds(self):
        for change in ("heads", "flatgate", "position_hole", "position_gap", "length", "object", "huge"):
            with self.subTest(change=change):
                args = list(fixture())
                if change == "heads": args[0] = args[0][:, :, :3]
                elif change == "flatgate": args[3] = args[3].reshape(2, 3, 12)
                elif change == "position_hole": args[5][0] = [2, -1, 4]
                elif change == "position_gap": args[5][0] = [1, 3, 4]
                elif change == "length": args[6][0] = 8
                elif change == "object": args[0] = args[0].astype(object)
                else: args[0] = np.broadcast_to(np.array(0.), (2, 64, 24, 4096))
                with self.assertRaises(ValueError): a.attend(*args)

    def test_nonfinite_inputs_and_overflow_refused(self):
        for i in range(4):
            args = list(fixture())
            args[i].flat[0] = np.nan
            with self.subTest(i=i), self.assertRaises(ValueError): a.attend(*args)
        args = list(fixture())
        args[0].fill(1e308)
        args[1].fill(1e308)
        with self.assertRaisesRegex(ValueError, "Nonfinite reference arithmetic"):
            a.attend(*args)

    def test_array_subclasses_refused_before_arithmetic_in_all_roles(self):
        class CustomArray(np.ndarray):
            pass

        for role in range(7):
            for kind in ("masked", "empty_mask", "custom_subclass"):
                args = list(fixture())
                if kind == "custom_subclass":
                    args[role] = args[role].view(CustomArray)
                else:
                    mask = np.zeros(args[role].shape, dtype=bool)
                    if kind == "masked":
                        mask.flat[0] = True
                    args[role] = np.ma.array(args[role], mask=mask)
                with self.subTest(role=role, kind=kind), self.assertRaisesRegex(ValueError, "Plain NumPy ndarrays required"):
                    a.attend(*args)

    def test_no_device_libraries_imported(self):
        self.assertFalse(any(name in ("torch", "mlx", "mlx_lm", "triton") or name.startswith(("torch.", "mlx.", "mlx_lm.", "triton.")) for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
