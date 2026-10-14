"""Tiny scalar/finite-difference adjoint checks; no native/device qualification."""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qwen_qsa_attention_adjoint as a
import qwen_qsa_attention_reference as f
import qwen_qsa_selector_reference as selector


def fixture():
    rng = np.random.default_rng(105)
    q = rng.normal(scale=0.3, size=(2, 2, 4, 2))
    k = rng.normal(scale=0.3, size=(2, 5, 2, 2))
    v = rng.normal(size=k.shape)
    gate = rng.normal(scale=0.4, size=q.shape)
    positions = np.array([[2, 3], [2, -1]], np.int64)
    lengths = np.array([5, 4], np.int64)
    routes = np.full((2, 2, 32), -1, np.int32)
    routes[0, 0, :2] = [0, 2]
    routes[0, 1, :3] = [0, 1, 3]
    routes[1, 0, :2] = [1, 2]
    dy = rng.normal(size=(2, 2, 8))
    return [q, k, v, gate, routes, positions, lengths, dy]


def gradients(result):
    return [x.array() for x in (result.dq, result.dk, result.dv, result.dgate)]


def scalar_output(args):
    """Independent math.fsum forward using lists and scalar dot/softmax/gate."""
    q, k, v, gate, routes, positions, _, _ = args
    b, n, h, d = q.shape
    out = np.zeros((b, n, h, d))
    for bi in range(b):
        for qi in range(n):
            if positions[bi, qi] == -1:
                continue
            ids = [int(x) for x in routes[bi, qi] if x >= 0]
            for hi in range(h):
                ki = hi // (h // k.shape[2])
                scores = [math.fsum(float(q[bi, qi, hi, j]) * float(k[bi, t, ki, j]) for j in range(d)) / math.sqrt(d) for t in ids]
                exps = [math.exp(s - max(scores)) for s in scores]
                weights = [e / math.fsum(exps) for e in exps]
                for j in range(d):
                    sig = 1 / (1 + math.exp(-float(gate[bi, qi, hi, j])))
                    out[bi, qi, hi, j] = math.fsum(w * float(v[bi, t, ki, j]) for w, t in zip(weights, ids)) * sig
    return out.reshape(b, n, h * d)


def scalar_loss(args):
    output = scalar_output(args)
    return math.fsum(float(x) * float(y) for x, y in zip(output.flat, args[7].flat))


class AdjointTests(unittest.TestCase):
    def test_all_four_gradients_against_scalar_finite_differences(self):
        args = fixture()
        actual = gradients(a.adjoint(*args))
        step = 1e-5
        for role in range(4):
            expected = np.zeros_like(args[role])
            for index in np.ndindex(args[role].shape):
                old = args[role][index]
                args[role][index] = old + step
                plus = scalar_loss(args)
                args[role][index] = old - step
                minus = scalar_loss(args)
                args[role][index] = old
                expected[index] = (plus - minus) / (2 * step)
            with self.subTest(role=role):
                np.testing.assert_allclose(actual[role], expected, rtol=3e-7, atol=6e-11)

    def test_directional_adjoint_identity_with_approved_forward(self):
        args = fixture()
        rng = np.random.default_rng(106)
        tangent = [rng.normal(size=x.shape) for x in args[:4]]
        lhs = math.fsum(float(x) * float(y) for grad, direction in zip(gradients(a.adjoint(*args)), tangent) for x, y in zip(grad.flat, direction.flat))
        for epsilon in (1e-4, 2e-5):
            plus = [x + epsilon * direction for x, direction in zip(args[:4], tangent)] + args[4:7]
            minus = [x - epsilon * direction for x, direction in zip(args[:4], tangent)] + args[4:7]
            jvp = (f.attend(*plus).context() - f.attend(*minus).context()) / (2 * epsilon)
            rhs = math.fsum(float(x) * float(y) for x, y in zip(jvp.flat, args[7].flat))
            with self.subTest(epsilon=epsilon):
                self.assertAlmostEqual(lhs, rhs, delta=3e-8)

    def test_selector_replay_is_fixed_while_main_qkv_gradients_vary(self):
        rng = np.random.default_rng(107)
        positions = np.array([[8, 9]], np.int64)
        lengths = np.array([10], np.int64)
        iq = rng.normal(size=(1, 2, 2, 4)).astype(np.float32)
        pooled = rng.normal(size=(1, 2, 4)).astype(np.float32)
        captured = selector.select(iq, pooled, positions, lengths, budget=4)
        routes = selector.replay(captured, iq, pooled, positions, lengths, budget=4)
        q = rng.normal(scale=0.2, size=(1, 2, 4, 2))
        k = rng.normal(scale=0.2, size=(1, 10, 2, 2))
        args = [q, k, rng.normal(size=k.shape), rng.normal(size=q.shape), routes, positions, lengths, rng.normal(size=(1, 2, 8))]
        direction = rng.normal(size=q.shape)
        grad = a.adjoint(*args).dq.array()
        epsilon = 1e-5
        plus, minus = args.copy(), args.copy()
        plus[0], minus[0] = q + epsilon * direction, q - epsilon * direction
        expected = (scalar_loss(plus) - scalar_loss(minus)) / (2 * epsilon)
        self.assertAlmostEqual(float(np.sum(grad * direction)), expected, delta=3e-10)
        np.testing.assert_array_equal(routes, captured.array())
        with self.assertRaisesRegex(ValueError, "capture/input mismatch"):
            selector.replay(captured, iq + 1, pooled, positions, lengths, budget=4)

    def test_exact_single_token_shared_kv_accumulation_and_gate(self):
        q = np.zeros((1, 2, 24, 256))
        k = np.zeros((1, 2, 2, 256))
        v = np.empty_like(k)
        v[:, :, 0] = 2
        v[:, :, 1] = 6
        routes = np.full((1, 2, 32), -1, np.int32)
        routes[:, :, 0] = 0
        args = [q, k, v, q.copy(), routes, np.array([[0, 1]], np.int64), np.array([2], np.int64), np.ones((1, 2, 6144))]
        dq, dk, dv, dg = gradients(a.adjoint(*args))
        np.testing.assert_array_equal(dq, 0)
        np.testing.assert_array_equal(dk, 0)
        np.testing.assert_array_equal(dv[0, 0], 12)  # 2 queries *12 heads *sigmoid(0).
        np.testing.assert_array_equal(dv[0, 1], 0)
        np.testing.assert_array_equal(dg[:, :, :12], 0.5)
        np.testing.assert_array_equal(dg[:, :, 12:], 1.5)

    def test_shared_kv_equals_sum_of_separate_query_head_contributions(self):
        args = fixture()
        actual = gradients(a.adjoint(*args))
        accumulated = [np.zeros_like(x) for x in actual]
        for bi in range(2):
            for qi in range(2):
                for hi in range(4):
                    separate = args.copy()
                    separate[7] = np.zeros_like(args[7])
                    separate[7][bi, qi, hi * 2:hi * 2 + 2] = args[7][bi, qi, hi * 2:hi * 2 + 2]
                    for total, grad in zip(accumulated, gradients(a.adjoint(*separate))):
                        total += grad
        for total, grad in zip(accumulated, actual):
            np.testing.assert_allclose(total, grad, rtol=2e-14, atol=2e-14)

    def test_cotangent_linearity_zero_and_readonly_results(self):
        args = fixture()
        first = gradients(a.adjoint(*args))
        scaled = args.copy()
        scaled[7] = args[7] * -2.5
        for left, right in zip(gradients(a.adjoint(*scaled)), first):
            np.testing.assert_allclose(left, -2.5 * right, rtol=3e-14, atol=3e-14)
            self.assertFalse(left.flags.writeable)
            with self.assertRaises(ValueError):
                left.setflags(write=True)
        scaled[7] = np.zeros_like(args[7])
        for grad in gradients(a.adjoint(*scaled)):
            np.testing.assert_array_equal(grad, 0)

    def test_padding_unselected_future_tokens_have_zero_gradients(self):
        args = fixture()
        args[7][1, 1] = 999
        dq, dk, dv, dg = gradients(a.adjoint(*args))
        np.testing.assert_array_equal(dq[1, 1], 0)
        np.testing.assert_array_equal(dg[1, 1], 0)
        np.testing.assert_array_equal(dk[0, 4], 0)
        np.testing.assert_array_equal(dv[0, 4], 0)
        np.testing.assert_array_equal(dk[1, [0, 3, 4]], 0)
        np.testing.assert_array_equal(dv[1, [0, 3, 4]], 0)
        args[5].fill(-1)
        args[4].fill(-1)
        result = a.adjoint(*args)
        self.assertEqual(result.selected_components, 0)
        for grad in gradients(result):
            np.testing.assert_array_equal(grad, 0)

    def test_no_mutation_readonly_noncontiguous_and_route_permutation(self):
        args = fixture()
        for role in range(4):
            args[role] = args[role][..., ::-1]
        args[7] = args[7][..., ::-1]
        before = [x.tobytes() for x in args]
        for x in args:
            x.setflags(write=False)
        expected = gradients(a.adjoint(*args))
        permuted = args.copy()
        permuted[4] = args[4][:, :, ::-1]
        for left, right in zip(gradients(a.adjoint(*permuted)), expected):
            np.testing.assert_allclose(left, right, rtol=3e-14, atol=3e-14)
        self.assertEqual(before, [x.tobytes() for x in args])

    def test_extreme_gate_derivative_is_stable_real_valued_slope(self):
        args = fixture()
        args[3].fill(40)
        result = a.adjoint(*args)
        for grad in gradients(result):
            self.assertTrue(np.isfinite(grad).all())
        self.assertTrue(np.any(result.dgate.array() != 0))
        args[3].fill(-40)
        np.testing.assert_allclose(result.dgate.array(), a.adjoint(*args).dgate.array(), rtol=0, atol=0)
        for value in (-1000, 1000):
            args[3].fill(value)
            np.testing.assert_array_equal(a.adjoint(*args).dgate.array(), 0)

    def test_all_input_roles_reject_subclasses_before_arithmetic(self):
        class Subclass(np.ndarray):
            pass

        for role in range(8):
            for kind in ("masked", "unmasked", "subclass"):
                args = fixture()
                if kind == "subclass":
                    args[role] = args[role].view(Subclass)
                else:
                    mask = np.zeros(args[role].shape, dtype=bool)
                    if kind == "masked":
                        mask.flat[0] = True
                    args[role] = np.ma.array(args[role], mask=mask)
                with self.subTest(role=role, kind=kind), self.assertRaisesRegex(ValueError, "Plain NumPy ndarrays"), patch.object(np, "zeros_like", side_effect=AssertionError("arithmetic entered")):
                    a.adjoint(*args)

    def test_invalid_float_dtype_nonfinite_and_cotangent_shape(self):
        for role in (0, 1, 2, 3, 7):
            for change in ("nan", "inf", "float32", "object"):
                args = fixture()
                if change in ("nan", "inf"):
                    args[role].flat[0] = float(change)
                else:
                    args[role] = args[role].astype(change)
                with self.subTest(role=role, change=change), self.assertRaises(ValueError):
                    a.adjoint(*args)
        args = fixture()
        args[7] = args[7].reshape(2, 2, 4, 2)
        with self.assertRaisesRegex(ValueError, "Cotangent must match"):
            a.adjoint(*args)

    def test_route_causality_unique_padding_and_geometry_gates(self):
        for change in ("duplicate", "future", "empty", "padded", "negative", "dtype", "gqa", "querygap"):
            args = fixture()
            if change == "duplicate": args[4][0, 0, 1] = 0
            elif change == "future": args[4][0, 0, 1] = 3
            elif change == "empty": args[4][0, 0].fill(-1)
            elif change == "padded": args[4][1, 1, 0] = 0
            elif change == "negative": args[4][0, 0, 0] = -2
            elif change == "dtype": args[4] = args[4].astype(np.int64)
            elif change == "gqa": args[0] = args[0][:, :, :3]
            else: args[5][0] = [0, 2]
            with self.subTest(change=change), self.assertRaises(ValueError):
                a.adjoint(*args)

    def test_input_gradient_and_work_bounds_precede_allocations(self):
        args = fixture()
        for name, limit in (("MAX_INPUT_BYTES", 1), ("MAX_GRADIENT_BYTES", 1), ("MAX_SELECTED_COMPONENTS", 1)):
            with self.subTest(name=name), patch.object(a, name, limit), patch.object(np, "zeros_like", side_effect=AssertionError("allocation entered")), self.assertRaisesRegex(ValueError, "bound exceeded"):
                a.adjoint(*args)
        result = a.adjoint(*args)
        self.assertEqual(result.gradient_bytes, sum(x.nbytes for x in args[:4]))
        self.assertEqual(result.selected_components, 7 * 4 * 2)

    def test_fixed_work_limit_refuses_real_oversized_fixture(self):
        q = np.zeros((1, 8, 24, 32))
        k = np.zeros((1, 64, 2, 32))
        routes = np.full((1, 8, 64), -1, np.int32)
        routes[:, :, :56] = np.arange(56)
        args = [q, k, k.copy(), q.copy(), routes, np.arange(56, 64, dtype=np.int64)[None], np.array([64], np.int64), np.zeros((1, 8, 768))]
        with patch.object(np, "zeros_like", side_effect=AssertionError("allocation entered")), self.assertRaisesRegex(ValueError, "selected component work bound exceeded"):
            a.adjoint(*args)

    def test_exact_count_bounds_and_actual_input_byte_refusal(self):
        args = fixture()
        input_bytes = sum(x.nbytes for x in args)
        grad_bytes = sum(x.nbytes for x in args[:4])
        work = 7 * 4 * 2
        with patch.object(a, "MAX_INPUT_BYTES", input_bytes), patch.object(a, "MAX_GRADIENT_BYTES", grad_bytes), patch.object(a, "MAX_SELECTED_COMPONENTS", work):
            self.assertEqual(a.adjoint(*args).selected_components, work)
        for name, count in (("MAX_INPUT_BYTES", input_bytes), ("MAX_GRADIENT_BYTES", grad_bytes), ("MAX_SELECTED_COMPONENTS", work)):
            with self.subTest(name=name), patch.object(a, name, count - 1), self.assertRaisesRegex(ValueError, "bound exceeded"):
                a.adjoint(*args)
        args[0] = np.broadcast_to(np.array(0.), (2, 8, 24, 1024))
        with patch.object(np, "isfinite", side_effect=AssertionError("input arithmetic entered")), self.assertRaisesRegex(ValueError, "input byte bound exceeded"):
            a.adjoint(*args)

    def test_overflow_refused_not_returned_as_gradients(self):
        args = fixture()
        args[2].fill(1e308)
        args[7].fill(1e308)
        with self.assertRaisesRegex(ValueError, "Nonfinite adjoint arithmetic"):
            a.adjoint(*args)

    def test_no_device_or_upstream_libraries_imported(self):
        prefixes = ("mlx", "mlx_lm", "torch", "triton", "exllamav3")
        self.assertFalse(any(name == p or name.startswith(p + ".") for p in prefixes for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
