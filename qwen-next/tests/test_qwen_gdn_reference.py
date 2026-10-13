"""Tiny synthetic CPU GDN; scalar recurrence and finite-difference adjoints."""

import importlib.util
import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qwen_gdn_reference", ROOT / "tools/qwen_gdn_reference.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def scalar_scan(q, k, v, g, beta, state=None, eps=1e-6):
    """Independent scalar equations: decay state FIRST, then its key projection."""
    bsz, steps, nk, dk = q.shape
    nv, dv = v.shape[2:]
    state = (
        np.zeros((bsz, nv, dk, dv))
        if state is None
        else state.astype(np.float64).copy()
    )
    output = np.empty((bsz, steps, nv, dv), dtype=np.float64)
    group = nv // nk
    for t in range(steps):
        for b in range(bsz):
            for h in range(nv):
                qraw, kraw = q[b, t, h // group], k[b, t, h // group]
                qnorm = [
                    float(a) / math.sqrt(math.fsum(float(x) ** 2 for x in qraw) + eps)
                    for a in qraw
                ]
                knorm = [
                    float(a) / math.sqrt(math.fsum(float(x) ** 2 for x in kraw) + eps)
                    for a in kraw
                ]
                decayed = [
                    [
                        float(state[b, h, r, c]) * math.exp(float(g[b, t, h]))
                        for c in range(dv)
                    ]
                    for r in range(dk)
                ]
                error = [
                    float(v[b, t, h, c])
                    - math.fsum(knorm[r] * decayed[r][c] for r in range(dk))
                    for c in range(dv)
                ]
                for r in range(dk):
                    for c in range(dv):
                        state[b, h, r, c] = (
                            decayed[r][c] + knorm[r] * float(beta[b, t, h]) * error[c]
                        )
                for c in range(dv):
                    output[b, t, h, c] = math.fsum(
                        float(state[b, h, r, c]) * qnorm[r] for r in range(dk)
                    ) / math.sqrt(dk)
    return output, state


def finite_difference(fn, value):
    result = np.empty_like(value)
    step = 1e-6
    for index in np.ndindex(value.shape):
        upper, lower = value.copy(), value.copy()
        upper[index] += step
        lower[index] -= step
        result[index] = (fn(upper) - fn(lower)) / (2 * step)
    return result


class QwenGDNTests(unittest.TestCase):
    def fixture(self, dtype=np.float64, steps=5, nk=2, group=3, dk=3, dv=2, batch=1):
        rng = np.random.default_rng(493)
        nv = nk * group
        values = [
            rng.normal(size=(batch, steps, nk, dk)),
            rng.normal(size=(batch, steps, nk, dk)),
            rng.normal(size=(batch, steps, nv, dv)),
            -rng.uniform(0.1, 1.5, size=(batch, steps, nv)),
            rng.uniform(0.15, 0.85, size=(batch, steps, nv)),
        ]
        state = rng.normal(0, 0.3, size=(batch, nv, dk, dv)).astype(dtype)
        return [x.astype(dtype) for x in values], state

    def test_independent_scalar_recurrence_f32_f64_and_unequal_head_dims(self):
        for dtype, precision, tol in (
            (np.float32, "float32", 2e-6),
            (np.float64, "float64", 2e-12),
        ):
            with self.subTest(dtype=dtype):
                values, state = self.fixture(dtype, batch=2)
                actual = m.recurrent_forward(
                    *values, initial_state=state, precision=precision
                )
                expected = scalar_scan(*values, state)
                for a, e in zip(actual, expected):
                    np.testing.assert_allclose(a, e, rtol=tol, atol=tol)
                    self.assertEqual(a.dtype, dtype)

    def test_recurrence_chunking_is_exact_with_state_carry(self):
        for dtype, precision in ((np.float32, "float32"), (np.float64, "float64")):
            values, state = self.fixture(dtype, steps=7)
            original = [x.copy() for x in values] + [state.copy()]
            expected, end = m.recurrent_forward(
                *values, initial_state=state, precision=precision
            )
            for boundaries in ((0, 1, 4, 7), (0, 3, 7), (0, 1, 2, 3, 4, 5, 6, 7)):
                carry = state
                outputs = []
                for lo, hi in itertools.pairwise(boundaries):
                    part, carry = m.recurrent_forward(
                        *(x[:, lo:hi] for x in values),
                        initial_state=carry,
                        precision=precision,
                    )
                    outputs.append(part)
                np.testing.assert_array_equal(np.concatenate(outputs, axis=1), expected)
                np.testing.assert_array_equal(carry, end)
            for a, b in zip(values + [state], original):
                np.testing.assert_array_equal(a, b)
            end.fill(123)
            self.assertFalse(np.shares_memory(end, state))
            np.testing.assert_array_equal(state, original[-1])

    def test_three_value_heads_repeat_each_key_head_contiguously(self):
        q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])
        k = np.array([[[[1.0, 0.0], [1.0, 0.0]]]])
        v = np.arange(1, 7, dtype=np.float64).reshape(1, 1, 6, 1)
        g = np.zeros((1, 1, 6))
        beta = np.ones((1, 1, 6))
        output, _ = m.recurrent_forward(q, k, v, g, beta, precision="float64")
        expected = (
            np.array([1, 2, 3, 0, 0, 0], dtype=np.float64) / (1 + 1e-6) / math.sqrt(2)
        )
        np.testing.assert_allclose(output.reshape(-1), expected, rtol=1e-14, atol=1e-14)

    def test_known_zero_update_decay_and_zero_query(self):
        values, state = self.fixture(steps=3, nk=1, group=1, dk=2, dv=3)
        values[-1].fill(0)
        _, end = m.recurrent_forward(*values, initial_state=state, precision="float64")
        np.testing.assert_allclose(
            end,
            state * np.exp(values[3].sum(axis=1))[..., None, None],
            rtol=1e-14,
            atol=1e-14,
        )
        values[0].fill(0)
        values[-1].fill(0.5)
        output, end = m.recurrent_forward(
            *values, initial_state=state, precision="float64"
        )
        np.testing.assert_array_equal(output, np.zeros_like(output))
        self.assertGreater(np.linalg.norm(end - state), 0.01)

    def test_all_recurrence_input_adjoints_against_scalar_finite_differences(self):
        values, state = self.fixture(steps=3, nk=1, group=2, dk=2, dv=2)
        rng = np.random.default_rng(381)
        dout = rng.normal(size=values[2].shape)
        dend = rng.normal(size=state.shape)
        analytic = m.recurrent_input_adjoint(
            *values, dout, initial_state=state, d_final_state=dend
        )
        for index, name in enumerate(("q", "k", "v", "g", "beta", "initial_state")):

            def objective(changed, index=index):
                args = values.copy()
                s = state
                if index == 5:
                    s = changed
                else:
                    args[index] = changed
                out, end = scalar_scan(*args, s)
                return np.sum(out * dout) + np.sum(end * dend)

            candidate = state if index == 5 else values[index]
            with self.subTest(input=name):
                np.testing.assert_allclose(
                    analytic[name],
                    finite_difference(objective, candidate),
                    rtol=3e-6,
                    atol=3e-8,
                )

    def test_late_loss_reaches_early_inputs_and_chunk_adjoint_composes(self):
        values, state = self.fixture(steps=5, nk=2, group=2, dk=2, dv=2)
        dout = np.zeros_like(values[2])
        dout[:, -2:] = (
            np.arange(dout[:, -2:].size).reshape(dout[:, -2:].shape) / 7 - 0.5
        )
        full = m.recurrent_input_adjoint(*values, dout, initial_state=state)
        split = 3
        prefix = [x[:, :split] for x in values]
        suffix = [x[:, split:] for x in values]
        _, middle = m.recurrent_forward(
            *prefix, initial_state=state, precision="float64"
        )
        later = m.recurrent_input_adjoint(
            *suffix, dout[:, split:], initial_state=middle
        )
        earlier = m.recurrent_input_adjoint(
            *prefix,
            dout[:, :split],
            initial_state=state,
            d_final_state=later["initial_state"],
        )
        for name in ("q", "k", "v", "g", "beta"):
            np.testing.assert_allclose(
                np.concatenate([earlier[name], later[name]], axis=1),
                full[name],
                rtol=2e-12,
                atol=2e-12,
            )
        np.testing.assert_allclose(
            earlier["initial_state"], full["initial_state"], rtol=2e-12, atol=2e-12
        )
        self.assertGreater(np.linalg.norm(full["v"][:, 0]), 1e-5)
        self.assertGreater(np.linalg.norm(full["k"][:, 0]), 1e-5)
        # Query reads state but does not update it: early queries have zero late-only adjoint.
        np.testing.assert_array_equal(
            full["q"][:, :split], np.zeros_like(full["q"][:, :split])
        )
        detached = m.recurrent_input_adjoint(
            *prefix, dout[:, :split], initial_state=state
        )
        np.testing.assert_array_equal(detached["v"], np.zeros_like(detached["v"]))

    def test_initial_state_cotangent_alone_and_optional_zero_initial_state(self):
        values, state = self.fixture(steps=2, nk=1, group=1, dk=2, dv=2)
        explicit = m.recurrent_forward(
            *values, initial_state=np.zeros_like(state), precision="float64"
        )
        implicit = m.recurrent_forward(*values, precision="float64")
        for a, b in zip(explicit, implicit):
            np.testing.assert_array_equal(a, b)
        grad = m.recurrent_input_adjoint(
            *values,
            np.zeros_like(values[2]),
            initial_state=state,
            d_final_state=np.ones_like(state),
        )
        self.assertGreater(np.linalg.norm(grad["v"]), 0.01)
        np.testing.assert_array_equal(grad["q"], np.zeros_like(values[0]))

    def test_split_qkv_is_contiguous_channel_layout_after_conv_only(self):
        packed = np.arange(2 * 3 * (2 * 2 * 2 + 6 * 3), dtype=np.float32).reshape(
            2, 3, -1
        )
        q, k, v = m.split_postconv_qkv(
            packed, num_k_heads=2, num_v_heads=6, key_dim=2, value_dim=3
        )
        np.testing.assert_array_equal(q.reshape(2, 3, -1), packed[..., :4])
        np.testing.assert_array_equal(k.reshape(2, 3, -1), packed[..., 4:8])
        np.testing.assert_array_equal(v.reshape(2, 3, -1), packed[..., 8:])
        self.assertEqual(v.shape, (2, 3, 6, 3))
        q.fill(-1)
        self.assertFalse(np.any(packed < 0))

    def test_gate_helpers_qwen_scale_one_and_stability(self):
        b = np.array([[[-1000.0, 0.0, 1000.0]]])
        a = np.array([[[-1000.0, 1.0, 1000.0]]])
        al = np.array([0.0, math.log(2), -1.0])
        dt = np.array([0.0, -0.5, 0.0])
        beta, g = m.qwen_beta_g(b, a, al, dt, precision="float64")
        np.testing.assert_array_equal(beta, [[[0, 0.5, 1]]])
        expected = np.array(
            [[[-0.0, -2 * math.log1p(math.exp(0.5)), -math.exp(-1) * 1000]]]
        )
        np.testing.assert_allclose(g, expected, rtol=1e-14, atol=1e-14)
        self.assertTrue((g <= 0).all())
        beta32, g32 = m.qwen_beta_g(b.astype(np.float16), a.astype(np.float16), al, dt)
        self.assertEqual(beta32.dtype, np.float32)
        self.assertEqual(g32.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.qwen_beta_g(b, a, np.array([1000.0, 0.0, 0.0]), dt)

    def test_sigmoid_output_norm_is_after_norm_without_one_plus_weight(self):
        core = np.array([[[[3.0, 4.0], [2.0, -1.0]]]])
        gate = np.array([[[[-2.0, 0.5], [3.0, -4.0]]]])
        weight = np.array([0.25, -2.0])
        actual = m.qwen_output_norm(core, gate, weight, precision="float64")
        expected = np.empty_like(core)
        for h in range(2):
            inv = 1 / math.sqrt(sum(core[0, 0, h] ** 2) / 2 + 1e-6)
            for j in range(2):
                expected[0, 0, h, j] = (
                    core[0, 0, h, j]
                    * inv
                    * weight[j]
                    / (1 + math.exp(-gate[0, 0, h, j]))
                )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertEqual(m.qwen_output_norm(core, gate, weight).dtype, np.float32)
        np.testing.assert_array_equal(
            m.qwen_output_norm(np.zeros_like(core), gate, weight),
            np.zeros_like(core, dtype=np.float32),
        )

    def test_noncontiguous_inputs_and_float16_promote_to_float32_state(self):
        values, _ = self.fixture(np.float16)
        strided = [x[:, ::-1] for x in values]
        expected = m.recurrent_forward(*(x.copy() for x in strided))
        actual = m.recurrent_forward(*strided)
        for a, b in zip(actual, expected):
            self.assertEqual(a.dtype, np.float32)
            np.testing.assert_array_equal(a, b)

    def test_fail_closed_shapes_bounds_state_precision_and_domains(self):
        values, state = self.fixture()
        for precision in ("bf16", None, True):
            with self.assertRaises(ValueError):
                m.recurrent_forward(*values, precision=precision)
        for index, replacement in (
            (0, values[0][:, :, :1]),
            (1, values[1].astype(np.int32)),
            (3, np.ones_like(values[3])),
            (4, np.full_like(values[4], 1.1)),
            (2, np.full_like(values[2], np.nan)),
        ):
            changed = values.copy()
            changed[index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError):
                m.recurrent_forward(*changed)
        for badstate in (state, state.astype(np.float16), state[:, :, :1]):
            with self.assertRaises(ValueError):
                m.recurrent_forward(*values, initial_state=badstate)
        for eps in (0, -1, True, float("nan")):
            with self.assertRaises(ValueError):
                m.recurrent_forward(*values, eps=eps)
        for badsteps in (0, 33):
            changed = [
                np.zeros((x.shape[0], badsteps, *x.shape[2:]), dtype=np.float32)
                for x in values
            ]
            with self.assertRaises(ValueError):
                m.recurrent_forward(*changed)
        with self.assertRaises(ValueError):
            m.recurrent_input_adjoint(
                *[x.astype(np.float32) for x in values], np.zeros_like(values[2])
            )
        with self.assertRaises(ValueError):
            m.recurrent_input_adjoint(
                *values,
                np.zeros_like(values[2]),
                initial_state=state,
                d_final_state=np.zeros_like(state, dtype=np.float32),
            )

    def test_finite_overflow_does_not_become_plausible_state(self):
        values, _ = self.fixture(np.float32)
        values[0].fill(1e30)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.recurrent_forward(*values)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.qwen_output_norm(
                np.full((1, 1, 1, 2), 1e30, dtype=np.float32),
                np.zeros((1, 1, 1, 2)),
                np.ones(2),
            )

    def test_epsilon_rejects_precision_overflow_and_underflow_to_zero(self):
        smallest32 = float(np.nextafter(np.float32(0), np.float32(1)))
        for dtype, precision, invalid in (
            (np.float32, "float32", (1e100, smallest32 / 4)),
            (np.float64, "float64", (10**400, float("inf"), 0.0)),
        ):
            values, state = self.fixture(dtype, steps=1, nk=1, group=1, dk=2, dv=2)
            core = np.ones((1, 1, 1, 2), dtype=dtype)
            for eps in invalid:
                with self.subTest(precision=precision, eps=eps):
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        m.recurrent_forward(
                            *values, initial_state=state, precision=precision, eps=eps
                        )
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        m.qwen_output_norm(
                            core,
                            np.zeros_like(core),
                            np.ones(2, dtype=dtype),
                            precision=precision,
                            eps=eps,
                        )

    def test_finite_epsilon_endpoints_work_when_denominator_is_finite(self):
        for dtype, precision in ((np.float32, "float32"), (np.float64, "float64")):
            for eps in (
                float(np.nextafter(dtype(0), dtype(1))),
                float(np.finfo(dtype).max),
            ):
                with self.subTest(precision=precision, eps=eps):
                    shape = (1, 1, 1, 2)
                    qkv = [np.zeros(shape, dtype=dtype) for _ in range(3)]
                    gates = np.zeros((1, 1, 1), dtype=dtype)
                    output, state = m.recurrent_forward(
                        *qkv, gates, gates, precision=precision, eps=eps
                    )
                    np.testing.assert_array_equal(output, np.zeros_like(output))
                    np.testing.assert_array_equal(state, np.zeros_like(state))
                    normalized = m.qwen_output_norm(
                        np.ones(shape, dtype=dtype),
                        np.zeros(shape, dtype=dtype),
                        np.ones(2, dtype=dtype),
                        precision=precision,
                        eps=eps,
                    )
                    self.assertTrue(np.isfinite(normalized).all())
                    self.assertTrue((normalized > 0).all())

    def test_finite_epsilon_plus_finite_squared_input_must_not_overflow(self):
        for dtype, precision in ((np.float32, "float32"), (np.float64, "float64")):
            with self.subTest(precision=precision):
                eps = float(np.finfo(dtype).max)
                magnitude = dtype(math.sqrt(eps) / 2)
                values, state = self.fixture(dtype, steps=1, nk=1, group=1, dk=2, dv=2)
                values[0].fill(magnitude)
                self.assertTrue(np.isfinite(np.sum(values[0] ** 2, axis=-1)).all())
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    m.recurrent_forward(
                        *values, initial_state=state, precision=precision, eps=eps
                    )
                core = np.full((1, 1, 1, 2), magnitude, dtype=dtype)
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    m.qwen_output_norm(
                        core,
                        np.zeros_like(core),
                        np.ones(2, dtype=dtype),
                        precision=precision,
                        eps=eps,
                    )

    def test_extreme_normalization_adjoint(self):
        q = np.array([[[[1e110, 2e110]]]])
        k = np.array([[[[1., -1.]]]])
        v = np.array([[[[3.]]]])
        g = np.array([[[-0.5]]])
        beta = np.array([[[0.5]]])
        dy = np.ones_like(v)
        grads = m.recurrent_input_adjoint(q, k, v, g, beta, dy)
        self.assertAlmostEqual(float(np.sum(grads["q"] * q)), 0., places=12)
        # The true zero-query derivative remains finite at subnormal epsilon.
        tiny = float(np.nextafter(0., 1.))
        grads = m.recurrent_input_adjoint(np.zeros_like(q), k, v, g, beta, dy, eps=tiny)
        self.assertTrue(np.isfinite(grads["q"]).all())
        self.assertGreater(float(np.linalg.norm(grads["q"] / 1e161)), 0.)

    def test_pinned_source_provenance_and_tamper(self):
        provenance = m.source_provenance()
        self.assertEqual(provenance["target_config"]["linear_num_key_heads"], 16)
        self.assertEqual(provenance["target_config"]["linear_num_value_heads"], 48)
        self.assertFalse(provenance["convolution_implemented"])
        self.assertFalse(provenance["production_backward_qualified"])
        with tempfile.TemporaryDirectory() as temp:
            for name in m.SOURCE_SHA256:
                path = Path(temp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((m.SOURCE_ROOT / name).read_bytes())
            path = Path(temp) / "exllamav3/training/gdn.py"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                m.source_provenance(temp)


if __name__ == "__main__":
    unittest.main()
