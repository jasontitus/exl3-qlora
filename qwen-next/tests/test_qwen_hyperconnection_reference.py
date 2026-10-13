"""CPU-only Qwen gates; scalar formula oracle and continuous finite differences."""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qwen_hyperconnection_reference", ROOT / "tools/qwen_hyperconnection_reference.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def scalar_sigmoid(v):
    return 1 / (1 + math.exp(-v)) if v >= 0 else math.exp(v) / (1 + math.exp(v))


def scalar_oracle(x, mixer):
    """Separately expressed scalar formula; no NumPy dot/reduction or helper reuse."""
    batches, tokens, _, width = x.shape
    result = np.zeros((batches, tokens, width), dtype=np.float64)
    post = (
        np.zeros((batches, tokens, 4), dtype=np.float64)
        if mixer.inject is not None
        else None
    )
    rank = mixer.down.shape[0]
    for b in range(batches):
        for t in range(tokens):
            z = []
            for h in range(4):
                inv = 1 / math.sqrt(
                    math.fsum(float(v) ** 2 for v in x[b, t, h]) / width + mixer.eps
                )
                z.extend(
                    float(x[b, t, h, d]) * inv * float(mixer.norm_w[h, d])
                    for d in range(width)
                )
            low = []
            for r in range(rank):
                a = (
                    math.fsum(z[j] * float(mixer.down[r, j]) for j in range(4 * width))
                    / 4
                )
                low.append(a * scalar_sigmoid(a))
            gates = [
                scalar_sigmoid(
                    math.fsum(low[r] * float(mixer.up[j, r]) for r in range(rank))
                )
                for j in range(4 * width)
            ]
            for d in range(width):
                result[b, t, d] = (
                    math.fsum(z[h * width + d] * gates[h * width + d] for h in range(4))
                    / 4
                )
            if post is not None:
                for h in range(4):
                    post[b, t, h] = 2 * scalar_sigmoid(
                        math.fsum(
                            z[j] * float(mixer.inject[h, j]) for j in range(4 * width)
                        )
                        / 4
                    )
    return post, result


def difference(f, x):
    result = np.empty_like(x)
    step = 1e-6
    for index in np.ndindex(x.shape):
        upper, lower = x.copy(), x.copy()
        upper[index] += step
        lower[index] -= step
        result[index] = (f(upper) - f(lower)) / (2 * step)
    return result


class QwenHyperconnectionTests(unittest.TestCase):
    def fixture(self, site=True, width=3, rank=2, dtype=np.float64):
        rng = np.random.default_rng(918)
        mixer = m.QwenGatedResidual(
            (1 + rng.normal(0, 0.2, (4, width))).astype(dtype),
            rng.normal(0, 0.6, (rank, 4 * width)).astype(dtype),
            rng.normal(0, 0.7, (4 * width, rank)).astype(dtype),
            rng.normal(0, 0.8, (4, 4 * width)).astype(dtype) if site else None,
            eps=1e-6,
        )
        x = rng.normal(0, 2, (2, 3, 4, width)).astype(dtype)
        return mixer, x

    def test_scalar_oracle_site_final_and_gate_order(self):
        for site in (False, True):
            for dtype, tolerance in (
                (np.float64, 2e-12),
                (np.float32, 2e-6),
                (np.float16, 2e-6),
            ):
                with self.subTest(site=site, dtype=dtype):
                    mixer, x = self.fixture(site=site, dtype=dtype)
                    post, mixed = mixer.mix(x)
                    expected_post, expected = scalar_oracle(x, mixer)
                    np.testing.assert_allclose(
                        mixed, expected, rtol=tolerance, atol=tolerance
                    )
                    self.assertEqual(
                        mixed.dtype, np.float64 if dtype == np.float64 else np.float32
                    )
                    self.assertFalse(
                        np.allclose(mixed, x.astype(np.float64).mean(axis=2))
                    )
                    if site:
                        np.testing.assert_allclose(
                            post, expected_post, rtol=tolerance, atol=tolerance
                        )
                        self.assertEqual(post.dtype, mixed.dtype)
                        self.assertGreater(np.ptp(post), 0.1)
                    else:
                        self.assertIsNone(post)

    def test_zero_gate_known_answer_per_stream_norm_and_raw_residual(self):
        norm = np.array([[1, 2], [2, 1], [-1, 0.5], [0.25, -2]], dtype=np.float64)
        mixer = m.QwenGatedResidual(
            norm, np.zeros((1, 8)), np.zeros((8, 1)), np.zeros((4, 8))
        )
        x = np.array([[[[3.0, 4.0], [0.0, 1.0], [2.0, -2.0], [0.0, 0.0]]]])
        post, mixed = mixer.mix(x)
        expected = []
        for d in range(2):
            expected.append(
                sum(
                    float(x[0, 0, h, d])
                    * float(norm[h, d])
                    / math.sqrt(sum(v * v for v in x[0, 0, h]) / 2 + 1e-6)
                    for h in range(4)
                )
                / 8
            )
        np.testing.assert_array_equal(post, np.ones((1, 1, 4)))
        np.testing.assert_allclose(mixed[0, 0], expected, rtol=1e-14, atol=1e-14)
        y = np.array([[[7.0, -3.0]]])
        before = x.copy()
        out = m.residual_apply(x, y, post)
        np.testing.assert_array_equal(out, x + y[:, :, None, :])
        np.testing.assert_array_equal(x, before)
        self.assertFalse(np.shares_memory(x, out))

    def test_raw_checkpoint_tables_add_one_once_and_cast_resident_tables(self):
        mixer, _ = self.fixture()
        raw = np.linspace(-0.2, 0.3, 12, dtype=np.float64)
        loaded = m.QwenGatedResidual.from_checkpoint_tables(
            raw, mixer.down, mixer.up, mixer.inject
        )
        np.testing.assert_array_equal(
            loaded.norm_w, (raw.astype(np.float32) + np.float32(1)).reshape(4, 3)
        )
        for field in ("down", "up", "inject"):
            self.assertEqual(getattr(loaded, field).dtype, np.float16)
            np.testing.assert_array_equal(
                getattr(loaded, field), getattr(mixer, field).astype(np.float16)
            )
        direct = m.QwenGatedResidual(
            loaded.norm_w, loaded.down, loaded.up, loaded.inject
        )
        np.testing.assert_array_equal(direct.norm_w, loaded.norm_w)

    def test_expand_copies_fp32_and_feature_mean_is_not_final_mixer(self):
        hidden = np.arange(12, dtype=np.float16).reshape(2, 2, 3) / 3
        stack = m.expand_streams(hidden)
        self.assertEqual(stack.dtype, np.float32)
        np.testing.assert_array_equal(m.streams_mean(stack), hidden.astype(np.float32))
        stack[0, 0, 0, 0] = 99
        self.assertNotEqual(stack[0, 0, 1, 0], 99)
        self.assertNotEqual(hidden[0, 0, 0], 99)

    def test_noncontiguous_batch_and_token_independence(self):
        mixer, x = self.fixture()
        x = x[:, ::-1, :, ::-1]
        self.assertFalse(x.flags.c_contiguous)
        post, mixed = mixer.mix(x)
        for b in range(2):
            for t in range(3):
                isolated = mixer.mix(x[b : b + 1, t : t + 1].copy())
                np.testing.assert_allclose(
                    isolated[0][0, 0], post[b, t], rtol=1e-14, atol=1e-14
                )
                np.testing.assert_allclose(
                    isolated[1][0, 0], mixed[b, t], rtol=1e-14, atol=1e-14
                )

    def test_continuous_mix_adjoint_site_final_and_nonuniform_cotangents(self):
        rng = np.random.default_rng(291)
        for site in (False, True):
            mixer, x = self.fixture(site=site, width=2)
            x = x[:1, :2].copy()
            dm = rng.normal(size=(1, 2, 2))
            dp = rng.normal(size=(1, 2, 4)) if site else None
            analytic = mixer.input_adjoint(x, dm, dp)

            def objective(z, mixer=mixer, dm=dm, dp=dp, site=site):
                post, mixed = scalar_oracle(z, mixer)
                return np.sum(dm * mixed) + (np.sum(dp * post) if site else 0)

            numerical = difference(objective, x)
            np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-8)
            self.assertEqual(analytic.dtype, np.float64)

    def test_composed_residual_input_and_sublayer_output_adjoint(self):
        mixer, x = self.fixture(width=2)
        x = x[:1, :1].copy()
        y = np.array([[[0.4, -0.7]]])
        dm = np.array([[[0.9, -0.3]]])
        dout = np.arange(8, dtype=np.float64).reshape(1, 1, 4, 2) / 5 - 0.4
        post, _ = mixer.mix(x)
        dx, dy, dp = m.residual_input_adjoint(x, y, post, dout)
        dx += mixer.input_adjoint(x, dm, dp)

        def objective(z, yy):
            p, mixed = scalar_oracle(z, mixer)
            return np.sum(mixed * dm) + np.sum(
                (z + p[..., None] * yy[..., None, :]) * dout
            )

        np.testing.assert_allclose(
            dx, difference(lambda z: objective(z, y), x), rtol=2e-7, atol=2e-8
        )
        np.testing.assert_allclose(
            dy, difference(lambda yy: objective(x, yy), y), rtol=2e-7, atol=2e-8
        )

    def test_zero_stream_stability_and_saturated_gates(self):
        mixer, x = self.fixture(dtype=np.float32)
        post, mixed = mixer.mix(np.zeros_like(x))
        np.testing.assert_array_equal(post, np.ones_like(post))
        np.testing.assert_array_equal(mixed, np.zeros_like(mixed))
        extreme = m.QwenGatedResidual(
            mixer.norm_w, mixer.down * 1e4, mixer.up * 1e4, mixer.inject * 1e4
        )
        post, mixed = extreme.mix(x)
        self.assertTrue(np.isfinite(mixed).all())
        self.assertTrue(np.isfinite(post).all())
        self.assertTrue(((post >= 0) & (post <= 2)).all())

    def test_weight_copies_cannot_be_mutated_or_reenabled(self):
        norm = np.ones((4, 2))
        mixer = m.QwenGatedResidual(norm, np.ones((1, 8)), np.ones((8, 1)), None)
        norm[:] = 99
        np.testing.assert_array_equal(mixer.norm_w, np.ones((4, 2)))
        with self.assertRaises(ValueError):
            mixer.norm_w.flags.writeable = True
        with self.assertRaises(ValueError):
            mixer.down[0, 0] = 4

    def test_bad_geometry_dtype_eps_and_nonfinite_rejected(self):
        mixer, x = self.fixture()
        for bad in (
            x[:, :, :3],
            x.astype(np.int32),
            np.full_like(x, np.nan),
            np.zeros((1, 1, 4, 65)),
            np.zeros((1, 33, 4, 2)),
        ):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                mixer.mix(bad)
        for eps in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                m.QwenGatedResidual(
                    mixer.norm_w, mixer.down, mixer.up, mixer.inject, eps
                )
        for down, up, inject in (
            (mixer.down.T, mixer.up, None),
            (mixer.down, mixer.up.T, None),
            (mixer.down, mixer.up, np.zeros((4, 4))),
        ):
            with self.assertRaises(ValueError):
                m.QwenGatedResidual(mixer.norm_w, down, up, inject)
        with self.assertRaises(ValueError):
            m.QwenGatedResidual.from_checkpoint_tables(
                mixer.norm_w, mixer.down * 1e10, mixer.up
            )

    def test_adjoint_rejects_rounded_dtype_and_final_post_cotangent(self):
        mixer, x = self.fixture(site=False)
        dm = np.zeros((2, 3, 3))
        with self.assertRaisesRegex(ValueError, "float64"):
            mixer.input_adjoint(x.astype(np.float32), dm)
        with self.assertRaisesRegex(ValueError, "no post"):
            mixer.input_adjoint(x, dm, np.zeros((2, 3, 4)))
        with self.assertRaises(ValueError):
            mixer.input_adjoint(x, dm.astype(np.float32))

    def test_fp32_overflow_fails_instead_of_returning_plausible_zero(self):
        mixer, x = self.fixture(dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            mixer.mix(np.full_like(x, 1e30))

    def test_pinned_sources_config_and_source_tampering(self):
        receipt = m.source_provenance()
        self.assertEqual(receipt["target_config_dimensions"]["hc_count"], 4)
        self.assertFalse(receipt["gpu_parity_qualified"])
        self.assertFalse(receipt["end_to_end_model_supported"])
        with tempfile.TemporaryDirectory() as tmp:
            for name in m.SOURCE_SHA256:
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((m.SOURCE_ROOT / name).read_bytes())
            victim = Path(tmp) / "exllamav3/training/hyperconnections.py"
            victim.write_bytes(victim.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                m.source_provenance(tmp)


if __name__ == "__main__":
    unittest.main()
