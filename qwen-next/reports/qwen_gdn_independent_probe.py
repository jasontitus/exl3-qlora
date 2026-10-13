"""CPU-only independent projector-form GDN checks; never imports a GPU backend."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/qwen_gdn_reference.py"
source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location("reviewed_gdn", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def oracle(q, k, v, g, beta, initial, eps=1e-6):
    """Separate formulation: state←exp(g)(I−βkkᵀ)state+βkvᵀ."""
    b, steps, nk, width = q.shape
    nv, dv = v.shape[-2:]
    result = np.zeros_like(v, dtype=np.float64)
    state = initial.copy()
    for t in range(steps):
        for bi in range(b):
            for h in range(nv):
                q0, k0 = q[bi, t, h // (nv // nk)], k[bi, t, h // (nv // nk)]
                q1 = q0 / math.sqrt(math.fsum(float(a) ** 2 for a in q0) + eps)
                k1 = k0 / math.sqrt(math.fsum(float(a) ** 2 for a in k0) + eps)
                strength = float(beta[bi, t, h])
                transition = math.exp(float(g[bi, t, h])) * (
                    np.eye(width) - strength * np.outer(k1, k1)
                )
                state[bi, h] = transition @ state[bi, h] + strength * np.outer(k1, v[bi, t, h])
                result[bi, t, h] = (q1 @ state[bi, h]) / math.sqrt(width)
    return result, state


rows = []
for seed, shape, scale, eps in (
    (887, (2, 7, 2, 6, 3, 2), 1.0, 1e-6),
    (143, (1, 5, 1, 3, 2, 4), 1e-5, 1e-6),
    (298, (2, 3, 2, 4, 4, 2), 0.2, 0.03),
):
    rng = np.random.default_rng(seed)
    b, t, nk, nv, dk, dv = shape
    values = [
        rng.normal(size=(b, t, nk, dk)) * scale,
        rng.normal(size=(b, t, nk, dk)) * scale,
        rng.normal(size=(b, t, nv, dv)),
        -rng.uniform(0.1, 2, (b, t, nv)),
        rng.uniform(0.1, 0.9, (b, t, nv)),
    ]
    state = rng.normal(size=(b, nv, dk, dv)) * 0.2
    do, ds = rng.normal(size=values[2].shape), rng.normal(size=state.shape)
    expected = oracle(*values, state, eps)
    for precision, dtype, tolerance in [
        ("float64", np.float64, 4e-12),
        ("float32", np.float32, 3e-6),
    ]:
        cast = [a.astype(dtype) for a in values]
        actual = m.recurrent_forward(
            *cast, initial_state=state.astype(dtype), precision=precision, eps=eps
        )
        errors = [float(np.max(np.abs(a - e))) for a, e in zip(actual, expected)]
        passed = all(
            np.allclose(a, e, rtol=tolerance, atol=tolerance) for a, e in zip(actual, expected)
        )
        rows.append(
            {
                "kind": "independent_projector_forward",
                "seed": seed,
                "precision": precision,
                "eps": eps,
                "shape": shape,
                "max_abs_output_error": errors[0],
                "max_abs_state_error": errors[1],
                "passed": passed,
            }
        )
    grads = m.recurrent_input_adjoint(*values, do, initial_state=state, d_final_state=ds, eps=eps)
    for index, name in enumerate(("q", "k", "v", "g", "beta", "initial_state")):
        x = state if index == 5 else values[index]
        direction = rng.normal(size=x.shape)
        direction *= max(float(np.linalg.norm(x)) / math.sqrt(x.size), 1e-3) / np.linalg.norm(
            direction
        )

        def objective(factor):
            args = values.copy()
            carry = state
            if index == 5:
                carry = state + factor * direction
            else:
                args[index] = args[index] + factor * direction
            output, end = oracle(*args, carry, eps)
            return float(np.sum(output * do) + np.sum(end * ds))

        step = 1e-5
        fd = (objective(step) - objective(-step)) / (2 * step)
        predicted = float(np.sum(grads[name] * direction))
        error = abs(predicted - fd)
        tolerance = 3e-8 + 3e-6 * abs(fd)
        rows.append(
            {
                "kind": "independent_projector_directional_vjp",
                "seed": seed,
                "input": name,
                "eps": eps,
                "predicted": predicted,
                "finite_difference": fd,
                "abs_error": error,
                "tolerance": tolerance,
                "passed": error <= tolerance,
            }
        )

q = np.array([[[[1e110, 2e110]]]], dtype=np.float64)
k = np.array([[[[1.0, -1.0]]]], dtype=np.float64)
v = np.array([[[[3.0]]]], dtype=np.float64)
g = np.array([[[-0.5]]], dtype=np.float64)
beta = np.array([[[0.5]]], dtype=np.float64)
state = np.zeros((1, 1, 2, 1), dtype=np.float64)
for name, index in [("q", 0), ("k", 1)]:
    values = [q.copy(), k.copy(), v, g, beta]
    if name == "k":
        values[0] = k.copy()
        values[1] = q.copy()
    gradients = m.recurrent_input_adjoint(*values, np.ones_like(v))
    x = values[index]

    def radial(factor):
        args = values.copy()
        args[index] = x * factor
        return float(oracle(*args, state)[0].sum())

    fd = (radial(1 + 1e-5) - radial(1 - 1e-5)) / (2e-5)
    predicted = float(np.sum(gradients[name] * x))
    rows.append(
        {
            "kind": "large_finite_normalization_radial",
            "input": name,
            "max_abs_input": float(abs(x).max()),
            "predicted": predicted,
            "finite_difference": fd,
            "passed": abs(predicted - fd) <= 1e-8,
        }
    )
eps = float(np.nextafter(np.float64(0), np.float64(1)))
try:
    values = [np.zeros_like(q), k, v, g, beta]
    gradients = m.recurrent_input_adjoint(*values, np.ones_like(v), eps=eps)
    _, end = oracle(*values, state, eps)
    expected = end[:, 0, :, 0].reshape(q.shape) / math.sqrt(2) / math.sqrt(eps)
    passed = bool(
        np.isfinite(gradients["q"]).all()
        and np.allclose(gradients["q"], expected, rtol=3e-15, atol=0)
    )
    row = {
        "kind": "smallest_finite_epsilon_zero_query_adjoint",
        "eps": eps,
        "passed": passed,
        "max_abs_gradient": float(np.abs(gradients["q"]).max()),
    }
except ValueError as error:
    row = {
        "kind": "smallest_finite_epsilon_zero_query_adjoint",
        "eps": eps,
        "passed": False,
        "error": str(error),
    }
rows.append(row)
assert not any(name == "mlx" or name.startswith("mlx.") or name == "torch" for name in sys.modules)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha, "Source drift"
print(
    json.dumps(
        {
            "format": "qwen-gdn-independent-probe-v1",
            "source_sha256": source_sha,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy": np.__version__,
            "checks": rows,
            "passed": all(r["passed"] for r in rows),
            "source_provenance": m.source_provenance(),
        },
        indent=2,
    )
)
