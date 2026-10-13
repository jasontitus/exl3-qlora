"""Independent CPU-only scalar/history checks for the tiny Qwen conv oracle."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import qwen_conv_reference as subject  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_oracle(x, weight, history, bias):
    """Direct time-index definition; no subject forward/helper or concat windows."""
    bsz, channels, steps = x.shape
    width = weight.shape[1]
    output = np.empty_like(x)
    ending = np.empty_like(history)

    def raw(batch, channel, time):
        if time < 0:
            return float(history[batch, channel, time + width - 1])
        return float(x[batch, channel, time])

    for b in range(bsz):
        for c in range(channels):
            for t in range(steps):
                z = math.fsum(
                    raw(b, c, t - width + 1 + j) * float(weight[c, j]) for j in range(width)
                )
                z += 0.0 if bias is None else float(bias[c])
                sig = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                output[b, c, t] = z * sig
            for j in range(width - 1):
                ending[b, c, j] = raw(b, c, steps - width + 1 + j)
    return output, ending


def run():
    source_path = Path(subject.__file__)
    source_before = sha(source_path)
    provenance = subject.provenance()
    rng = np.random.default_rng(882199)
    checks = []

    def record(name, error, tolerance, **extra):
        checks.append(
            {
                "name": name,
                "max_abs_error": float(error),
                "tolerance": tolerance,
                "passed": bool(error <= tolerance),
                **extra,
            }
        )

    # K=1, shorter-than-history, exact-history and longer chunks; positive/negative strides.
    shapes = [
        (1, 1, 1, 1, False),
        (2, 3, 1, 8, True),
        (1, 2, 3, 4, True),
        (2, 2, 9, 4, False),
        (2, 3, 32, 8, True),
    ]
    for case, (b, c, t, k, with_bias) in enumerate(shapes):
        x = rng.normal(scale=0.35, size=(b, c, t * 2))[:, :, ::2]
        w = rng.normal(scale=0.4, size=(c, k))[:, ::-1]
        h = rng.normal(scale=0.3, size=(b, c, k - 1))[:, :, ::-1]
        bias = rng.normal(scale=0.2, size=c) if with_bias else None
        args = [x, w, h, bias]
        saved = [None if v is None else v.copy() for v in args]
        expected, expected_end = scalar_oracle(*args)
        actual, actual_end = subject.forward(*args)
        record(
            f"scalar-forward-{case}",
            np.max(np.abs(expected - actual)),
            2e-14,
            shape=[b, c, t, k],
            bias=with_bias,
        )
        record(f"raw-history-{case}", np.max(np.abs(expected_end - actual_end), initial=0), 0)
        dy = rng.normal(size=x.shape)
        dend = rng.normal(size=h.shape)
        gradients = subject.vjp(x, w, h, dy, dend, bias)
        for index, name in enumerate(("x", "weight", "history", "bias")):
            if args[index] is None or args[index].size == 0:
                continue
            direction = rng.normal(size=args[index].shape)
            direction /= np.linalg.norm(direction)
            values = []
            eps = 2e-6
            for sign in (1, -1):
                shifted = [None if a is None else a.copy() for a in args]
                shifted[index] += sign * eps * direction
                y, end = scalar_oracle(*shifted)
                values.append(float(np.sum(y * dy) + np.sum(end * dend)))
            numeric = (values[0] - values[1]) / (2 * eps)
            analytical = float(np.sum(gradients[name] * direction))
            record(
                f"scalar-directional-{case}-{name}",
                abs(numeric - analytical),
                2e-8,
                numerical=numeric,
                analytical=analytical,
            )
        record(
            f"no-mutation-{case}",
            max(
                (
                    np.max(np.abs(a - old), initial=0)
                    for a, old in zip(args, saved)
                    if a is not None
                ),
                default=0,
            ),
            0,
        )

    # State-only loss must reach old history when this chunk is shorter than K-1.
    x = rng.normal(size=(1, 2, 1))
    h = rng.normal(size=(1, 2, 7))
    w = rng.normal(size=(2, 8))
    dend = rng.normal(size=h.shape)
    got = subject.vjp(x, w, h, np.zeros_like(x), dend)
    dh = np.zeros_like(h)
    dh[:, :, 1:] = dend[:, :, :-1]
    record("end-only-old-history-adjoint", np.max(np.abs(got["history"] - dh)), 0)
    record("end-only-current-input-adjoint", np.max(np.abs(got["x"] - dend[:, :, -1:])), 0)
    record("end-only-weight-adjoint", np.max(np.abs(got["weight"])), 0)

    # Compose several one-token chunks, carrying the actual end-history cotangent.
    x = rng.normal(scale=0.3, size=(1, 2, 5))
    w = rng.normal(scale=0.2, size=(2, 8))
    initial = rng.normal(scale=0.3, size=(1, 2, 7))
    dy = np.zeros_like(x)
    dy[:, :, -1] = [0.7, -0.4]
    dend = rng.normal(scale=0.3, size=initial.shape)
    expected = subject.vjp(x, w, initial, dy, dend)
    states = [initial]
    outputs = []
    for i in range(x.shape[-1]):
        y, state = subject.forward(x[:, :, i : i + 1], w, states[-1])
        outputs.append(y)
        states.append(state)
    carry = dend
    dx = np.empty_like(x)
    dw = np.zeros_like(w)
    for i in reversed(range(x.shape[-1])):
        grad = subject.vjp(x[:, :, i : i + 1], w, states[i], dy[:, :, i : i + 1], carry)
        dx[:, :, i : i + 1] = grad["x"]
        dw += grad["weight"]
        carry = grad["history"]
    oracle_y, oracle_end = scalar_oracle(x, w, initial, None)
    record(
        "one-token-chunk-forward",
        np.max(np.abs(np.concatenate(outputs, axis=-1) - oracle_y)),
        2e-14,
    )
    record("one-token-chunk-end", np.max(np.abs(states[-1] - oracle_end)), 0)
    for key, actual in (("x", dx), ("weight", dw), ("history", carry)):
        record(f"one-token-chunk-adjoint-{key}", np.max(np.abs(expected[key] - actual)), 2e-14)

    # Check inference's full K-column convention independently: column0 is unused.
    state = rng.normal(size=(1, 2, 4))
    x = rng.normal(scale=0.3, size=(1, 2, 2))
    w = rng.normal(scale=0.3, size=(2, 4))
    reference = np.empty_like(x)
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            full = list(state[b, c]) + list(x[b, c])
            for t in range(x.shape[-1]):
                z = math.fsum(float(full[t + j + 1]) * float(w[c, j]) for j in range(4))
                reference[b, c, t] = z / (1 + math.exp(-z))
    actual, end = subject.forward(x, w, state[:, :, 1:])
    record("inference-K-column-output-mapping", np.max(np.abs(actual - reference)), 2e-14)
    full_end = np.concatenate((state, x), axis=-1)[:, :, -4:]
    record("inference-K-column-history-mapping", np.max(np.abs(end - full_end[:, :, 1:])), 0)

    # Saturated SiLU regimes must stay finite and approach the correct derivative.
    x = np.array([[[-1000.0, 1000.0]]])
    w = np.ones((1, 1))
    h = np.empty((1, 1, 0))
    y, _ = subject.forward(x, w, h)
    g = subject.vjp(x, w, h, np.ones_like(x))
    record("saturated-silu-output", np.max(np.abs(y - np.array([[[0.0, 1000.0]]]))), 0)
    record("saturated-silu-input-adjoint", np.max(np.abs(g["x"] - np.array([[[0.0, 1.0]]]))), 0)

    forbidden = [
        m
        for m in sys.modules
        if m == "torch" or m.startswith("torch.") or m == "mlx" or m.startswith("mlx.")
    ]
    assert not forbidden, forbidden
    assert source_before == sha(source_path), "Implementation drift during review"
    return {
        "format": "qwen-conv-independent-probe-v1",
        "source_sha256": source_before,
        "script_sha256": sha(__file__),
        "source_provenance": provenance,
        "numpy_version": np.__version__,
        "gpu_used": False,
        "torch_imported": False,
        "mlx_imported": False,
        "checks": checks,
        "checks_passed": sum(c["passed"] for c in checks),
        "passed": all(c["passed"] for c in checks),
        "scope": "Tiny FP64 CPU mathematical conv/SiLU/history contract only; no serving or full-model qualification.",
    }


if __name__ == "__main__":
    result = run()
    destination = Path(sys.argv[1])
    with destination.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": len(result["checks"]),
                "checks_passed": result["checks_passed"],
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
