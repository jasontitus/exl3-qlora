"""Independent CPU scalar-conv/projector-GDN composition and state-path checks."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_qwen_conv_gdn_composition as subject  # noqa: E402
from qwen_conv_independent_probe import scalar_oracle  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle(x, weight, history, initial, decay, beta):
    """Independent scalar conv then per-head matrix projector; no component helpers."""
    packed, ending = scalar_oracle(x, weight, history, None)
    batches, _, steps = packed.shape
    state = initial.copy()
    y = np.empty((batches, steps, 2, 2), dtype=np.float64)
    for b in range(batches):
        for t in range(steps):
            q = packed[b, :2, t]
            k = packed[b, 2:4, t]
            q = q / math.sqrt(math.fsum(float(a) ** 2 for a in q) + 1e-6)
            k = k / math.sqrt(math.fsum(float(a) ** 2 for a in k) + 1e-6)
            for head in range(2):
                v = packed[b, 4 + 2 * head : 6 + 2 * head, t]
                strength = float(beta[b, t, head])
                projector = math.exp(float(decay[b, t, head])) * (
                    np.eye(2) - strength * np.outer(k, k)
                )
                state[b, head] = projector @ state[b, head] + strength * np.outer(k, v)
                y[b, t, head] = (q @ state[b, head]) / math.sqrt(2)
    return y, ending, state


def run():
    inputs = [
        Path(subject.__file__),
        ROOT / "tools/qwen_conv_reference.py",
        ROOT / "tools/qwen_gdn_reference.py",
        Path(__file__),
        ROOT / "reports/qwen_conv_independent_probe.py",
    ]
    pins = {str(p): sha(p) for p in inputs}
    subject.conv.provenance()
    subject.gdn.source_provenance()
    checks = []

    def record(name, actual, expected, tol=1e-11):
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0))
        checks.append(
            {"name": name, "max_abs_error": error, "tolerance": tol, "passed": error <= tol}
        )

    for case, (batch, steps, width, cut) in enumerate(((1, 7, 4, 2), (2, 8, 8, 1), (1, 3, 4, 2))):
        rng = np.random.default_rng(7750 + case)
        x = rng.normal(scale=0.4, size=(batch, 8, steps))
        w = rng.normal(scale=0.4, size=(8, width))
        h = rng.normal(scale=0.3, size=(batch, 8, width - 1))
        s = rng.normal(scale=0.2, size=(batch, 2, 2, 2))
        g = -rng.uniform(0.1, 0.7, size=(batch, steps, 2))
        beta = rng.uniform(0.1, 0.9, size=(batch, steps, 2))
        args = [x, w, h, s, g, beta]
        expected = oracle(*args)
        actual = subject.forward(*args)[:3]
        for key, a, e in zip(("output", "conv_history", "gdn_state"), actual, expected):
            record(f"projector-forward-{case}-{key}", a, e)
        dy = np.zeros_like(actual[0])
        dy[:, -1] = rng.normal(size=dy[:, -1].shape)
        dh = rng.normal(scale=0.3, size=h.shape)
        ds = rng.normal(scale=0.3, size=s.shape)
        c, r = subject.backward(*args, dy, dh, ds)
        gradients = [c["x"], c["weight"], c["history"], r["initial_state"], r["g"], r["beta"]]
        for i, key in enumerate(("x", "weight", "history", "initial_state", "g", "beta")):
            direction = rng.normal(size=args[i].shape)
            direction /= np.linalg.norm(direction)
            values = []
            for sign in (1, -1):
                shifted = [a.copy() for a in args]
                shifted[i] += sign * 1e-6 * direction
                y, endh, ends = oracle(*shifted)
                values.append(float(np.sum(y * dy) + np.sum(endh * dh) + np.sum(ends * ds)))
            numeric = (values[0] - values[1]) / 2e-6
            analytic = float(np.sum(gradients[i] * direction))
            record(
                f"independent-directional-{case}-{key}",
                analytic,
                numeric,
                1e-7 + 1e-6 * abs(numeric),
            )
        prefix = [x[:, :, :cut], w, h, s, g[:, :cut], beta[:, :cut]]
        ya, ha, sa, _ = subject.forward(*prefix)
        suffix = [x[:, :, cut:], w, ha, sa, g[:, cut:], beta[:, cut:]]
        yb, hb, sb, _ = subject.forward(*suffix)
        for key, a, e in (
            ("output", np.concatenate((ya, yb), axis=1), actual[0]),
            ("conv_history", hb, actual[1]),
            ("gdn_state", sb, actual[2]),
        ):
            record(f"chunk-forward-{case}-{key}", a, e)
        cb, rb = subject.backward(*suffix, dy[:, cut:], dh, ds)
        ca, ra = subject.backward(*prefix, dy[:, :cut], cb["history"], rb["initial_state"])
        combined = [
            np.concatenate((ca["x"], cb["x"]), axis=2),
            ca["weight"] + cb["weight"],
            ca["history"],
            ra["initial_state"],
            np.concatenate((ra["g"], rb["g"]), axis=1),
            np.concatenate((ra["beta"], rb["beta"]), axis=1),
        ]
        for key, a, e in zip(
            ("x", "weight", "history", "initial_state", "g", "beta"), combined, gradients
        ):
            record(f"chunk-adjoint-{case}-{key}", a, e)
        # Deliberately sever exactly one boundary path. Each must spoil the gradient.
        for omitted in ("conv_history", "gdn_state"):
            wrong_c, wrong_r = subject.backward(
                *prefix,
                dy[:, :cut],
                np.zeros_like(ha) if omitted == "conv_history" else cb["history"],
                np.zeros_like(sa) if omitted == "gdn_state" else rb["initial_state"],
            )
            del wrong_r
            difference = float(np.linalg.norm(wrong_c["x"] - ca["x"]))
            checks.append(
                {
                    "name": f"detached-negative-control-{case}-{omitted}",
                    "input_gradient_difference_norm": difference,
                    "minimum_detectable_difference": 1e-7,
                    "passed": difference > 1e-7,
                }
            )
    assert all(sha(p) == digest for p, digest in pins.items()), "Input drift"
    assert not any(
        k == "torch" or k.startswith("torch.") or k == "mlx" or k.startswith("mlx.")
        for k in sys.modules
    )
    return {
        "format": "qwen-composition-independent-probe-v1",
        "inputs_sha256": pins,
        "checks": checks,
        "checks_passed": sum(c["passed"] for c in checks),
        "passed": all(c["passed"] for c in checks),
        "numpy_version": np.__version__,
        "gpu_used": False,
        "torch_imported": False,
        "mlx_imported": False,
        "scope": "Tiny continuous FP64 conv/SiLU+GDN composition, not a full layer or backend.",
    }


if __name__ == "__main__":
    result = run()
    with Path(sys.argv[1]).open("x") as out:
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
