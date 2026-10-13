"""Measure nested-boundary drift after exact-parity tests failed; no auto-pass.

Run only when the parent releases the GPU. Every compared leaf and both Adam
updates are retained. This diagnostic does not choose or widen tolerances.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise FileExistsError("Choose a new diagnostic report")
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_nested_attention_recompute import run, setup

    from exl3_mlx.attention import empty_attention_state
    from exl3_mlx.recompute import recompute
    from exl3_mlx.sft import implementation_identity

    identity = implementation_identity()
    inputs = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "tests/test_nested_attention_recompute.py",
            Path(__file__).resolve().parents[1] / "tests/test_composed_gradient.py",
        )
    }

    def scalar(value):
        return float(value) if math.isfinite(float(value)) else None

    def stats(left, right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("Shape/dtype mismatch")
        lraw = np.array(left.view(mx.uint16)) if left.dtype == mx.bfloat16 else np.array(left)
        rraw = np.array(right.view(mx.uint16)) if right.dtype == mx.bfloat16 else np.array(right)
        lf = np.array(left.astype(mx.float32)).astype(np.float64).ravel()
        rf = np.array(right.astype(mx.float32)).astype(np.float64).ravel()
        difference = rf - lf
        norm = np.linalg.norm(lf)
        denominator = norm * np.linalg.norm(rf)
        floating = mx.issubdtype(left.dtype, mx.floating)
        ulp = None
        if left.dtype in (mx.float32, mx.bfloat16):
            bits = 16 if left.dtype == mx.bfloat16 else 32
            integers = np.uint16 if bits == 16 else np.uint32

            def ordered(raw):
                u = raw.view(integers).astype(np.uint64)
                sign = np.uint64(1 << (bits - 1))
                mask = np.uint64((1 << bits) - 1)
                return np.where((u & sign) != 0, (~u) & mask, u | sign).astype(np.int64)

            ulp = int(np.max(np.abs(ordered(lraw) - ordered(rraw)), initial=0))
        nonzero = np.abs(lf) > 0
        exact = bool(np.array_equal(lraw, rraw))
        return {
            "shape": list(left.shape),
            "dtype": str(left.dtype),
            "elements": int(left.size),
            "floating": bool(floating),
            "finite": bool(np.all(np.isfinite(lf)) and np.all(np.isfinite(rf))),
            "exact_equal": exact,
            "different_elements": int(np.count_nonzero(lraw != rraw)),
            "max_abs": scalar(np.max(np.abs(difference), initial=0)),
            "max_abs_over_reference_max": scalar(
                np.max(np.abs(difference), initial=0) / max(np.max(np.abs(lf), initial=0), 1e-300)
            ),
            "max_element_relative_nonzero": scalar(
                np.max(np.abs(difference[nonzero] / lf[nonzero]), initial=0)
            ),
            "relative_l2": scalar(np.linalg.norm(difference) / max(norm, 1e-300)),
            "cosine": scalar(np.dot(lf, rf) / denominator)
            if denominator
            else (1.0 if exact else None),
            "max_ulp": ulp,
            "sign_changed_nonzero": int(np.count_nonzero((lf * rf < 0))),
            "reference_zero_became_nonzero": int(np.count_nonzero((lf == 0) & (rf != 0))),
        }

    def compare(left, right):
        a, b = tree_flatten(left), tree_flatten(right)
        if [k for k, _ in a] != [k for k, _ in b]:
            raise ValueError("Tree structure mismatch")
        return {k: stats(x, y) for (k, x), (_, y) in zip(a, b, strict=True)}

    def forward_states(model, data, h, pre):
        states = []
        kv = empty_attention_state(1, h.shape[1], 32, 1, h.dtype)
        for index, block in enumerate(model.blocks):
            embedding = mx.array(data["embedding"]) if index == 0 else mx.zeros((0,))
            h, pre, kv = recompute(block, h, pre, kv, embedding)
            mx.eval(h, pre, kv)
            states.append((h, pre, kv))
        return states

    cases = []
    for dtype in (mx.float32, mx.bfloat16):
        for nonzero in (False, True):
            for mode in ("checkpoint", "bf16"):
                data, reference, h, pre = setup(dtype, nonzero, mode, False)
                _, nested, _, _ = setup(dtype, nonzero, mode, True)
                optimizers = [optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)]
                case = {
                    "dtype": str(dtype),
                    "nonzero_initial_b": nonzero,
                    "activation_mode": mode,
                    "steps": [],
                }
                for step in range(2):
                    outputs = [forward_states(model, data, h, pre) for model in (reference, nested)]
                    results = []
                    for model in (reference, nested):
                        params = model.trainable_parameters()
                        value = mx.value_and_grad(
                            lambda p, x, a: run(model, data, p, x, a), argnums=(0, 1, 2)
                        )(params, h, pre)
                        mx.eval(value)
                        results.append(value)
                    record = {
                        "step": step + 1,
                        "forward_state_leaves": compare(*outputs),
                        "objective_and_gradient_leaves": compare(*results),
                    }
                    for model, optimizer, (_, gradients) in zip(
                        (reference, nested), optimizers, results, strict=True
                    ):
                        optimizer.update(model, gradients[0])
                        mx.eval(model.parameters(), optimizer.state)
                    record["post_update_parameter_leaves"] = compare(
                        reference.trainable_parameters(), nested.trainable_parameters()
                    )
                    record["post_update_optimizer_leaves"] = compare(
                        optimizers[0].state, optimizers[1].state
                    )
                    case["steps"].append(record)
                cases.append(case)
                print(
                    json.dumps({"finished_case": {k: v for k, v in case.items() if k != "steps"}}),
                    flush=True,
                )
    if implementation_identity() != identity or any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != d for p, d in inputs.items()
    ):
        raise ValueError("Diagnostic implementation changed during execution")
    report = {
        "scope": "Small two-block numerical diagnostic with trainable shared KV; no large-model or higher-derivative claim",
        "acceptance": "No automatic numerical pass; original bitwise-parity requirement remains failed until separately reviewed",
        "implementation": identity,
        "inputs": inputs,
        "cases": cases,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
