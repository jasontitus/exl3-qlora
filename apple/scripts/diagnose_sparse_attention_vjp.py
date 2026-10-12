#!/usr/bin/env python3
"""Bounded, real-head-dimension sparse-attention forward/VJP diagnostic.

Run sequentially with other GPU jobs, from apple/:
  PYTHONPATH=src .venv/bin/python scripts/diagnose_sparse_attention_vjp.py \
      --output /tmp/sparse-attention-vjp.json

--plan-only and --self-check do not import MLX or start GPU work. This checks the
fixed-selection attention primitive, not indexer gradients, the whole model, or
fine-tuning quality. Tolerances below are fixed before any real-dimension run.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TOLERANCES = {
    "float32": {"atol": 2e-5, "rtol": 4e-4, "relative_l2": 1e-4, "cosine_min": 0.999999},
    "bfloat16": {"atol": 2e-3, "rtol": 3e-2, "relative_l2": 1.5e-2, "cosine_min": 0.9999},
}
SHAPE = {
    "batch": 1,
    "heads": 64,
    "dim": 512,
    "window": 128,
    "selected": 512,
    "pooled": 48,
    "query_chunk": 16,
}
LENGTHS = (145, 160)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_bytes(value.tobytes()),
    }


def selection_fixture(length):
    """Synthetic ratio-four visibility, with intentional repeated key reads.

    The width is the model's real K512, but repetitions are deliberate stress
    cases, not a claim that the model's unique top-k selector emits duplicates.
    Invalid padding appears both between valid entries and at early positions.
    """
    selected = np.full((1, length, SHAPE["selected"]), -1, dtype=np.int32)
    for position in range(length):
        visible = (position + 1) // 4
        if not visible:
            continue
        slots = np.arange(SHAPE["selected"])
        row = (slots * slots + 3 * slots + 7 * position) % visible
        if position % 3 == 0:
            row[(slots + position) % 5 == 0] = -1
        selected[0, position] = row
    return selected


def dense_multiplicity(selected, raw_length, pooled_length, window_size):
    """Independent global-key masks/counts; no tiled gather/scatter logic."""
    batch, query_count, _ = selected.shape
    counts = np.zeros((batch, query_count, raw_length + pooled_length), np.int32)
    for b in range(batch):
        for query in range(query_count):
            for raw in range(raw_length):
                if 0 <= query - raw < window_size:
                    counts[b, query, raw] = 1
            for pooled in selected[b, query]:
                if pooled != -1:
                    if pooled < 0 or pooled >= pooled_length:
                        raise ValueError("Invalid fixture selection")
                    counts[b, query, raw_length + pooled] += 1
    return counts


def metrics(actual, expected, tolerances):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "actual": list(actual.shape),
            "expected": list(expected.shape),
        }
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {"passed": False, "finite": False}
    error = actual - expected
    error_l2 = float(np.linalg.norm(error.ravel()))
    expected_l2 = float(np.linalg.norm(expected.ravel()))
    actual_l2 = float(np.linalg.norm(actual.ravel()))
    relative_l2 = error_l2 / max(expected_l2, 1e-30)
    cosine = (
        float(np.dot(actual.ravel(), expected.ravel()) / (actual_l2 * expected_l2))
        if actual_l2 and expected_l2
        else float(actual_l2 == expected_l2)
    )
    allowed = tolerances["atol"] + tolerances["rtol"] * np.abs(expected)
    failures = int(np.count_nonzero(np.abs(error) > allowed))
    worst = np.unravel_index(np.argmax(np.abs(error)), error.shape)
    return {
        "passed": failures == 0
        and relative_l2 <= tolerances["relative_l2"]
        and cosine >= tolerances["cosine_min"],
        "finite": True,
        "elements": int(error.size),
        "allclose_failures": failures,
        "max_abs_error": float(np.max(np.abs(error))),
        "error_l2": error_l2,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "actual_l2": actual_l2,
        "reference_l2": expected_l2,
        "worst_coordinate": [int(i) for i in worst],
        "worst_actual": float(actual[worst]),
        "worst_reference": float(expected[worst]),
    }


def self_check():
    """CPU-only fixture/oracle-equivalence and grading sanity checks."""
    rng = np.random.default_rng(812)
    for length in LENGTHS:
        selected = selection_fixture(length)
        count = dense_multiplicity(selected, length, SHAPE["pooled"], SHAPE["window"])
        if np.any(selected[0, :3] != -1) or np.max(count[..., length:]) < 2:
            raise AssertionError("Fixture must exercise padding and repeated keys")
        for query in range(length):
            if np.any(selected[0, query] >= (query + 1) // 4):
                raise AssertionError("Noncausal synthetic compressed selection")
        if np.any(count[..., length + 40 :]):
            raise AssertionError("Fixture must include unused compressed rows")
        # Float64, explicit duplicated-key softmax vs global multiplicities.
        # Use a small head dimension here; this validates oracle construction.
        keys = rng.normal(size=(length + SHAPE["pooled"], 7))
        query = rng.normal(size=(7,))
        for position in (0, 3, 15, 127, length - 1):
            raw_ids = list(range(max(0, position - SHAPE["window"] + 1), position + 1))
            ids = raw_ids + [length + int(i) for i in selected[0, position] if i >= 0]
            repeated = keys[ids]
            logits = np.append(repeated @ query / math.sqrt(7), 0.25)
            p = np.exp(logits - np.max(logits))
            p /= p.sum()
            explicit = p[:-1] @ repeated
            live = count[0, position] > 0
            logits = np.append(
                keys[live] @ query / math.sqrt(7) + np.log(count[0, position, live]), 0.25
            )
            p = np.exp(logits - np.max(logits))
            p /= p.sum()
            np.testing.assert_allclose(p[:-1] @ keys[live], explicit, rtol=1e-12, atol=1e-12)
    good = np.arange(1, 20, dtype=np.float32)
    if not metrics(good, good, TOLERANCES["float32"])["passed"]:
        raise AssertionError("Identical tensors did not pass")
    if metrics(-good, good, TOLERANCES["float32"])["passed"]:
        raise AssertionError("Opposite gradients incorrectly passed")
    if metrics(np.full_like(good, np.nan), good, TOLERANCES["float32"])["passed"]:
        raise AssertionError("Nonfinite gradients incorrectly passed")
    return {
        "passed": True,
        "checks": "causality, repetitions, unused rows, dense multiplicity equivalence, grading",
    }


def plan():
    return {
        "schema": "sparse-attention-real-dim-vjp-v1",
        "shape": SHAPE,
        "sequence_lengths": list(LENGTHS),
        "dtypes": ["float32", "bfloat16"],
        "sink_dtype": "float32 in both cases (model-like sink parameter)",
        "tolerances_predeclared": TOLERANCES,
        "oracle": "Dense global keys, causal window mask and log integer selection multiplicities; MLX autodiff",
        "scope": "B1, H64, D512, window128, selected512, query tiles16; 10 tiles with partial/full tails",
        "limits": [
            "Synthetic fixed selections; no top-k/indexer/compression/RoPE or whole-model differentiation",
            "Repeated selections deliberately stress accumulation; not claimed to be real unique top-k output",
            "BF16 query/raw/pooled/output/cotangent with F32 sink; F32 uses F32 throughout",
            "Forward output casts match the primitive; derivatives are with respect to supplied rounded inputs",
            "Independent expression shares MLX matmul/softmax/autodiff backend, not a separate hardware oracle",
            "Passing this diagnostic does not certify training efficacy or generated kernel quality",
        ],
    }


def run_case(mx, attention, length, dtype_name, budget):
    rng = np.random.default_rng(1729 + length)
    b, h, d, c = (SHAPE[k] for k in ("batch", "heads", "dim", "pooled"))
    dtype = getattr(mx, dtype_name)
    values = [
        (rng.standard_normal((b, length, h, d)) * 0.8).astype(np.float32),
        (rng.standard_normal((b, length, d)) * 0.65).astype(np.float32),
        (rng.standard_normal((b, c, d)) * 0.9).astype(np.float32),
        np.linspace(-1.25, 1.0, h, dtype=np.float32),
    ]
    dy_np = (
        0.03 * rng.standard_normal(values[0].shape)
        + 0.02
        * np.sin(np.arange(length)[None, :, None, None] * 0.2 + np.arange(h)[None, None, :, None])
    ).astype(np.float32)
    primals = [mx.array(v, dtype=dtype if i < 3 else mx.float32) for i, v in enumerate(values)]
    dy = mx.array(dy_np, dtype=dtype)
    selected_np = selection_fixture(length)
    selected = mx.array(selected_np)
    counts = dense_multiplicity(selected_np, length, c, SHAPE["window"])
    # Mask construction and integer counts are fixed, outside the gradient tape.
    log_count_np = np.full(counts.shape, -np.inf, np.float32)
    live = counts > 0
    log_count_np[live] = np.log(counts[live]).astype(np.float32)
    log_count = mx.array(log_count_np)

    def candidate(q, raw, pooled, sink):
        return attention(
            q,
            raw,
            pooled,
            selected,
            sink,
            window_size=SHAPE["window"],
            query_chunk=SHAPE["query_chunk"],
        )

    def dense_oracle(q, raw, pooled, sink):
        # Shared global keys are each stored once. Multiplicity m replaces m
        # duplicate softmax entries by one logit + log(m). Its gradient includes
        # the shared latent's key AND value paths through ordinary autodiff.
        keys = mx.concatenate([raw.astype(mx.float32), pooled.astype(mx.float32)], axis=1)
        queries = mx.transpose(q.astype(mx.float32), (0, 2, 1, 3))
        logits = (queries @ mx.swapaxes(keys[:, None, :, :], -1, -2)) / math.sqrt(d)
        logits = logits + log_count[:, None, :, :]
        sink_logits = mx.broadcast_to(sink[None, :, None, None], (*logits.shape[:-1], 1))
        probabilities = mx.softmax(mx.concatenate([logits, sink_logits], axis=-1), axis=-1)
        result = probabilities[..., :-1] @ keys[:, None, :, :]
        return mx.transpose(result, (0, 2, 1, 3)).astype(q.dtype)

    mx.eval(*primals, dy, selected, log_count)
    actual_input_hashes = {
        name: array_hash(np.array(v.astype(mx.float32)))
        for name, v in zip(
            ("query", "raw", "pooled", "sink", "cotangent"), [*primals, dy], strict=True
        )
    }
    del values, dy_np
    gc.collect()
    mx.reset_peak_memory()
    times = {}
    results = []
    for label, function in (("candidate", candidate), ("dense_autodiff", dense_oracle)):
        start = time.perf_counter()
        (out,), grads = mx.vjp(function, primals, [dy])
        mx.eval(out, *grads)
        budget.check()
        results.append([np.array(v.astype(mx.float32)) for v in [out, *grads]])
        times[label] = time.perf_counter() - start
        del out, grads
        gc.collect()
        mx.clear_cache()
    tolerance = TOLERANCES[dtype_name]
    names = ("forward", "dQuery", "dRaw", "dPooled", "dSink")
    tensor_results = {
        name: metrics(a, e, tolerance) for name, a, e in zip(names, *results, strict=True)
    }
    slices = {
        "early": (0, 4),
        "tile_boundary": (15, 18),
        "window_boundary": (126, 131),
        "tail": (length - 3, length),
    }
    query_results = {}
    for name, index in (("forward", 0), ("dQuery", 1)):
        query_results[name] = {
            part: {
                "query_range": [start, end],
                **metrics(
                    results[0][index][:, start:end], results[1][index][:, start:end], tolerance
                ),
            }
            for part, (start, end) in slices.items()
        }
    unused = np.sum(counts[..., length:], axis=(0, 1)) == 0
    unused_exact_zero = all(np.count_nonzero(result[3][:, unused]) == 0 for result in results)
    passed = (
        all(item["passed"] for item in tensor_results.values())
        and all(item["passed"] for group in query_results.values() for item in group.values())
        and unused_exact_zero
    )
    receipt = {
        "case_id": f"s{length}-{dtype_name}",
        "passed": passed,
        "shape": {**SHAPE, "sequence": length},
        "dtype": dtype_name,
        "numpy_seed": 1729 + length,
        "tensor_results": tensor_results,
        "query_slice_results": query_results,
        "unselected_pooled_rows": int(np.count_nonzero(unused)),
        "unselected_pooled_gradient_exact_zero": unused_exact_zero,
        "maximum_pooled_multiplicity": int(np.max(counts[..., length:])),
        "selected_hash": array_hash(selected_np),
        "actual_inputs_as_float32": actual_input_hashes,
        "timing_seconds_not_a_benchmark": times,
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
        "process_observed_gib": budget.usage() / 2**30,
    }
    del results, primals, dy
    gc.collect()
    mx.clear_cache()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--self-check", action="store_true", help="CPU-only oracle/fixture checks")
    parser.add_argument("--memory-gib", type=float, default=3.0)
    parser.add_argument("--reserve-gib", type=float, default=16.0)
    args = parser.parse_args()
    if args.plan_only or args.self_check:
        document = plan()
        if args.self_check:
            document["cpu_self_check"] = self_check()
        print(json.dumps(document, indent=2, allow_nan=False))
        return 0
    if args.output is None:
        parser.error("--output is required for a GPU run")
    output = args.output.expanduser().resolve()
    if args.output.is_symlink() or output.exists():
        parser.error("--output must be a new file, not an existing path or symlink")
    if not output.parent.is_dir():
        parser.error("--output parent directory must already exist")
    # Import GPU dependencies only after explicit run arguments are validated.
    import mlx.core as mx

    from exl3_mlx import sparse_attention
    from exl3_mlx.memory import MemoryBudget
    from exl3_mlx.sparse_attention import sparse_latent_attention

    report = {
        **plan(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "numpy")},
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "primitive_sha256": sha256_bytes(Path(sparse_attention.__file__).read_bytes()),
        "materialize_sha256": sha256_bytes(
            Path(sparse_attention.__file__).with_name("materialize.py").read_bytes()
        ),
        "memory_gib": args.memory_gib,
        "reserve_gib": args.reserve_gib,
        "cpu_self_check": self_check(),
        "cases": [],
        "passed": False,
    }
    try:
        with MemoryBudget(args.memory_gib, cache_mib=32, reserve_gib=args.reserve_gib) as budget:
            for length in LENGTHS:
                for dtype in TOLERANCES:
                    print(f"Checking S{length} {dtype}, H64 D512 K512...", flush=True)
                    case = run_case(mx, sparse_latent_attention, length, dtype, budget)
                    report["cases"].append(case)
                    print(
                        f"  passed={case['passed']} peak_mlx_gib={case['peak_mlx_gib']:.3f}",
                        flush=True,
                    )
        report["passed"] = len(report["cases"]) == 4 and all(c["passed"] for c in report["cases"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Diagnostic passed={report['passed']}; report: {output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
