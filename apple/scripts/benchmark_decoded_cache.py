"""Isolated real-matrix cache/chunk sweep; run without another GPU workload.

Example (uses the checkpoint in place, without copying its weight bank):
  PYTHONPATH=src python scripts/benchmark_decoded_cache.py --checkpoint MODEL \
    --keys layers.0.ffn.experts.0.w1 layers.0.ffn.experts.0.w2 \
    --rows 32 128 --chunk-sizes 512 2048 8192 --output reports/tile-benchmark.json

Warm single-matrix reuse is an upper bound on cache benefit. A one-pass whole
model evaluation can miss every decoded tile; these timings do not predict its
throughput. First use can include JIT and OS cache effects, so report it separately.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import exl3_mlx.linear as linear_source
import exl3_mlx.streaming as streaming_source
from exl3_mlx.checkpoint import Checkpoint
from exl3_mlx.materialize import materialize
from exl3_mlx.memory import MemoryBudget
from exl3_mlx.storage import ExpertBank
from exl3_mlx.streaming import MatrixStore


def payload(value):
    integer = mx.uint32 if value.dtype == mx.float32 else mx.uint16
    return np.array(value.view(integer)).tobytes()


def signature(value):
    return hashlib.sha256(payload(value)).hexdigest()


def metrics(value, reference, dtype):
    actual = np.array(value.astype(mx.float32)).astype(np.float64)
    delta = actual - reference
    tolerances = {"float32": 2e-5, "float16": 2e-3, "bfloat16": 2e-2}
    tolerance = tolerances[dtype]
    return {
        "max_abs_error": float(np.max(np.abs(delta))),
        "relative_l2_error": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-30)),
        "atol": tolerance,
        "rtol": tolerance,
        "within_tolerance": bool(np.allclose(actual, reference, atol=tolerance, rtol=tolerance)),
    }


def run_sequence(layer, x, dy):
    # Emulate a forward followed by backward recomputation. The public VJP call
    # includes its own forward; no private backward path or altered GEMM is used.
    begin = time.perf_counter()
    forward = layer(x)
    mx.eval(forward)
    middle = time.perf_counter()
    (recomputed,), (dx,) = mx.vjp(layer, [x], [dy])
    mx.eval(recomputed, dx)
    end = time.perf_counter()
    if payload(forward) != payload(recomputed):
        raise AssertionError("Repeated forward changed within one sequence")
    if not mx.all(mx.isfinite(forward)).item() or not mx.all(mx.isfinite(dx)).item():
        raise FloatingPointError("Benchmark produced nonfinite output or input gradient")
    return (
        forward,
        dx,
        {
            "forward_seconds": middle - begin,
            "forward_plus_vjp_seconds": end - middle,
            "recompute_sequence_seconds": end - begin,
        },
    )


def benchmark(args):
    root = Path(args.checkpoint).expanduser().resolve()
    checkpoint = (
        ExpertBank(root) if (root / "expert-manifest-v2.json").is_file() else Checkpoint(root)
    )
    dtype = getattr(mx, args.dtype)
    # 512 is the numerical baseline even if the requested sweep omits it.
    chunks = list(dict.fromkeys([512, *args.chunk_sizes]))
    result = {
        "scope": "isolated EXL3 matrix; warm-cache reuse is not whole-model throughput",
        "timing_notes": [
            "Cold means empty MatrixStore caches, not cold OS file cache or JIT state.",
            "Measured warm sequences follow an initial sequence and requested warmup sequences.",
            "Forward-plus-VJP includes a forward recomputation; sequence is forward then that VJP.",
            "Different chunk sizes can change FP32 matmul and transpose accumulation order.",
            "Run with no concurrent GPU workload; this script does not control other processes.",
        ],
        "checkpoint": str(root),
        "source": getattr(checkpoint, "source", None),
        "manifest_sha256": getattr(checkpoint, "manifest_sha256", None),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")},
        "platform": platform.platform(),
        "source_files": {
            str(Path(module.__file__).resolve()): hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (linear_source, streaming_source)
        },
        "settings": vars(args),
        "cases": [],
    }
    with MemoryBudget(args.memory_gib, cache_mib=64, reserve_gib=args.reserve_gib) as budget:
        for key in args.keys:
            _, packed_shape, _ = checkpoint.tensors[key + ".trellis"]
            k, n = packed_shape[0] * 16, packed_shape[1] * 16
            for rows in args.rows:
                mx.random.seed(args.seed)
                x = materialize(mx.random.normal((rows, k)).astype(dtype))
                dy = materialize(mx.random.normal((rows, n)).astype(dtype))
                baseline, baseline_signatures = None, None
                for chunk in chunks:
                    chunk_signatures = None
                    for decoded_mib in (0, args.decoded_cache_mib):
                        store = MatrixStore(
                            checkpoint,
                            cache_bytes=args.packed_cache_mib << 20,
                            decoded_cache_bytes=decoded_mib << 20,
                            chunk_size=chunk,
                        )
                        layer = store.linear(key)
                        gc.collect()
                        mx.clear_cache()
                        memory_before = mx.get_active_memory()
                        mx.reset_peak_memory()
                        y, dx, cold = run_sequence(layer, x, dy)
                        budget.check()
                        cold_stats = store.report()
                        reference_signatures = (signature(y), signature(dx))
                        if chunk == 512 and decoded_mib == 0:
                            baseline = tuple(
                                np.array(v.astype(mx.float32)).astype(np.float64) for v in (y, dx)
                            )
                            baseline_signatures = reference_signatures
                        if decoded_mib == 0:
                            chunk_signatures = reference_signatures
                        if reference_signatures != chunk_signatures:
                            raise AssertionError(
                                f"Decoded cache changed exact output/dX at {key}, chunk {chunk}"
                            )
                        numerical = {
                            name: metrics(value, ref, args.dtype)
                            for name, value, ref in zip(
                                ("output", "dx"), (y, dx), baseline, strict=True
                            )
                        }
                        for _ in range(args.warmup):
                            y, dx, _ = run_sequence(layer, x, dy)
                            budget.check()
                            if (signature(y), signature(dx)) != reference_signatures:
                                raise AssertionError("Warmup changed exact output/dX")
                        times = []
                        for _ in range(args.repetitions):
                            y, dx, timing = run_sequence(layer, x, dy)
                            budget.check()
                            times.append(timing)
                            if (signature(y), signature(dx)) != reference_signatures:
                                raise AssertionError("Warm cache changed exact output/dX")
                        result["cases"].append(
                            {
                                "key": key,
                                "shape": [k, n],
                                "bpw": packed_shape[-1] / 16,
                                "rows": rows,
                                "dtype": args.dtype,
                                "chunk_size": chunk,
                                "decoded_cache_mib": decoded_mib,
                                "cold_store_cache": cold,
                                "cold_stats": cold_stats,
                                "warm_median": {
                                    name: statistics.median(t[name] for t in times)
                                    for name in times[0]
                                },
                                "warm_samples": times,
                                "final_stats": store.report(),
                                "extra_peak_mlx_bytes": mx.get_peak_memory() - memory_before,
                                "same_chunk_cache_bitwise_equal": reference_signatures
                                == chunk_signatures,
                                "baseline_512_bitwise_equal": reference_signatures
                                == baseline_signatures,
                                "vs_baseline_512": numerical,
                                "output_sha256": reference_signatures[0],
                                "dx_sha256": reference_signatures[1],
                            }
                        )
                        print(
                            f"Finished {key} rows={rows} chunk={chunk} decoded_mib={decoded_mib}",
                            flush=True,
                        )
                        store.clear()
                        del layer, store, y, dx
                del x, dy
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--packed-cache-mib", type=int, default=128)
    parser.add_argument("--decoded-cache-mib", type=int, default=128)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--reserve-gib", type=float, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(row < 1 for row in args.rows) or any(
        chunk < 128 or chunk % 128 for chunk in args.chunk_sizes
    ):
        parser.error("Positive row counts and chunk sizes that are multiples of 128 are required")
    if (
        args.decoded_cache_mib < 1
        or args.packed_cache_mib < 0
        or args.repetitions < 1
        or args.warmup < 0
    ):
        parser.error("Use positive decoded cache/repetitions and nonnegative packed cache/warmup")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new benchmark report path")
    report = benchmark(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
