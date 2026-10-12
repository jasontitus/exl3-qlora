"""Add packed-only conditions to a completed actual-MoE benchmark report.

Run after the reference benchmark and all other GPU workloads have finished.
This reuses the exact checkpoint, synthetic inputs, layer, seed, and chunk sizes
from the supplied report. It requires bitwise output/dX equality with each
recorded low-cache condition, without repeating those baseline executions.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import benchmark_moe_cache as original
import mlx.core as mx
import numpy as np

from exl3_mlx.hyperconnections import rms_norm
from exl3_mlx.materialize import materialize
from exl3_mlx.memory import MemoryBudget
from exl3_mlx.storage import ExpertBank, TensorArchive
from exl3_mlx.streaming import MatrixStore


def module_hashes():
    return {
        Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (
            original.attention_source,
            original.linear_source,
            original.moe_source,
            original.streaming_source,
        )
    }


def run(args):
    reference_path = Path(args.reference_report).expanduser().resolve()
    reference_bytes = reference_path.read_bytes()
    reference_hash = hashlib.sha256(reference_bytes).hexdigest()
    reference = json.loads(reference_bytes)
    if reference.get("complete") is not True:
        raise ValueError("Reference benchmark must be complete before the packed-only run")
    expected_hashes = {Path(path).name: value for path, value in reference["source_files"].items()}
    if module_hashes() != expected_hashes:
        raise ValueError("Engine source files differ from the reference benchmark")
    versions = {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")}
    if versions != reference["versions"] or platform.platform() != reference["platform"]:
        raise ValueError("Runtime versions/platform differ from the reference benchmark")
    settings = reference["settings"]
    layer, rows, seed = settings["layer"], settings["rows"], settings["seed"]
    root = Path(reference["checkpoint"]).expanduser().resolve()
    config = json.loads((root / "config.json").read_text())["text_config"]
    baselines = {}
    for chunk in args.chunk_sizes:
        cases = [
            case
            for case in reference["cases"]
            if case["profile"] == "low" and case["chunk_size"] == chunk
        ]
        if len(cases) != 1 or cases[0]["layer"] != layer or cases[0]["rows"] != rows:
            raise ValueError(f"Reference must contain exactly one matching low-cache chunk {chunk}")
        if cases[0]["packed_cache_mib"] != 128 or cases[0]["decoded_cache_mib"] != 0:
            raise ValueError("Reference low-cache budgets differ from the intended baseline")
        baselines[chunk] = cases[0]
    bank, archive = ExpertBank(root), TensorArchive(root)
    if bank.manifest_sha256 != reference["manifest_sha256"] or bank.source != reference["source"]:
        raise ValueError("Expert manifest/source differs from the reference benchmark")
    report = {
        "scope": "actual single MoE layer, packed-only cache, paired against recorded baseline hashes",
        "paired_reference": str(reference_path),
        "paired_reference_sha256": reference_hash,
        "checkpoint": str(root),
        "manifest_sha256": bank.manifest_sha256,
        "source": bank.source,
        "versions": versions,
        "platform": platform.platform(),
        "source_files_by_name": module_hashes(),
        "settings": vars(args) | {"layer": layer, "rows": rows, "seed": seed},
        "notes": [
            "No low-cache baseline is rerun; outputs and input gradients must match saved same-chunk hashes.",
            "A new process/order may have different JIT and OS cache state; this is a one-sequence timing.",
            "Forward-plus-VJP includes another MoE forward and per-expert backward recomputations.",
            "Inputs are synthetic, normalized with the real FFN norm, not captured training activations.",
            "Whole-model throughput and training effectiveness are not established by this benchmark.",
        ],
        "cases": [],
        "complete": False,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    budget = MemoryBudget(args.memory_gib, cache_mib=64, reserve_gib=args.reserve_gib)
    budget.abort_path = str(output_path.with_suffix(".memory-abort.json"))
    with budget:
        mx.random.seed(seed)
        x = materialize(
            rms_norm(
                mx.random.normal((rows, config["hidden_size"])).astype(mx.bfloat16),
                archive.array(f"layers.{layer}.ffn_norm.weight"),
                config["rms_norm_eps"],
            )
        )
        dy = materialize(mx.random.normal(x.shape).astype(mx.bfloat16))
        input_hash, cotangent_hash = original.digest(x), original.digest(dy)
        if (
            input_hash != reference["input_sha256"]
            or cotangent_hash != reference["cotangent_sha256"]
        ):
            raise ValueError("Synthetic input/cotangent bytes differ from the reference benchmark")
        report["input_sha256"], report["cotangent_sha256"] = input_hash, cotangent_hash
        for chunk in args.chunk_sizes:
            expected = baselines[chunk]
            print(f"Starting packed-only layer={layer} rows={rows} chunk={chunk}", flush=True)
            store = MatrixStore(
                bank, cache_bytes=8192 << 20, decoded_cache_bytes=0, chunk_size=chunk
            )
            model = original.make_moe(archive, store, config, layer)
            route_weights, indices = model.gate(x)
            route_counts = np.bincount(
                np.array(indices).reshape(-1), minlength=config["n_routed_experts"]
            )
            del route_weights, indices
            if route_counts.tolist() != expected["route_counts"]:
                raise ValueError("Routing coverage differs from the reference benchmark")
            gc.collect()
            mx.clear_cache()
            mx.reset_peak_memory()
            initial_memory = mx.get_active_memory()
            output, forward = original.phase(lambda m=model: m(x), store, budget)
            print(f"Packed-only forward {forward['seconds']:.3f}s", flush=True)
            ((recomputed,), (dx,)), backward = original.phase(
                lambda m=model: mx.vjp(m, [x], [dy]), store, budget
            )
            print(f"Packed-only forward+VJP {backward['seconds']:.3f}s", flush=True)
            output_hash, dx_hash = original.digest(output), original.digest(dx)
            if original.digest(recomputed) != output_hash:
                raise AssertionError("Repeated packed-only forward changed")
            if output_hash != expected["output_sha256"] or dx_hash != expected["dx_sha256"]:
                raise AssertionError(
                    f"Packed-only output/dX differ from the recorded chunk {chunk} baseline"
                )
            if not mx.all(mx.isfinite(output)).item() or not mx.all(mx.isfinite(dx)).item():
                raise FloatingPointError("Nonfinite packed-only output or input gradient")
            if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_hash:
                raise ValueError("Reference benchmark changed during the paired run")
            total = forward["seconds"] + backward["seconds"]
            report["cases"].append(
                {
                    "profile": "packed_only",
                    "layer": layer,
                    "rows": rows,
                    "chunk_size": chunk,
                    "packed_cache_mib": 8192,
                    "decoded_cache_mib": 0,
                    "route_counts": route_counts.tolist(),
                    "experts_used": int(np.count_nonzero(route_counts)),
                    "initial_active_mlx_bytes": initial_memory,
                    "forward": forward,
                    "forward_plus_vjp": backward,
                    "recompute_sequence_seconds": total,
                    "reference_low_recompute_sequence_seconds": expected[
                        "recompute_sequence_seconds"
                    ],
                    "reference_low_seconds_over_packed_seconds": expected[
                        "recompute_sequence_seconds"
                    ]
                    / total,
                    "output_sha256": output_hash,
                    "dx_sha256": dx_hash,
                    "same_chunk_reference_bitwise_equal": True,
                }
            )
            report["complete"] = len(report["cases"]) == len(args.chunk_sizes)
            report["saved_at_unix_seconds"] = time.time()
            temporary = output_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(output_path)
            store.clear()
            del model, store, output, recomputed, dx
            gc.collect()
            mx.clear_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[512, 8192])
    parser.add_argument("--memory-gib", type=float, default=48)
    parser.add_argument("--reserve-gib", type=float, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(set(args.chunk_sizes)) != len(args.chunk_sizes) or any(
        c < 128 or c % 128 for c in args.chunk_sizes
    ):
        parser.error("Choose unique chunk sizes that are positive multiples of 128")
    if Path(args.output).exists():
        parser.error("Output exists; choose a new report path")
    run(args)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
