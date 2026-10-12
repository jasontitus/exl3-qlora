"""Actual one-layer DeepSeek MoE cache sweep, without another GPU workload.

Uses real frozen router/shared/routed expert weights and a long synthetic BF16
activation batch. This exercises whole-layer cache capacity/reuse, not an entire
model or captured training activations. A forward is followed by public mx.vjp,
which includes a second forward and the MoE's expert backward recomputations.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import exl3_mlx.attention as attention_source
import exl3_mlx.linear as linear_source
import exl3_mlx.moe as moe_source
import exl3_mlx.streaming as streaming_source
from exl3_mlx.hyperconnections import rms_norm
from exl3_mlx.materialize import materialize
from exl3_mlx.memory import MemoryBudget
from exl3_mlx.moe import DeepSeekExpert, DeepSeekGate, DeepSeekMoE
from exl3_mlx.native import NativeLinear
from exl3_mlx.storage import ExpertBank, TensorArchive
from exl3_mlx.streaming import MatrixStore


def array_bytes(value):
    return np.array(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value).tobytes()


def digest(value):
    return hashlib.sha256(array_bytes(value)).hexdigest()


def error_metrics(value, reference):
    actual = np.array(value.astype(mx.float32)).astype(np.float64)
    difference = actual - reference
    return {
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)
        ),
        "atol": 2e-2,
        "rtol": 2e-2,
        "within_bf16_tolerance": bool(np.allclose(actual, reference, atol=2e-2, rtol=2e-2)),
    }


def make_moe(archive, store, config, layer):
    prefix = f"layers.{layer}.ffn"
    gate = DeepSeekGate(
        archive.array(prefix + ".gate.weight"),
        archive.array(prefix + ".gate.bias"),
        top_k=config["num_experts_per_tok"],
        route_scale=config["routed_scaling_factor"],
        gate_temp=config.get("gate_temp", 1.0),
        norm_topk_prob=config["norm_topk_prob"],
    )
    experts = [
        DeepSeekExpert(
            *(store.linear(f"{prefix}.experts.{index}.w{projection}") for projection in (1, 2, 3)),
            swiglu_limit=config["swiglu_limit"],
        )
        for index in range(config["n_routed_experts"])
    ]
    shared = DeepSeekExpert(
        *(
            NativeLinear(archive, f"{prefix}.shared_experts.w{projection}")
            for projection in (1, 2, 3)
        ),
        swiglu_limit=config["swiglu_limit"],
    )
    model = DeepSeekMoE(gate, experts, shared)
    if tree_flatten(model.trainable_parameters()):
        raise AssertionError("The layer benchmark must have no trainable parameters")
    mx.eval(model.parameters())
    return model


def phase(function, store, budget):
    before = store.report()
    begin = time.perf_counter()
    result = function()
    mx.eval(result)
    seconds = time.perf_counter() - begin
    budget.check()
    after = store.report()
    return result, {
        "seconds": seconds,
        "stats_before": before,
        "stats_after": after,
        "stats_delta": {key: after[key] - before[key] for key in before},
        "active_mlx_bytes": mx.get_active_memory(),
        "peak_mlx_bytes": mx.get_peak_memory(),
        "process_accounted_bytes": budget.usage(),
    }


def benchmark(args):
    root = Path(args.checkpoint).expanduser().resolve()
    config = json.loads((root / "config.json").read_text())["text_config"]
    if (
        not 0 <= args.layer < config["num_hidden_layers"]
        or config["scoring_func"] != "sqrtsoftplus"
        or config["n_shared_experts"] != 1
    ):
        raise ValueError("Select a supported DeepSeek V4.1 MoE layer")
    bank, archive = ExpertBank(root), TensorArchive(root)
    report = {
        "scope": "actual single MoE layer with synthetic normalized BF16 inputs; not full-model throughput",
        "checkpoint": str(root),
        "source": bank.source,
        "manifest_sha256": bank.manifest_sha256,
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")},
        "source_files": {
            str(Path(module.__file__).resolve()): hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (attention_source, linear_source, moe_source, streaming_source)
        },
        "settings": vars(args),
        "notes": [
            "Inputs are seeded random BF16 values normalized with this layer's genuine FFN norm weight.",
            "Cold means empty per-condition stores, not cold OS page cache or JIT state.",
            "The VJP phase includes another MoE forward and per-expert backward recomputations.",
            "Same-chunk cache comparisons must be bitwise; larger chunks use separate numerical metrics.",
            "Both caches can hold this layer, but cannot retain every layer of the full model.",
            "First-use JIT and condition order may affect timings; this is a one-sequence layer experiment.",
        ],
        "cases": [],
    }
    budget = MemoryBudget(args.memory_gib, cache_mib=64, reserve_gib=args.reserve_gib)
    budget.abort_path = str(Path(args.output).with_suffix(".memory-abort.json"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with budget:
        mx.random.seed(args.seed)
        x = materialize(
            rms_norm(
                mx.random.normal((args.rows, config["hidden_size"])).astype(mx.bfloat16),
                archive.array(f"layers.{args.layer}.ffn_norm.weight"),
                config["rms_norm_eps"],
            )
        )
        dy = materialize(mx.random.normal(x.shape).astype(mx.bfloat16))
        report["input_sha256"], report["cotangent_sha256"] = digest(x), digest(dy)
        baseline_values, baseline_digests = None, None
        for chunk in (512, 8192):
            uncached_digests = None
            for name, packed_mib, decoded_mib in (("low", 128, 0), ("development", 8192, 32768)):
                print(
                    f"Starting layer={args.layer} rows={args.rows} chunk={chunk} profile={name}",
                    flush=True,
                )
                store = MatrixStore(
                    bank,
                    cache_bytes=packed_mib << 20,
                    decoded_cache_bytes=decoded_mib << 20,
                    chunk_size=chunk,
                )
                model = make_moe(archive, store, config, args.layer)
                route_weights, indices = model.gate(x)
                route_counts = np.bincount(
                    np.array(indices).reshape(-1), minlength=config["n_routed_experts"]
                )
                del route_weights, indices
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                initial_memory = mx.get_active_memory()
                output, forward = phase(lambda m=model: m(x), store, budget)
                print(
                    f"Forward {forward['seconds']:.3f}s; active {forward['active_mlx_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
                ((recomputed,), (dx,)), backward = phase(
                    lambda m=model: mx.vjp(m, [x], [dy]), store, budget
                )
                print(
                    f"Forward+VJP {backward['seconds']:.3f}s; peak {backward['peak_mlx_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
                if digest(output) != digest(recomputed):
                    raise AssertionError("Repeated MoE forward changed within a condition")
                if not mx.all(mx.isfinite(output)).item() or not mx.all(mx.isfinite(dx)).item():
                    raise FloatingPointError("Nonfinite layer benchmark output or input gradient")
                signatures = digest(output), digest(dx)
                if name == "low":
                    uncached_digests = signatures
                elif signatures != uncached_digests:
                    raise AssertionError(
                        f"Decoded cache changed bitwise MoE output/dX at chunk {chunk}"
                    )
                if chunk == 512 and name == "low":
                    baseline_digests = signatures
                    baseline_values = tuple(
                        np.array(v.astype(mx.float32)).astype(np.float64) for v in (output, dx)
                    )
                report["cases"].append(
                    {
                        "profile": name,
                        "layer": args.layer,
                        "rows": args.rows,
                        "hidden_size": config["hidden_size"],
                        "chunk_size": chunk,
                        "packed_cache_mib": packed_mib,
                        "decoded_cache_mib": decoded_mib,
                        "experts_used": int(np.count_nonzero(route_counts)),
                        "routed_pairs": int(route_counts.sum()),
                        "route_counts": route_counts.tolist(),
                        "initial_active_mlx_bytes": initial_memory,
                        "forward": forward,
                        "forward_plus_vjp": backward,
                        "recompute_sequence_seconds": forward["seconds"] + backward["seconds"],
                        "output_sha256": signatures[0],
                        "dx_sha256": signatures[1],
                        "same_chunk_cache_bitwise_equal": signatures == uncached_digests,
                        "baseline_512_bitwise_equal": signatures == baseline_digests,
                        "vs_baseline_512": {
                            role: error_metrics(value, reference)
                            for role, value, reference in zip(
                                ("output", "dx"), (output, dx), baseline_values, strict=True
                            )
                        },
                    }
                )
                report["complete"] = len(report["cases"]) == 4
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                store.clear()
                del model, store, output, recomputed, dx
                gc.collect()
                mx.clear_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, default=5808)
    parser.add_argument("--memory-gib", type=float, default=48)
    parser.add_argument("--reserve-gib", type=float, default=32)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("Use a positive row count")
    if Path(args.output).exists():
        parser.error("Output exists; choose a new report path")
    benchmark(args)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
