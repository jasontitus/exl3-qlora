"""Synthetic final-block memory feasibility probe using actual model weights.

Run alone, after other Metal workloads finish. This performs no optimizer update
and does not establish full-model training, numerical parity, or learned quality.
Example (output must be a new directory):
  PYTHONPATH=src python scripts/probe_final_block_memory.py --model MODEL \
    --output artifacts/final-block-probe --length 5808 --chunk-size 8192

Imports that initialize MLX are deliberately inside run(); --describe-only and
the shape helpers do not load weights or execute Metal.
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path


def read_plan(args):
    root = Path(args.model).expanduser().resolve()
    raw = (root / "config.json").read_bytes()
    config = json.loads(raw)["text_config"]
    layer = config["num_hidden_layers"] - 1
    if (
        layer != 39
        or config["compress_ratios"][layer] != 1
        or layer in config["kv_source_layer_ids"]
        or layer in config["index_source_layer_ids"]
        or layer in config["engram_layer_ids"]
    ):
        raise ValueError("Probe supports only the non-source, non-Engram final block39 with ratio1")
    if not 2 <= args.length <= 32768:
        raise ValueError("length must be in2..32768")
    if args.chunk_size < 128 or args.chunk_size % 128:
        raise ValueError("chunk_size must be a positive multiple of128")
    if args.rank < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("Positive adapter rank and alpha required")
    if (
        not math.isfinite(args.retained_baseline_gib)
        or args.retained_baseline_gib <= 0
        or args.retained_baseline_gib >= args.memory_gib
    ):
        raise ValueError("Retained baseline must be positive and smaller than process budget")
    return {
        "model": str(root),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "layer": layer,
        "length": args.length,
        "stream_shape": [1, args.length, config["hc_mult"], config["hidden_size"]],
        "pre_shape": [1, args.length, config["hc_mult"]],
        "compressed_kv_shape": [1, args.length, config["head_dim"]],
        "index_keys_shape": [1, args.length, config["index_head_dim"]],
        "selected_shape": [1, args.length, min(args.length, config["index_topk"])],
        "candidate_block_size": config["candidate_block_size"],
        "candidate_topk_blocks": config["candidate_topk_blocks"],
        "index_topk": config["index_topk"],
        "requested_retained_baseline_gib": args.retained_baseline_gib,
        "frozen_inputs": getattr(args, "frozen_inputs", False),
        "chunk_size": args.chunk_size,
        "scope": "synthetic final-block backward memory feasibility; no optimizer update",
        "limitations": [
            "Synthetic hidden/KV state and recent causal selections are not a real preceding39-layer state.",
            "Streams begin with expand_streams shape/pre and independent perturbations so pre-mix gradients are exercised.",
            "Dummy retained allocations simulate capacity, not the preceding autograd graph or allocator fragmentation.",
            "Data-dependent expert routing and sparse attention may differ on actual examples.",
            "A scalar final-collapse surrogate replaces the vocabulary head and supervised loss.",
            "Passing this probe does not establish full-model training feasibility or numerical parity.",
        ],
    }


def selection_arrays(length, topk, block_size, candidate_topk):
    """CPU-only plausible causal state: newest entries and newest candidate blocks."""
    import numpy as np

    if min(length, topk, block_size, candidate_topk) < 1:
        raise ValueError("Selection dimensions must be positive")
    width = min(length, topk)
    visible = np.arange(1, length + 1, dtype=np.int32)[:, None]
    selected = np.maximum(visible - width, 0) + np.arange(width, dtype=np.int32)[None, :]
    selected = np.where(selected < visible, selected, -1).astype(np.int32)[None, ...]
    count = (length + block_size - 1) // block_size
    blocks = np.arange(count, dtype=np.int32)[None, :]
    reachable = (visible + block_size - 1) // block_size
    candidates = (blocks < reachable) & (blocks >= np.maximum(reachable - candidate_topk, 0))
    packed = np.packbits(candidates, axis=-1, bitorder="little")[None, ...]
    return selected, packed


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, help="New receipt directory")
    p.add_argument("--length", type=int, default=5808)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--memory-gib", type=float, default=80.0)
    p.add_argument("--reserve-gib", type=float, default=16.0)
    p.add_argument("--packed-cache-mib", type=int, default=8192)
    p.add_argument("--decoded-cache-mib", type=int, default=32768)
    p.add_argument("--allocator-cache-mib", type=int, default=256)
    p.add_argument("--retained-baseline-gib", type=float, default=17.95)
    p.add_argument(
        "--frozen-inputs",
        action="store_true",
        help="Test final-only adapters with frozen prefix state and zero input cotangents",
    )
    p.add_argument("--describe-only", action="store_true")
    return p


def run(args, plan, resources):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    from exl3_mlx.attention import quantize_activation
    from exl3_mlx.deepseek import DeepSeekTextModel
    from exl3_mlx.hyperconnections import expand_streams, hc_pre, rms_norm
    from exl3_mlx.materialize import materialize
    from exl3_mlx.memory import MemoryBudget
    from exl3_mlx.recompute import recompute
    from exl3_mlx.sft import implementation_identity, write_json

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "starting",
        "plan": plan,
        "resources": resources,
        "settings": vars(args),
        "implementation": implementation_identity(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "python": sys.version,
        "events": [],
    }
    model = None

    def event(phase, **extra):
        item = {
            "phase": phase,
            "elapsed_seconds": time.monotonic() - started,
            "active_mlx_gib": mx.get_active_memory() / 2**30,
            "allocator_cache_gib": mx.get_cache_memory() / 2**30,
            "peak_mlx_gib": mx.get_peak_memory() / 2**30,
            "observed_process_gib": MemoryBudget.usage() / 2**30,
            **extra,
        }
        if model is not None:
            item["matrix_store"] = model._store.report()
        report["events"].append(item)
        with (output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(item, allow_nan=False) + "\n")
            handle.flush()
        write_json(output / "probe.json", report)
        print(
            f"[{item['elapsed_seconds']:.1f}s] {phase}: active {item['active_mlx_gib']:.2f}GiB, "
            f"peak {item['peak_mlx_gib']:.2f}GiB",
            flush=True,
        )

    def gradient_summary(value):
        # Bound inspection scratch as well; avoid a full FP32 copy of dH.
        flat = value.reshape(-1)
        maximum = 0.0
        for start in range(0, flat.size, 1 << 20):
            piece = flat[start : start + (1 << 20)]
            if not mx.all(mx.isfinite(piece)).item():
                raise FloatingPointError("Nonfinite final-block gradient")
            maximum = max(maximum, mx.max(mx.abs(piece).astype(mx.float32)).item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": True,
            "max_abs": maximum,
            "nonzero": maximum > 0,
        }

    try:
        with MemoryBudget(
            args.memory_gib, cache_mib=args.allocator_cache_mib, reserve_gib=args.reserve_gib
        ) as budget:
            budget.abort_path = output / "memory-abort.json"
            event("loading_model")
            mx.random.seed(args.seed)
            model = DeepSeekTextModel(
                plan["model"],
                rank=args.rank,
                alpha=args.alpha,
                cache_mib=args.packed_cache_mib,
                decoded_cache_mib=args.decoded_cache_mib,
                chunk_size=args.chunk_size,
                checkpoint_blocks=True,
            )
            model.train()
            block = model.blocks[plan["layer"]]
            if (
                block.engram is not None
                or block.attn.indexer is not None
                or block.attn.compressor is not None
            ):
                raise ValueError(
                    "Final block unexpectedly requires Engram or source-state computation"
                )
            report["model"] = model.report()
            event("model_loaded")
            budget.check()

            mx.random.seed(args.seed + 1)
            c = model.config
            embedding = materialize(
                mx.random.normal((1, args.length, c["hidden_size"])).astype(mx.bfloat16)
            )
            h, pre = expand_streams(embedding, c["hc_mult"])
            # Identical expanded streams make pre-mix derivatives degenerate
            # under RMS normalization. Later real blocks have distinct streams.
            h = materialize((h + 0.05 * mx.random.normal(h.shape)).astype(mx.bfloat16))
            pre = materialize(pre)
            del embedding
            kv = materialize(
                quantize_activation(
                    mx.random.normal(tuple(plan["compressed_kv_shape"])).astype(mx.bfloat16), "fp4"
                )
            )
            index_keys = materialize(
                mx.random.normal(tuple(plan["index_keys_shape"])).astype(mx.bfloat16)
            )
            selected_np, candidates_np = selection_arrays(
                args.length, c["index_topk"], c["candidate_block_size"], c["candidate_topk_blocks"]
            )
            selected, candidates = mx.array(selected_np), mx.array(candidates_np)
            del selected_np, candidates_np
            no_engram = mx.zeros((0,), dtype=mx.bfloat16)
            readout = materialize(
                mx.random.normal((1, args.length, c["hidden_size"])).astype(mx.bfloat16)
            )
            mx.eval(h, pre, kv, index_keys, selected, candidates, no_engram, readout)
            mx.clear_cache()
            event(
                "synthetic_state_ready",
                selected_shape=list(selected.shape),
                candidate_shape=list(candidates.shape),
            )

            # No routed expert has run yet. Keep these evaluated dense arrays
            # alive through the VJP to mimic the remaining layer-state footprint.
            if model._store.stats.cached_bytes or model._store.stats.decoded_cached_bytes:
                raise AssertionError("Expected empty expert caches before retained-state padding")
            padding = []
            target = int(args.retained_baseline_gib * 2**30)
            before_padding = mx.get_active_memory()
            while mx.get_active_memory() < target:
                count = min(16 << 20, (target - mx.get_active_memory() + 3) // 4)
                item = mx.arange(count, dtype=mx.float32)
                mx.eval(item)
                padding.append(item)
                budget.check()
                if len(padding) > 1024:
                    raise RuntimeError(
                        "Retained-state allocation did not reach requested live baseline"
                    )
            mx.clear_cache()
            report["padding"] = {
                "payload_bytes": sum(v.nbytes for v in padding),
                "chunks": len(padding),
                "before_active_gib": before_padding / 2**30,
                "after_active_gib": mx.get_active_memory() / 2**30,
                "target_already_exceeded": before_padding > target,
            }
            budget.check_system_headroom()
            report["peak_before_block_gib"] = mx.get_peak_memory() / 2**30
            mx.reset_peak_memory()
            event("retained_state_ready")

            original = block.trainable_parameters()
            leaves = tree_flatten(original)
            if not leaves or any(not name.endswith(("lora_a", "lora_b")) for name, _ in leaves):
                raise AssertionError("Expected only final-block adapter parameters")
            count = len(leaves)
            primals = [value for _, value in leaves] + [h, pre, kv]

            def objective(*values):
                block.update(
                    tree_unflatten(
                        [
                            (name, value)
                            for (name, _), value in zip(leaves, values[:count], strict=True)
                        ]
                    )
                )
                streams, mix, latent = values[count:]
                a, b, _ = recompute(
                    block,
                    streams,
                    mix,
                    (latent, index_keys, selected, candidates),
                    no_engram,
                    progress=lambda message: event("recompute: " + message),
                    frozen_inputs=args.frozen_inputs,
                )
                final = rms_norm(hc_pre(a, b), model.norm)
                return mx.sum(final.astype(mx.float32) * readout.astype(mx.float32)) / math.sqrt(
                    final.size
                )

            event("block_forward_and_vjp_started")
            try:
                losses, gradients = mx.vjp(objective, primals, [mx.array(1.0, dtype=mx.float32)])
                mx.eval(losses, gradients)
            finally:
                block.update(original)
            event("block_vjp_complete")
            budget.check()
            loss = losses[0].item()
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite surrogate loss")
            names = [name for name, _ in leaves] + ["input.h", "input.pre", "input.shared_kv"]
            summary = {
                name: gradient_summary(value) for name, value in zip(names, gradients, strict=True)
            }
            if not any(summary[name]["nonzero"] for name, _ in leaves):
                raise FloatingPointError("All final-block adapter gradients were zero")
            input_nonzero = [
                summary[name]["nonzero"] for name in ("input.h", "input.pre", "input.shared_kv")
            ]
            if args.frozen_inputs:
                if any(input_nonzero):
                    raise AssertionError("Frozen input cotangents must all be zero")
            elif not all(input_nonzero):
                raise FloatingPointError("An expected upstream state gradient was entirely zero")
            for (name, value), (expected_name, expected_value) in zip(
                tree_flatten(block.trainable_parameters()), leaves, strict=True
            ):
                if name != expected_name or value is not expected_value:
                    raise AssertionError("Original parameter objects were not restored")
            # These references must still be live when the final memory is read.
            report["retained_padding_bytes_at_end"] = sum(v.nbytes for v in padding)
            report.update(
                status="gradients_verified",
                surrogate_loss=loss,
                gradients=summary,
                final_cache_stats=model._store.report(),
            )
            event("gradients_verified")
            budget.check()
        # Publish success after the watchdog has stopped, so an abort cannot
        # leave a success receipt from just before a final guard check.
        report.update(
            status="passed",
            acceptance=(
                "One synthetic final-block VJP completed with finite nonzero adapter gradients and "
                + (
                    "zero frozen-input cotangents"
                    if args.frozen_inputs
                    else "nonzero upstream gradients"
                )
                + " under the requested guard; no full-training claim."
            ),
        )
        event("probe_passed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        event("probe_failed")
        raise
    return report


def main():
    args = parser().parse_args()
    # Resource parsing is CPU-only and shared with the actual training workflow.
    from exl3_mlx.resources import resolve_resources

    resources = resolve_resources(
        "development",
        memory_gib=args.memory_gib,
        reserve_gib=args.reserve_gib,
        packed_cache_mib=args.packed_cache_mib,
        decoded_cache_mib=args.decoded_cache_mib,
        allocator_cache_mib=args.allocator_cache_mib,
    )
    plan = read_plan(args)
    if args.describe_only:
        print(json.dumps({"plan": plan, "resources": resources}, indent=2))
        return
    run(args, plan, resources)


if __name__ == "__main__":
    main()
