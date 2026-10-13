#!/usr/bin/env python3
"""Observe final-block BF16 adapter additions using an immutable prefix capture.

Run AFTER the matching frozen-prefix training experiment finishes:
  python diagnose_adapter_rounding.py --runtime PATH/runtime-v1 \
    --experiment PATH/last-block-original-row --output NEW_DIRECTORY

The runtime supplies the existing final-block evaluation algorithm. Four suffix
scores run: initial/unobserved, initial/observed, step1/unobserved, step1/observed.
Observer wrappers call the original layers and return their original arrays.
Exact score/logit-hash parity is required before interpreting observations.
Normal MoE routes are recomputed independently; they are never frozen.

This is a training-row arithmetic diagnostic, not an optimization-quality test
or a causal proof that rounding explains a loss change. No optimizer is run.
--describe-only and --self-test are CPU-only and do not import MLX.
"""

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def checked_file(root, name, checksum):
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe receipt path: {name}")
    if not path.is_file() or digest(path) != checksum:
        raise ValueError(f"Checksum mismatch: {path}")
    return path


def load_helper(runtime):
    if any(name == "exl3_mlx" or name.startswith("exl3_mlx.") for name in sys.modules):
        raise RuntimeError("Run in a fresh Python process; exl3_mlx is already imported")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(runtime / "src"))
    path = runtime / "scripts/diagnose_frozen_prefix.py"
    spec = importlib.util.spec_from_file_location("pinned_prefix_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    for name in ("runtime", "experiment", "output"):
        value = getattr(args, name)
        if value is None:
            raise ValueError(f"--{name} is required")
        setattr(args, name, value.expanduser().resolve())
    if args.output.exists():
        raise ValueError("Output must be a new directory")
    if args.output.is_relative_to(args.runtime) or args.output.is_relative_to(args.experiment):
        raise ValueError("Output must be outside the immutable runtime and experiment")
    snapshot = json.loads((args.runtime / "snapshot.json").read_text())
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("Missing runtime snapshot checksums")
    for name, checksum in snapshot.items():
        checked_file(args.runtime, name, checksum)
    helper_name = "scripts/diagnose_frozen_prefix.py"
    if helper_name not in snapshot:
        raise ValueError("Snapshot must bind the original diagnostic script")
    diagnostic = json.loads((args.experiment / "diagnostic.json").read_text())
    if diagnostic.get("schema") != "frozen-prefix-training-diagnostic-v1":
        raise ValueError("Expected the frozen-prefix training diagnostic")
    settings = diagnostic["settings"]
    if settings.get("resume") is not None:
        raise ValueError("This first-step diagnostic requires a fresh, non-resumed experiment")
    if diagnostic["script_sha256"] != snapshot[helper_name]:
        raise ValueError("Training script differs from the pinned runtime")
    args.model = Path(settings["model"]).resolve()
    args.data = Path(settings["data"]).resolve()
    if args.output.is_relative_to(args.model):
        raise ValueError("Output must be outside the frozen base model")
    args.row = settings["rows"][0] if args.row is None else args.row
    if args.row != settings["rows"][0]:
        raise ValueError("Use the row trained at step1 for the first-order comparison")
    args.rank, args.alpha, args.chunk_size = (
        settings[key] for key in ("rank", "alpha", "chunk_size")
    )
    args.prefix_cache = Path(
        settings.get("prefix_cache") or args.experiment / "prefix-cache"
    ).resolve()
    args.step_adapter = args.experiment / "step-000001"
    helper = load_helper(args.runtime)
    row = helper.selected_rows(args.data, [args.row])[args.row]
    description = {
        "runtime": str(args.runtime),
        "runtime_snapshot_sha256": digest(args.runtime / "snapshot.json"),
        "experiment": str(args.experiment),
        "training_status": diagnostic.get("status"),
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": digest(args.data),
        "row": args.row,
        "serialized_tokens": len(row["input_ids"]),
        "supervised_tokens": len(row["input_ids"]) - row["loss_start"],
        "adapter_layers": [39],
        "rank": args.rank,
        "alpha": args.alpha,
        "chunk_size": args.chunk_size,
        "evaluations": 4,
        "normal_router_selections": True,
        "requires_completed_experiment": True,
        "no_optimizer_update": True,
    }
    if args.describe_only:
        return helper, row, diagnostic, description
    if diagnostic.get("status") != "completed":
        raise ValueError("Wait until the experiment completes before running this GPU diagnostic")
    cache = json.loads((args.prefix_cache / "prefix-cache.json").read_text())
    # All cache checks and source/runtime/model/data binding are repeated in run().
    if cache.get("status") != "complete":
        raise ValueError("Prefix capture is not complete")
    state = json.loads((args.step_adapter / "trainer_state.json").read_text())
    if state.get("format") != "exl3-mlx-training-state-v1" or state.get("step") != 1:
        raise ValueError("Expected the exact saved first optimizer step")
    for name, checksum in state["files"].items():
        checked_file(args.step_adapter, name, checksum)
    required = {
        "adapter.safetensors",
        "adapter_config.json",
        "optimizer.safetensors",
        "random.safetensors",
    }
    if set(state["files"]) != required:
        raise ValueError("Incomplete first-step checkpoint receipt")
    contract = state["contract"]
    if (
        contract.get("diagnostic") != "frozen-prefix-v1"
        or contract.get("prefix_cache_sha256") != digest(args.prefix_cache / "prefix-cache.json")
        or contract.get("binding") != cache["binding"]
        or contract.get("rows") != settings["rows"]
        or state["metrics"].get("row") != args.row
    ):
        raise ValueError("First-step checkpoint/cache/training-row contract mismatch")
    description["trainer_state_sha256"] = digest(args.step_adapter / "trainer_state.json")
    description["adapter_sha256"] = digest(args.step_adapter / "adapter.safetensors")
    description["prefix_manifest_sha256"] = digest(args.prefix_cache / "prefix-cache.json")
    description["training_diagnostic_sha256"] = digest(args.experiment / "diagnostic.json")
    return helper, row, diagnostic, description


def bf16_float(bits):
    import numpy as np

    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def arithmetic_counts(base_bits, output_bits, residual, cast_residual):
    """Pure NumPy, one bounded row tile; counts use exact BF16 bit comparisons."""
    import numpy as np

    if not (base_bits.shape == output_bits.shape == residual.shape == cast_residual.shape):
        raise ValueError("Arithmetic observation shapes differ")
    base, output = bf16_float(base_bits), bf16_float(output_bits)
    if not all(np.isfinite(v).all() for v in (base, output, residual, cast_residual)):
        raise FloatingPointError("Nonfinite arithmetic observation")
    unchanged = base_bits == output_bits
    nonzero = residual != 0
    cast_nonzero = cast_residual != 0
    ideal = base.astype(np.float64) + residual.astype(np.float64)
    actual_delta = output.astype(np.float64) - base.astype(np.float64)
    return {
        "elements": int(base.size),
        "fp32_residual_nonzero": int(np.count_nonzero(nonzero)),
        "cast_residual_nonzero": int(np.count_nonzero(cast_nonzero)),
        "nonzero_fp32_cast_to_zero": int(np.count_nonzero(nonzero & ~cast_nonzero)),
        "nonzero_cast_absorbed_by_addition": int(np.count_nonzero(cast_nonzero & unchanged)),
        "nonzero_fp32_without_output_bit_change": int(np.count_nonzero(nonzero & unchanged)),
        "output_bits_changed_from_base": int(np.count_nonzero(~unchanged)),
        "base_sum_squared": float(np.sum(base.astype(np.float64) ** 2)),
        "fp32_residual_sum_squared": float(np.sum(residual.astype(np.float64) ** 2)),
        "cast_residual_sum_squared": float(np.sum(cast_residual.astype(np.float64) ** 2)),
        "actual_output_delta_sum_squared": float(np.sum(actual_delta**2)),
        "output_minus_unrounded_sum_squared": float(np.sum((output - ideal) ** 2)),
    }


def finish_counts(counts):
    counts = dict(counts)
    base_norm = math.sqrt(counts["base_sum_squared"])
    for key in ("base", "fp32_residual", "cast_residual", "actual_output_delta"):
        counts[key + "_l2_norm"] = math.sqrt(counts[key + "_sum_squared"])
    counts["fp32_residual_over_base_l2"] = (
        counts["fp32_residual_l2_norm"] / base_norm if base_norm else None
    )
    nonzero = counts["fp32_residual_nonzero"]
    counts["fraction_nonzero_fp32_without_output_bit_change"] = (
        counts["nonzero_fp32_without_output_bit_change"] / nonzero if nonzero else None
    )
    return counts


def router_difference(before, after):
    import numpy as np

    bi, bw = before["indices"], before["weights"]
    ai, aw = after["indices"], after["weights"]
    if bi.shape != ai.shape or bw.shape != aw.shape or bi.shape != bw.shape:
        raise ValueError("Router observation shapes differ")
    ordered = np.any(bi != ai, axis=-1)
    sets = np.any(np.sort(bi, axis=-1) != np.sort(ai, axis=-1), axis=-1)
    same_order = ~ordered
    return {
        "token_rows": len(bi),
        "top_k": bi.shape[-1],
        "changed_ordered_route_rows": int(np.count_nonzero(ordered)),
        "changed_expert_set_rows": int(np.count_nonzero(sets)),
        "changed_route_slots": int(np.count_nonzero(bi != ai)),
        "same_order_route_rows": int(np.count_nonzero(same_order)),
        "same_order_weight_delta_l2": float(
            np.linalg.norm((aw[same_order] - bw[same_order]).astype(np.float64))
        ),
        "weight_comparison_scope": "Only rows with identical ordered expert IDs; no mismatched-slot comparison",
        "routes_recomputed_normally": True,
    }


def first_order_report(args, diagnostic, measured_delta):
    import numpy as np
    from safetensors.numpy import load_file

    receipt = diagnostic.get("first_step_gradients")
    if not receipt:
        return {"available": False, "reason": "No saved first-step gradient receipt"}
    if receipt.get("step") != 1:
        raise ValueError("Saved gradient is not for the first step")
    gradient_path = checked_file(
        args.experiment, "first-step-gradients.safetensors", receipt["sha256"]
    )
    before = load_file(str(args.prefix_cache / "initial-adapter/adapter.safetensors"))
    after = load_file(str(args.step_adapter / "adapter.safetensors"))
    gradients = load_file(str(gradient_path))
    if set(before) != set(after) or set(before) != set(gradients) or len(before) != 4:
        raise ValueError("First-order comparison tensor keys differ")
    details = {}
    for name, g in gradients.items():
        if (
            not name.startswith("blocks.39.")
            or g.shape != before[name].shape
            or g.shape != after[name].shape
        ):
            raise ValueError("First-order comparison tensor scope/shape differs")
        if any(
            v.dtype != np.float32 or not np.isfinite(v).all()
            for v in (g, before[name], after[name])
        ):
            raise ValueError("Expected finite FP32 adapters and gradients")
        delta = after[name].astype(np.float64) - before[name].astype(np.float64)
        details[name] = {
            "gradient_dot_parameter_delta": float(np.sum(g.astype(np.float64) * delta)),
            "parameter_delta_l2": float(np.linalg.norm(delta)),
            "gradient_l2": float(np.linalg.norm(g.astype(np.float64))),
        }
    predicted = sum(v["gradient_dot_parameter_delta"] for v in details.values())
    return {
        "available": True,
        "gradient_sha256": digest(gradient_path),
        "gradient_scope": receipt["scope"],
        "predicted_first_order_loss_delta": predicted,
        "measured_loss_delta": measured_delta,
        "measured_minus_first_order": measured_delta - predicted,
        "tensors": details,
        "interpretation": "g dot the actual saved parameter update, accumulated on CPU in FP64. The gradient includes identity STE surrogates and locally fixed discrete selections. Nonlinearity, changed routes and finite-precision rounding can all invalidate first-order descent; disagreement alone does not identify a cause.",
    }


@contextlib.contextmanager
def observe_block(model, trace, check):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from exl3_mlx.native import LoRALinear

    block = model.blocks[39]
    replacements = []

    class BaseObserver(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self._outputs = []

        def __call__(self, x):
            result = self.original(x)
            self._outputs.append(result)
            return result

    class AdapterObserver(nn.Module):
        def __init__(self, original, name):
            super().__init__()
            self.original = original
            self._name = name

        def __call__(self, x):
            original = self.original
            if self._name in trace["linears"]:
                raise AssertionError("Expected one call per final-block adapter")
            if original.training or original.dropout.training:
                raise AssertionError("Observe deterministic evaluation only")
            base_original = original.linear
            capture = BaseObserver(base_original)
            original.linear = capture
            try:
                actual = original(x)  # The existing upstream MLX-LM __call__, unchanged.
            finally:
                original.linear = base_original
            if len(capture._outputs) != 1:
                raise AssertionError("Expected one unchanged frozen-base call")
            base = capture._outputs[0]
            # Auxiliary reconstruction is observed only; it never replaces a model array.
            residual = original.scale * ((original.dropout(x) @ original.lora_a) @ original.lora_b)
            cast = residual.astype(x.dtype)
            reconstructed = base + cast
            mx.eval(base, actual, residual, cast, reconstructed)
            if (
                any(value.dtype != mx.bfloat16 for value in (x, base, actual, cast))
                or residual.dtype != mx.float32
            ):
                raise AssertionError(
                    "This diagnostic expects BF16 input/base/addition and FP32 adapters"
                )
            if not mx.array_equal(actual.view(mx.uint16), reconstructed.view(mx.uint16)).item():
                raise AssertionError(
                    "Auxiliary existing-form reconstruction differs from original output"
                )
            width = actual.shape[-1]
            bases, outputs, residuals, casts = (
                value.reshape(-1, width) for value in (base, actual, residual, cast)
            )
            counts, base_hash, result_hash = {}, hashlib.sha256(), hashlib.sha256()
            output_parts = []
            for start in range(0, outputs.shape[0], 128):
                check()
                end = start + 128
                b = np.array(bases[start:end].view(mx.uint16))
                o = np.array(outputs[start:end].view(mx.uint16))
                r = np.array(residuals[start:end])
                c = np.array(casts[start:end].astype(mx.float32))
                values = arithmetic_counts(b, o, r, c)
                for key, value in values.items():
                    counts[key] = counts.get(key, 0) + value
                base_hash.update(b.tobytes())
                result_hash.update(o.tobytes())
                output_parts.append(o)
            trace["linears"][self._name] = {
                "shape": list(actual.shape),
                "input_dtype": str(x.dtype),
                "residual_dtype": str(residual.dtype),
                "base_dtype": str(base.dtype),
                "addition_dtype": str(actual.dtype),
                "base_bits_sha256": base_hash.hexdigest(),
                "result_bits_sha256": result_hash.hexdigest(),
                "existing_form_reconstruction_exact": True,
                "counts": finish_counts(counts),
            }
            trace["outputs"][self._name] = np.concatenate(output_parts, axis=0)
            return actual

    class RouterObserver(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def __call__(self, x):
            if trace.get("router") is not None:
                raise AssertionError("Expected one router call during evaluation")
            weights, indices = self.original(x)  # No alternate or frozen selection.
            mx.eval(weights, indices)
            w, i = np.array(weights), np.array(indices)
            if not np.isfinite(w).all():
                raise FloatingPointError("Nonfinite observed router weights")
            trace["router"] = {"weights": w, "indices": i}
            return weights, indices

    try:
        selected = {
            key: value for key, value in block.attn.linears.items() if isinstance(value, LoRALinear)
        }
        if set(selected) != {"wq_a", "wo_b"}:
            raise ValueError("Expected exactly final-block wq_a and wo_b adapters")
        for name, original in selected.items():
            replacements.append((block.attn.linears, name, original))
            block.attn.linears[name] = AdapterObserver(original, name)
        original_gate = block.ffn.gate
        replacements.append((block.ffn, "gate", original_gate))
        block.ffn.gate = RouterObserver(original_gate)
        yield
        if set(trace["linears"]) != {"wq_a", "wo_b"} or trace.get("router") is None:
            raise AssertionError("Incomplete final-block observation")
    finally:
        for container, name, original in reversed(replacements):
            container[name] = original


def run(args, helper, row, diagnostic, description):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear as UpstreamLoRALinear

    from exl3_mlx.context import ENGINE_SEMANTICS
    from exl3_mlx.deepseek import DeepSeekTextModel
    from exl3_mlx.memory import MemoryBudget
    from exl3_mlx.train import load_adapter, load_adapter_config

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "final-block-adapter-arithmetic-v1",
        "status": "starting",
        "description": description,
        "script_sha256": digest(__file__),
        "upstream_lora_source_sha256": digest(inspect.getfile(UpstreamLoRALinear)),
        "conditions": {},
        "quality_certified": False,
        "causal_rounding_explanation_established": False,
        "scope": "Observed arithmetic on one actual training prefix. No changed arithmetic intervention, no route freezing, no optimizer update, no new full-model parity claim.",
        "limitations": [
            "Absorbed residual counts do not alone identify the cause of a loss change.",
            "This observes the final adapted block, not activation quantizers elsewhere or all40-layer adapters.",
            "Scores use the existing exact-base-validated prefix; updated full-model parity remains the original experiment's separate test.",
            "Layer output bit differences across conditions include upstream changes within this block; compare absorption to each condition's own frozen-base output.",
            "Statistics cover every serialized input row, including prompt positions, not only supervised answer positions.",
        ],
    }
    result_path = args.output / "arithmetic.json"
    write_json(result_path, report)

    def event(message):
        print(f"[{time.monotonic() - started:.1f}s] {message}", flush=True)

    try:
        binding = helper.binding_for(args)
        cache = json.loads((args.prefix_cache / "prefix-cache.json").read_text())
        helper.validate_cache_manifest(cache, binding, [args.row], args.prefix_cache)
        for path in (args.prefix_cache / "initial-adapter", args.step_adapter):
            config = load_adapter_config(path)
            if config.get("engine_semantics") != ENGINE_SEMANTICS:
                raise ValueError("Adapter engine semantics differ from pinned runtime")
        with MemoryBudget(
            args.memory_gib, cache_mib=args.allocator_cache_mib, reserve_gib=args.reserve_gib
        ) as budget:
            budget.abort_path = args.output / "memory-abort.json"
            model = DeepSeekTextModel(
                args.model,
                rank=args.rank,
                alpha=args.alpha,
                adapter_layers=[39],
                expert_adapter_layers=[],
                chunk_size=args.chunk_size,
                cache_mib=args.packed_cache_mib,
                decoded_cache_mib=0,
                progress=event,
            )
            prefix = mx.load(str(args.prefix_cache / f"row-{args.row:04d}.safetensors"))
            mx.eval(prefix)
            traces = {}
            for label, directory in (
                ("initial_zero_B", args.prefix_cache / "initial-adapter"),
                ("step1", args.step_adapter),
            ):
                event(f"{label}: loading exact saved adapters")
                load_adapter(model, directory)
                if label == "initial_zero_B" and any(
                    mx.count_nonzero(value).item()
                    for key, value in tree_flatten(model.trainable_parameters())
                    if key.endswith("lora_b")
                ):
                    raise AssertionError("Initial adapter B must be exactly zero")
                event(f"{label}: unobserved suffix evaluation")
                unobserved = helper.evaluate_cached(model, prefix, row, check=budget.check)
                if (
                    label == "initial_zero_B"
                    and unobserved != cache["rows"][str(args.row)]["cached_base"]
                ):
                    raise AssertionError("Replayed baseline no longer matches prefix capture")
                if label == "step1":
                    original_step = next(
                        (item for item in diagnostic["history"] if item["step"] == 1), None
                    )
                    if original_step is None:
                        raise ValueError("Training report lacks the first-step score")
                    recorded = original_step["post_update_scores"][str(args.row)]
                    if any(recorded[key] != value for key, value in unobserved.items()):
                        raise AssertionError(
                            "First-step replay differs from its original score/logits"
                        )
                trace = {"linears": {}, "outputs": {}}
                event(f"{label}: observed suffix evaluation")
                with observe_block(model, trace, budget.check):
                    observed = helper.evaluate_cached(model, prefix, row, check=budget.check)
                if observed != unobserved:
                    raise AssertionError(
                        f"{label}: observer changed score/logits; discard arithmetic conclusions"
                    )
                router_path = args.output / f"{label}-router.npz"
                np.savez(router_path, **trace["router"])
                report["conditions"][label] = {
                    "adapter_sha256": digest(directory / "adapter.safetensors"),
                    "unobserved": unobserved,
                    "observed": observed,
                    "observer_output_parity_exact": True,
                    "original_experiment_score_parity_exact": True,
                    "linears": trace["linears"],
                    "router_file": router_path.name,
                    "router_file_sha256": digest(router_path),
                }
                traces[label] = trace
                write_json(result_path, report)
            before, after = traces["initial_zero_B"], traces["step1"]
            report["router_comparison"] = router_difference(before["router"], after["router"])
            report["linear_output_comparison"] = {}
            for name, original in before["outputs"].items():
                updated = after["outputs"][name]
                if original.shape != updated.shape:
                    raise AssertionError("Compared output shapes differ")
                report["linear_output_comparison"][name] = {
                    "elements": int(original.size),
                    "changed_elements": int(np.count_nonzero(original != updated)),
                    "rows_with_changed_elements": int(
                        np.count_nonzero(np.any(original != updated, axis=-1))
                    ),
                }
            delta = (
                report["conditions"]["step1"]["unobserved"]["loss"]
                - report["conditions"]["initial_zero_B"]["unobserved"]["loss"]
            )
            report["first_order"] = first_order_report(args, diagnostic, delta)
            report["measured_loss_delta"] = delta
            report["binding"] = binding
            report["resources"] = {
                key: getattr(args, key)
                for key in ("memory_gib", "reserve_gib", "packed_cache_mib", "allocator_cache_mib")
            }
            report["peak_mlx_gib"] = mx.get_peak_memory() / 2**30
            report["observed_process_gib"] = budget.usage() / 2**30
            report["elapsed_seconds"] = time.monotonic() - started
            report["status"] = "completed"
            write_json(result_path, report)
            event(f"Completed observational diagnostic; measured NLL delta={delta:+.9f}")
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(result_path, report)
        raise


def self_test():
    import tempfile

    import numpy as np

    base = np.array([1.0, 1.0, 0.0], dtype=np.float32).view(np.uint32)
    bits = (base >> 16).astype(np.uint16)
    residual = np.array([0.001, 0.01, 1e-42], dtype=np.float32)
    cast = np.array([0.00099945068359375, 0.010009765625, 0.0], dtype=np.float32)
    output = np.array([1.0, 1.0078125, 0.0], dtype=np.float32)
    outbits = (output.view(np.uint32) >> 16).astype(np.uint16)
    result = finish_counts(arithmetic_counts(bits, outbits, residual, cast))
    assert result["fp32_residual_nonzero"] == 3
    assert result["nonzero_fp32_cast_to_zero"] == 1
    assert result["nonzero_cast_absorbed_by_addition"] == 1
    assert result["nonzero_fp32_without_output_bit_change"] == 2
    assert result["output_bits_changed_from_base"] == 1
    assert (
        finish_counts(arithmetic_counts(bits, bits, residual * 0, cast * 0))[
            "fraction_nonzero_fp32_without_output_bit_change"
        ]
        is None
    )
    before = {"indices": np.array([[1, 2], [3, 4]]), "weights": np.ones((2, 2), dtype=np.float32)}
    after = {"indices": np.array([[2, 1], [3, 5]]), "weights": before["weights"].copy()}
    routes = router_difference(before, after)
    assert routes["changed_ordered_route_rows"] == 2 and routes["changed_expert_set_rows"] == 1
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "receipt.json"
        write_json(path, {"ok": True})
        assert checked_file(root, path.name, digest(path)) == path
        try:
            checked_file(root, "../outside", "x")
        except ValueError:
            pass
        else:
            raise AssertionError("Traversal should be rejected")
    print("CPU-only arithmetic/router/receipt helper checks passed; GPU observers unexecuted.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--row", type=int, help="Defaults to the first training row; other rows are rejected"
    )
    parser.add_argument("--memory-gib", type=float, default=80)
    parser.add_argument("--reserve-gib", type=float, default=12)
    parser.add_argument("--packed-cache-mib", type=int, default=8192)
    parser.add_argument("--allocator-cache-mib", type=int, default=256)
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    helper, row, diagnostic, description = prepare(args)
    if args.describe_only:
        print(json.dumps(description, indent=2))
        return
    run(args, helper, row, diagnostic, description)


if __name__ == "__main__":
    main()
