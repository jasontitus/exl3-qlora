"""Observe final-block memory phases without modifying engine arithmetic.

Run serially like probe_final_block_memory.py; all its flags are supported.
Writes OUTPUT.phases.jsonl and OUTPUT.phase-plan.json exclusively beside the
original probe output. Sampled before/after observations are not per-op peaks.
The wrappers add no eval, materialize, tensor copies, or retained tensor values.
No MLX import or GPU work occurs until main() runs the actual diagnostic.
"""

import contextlib
import functools
import hashlib
import json
import time
from pathlib import Path


@contextlib.contextmanager
def instrument(specifications, emit):
    """Temporarily wrap callables; descriptors contain only shapes/scalars."""
    originals, counts = [], {}
    try:
        for owner, attribute, classify, every in specifications:
            original = getattr(owner, attribute)

            def make_wrapper(original, classify, every):
                @functools.wraps(original)
                def wrapped(*args, **kwargs):
                    label, metadata = classify(args, kwargs)
                    count = counts.get(label, 0) + 1
                    counts[label] = count
                    sampled = count == 1 or count % every == 0
                    if sampled:
                        emit(label + ".enter", call=count, **metadata)
                    try:
                        result = original(*args, **kwargs)
                    except BaseException as error:
                        emit(label + ".error", call=count, error=type(error).__name__, **metadata)
                        raise
                    if sampled:
                        emit(label + ".exit", call=count, **metadata)
                    return result

                return wrapped

            originals.append((owner, attribute, original))
            setattr(owner, attribute, make_wrapper(original, classify, every))
        yield counts
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def main():
    import probe_final_block_memory as probe

    parser = probe.parser()
    parser.description = __doc__
    parser.add_argument("--sample-every", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.sample_every <= 4096:
        parser.error("sample-every must be1..4096")
    from exl3_mlx.resources import resolve_resources

    plan = probe.read_plan(args)
    resources = resolve_resources(
        "development",
        memory_gib=args.memory_gib,
        reserve_gib=args.reserve_gib,
        packed_cache_mib=args.packed_cache_mib,
        decoded_cache_mib=args.decoded_cache_mib,
        allocator_cache_mib=args.allocator_cache_mib,
    )
    if args.describe_only:
        print(json.dumps({"plan": plan, "resources": resources}, indent=2))
        return
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose new probe output")
    output.parent.mkdir(parents=True, exist_ok=True)
    phases = output.with_name(output.name + ".phases.jsonl")
    declaration = output.with_name(output.name + ".phase-plan.json")
    if phases.exists() or declaration.exists():
        raise FileExistsError("Choose fresh diagnostic receipts")
    with declaration.open("x") as handle:
        json.dump(
            {
                "scope": "Process-local observer wrappers; original engine operations and probe unchanged",
                "limitations": "Observations do not synchronize Metal and cannot prove a precise individual allocating operation; no memory-win or full-training claim",
                "settings": vars(args),
                "plan": plan,
                "scripts": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).resolve(), Path(probe.__file__).resolve())
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    import mlx.core as mx

    import exl3_mlx.attention as attention
    import exl3_mlx.deepseek as deepseek
    import exl3_mlx.moe as moe
    import exl3_mlx.native as native
    import exl3_mlx.sparse_attention as sparse
    import exl3_mlx.streaming as streaming
    from exl3_mlx.memory import MemoryBudget

    started = time.monotonic()

    def shape(value):
        return list(value.shape) if hasattr(value, "shape") else None

    def named(label, index=0):
        return lambda args, kwargs: (label, {"input_shape": shape(args[index])})

    specs = [
        (attention.ShortAttention, "__call__", named("attention", 1), 1),
        (moe.DeepSeekMoE, "__call__", named("moe", 1), 1),
        (
            native.NativeLinear,
            "__call__",
            lambda a, k: (
                "native_projection",
                {"input_shape": shape(a[1]), "output_features": a[0].out_features},
            ),
            1,
        ),
        (
            attention,
            "quantize_activation",
            lambda a, k: (
                "activation_quantize",
                {"input_shape": shape(a[0]), "kind": a[1] if len(a) > 1 else k.get("kind", "fp8")},
            ),
            1,
        ),
        (attention, "rope", named("rope"), 1),
        (deepseek, "hc_pre", named("hc_pre"), 1),
        (deepseek, "hc_post", named("hc_post"), 1),
        (
            sparse,
            "_tile",
            lambda a, k: ("sparse_tile", {"query_shape": shape(a[0]), "query_start": int(a[5])}),
            args.sample_every,
        ),
        (
            sparse,
            "_scatter",
            lambda a, k: (
                "sparse_scatter",
                {"accumulator_shape": shape(a[0]), "indices_shape": shape(a[1])},
            ),
            args.sample_every,
        ),
        (
            moe,
            "_component_vjp",
            lambda a, k: (
                "component_vjp_" + type(a[0]).__name__,
                {"input_shapes": [shape(v) for v in a[1]]},
            ),
            max(1, args.sample_every // 2),
        ),
        (
            streaming,
            "_base",
            lambda a, k: (
                "streamed_backward" if k.get("transpose", False) else "streamed_forward",
                {"input_shape": shape(a[0])},
            ),
            args.sample_every,
        ),
    ]
    with phases.open("x", buffering=1) as handle:

        def emit(phase, **extra):
            item = {
                "phase": phase,
                "elapsed_seconds": time.monotonic() - started,
                "active_mlx_gib": mx.get_active_memory() / 2**30,
                "allocator_cache_gib": mx.get_cache_memory() / 2**30,
                "peak_mlx_gib": mx.get_peak_memory() / 2**30,
                "observed_process_gib": MemoryBudget.usage() / 2**30,
                **extra,
            }
            handle.write(json.dumps(item, allow_nan=False) + "\n")
            handle.flush()

        emit("diagnostic.started")
        with instrument(specs, emit) as counts:
            report = probe.run(args, plan, resources)
        emit("diagnostic.completed", counts=counts, probe_status=report["status"])


if __name__ == "__main__":
    main()
