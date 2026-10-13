#!/usr/bin/env python3
"""Train real block39 adapters against an exact, frozen full-context prefix.

Capture once (GPU): --model MODEL --data PREPARED --rows 1 --output NEW_CACHE
  --initial-adapter ORIGINAL_STEP1 --capture-only
Train (GPU): same model/data/rows, --prefix-cache NEW_CACHE --output NEW_RUN
  --steps 8 --learning-rate 1e-4

Only block39 adapters train. Prefix states before block39 are immutable full
sequence states, not truncated context. Block39 attention and MoE routing execute
normally on every call. A cache is usable only after exact uncached/cached base
logit parity, and matching source/runtime, model metadata/payload stats and data.
--describe-only is CPU-only. Training-row fitting is not held-out kernel quality.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def selected_rows(data, rows):
    values = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    if (
        not rows
        or len(set(rows)) != len(rows)
        or any(type(i) is not int or not 0 <= i < len(values) for i in rows)
    ):
        raise ValueError("Select unique existing zero-based row indices")
    selected = {}
    for index in rows:
        row = values[index]
        if set(row) != {"input_ids", "loss_start"}:
            raise ValueError("Use prepared input_ids/loss_start rows only")
        if (
            type(row["loss_start"]) is not int
            or not 1 <= row["loss_start"] < len(row["input_ids"])
            or not 2 <= len(row["input_ids"]) <= 32768
            or any(type(t) is not int or t < 0 for t in row["input_ids"])
        ):
            raise ValueError("Invalid prepared row")
        selected[index] = row
    return selected


def validate_cache_manifest(manifest, binding, required_rows, directory):
    if manifest.get("schema") != "frozen-prefix-cache-v1" or manifest.get("status") != "complete":
        raise ValueError("Prefix capture is not complete")
    if manifest.get("binding") != binding:
        raise ValueError("Prefix cache source/runtime/model/data/adapter configuration differs")
    records = manifest.get("rows", {})
    if any(str(i) not in records for i in required_rows):
        raise ValueError("Prefix cache lacks a requested row")
    for name, checksum in manifest.get("files", {}).items():
        path = Path(directory) / name
        if path.is_symlink() or not path.resolve().is_relative_to(Path(directory).resolve()):
            raise ValueError("Unsafe prefix cache file")
        if digest(path) != checksum:
            raise ValueError("Prefix cache checksum mismatch")
    required = {"initial-adapter/adapter.safetensors", "initial-adapter/adapter_config.json"}
    required.update(f"row-{i:04d}.safetensors" for i in required_rows)
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("Prefix cache has incomplete checksums")
    for index in required_rows:
        row = records[str(index)]
        if not row.get("base_parity_exact") or row["uncached_base"] != row["cached_base"]:
            raise ValueError("Prefix cache lacks exact base parity")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument(
        "--rows",
        type=int,
        nargs="+",
        required=True,
        help="Training rows, zero-based; cycle one per step",
    )
    p.add_argument(
        "--eval-rows",
        type=int,
        nargs="*",
        default=[],
        help="Extra diagnostic rows, not asserted independent",
    )
    p.add_argument("--output", type=Path, required=True, help="New directory only")
    p.add_argument("--prefix-cache", type=Path)
    p.add_argument("--initial-adapter", type=Path, help="Import block39 A only; B is zeroed")
    p.add_argument("--capture-only", action="store_true")
    p.add_argument("--steps", type=int, default=8, help="Additional optimizer steps")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--bias-correction",
        action="store_true",
        help="Explicit Adam option; default false matches existing engine",
    )
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--memory-gib", type=float, default=80)
    p.add_argument("--reserve-gib", type=float, default=12)
    p.add_argument("--packed-cache-mib", type=int, default=8192)
    p.add_argument("--decoded-cache-mib", type=int, default=0)
    p.add_argument("--allocator-cache-mib", type=int, default=256)
    p.add_argument(
        "--resume",
        type=Path,
        help="Exact diagnostic optimizer checkpoint; output still must be new",
    )
    p.add_argument(
        "--verify-uncached-final", action="store_true", help="Extra full40-layer final evaluation"
    )
    p.add_argument("--include-token-metrics", action="store_true")
    p.add_argument("--describe-only", action="store_true")
    return p


def prepare(args):
    args.model, args.data, args.output = (
        p.expanduser().resolve() for p in (args.model, args.data, args.output)
    )
    if args.output.is_relative_to(args.model):
        raise ValueError("Output must be outside the frozen model directory")
    for key in ("prefix_cache", "initial_adapter", "resume"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.output.exists():
        raise ValueError("Output must be a new directory")
    if args.prefix_cache and (args.capture_only or args.initial_adapter):
        raise ValueError(
            "Existing prefix cache supplies initial adapters; do not combine with capture/initial-adapter"
        )
    if args.resume and not args.prefix_cache:
        raise ValueError("Resume requires its verified prefix cache")
    if (
        args.steps < 1
        or args.rank < 1
        or any(
            not math.isfinite(v) or v <= 0 for v in (args.alpha, args.learning_rate, args.clip_grad)
        )
    ):
        raise ValueError("Use positive steps/rank and finite positive alpha/learning-rate/clip")
    if not 128 <= args.chunk_size <= 32768 or args.chunk_size % 128:
        raise ValueError("chunk-size must be a multiple of128 in128..32768")
    if len(set(args.rows)) != len(args.rows) or len(set(args.eval_rows)) != len(args.eval_rows):
        raise ValueError("Rows must be unique")
    requested = list(dict.fromkeys(args.rows + args.eval_rows))
    rows = selected_rows(args.data, requested)
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    if (
        config["num_hidden_layers"] != 40
        or config["compress_ratios"][39] != 1
        or any(
            39 in config[key]
            for key in ("kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids")
        )
    ):
        raise ValueError("Requires final block39 without an indexer/compressor/Engram")
    if any(
        t >= config["vocab_size"] or t == 129264 for row in rows.values() for t in row["input_ids"]
    ):
        raise ValueError("Invalid text token IDs")
    return rows


def binding_for(args):
    from exl3_mlx.sft import implementation_identity
    from exl3_mlx.train import checkpoint_identity

    payloads = {}
    for path in sorted(args.model.rglob("*")):
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".u32"):
            stat = path.stat()
            payloads[str(path.relative_to(args.model))] = [
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ino,
            ]
    return {
        "diagnostic_script_sha256": digest(__file__),
        "implementation": implementation_identity(),
        "model": str(args.model),
        "model_metadata": checkpoint_identity(args.model),
        "payload_stat_fingerprints": payloads,
        "payload_identity_scope": "metadata hashes plus payload size/mtime/inode, not full weight-byte hashes",
        "data_sha256": digest(args.data),
        "rank": args.rank,
        "alpha": args.alpha,
        "chunk_size": args.chunk_size,
        "adapter_layers": [39],
        "expert_adapter_layers": [],
    }


def suffix_hidden(model, prefix, *, train=False):
    from exl3_mlx.hyperconnections import hc_pre, rms_norm
    from exl3_mlx.recompute import recompute

    inputs = (
        prefix["h"],
        prefix["pre"],
        tuple(prefix[f"kv{i}"] for i in range(4)),
        prefix["engram"],
    )
    block = model.blocks[39]
    if train:
        h, pre, _ = recompute(block, *inputs)
    else:
        h, pre, _ = block(*inputs)
    return rms_norm(hc_pre(h, pre), model.norm)


def evaluate_cached(model, prefix, example, *, include_token_metrics=False, check=lambda: None):
    """Reusable actual-forward score of the entire original answer mask."""
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    model.eval()
    start = example["loss_start"]
    targets = mx.array(example["input_ids"][start:], dtype=mx.int32)
    hidden = suffix_hidden(model, prefix)[0, start - 1 :]
    mx.eval(hidden)
    nll, hashes, losses, predicted = 0.0, hashlib.sha256(), [], []
    for offset in range(0, len(targets), 32):
        check()
        logits = model.project_logits(hidden[offset : offset + 32])
        values = nn.losses.cross_entropy(logits, targets[offset : offset + 32], reduction="none")
        value = mx.sum(values)
        mx.eval(value, logits)
        if not math.isfinite(value.item()):
            raise FloatingPointError("Nonfinite cached evaluation")
        nll += value.item()
        hashes.update(np.array(logits).tobytes())
        if include_token_metrics:
            losses.extend(np.array(values).tolist())
            predicted.extend(np.array(mx.argmax(logits, axis=-1)).tolist())
    report = {
        "loss": nll / len(targets),
        "perplexity": math.exp(nll / len(targets)) if nll / len(targets) < 700 else None,
        "supervised_tokens": len(targets),
        "examples": 1,
        "logits_sha256": hashes.hexdigest(),
        "logits_scope": "supervised_rows",
    }
    if include_token_metrics:
        report.update(
            token_nll=losses, target_ids=example["input_ids"][start:], predicted_ids=predicted
        )
    check()
    return report


def run(args, rows):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from exl3_mlx.deepseek import DeepSeekTextModel
    from exl3_mlx.materialize import materialize
    from exl3_mlx.memory import MemoryBudget
    from exl3_mlx.train import checkpoint_identity, load_adapter, load_adapter_config, save_adapter
    from exl3_mlx.training import (
        evaluate,
        load_checkpoint,
        projected_cross_entropy,
        save_checkpoint,
    )

    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "frozen-prefix-training-diagnostic-v1",
        "status": "starting",
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": digest(__file__),
        "scope": "Actual last-block SFT updates, full original context; training/development fit only",
        "quality_certified": False,
        "history": [],
    }
    write_json(args.output / "diagnostic.json", report)

    def event(message):
        print(f"[{time.monotonic() - started:.1f}s] {message}", flush=True)

    try:
        with MemoryBudget(
            args.memory_gib, cache_mib=args.allocator_cache_mib, reserve_gib=args.reserve_gib
        ) as budget:
            budget.abort_path = args.output / "memory-abort.json"
            binding = binding_for(args)
            report["binding"] = binding
            cache = None
            if args.prefix_cache:
                cache_dir = args.prefix_cache.expanduser().resolve()
                cache = json.loads((cache_dir / "prefix-cache.json").read_text())
                validate_cache_manifest(cache, binding, rows, cache_dir)
            else:
                cache_dir = args.output / "prefix-cache"
                cache_dir.mkdir()
            mx.random.seed(args.seed)
            model = DeepSeekTextModel(
                args.model,
                rank=args.rank,
                alpha=args.alpha,
                adapter_layers=[39],
                expert_adapter_layers=[],
                cache_mib=args.packed_cache_mib,
                decoded_cache_mib=args.decoded_cache_mib,
                chunk_size=args.chunk_size,
                progress=event,
            )
            expected = dict(tree_flatten(model.trainable_parameters()))
            if len(expected) != 4 or any(not key.startswith("blocks.39.") for key in expected):
                raise AssertionError("Expected only block39 wq_a/wo_b A/B adapters")
            if cache:
                load_adapter(model, cache_dir / "initial-adapter")
            elif args.initial_adapter:
                config = load_adapter_config(args.initial_adapter)
                if (
                    config["base_identity"] != checkpoint_identity(args.model)
                    or config["rank"] != args.rank
                    or config["alpha"] != args.alpha
                ):
                    raise ValueError("Initial adapter base/rank/alpha mismatch")
                saved = mx.load(str(args.initial_adapter / "adapter.safetensors"))
                selected = {}
                for key, value in expected.items():
                    if key.endswith("lora_a"):
                        if (
                            key not in saved
                            or saved[key].shape != value.shape
                            or saved[key].dtype != value.dtype
                        ):
                            raise ValueError("Initial A keys/shapes/dtypes mismatch")
                        selected[key] = saved[key]
                    else:
                        selected[key] = mx.zeros_like(value)
                if any(not mx.all(mx.isfinite(value)).item() for value in selected.values()):
                    raise ValueError("Nonfinite initial A")
                model.load_weights(list(selected.items()), strict=False)
                report["initial_A_import"] = {
                    "path": str(args.initial_adapter),
                    "adapter_sha256": digest(args.initial_adapter / "adapter.safetensors"),
                    "scope": "block39 A only; all B zero",
                }
            if any(
                mx.count_nonzero(v).item()
                for k, v in tree_flatten(model.trainable_parameters())
                if k.endswith("lora_b")
            ):
                raise AssertionError("Capture/replay initialization requires exactly zero B")
            prefixes = {}
            if cache is None:
                cache = {
                    "schema": "frozen-prefix-cache-v1",
                    "status": "capturing",
                    "binding": binding,
                    "rows": {},
                    "files": {},
                    "initial_A_import": report.get("initial_A_import"),
                }
                write_json(cache_dir / "prefix-cache.json", cache)
                save_adapter(
                    model,
                    cache_dir / "initial-adapter",
                    {"scope": "Frozen-prefix diagnostic zero-B initialization"},
                )

                class Capture(nn.Module):
                    def __init__(self, original):
                        super().__init__()
                        self.original = original
                        self._captured = None

                    def __call__(self, h, pre, state, engram):
                        if self._captured is not None:
                            raise AssertionError("Expected one prefix capture per example")
                        self._captured = {
                            "h": materialize(h),
                            "pre": materialize(pre),
                            **{f"kv{i}": materialize(v) for i, v in enumerate(state)},
                            "engram": materialize(engram),
                        }
                        return self.original(h, pre, state, engram)

                for index, example in rows.items():
                    event(f"Capturing row{index}: {len(example['input_ids'])} full tokens")
                    original = model.blocks[39]
                    wrapper = Capture(original)
                    model.blocks[39] = wrapper
                    try:
                        uncached = evaluate(model, [example], check=budget.check)
                        prefix = wrapper._captured
                    finally:
                        model.blocks[39] = original
                    if prefix is None:
                        raise AssertionError("No prefix captured")
                    file = cache_dir / f"row-{index:04d}.safetensors"
                    mx.save_safetensors(str(file), prefix)
                    loaded = mx.load(str(file))
                    if set(loaded) != set(prefix) or any(
                        not mx.array_equal(loaded[k], v).item() for k, v in prefix.items()
                    ):
                        raise AssertionError("Prefix serialization mismatch")
                    cached = evaluate_cached(model, loaded, example, check=budget.check)
                    if uncached != cached:
                        raise AssertionError(f"Uncached/cached base parity failed on row{index}")
                    prefixes[index] = loaded
                    cache["rows"][str(index)] = {
                        "uncached_base": uncached,
                        "cached_base": cached,
                        "base_parity_exact": True,
                        "prefix_shapes": {k: list(v.shape) for k, v in prefix.items()},
                        "prefix_dtypes": {k: str(v.dtype) for k, v in prefix.items()},
                    }
                    cache["files"][file.name] = digest(file)
                    write_json(cache_dir / "prefix-cache.json", cache)
                    event(f"Row{index} exact cached/uncached parity, NLL={cached['loss']:.9f}")
                for name in ("adapter.safetensors", "adapter_config.json"):
                    cache["files"]["initial-adapter/" + name] = digest(
                        cache_dir / "initial-adapter" / name
                    )
                cache["status"] = "complete"
                write_json(cache_dir / "prefix-cache.json", cache)
            else:
                for index in rows:
                    prefixes[index] = mx.load(str(cache_dir / f"row-{index:04d}.safetensors"))
                    mx.eval(prefixes[index])
            report["prefix_cache"] = str(cache_dir)
            report["prefix_cache_sha256"] = digest(cache_dir / "prefix-cache.json")
            report["base_scores"] = {
                str(i): evaluate_cached(
                    model,
                    prefixes[i],
                    row,
                    include_token_metrics=args.include_token_metrics,
                    check=budget.check,
                )
                for i, row in rows.items()
            }
            if args.capture_only:
                report["status"] = "captured"
            else:
                optimizer = optim.Adam(
                    learning_rate=args.learning_rate, bias_correction=args.bias_correction
                )
                contract = {
                    "diagnostic": "frozen-prefix-v1",
                    "prefix_cache_sha256": report["prefix_cache_sha256"],
                    "learning_rate": args.learning_rate,
                    "bias_correction": args.bias_correction,
                    "adam_betas": [0.9, 0.999],
                    "adam_eps": 1e-8,
                    "clip_grad": args.clip_grad,
                    "rows": args.rows,
                    "seed": args.seed,
                    "binding": binding,
                }
                step = 0
                if args.resume:
                    state = load_checkpoint(model, optimizer, args.resume, contract)
                    step = state["step"]
                    report["resumed_initial_scores"] = {
                        str(i): evaluate_cached(
                            model,
                            prefixes[i],
                            row,
                            include_token_metrics=args.include_token_metrics,
                            check=budget.check,
                        )
                        for i, row in rows.items()
                    }
                initial_step = step
                for _ in range(args.steps):
                    index = args.rows[step % len(args.rows)]
                    example, prefix = rows[index], prefixes[index]
                    targets = mx.array(
                        example["input_ids"][example["loss_start"] :], dtype=mx.int32
                    )
                    model.train()

                    def loss_fn(current):
                        hidden = suffix_hidden(current, prefix, train=True)[
                            0, example["loss_start"] - 1 :
                        ]
                        return projected_cross_entropy(current.project_logits, hidden, targets)

                    event(f"Update{step + 1} row{index}")
                    loss, grads = nn.value_and_grad(model, loss_fn)(model)
                    mx.eval(loss, grads)
                    if not math.isfinite(loss.item()) or any(
                        not mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(grads)
                    ):
                        raise FloatingPointError("Nonfinite pre-update loss/gradient")
                    if step == initial_step:
                        gradient_file = args.output / "first-step-gradients.safetensors"
                        mx.save_safetensors(str(gradient_file), dict(tree_flatten(grads)))
                        report["first_step_gradients"] = {
                            "sha256": digest(gradient_file),
                            "step": step + 1,
                            "scope": "unclipped d(mean full-answer NLL)/d(adapter)",
                        }
                    clipped, norm = optim.clip_grad_norm(grads, args.clip_grad)
                    mx.eval(clipped, norm)
                    if not math.isfinite(norm.item()):
                        raise FloatingPointError("Nonfinite gradient norm")
                    optimizer.update(model, clipped)
                    mx.eval(model.trainable_parameters(), optimizer.state)
                    if any(
                        not mx.all(mx.isfinite(v)).item()
                        for _, v in tree_flatten(model.trainable_parameters())
                    ):
                        raise FloatingPointError("Nonfinite adapter after update")
                    step += 1
                    record = {
                        "step": step,
                        "row": index,
                        "pre_update_loss": loss.item(),
                        "gradient_norm": norm.item(),
                        "supervised_tokens": len(targets),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    checkpoint = args.output / f"step-{step:06d}"
                    save_checkpoint(
                        model,
                        optimizer,
                        checkpoint,
                        {"step": step, "cursor": step, "contract": contract, "metrics": record},
                    )
                    report["last_checkpoint"] = str(checkpoint)
                    record["post_update_scores"] = {
                        str(i): evaluate_cached(
                            model,
                            prefixes[i],
                            row,
                            include_token_metrics=args.include_token_metrics,
                            check=budget.check,
                        )
                        for i, row in rows.items()
                    }
                    report["history"].append(record)
                    report["status"] = "training"
                    write_json(args.output / "diagnostic.json", report)
                    event(
                        f"Update{step}: before={record['pre_update_loss']:.9f}; after="
                        + ", ".join(
                            f"row{i}={s['loss']:.9f}"
                            for i, s in record["post_update_scores"].items()
                        )
                    )
                    del grads, clipped, loss, targets
                if args.verify_uncached_final:
                    report["uncached_final_scores"] = {
                        str(i): evaluate(model, [row], check=budget.check)
                        for i, row in rows.items()
                    }
                    for i in rows:
                        actual = report["uncached_final_scores"][str(i)]
                        cached = report["history"][-1]["post_update_scores"][str(i)]
                        if any(actual[k] != cached[k] for k in actual):
                            raise AssertionError(f"Final uncached/cached parity failed on row{i}")
                    report["uncached_final_parity_exact"] = True
                report["status"] = "completed"
            report["peak_mlx_gib"] = mx.get_peak_memory() / 2**30
            report["observed_process_gib"] = budget.usage() / 2**30
            report["elapsed_seconds"] = time.monotonic() - started
            write_json(args.output / "diagnostic.json", report)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(args.output / "diagnostic.json", report)
        raise
    return report


def main():
    args = parser().parse_args()
    rows = prepare(args)
    if args.describe_only:
        print(
            json.dumps(
                {
                    "training_rows": args.rows,
                    "extra_diagnostic_rows": args.eval_rows,
                    "lengths": {str(i): len(row["input_ids"]) for i, row in rows.items()},
                    "adapter_layers": [39],
                    "scope": "full-context frozen-prefix final-block diagnostic",
                },
                indent=2,
            )
        )
        return
    run(args, rows)


if __name__ == "__main__":
    main()
