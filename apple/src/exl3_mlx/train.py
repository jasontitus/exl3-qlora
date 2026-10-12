"""Experimental causal-text adapter training on the local streaming repack."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from .deepseek import DeepSeekTextModel
from .memory import MemoryBudget


def load_examples(path, tokenizer, *, seq_length, vocab_size):
    if not 2 <= seq_length <= 512:
        raise ValueError("seq_length must be in 2..512")
    examples = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) == {"text"} and isinstance(row["text"], str):
            # Each document has its own causal/Engram boundary. Never concatenate
            # unrelated samples or silently reinterpret a chat messages schema.
            ids = [0] + tokenizer.encode(row["text"], add_special_tokens=False).ids + [1]
        elif set(row) == {"input_ids"}:
            ids = row["input_ids"]
        else:
            raise ValueError(f"Line {number}: use exactly text or input_ids")
        if (
            not isinstance(ids, list)
            or len(ids) < 2
            or any(type(t) is not int or not 0 <= t < vocab_size or t == 129264 for t in ids)
        ):
            raise ValueError(f"Line {number}: invalid text token IDs")
        # Explicit prefix truncation: windows reset Engram and attention history.
        examples.append(ids[: seq_length + 1])
    if not examples:
        raise ValueError("Dataset contains no examples")
    return examples


def text_loss(model, tokens):
    return nn.losses.cross_entropy(model(tokens[:, :-1]), tokens[:, 1:], reduction="mean")


def checkpoint_identity(root):
    names = [
        "config.json",
        "expert-manifest-v2.json",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def save_adapter(model, directory, extra):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(directory / "adapter.safetensors"), weights)
    config = {
        "format": "exl3-mlx-deepseek-short-text-v1",
        **model.adapter_config,
        "base_identity": checkpoint_identity(model._root),
        "max_seq_length": 512,
        "activation_quantization": "fp8/fp4 with identity STE",
        "training": extra,
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Verify serialized values; full prediction reload is exercised separately.
    loaded = mx.load(str(directory / "adapter.safetensors"))
    if set(loaded) != set(weights) or any(
        not mx.array_equal(loaded[k], v).item() for k, v in weights.items()
    ):
        raise AssertionError("Adapter serialization mismatch")


def load_adapter(model, directory):
    directory = Path(directory)
    c = json.loads((directory / "adapter_config.json").read_text())
    if c["format"] != "exl3-mlx-deepseek-short-text-v1":
        raise ValueError("Unsupported adapter format")
    if c["base_identity"] != checkpoint_identity(model._root):
        raise ValueError("Adapter checkpoint identity mismatch")
    if any(c[k] != v for k, v in model.adapter_config.items()):
        raise ValueError("Construct the model with the saved rank, alpha and layer selection")
    weights = mx.load(str(directory / "adapter.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(weights) != set(expected) or any(
        v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in weights.items()
    ):
        raise ValueError("Adapter tensor keys, shapes or dtypes differ")
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())


def _main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--data", required=True, help="JSONL with text or input_ids; causal loss on every token"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--memory-gib", type=float, default=16.0)
    p.add_argument("--allocator-cache-mib", type=int, default=64)
    p.add_argument("--reserve-gib", type=float, default=32.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seq-length", type=int, default=16)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument(
        "--adapter-layers",
        type=int,
        nargs="*",
        default=None,
        help="Attention adapter layers; defaults to all 40",
    )
    p.add_argument(
        "--expert-adapter-layers",
        type=int,
        nargs="*",
        default=[],
        help="Also adapt all routed experts in these layers (more adapter memory)",
    )
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--resume-adapter", help="Load adapter weights; optimizer restarts fresh")
    args = p.parse_args()
    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("Positive steps and learning rate required")
    output = Path(args.output)
    if output.exists():
        p.error("Output already exists; choose a new directory")
    root = Path(args.model).expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    config = json.loads((root / "config.json").read_text())["text_config"]
    examples = load_examples(
        args.data, tokenizer, seq_length=args.seq_length, vocab_size=config["vocab_size"]
    )
    mx.random.seed(args.seed)
    start = time.monotonic()

    def progress(message):
        print(f"[{time.monotonic() - start:.1f}s] {message}", flush=True)

    with MemoryBudget(
        args.memory_gib, cache_mib=args.allocator_cache_mib, reserve_gib=args.reserve_gib
    ) as budget:
        model = DeepSeekTextModel(
            root,
            rank=args.rank,
            alpha=args.alpha,
            cache_mib=args.cache_mib,
            chunk_size=args.chunk_size,
            checkpoint_blocks=not args.no_checkpoint,
            adapter_layers=args.adapter_layers,
            expert_adapter_layers=args.expert_adapter_layers,
            progress=progress,
        )
        if args.resume_adapter:
            load_adapter(model, args.resume_adapter)
        vg = nn.value_and_grad(model, text_loss)
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        history = []
        for step in range(args.steps):
            budget.check()
            ids = mx.array([examples[step % len(examples)]], dtype=mx.int32)
            loss, grads = vg(model, ids)
            mx.eval(loss, grads)
            flat = tree_flatten(grads)
            if any(not k.endswith(("lora_a", "lora_b")) for k, _ in flat):
                raise AssertionError("Frozen parameters leaked into gradients")
            if not math.isfinite(loss.item()) or any(
                not mx.all(mx.isfinite(g)).item() for _, g in flat
            ):
                raise FloatingPointError("Nonfinite loss/gradient; optimizer was not updated")
            grad_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in flat)).item()
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state)
            record = {
                "step": step + 1,
                "loss": loss.item(),
                "gradient_norm": grad_norm,
                "tokens": ids.shape[1] - 1,
                "elapsed_seconds": time.monotonic() - start,
            }
            history.append(record)
            progress(json.dumps(record))
            # Persist a recoverable adapter after every completed update.
            save_adapter(model, output / f"step-{step + 1:06d}", record)
        report = model.report()
        report.update(
            history=history,
            steps=args.steps,
            dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            elapsed_seconds=time.monotonic() - start,
            mlx=mx.__version__,
            device=mx.device_info()["device_name"],
            adapter_serialization_exact=True,
            optimizer_resume_supported=False,
        )
        (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    try:
        _main()
    except MemoryError as error:
        raise SystemExit(f"Memory budget stopped training: {error}") from error


if __name__ == "__main__":
    main()
