"""Experimental causal-text adapter training on the local streaming repack."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


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


def main():
    # Preserve the original entry point while sharing the complete workflow CLI.
    import sys

    from .workflow import main as workflow_main

    workflow_main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
