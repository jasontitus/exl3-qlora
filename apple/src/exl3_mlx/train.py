"""Experimental causal-text adapter training on the local streaming repack."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .activation_config import activation_description, normalize_adapter_config
from .context import ENGINE_SEMANTICS, MAX_CONTEXT_TOKENS


def load_examples(path, tokenizer, *, seq_length, vocab_size):
    if not 2 <= seq_length <= MAX_CONTEXT_TOKENS:
        raise ValueError("seq_length must be in 2..32768")
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


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_adapter_integrity(directory, config):
    """Verify available receipts before replacing any destination parameters."""
    expected = []
    if "adapter_sha256" in config:
        expected.append(config["adapter_sha256"])
    state_path = directory / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not isinstance(state, dict) or state.get("format") != "exl3-mlx-training-state-v1":
            raise ValueError("Unsupported training checkpoint integrity receipt")
        files = state.get("files")
        required = ("adapter.safetensors", "adapter_config.json")
        if not isinstance(files, dict) or any(not isinstance(files.get(n), str) for n in required):
            raise ValueError("Training checkpoint is missing adapter checksums")
        if _file_sha256(directory / "adapter_config.json") != files["adapter_config.json"]:
            raise ValueError("Adapter configuration checksum mismatch")
        expected.append(files["adapter.safetensors"])
    if not expected:
        raise ValueError(
            "Adapter has no saved checksum; recover a trusted training checkpoint "
            "or explicitly re-save verified weights"
        )
    actual = _file_sha256(directory / "adapter.safetensors")
    if any(value != actual for value in expected):
        raise ValueError("Adapter weight checksum mismatch")


def save_adapter(model, directory, extra):
    model_config = normalize_adapter_config(model.adapter_config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(directory / "adapter.safetensors"), weights)
    config = {
        "format": "exl3-mlx-deepseek-short-text-v1",
        "adapter_sha256": _file_sha256(directory / "adapter.safetensors"),
        **model_config,
        "base_identity": checkpoint_identity(model._root),
        "max_seq_length": MAX_CONTEXT_TOKENS,
        "engine_semantics": ENGINE_SEMANTICS,
        "activation_quantization": activation_description(model_config["activation_mode"]),
        "training": extra,
    }
    (directory / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Verify serialized values; full prediction reload is exercised separately.
    loaded = mx.load(str(directory / "adapter.safetensors"))
    if set(loaded) != set(weights) or any(
        not mx.array_equal(loaded[k], v).item() for k, v in weights.items()
    ):
        raise AssertionError("Adapter serialization mismatch")


def load_adapter_config(directory):
    """Read verified settings before model allocation or parameter mutation.

    Legacy adapters without an arithmetic version remain readable, with unknown
    numerical provenance. An explicit incompatible version is never silently
    migrated. Exact project/checkpoint resume has its separate strict contract.
    """
    directory = Path(directory).expanduser()
    config = json.loads((directory / "adapter_config.json").read_text())
    if not isinstance(config, dict) or config.get("format") != "exl3-mlx-deepseek-short-text-v1":
        raise ValueError("Unsupported adapter format")
    _verify_adapter_integrity(directory, config)
    if "engine_semantics" in config and config["engine_semantics"] != ENGINE_SEMANTICS:
        raise ValueError(
            "Adapter engine_semantics differ; use its recorded implementation "
            "or explicitly migrate and revalidate it"
        )
    return normalize_adapter_config(config)


def load_adapter(model, directory):
    directory = Path(directory).expanduser()
    c = load_adapter_config(directory)
    if c["base_identity"] != checkpoint_identity(model._root):
        raise ValueError("Adapter checkpoint identity mismatch")
    expected_config = normalize_adapter_config(model.adapter_config)
    if any(c[k] != v for k, v in expected_config.items()):
        raise ValueError(
            "Construct the model with the saved rank, alpha, layer selection and activation_mode"
        )
    weights = mx.load(str(directory / "adapter.safetensors"))
    expected = dict(tree_flatten(model.trainable_parameters()))
    if set(weights) != set(expected) or any(
        v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in weights.items()
    ):
        raise ValueError("Adapter tensor keys, shapes or dtypes differ")
    if any(not mx.all(mx.isfinite(value)).item() for value in weights.values()):
        raise ValueError("Adapter contains nonfinite weights")
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.trainable_parameters())


def main():
    # Preserve the original entry point while sharing the complete workflow CLI.
    import sys

    from .workflow import main as workflow_main

    workflow_main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
