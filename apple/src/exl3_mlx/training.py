"""Eager streamed training, MLX-LM loss, atomic resumable checkpoints."""

import hashlib
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm.tuner.trainer import default_loss

from .data import sha256
from .materialize import materialize


def batch_for(example):
    ids = example["input_ids"]
    return mx.array([ids], dtype=mx.int32), mx.array([[example["loss_start"], len(ids) - 1]])


def supervised_hidden(model, tokens, lengths):
    """Keep full causal context but project only rows with supervised targets."""
    hidden = model.hidden_states(tokens[:, :-1])
    spans = np.array(lengths)
    pieces, targets = [], []
    for batch, (start, end) in enumerate(spans.tolist()):
        if not 1 <= start <= end < tokens.shape[1]:
            raise ValueError("Invalid supervised token span")
        pieces.append(hidden[batch, start - 1 : end])
        targets.append(tokens[batch, start : end + 1])
    return mx.concatenate(pieces), mx.concatenate(targets)


def projected_cross_entropy(project, hidden, targets, *, chunk_size=32):
    """Exact target-only CE with an eager first-order VJP for a frozen LM head.

    Never retain an [sequence, vocabulary] graph. Chunking does not detach the
    backbone or reset its history. The head is frozen; its input receives dL/dh.
    """
    if chunk_size < 1 or not len(targets) or hidden.shape[0] != targets.shape[0]:
        raise ValueError("Positive chunk size and aligned nonempty targets required")
    count = len(targets)

    def part(value, start):
        return nn.losses.cross_entropy(
            project(value), targets[start : start + value.shape[0]], reduction="sum"
        )

    @mx.custom_function
    def forward(values):
        values = mx.stop_gradient(values)
        total = 0.0
        for start in range(0, count, chunk_size):
            total += part(values[start : start + chunk_size], start).item()
        return mx.array(total / count, dtype=mx.float32)

    @forward.vjp
    def backward(primals, cotangent, output):
        values = primals
        gradients = []
        for start in range(0, count, chunk_size):
            piece = mx.stop_gradient(values[start : start + chunk_size])
            _, (gradient,) = mx.vjp(lambda value: part(value, start), [piece], [cotangent / count])
            mx.eval(gradient)
            gradients.append(materialize(gradient))
        return mx.concatenate(gradients)

    return forward(hidden)


def masked_loss(model, tokens, lengths):
    if hasattr(model, "hidden_states") and hasattr(model, "project_logits"):
        hidden, targets = supervised_hidden(model, tokens, lengths)
        return projected_cross_entropy(model.project_logits, hidden, targets)
    return default_loss(model, tokens, lengths)[0]


def sample_indices(count, cursor, size, seed):
    """Deterministic epoch shuffles; resume needs only the sample cursor and seed."""
    if count < 1 or size < 1 or cursor < 0:
        raise ValueError("Invalid sampler arguments")
    indices = []
    epoch_order, last_epoch = None, None
    for position in range(cursor, cursor + count):
        epoch, offset = divmod(position, size)
        if epoch != last_epoch:
            epoch_order = list(range(size))
            random.Random(seed + epoch).shuffle(epoch_order)
            last_epoch = epoch
        indices.append(epoch_order[offset])
    return indices


def optimizer_step(model, optimizer, examples, *, clip=1.0, check=lambda: None):
    """Microbatch one, accumulating a token-weighted mean before one update."""
    model.train()
    vg = nn.value_and_grad(model, masked_loss)
    accumulated, total, loss_sum = None, 0, 0.0
    for example in examples:
        check()
        tokens, lengths = batch_for(example)
        value, grads = vg(model, tokens, lengths)
        mx.eval(value, grads)
        flat = tree_flatten(grads)
        if not flat or any(not k.endswith(("lora_a", "lora_b")) for k, _ in flat):
            raise AssertionError("Only adapter parameters may enter the optimizer")
        if not math.isfinite(value.item()) or any(
            not mx.all(mx.isfinite(g)).item() for _, g in flat
        ):
            raise FloatingPointError("Nonfinite loss/gradient; optimizer was not updated")
        n = len(example["input_ids"]) - example["loss_start"]
        accumulated = (
            tree_map(lambda g: g * n, grads)
            if accumulated is None
            else tree_map(lambda a, g: a + g * n, accumulated, grads)
        )
        mx.eval(accumulated)
        total += n
        loss_sum += value.item() * n
        del grads, flat, tokens, lengths, value
    grads = tree_map(lambda g: g / total, accumulated)
    grads, norm = optim.clip_grad_norm(grads, clip)
    mx.eval(grads, norm)
    if not math.isfinite(norm.item()):
        raise FloatingPointError("Nonfinite accumulated gradient; optimizer was not updated")
    optimizer.update(model, grads)
    mx.eval(model.trainable_parameters(), optimizer.state)
    if any(
        not mx.all(mx.isfinite(v)).item() for _, v in tree_flatten(model.trainable_parameters())
    ):
        raise FloatingPointError("Nonfinite updated adapter; checkpoint not saved")
    check()
    return {
        "loss": loss_sum / total,
        "gradient_norm": norm.item(),
        "supervised_tokens": total,
        "microbatches": len(examples),
    }


def evaluate(model, examples, *, check=lambda: None):
    was_training = model.training
    model.eval()
    total, nll, digest = 0, 0.0, hashlib.sha256()
    try:
        for example in examples:
            check()
            tokens, lengths = batch_for(example)
            if hasattr(model, "hidden_states") and hasattr(model, "project_logits"):
                hidden, targets = supervised_hidden(model, tokens, lengths)
                mx.eval(hidden, targets)
                for start in range(0, len(targets), 32):
                    check()
                    logits = model.project_logits(hidden[start : start + 32])
                    value = nn.losses.cross_entropy(
                        logits, targets[start : start + 32], reduction="sum"
                    )
                    mx.eval(value, logits)
                    if not math.isfinite(value.item()):
                        raise FloatingPointError("Nonfinite evaluation loss")
                    nll += value.item()
                    total += logits.shape[0]
                    digest.update(np.array(logits).tobytes())
                    del logits, value
                del hidden, targets
            else:
                logits = model(tokens[:, :-1])
                value, n = default_loss(lambda _, values=logits: values, tokens, lengths)
                mx.eval(value, n, logits)
                if not math.isfinite(value.item()):
                    raise FloatingPointError("Nonfinite evaluation loss")
                nll += value.item() * n.item()
                total += n.item()
                digest.update(np.array(logits).tobytes())
                del logits, value
            del tokens, lengths
            check()
    finally:
        model.train(was_training)
    if not total:
        raise ValueError("Evaluation needs supervised tokens")
    loss = nll / total
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else None,
        "supervised_tokens": total,
        "examples": len(examples),
        "logits_sha256": digest.hexdigest(),
        "logits_scope": "supervised_rows" if hasattr(model, "hidden_states") else "all_rows",
    }


def save_checkpoint(model, optimizer, directory, state):
    from .train import save_adapter

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=directory.parent))
    payload = stage / "payload"
    try:
        save_adapter(model, payload, state)
        arrays = dict(tree_flatten(optimizer.state))
        if not arrays or any(not isinstance(v, mx.array) for v in arrays.values()):
            raise ValueError("Optimizer state must contain only array leaves")
        mx.save_safetensors(str(payload / "optimizer.safetensors"), arrays)
        mx.save_safetensors(
            str(payload / "random.safetensors"), {str(i): v for i, v in enumerate(mx.random.state)}
        )
        saved = {
            "format": "exl3-mlx-training-state-v1",
            **state,
            "files": {
                name: sha256(payload / name)
                for name in (
                    "adapter.safetensors",
                    "adapter_config.json",
                    "optimizer.safetensors",
                    "random.safetensors",
                )
            },
        }
        (payload / "trainer_state.json").write_text(json.dumps(saved, indent=2) + "\n")
        payload.rename(directory)
    finally:
        shutil.rmtree(stage)


def load_checkpoint(model, optimizer, directory, contract):
    from .train import load_adapter

    directory = Path(directory)
    state = json.loads((directory / "trainer_state.json").read_text())
    if state.get("format") != "exl3-mlx-training-state-v1" or state.get("contract") != contract:
        raise ValueError("Resume settings/data differ; use --resume-adapter for a fresh optimizer")
    names = {
        "adapter.safetensors",
        "adapter_config.json",
        "optimizer.safetensors",
        "random.safetensors",
    }
    if set(state["files"]) != names or any(
        sha256(directory / n) != state["files"][n] for n in names
    ):
        raise ValueError("Training checkpoint checksum mismatch")
    optimizer.init(model.trainable_parameters())
    arrays = mx.load(str(directory / "optimizer.safetensors"))
    expected = dict(tree_flatten(optimizer.state))
    if set(arrays) != set(expected) or any(
        v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in arrays.items()
    ):
        raise ValueError("Optimizer state structure mismatch")
    rng = mx.load(str(directory / "random.safetensors"))
    if set(rng) != {str(i) for i in range(len(mx.random.state))} or any(
        v.shape != mx.random.state[int(k)].shape or v.dtype != mx.random.state[int(k)].dtype
        for k, v in rng.items()
    ):
        raise ValueError("Random state structure mismatch")
    if int(arrays["step"].item()) != state["step"] or state["cursor"] < 0:
        raise ValueError("Optimizer step/cursor mismatch")
    load_adapter(model, directory)
    optimizer.state = tree_unflatten(list(arrays.items()))
    for i, key in enumerate(mx.random.state):
        key[:] = rng[str(i)]
    mx.eval(optimizer.state, mx.random.state)
    return state
