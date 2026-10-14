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


def evaluate(
    model,
    examples,
    *,
    check=lambda: None,
    include_token_metrics=False,
    preserve_mode=False,
    log=lambda message: None,
):
    """Score complete supervised targets without updating model parameters.

    By default, temporarily use evaluation mode. The opt-in preserve_mode path
    keeps the caller's mode, allowing a final diagnostic to reuse exact frozen
    training-prefix cache entries. It does not disable training-mode behavior
    such as dropout on arbitrary models. Both paths restore the entry mode even
    if scoring or a memory check raises; neither computes gradients or updates
    an optimizer. The supported DeepSeek training path has no stochastic dropout.
    """
    if type(preserve_mode) is not bool:
        raise ValueError("preserve_mode must be boolean")
    was_training = model.training
    if not preserve_mode:
        model.eval()
    total, nll, digest = 0, 0.0, hashlib.sha256()
    row_metrics = []
    try:
        for index, example in enumerate(examples, 1):
            check()
            log(
                f"Evaluation example {index}/{len(examples)} started; "
                f"serialized tokens {len(example['input_ids'])}"
            )
            tokens, lengths = batch_for(example)
            details = {"token_nll": [], "target_ids": [], "predicted_ids": []}
            row_digest = hashlib.sha256() if include_token_metrics else None
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
                    logit_bytes = np.array(logits).tobytes()
                    digest.update(logit_bytes)
                    if include_token_metrics:
                        row_digest.update(logit_bytes)
                        gold = targets[start : start + 32]
                        details["token_nll"].extend(nn.losses.cross_entropy(logits, gold).tolist())
                        details["target_ids"].extend(gold.tolist())
                        details["predicted_ids"].extend(mx.argmax(logits, axis=-1).tolist())
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
                if include_token_metrics:
                    row_digest.update(np.array(logits).tobytes())
                    start = example["loss_start"]
                    selected = logits[0, start - 1 :]
                    gold = tokens[0, start:]
                    details["token_nll"] = nn.losses.cross_entropy(selected, gold).tolist()
                    details["target_ids"] = gold.tolist()
                    details["predicted_ids"] = mx.argmax(selected, axis=-1).tolist()
                del logits, value
            if include_token_metrics:
                details["logits_sha256"] = row_digest.hexdigest()
                count = len(details["target_ids"])
                details["loss"] = sum(details["token_nll"]) / count
                details["supervised_tokens"] = count
                details["correct_tokens"] = sum(
                    a == b for a, b in zip(details["target_ids"], details["predicted_ids"])
                )
                row_metrics.append(details)
            del tokens, lengths
            check()
            log(
                f"Evaluation example {index}/{len(examples)} completed; "
                f"cumulative supervised tokens {total}"
            )
    finally:
        model.train(was_training)
    if not total:
        raise ValueError("Evaluation needs supervised tokens")
    loss = nll / total
    result = {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else None,
        "supervised_tokens": total,
        "examples": len(examples),
        "logits_sha256": digest.hexdigest(),
        "logits_scope": "supervised_rows" if hasattr(model, "hidden_states") else "all_rows",
    }
    if include_token_metrics:
        result["rows"] = row_metrics
    return result


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
    """Restore the workflow's fixed-Adam state after complete validation."""
    from .train import load_adapter

    if (
        type(optimizer) is not optim.Adam
        or tuple(optimizer.betas) != (0.9, 0.999)
        or optimizer.eps != 1e-8
        or optimizer.bias_correction is not False
        or optimizer._schedulers
    ):
        raise ValueError(
            "Resume supports only fixed-rate Adam with the workflow's default settings"
        )
    directory = Path(directory)
    state = json.loads((directory / "trainer_state.json").read_text())
    if state.get("format") != "exl3-mlx-training-state-v1" or state.get("contract") != contract:
        raise ValueError("Resume settings/data differ; use --resume-adapter for a fresh optimizer")
    if any(type(state.get(key)) is not int or state[key] < 0 for key in ("step", "cursor")):
        raise ValueError("Optimizer step/cursor must be nonnegative integers")
    if "grad_accumulation" in contract:
        accumulation = contract["grad_accumulation"]
        if (
            type(accumulation) is not int
            or accumulation < 1
            or state["cursor"] != state["step"] * accumulation
        ):
            raise ValueError("Sample cursor differs from the recorded accumulation contract")
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
    # Never derive the incoming schema from destination state: MLX init retains
    # nonempty moment dictionaries, including malformed existing ones. The
    # workflow uses fixed Adam, whose fresh constructor defines canonical scalar
    # metadata and whose init derives every moment shape/dtype from the model.
    # The placeholder learning rate defines only its dtype, not the loaded value.
    template = optim.Adam(learning_rate=1.0)
    template.init(model.trainable_parameters())
    arrays = mx.load(str(directory / "optimizer.safetensors"))
    expected = dict(tree_flatten(template.state))
    if set(arrays) != set(expected) or any(
        v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in arrays.items()
    ):
        raise ValueError("Optimizer state structure mismatch")
    if any(not mx.all(mx.isfinite(value)).item() for value in arrays.values()):
        raise ValueError("Optimizer state contains nonfinite values")
    if any(
        not mx.all(value >= 0).item()
        for name, value in arrays.items()
        if name == "v" or name.endswith(".v")
    ):
        raise ValueError("Adam second moments must be nonnegative")
    learning_rate = arrays["learning_rate"]
    if learning_rate.size != 1 or learning_rate.item() <= 0:
        raise ValueError("Optimizer learning rate must be a positive scalar")
    if "learning_rate" in contract:
        recorded_rate = contract["learning_rate"]
        if (
            type(recorded_rate) not in (int, float)
            or not math.isfinite(recorded_rate)
            or recorded_rate <= 0
            or learning_rate.item() != mx.array(recorded_rate, dtype=learning_rate.dtype).item()
        ):
            raise ValueError("Optimizer learning rate differs from the recorded contract")
    rng = mx.load(str(directory / "random.safetensors"))
    if set(rng) != {str(i) for i in range(len(mx.random.state))} or any(
        v.shape != mx.random.state[int(k)].shape or v.dtype != mx.random.state[int(k)].dtype
        for k, v in rng.items()
    ):
        raise ValueError("Random state structure mismatch")
    if arrays["step"].size != 1 or arrays["step"].item() != state["step"]:
        raise ValueError("Optimizer step/cursor mismatch")
    load_adapter(model, directory)
    optimizer.state = tree_unflatten(list(arrays.items()))
    for i, key in enumerate(mx.random.state):
        key[:] = rng[str(i)]
    mx.eval(optimizer.state, mx.random.state)
    return state
