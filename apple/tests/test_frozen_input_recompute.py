"""Actual MLX checks of explicitly frozen recompute inputs; parent runs GPU tests."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from test_prefix_cache_model import assert_bits_equal, make_model

import exl3_mlx.deepseek as deepseek
from exl3_mlx.recompute import recompute


class ProbeBlock(nn.Module):
    def __init__(self, nonzero):
        super().__init__()
        rng = np.random.default_rng(514)
        self.weight = mx.array(rng.normal(0, 0.15, (8, 8)).astype(np.float32))
        self.freeze()
        self.lora_a = mx.array(rng.normal(0, 0.1, (8, 2)).astype(np.float32))
        self.lora_b = (
            mx.array(rng.normal(0, 0.1, (2, 8)).astype(np.float32)) if nonzero else mx.zeros((2, 8))
        )

    def __call__(self, h, pre, state, bank):
        pooled, keys, selected, candidates = state
        x = mx.sum(h.astype(mx.float32) * pre[..., None], axis=-2)
        x = x + mx.mean(pooled.astype(mx.float32), axis=1, keepdims=True) * 0.2
        x = x + keys.astype(mx.float32) * 0.1 + bank.astype(mx.float32) * 0.03
        value = mx.tanh(x @ self.weight + (x @ self.lora_a) @ self.lora_b).astype(h.dtype)
        out = (h.astype(mx.float32) + value.astype(mx.float32)[..., None, :]).astype(h.dtype)
        mix = mx.sigmoid(mx.mean(out.astype(mx.float32), axis=-1))
        return out, mix, (value, keys, selected, candidates)


def inputs(dtype):
    rng = np.random.default_rng(315)
    h = mx.array(rng.normal(0, 0.2, (1, 5, 4, 8)).astype(np.float32)).astype(dtype)
    pre = mx.array(rng.uniform(0.1, 0.9, (1, 5, 4)).astype(np.float32))
    pooled = mx.array(rng.normal(0, 0.2, (1, 5, 8)).astype(np.float32)).astype(dtype)
    keys = pooled * 0.5
    selected = mx.zeros((1, 5, 2), dtype=mx.int32)
    candidates = mx.ones((1, 5, 1), dtype=mx.uint8)
    bank = pooled * 0.1
    return h, pre, (pooled, keys, selected, candidates), bank


def objective(block, state, frozen_inputs):
    h, pre, kv = recompute(block, *state, frozen_inputs=frozen_inputs)
    weights = mx.arange(h.size, dtype=mx.float32).reshape(h.shape) / h.size
    return (
        mx.mean(h.astype(mx.float32) ** 2 * weights)
        + mx.mean(pre**2)
        + 0.3 * mx.mean(kv[0].astype(mx.float32) ** 2)
    )


def model_loss(model, tokens):
    values = model.hidden_states(tokens).astype(mx.float32)
    target = mx.arange(values.size, dtype=mx.float32).reshape(values.shape) / values.size
    return mx.mean((values - target) ** 2)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("nonzero", [False, True])
def test_parameter_gradients_forward_and_adam_exact(dtype, nonzero):
    full, frozen = ProbeBlock(nonzero), ProbeBlock(nonzero)
    state = inputs(dtype)
    optimizers = [optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)]
    assert_bits_equal(recompute(full, *state), recompute(frozen, *state, frozen_inputs=True))
    for _ in range(2):
        a = nn.value_and_grad(full, lambda model: objective(model, state, False))(full)
        b = nn.value_and_grad(frozen, lambda model: objective(model, state, True))(frozen)
        mx.eval(a, b)
        assert_bits_equal(a, b)
        for model, optimizer, (_, gradient) in zip((full, frozen), optimizers, (a, b), strict=True):
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state)
        assert_bits_equal(full.trainable_parameters(), frozen.trainable_parameters())
        assert_bits_equal(optimizers[0].state, optimizers[1].state)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_default_keeps_input_gradients_opt_in_returns_exact_zero(dtype):
    model, state = ProbeBlock(True), inputs(dtype)
    h, pre, kv, bank = state

    def apply(h, pre, pooled, frozen):
        return objective(model, (h, pre, (pooled, *kv[1:]), bank), frozen)

    outputs, full = mx.vjp(
        lambda h, pre, pooled: apply(h, pre, pooled, False), [h, pre, kv[0]], [mx.array(1.0)]
    )
    outputs2, frozen = mx.vjp(
        lambda h, pre, pooled: apply(h, pre, pooled, True), [h, pre, kv[0]], [mx.array(1.0)]
    )
    mx.eval(outputs, outputs2, full, frozen)
    assert_bits_equal(outputs, outputs2)
    assert all(mx.any(value != 0).item() for value in full)
    assert all(mx.all(value == 0).item() for value in frozen)


def test_explicit_contract_validation():
    module, state = ProbeBlock(False), inputs(mx.float32)
    with pytest.raises(ValueError, match="boolean"):
        recompute(module, *state, frozen_inputs=1)
    module.freeze()
    with pytest.raises(ValueError, match="trainable"):
        recompute(module, *state, frozen_inputs=True)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("cache_bytes", [0, 1 << 20])
@pytest.mark.parametrize("expert_layers", [(1,), (0,)])
def test_actual_model_boundary_matches_unpruned_reference(
    dtype, cache_bytes, expert_layers, monkeypatch
):
    baseline = make_model(
        cache_bytes=cache_bytes, dtype=dtype, attention_layers=(2,), expert_layers=expert_layers
    )
    changed = make_model(
        cache_bytes=cache_bytes, dtype=dtype, attention_layers=(2,), expert_layers=expert_layers
    )
    optimizers = [optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)]
    tokens = mx.array([[1, 2, 3, 4, 5]])
    observations = []

    def unpruned(module, *args, **kwargs):
        observations.append(
            (
                next(i for i, block in enumerate(baseline.blocks) if block is module),
                kwargs["frozen_inputs"],
            )
        )
        return recompute(module, *args, **{**kwargs, "frozen_inputs": False})

    for _ in range(2):
        with monkeypatch.context() as m:
            m.setattr(deepseek, "recompute", unpruned)
            left = nn.value_and_grad(baseline, lambda model: model_loss(model, tokens))(baseline)
        right = nn.value_and_grad(changed, lambda model: model_loss(model, tokens))(changed)
        mx.eval(left, right)
        assert_bits_equal(left, right)
        assert mx.any(left[1]["blocks"][expert_layers[0]]["lora_b"] != 0).item()
        for model, optimizer, (_, gradient) in zip(
            (baseline, changed), optimizers, (left, right), strict=True
        ):
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state)
        assert_bits_equal(baseline.trainable_parameters(), changed.trainable_parameters())
        assert_bits_equal(optimizers[0].state, optimizers[1].state)
    if expert_layers == (1,):
        assert any(index == 1 and flag for index, flag in observations)
        assert all(not flag for index, flag in observations if index != 1)
    else:
        # Block0 produces trainable pooled KV consumed by later blocks. Cutting
        # at block2 would silently discard this required shared-state gradient.
        assert observations and not any(flag for _, flag in observations)


@pytest.mark.parametrize("mutation", ["earlier_block", "nonblock"])
def test_live_trainability_prevents_stale_boundary_cut(mutation, monkeypatch):
    model = make_model(cache_bytes=0, dtype=mx.float32)
    if mutation == "earlier_block":
        model.blocks[0].unfreeze(keys=["weight"], recurse=False)
    else:
        model.unfreeze(keys=["norm"], recurse=False)
    flags = []

    def observed(module, *args, **kwargs):
        flags.append(kwargs["frozen_inputs"])
        return recompute(module, *args, **kwargs)

    monkeypatch.setattr(deepseek, "recompute", observed)
    value, gradient = nn.value_and_grad(
        model, lambda model: model_loss(model, mx.array([[1, 2, 3]]))
    )(model)
    mx.eval(value, gradient)
    assert flags and not any(flags)
    expected = gradient["blocks"][0]["weight"] if mutation == "earlier_block" else gradient["norm"]
    assert mx.any(expected != 0).item()
