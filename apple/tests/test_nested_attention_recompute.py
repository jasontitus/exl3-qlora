"""New numerical implementation of the same first-order shared-KV derivative.

The original bitwise-gradient requirement failed and is preserved in reports.
Use existing composed-oracle tolerances; never promise exact update trajectories.
"""

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_unflatten
from test_composed_gradient import fixture, make_model
from test_prefix_cache_model import assert_bits_equal

import exl3_mlx.deepseek as deepseek
from exl3_mlx.attention import empty_attention_state
from exl3_mlx.native import LoRALinear
from exl3_mlx.recompute import recompute


def setup(dtype, nonzero, mode, nested):
    data = fixture()
    model = make_model(data)
    mx.random.seed(947)
    for block in model.blocks:
        block.checkpoint_attention = nested
        block.attn.activation_mode = mode
        if block.attn.compressor is not None:
            projection = LoRALinear(block.attn.compressor["wkv"], rank=2, alpha=4.0)
            projection.lora_b = mx.full(projection.lora_b.shape, 0.0125)
            block.attn.compressor["wkv"] = projection
    if not nonzero:
        model.update(
            tree_unflatten(
                [
                    (name, mx.zeros_like(value) if name.endswith("lora_b") else value)
                    for name, value in tree_flatten(model.trainable_parameters())
                ]
            )
        )
    model.train()
    streams = mx.array(data["streams"]).astype(dtype)
    pre = mx.array(data["pre"])
    return data, model, streams, pre


def run(model, data, parameters, streams, pre):
    model.update(parameters)
    kv = empty_attention_state(1, streams.shape[1], 32, 1, streams.dtype)
    for index, block in enumerate(model.blocks):
        embedding = mx.array(data["embedding"]) if index == 0 else mx.zeros((0,))
        streams, pre, kv = recompute(block, streams, pre, kv, embedding)
    # Nonuniform objective explicitly includes shared KV and mixer outputs.
    weights = mx.arange(streams.size, dtype=mx.float32).reshape(streams.shape) / streams.size
    loss = (
        mx.mean(streams.astype(mx.float32) ** 2 * weights)
        + 0.1 * mx.mean(kv[0].astype(mx.float32) ** 2)
        + 0.03 * mx.mean(pre.astype(mx.float32) ** 2)
    )
    return loss


def assert_numerically_same(left, right):
    a, b = tree_flatten(left), tree_flatten(right)
    assert [key for key, _ in a] == [key for key, _ in b]
    for (key, x), (_, y) in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.dtype == mx.bfloat16 or not mx.issubdtype(x.dtype, mx.floating):
            assert_bits_equal(x, y)
            continue
        expected, actual = np.array(x).astype(np.float64), np.array(y).astype(np.float64)
        assert np.all(np.isfinite(expected)) and np.all(np.isfinite(actual)), key
        # Reuse the independently declared composed-oracle bounds. An extra
        # norm guard prevents atol from hiding an entirely tiny nonzero leaf.
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-7, err_msg=key)
        assert np.all(actual[expected == 0] == 0), key
        norm = np.linalg.norm(expected)
        if norm:
            assert np.linalg.norm(actual - expected) / norm <= 2e-4, key
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=key)


def forward_states(model, data, h, pre):
    states = []
    kv = empty_attention_state(1, h.shape[1], 32, 1, h.dtype)
    for index, block in enumerate(model.blocks):
        embedding = mx.array(data["embedding"]) if index == 0 else mx.zeros((0,))
        h, pre, kv = recompute(block, h, pre, kv, embedding)
        mx.eval(h, pre, kv)
        states.append((h, pre, kv))
    return states


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("nonzero", [False, True])
@pytest.mark.parametrize("mode", ["checkpoint", "bf16"])
def test_two_blocks_same_derivative_numerical_grads_adam_and_exact_initial_forward(
    dtype, nonzero, mode
):
    data, original, streams, pre = setup(dtype, nonzero, mode, False)
    _, nested, _, _ = setup(dtype, nonzero, mode, True)
    optimizers = [optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)]
    assert_bits_equal(
        forward_states(original, data, streams, pre), forward_states(nested, data, streams, pre)
    )
    for _ in range(2):
        results = []
        for model in (original, nested):
            params = model.trainable_parameters()
            result = mx.value_and_grad(
                lambda p, h, a: run(model, data, p, h, a), argnums=(0, 1, 2)
            )(params, streams, pre)
            mx.eval(result)
            results.append(result)
        assert_bits_equal(results[0][0], results[1][0])
        assert_numerically_same(results[0][1], results[1][1])
        assert mx.any(results[1][1][1] != 0).item()
        assert mx.any(results[1][1][2] != 0).item()
        # Source-block attention feeds shared KV consumed in the later block.
        assert mx.any(
            results[1][1][0]["blocks"][0]["attn"]["compressor"]["wkv"]["lora_b"] != 0
        ).item()
        for model, optimizer, (_, gradients) in zip(
            (original, nested), optimizers, results, strict=True
        ):
            optimizer.update(model, gradients[0])
            mx.eval(model.parameters(), optimizer.state)
        assert_numerically_same(original.trainable_parameters(), nested.trainable_parameters())
        assert_numerically_same(optimizers[0].state, optimizers[1].state)
        assert_numerically_same(
            forward_states(original, data, streams, pre), forward_states(nested, data, streams, pre)
        )


def test_default_and_eval_keep_direct_path_and_training_calls_complete_vjp(monkeypatch):
    data, model, streams, pre = setup(mx.float32, True, "checkpoint", False)
    block = model.blocks[0]
    assert block.checkpoint_attention is False
    kv = empty_attention_state(1, streams.shape[1], 32, 1, streams.dtype)
    embedding = mx.array(data["embedding"])
    calls = []

    def observed(module, *args, **kwargs):
        calls.append((module, kwargs))
        assert not kwargs.get("frozen_inputs", False)
        return recompute(module, *args, **kwargs)

    monkeypatch.setattr(deepseek, "recompute", observed)
    original = block(streams, pre, kv, embedding)
    mx.eval(original)
    assert not calls
    block.checkpoint_attention = True
    block.eval()
    evaluated = block(streams, pre, kv, embedding)
    mx.eval(evaluated)
    assert not calls
    assert_bits_equal(original, evaluated)
    block.train()
    checked = block(streams, pre, kv, embedding)
    mx.eval(checked)
    assert calls == [(block.attn, {})]
    assert_bits_equal(original, checked)
