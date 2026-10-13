"""Exact projection/gradient/optimizer comparisons; parent runs Metal tests."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_prefix_cache_model import assert_bits_equal

import exl3_mlx.grouped_projection as grouped
from exl3_mlx.native import NativeLinear
from exl3_mlx.recompute import recompute


def groups(quantized=False, *, quantize_input=True):
    rng = np.random.default_rng(942)
    modules = []
    for width in (32, 64, 32):
        if quantized:
            weight = rng.integers(0x20, 0x50, (width, 64), dtype=np.uint8).copy().view(np.uint32)
            tensors = {
                "x.weight": mx.array(weight),
                "x.scales": mx.full((width, 2), 122, dtype=mx.uint8),
            }
        else:
            tensors = {"x.weight": mx.array(rng.normal(0, 0.08, (width, 64)).astype(np.float32))}
        archive = SimpleNamespace(array=lambda name, rows=None, t=tensors: t[name])
        modules.append(NativeLinear(archive, "x", quantize_input=quantize_input))
    return modules


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("quantized,quantize_input", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_forward_and_input_gradient_exact(dtype, quantized, quantize_input, noncontiguous):
    projections = groups(quantized, quantize_input=quantize_input)
    rng = np.random.default_rng(944)
    shape = (2, 3, 7, 64) if noncontiguous else (2, 7, 3, 64)
    x = mx.array(rng.normal(0, 0.1, shape).astype(np.float32)).astype(dtype)
    if noncontiguous:
        x = x.transpose(0, 2, 1, 3)
    dy = mx.array(rng.normal(0, 0.2, (2, 7, 128)).astype(np.float32)).astype(dtype)
    reference = mx.vjp(
        lambda value: grouped.grouped_projection_reference(value, projections), [x], [dy]
    )
    bounded = mx.vjp(lambda value: grouped.grouped_projection(value, projections), [x], [dy])
    mx.eval(reference, bounded)
    assert_bits_equal(reference, bounded)
    assert mx.any(bounded[1][0] != 0).item()


class AdapterBeforeGroups(nn.Module):
    def __init__(self, dtype, nonzero, *, bounded):
        super().__init__()
        self.projections = groups(quantized=True, quantize_input=False)
        self.dtype, self.bounded = dtype, bounded
        rng = np.random.default_rng(945)
        self.lora_a = mx.array(rng.normal(0, 0.1, (192, 2)).astype(np.float32))
        self.lora_b = (
            mx.array(rng.normal(0, 0.1, (2, 192)).astype(np.float32))
            if nonzero
            else mx.zeros((2, 192))
        )

    def __call__(self, x):
        y = (x.astype(mx.float32) + (x.astype(mx.float32) @ self.lora_a) @ self.lora_b).astype(
            self.dtype
        )
        y = y.reshape(*y.shape[:-1], 3, 64)
        function = (
            grouped.grouped_projection if self.bounded else grouped.grouped_projection_reference
        )
        return function(y, self.projections)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("nonzero", [False, True])
def test_enclosing_recompute_parameter_gradients_and_two_adam_updates(dtype, nonzero):
    reference = AdapterBeforeGroups(dtype, nonzero, bounded=False)
    bounded = AdapterBeforeGroups(dtype, nonzero, bounded=True)
    rng = np.random.default_rng(946)
    x = mx.array(rng.normal(0, 0.1, (2, 5, 192)).astype(np.float32)).astype(dtype)
    target = mx.array(rng.normal(0, 0.05, (2, 5, 128)).astype(np.float32))
    optimizers = [optim.Adam(learning_rate=1e-3), optim.Adam(learning_rate=1e-3)]

    def loss(model):
        return mx.mean((recompute(model, x).astype(mx.float32) - target) ** 2)

    for _ in range(2):
        left = nn.value_and_grad(reference, loss)(reference)
        right = nn.value_and_grad(bounded, loss)(bounded)
        mx.eval(left, right)
        assert_bits_equal(left, right)
        for model, optimizer, (_, gradient) in zip(
            (reference, bounded), optimizers, (left, right), strict=True
        ):
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state)
        assert_bits_equal(reference.trainable_parameters(), bounded.trainable_parameters())
        assert_bits_equal(optimizers[0].state, optimizers[1].state)


def test_trainable_projection_falls_back_and_preserves_weight_gradient(monkeypatch):
    reference = AdapterBeforeGroups(mx.float32, True, bounded=False)
    bounded = AdapterBeforeGroups(mx.float32, True, bounded=True)
    for model in (reference, bounded):
        # Use the plain BF16/FP32 native branch so weights are differentiable.
        model.projections = groups(quantized=False)
        model.projections[1].unfreeze()
    x = mx.full((1, 3, 192), 0.125)
    left = nn.value_and_grad(reference, lambda model: mx.sum(model(x) ** 2))(reference)

    def forbidden(value):
        raise AssertionError("Frozen custom boundary must not run for trainable groups")

    monkeypatch.setattr(grouped, "materialize", forbidden)
    right = nn.value_and_grad(bounded, lambda model: mx.sum(model(x) ** 2))(bounded)
    mx.eval(left, right)
    assert_bits_equal(left, right)
    assert mx.any(right[1]["projections"][1]["weight"] != 0).item()
    assert any("weight" in name for name, _ in tree_flatten(right[1]))


def test_invalid_group_count_rejected():
    with pytest.raises(ValueError, match="one projection"):
        grouped.grouped_projection(mx.zeros((1, 3, 2, 64)), groups())
    with pytest.raises(ValueError, match="one projection"):
        grouped.grouped_projection(mx.zeros((1, 3, 0, 64)), [])
