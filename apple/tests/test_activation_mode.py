"""Experimental differentiable-path bypass; checkpoint indexer policy is preserved."""

from collections import Counter
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_composed_gradient import FrozenLinear, fixture, make_model

import exl3_mlx.attention as attention
import exl3_mlx.native as native
from exl3_mlx.native import NativeLinear
from exl3_mlx.recompute import recompute


def assert_same(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    if left.dtype == mx.bfloat16:
        left, right = left.view(mx.uint16), right.view(mx.uint16)
    np.testing.assert_array_equal(np.array(left), np.array(right))


def archive():
    rng = np.random.default_rng(8001)
    encoded = rng.integers(0x20, 0x50, size=(32, 64), dtype=np.uint8).copy().view(np.uint32)
    values = {
        "linear.weight": mx.array(encoded),
        "linear.scales": mx.full((32, 2), 122, dtype=mx.uint8),
    }
    return SimpleNamespace(array=lambda key, rows=None: values[key])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("mode", ["checkpoint", "bf16"])
def test_native_mode_keeps_packed_weights_and_matches_explicit_input_policy(dtype, mode):
    source = archive()
    layer = NativeLinear(source, "linear", activation_mode=mode)
    x = mx.array(np.random.default_rng(91).normal(size=(3, 64)).astype(np.float32)).astype(dtype)
    dy = mx.array(np.random.default_rng(92).normal(size=(3, 32)).astype(np.float32)).astype(dtype)

    def expected(value):
        if mode == "checkpoint":
            value = attention.quantize_activation(value)
        return mx.quantized_matmul(
            value, layer.weight, layer.scales, group_size=32, bits=8, mode="mxfp8"
        )

    (got,), (dx,) = mx.vjp(layer, [x], [dy])
    (want,), (expected_dx,) = mx.vjp(expected, [x], [dy])
    assert_same(got, want)
    assert_same(dx, expected_dx)
    assert layer.weight.dtype == mx.uint32 and layer.scales.dtype == mx.uint8
    assert_same(layer.weight, source.array("linear.weight"))
    assert_same(layer.scales, source.array("linear.scales"))
    assert not tree_flatten(layer.trainable_parameters())
    assert mx.linalg.norm(dx.astype(mx.float32)).item() > 0


def test_checkpoint_default_bitwise_and_bf16_does_not_invoke_native_quantizer(monkeypatch):
    source = archive()
    implicit = NativeLinear(source, "linear")
    explicit = NativeLinear(source, "linear", activation_mode="checkpoint")
    bypass = NativeLinear(source, "linear", activation_mode="bf16")
    prequantized = NativeLinear(source, "linear", quantize_input=False)
    x = mx.arange(128, dtype=mx.float32).reshape(2, 64) / 97
    assert_same(implicit(x), explicit(x))

    def forbidden(value):
        raise AssertionError("Explicit activation quantizer must not execute")

    monkeypatch.setattr(native, "quantize_activation", forbidden)
    # The choice is a persistent module attribute, including during a later
    # custom recomputation's backward. There is no ambient/global mode context.
    assert_same(bypass(x), prequantized(x))
    left = mx.grad(lambda value: mx.sum(recompute(bypass, value)))(x)
    right = mx.grad(lambda value: mx.sum(prequantized(value)))(x)
    assert_same(left, right)


def make_attention(mode):
    data = fixture()
    reference = make_model(data).blocks[0].attn
    # Dense trunk projections isolate the three attention quantization sites.
    # The separate NativeLinear cases cover packed MXFP8 input quantization.
    for layer in reference.linears.values():
        (layer.linear if "linear" in layer else layer).quant = False
    config = dict(reference.config)
    config.update(index_n_heads=2, index_head_dim=32, index_topk=3)
    rng = np.random.default_rng(8002)
    indexer = attention.FrozenIndexer(
        0,
        config,
        {
            name: FrozenLinear(
                {"weight": rng.normal(0, 0.1, (out, 32)).astype(np.float32), "quant": True}
            )
            for name, out in (("wk", 32), ("wq_b", 64), ("weights_proj", 2))
        },
        mx.ones(32),
        query_chunk=4,
    )
    layer = attention.ShortAttention(
        0,
        config,
        reference.linears,
        reference.norms,
        reference.sink,
        reference.wo_a_groups,
        reference.compressor,
        indexer,
        activation_mode=mode,
    )
    return layer


@pytest.mark.parametrize("mode", ["checkpoint", "bf16"])
def test_attention_bypass_preserves_indexer_quantizers_and_recompute_gradients(monkeypatch, mode):
    layer = make_attention(mode)
    x = mx.array(np.random.default_rng(94).normal(size=(1, 19, 32)).astype(np.float32))
    state = attention.empty_attention_state(1, 19, 32, 32, x.dtype)
    original = attention.quantize_activation
    calls = []

    def count(value, kind="fp8"):
        calls.append(kind)
        return original(value, kind)

    with monkeypatch.context() as patch:
        patch.setattr(attention, "quantize_activation", count)
        output, saved = layer(x, state)
        mx.eval(output, saved)
    assert layer.indexer.activation_mode == "checkpoint"
    # Indexer wk/wq_b/weights_proj still quantize inputs, and both query/key
    # vectors still use FP4 E8M0. Window/compressed/output differ by mode only.
    assert Counter(calls) == (
        Counter(fp8=5, fp4=1, fp4_e8m0=2) if mode == "checkpoint" else Counter(fp8=3, fp4_e8m0=2)
    )
    assert saved[2].shape[-1] == 3 and not tree_flatten(layer.indexer.trainable_parameters())

    def loss(model, checkpoint):
        out, values = recompute(model, x, state) if checkpoint else model(x, state)
        return mx.mean(out**2) + 0.1 * mx.mean(values[0] ** 2)

    plain, pg = nn.value_and_grad(layer, lambda m: loss(m, False))(layer)
    checked, cg = nn.value_and_grad(layer, lambda m: loss(m, True))(layer)
    mx.eval(plain, pg, checked, cg)
    assert_same(plain, checked)
    for (pk, pv), (ck, cv) in zip(tree_flatten(pg), tree_flatten(cg), strict=True):
        assert pk == ck
        assert_same(pv, cv)
        assert mx.linalg.norm(pv).item() > 0


def test_indexer_rejects_projection_with_bypassed_input_quantization():
    with pytest.raises(ValueError, match="FrozenIndexer"):
        attention.FrozenIndexer(0, {}, {"wk": SimpleNamespace(activation_mode="bf16")})


def test_attention_mode_changes_forward_without_changing_stored_parameters():
    baseline, bypass = make_attention("checkpoint"), make_attention("bf16")
    x = mx.array(np.random.default_rng(95).normal(size=(1, 19, 32)).astype(np.float32))
    state = attention.empty_attention_state(1, 19, 32, 32, x.dtype)
    a, _ = baseline(x, state)
    b, _ = bypass(x, state)
    assert not mx.array_equal(a, b).item()
    for (ak, av), (bk, bv) in zip(
        tree_flatten(baseline.parameters()), tree_flatten(bypass.parameters()), strict=True
    ):
        assert ak == bk
        assert_same(av, bv)
