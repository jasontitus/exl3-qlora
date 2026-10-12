"""Dense dispatch and retained-allocation regressions for the eager MoE VJP."""

import gc

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_block_training import Dense

from exl3_mlx.materialize import materialize
from exl3_mlx.moe import DeepSeekExpert, DeepSeekGate, DeepSeekMoE
from exl3_mlx.native import LoRALinear


def dense_dispatch(model, x, *, detach_weights=False):
    flat = x.reshape(-1, x.shape[-1])
    weights, ids = model.gate(flat)
    if detach_weights:
        weights = mx.stop_gradient(weights)
    output = model.shared_expert(flat).astype(mx.float32)
    # Independently evaluate every expert for every token with a dense routing
    # coefficient. Bias-free experts output zero for a zero weighted hidden row.
    for expert_id, expert in enumerate(model.experts):
        coefficient = mx.sum(mx.where(ids == expert_id, weights, 0), axis=-1, keepdims=True)
        output = output + expert(flat, coefficient).astype(mx.float32)
    return output.astype(x.dtype).reshape(x.shape)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16], ids=["fp32", "bf16"])
def test_bounded_moe_dense_input_router_and_all_expert_adapter_gradients(dtype, record_property):
    mx.random.seed(94)
    dim, intermediate, count = 16, 24, 5

    class InputDtypeDense(Dense):
        def __call__(self, x):
            # Match NativeLinear's dense path instead of promoting BF16 inputs
            # through a float32 weight in the generic test helper.
            return x @ self.weight.astype(x.dtype).T

    errors = {}

    def equivalent(value, expected_value, role):
        value = np.array(mx.array(value).astype(mx.float32))
        expected_value = np.array(mx.array(expected_value).astype(mx.float32))
        error = float(np.max(np.abs(value - expected_value)))
        scale = float(np.max(np.abs(expected_value)))
        errors[role] = {"max_abs_error": error, "max_reference_magnitude": scale}
        if dtype == mx.bfloat16:
            # Four BF16 epsilons at the tensor's maximum magnitude allow the
            # different cast/accumulation order across routed expert gradients.
            # This also bounds float32 LoRA gradients fed by BF16 intermediates.
            assert error <= scale * (4 / 128) + 3e-7, (role, error, scale)
        else:
            np.testing.assert_allclose(value, expected_value, rtol=5e-5, atol=3e-7)

    def make_expert():
        projections = [
            InputDtypeDense(dim, intermediate),
            InputDtypeDense(intermediate, dim),
            InputDtypeDense(dim, intermediate),
        ]
        projections = [LoRALinear(p, rank=2, alpha=2) for p in projections]
        for p in projections:
            p.lora_b = mx.random.normal(p.lora_b.shape) * 0.1
        return DeepSeekExpert(*projections, swiglu_limit=0.8)

    gate = DeepSeekGate(
        mx.random.normal((count, dim)) * 0.4,
        mx.array([0.1, -0.1, -20, 0.0, 0.0]),
        top_k=2,
    )
    model = DeepSeekMoE(gate, [make_expert() for _ in range(count)], make_expert())
    x = mx.random.normal((2, 7, dim)).astype(dtype)
    dy = mx.random.normal(x.shape).astype(dtype)
    (out,), (dx,) = mx.vjp(model, [x], [dy])
    (expected,), (dx_expected,) = mx.vjp(lambda value: dense_dispatch(model, value), [x], [dy])
    equivalent(out, expected, "output")
    equivalent(dx, dx_expected, "input_gradient")
    (_,), (without_router,) = mx.vjp(
        lambda value: dense_dispatch(model, value, detach_weights=True), [x], [dy]
    )
    assert mx.max(mx.abs(dx - without_router)).item() > 1e-5

    before = {key: np.array(value).copy() for key, value in tree_flatten(model.parameters())}
    loss, actual = nn.value_and_grad(model, lambda m: mx.sum(m(x) * dy))(model)
    wanted_loss, wanted = nn.value_and_grad(model, lambda m: mx.sum(dense_dispatch(m, x) * dy))(
        model
    )
    equivalent(loss, wanted_loss, "loss")
    for (key, value), (expected_key, expected_value) in zip(
        tree_flatten(actual), tree_flatten(wanted), strict=True
    ):
        assert key == expected_key and key.endswith(("lora_a", "lora_b"))
        equivalent(value, expected_value, key)
        if key.startswith("experts.2."):
            assert mx.max(mx.abs(value)).item() == 0
        else:
            assert mx.max(mx.abs(value)).item() > 0
    for key, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(np.array(value), before[key])
    record_property("numerical_errors", errors)
    if dtype == mx.bfloat16:
        summary = {key: errors[key] for key in ("output", "input_gradient", "loss")}
        summary["adapter_max_abs_error"] = max(
            value["max_abs_error"] for key, value in errors.items() if "lora_" in key
        )
        print("BF16 routed-vs-dense errors:", summary)


class EagerScale(nn.Module):
    """Small eager custom primitive reproducing streamed-projection boundaries."""

    def __init__(self, dim, factor):
        super().__init__()
        self.in_features = self.out_features = dim
        self.factor = factor

    def __call__(self, x):
        factor = self.factor

        @mx.custom_function
        def forward(value):
            return materialize(mx.stop_gradient(value) * factor)

        @forward.vjp
        def backward(primals, cotangent, output):
            return materialize(cotangent * factor)

        return forward(x)


def test_many_expert_accumulation_has_bounded_peak_memory():
    # 64 full [4096,128] float32 accumulators would consume 128 MiB. The bounded
    # forward/backward should retain only a handful, plus small route tensors.
    count, length, dim = 64, 4096, 128
    gate = DeepSeekGate(mx.eye(count, dim) * 4, mx.zeros(count), top_k=2)

    def expert(i):
        return DeepSeekExpert(
            EagerScale(dim, 0.1), EagerScale(dim, 0.2 + i * 0.001), EagerScale(dim, 0.1)
        )

    model = DeepSeekMoE(gate, [expert(i) for i in range(count)], expert(count))
    x = mx.eye(count, dim)[mx.arange(length) % count]
    mx.eval(x, model.parameters())
    gc.collect()
    old_cache = mx.set_cache_limit(0)
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    try:
        (_,), (gradient,) = mx.vjp(model, [x], [mx.ones_like(x)])
        mx.eval(gradient)
        assert mx.all(mx.isfinite(gradient)).item()
        assert mx.max(mx.abs(gradient)).item() > 0
        assert mx.get_peak_memory() - baseline < 64 << 20
    finally:
        mx.set_cache_limit(old_cache)
