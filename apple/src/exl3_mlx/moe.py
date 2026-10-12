"""DeepSeek V4.1 text router and complete experts, with input gradients preserved.

Math follows DeepSeek's pinned inference/model.py Gate, Expert and MoE. Frozen
router parameters still allow differentiation of routing weights with respect to
the input. Discrete expert selection has no derivative.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .materialize import materialize


def _component_vjp(module, inputs, cotangent, *, weights_only=False):
    """Recompute exactly one component and release its graph before continuing."""
    parameters = module.trainable_parameters()
    items = tree_flatten(parameters)
    count = len(items)

    def apply(*primals):
        if count:
            module.update(tree_unflatten([(key, v) for (key, _), v in zip(items, primals)]))
        result = module(*primals[count:])
        return result[0] if weights_only else result

    primals = [mx.stop_gradient(value) for _, value in items]
    primals += [mx.stop_gradient(value) for value in inputs]
    try:
        _, gradients = mx.vjp(apply, primals, [cotangent])
        # eval alone does not cut nested custom-function traces. Round-trip the
        # small results before the next expert's backward graph is constructed.
        gradients = [materialize(value) for value in gradients]
    finally:
        module.update(parameters)
    return {key: value for (key, _), value in zip(items, gradients)}, gradients[count:]


def _route_plan(indices):
    ids = np.array(indices)
    for expert_id in np.unique(ids):
        token_rows, slots = np.nonzero(ids == expert_id)
        yield (
            int(expert_id),
            mx.array(token_rows.astype(np.int32)),
            mx.array(slots.astype(np.int32)),
        )


class DeepSeekExpert(nn.Module):
    def __init__(self, w1, w2, w3, *, swiglu_limit=10.0):
        super().__init__()
        if not math.isfinite(swiglu_limit) or swiglu_limit < 0:
            raise ValueError("swiglu_limit must be nonnegative and finite")
        if (
            w1.in_features != w3.in_features
            or w1.out_features != w3.out_features
            or w2.in_features != w1.out_features
            or w2.out_features != w1.in_features
        ):
            raise ValueError("Expert projection dimensions do not match")
        if any("bias" in w for w in (w1, w2, w3)):
            raise ValueError("DeepSeek experts do not have projection biases")
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.in_features = w1.in_features
        self.swiglu_limit = float(swiglu_limit)

    def __call__(self, x, weights=None):
        gate = self.w1(x).astype(mx.float32)
        up = self.w3(x).astype(mx.float32)
        if self.swiglu_limit > 0:
            gate = mx.minimum(gate, self.swiglu_limit)  # no lower clamp on gate
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
        hidden = nn.silu(gate) * up
        if weights is not None:
            hidden = hidden * weights
        # Weight before the down projection, including the pre-matmul dtype rounding.
        return self.w2(hidden.astype(x.dtype))


class DeepSeekGate(nn.Module):
    def __init__(
        self, weight, bias, *, top_k=6, route_scale=1.5, gate_temp=1.0, norm_topk_prob=True
    ):
        super().__init__()
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("Router weight must be [experts, hidden], bias [experts]")
        if not 1 <= top_k <= weight.shape[0]:
            raise ValueError("top_k outside expert count")
        if gate_temp <= 0 or route_scale <= 0 or not math.isfinite(gate_temp + route_scale):
            raise ValueError("Routing scale and temperature must be positive and finite")
        self.weight, self.bias = weight, bias
        self.top_k, self.route_scale = top_k, route_scale
        self.gate_temp, self.norm_topk_prob = gate_temp, norm_topk_prob
        self.in_features, self.num_experts = weight.shape[1], weight.shape[0]
        self.freeze()

    def __call__(self, x):
        logits = x.astype(mx.float32) @ mx.stop_gradient(self.weight).astype(mx.float32).T
        scores = mx.sqrt(mx.logaddexp(logits / self.gate_temp, 0.0))
        biased = mx.stop_gradient(scores + self.bias)
        # Stable descending sort gives an explicit low-index tie policy. Upstream
        # torch.topk tie ordering is unspecified; parity fixtures avoid ties.
        indices = mx.argsort(-biased, axis=-1)[..., : self.top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.norm_topk_prob and self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return weights * self.route_scale, indices


class DeepSeekMoE(nn.Module):
    """Group only selected token/expert pairs and add the always-on shared expert.

    A first-order custom VJP recomputes one expert at a time, including its
    router-weight derivative. Eager materialized accumulation avoids retaining
    one full token-by-hidden scatter buffer per expert. Adapter parameters are
    explicit primals; both expert adapters and upstream adapters keep gradients.
    """

    def __init__(self, gate, experts, shared_expert):
        super().__init__()
        if len(experts) != gate.num_experts:
            raise ValueError("Router and expert counts differ")
        if any(e.in_features != gate.in_features for e in [*experts, shared_expert]):
            raise ValueError("Expert and router hidden dimensions differ")
        self.gate, self.experts, self.shared_expert = gate, list(experts), shared_expert

    def __call__(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if flat.shape[0] == 0:
            return mx.zeros_like(x)
        original = self.trainable_parameters()

        @mx.custom_function
        def forward(parameters, inp):
            inp = mx.stop_gradient(inp)
            self.update(tree_map(mx.stop_gradient, parameters))
            try:
                weights, indices = self.gate(inp)
                weights, indices = materialize(weights), materialize(indices)
                output = mx.zeros(inp.shape, dtype=mx.float32)
                for expert_id, rows, slots in _route_plan(indices):
                    value = self.experts[expert_id](inp[rows], weights[rows, slots, None])
                    output = materialize(output.at[rows].add(value.astype(mx.float32)))
                    del value
                output = output + self.shared_expert(inp).astype(mx.float32)
                return materialize(output.astype(inp.dtype))
            finally:
                self.update(original)

        @forward.vjp
        def backward(primals, cotangent, output):
            parameters, inp = primals
            inp, cotangent = mx.stop_gradient(inp), mx.stop_gradient(cotangent)
            self.update(tree_map(mx.stop_gradient, parameters))
            try:
                weights, indices = self.gate(inp)
                weights, indices = materialize(weights), materialize(indices)
                dx = mx.zeros(inp.shape, dtype=mx.float32)
                dw = mx.zeros(weights.shape, dtype=mx.float32)
                parameter_grads = {}

                def collect(prefix, values):
                    parameter_grads.update({prefix + key: value for key, value in values.items()})

                for expert_id, rows, slots in _route_plan(indices):
                    dp, (di, dr) = _component_vjp(
                        self.experts[expert_id],
                        (inp[rows], weights[rows, slots, None]),
                        cotangent[rows],
                    )
                    collect(f"experts.{expert_id}.", dp)
                    dx = materialize(dx.at[rows].add(di.astype(mx.float32)))
                    # Each token/expert pair occurs once, with a unique slot.
                    dw = materialize(dw.at[rows, slots].add(dr[:, 0].astype(mx.float32)))
                    del dp, di, dr
                dp, (di,) = _component_vjp(self.shared_expert, (inp,), cotangent)
                collect("shared_expert.", dp)
                dx = materialize(dx + di.astype(mx.float32))
                del dp, di
                dp, (di,) = _component_vjp(self.gate, (inp,), dw, weights_only=True)
                collect("gate.", dp)
                dx = materialize((dx + di.astype(mx.float32)).astype(inp.dtype))
                # Preserve the exact pytree, including empty frozen submodules.
                leaves = iter(
                    parameter_grads.get(key, mx.zeros_like(value))
                    for key, value in tree_flatten(parameters)
                )
                return tree_map(lambda _: next(leaves), parameters), dx
            finally:
                self.update(original)

        return forward(original, flat).reshape(shape)
