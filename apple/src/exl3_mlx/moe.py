"""DeepSeek V4.1 text router and complete experts, with input gradients preserved.

Math follows DeepSeek's pinned inference/model.py Gate, Expert and MoE. Frozen
router parameters still allow differentiation of routing weights with respect to
the input. Discrete expert selection has no derivative.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


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

    The small integer route plan is synchronized to CPU; floating-point activation
    and gradient math remains on MLX. This supports eager autodiff/checkpointing,
    not compilation of data-dependent expert counts.
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
        weights, indices = self.gate(flat)
        ids = np.array(indices)
        output = mx.zeros(flat.shape, dtype=mx.float32)
        for expert_id in np.unique(ids):
            token_rows, slots = np.nonzero(ids == expert_id)
            rows, slots = mx.array(token_rows.astype(np.int32)), mx.array(slots.astype(np.int32))
            value = self.experts[int(expert_id)](flat[rows], weights[rows, slots, None])
            output = output.at[rows].add(value.astype(mx.float32))
        output = output + self.shared_expert(flat).astype(mx.float32)
        return output.astype(x.dtype).reshape(shape)
