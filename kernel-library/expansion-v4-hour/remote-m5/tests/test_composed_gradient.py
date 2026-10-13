"""Independent smooth two-block oracle for the declared first-order surrogate.

This uses synthetic FP32 weights, not a full-model/native-engine parity claim.
The NumPy float64 replay does not call model math, custom VJPs, or MLX autodiff.
It holds each nominal activation quantization residual and each router's selected
IDs fixed. Thus finite differences test the declared STE, not quantized stairs.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import exl3_mlx.attention as attention
from exl3_mlx.deepseek import TrainingBlock
from exl3_mlx.engram import Engram
from exl3_mlx.hyperconnections import HyperConnection, hc_pre, rms_norm
from exl3_mlx.moe import DeepSeekExpert, DeepSeekGate, DeepSeekMoE
from exl3_mlx.native import LoRALinear
from exl3_mlx.recompute import recompute
from exl3_mlx.training import projected_cross_entropy


def fixture():
    rng = np.random.default_rng(65123)

    def random(shape, scale=0.08):
        return rng.normal(0, scale, shape).astype(np.float32)

    def linear(k, n, *, quant=True, adapter=False):
        values = {"weight": random((n, k)), "quant": quant}
        if adapter:
            values.update(a=random((k, 2), 0.15), b=random((2, n), 0.025), scale=2.0)
        return values

    def expert(*, routed=False, shared=False):
        # Routed EXL3 projections do not quantize their activations. Dense frozen
        # weights here isolate MoE/custom-boundary composition from EXL3 decoding.
        return {
            name: linear(
                32, 32, quant=shared, adapter=(routed and name == "w1") or (shared and name == "w3")
            )
            for name in ("w1", "w2", "w3")
        }

    config = {
        "num_attention_heads": 2,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "index_topk": 512,
        "rms_norm_eps": 1e-20,
        "compress_ratios": [2, 2],
        "o_groups": 2,
        "sliding_window": 3,
        "rope_scaling": {
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    }
    blocks = []
    for index in range(2):
        blocks.append(
            {
                "linears": {
                    "wq_a": linear(32, 32, adapter=True),
                    "wq_b": linear(32, 64),
                    "wkv": linear(32, 32),
                    "wo_b": linear(32, 32, adapter=True),
                },
                "norms": {
                    name: rng.uniform(0.7, 1.3, 32).astype(np.float32)
                    for name in ("q", "kv", "compressor", "attn", "ffn")
                },
                "groups": [linear(32, 16, quant=False) for _ in range(2)],
                "compressor": {name: linear(32, 32, quant=False) for name in ("wkv", "wgate")}
                if index == 0
                else None,
                "sink": random((2,), 0.5),
                "connections": [
                    {
                        "weight": random((24, 128)),
                        "scale": np.array([0.4, 0.3, 0.2], np.float32),
                        "bias": random((24,), 0.2),
                    }
                    for _ in range(2)
                ],
                "router": {
                    "weight": random((3, 32)),
                    "bias": np.array([0.2, 0.0, -0.3], np.float32),
                },
                "experts": [expert(routed=i == 0) for i in range(3)],
                "shared": expert(shared=True),
                "engram": {
                    "projection": linear(32, 160),
                    "q": rng.uniform(0.6, 1.4, (4, 32)).astype(np.float32),
                    "k": rng.uniform(0.6, 1.4, (4, 32)).astype(np.float32),
                }
                if index == 0
                else None,
            }
        )
    return {
        "blocks": blocks,
        "config": config,
        "streams": random((1, 19, 4, 32), 1.0)
        * np.array([0.4, 0.8, 1.2, 1.6], np.float32)[None, None, :, None],
        "pre": rng.uniform(0.2, 0.8, (1, 19, 4)).astype(np.float32),
        "embedding": random((1, 19, 2, 16), 0.5),
        "head": random((13, 32), 0.15),
        "labels": rng.integers(0, 13, size=19, dtype=np.int32),
        "norm": rng.uniform(0.7, 1.3, 32).astype(np.float32),
    }


class FrozenLinear(nn.Module):
    def __init__(self, description):
        super().__init__()
        self.weight = mx.array(description["weight"])
        self.quant = description["quant"]
        self.out_features, self.in_features = self.weight.shape
        self.freeze()

    def __call__(self, x):
        values = attention.quantize_activation(x) if self.quant else x
        return values @ self.weight.T


def make_model(data):
    def linear(description):
        result = FrozenLinear(description)
        if "a" in description:
            result = LoRALinear(result, rank=2, alpha=4.0)
            result.lora_a, result.lora_b = map(mx.array, (description["a"], description["b"]))
        return result

    def expert(description):
        return DeepSeekExpert(
            *(linear(description[n]) for n in ("w1", "w2", "w3")), swiglu_limit=1.0
        )

    result = nn.Module()
    result.blocks = []
    for index, description in enumerate(data["blocks"]):
        norm = {name: mx.array(value) for name, value in description["norms"].items()}
        attn = attention.ShortAttention(
            index,
            data["config"],
            {k: linear(v) for k, v in description["linears"].items()},
            {k: norm[k] for k in ("q", "kv", "compressor")},
            mx.array(description["sink"]),
            [linear(v) for v in description["groups"]],
            None
            if description["compressor"] is None
            else {k: linear(v) for k, v in description["compressor"].items()},
        )
        gate = DeepSeekGate(
            mx.array(description["router"]["weight"]),
            mx.array(description["router"]["bias"]),
            top_k=2,
        )
        moe = DeepSeekMoE(
            gate, [expert(v) for v in description["experts"]], expert(description["shared"])
        )
        connections = [
            HyperConnection(*(mx.array(v[name]) for name in ("weight", "scale", "bias")))
            for v in description["connections"]
        ]
        e = description["engram"]
        engram = (
            None
            if e is None
            else Engram(linear(e["projection"]), mx.array(e["q"]), mx.array(e["k"]))
        )
        result.blocks.append(
            TrainingBlock(attn, moe, *connections, norm["attn"], norm["ffn"], engram)
        )
    return result


class AnchorTape:
    def __init__(self, anchors):
        self.anchors, self.cursor = anchors, 0

    def quant(self, value, kind="fp8"):
        expected_kind, nominal, rounded = self.anchors[self.cursor]
        self.cursor += 1
        assert expected_kind == kind and nominal.shape == value.shape
        # The explicit anchored form is exact at nominal inputs and has derivative
        # identity. An ordinary quantizer is intentionally absent from this oracle.
        return rounded + (value - nominal)


def softmax(x, axis=-1):
    values = np.exp(x - x.max(axis=axis, keepdims=True))
    return values / values.sum(axis=axis, keepdims=True)


def norm(x, weight):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-20) * weight


def np_linear(x, description, parameters, prefix, tape):
    value = tape.quant(x) if description["quant"] else x
    result = value @ description["weight"].astype(np.float64).T
    if "a" in description:
        result += description["scale"] * (
            x @ parameters[prefix + ".lora_a"] @ parameters[prefix + ".lora_b"]
        )
    return result


def rotate(x, positions, config, inverse=False):
    dim = config["qk_rope_head_dim"]
    base, scaling = config["compress_rope_theta"], config["rope_scaling"]
    frequency = 1.0 / base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)

    def correction(r):
        return (
            dim
            * math.log(scaling["original_max_position_embeddings"] / (r * 2 * math.pi))
            / (2 * math.log(base))
        )

    lower = max(math.floor(correction(scaling["beta_fast"])), 0)
    upper = min(math.ceil(correction(scaling["beta_slow"])), dim - 1)
    ramp = np.clip((np.arange(dim // 2) - lower) / max(upper - lower, 1e-3), 0, 1)
    frequency *= 1 - ramp + ramp / scaling["factor"]
    angle = np.asarray(positions)[:, None] * frequency * (-1 if inverse else 1)
    shape = (1, x.shape[1], *((1,) * (x.ndim - 3)), dim // 2)
    cosine, sine = np.cos(angle).reshape(shape), np.sin(angle).reshape(shape)
    result = x.copy()
    a, b = x[..., -dim::2], x[..., -dim + 1 :: 2]
    result[..., -dim::2] = a * cosine - b * sine
    result[..., -dim + 1 :: 2] = a * sine + b * cosine
    return result


def np_attention(x, shared, d, config, params, prefix, tape):
    b, s, _ = x.shape

    def project(name, value):
        return np_linear(value, d["linears"][name], params, prefix + ".attn.linears." + name, tape)

    query_reduced = norm(project("wq_a", x), d["norms"]["q"])
    query = rotate(project("wq_b", query_reduced).reshape(b, s, 2, 32), np.arange(s), config)
    window = tape.quant(rotate(norm(project("wkv", x), d["norms"]["kv"]), np.arange(s), config))
    if d["compressor"] is not None:

        def project_compressed(name):
            return np_linear(
                x, d["compressor"][name], params, prefix + ".attn.compressor." + name, tape
            )

        count = s // 2
        values = project_compressed("wkv")[:, : count * 2].reshape(b, count, 2, 32)
        weights = softmax(project_compressed("wgate")[:, : count * 2].reshape(values.shape), 2)
        latent = norm(np.sum(values * weights, axis=2), d["norms"]["compressor"])
        shared = tape.quant(rotate(latent, np.arange(count) * 2, config), "fp4")
    output = np.empty(query.shape, np.float64)
    for token in range(s):
        keys = np.concatenate(
            [
                window[:, max(0, token - config["sliding_window"] + 1) : token + 1],
                shared[:, : (token + 1) // 2],
            ],
            axis=1,
        )
        scores = np.einsum("bhd,bkd->bhk", query[:, token], keys) / math.sqrt(32)
        sinks = np.broadcast_to(d["sink"][None, :, None], (*scores.shape[:-1], 1))
        probability = softmax(np.concatenate([scores, sinks], axis=-1))[..., :-1]
        output[:, token] = np.einsum("bhk,bkd->bhd", probability, keys)
    output = tape.quant(rotate(output, np.arange(s), config, inverse=True).reshape(b, s, 2, 32))
    output = np.concatenate(
        [
            np_linear(
                output[:, :, group], desc, params, prefix + f".attn.wo_a_groups.{group}", tape
            )
            for group, desc in enumerate(d["groups"])
        ],
        axis=-1,
    )
    return project("wo_b", output), shared


def np_connection(streams, description):
    flat = streams.reshape(*streams.shape[:-2], -1)
    mixes = (flat @ description["weight"].astype(np.float64).T) / np.sqrt(
        np.mean(flat * flat, axis=-1, keepdims=True) + 1e-20
    )
    scale, bias = description["scale"], description["bias"]
    pre = 1 / (1 + np.exp(-(mixes[..., :4] * scale[0] + bias[:4]))) + 1e-6
    post = 2 / (1 + np.exp(-(mixes[..., 4:8] * scale[1] + bias[4:8])))
    matrix = softmax((mixes[..., 8:] * scale[2] + bias[8:]).reshape(*flat.shape[:-1], 4, 4)) + 1e-6
    matrix /= matrix.sum(axis=-2, keepdims=True) + 1e-6
    for _ in range(19):
        matrix /= matrix.sum(axis=-1, keepdims=True) + 1e-6
        matrix /= matrix.sum(axis=-2, keepdims=True) + 1e-6
    return pre, post, matrix


def np_post(value, streams, post, matrix):
    return value[..., None, :] * post[..., None] + np.einsum("...jk,...jd->...kd", matrix, streams)


def np_engram(streams, embedding, description, params, prefix, tape):
    projection = np_linear(
        embedding.reshape(1, 19, -1),
        description["projection"],
        params,
        prefix + ".engram.wkv",
        tape,
    )
    key, value = projection[..., :128].reshape(streams.shape), projection[..., 128:]
    dot = np.sum(streams * description["q"] * description["k"] * key, axis=-1)
    dot /= np.sqrt(np.mean(streams * streams, axis=-1) + 1e-20)
    dot /= np.sqrt(np.mean(key * key, axis=-1) + 1e-20) * math.sqrt(32)
    signed_root = np.where(dot < 0, -1.0, 1.0) * np.sqrt(np.maximum(np.abs(dot), 1e-6))
    return streams + value[..., None, :] / (1 + np.exp(-signed_root[..., None]))


def np_expert(x, description, params, prefix, tape, weights=None):
    def project(name, value):
        return np_linear(value, description[name], params, prefix + "." + name, tape)

    gate = np.minimum(project("w1", x), 1.0)
    up = np.clip(project("w3", x), -1.0, 1.0)
    hidden = gate / (1 + np.exp(-gate)) * up
    return project("w2", hidden if weights is None else hidden * weights)


def np_moe(x, description, params, prefix, ids, tape):
    flat = x.reshape(-1, 32)
    logits = flat @ description["router"]["weight"].astype(np.float64).T
    scores = np.sqrt(np.logaddexp(logits, 0.0))
    weights = np.take_along_axis(scores, ids, axis=-1)
    weights = 1.5 * weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    result = np.zeros(flat.shape, np.float64)
    for expert in np.unique(ids):
        rows, slots = np.nonzero(ids == expert)
        result[rows] += np_expert(
            flat[rows],
            description["experts"][expert],
            params,
            prefix + f".ffn.experts.{expert}",
            tape,
            weights[rows, slots, None],
        )
    result += np_expert(flat, description["shared"], params, prefix + ".ffn.shared_expert", tape)
    return result.reshape(x.shape)


def numpy_loss(data, parameters, streams, pre, anchors, route_ids):
    streams, pre = streams.astype(np.float64), pre.astype(np.float64)
    tape, shared = AnchorTape(anchors), np.zeros((1, 0, 32), np.float64)
    for index, description in enumerate(data["blocks"]):
        prefix = f"blocks.{index}"
        if description["engram"] is not None:
            streams = np_engram(
                streams,
                data["embedding"].astype(np.float64),
                description["engram"],
                parameters,
                prefix,
                tape,
            )
        attn_pre, post, matrix = np_connection(streams, description["connections"][0])
        mixed = np.sum(streams * pre[..., None], axis=-2)
        value, shared = np_attention(
            norm(mixed, description["norms"]["attn"]),
            shared,
            description,
            data["config"],
            parameters,
            prefix,
            tape,
        )
        streams = np_post(value, streams, post, matrix)
        pre, post, matrix = np_connection(streams, description["connections"][1])
        mixed = np.sum(streams * attn_pre[..., None], axis=-2)
        value = np_moe(
            norm(mixed, description["norms"]["ffn"]),
            description,
            parameters,
            prefix,
            route_ids[index],
            tape,
        )
        streams = np_post(value, streams, post, matrix)
    hidden = norm(np.sum(streams * pre[..., None], axis=-2), data["norm"])[0]
    logits = hidden @ data["head"].astype(np.float64).T
    probability = softmax(logits)
    loss = -np.log(probability[np.arange(len(logits)), data["labels"]]).mean()
    loss += 0.1 * np.mean(shared * shared) + 0.03 * np.mean(pre * pre)
    assert tape.cursor == len(anchors)
    return float(loss)


def mlx_loss(model, data, parameters, streams, pre, *, checkpoint):
    model.update(parameters)
    shared = attention.empty_attention_state(1, 19, 32, 1, mx.float32)
    for index, block in enumerate(model.blocks):
        embedding = mx.array(data["embedding"]) if index == 0 else mx.zeros((0,))
        if checkpoint:
            streams, pre, shared = recompute(block, streams, pre, shared, embedding)
        else:
            streams, pre, shared = block(streams, pre, shared, embedding)
    hidden = rms_norm(hc_pre(streams, pre), mx.array(data["norm"]))[0]
    result = projected_cross_entropy(
        lambda x: x @ mx.array(data["head"]).T, hidden, mx.array(data["labels"]), chunk_size=3
    )
    return result + 0.1 * mx.mean(shared[0] ** 2) + 0.03 * mx.mean(pre**2)


def test_composed_recompute_gradient_matches_independent_anchored_replay(monkeypatch):
    data, anchors, routes = fixture(), [], []
    model = make_model(data)
    parameters = model.trainable_parameters()
    streams, pre = mx.array(data["streams"]), mx.array(data["pre"])
    original_quant, original_gate = attention.quantize_activation, DeepSeekGate.__call__

    def record_quant(value, kind="fp8"):
        rounded = original_quant(value, kind)
        anchors.append(
            (kind, np.array(value).astype(np.float64), np.array(rounded).astype(np.float64))
        )
        return rounded

    def record_gate(gate, value):
        weights, indices = original_gate(gate, value)
        routes.append(np.array(indices))
        return weights, indices

    # Capture a plain forward only. Backward recomputation must not append more
    # quantization or routing anchors to this nominal execution's tape.
    with monkeypatch.context() as patch:
        patch.setattr(attention, "quantize_activation", record_quant)
        patch.setattr(DeepSeekGate, "__call__", record_gate)
        value = mlx_loss(model, data, parameters, streams, pre, checkpoint=False)
        mx.eval(value)
    assert len(routes) == 2 and len(anchors) >= 15

    def plain(p, x, a):
        return mlx_loss(model, data, p, x, a, checkpoint=False)

    def checked(p, x, a):
        return mlx_loss(model, data, p, x, a, checkpoint=True)

    value_plain, grad_plain = mx.value_and_grad(plain, argnums=(0, 1, 2))(parameters, streams, pre)
    value_checked, grad_checked = mx.value_and_grad(checked, argnums=(0, 1, 2))(
        parameters, streams, pre
    )
    mx.eval(value_plain, value_checked, grad_plain, grad_checked)
    assert value_checked.item() == pytest.approx(value_plain.item(), abs=1e-6)
    for (name, actual), (other, expected) in zip(
        tree_flatten(grad_checked), tree_flatten(grad_plain), strict=True
    ):
        assert name == other
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=2e-4, atol=2e-7)

    flat_parameters = {k: np.array(v).astype(np.float64) for k, v in tree_flatten(parameters)}
    flat_gradients = {k: np.array(v).astype(np.float64) for k, v in tree_flatten(grad_checked[0])}

    def independent(p, x, a):
        return numpy_loss(data, p, x, a, anchors, routes)

    expected_loss = independent(flat_parameters, data["streams"], data["pre"])
    assert value_checked.item() == pytest.approx(expected_loss, abs=8e-6)

    # Every trainable array is probed separately along its normalized gradient,
    # including nonzero-B A gradients, routed and shared expert adapters. Each
    # derivative is independently obtained from the NumPy-only replay.
    epsilon = 2e-5
    for name, gradient in flat_gradients.items():
        magnitude = np.linalg.norm(gradient)
        assert magnitude > 1e-10, f"Unexercised adapter: {name}"
        direction = gradient / magnitude
        plus, minus = dict(flat_parameters), dict(flat_parameters)
        plus[name] = plus[name] + epsilon * direction
        minus[name] = minus[name] - epsilon * direction
        numeric = (
            independent(plus, data["streams"], data["pre"])
            - independent(minus, data["streams"], data["pre"])
        ) / (2 * epsilon)
        assert magnitude == pytest.approx(numeric, rel=8e-4, abs=1e-6), name

    for index, initial in enumerate((data["streams"], data["pre"]), 1):
        gradient = np.array(grad_checked[index]).astype(np.float64)
        direction = gradient / np.linalg.norm(gradient)
        high = [data["streams"].astype(np.float64), data["pre"].astype(np.float64)]
        low = [value.copy() for value in high]
        high[index - 1] += epsilon * direction
        low[index - 1] -= epsilon * direction
        numeric = (independent(flat_parameters, *high) - independent(flat_parameters, *low)) / (
            2 * epsilon
        )
        assert np.linalg.norm(gradient) == pytest.approx(numeric, rel=8e-4, abs=1e-6)

    # A fixed unrelated joint direction can expose cross-leaf summation errors
    # that separate gradient-aligned probes alone would not exercise.
    rng = np.random.default_rng(113)
    delta = {k: rng.normal(size=v.shape) / math.sqrt(v.size) for k, v in flat_parameters.items()}
    plus = {k: v + epsilon * delta[k] for k, v in flat_parameters.items()}
    minus = {k: v - epsilon * delta[k] for k, v in flat_parameters.items()}
    numeric = (
        independent(plus, data["streams"], data["pre"])
        - independent(minus, data["streams"], data["pre"])
    ) / (2 * epsilon)
    predicted = sum(np.sum(flat_gradients[k] * delta[k]) for k in delta)
    assert predicted == pytest.approx(numeric, rel=2e-3, abs=2e-6)
