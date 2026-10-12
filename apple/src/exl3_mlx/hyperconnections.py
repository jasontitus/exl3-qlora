"""Differentiable V4.1 hyper-connections and threaded pre-mixes.

Adapted from DeepSeek-V4.1-Flash inference/model.py and kernel.py, MIT license;
see configs/DEEPSEEK_LICENSE and reports/deepseek-reference-source.json.
"""

import mlx.core as mx
import mlx.nn as nn


def rms_norm(x, weight, eps=1e-20):
    f = x.astype(mx.float32)
    return (f * mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + eps) * weight).astype(x.dtype)


def sinkhorn(logits, *, iters=20, eps=1e-6):
    if iters < 1 or eps <= 0:
        raise ValueError("Sinkhorn needs positive iterations and epsilon")
    comb = mx.softmax(logits.astype(mx.float32), axis=-1) + eps
    comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    for _ in range(1, iters):
        comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
        comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
    return comb


def hc_pre(streams, pre):
    return mx.sum(streams.astype(mx.float32) * pre[..., None], axis=-2).astype(streams.dtype)


def hc_post(value, residual, post, comb):
    mixed = mx.einsum("...jk,...jd->...kd", comb, residual.astype(mx.float32))
    return (post[..., None] * value.astype(mx.float32)[..., None, :] + mixed).astype(value.dtype)


def expand_streams(embedding, hc_mult=4):
    streams = mx.broadcast_to(
        embedding[..., None, :], (*embedding.shape[:-1], hc_mult, embedding.shape[-1])
    )
    pre = mx.broadcast_to(mx.eye(hc_mult, dtype=mx.float32)[0], (*embedding.shape[:-1], hc_mult))
    return streams, pre


class HyperConnection(nn.Module):
    def __init__(
        self, weight, scale, bias, *, hc_mult=4, norm_eps=1e-20, sinkhorn_iters=20, eps=1e-6
    ):
        super().__init__()
        mixes = (2 + hc_mult) * hc_mult
        if (
            weight.ndim != 2
            or weight.shape[0] != mixes
            or weight.shape[1] % hc_mult
            or bias.shape != (mixes,)
            or scale.shape != (3,)
        ):
            raise ValueError("Invalid hyper-connection parameter shapes")
        self.weight, self.scale, self.bias = weight, scale, bias
        self.hc_mult, self.norm_eps = hc_mult, norm_eps
        self.sinkhorn_iters, self.eps = sinkhorn_iters, eps
        self.freeze()

    def __call__(self, streams):
        h = self.hc_mult
        flat = streams.astype(mx.float32).reshape(*streams.shape[:-2], -1)
        mixes = (flat @ self.weight.T) * mx.rsqrt(
            mx.mean(flat * flat, axis=-1, keepdims=True) + self.norm_eps
        )
        pre = mx.sigmoid(mixes[..., :h] * self.scale[0] + self.bias[:h]) + self.eps
        post = 2 * mx.sigmoid(mixes[..., h : 2 * h] * self.scale[1] + self.bias[h : 2 * h])
        comb = (mixes[..., 2 * h :] * self.scale[2] + self.bias[2 * h :]).reshape(
            *flat.shape[:-1], h, h
        )
        return pre, post, sinkhorn(comb, iters=self.sinkhorn_iters, eps=self.eps)


class DeepSeekBlock(nn.Module):
    """Exact V4.1 residual ordering around supplied attention and MoE modules.

    Attention uses the previous block's pre-mix; FFN uses this attention's new
    pre-mix. FFN's new pre-mix is returned to the next block (or final collapse).
    Engram injection occurs before entering a block, as in the source model.
    """

    def __init__(self, attention, ffn, attn_hc, ffn_hc, attn_norm, ffn_norm, *, eps=1e-20):
        super().__init__()
        self.attention, self.ffn = attention, ffn
        self.attn_hc, self.ffn_hc = attn_hc, ffn_hc
        self.attn_norm, self.ffn_norm = attn_norm, ffn_norm
        self.eps = eps
        self.freeze(keys=["attn_norm", "ffn_norm"], recurse=False)

    def __call__(self, streams, pre_mix):
        attn_pre, post, comb = self.attn_hc(streams)
        value = self.attention(rms_norm(hc_pre(streams, pre_mix), self.attn_norm, self.eps))
        streams = hc_post(value, streams, post, comb)
        ffn_pre, post, comb = self.ffn_hc(streams)
        value = self.ffn(rms_norm(hc_pre(streams, attn_pre), self.ffn_norm, self.eps))
        return hc_post(value, streams, post, comb), ffn_pre
