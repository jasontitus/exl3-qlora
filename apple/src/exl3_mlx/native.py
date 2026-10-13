"""Frozen native BF16/MXFP8 trunk linears and original-basis adapters."""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.lora import LoRALinear as MLXLMLoRALinear

from .activation_config import validate_activation_mode
from .attention import quantize_activation


class NativeLinear(nn.Module):
    def __init__(
        self, archive, key, *, rows=None, quantize_input=True, activation_mode="checkpoint"
    ):
        super().__init__()
        self.activation_mode = validate_activation_mode(activation_mode)
        self.weight = archive.array(key + ".weight", rows)
        self.quantize_input = quantize_input
        self.quantized = self.weight.dtype == mx.uint32
        if self.quantized:
            self.scales = archive.array(key + ".scales", rows)
            if self.scales.dtype != mx.uint8 or self.scales.shape != (
                self.weight.shape[0],
                self.weight.shape[1] // 8,
            ):
                raise ValueError("Expected MXFP8 group-32 weights")
        elif self.weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
            raise ValueError("Unsupported trunk weight dtype")
        self.out_features = self.weight.shape[0]
        self.in_features = self.weight.shape[1] * (4 if self.quantized else 1)
        self.freeze()
        mx.eval(self.parameters())

    def __call__(self, x):
        if self.quantized:
            inp = (
                quantize_activation(x)
                if self.quantize_input and self.activation_mode == "checkpoint"
                else x
            )
            return mx.quantized_matmul(
                inp, self.weight, self.scales, group_size=32, bits=8, mode="mxfp8"
            )
        return x @ self.weight.astype(x.dtype).T


class LoRALinear(MLXLMLoRALinear):
    """MLX-LM's adapter forward, initialized without a dense placeholder base.

    ``from_base`` assumes MLX's affine quantization and temporarily allocates an
    nn.Linear. Neither is appropriate for disk-streamed EXL3 experts.
    """

    def __init__(self, base, *, rank=8, alpha=16.0):
        nn.Module.__init__(self)
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Positive rank and alpha required")
        self.linear = base
        self.dropout = nn.Dropout(0.0)
        self.scale = alpha / rank
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.alpha = rank, alpha
        bound = 1 / math.sqrt(self.in_features)
        self.lora_a = mx.random.uniform(-bound, bound, (self.in_features, rank))
        self.lora_b = mx.zeros((rank, self.out_features))
        self.linear.freeze()

    def fuse(self, dequantize=False):
        raise NotImplementedError("Keep EXL3 adapters separate; fusion requires requantization")
