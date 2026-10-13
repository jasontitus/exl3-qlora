"""Bounded NumPy Qwen4Exp GatedResidual math, not DeepSeek mHC.

Four per-stream RMS norms -> flattened low-rank SiLU/sigmoid channel gate ->
MEAN across streams. Sites inject y with2*sigmoid gates into the RAW residual;
the final mixer has no injection and no additional RMSNorm. There is no
Sinkhorn or combine matrix. Semantic sources are pinned below, never imported.

Normal math uses float32; float64 input preserves float64 for derivative checks,
as the local training reference does. NumPy reduction/GEMM rounding is not a
bitwise emulation of Torch/CUDA/Metal or Sushi's serving kernels. The analytical
adjoint is explicitly continuous float64 math, not a derivative of rounded
floating-point execution. No sublayer, complete model, or training is provided.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STREAMS = 4
MAX_HIDDEN = 64
MAX_RANK = 64
MAX_TOKENS = 32
MAX_ARRAY_ELEMENTS = 4 * MAX_HIDDEN * MAX_RANK
SOURCE_REVISION = "b766887941695bc0c28e25b59ca24db63916e8b8"
SOURCE_ROOT = Path("/Users/jasontitus/experiments/exl3-qlora")
SOURCE_SHA256 = {
    "exllamav3/training/hyperconnections.py": "0d2873bf4fe26416f23db6c8cbd884cded24a473dbb72710f0f25479671dbe6c",
    "exllamav3/modules/hyperconnections.py": "841ca30d37ea8900b5ae1c9e4b4862d9580ec0c4c92342b1b460968c0926c2f9",
    "exllamav3/architecture/qwen4_exp.py": "f0781ec11aabc78694ed5084d24fb31093c0500a5c985a00276188c241c165bb",
    "exllamav3/training/backbone.py": "0265a93b28089470d9d109a61a997403201145456e70c098f3061026c3c076d9",
    "exllamav3/training/native_llama.py": "12c99ff24bb54067a081ceb9e6b52154cf811afd231bd30a13847c70d5729f5e",
}
CONFIG_SHA256 = "f8adca807ce1deef83a781d28560fc38d2335e592939d9cf31930fa9a553eda1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_provenance(root=SOURCE_ROOT):
    """Verify exact local semantic sources/config without importing GPU packages."""
    actual = {
        name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
        for name in SOURCE_SHA256
    }
    require(actual == SOURCE_SHA256, "Semantic source identity changed")
    config = Path(__file__).resolve().parents[1] / "sources/config.json"
    raw = config.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "Pinned Qwen config changed"
    )
    parsed = json.loads(raw)
    require(
        parsed.get("model_type") == "qwen4_exp"
        and parsed.get("text_config", {}).get("hc_count") == STREAMS,
        "Expected Qwen4Exp four-stream metadata",
    )
    return {
        "semantic_sources_sha256": actual,
        "source_revision_context": SOURCE_REVISION,
        "config_sha256": CONFIG_SHA256,
        "numpy_version": np.__version__,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dtype_scope": "FP32 mathematical reference; FP64 continuous derivative oracle. No backend bitwise parity.",
        "target_config_dimensions": {
            key: parsed["text_config"][key]
            for key in ("hc_count", "hc_lowrank", "hidden_size", "rms_norm_eps")
        },
        "fixture_caps": {
            "hidden_size": MAX_HIDDEN,
            "rank": MAX_RANK,
            "tokens": MAX_TOKENS,
        },
        "real_payload_verified": False,
        "gpu_parity_qualified": False,
        "end_to_end_model_supported": False,
    }


def _array(value, name):
    require(isinstance(value, np.ndarray), f"{name} must be a NumPy array")
    require(
        value.size <= MAX_ARRAY_ELEMENTS and 0 < value.size,
        f"{name} exceeds tiny fixture bounds",
    )
    require(
        value.dtype in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")),
        f"{name} must use float16/32/64",
    )
    require(np.isfinite(value).all(), f"{name} must be finite")
    return value


def _immutable(value):
    # bytes-backed views cannot be made writable, unlike owned read-only ndarrays.
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(
        value.shape
    )


def _sigmoid(x):
    result = np.empty_like(x)
    positive = x >= 0
    with np.errstate(under="ignore"):
        result[positive] = 1 / (1 + np.exp(-x[positive]))
        exp = np.exp(x[~positive])
        result[~positive] = exp / (1 + exp)
    return result


def _finite(*values):
    require(
        all(value is None or np.isfinite(value).all() for value in values),
        "Reference arithmetic produced nonfinite values",
    )


def _streams(value, *, continuous=False):
    value = _array(value, "streams")
    require(
        value.ndim == 4
        and value.shape[2] == STREAMS
        and 0 < value.shape[-1] <= MAX_HIDDEN
        and 0 < value.shape[0] * value.shape[1] <= MAX_TOKENS,
        "Expected bounded[b,t,4,d] stream stack",
    )
    if continuous:
        require(
            value.dtype == np.float64,
            "Continuous input adjoint requires float64 streams",
        )
    return value


@dataclass(frozen=True)
class QwenGatedResidual:
    """Prepared tables: norm_w ALREADY includes1+w; down/up/inject[out,in].

    Constructor preserves supplied floating precision for mathematical fixtures.
    from_checkpoint_tables explicitly reproduces the local loader's FP16 tables
    and FP32(1+raw_norm) preparation, independently of the checkpoint's storage
    dtype. No weights are loaded or dequantized by this module.
    """

    norm_w: np.ndarray
    down: np.ndarray
    up: np.ndarray
    inject: np.ndarray | None
    eps: float = 1e-6

    def __post_init__(self):
        require(
            type(self.eps) in (int, float) and math.isfinite(self.eps) and self.eps > 0,
            "eps must be positive finite scalar",
        )
        norm = _array(self.norm_w, "norm_w")
        require(
            norm.ndim == 2
            and norm.shape[0] == STREAMS
            and 0 < norm.shape[1] <= MAX_HIDDEN,
            "norm_w must have shape[4,d] within fixture bounds",
        )
        width = norm.size
        down = _array(self.down, "down")
        require(
            down.ndim == 2 and down.shape[1] == width and 0 < down.shape[0] <= MAX_RANK,
            "down must have shape[rank,4*d]",
        )
        require(
            _array(self.up, "up").shape == (width, down.shape[0]),
            "up must have shape[4*d,rank]",
        )
        if self.inject is not None:
            require(
                _array(self.inject, "inject").shape == (STREAMS, width),
                "inject must have shape[4,4*d]",
            )
        for name in ("norm_w", "down", "up", "inject"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _immutable(value))

    @classmethod
    def from_checkpoint_tables(cls, norm_raw, down, up, inject=None, *, eps=1e-6):
        """Local ExLlama loader convention; no claim of Sushi resident dtype parity."""
        for name, array in (("norm_raw", norm_raw), ("down", down), ("up", up)):
            _array(array, name)
        if inject is not None:
            _array(inject, "inject")
        require(
            norm_raw.ndim in (1, 2) and norm_raw.size % STREAMS == 0,
            "Raw norm table must flatten to[4,d]",
        )
        with np.errstate(over="ignore", invalid="ignore"):
            return cls(
                (norm_raw.astype(np.float32) + np.float32(1)).reshape(STREAMS, -1),
                down.astype(np.float16),
                up.astype(np.float16),
                inject.astype(np.float16) if inject is not None else None,
                eps,
            )

    def _forward(self, streams):
        streams = _streams(streams)
        require(
            streams.shape[-1] == self.norm_w.shape[1],
            "Stream/table hidden dimension mismatch",
        )
        dtype = np.float64 if streams.dtype == np.float64 else np.float32
        x = streams.astype(dtype, copy=False)
        norm, down, up = (
            value.astype(dtype, copy=False)
            for value in (self.norm_w, self.down, self.up)
        )
        with np.errstate(
            over="ignore", invalid="ignore", divide="ignore", under="ignore"
        ):
            variance = np.mean(x * x, axis=-1, keepdims=True)
            _finite(variance)
            inv = 1 / np.sqrt(variance + dtype(self.eps))
            z = x * inv * norm
            flat = z.reshape(*z.shape[:2], -1)
            a = (flat @ down.T) / STREAMS
            sigmoid_a = _sigmoid(a)
            t = a * sigmoid_a
            logits = t @ up.T
            _finite(a, t, logits)
            weights = _sigmoid(logits).reshape(z.shape)
            mixed = np.mean(weights * z, axis=-2)
            post = None
            if self.inject is not None:
                inject_logits = (flat @ self.inject.astype(dtype).T) / STREAMS
                _finite(inject_logits)
                post = 2 * _sigmoid(inject_logits)
            _finite(inv, z, weights, mixed, post)
        return post, mixed, (x, norm, down, up, inv, z, a, sigmoid_a, weights)

    def mix(self, streams):
        """Return(post,mixed), both FP32 except FP64-input oracle mode."""
        post, mixed, _ = self._forward(streams)
        return post, mixed

    def input_adjoint(self, streams, d_mixed, d_post=None):
        """Analytical float64 VJP of mix wrt streams, with fixed tables.

        Treats operations as continuous real math. Does not differentiate casts,
        model sublayers, parameters or rounded FP32 execution.
        """
        _streams(streams, continuous=True)
        post, mixed, cache = self._forward(streams)
        require(
            _array(d_mixed, "d_mixed").shape == mixed.shape
            and d_mixed.dtype == np.float64,
            "d_mixed must be matching float64 cotangent",
        )
        if post is None:
            require(d_post is None, "Final mixer has no post output")
        elif d_post is not None:
            require(
                _array(d_post, "d_post").shape == post.shape
                and d_post.dtype == np.float64,
                "d_post must be matching float64 cotangent",
            )
        x, norm, down, up, inv, z, a, sigmoid_a, weights = cache
        with np.errstate(
            over="ignore", invalid="ignore", divide="ignore", under="ignore"
        ):
            direct = d_mixed[..., None, :] / STREAMS
            dz = direct * weights
            dg = (direct * z * weights * (1 - weights)).reshape(*z.shape[:2], -1)
            dt = dg @ up
            da = dt * (sigmoid_a + a * sigmoid_a * (1 - sigmoid_a))
            dz += ((da @ down) / STREAMS).reshape(z.shape)
            if d_post is not None:
                dp = d_post * post * (1 - post / 2)
                dz += ((dp @ self.inject.astype(np.float64)) / STREAMS).reshape(z.shape)
            normalized_grad = dz * norm
            dx = inv * normalized_grad - x * inv**3 * np.mean(
                normalized_grad * x, axis=-1, keepdims=True
            )
            _finite(dx)
        return dx


def expand_streams(hidden):
    """Broadcast[b,t,d] embedding to an independent FP32[b,t,4,d] copy."""
    hidden = _array(hidden, "hidden")
    require(
        hidden.ndim == 3
        and 0 < hidden.shape[-1] <= MAX_HIDDEN
        and 0 < hidden.shape[0] * hidden.shape[1] <= MAX_TOKENS,
        "Expected bounded[b,t,d] hidden state",
    )
    with np.errstate(over="ignore"):
        result = np.repeat(hidden.astype(np.float32)[..., None, :], STREAMS, axis=-2)
    _finite(result)
    return result


def residual_apply(streams, y, post):
    """Raw stream + per-stream gate*sublayer output, out of place; no norm/combine."""
    x = _streams(streams)
    require(
        x.dtype in (np.dtype("float32"), np.dtype("float64")),
        "Residual stack must be FP32 or FP64",
    )
    require(_array(y, "y").shape == x.shape[:2] + x.shape[-1:], "y shape mismatch")
    require(_array(post, "post").shape == x.shape[:-1], "post shape mismatch")
    require(post.dtype == x.dtype, "post and residual stack must share dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        result = x + post[..., None] * y.astype(x.dtype)[..., None, :]
    _finite(result)
    return result


def residual_input_adjoint(streams, y, post, d_output):
    """Float64 partial adjoints(dx,dy,dpost); caller composes dpost through mix."""
    _streams(streams, continuous=True)
    result = residual_apply(streams, y, post)
    require(
        y.dtype == np.float64
        and _array(d_output, "d_output").dtype == np.float64
        and d_output.shape == result.shape,
        "Residual adjoint requires matching float64 inputs",
    )
    dx = d_output.copy()
    dy = np.sum(d_output * post[..., None], axis=-2)
    dp = np.sum(d_output * y[..., None, :], axis=-1)
    _finite(dx, dy, dp)
    return dx, dy, dp


def streams_mean(streams):
    """Feature export mean only; this is NOT the learned final mixer."""
    streams = _streams(streams)
    require(
        streams.dtype in (np.dtype("float32"), np.dtype("float64")),
        "Stream stack must be FP32 or FP64",
    )
    result = np.mean(streams, axis=-2)
    _finite(result)
    return result
