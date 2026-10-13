"""Tiny CPU Qwen4Exp GatedDeltaNet recurrence and continuous input-adjoint oracle.

Inputs q/k/v are AFTER packed-channel causal depthwise conv + SiLU; that
convolution and its state are deliberately not implemented here. Qwen uses
split q|k|v, repeated contiguous key-head groups, scalar decay per VALUE head,
beta_scale=1, and sigmoid (not SiLU) output gating after weighted RMSNorm.
This is not fused Qwen3-Next layout or channelwise-decay KDA.

The default scan/state is FP32. Explicit precision='float64' is a mathematical
oracle, not serving parity. Analytical adjoints are continuous FP64 derivatives
of the recurrence wrt q/k/v/g/beta/initial_state, not gradients through casts,
gate helpers, convolution, output norm, projections or a complete model. All
dimensions/tape are fixture-bounded; no production16K backward is provided.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MAX_BATCH, MAX_STEPS, MAX_HEADS, MAX_HEAD_DIM = 2, 32, 8, 16
MAX_ARRAY_ELEMENTS = 1 << 18
MAX_STATE_TAPE_ELEMENTS = MAX_BATCH * (MAX_STEPS + 1) * MAX_HEADS * MAX_HEAD_DIM**2
SOURCE_ROOT = Path("/Users/jasontitus/experiments/exl3-qlora")
SOURCE_SHA256 = {
    "exllamav3/training/gdn.py": "e1de1f6112715a7868388972872a5f37b0d8198d98246527650ce791d9b33a57",
    "exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py": "a47e3ac514277ca77e73db51ee6306515dc5f1fbb24747dc7d7da0538823c99d",
    "exllamav3/modules/gated_delta_net.py": "0c7c2e53a20445991a6b750b9a17e58f4eed6a349b7876f9dfc148e47cd1f87e",
    "exllamav3/modules/gated_rmsnorm.py": "0823e3f45abfaf76707f213821980af08ab53cbb7c93c6a3c33a0f2f8548808f",
    "exllamav3/architecture/qwen4_exp.py": "f0781ec11aabc78694ed5084d24fb31093c0500a5c985a00276188c241c165bb",
    "exllamav3/training/native_llama.py": "12c99ff24bb54067a081ceb9e6b52154cf811afd231bd30a13847c70d5729f5e",
}
CONFIG_SHA256 = "f8adca807ce1deef83a781d28560fc38d2335e592939d9cf31930fa9a553eda1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_provenance(root=SOURCE_ROOT):
    actual = {
        name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
        for name in SOURCE_SHA256
    }
    require(actual == SOURCE_SHA256, "Semantic source identity changed")
    path = Path(__file__).resolve().parents[1] / "sources/config.json"
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, "Pinned Qwen config changed"
    )
    config = json.loads(raw)
    require(config.get("model_type") == "qwen4_exp", "Expected Qwen4Exp architecture")
    keys = (
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
        "output_gate_type",
        "rms_norm_eps",
    )
    metadata = {key: config["text_config"][key] for key in keys}
    require(
        metadata["output_gate_type"] == "sigmoid", "Expected Qwen sigmoid output gate"
    )
    return {
        "semantic_sources_sha256": actual,
        "config_sha256": CONFIG_SHA256,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "numpy_version": np.__version__,
        "target_config": metadata,
        "head_mapping": "value_head j -> key_head j//(num_value_heads/num_key_heads)",
        "convolution_implemented": False,
        "real_payload_verified": False,
        "gpu_parity_qualified": False,
        "production_backward_qualified": False,
        "scope": "Bounded recurrence only; FP32 sequential state and explicit continuousFP64 oracle.",
    }


def _array(value, name):
    require(
        isinstance(value, np.ndarray) and 0 < value.size <= MAX_ARRAY_ELEMENTS,
        f"{name} must be a nonempty bounded NumPy array",
    )
    require(
        value.dtype in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")),
        f"{name} must have float16/32/64 dtype",
    )
    require(np.isfinite(value).all(), f"{name} must be finite")
    return value


def _dtype(precision):
    require(
        precision in ("float32", "float64"),
        "precision must be float32 or float64 oracle",
    )
    return np.float32 if precision == "float32" else np.float64


def _epsilon(eps, dtype):
    require(type(eps) in (int, float), "eps must be positive finite")
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            converted = dtype(eps)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "eps must remain positive finite in computation precision"
        ) from exc
    require(
        np.isfinite(converted) and converted > 0,
        "eps must remain positive finite in computation precision",
    )
    return converted


def _finite(*values):
    require(
        all(np.isfinite(value).all() for value in values),
        "Reference arithmetic became nonfinite",
    )


def _sigmoid(x):
    result = np.empty_like(x)
    mask = x >= 0
    with np.errstate(under="ignore"):
        result[mask] = 1 / (1 + np.exp(-x[mask]))
        ex = np.exp(x[~mask])
        result[~mask] = ex / (1 + ex)
    return result


def _softplus(x):
    # torch.nn.functional.softplus's default beta1/threshold20.
    result = x.copy()
    mask = x <= 20
    with np.errstate(under="ignore"):
        result[mask] = np.log1p(np.exp(x[mask]))
    return result


def qwen_beta_g(b_raw, a_raw, a_log, dt_bias, *, precision="float32"):
    """Qwen's beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias).

    Default FP32 matches training helper precision; float64 is explicitly an
    extended mathematical oracle. No generic beta_scale override is inferred.
    """
    dtype = _dtype(precision)
    b_raw, a_raw, a_log, dt_bias = [
        _array(x, name)
        for x, name in (
            (b_raw, "b_raw"),
            (a_raw, "a_raw"),
            (a_log, "a_log"),
            (dt_bias, "dt_bias"),
        )
    ]
    require(
        b_raw.ndim == 3
        and a_raw.shape == b_raw.shape
        and 0 < b_raw.shape[0] <= MAX_BATCH
        and 0 < b_raw.shape[1] <= MAX_STEPS
        and 0 < b_raw.shape[2] <= MAX_HEADS,
        "Expected bounded[b,t,value_heads] gate inputs",
    )
    require(
        a_log.shape == dt_bias.shape == (b_raw.shape[2],),
        "Gate metadata must be[value_heads]",
    )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        b, a, al, dt = [x.astype(dtype) for x in (b_raw, a_raw, a_log, dt_bias)]
        _finite(b, a, al, dt)
        beta = _sigmoid(b)
        argument = a + dt
        exp_a = np.exp(al)
        _finite(argument, exp_a)
        g = -exp_a * _softplus(argument)
    _finite(beta, g)
    return beta, g


def split_postconv_qkv(packed, *, num_k_heads, num_v_heads, key_dim, value_dim):
    """Split ALREADY convolved/SiLU'd contiguous q|k|v channels; no conv/state."""
    packed = _array(packed, "postconv packed qkv")
    require(
        all(
            type(n) is int and 0 < n <= limit
            for n, limit in (
                (num_k_heads, MAX_HEADS),
                (num_v_heads, MAX_HEADS),
                (key_dim, MAX_HEAD_DIM),
                (value_dim, MAX_HEAD_DIM),
            )
        ),
        "Unsupported head geometry",
    )
    require(
        num_v_heads % num_k_heads == 0, "Value heads must be a multiple of key heads"
    )
    width = 2 * num_k_heads * key_dim + num_v_heads * value_dim
    require(
        packed.ndim == 3
        and packed.shape[-1] == width
        and 0 < packed.shape[0] <= MAX_BATCH
        and 0 < packed.shape[1] <= MAX_STEPS,
        "Expected bounded[b,t,2*nk*dk+nv*dv] postconv input",
    )
    q, k, v = np.split(
        packed, [num_k_heads * key_dim, 2 * num_k_heads * key_dim], axis=-1
    )
    prefix = packed.shape[:2]
    return (
        q.reshape(*prefix, num_k_heads, key_dim).copy(),
        k.reshape(*prefix, num_k_heads, key_dim).copy(),
        v.reshape(*prefix, num_v_heads, value_dim).copy(),
    )


def _validate(q, k, v, g, beta, initial_state, precision, eps):
    dtype = _dtype(precision)
    eps = _epsilon(eps, dtype)
    for name, x in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        _array(x, name)
    require(
        q.ndim == 4
        and k.shape == q.shape
        and v.ndim == 4
        and v.shape[:2] == q.shape[:2],
        "Expected q/k[b,t,nk,dk],v[b,t,nv,dv]",
    )
    b, t, nk, dk = q.shape
    nv, dv = v.shape[2:]
    require(
        0 < b <= MAX_BATCH
        and 0 < t <= MAX_STEPS
        and 0 < nk <= MAX_HEADS
        and 0 < nv <= MAX_HEADS
        and 0 < dk <= MAX_HEAD_DIM
        and 0 < dv <= MAX_HEAD_DIM,
        "Recurrence exceeds tiny fixture dimensions",
    )
    require(
        nv % nk == 0 and g.shape == beta.shape == (b, t, nv),
        "Invalid value-head grouping/gate shapes",
    )
    require(
        np.all(g <= 0) and np.all((beta >= 0) & (beta <= 1)),
        "Qwen requires g<=0 and beta in[0,1]",
    )
    require(
        b * (t + 1) * nv * dk * dv <= MAX_STATE_TAPE_ELEMENTS,
        "Retained-state fixture cap exceeded",
    )
    shape = (b, nv, dk, dv)
    if initial_state is None:
        state = np.zeros(shape, dtype=dtype)
    else:
        state = _array(initial_state, "initial_state")
        require(
            state.shape == shape and state.dtype == dtype,
            "State shape/dtype must match computation precision",
        )
        state = state.copy()
    with np.errstate(over="ignore"):
        inputs = [x.astype(dtype, copy=False) for x in (q, k, v, g, beta)]
    _finite(*inputs, state)
    return (*inputs, state, eps)


def _scan(q, k, v, g, beta, *, initial_state, precision, eps, record=False):
    q, k, v, g, beta, state, eps = _validate(
        q, k, v, g, beta, initial_state, precision, eps
    )
    dtype = _dtype(precision)
    b, t, nk, dk = q.shape
    nv, dv = v.shape[2:]
    group = nv // nk
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        qsq = np.sum(q * q, axis=-1, keepdims=True)
        ksq = np.sum(k * k, axis=-1, keepdims=True)
        _finite(qsq, ksq)
        q_denominator, k_denominator = qsq + eps, ksq + eps
        _finite(q_denominator, k_denominator)
        qi = 1 / np.sqrt(q_denominator)
        ki = 1 / np.sqrt(k_denominator)
        qn = np.repeat(q * qi, group, axis=2)
        kn = np.repeat(k * ki, group, axis=2)
        decay = np.exp(g)
        _finite(qn, kn, qi, ki, decay)
        output = np.empty((b, t, nv, dv), dtype=dtype)
        tape = []
        scale = dtype(dk**-0.5)
        for i in range(t):
            previous = state
            memory = np.sum(previous * kn[:, i, :, :, None], axis=-2)
            error = v[:, i] - memory * decay[:, i, :, None]
            delta = error * beta[:, i, :, None]
            state = (
                previous * decay[:, i, :, None, None]
                + kn[:, i, :, :, None] * delta[:, :, None, :]
            )
            output[:, i] = np.sum(state * qn[:, i, :, :, None], axis=-2) * scale
            _finite(state, output[:, i])
            if record:
                tape.append((previous, state, memory, error, delta))
    return output, state, (q, k, qn, kn, qi, ki, decay, beta, scale, tape)


def recurrent_forward(
    q, k, v, g, beta, *, initial_state=None, precision="float32", eps=1e-6
):
    """Return(outputs[b,t,nv,dv],final_state[b,nv,dk,dv]); inputs unchanged.

    Only the recurrence state is carried. Reusing this state does NOT supply the
    convolution history required to stream a complete Qwen GDN block.
    """
    out, state, _ = _scan(
        q, k, v, g, beta, initial_state=initial_state, precision=precision, eps=eps
    )
    return out, state


def recurrent_input_adjoint(
    q, k, v, g, beta, d_output, *, initial_state=None, d_final_state=None, eps=1e-6
):
    """Continuous FP64 VJP; returns dq,dk,dv,dg,dbeta,dinitial_state.

    Nonzero d_final_state composes across chunks without detaching the recurrent
    path. This stores a tiny per-step tape and is not a long-context backward.
    """
    for name, x in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("d_output", d_output),
    ):
        require(
            _array(x, name).dtype == np.float64,
            "Continuous recurrence adjoint requires float64 arrays",
        )
    output, state, cache = _scan(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        precision="float64",
        eps=eps,
        record=True,
    )
    require(d_output.shape == output.shape, "Output cotangent shape mismatch")
    if d_final_state is None:
        ds = np.zeros_like(state)
    else:
        require(
            _array(d_final_state, "d_final_state").shape == state.shape
            and d_final_state.dtype == np.float64,
            "Final state cotangent shape/dtype mismatch",
        )
        ds = d_final_state.copy()
    rawq, rawk, qn, kn, qi, ki, decay, beta, scale, tape = cache
    dq, dk, dv, dg, dbeta = (np.zeros_like(x) for x in (qn, kn, v, g, beta))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        for i in reversed(range(q.shape[1])):
            previous, current, memory, error, delta = tape[i]
            do = d_output[:, i] * scale
            dq[:, i] = np.sum(current * do[:, :, None, :], axis=-1)
            ds = ds + qn[:, i, :, :, None] * do[:, :, None, :]
            dk[:, i] = np.sum(ds * delta[:, :, None, :], axis=-1)
            dd = np.sum(ds * kn[:, i, :, :, None], axis=-2)
            dbeta[:, i] = np.sum(dd * error, axis=-1)
            de = dd * beta[:, i, :, None]
            dv[:, i] = de
            dm = -de * decay[:, i, :, None]
            dg[:, i] = (
                np.sum(ds * previous, axis=(-2, -1)) - np.sum(de * memory, axis=-1)
            ) * decay[:, i]
            dk[:, i] += np.sum(previous * dm[:, :, None, :], axis=-1)
            ds = (
                ds * decay[:, i, :, None, None]
                + kn[:, i, :, :, None] * dm[:, :, None, :]
            )
        b, t, nk, width = q.shape
        group = v.shape[2] // nk
        dq = dq.reshape(b, t, nk, group, width).sum(axis=3)
        dk = dk.reshape(b, t, nk, group, width).sum(axis=3)
        # Normalize first: inverse-norm cubing can underflow/overflow even
        # when the normalization Jacobian and its result are representable.
        unitq, unitk = rawq * qi, rawk * ki
        dq = qi * (dq - unitq * np.sum(dq * unitq, axis=-1, keepdims=True))
        dk = ki * (dk - unitk * np.sum(dk * unitk, axis=-1, keepdims=True))
        _finite(dq, dk, dv, dg, dbeta, ds)
    return {"q": dq, "k": dk, "v": dv, "g": dg, "beta": dbeta, "initial_state": ds}


def qwen_output_norm(core, gate, weight, *, eps=1e-6, precision="float32"):
    """Qwen weighted RMSNorm THEN sigmoid(z), over each value head's dv.

    No implicit1+weight bias, gate-first operation or SiLU alternative. FP32
    core/default yields FP32 output, as the training reference; explicit FP64
    is a continuous oracle extension, not its unconditional.float() execution.
    """
    dtype = _dtype(precision)
    eps = _epsilon(eps, dtype)
    core, gate, weight = (
        _array(x, name)
        for x, name in ((core, "core"), (gate, "gate"), (weight, "weight"))
    )
    require(
        core.ndim == 4
        and core.shape == gate.shape
        and weight.shape == (core.shape[-1],),
        "Expected core/gate[b,t,nv,dv] and weight[dv]",
    )
    require(
        0 < core.shape[0] <= MAX_BATCH
        and 0 < core.shape[1] <= MAX_STEPS
        and 0 < core.shape[2] <= MAX_HEADS
        and 0 < core.shape[3] <= MAX_HEAD_DIM,
        "Output norm exceeds tiny fixture dimensions",
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        x, z, w = (a.astype(dtype, copy=False) for a in (core, gate, weight))
        variance = np.mean(x * x, axis=-1, keepdims=True)
        _finite(x, z, w, variance)
        denominator = variance + eps
        _finite(denominator)
        result = x / np.sqrt(denominator) * w * _sigmoid(z)
    _finite(result)
    return result
