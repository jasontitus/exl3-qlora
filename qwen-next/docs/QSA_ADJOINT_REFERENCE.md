# Fixed-route selected-attention adjoint reference

This NumPy reference component supplies the analytic input adjoint of the approved selected-attention **mathematical** reference. It extends CPU architecture work toward M2b; it does not complete the A0 forward boundary or qualify a model, trainable suffix, native backward kernel or 16K training. Production dispatch, the existing project and the approved forward v2 remain unchanged. This component adds no model loader or training path.

`adjoint(q, k, v, gate, routes, positions, lengths, cotangent)` returns immutable FP64 `dq`, `dk`, `dv` and `dgate` arrays. All eight arguments must be plain NumPy ndarrays; floating arguments must already be FP64. Queries and gates are `[B,Q,Hq,D]`, K/V are `[B,T,Hkv,D]`, and the cotangent is the flattened forward shape `[B,Q,Hq*D]`. Noncontiguous and read-only plain arrays are supported. Subclasses, masked arrays and implicit dtype conversions are refused.

The boundary starts after Q/K/V projection, normalization, partial RoPE and indexer preparation. Routes are held fixed and must have the approved unique, causal, sequence-local contract. No derivative is returned for routes, indices, lengths, positions, hard selection or a change in selected membership. A caller may supply routes obtained from the separately reviewed selector replay; this does not establish authentication of a caller-supplied capture. No parameter, adapter, checkpoint, cache or upstream preprocessing derivative is implemented.

For one query head and selected token set, let `p = softmax(Q Kᵀ / sqrt(D))`, `c = p V`, `s = sigmoid(gate)` and `u` be the output cotangent. The reference computes:

```text
dc = u * s
dp[t] = dot(dc, V[t])
dz[t] = p[t] * (dp[t] - sum_j p[j] * dp[j])
dQ += sum_t dz[t] * K[t] / sqrt(D)
dK[t] += dz[t] * Q / sqrt(D)
dV[t] += p[t] * dc
dgate = u * c * sigmoid'(gate)
```

Query head `h` uses KV head `h // (Hq/Hkv)`. K/V adjoints accumulate over every selected query and every sharing query head, including repeated use of a token across queries. Routes are unique within each query. Unselected tokens, out-of-length positions and padded queries receive zero derivatives. A finite nonzero cotangent for a padded query is ignored because that forward output is constant zero.

These formulas are a curator-derived analytic adjoint, not an extracted upstream backward implementation. Like the forward reference, they represent real-valued attention evaluated in FP64 and do not differentiate finite machine rounding, native FP16 probability casts, native reduction order or score-selection discontinuities. The sigmoid slope uses `exp(-abs(g))/(1+exp(-abs(g)))²`; at `g=40` it remains small and positive even when a rounded forward sigmoid is 1. Finite differences around a rounding plateau therefore do not test this mathematical slope. Underflow may round tiny values to zero; nonfinite inputs, overflow or nonfinite intermediate/output arithmetic are rejected.

The explicit fixture limits are:

| Quantity | Maximum |
| --- | ---: |
| Aggregate bytes of all eight input arrays | 1 MiB |
| Sum of the four gradient array sizes | 1 MiB |
| Batch / queries / history | 2 / 8 / 64 |
| Query heads / KV heads / dimension | 24 / 2 / 256 |
| Padded route width | 64, multiple of 32 |
| Selected token count summed over queries × Hq × D | 262,144 |

The work count is a logical selected-component bound, not a FLOP count or timing estimate. It is checked before allocating gradients. Only one query/head's selected K/V and score intermediates are gathered at a time. Immutable output bytes coexist temporarily with the four gradient arrays; listed input/output sizes are not a process RSS or complete temporary-allocation bound. No measured memory or speed claim follows.

The implementation uses the private validation/sigmoid seams from the byte-identical committed forward v2. Those exact dependency identities are bound by the portable reference contract; updating them requires explicit review. The existing selector is used unchanged for composition tests only. The runtime component does not import the selector, Torch, Triton, MLX, upstream model code or device APIs.

Run the CPU mathematical and portable dependency-contract tests from the repository root with NumPy already installed:

```sh
python -m unittest discover -s qwen-next/tests -p 'test_qwen_qsa_attention_adjoint.py' -v
python -m unittest discover -s qwen-next/tests -p 'test_qsa_adjoint_source_contract.py' -v
```

The tests check all four gradients against central differences of an independent scalar `math.fsum` forward, a directional adjoint identity against the approved forward, fixed selector replay, exact 24:2:D256 shared-K/V accumulation, per-query/head contribution sums, cotangent linearity, padding, immutable outputs/read-only inputs, route permutation, overflow, malformed roles and pre-allocation resource refusal. These are CPU synthetic reference checks, not an actual model resume, training, native kernel or device qualification.

The portable contract is [`../reference-contracts/qsa-adjoint-cpu-v1.json`](../reference-contracts/qsa-adjoint-cpu-v1.json). It binds the existing forward source contract, exact dependency bytes, new adjoint and mathematical tests. The older forward-only contract remains an unchanged record of that component's scope.

A tiny uniform-attention example checks shared-K/V accumulation:

```python
import sys
import numpy as np
sys.path.insert(0, "qwen-next/tools")
from qwen_qsa_attention_adjoint import adjoint

q = np.zeros((1, 1, 4, 2), dtype=np.float64)
k = np.zeros((1, 8, 2, 2), dtype=np.float64)
v = np.full_like(k, 2.0)
gate = np.zeros_like(q)
routes = np.full((1, 1, 32), -1, dtype=np.int32)
routes[0, 0, :8] = np.arange(8)
result = adjoint(q, k, v, gate, routes,
                 np.array([[7]], dtype=np.int64),
                 np.array([8], dtype=np.int64),
                 np.ones((1, 1, 8), dtype=np.float64))
assert np.array_equal(result.dq.array(), np.zeros_like(q))
assert np.array_equal(result.dk.array(), np.zeros_like(k))
assert np.array_equal(result.dv.array(), np.full_like(v, 0.125))
assert np.array_equal(result.dgate.array(), np.full_like(gate, 0.5))
assert not result.dk.array().flags.writeable
```
