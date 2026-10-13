# Experimental activation arithmetic intervention — 2026-10-12

`activation_mode="bf16"` is an opt-in causal experiment, not an established fix
for the observed first-step regression. The default remains `"checkpoint"`.
All changes are in the staged tree; the running experiment's immutable runtime
was not edited. Static checks pass; nine authored GPU cases await the parent's
sequential run.

## Exact intervention

Both modes retain the same packed EXL3 expert weights, packed MXFP8 trunk
weights, frozen scales, tensor dtypes, adapter formulation, and accumulation
algorithms. The experimental mode bypasses explicit activation round trips at:

- Non-indexer `NativeLinear` inputs, including shared experts and the Engram
  projection. Native BF16 projections already have no such round trip.
- Attention window KV after normalization/RoPE.
- Compressed attention KV after normalization/RoPE.
- Attention output after inverse RoPE and before grouped output projections.

The **FrozenIndexer policy remains the checkpoint policy**: native projection
inputs still use FP8 round trips, and index query/key vectors still use FP4 E8M0.
The indexer rejects projections configured to bypass input quantization. Its
actual selections can still change because current backbone inputs can change.
Indexer and MoE selections are recomputed normally; no routes are replayed or
held fixed during this intervention.

BF16 casts, residual-addition rounding, low-bit frozen weights, discrete routes,
and other nonlinearities remain. The name does not mean every intermediate is
BF16: existing FP32 compressor, normalization, accumulation, and adapter
arithmetic is preserved. It is neither a quantization-free nor a mathematically
smooth model.

## State and comparison rules

The mode is stored explicitly on the model and relevant modules, and appears in
`model.adapter_config`. No ambient context or ContextVar is used; recomputation
and nested custom VJPs therefore read the same persistent mode as forward.
All three indexer native projections explicitly receive checkpoint mode, while
other native construction sites receive the selected model mode. Frozen-prefix
cache structure guards include these public module attributes.

The serialization/configuration owner is wiring absent legacy fields to
checkpoint mode and rejecting cross-mode adapter load/resume. This changes the
zero-adapter forward, so a BF16-mode evaluation must use a fresh matching-mode
base and the same data/target mask. Its scores cannot silently replace the
checkpoint-mode baseline or the original failed fit. An improvement would
support this intervention for that experiment, not certify kernel quality or
prove activation quantization was the only cause of the earlier regression.

## Pending numerical checks

`tests/test_activation_mode.py` covers packed-native FP32/BF16 forward and input
VJP against the explicit input policy; implicit versus explicit checkpoint
default; bypass without calling the native quantizer; mode persistence through
recomputation; preserved indexer input/query/key quantizers; nonzero attention
adapter gradients in both modes; rejection of bypassed indexer projections; and
changed forward values with unchanged stored weights. Checkpoint mode takes the
same quantizer calls and mathematical operations as before. Bitwise claims
remain conditional on those tests passing and actual-model evidence.

The separate rounding-observer script was reviewed without execution. Its
wrappers return the original outputs and require exact observed/unobserved and
original-experiment score/hash parity before reporting findings. Local `wo_b`
absorption statistics compare that adapter with its own base branch; they do
not isolate changes already caused by `wq_a` upstream. No wrapper/lazy-evaluation
blocker was found in that read-only review.
