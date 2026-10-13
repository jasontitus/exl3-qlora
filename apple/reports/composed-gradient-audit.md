# Composed-gradient audit — 2026-10-12

The new independent smooth composition test passed: the parent ran
`PYTHONPATH=src python -m pytest -q tests/test_composed_gradient.py` and reported
**1 passed in 1.98 s**. This is additional evidence against a general missing or
reversed gradient path; it does not establish the cause of the real first-step
NLL regression, or certify the complete model's gradient.

## What was checked

The fixture composes two actual `TrainingBlock` objects with four residual
streams, Engram injection, delayed hyper-connection pre-mixes, Sinkhorn mixing,
causal attention and its denominator-only sink, a differentiable compressed KV
source reused by the next block, routed/shared experts, and chunked CE. It uses
19 positions, spanning the attention query-tile boundary, and nonzero LoRA B so
both A and B gradients participate. Attention, one routed expert, and the shared
expert carry adapters. The native/EXL3 projection kernels are represented by
small dense frozen matrices to isolate composition.

The direct path and block recomputation agree in loss and all gradient leaves.
The custom path is then checked against a separately written NumPy float64
forward of the complete composition. Each activation-quantization site records
its nominal input and rounded output once. NumPy replays it as
`rounded_nominal + (input - nominal_input)`, and holds nominal MoE selected IDs
fixed while differentiating routing weights normally. These anchors use the
engine's forward values intentionally: this test checks the engine's declared
identity surrogate, not the quantizer's format correctness or the derivative of
a quantized staircase. The remaining NumPy model arithmetic uses no MLX math,
model helpers, custom VJPs, or autodiff.

Independent central differences check every one of the 16 adapter arrays,
incoming stream activations, incoming pre-mix, and one unrelated joint parameter
direction: **19 directional checks**. Difference step is `2e-5`. Per-array/input
checks allow relative error `8e-4` and absolute error `1e-6`; the unrelated joint
direction allows `2e-3` and `2e-6`. All pass. Every adapter's tested gradient norm
is nonzero. The direct/recomputed comparison uses relative error `2e-4` and
absolute error `2e-7`.

## Read-only findings

No concrete end-to-end gradient defect was found in the inspected tree ordering,
parameter restoration, first-order materialization boundaries, Engram, or
hyper-connection ordering. Recompute explicitly receives trainable parameter
leaves and shared floating KV state; integer selection and bitmap leaves are
excluded from differentiation. The inspected materialization calls outside
custom VJP boundaries are frozen index-selection or frozen input-loading paths.

The frozen-prefix diagnostic's model isolation is mathematically appropriate:
only block 39's adapters change, so prefix h/pre/shared KV remain constant for
the full original sequence. Its last block has no compressor/indexer/Engram;
attention, MoE routing, activation quantization, and the complete answer loss are
executed afresh. It requires exact serialized-prefix and uncached/cached base
logit parity. Review feedback before execution was to bind the diagnostic script
hash into the cache/resume contract, distinguish zero-B baseline scores from
resumed initial scores, and reject outputs under the frozen model directory.

The standard SFT `adapter_layers` API/CLI/checkpoint constraints were coherent in
this read. A Studio parser edge was sent to its owner: `39,` must not silently
become `[39, 0]` through `Number('')`; discard or reject empty tokens explicitly.

## Interpretation and next discriminator

The test uses synthetic FP32 weights and short context. It does not test the real
BF16 rounding sequence, actual long-context top-k changes, a whole 40-layer loss
gradient, or independent native full-model parity. It also does not validate a
higher-order derivative; the engine's custom boundaries are first order.

The next useful evidence is the real final-block frozen-prefix gradient and
actual-forward fit, followed by a controlled smooth/normal replay if needed.
Disabling explicit FP8/FP4 activation round trips is a credible separate compute
experiment, but changes the zero-adapter forward. Such a mode requires its own
baseline and strict train/eval/checkpoint semantic identity. It cannot be used
to relabel the original failed fit as successful or claim native inference
parity. BF16 rounding and discrete routing would still remain.
