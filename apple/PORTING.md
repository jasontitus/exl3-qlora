# Full-model port plan

## Current boundary

The matrix decode / linear / first-order activation VJP / LoRA foundation works on
this M1 Ultra and on actual tensors from the chosen checkpoint. Full-model SFT,
generation, routed expert execution, data loading, and PEFT export are not
implemented. `inspect` reports `full_model_training_supported: false` for every
architecture. Never infer support from the ability to load a matrix.

The upstream PyTorch training layer's useful seam is its activation-transform
linear (`EXL3LoRAHadFunction`), not a differentiable integer quantizer. Base
trellis/scales stay frozen. Gradients through the base input must still flow to
earlier adapters. The new implementation follows that contract.

## 1. Memory and matrix kernels

Measure forward/backward peak memory at realistic token counts and 5120/2304
expert dimensions. Replace decode-to-fp32-plus-GEMM with tiled/fused Metal kernels
when measurements justify it. Keep the independent bitstream oracle and adjoint
tests as correctness gates; bf16/fp16 speedups must retain fp32 dX accumulation.
Check MLX scheduling explicitly rather than assuming chunk sizes cap live memory.

Build a tensor provider with bounded mapped-file residency and eviction, reusable
in both forward and recomputed backward. A Python closure retaining every MLX
weight array is not a streaming implementation. Benchmarks must include cold and
warm SSD reads and page-cache pressure. Do not rely on transparent OS swapping.

## 2. One complete DeepSeek expert and routed MoE block

Port the exact SwiGLU clamp and three-projection expert algebra. Compare a real
expert's output and dX against an independent reference. Then implement grouped
token dispatch, selection and routing-weight math, the shared expert, and
per-expert bitrates. Default to frozen routers and trunk adapters; trainable
routed-expert adapters are a separate configuration and memory budget.

DeepSeek's `sqrtsoftplus` scores, selection rules, renormalization and scale must
match its config. Tests must cover duplicate expert assignments, unused experts,
token order restoration and gradients through selected expert outputs. A dense
Qwen-style softmax router is not an equivalent substitute.

## 3. DeepSeek V4.1 text decoder

Implement against the pinned config and a trusted V4.1 reference:

- Four fp32 residual streams and the model's Sinkhorn hyper-connections.
- Sliding attention and compressed attention with the V4.1 compression schedule,
  shared KV/index source layers and candidate selection.
- RMSNorm with the recorded epsilon, rotary positions/YaRN, grouped output
  projection slices, routing, and the final head.
- The two Engram layers, exact n-gram hashing and projections; frozen table rows
  loaded on demand from their separate checkpoint data.

Do not treat the upstream V4 implementation as V4.1 or the `qwen4_exp` path as
DeepSeek. Existing local `mlx-stream` / `mlx-serve` inference work may provide
useful layout/kernels/reference outputs; none is imported or modified by this
initial port. An inference implementation alone does not supply backward rules.

Exclude vision and MTP explicitly in the first text-only trainer. Require
layerwise activations and logits to match a trusted reference before adding a
training command. Compare short prompts plus boundary cases that actually invoke
compression, shared source layers and Engram. No arbitrary context limit should
silently bypass architecture features.

## 4. Training and output compatibility

Add activation checkpointing, chunked cross-entropy, optimizer state budgeting,
dataset/tokenizer handling, assistant-token loss masks, evaluation and resume.
Quantify SSD traffic for forward + backward recomputation and expert-routing
coverage as batch/sequence size grows. Active parameters per token alone do not
predict the storage bandwidth or optimizer/activation working set.

Implement a named mapping between adapter tensor names and the eventual serving
engine. Test adapter-disable parity, save/reload, and inference application of the
same delta. Current primitive adapter artifacts intentionally make no PEFT/native
compatibility claim. Port upstream preference objectives only after SFT parity.

## Target evidence

Pinned checkpoint: `jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw` at
`915e1d0de8ee0a99944c5bc88ab802a639c63380`. Headers from layer 0 contain 1,167 EXL3
matrices: 906 at 2 bits, 240 at 3, 8 at 4, and 13 at 5. The checkpoint config
describes 40 decoder layers, hidden width 5,120, 384 experts per layer and six
selected experts per token. We downloaded only three matrices (~8.63 MiB total
payload), not the 203.8 GB model. Detailed sizes and hashes are in `reports/`.

Qwen3.8-Flash-Next remains a later target. Its upstream training support involves
gated residual hyper-connections, Gated DeltaNet, QSA and PLE; those components
need their own MLX port and parity gates. They are not implemented by this work.
