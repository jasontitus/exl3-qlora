# EXL3 QLoRA on Apple silicon / MLX

This is an experimental port of the upstream CUDA training path, developed on
an M1 Ultra with 128 GiB RAM and MLX 0.32.3. It includes real EXL3 decoding,
first-order input gradients, streamed experts, and a short-context DeepSeek V4.1
text adapter trainer. The CUDA package remains separate.

**Verified:** text forward and backward through all 40 layers of the existing
local **3.0-bpw streaming repack**, with finite gradients for 160 attention-adapter
tensors. The successful four-token probe peaked at **8.47 GiB of MLX allocations**
and took 55.8 seconds including loading, a separate inference pass, and a training
forward/backward. This is not a throughput benchmark or total system RAM figure.

**Still unverified:** a full-model optimizer update, trained-adapter text quality,
and full-model logit parity against native DeepSeek inference. Earlier runs caused
substantial memory pressure; they exposed retained MLX graphs. The streaming path
now materializes activation boundaries, and a regression test checks that decoded
matrices do not accumulate. The final budget/per-slice hardening has passed small
tests but has not been rerun on the full model after the user's memory warning.

The requested **2.8-bpw** target remains
[jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw),
pinned at `915e1d0de8ee0a99944c5bc88ab802a639c63380`. Its individual matrices and a
complete real expert have been verified. Its original full checkpoint layout is
not yet connected to the text trainer. The available 3.0-bpw repack is a distinct
validation checkpoint, not an assertion of 2.8-bpw model support. Nothing was deleted.

## Setup and small checks

```bash
cd ~/experiments/exl3-qlora/apple
uv sync --locked
uv run pytest
uv run exl3-mlx train-smoke --output artifacts/small-adapter
uv run python scripts/validate_target.py
uv run python scripts/validate_expert.py
```

`validate_target.py` fetches four matrices (under 12 MiB of tensor payload), not
whole shards. Downloads require a pinned revision, enforce a byte limit, and save
hashes. `train-smoke` fits a synthetic residual; it is not language-model training.

## Experimental text trainer

The loader accepts the local `mtplx-expert-manifest-v2` repack with native
MXFP8/BF16 trunk shards, `experts.bin`, both Engram banks, tokenizer, and token map.
The existing checkpoint is at:

```text
~/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw
```

JSONL data uses one document per line, either `{"text":"..."}` or
`{"input_ids":[0,671,6102,294,8760]}`. Text gets BOS/EOS; explicit token IDs are used
as supplied. Loss covers every target token. Each document starts its own
attention/Engram history; long examples are explicitly prefix-truncated. Chat
message conversion, assistant-only masks, packing and multimodal inputs are absent.

For a future guarded smoke update (not executed after the memory warning):

```bash
uv run exl3-mlx-train \
  --model ~/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
  --data data.jsonl --output artifacts/text-smoke \
  --steps 1 --seq-length 4 --rank 2 --alpha 4 \
  --memory-gib 16 --allocator-cache-mib 64 --cache-mib 128
```

Defaults attach LoRA to attention `wq_a` and `wo_b` across all 40 layers. Use
`--adapter-layers 0 1` to select fewer adapter layers, or
`--expert-adapter-layers 0` to also adapt all routed experts in layer 0. This adds
adapter/optimizer memory. Frozen EXL3 experts still participate in input gradients
throughout the model. Selecting fewer adapters does not remove base layers.

Completed updates save adapter-only `step-000001/adapter.safetensors`, metadata,
and checkpoint identity hashes. `--resume-adapter` reloads weights but starts a
fresh optimizer. The named format `exl3-mlx-deepseek-short-text-v1` is private to
this package: no PEFT or external serving-engine compatibility is claimed.
Prediction reload parity is tested on a small model, not yet the real 40-layer model.

For read-only full-model forward/backward verification, explicitly run
`uv run python scripts/probe_model.py --model PATH`. This is a substantial workload;
it performs no optimizer update and downloads nothing.

## Memory behavior and current limits

- Routed weights are read per selected matrix in forward and backward through a
  bounded LRU. The entire 190-GiB expert bank is never mapped or copied into RAM.
  Native trunk weights (about 8 GiB for this text path) remain resident.
- Eager block recomputation rebuilds one block's gradient graph at a time, including
  shared compressed-KV gradients. Packed base parameters never enter the optimizer.
- Python MLX custom VJPs plus `eval`/`stop_gradient` alone retained dense decode
  graphs. The prototype copies **activation results**, preserving BF16 bytes,
  through host buffers to break those graphs. Both weight decode and matrix math
  run on Metal. These synchronizations/copies cost speed; a native primitive is the
  next optimization. The streamed path also materializes each column slice.
- The CLI defaults to a **16-GiB fail-fast budget**, 64-MiB MLX allocator cache,
  256-MiB packed-weight cache and a startup check for 32 GiB of system headroom
  beyond the budget. It checks MLX allocated/cache memory and process peak RSS.
  A 100-ms watchdog exits this training process on violation, preserving earlier
  step checkpoints. **This is not an OS-enforced hard quota**: MLX's own limit is
  advisory and one allocation can overshoot before the watchdog runs.
- The 512-token limit is architectural, not a promise that 512-token training fits
  the default memory budget or runs quickly. For sequences up to the indexer's
  512 candidates, all causally eligible compressed positions fit. Longer contexts
  require the sparse indexer/candidate-selection port and fail explicitly.
- Text-only, eager first-order reverse-mode training. No vision, MTP, generation
  cache, optimizer-state resume, higher derivatives or `mx.compile` support.

## EXL3 and architecture verification

The Metal decoder implements tail-biting ring windows, tensor-core fragment
permutation, all three procedural codebooks, integer 1–8 bpw, and mul1
1.5/2.5/3.5 bpw. The checkpoint's 2.8 bpw is an aggregate of matrix bitrates.
The base algebra is:

```text
W_effective = diag(suh) H128 W_inner H128 diag(svh)
y_base      = H128(H128(x * suh) @ W_inner) * svh
dx_base     = H128(H128(dy * svh) @ W_inner.T) * suh
y           = y_base + alpha/rank * (x @ A) @ B + bias
```

H128 is a normalized transform per 128-element block. Codebooks round to fp16;
transforms and accumulation use fp32, then cast to the activation dtype. This is
not bitwise parity with CUDA fp16/bf16 inference. NumPy is used for reference
checks, I/O, route planning and activation materialization, not weight computation.

Tests cover exhaustive codebooks, layout, gradients, frozen bases, selected-expert
routing, SwiGLU, hyper-connections, sparse-attention masks/sinks, quantization
rounding and its explicit identity straight-through gradient, shared-KV block
recomputation, checkpoint integrity, memory retention, and adapter serialization.
Source math follows the pinned official DeepSeek V4.1 inference implementation.

| Real 2.8-bpw fixture | Bits | Max forward error | Max dX error |
| --- | --- | --- | --- |
| `layers.0.attn.wkv` | 5 | 1.86e-6 | 1.37e-6 |
| `layers.0.ffn.experts.0.w1` | 2 | 1.78e-6 | 2.06e-6 |
| `layers.0.ffn.experts.0.w2` | 3 | 1.11e-6 | 2.75e-6 |
| Complete expert `w1/w2/w3` | 2 / 3 / 2 | 2.42e-6 | 4.31e-6 |

These compare against independent float64 NumPy math. The complete expert test
includes routing weights before the down projection and the exact SwiGLU clamps.
A separate 60-step matrix adapter fit decreased synthetic loss from 0.0696842 to
0.00278163, kept base weights unchanged, and reloaded exactly. Evidence lives in
[`reports/`](reports/), including the full-model probe and its measurement scope.

## Sources and license

Upstream baseline: `068f586c00724e3eabbf4165fb66c355da9af4df`. Format references:
[`tests/testlib/trellis.py`](../tests/testlib/trellis.py),
[`quant/codebook.cuh`](../exllamav3/exllamav3_ext/quant/codebook.cuh),
[`LinearEXL3`](../exllamav3/modules/quant/exl3.py), and
[`EXL3LoRAHadFunction`](../exllamav3/training/qlora_linear.py).
Architecture source:
[official DeepSeek V4.1 model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/5254448103b6254fc646dacbca1cbbf20e9216c1/inference/model.py).
The port and adapted math retain MIT licenses; see [`LICENSE`](LICENSE) and
[`configs/DEEPSEEK_LICENSE`](configs/DEEPSEEK_LICENSE). Model weights stay untracked.
