# Qwen3.8-Flash-Next: low-memory fine-tuning with 16K-plus training items

Research and source review: **2026-10-12**. Status: **proposed design, not an implemented or hardware-qualified capability**.

## Recommendation and scope

Adding this model is feasible, and much of our streaming and adapter infrastructure is reusable. **Correct 16K training is a substantial extension:** it needs the Qwen architecture, sparse-attention forward/backward, compatible checkpoint decoding, and a bounded recurrent backward when adapters extend into earlier layers. Raising a sequence-length setting will not provide those capabilities.

Start with **one complete 16,384-token text item per microbatch, frozen base weights, and LoRA on final-layer QSA attention**. Layer 47 is a full-attention layer. Freezing layers 0–46 avoids recurrent backward for this first stage while retaining the complete item's context. Then expand adapter coverage after implementing and verifying bounded GatedDeltaNet backward. Qualify 32,768 tokens separately.

“Axel Lattle” is interpreted as **Axolotl**, and “Qwen 3.8 Next” as **Qwen3.8-Flash-Next**, the approximately 125B main model plus 51B n-gram table. This is different from Qwen3-Next-80B and Qwen3.8-27B. “16K-plus training items” means **tokens per serialized item**, including prompt, rationale, code, answer, and delimiters. Dataset size is a separate dimension. The proposal is adapter fine-tuning; full-parameter training of the entire model is outside the local-memory target. Vision and MTP training are excluded initially.

The preferred first checkpoint route is **Sushi's EXL3 experts plus MLX affine trunk and n-gram weights**. It preserves our existing EXL3 training investment while adding a comparatively natural Apple quantization backend. A GGUF Q2_0 route is also credible, with relevant public training prior art, but requires another loader and differentiable packed-matrix implementation. Neither route currently works in our Apple trainer.

Earlier conversational estimates of “a few hours” and “24–48 GiB at 4K” were hypotheses, not measurements. They should not be treated as a 16K engineering estimate or fit guarantee. This document replaces that confidence with explicit stages, buffer arithmetic, and acceptance gates.

## Evidence boundaries

The local review used HEAD `b766887941695bc0c28e25b59ca24db63916e8b8` **plus the actively changing working tree**, including uncommitted Apple changes and reports. HEAD alone does not reproduce this review. Core source fingerprints are recorded below. Reads were not an atomic snapshot.

This task inspected source, existing receipts, and public primary sources. It did not rerun training or GPU tests, download model weights, upgrade dependencies, or modify the training implementation. Existing reports are historical evidence; external performance numbers are author-reported on their hardware. The adversarial review is a separate critical pass over this design by its author, **not an independent reviewer or a hardware certification**.

### What our current training has actually demonstrated

The latest inspected [Metal tuning receipt](../apple/reports/metal-tuning-result.md) records **three distinct complete historical edits**, serialized at **3,606, 5,809, and 5,822 tokens**. There were 12 updates, four exposures per edit, and 11,568 supervised target presentations. Final-block attention LoRA had 78,848 trainable parameters. Fresh-reload token-weighted NLL improved 8.23%; the reported MLX allocation peak was 53.08 GiB.

This answers the earlier example-count question for that completed fit: **three independent examples**, not twelve. The receipt also limits the quality claim: 92.1% of the NLL-sum improvement came from comments and diff headers; added-code teacher-forced correctness rose by one token, and EOS correctness regressed. There is no demonstrated held-out kernel improvement from that adapter.

The current [context constant](../apple/src/exl3_mlx/context.py) permits up to 32,768 tokens. This supersedes old documentation describing a 512-token ceiling, but **an accepted maximum is not a measured training length or a memory guarantee**. The current architecture remains DeepSeek-specific. The three-example result cannot predict throughput for a large corpus: first encounters took hundreds of seconds, while repeated examples benefited from frozen-prefix caching.

## What Axolotl and others have implemented

### Axolotl: real support, with important hardware and context qualifications

Axolotl merged [PR #3965](https://github.com/axolotl-ai-cloud/axolotl/pull/3965) on September 1, 2026. Its author reports end-to-end testing, profiling, and forward/loss checks on one B300; the PR explicitly did not establish multi-GPU testing at creation.

The pinned [model guide](https://github.com/axolotl-ai-cloud/axolotl/blob/e767fe077c02a087bdf4c7da0cc2566800cddad3/examples/qwen3.8-flash-next/README.md) reports approximately **120 GiB GPU memory with n-gram CPU offload, versus 216 GiB without**, for text QLoRA. It lists SDPA support and limitations around LoRA kernels, Liger, and variable-length SDPA. Its full-fine-tuning path is untested. Moving the table to CPU saves GPU memory on that discrete-GPU system; it does not eliminate the table's host-RAM cost.

The inspected [QLoRA configuration](https://github.com/axolotl-ai-cloud/axolotl/blob/e767fe077c02a087bdf4c7da0cc2566800cddad3/examples/qwen3.8-flash-next/qlora.yaml) uses **4K sequences**, NF4, rank 16, gradient checkpointing, chunked cross entropy, explicit attention/shared-expert targets, and routed-expert parameter adapters. `quantize_moe_experts: true` is essential: ordinary bitsandbytes linear replacement misses fused expert parameters. Its CPU-offload comment calls for roughly 100 GB free host RAM. The separate [NVFP4 recipe](https://github.com/axolotl-ai-cloud/axolotl/blob/e767fe077c02a087bdf4c7da0cc2566800cddad3/examples/qwen3.8-flash-next/nvfp4-lora.yaml) uses 2K sequences and a different expert/kernel stack.

Their [Qwen patches](https://github.com/axolotl-ai-cloud/axolotl/blob/e767fe077c02a087bdf4c7da0cc2566800cddad3/src/axolotl/monkeypatch/models/qwen4_exp/modeling.py) are useful semantic references for vectorized QSA selection and document-boundary resets. These implementations and tests are more useful to us than copying configuration values. The inspected recipes do **not** demonstrate 16K training on a Mac.

### MS-SWIFT: measured distributed training through 8K

The pinned [Qwen best-practice guide](https://github.com/modelscope/ms-swift/blob/e132612cb3be4d1c92e00fd7a1be878a9b44b082/docs/source_en/BestPractices/Qwen3_8-Flash-Next-Best-Practice.md) describes Megatron LoRA with eight GPUs, tensor/expert/pipeline parallelism, rank 8, and full recomputation. Its synthetic equal-length measurements include:

| Length | Configuration | Reported peak per GPU |
| --- | --- | ---: |
| 4K | Without PLE offload | 77.8 GiB |
| 4K | With PLE offload | 63.6 GiB |
| 8K | Without PLE offload | 93.3 GiB |
| 8K | With PLE offload | 77.7 GiB |
| 8K | With PLE offload and padding-free execution | 72.5 GiB |

This establishes an implemented longer-context distributed path. Those are per-GPU measurements on a different precision, adapter, and sharding setup; they are not total-system memory or predictions for our streaming Mac trainer. The inspected table stops at 8K.

### NVIDIA NeMo Automodel: useful sparse-attention training semantics

The pinned [fine-tuning configuration](https://github.com/NVIDIA-NeMo/Automodel/blob/6c70585d941e78b4fc642a0c3c17caa91b0f48fa/examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml) targets full language-model training with FSDP2 and expert parallelism across 64 GPUs. It is architectural prior art, not a local low-memory recipe. The separate [8K benchmark configuration](https://github.com/NVIDIA-NeMo/Automodel/blob/6c70585d941e78b4fc642a0c3c17caa91b0f48fa/examples/llm_benchmark/qwen/qwen3_8_flash_next_fa4_ep64.yaml) uses random weights and mock tokens; that benchmark must not be described as a real-checkpoint fine-tune.

NeMo's [QSA implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/6c70585d941e78b4fc642a0c3c17caa91b0f48fa/nemo_automodel/components/models/qwen3_8_flash_next/qsa.py) exposes selected causal token IDs, invalid-slot handling, query chunking, and route replay during checkpoint recomputation. With a 2,048-token budget and compression ratio four, the selection width is at most **2,051 tokens per query**, including the incomplete causal tail. The indexer is a hard selection mechanism, not a differentiable dense-attention substitute. Its [FA4 implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/6c70585d941e78b4fc642a0c3c17caa91b0f48fa/nemo_automodel/components/models/qwen3_8_flash_next/fa4_qsa.py) is NVIDIA-specific; its preprocessing is also a reminder to audit auxiliary masks rather than assuming sparse attention automatically has bounded memory.

### Community GGUF training: the closest alternative-quant precedent

`woct0rdho`'s [training recipe](https://github.com/woct0rdho/transformers5-qwen3.5-recipe/tree/be48b82232b71cd679e687839cf38cfa3fe52f9e) trains Qwen3.8-Flash-Next with a **GSQ-RCO Q2_0 GGUF base**, PEFT adapters, packed GGML matrix operations, chunked loss, recurrent backward, and disk-resident PLE. The associated [torch-ggml-ops](https://github.com/woct0rdho/torch-ggml-ops) provides autograd bindings for AMD RDNA3/3.5.

Its pinned [training report/design](https://github.com/woct0rdho/transformers5-qwen3.5-recipe/blob/be48b82232b71cd679e687839cf38cfa3fe52f9e/docs/plan_qwen4_exp.md) reports batch one, 2,048 tokens, rank-four BF16 adapters, and a complete update around **9.31 seconds** on Strix Halo. Disk PLE reduced the reported allocator peak from 67.51 GiB to **40.69 GiB**. These are author-reported update audits, not an independently reproduced training campaign or held-out quality result.

Crucially, the [QSA kernel](https://github.com/woct0rdho/transformers5-qwen3.5-recipe/blob/be48b82232b71cd679e687839cf38cfa3fe52f9e/qwen4_exp_qsa_attention.py) explicitly serves dense 2K attention, where selection is exhaustive, and the [full-step audit](https://github.com/woct0rdho/transformers5-qwen3.5-recipe/blob/be48b82232b71cd679e687839cf38cfa3fe52f9e/audit_qwen4_exp_training_step.py) rejects its fixed dataset lengths above 2,048. This is unusually relevant proof that GGUF can support adapter training, **but it does not supply our missing 16K sparse backward**. Its ROCm/Triton/Gluon kernels also require an Apple implementation.

### Coverage conclusion

The inspected primary sources establish exact-model training support in several frameworks, with concrete short-context and 8K evidence. They do not establish **16K-plus low-memory training on Apple silicon**. This is a bounded research conclusion about the sources inspected, not a claim that no unpublished implementation exists. Generic Qwen3/Qwen3.5 support, 27B tutorials, GGUF inference releases, and inference throughput figures do not qualify this exact architecture for 16K training.

## Repository assessment

| Component | Present and reusable | Required change |
| --- | --- | --- |
| Model dispatch | [workflow.py](../apple/src/exl3_mlx/workflow.py) and [sft.py](../apple/src/exl3_mlx/sft.py) provide preparation, fitting, export, and receipts | Add model-specific capabilities, construction, checkpoint identity, and adapter schema; current dispatch and validation require DeepSeek |
| Frozen linear/adapters | [streaming.py](../apple/src/exl3_mlx/streaming.py), [linear.py](../apple/src/exl3_mlx/linear.py), [native.py](../apple/src/exl3_mlx/native.py) already propagate input gradients through frozen EXL3 weights | Generalize packed formats and metadata; current native packed linear expects MXFP8, not affine |
| Storage | [storage.py](../apple/src/exl3_mlx/storage.py) reads bounded tensor/row ranges | Add Sushi expert-axis views, unsigned trellis payloads, rate groups, index ownership, affine row decoding, and PLE layout |
| Architecture | CUDA/Torch [gdn.py](../exllamav3/training/gdn.py), [hyperconnections.py](../exllamav3/training/hyperconnections.py), [ple.py](../exllamav3/training/ple.py) already describe Qwen math | Port to MLX/Metal; DeepSeek gates, hyper-connections, Engram, and attention are different |
| QSA references | [QSA indexer](../exllamav3/modules/qsa_indexer.py) and [sparse prefill parity tests](../tests/parity/test_qwen4_qsa_prefill.py) cover inference semantics | Build differentiable bounded sparse attention; these inference paths are not training backward |
| Existing Qwen training | [native_llama.py](../exllamav3/training/native_llama.py) supports substantial Qwen structure | It refuses sequences above its QSA dense threshold; it cannot be used as a 16K forward oracle unchanged |
| Recompute/prefix cache | [recompute.py](../apple/src/exl3_mlx/recompute.py), [prefix_cache.py](../apple/src/exl3_mlx/prefix_cache.py), [prefix_disk.py](../apple/src/exl3_mlx/prefix_disk.py) provide materialized boundaries and bounded caching | Introduce Qwen identity/state semantics and recurrent chunk adjoints; layer recomputation alone is insufficient |
| Supervised loss | [training.py](../apple/src/exl3_mlx/training.py) selects answer rows and chunks frozen-head cross entropy | Reuse exact loss normalization and its custom VJP with Qwen's head; audit large evaluation buffers too |
| Resources | [resources.py](../apple/src/exl3_mlx/resources.py) has process/cache/reserve profiles | Include all host, device, staging, page-cache, and disk-cache pressure in qualification |

The pinned local environment is MLX 0.32.3 / MLX-LM 0.32.0. Its Qwen3-Next model is a useful reference, but is a different architecture. Its training recurrence falls back to sequential operations; an inference Metal kernel does not provide a proven long-context backward. Dependency upgrades must be tested in an isolated environment because the active installation may include local kernel work.

## Other quants and the Sushi hybrid

### Why low-memory adapter training is format-independent in principle

For a frozen decoded weight `Wq`, the effective linear is:

```text
y = x Wqᵀ + (alpha / rank) x A B
dx = dy Wq + adapter contribution
```

Training needs the frozen base's forward multiply and **transpose multiply for input gradients**, plus gradients for small floating-point adapter matrices. It does not require differentiation through the quantizer. A backend may reconstruct small tiles on demand or implement packed matmul directly. It must avoid keeping a complete decoded model alive across backward.

This is already the principle behind our streamed EXL3 custom VJP. Merely freezing a llama.cpp/Sushi inference module does not make it trainable: a compatible input-gradient path and framework integration still have to exist.

| Frozen format | Apple implementation prospect | Qualification needed |
| --- | --- | --- |
| Current EXL3 | Existing bounded forward/transpose reconstruction is the starting point | Decoder/layout parity for each new pack; long-context architecture remains separate |
| MLX affine 2/3/4/5/6/8-bit | Closest additional backend; uses familiar MLX packed matmul and per-group scales/biases | Correct per-tensor bits/group sizes, transpose VJP, row gathers, and zero-adapter parity |
| GGUF Q2_0 and simpler block integer formats | Credible, supported by the public training precedent above | GGUF tensor metadata, block decoding, packed/tiled forward and backward on Metal, and serving/export agreement |
| GGUF IQ formats | Feasible with the same mathematical contract | Format-specific codebook/sign/scaling decode and profiling; generic IQ inference support is not a training implementation |
| bitsandbytes NF4 / NVIDIA NVFP4 | Representable as frozen quantized weights | Their inspected training stacks depend on other GPU backends; equivalent Apple kernels and numerical semantics would be new work |

GSQ/RCO identify how a quantization was produced; **the actual tensor encoding determines our implementation**. A GSQ-RCO Q2_0 model is different from a mixed IQ model. Do not infer IQ status or speed from a repository title, and do not assume all IQ quants used GSQ/RCO.

### What Sushi actually uses

The inspected [3-bpw Sushi model](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-3bpw/tree/76bd8f55df2bd4ab6297101c57bf86d450ac0aa3) combines **EXL3 routed experts, mostly 8-bit MLX affine trunk weights, and a 4-bit affine n-gram table**. The model card reports 49.33 GiB resident weights and roughly 29.8 GiB of n-gram table on disk. These are checkpoint/inference figures, not our training peak. Excluding unused MTP/vision tensors may change our loaded footprint.

The actual pinned [config](https://huggingface.co/beamster/Qwen3.8-Flash-Next-Sushi-3bpw/blob/76bd8f55df2bd4ab6297101c57bf86d450ac0aa3/config.json) specifies **MCG codebook, window 14, nominal expert rate three**, with per-weight affine settings. Some pack examples illustrate window 15; those examples must not override checkpoint metadata.

The [Sushi pack specification](https://github.com/beamivalice/sushi/blob/bd4b3954f5a42a87effcdf8a68ea91a4b6a1a1b9/docs/pack-format.md) defines expert-axis trellis tensors, optional contiguous rate groups, Hadamard transforms, and scales. Codebook scale can already be folded into the input scale. Its grouped format allows rates our current Apple manifest does not expose, and its current SSD streaming rejects ragged banks. We should implement explicit group-to-expert mapping and reject unsupported combinations rather than falling back.

Our [trellis implementation](../apple/src/exl3_mlx/trellis.py) assumes a 16-bit codeword window; the active [expert-bank loader](../apple/src/exl3_mlx/storage.py) validates a specific MUL1 repack. **The Sushi window-14 MCG pack is not a drop-in replacement.** Required work includes window masking, unsigned payload handling, packed geometry, normalization-folding checks, source-converter decode fixtures, and forward/adjoint parity. Our transpose multiply must reconstruct the same matrix used by forward; it cannot apply a forward-only packing shortcut to a transposed code stream.

The implementation order should be: affine linear and row reader; Sushi metadata/expert views; MCG/window decoder qualification; architecture integration. Keep adapters separate. Serving requires an adapter application path in Sushi or another exact-base engine; inference support by itself does not establish that integration. Merge/requantize must be separately evaluated because rounding can remove small learned updates.

## Architecture for complete 16K-plus items

The pinned [official model configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) has 48 layers, hidden size 2,560, four residual streams, 36 GatedDeltaNet layers, 12 QSA layers, 512 routed experts with top-10 selection, and a vocabulary of 248,320. GDN uses 16 key heads and 48 value heads of dimension 128. QSA uses 24 query heads and two KV heads of dimension 256. Preserve these exact distinctions.

### Stage A: final-QSA adapters, frozen prefix

1. Port exact text forward: Qwen gated residual streams and mixer, sigmoid-gated GDN, QSA indexer/attention, MoE/shared expert, and PLE injection. Reuse the Torch references as math oracles; do not substitute DeepSeek operations with similar names.
2. Freeze layers 0–46, PLE, router/indexer weights, embeddings, head, and all expert weights. Place modest-rank adapters only on selected layer-47 attention projections.
3. Run the complete frozen prefix without retaining its backward graph. Give this execution an explicit frozen-forward mode rather than relying on the whole model's training flag to choose efficient GDN kernels. Verify that cached and uncached paths use the same arithmetic. Cache its output for the **entire exact item** when reuse justifies the disk/RAM cost. The boundary contains four residual streams and must preserve the defined precision.
4. Train layer 47 with bounded QSA backward, residual/mixer gradients, frozen-expert input gradients, and chunked head loss. A frozen router still needs the selected routing-weight derivatives through its input when that input is trainable.
5. Compute one loss over all supervised positions and perform one optimizer update per item, or accumulate globally token-normalized gradients across complete items. Internal tiles are implementation details, not independently truncated training examples.

All 12 QSA layers still need correct 16K **forward** selection on a cache miss. Only final-layer QSA needs backward initially. The frozen prefix contains all GDN layers, so Stage A does not need recurrent backward. It therefore has a materially lower implementation and activation-memory burden than broad adapter placement.

This is limited adapter capacity. It may fit the examples without improving new coding tasks; evaluation must decide whether it is sufficient. Cache hits accelerate repeats, but every new unique item still incurs a full frozen-prefix pass. This cache is a full-item hidden-state cache, not an autoregressive KV cache shared arbitrarily between documents.

### Stage B: wider attention coverage with exact recurrent gradients

Port a training-capable GDN recurrence with **internal chunk checkpointing**, in addition to layer checkpointing. Store a bounded number of FP32 state boundaries, recompute each chunk, and carry the state adjoint backwards between chunks. Include the causal-convolution history and the model's head-ordering convention.

A token-by-token autodiff reference is suitable for tiny numerical tests. It is unsuitable for 16K production if it retains a recurrent state for every token. Detaching state between chunks implements truncated backpropagation and changes the learning problem; it must not be marketed as full-item training.

Use this backwards schedule:

```text
recompute one decoder block
  regenerate that block's chunk boundary states
  traverse chunks in reverse, carrying d(state) to the previous chunk
  release each chunk's intermediates immediately
release the block's temporary states before moving to the preceding block
```

Avoid retaining all 36 layers' chunk-state histories simultaneously. Layer-boundary storage also needs its own budget: wider adapter coverage retains substantially more boundaries than Stage A. Add grouped checkpointing or bounded disk boundary offload if required for 32K.

### QSA forward and backward

Build pooled index keys once, score **query tiles** against causal completed blocks, select the highest-scoring blocks, append the incomplete causal tail, and mask invalid tokens. Expand and gather selected K/V for only the current query tile. Preserve normalization, partial RoPE, GQA grouping, output gating, causality, padding, and token-set semantics.

Use online/streaming softmax or small query tiles to avoid full score arrays. **Tiling a normal autodiff loop is insufficient if it saves every tile's gathers and probabilities.** Put the entire QSA operation behind a custom VJP: retain the necessary Q/K/V, compact routes and softmax statistics, then regenerate and consume one tile at a time during backward. Accumulate into bounded Q/K/V gradient buffers, with an explicit shared-K/V reduction strategy. Gradient correctness and repeatability matter more initially than fused-kernel speed. A portable tiled reference can precede optimized Metal kernels.

Capture the actual forward selection and replay it during checkpoint recomputation. Frozen indexer weights do not make its routes constant: its input changes with upstream adapters. Hard top-k selection has no conventional gradient through token IDs; keeping indexer weights frozen follows that contract. Replaying routes avoids changing the selected computation at near ties during backward. Test binding sparse selection with a reduced-budget tiny oracle and with full-budget sequences crossing 2,051/2,052 tokens.

The compressed indexer still does work proportional to `T² / compression_ratio`; tiling bounds memory, not that compute cost. Main attention has a fixed selection budget, but **gathering every query's selected K/V at once can still exhaust RAM**. Sparse naming alone is insufficient.

### Embeddings, PLE, loss, and data

Use bounded, deduplicated row reads for token embeddings and the frozen n-gram table, with asynchronous prefetch where measurement justifies it. Preserve PLE hash arithmetic, EOS history, gating, and dilated-convolution semantics. Its 16 hashed heads can request up to 262,144 row references in a 16K item before deduplication. A full decoded table or unlimited warm page cache must not become a hidden dependency.

Reuse the existing chunked frozen-head cross-entropy custom VJP. Mask prompt tokens only for the loss; their hidden states still participate in the full causal forward. Avoid `[T, vocabulary]` allocations in training, scoring, diagnostics, and export probes. Aggregate by the total supervised-token count, not an unweighted average of differently sized chunks.

Initially use one unpacked complete item. Sample packing requires resets for GDN state, both convolution histories, PLE hashing, positions, and QSA visibility. It should remain disabled until document-boundary parity is established.

Preparation must report actual post-template lengths, supervised-token counts, truncation/rejection counts, and p50/p90/p99 lengths. Retain rationale plus the complete change in one item. Fail explicitly if an item exceeds the selected capacity; never silently truncate to claim 16K support. Deduplicate and split by change lineage, repository, or time to limit leakage. If the corpus includes items between 16K and 32K, select the next supported capacity without splitting their reasoning/change relationship.

## Memory arithmetic and realistic limits

These are **buffer sizes, not measured peaks**. Assumptions: batch one, FP32 residual/state and score buffers, BF16 gathered K/V, no retained dense base model. `T` is the backbone's input length; ordinary next-token shifting makes it one token shorter than the serialized item. Qualification receipts should record both lengths. GiB means `2³⁰` bytes. Dtypes are explicit because changing them changes numerical behavior.

| Buffer / naive implementation | Formula in bytes | 16,384 tokens | 32,768 tokens |
| --- | --- | ---: | ---: |
| One four-stream residual boundary, FP32 | `T × 4 × 2560 × 4` | 0.625 GiB | 1.25 GiB |
| 48 such layer boundaries | `48 × boundary` | 30 GiB | 60 GiB |
| Dense attention scores, one QSA layer | `24 × T² × 4` | 24 GiB | 96 GiB |
| All logits, FP32 | `T × 248320 × 4` | 15.156 GiB | 30.313 GiB |
| GDN per-token state history, one layer | `T × 48 × 128 × 128 × 4` | 48 GiB | 96 GiB |
| Expanded selected K and V for all queries | `2 × T × 2051 × 2 × 256 × 2` | 64.094 GiB | 128.188 GiB |
| Full token route IDs, one QSA layer | `T × 2051 × 4` | 128.188 MiB | 256.375 MiB |
| Unreduced indexer scores, four heads | `4 × T × (T/4) × 4` | 1 GiB | 4 GiB |

These buffers belong to different strategies and lifetimes and must **not simply be added** as a peak estimate. They identify implementations to avoid. A boolean `[T,T]` mask adds 0.25 GiB at 16K and 1 GiB at 32K even before other mask representations or replication.

Bounded alternatives have concrete sizes:

- One FP32 GDN state is **3 MiB**. Chunk size 256 needs about **192 MiB at 16K / 384 MiB at 32K** for chunk boundaries, plus an endpoint and scratch. Even a naive in-chunk state history is 768 MiB for the active chunk. A specialized backward can reduce that further.
- A 64-query QSA tile gathering BF16 K and V uses approximately **256.375 MiB**, with about **12.018 MiB** of FP32 scores. This must be combined with bounded backward lifetimes, not accumulated across tiles.
- A 32-token FP32 logit chunk uses **30.313 MiB** rather than 15–30 GiB. The frozen head itself and backward working buffers are separate costs.
- Compact selected block IDs can reduce retained route storage; expansion should happen per tile. The representation must include validity and causal-tail semantics and reproduce the exact selection.

Weight memory depends on cache policy, not just nominal bits. The main routed expert matrices contain about 120.796B parameters; at exactly three bits their payload alone is **42.188 GiB**, excluding scales, metadata, MTP, and trunk. Keeping all of them resident is a choice; a bounded SSD-backed cache trades RAM for I/O. For 16K tokens, routing can touch most experts in a layer, so “top ten per token” does not imply reading only ten experts per entire item. Group work by expert and measure bytes reread during recomputation.

Prefix caching also scales with corpus size: one FP32 Stage A boundary is 0.625 GiB at 16K, so **1,000 unique items would require about 625 GiB of disk** before metadata. Use quotas and measured cache reuse rather than assuming every item's prefix can remain cached. A corpus larger than the cache can miss again on subsequent epochs.

Adapter placement also matters. With independent gate/up/down adapters on all 48 × 512 routed experts, rank `r` uses:

```text
P = 48 × 512 × 3 × r × (2560 + 640)
FP32 weights + gradients + two Adam moments ≈ 16 × P bytes
rank 4:   943,718,400 parameters, 14.063 GiB
rank 16: 3,774,873,600 parameters, 56.25 GiB
```

Other factor-sharing layouts, BF16 parameters, or 8-bit optimizers give different figures. This calculation explains why copying an all-expert Axolotl target list into our trainer would undermine the initial memory goal. Attention-only adapters avoid this cost.

### Operating target on a 128 GiB Mac

Use the existing **80 GiB development process budget and 32 GiB system reserve as an initial experimental envelope**, subject to measured available memory. This is a proposed operating budget, not a prediction that Stage A or Stage B will fit. The resource guards are advisory checks/watchdogs, not a hard OS memory partition.

For Stage A, inventory the loaded trunk/head/norm weights from headers; add one residual boundary, the bounded expert/row caches, one block's QSA/MoE scratch, adapter/optimizer state, staging, and allocator slack. For Stage B, add the retained layer boundaries and active recurrent backward. At 32K, 60 GiB of FP32 layer boundaries alone leave little room under an 80 GiB envelope; grouped recomputation or disk boundary offload becomes necessary unless fewer boundaries are retained.

There is insufficient evidence to promise “24–48 GiB” or any precise total. A 128 GiB machine is a plausible target for the bounded Stage A design, and possibly broader 16K attention tuning, but **fit remains an acceptance result**. Peak MLX allocation is only one observation. Record process RSS/physical footprint, Metal allocation, allocator cache, host staging, system available memory, swap, and filesystem-cache pressure, accounting for overlapping metrics rather than adding them blindly.

CPU offload on Apple unified memory moves computation/storage responsibility but does not create a second RAM pool. Our useful distinction is **resident RAM versus bounded disk reads**. Warm mmap/page-cache residency of a 30 GiB table still pressures the same machine. Cache quotas must cover retained payloads and loading overlaps; decoded caches remain off until a measured benefit justifies them.

## Implementation plan and estimates

These are planning estimates for one engineer familiar with the repo, with automated assistance and access to the actual checkpoint/Mac. They are not run-time estimates or delivery promises. Numerical mismatches, Metal kernel limitations, SSD behavior, and active-tree integration can widen them.

| Milestone | Work | Exit evidence | Approximate focused work |
| --- | --- | --- | --- |
| M0: freeze specifications | Item-length inventory, checkpoint headers, quant layout, adapter scope, source/dependency identities | Explicit scope, byte inventory, baseline numerical fixtures | 1–2 days |
| M1: short-context Qwen forward | Model dispatch, HC/GDN/MoE/PLE, affine + selected EXL3 pack | Zero-adapter parity and tiny input-gradient checks | 3–6 days |
| M2: genuine sparse QSA | Bounded selection/gather, route replay, Q/K/V backward and reductions | Sparse oracle, causal/tail tests, composed gradient parity | 3–7 days |
| M3: Stage A 16K training | Final-layer adapters, frozen-prefix identity, loss, receipts/reload | Full 16K update, fresh uncached evaluation, measured resource fit | 2–5 days |
| M4: Stage B recurrence | Chunk checkpoints, state adjoints, convolution gradients, wider adapters | Cross-chunk gradients and bounded memory scaling | 4–8 additional days |
| M5: useful quality and 32K | Held-out generated patches, compilation/parity/timing; 32K boundary strategy | Separate quality and capacity reports | 3–7 additional days |

A **first short-context smoke run could be a few days** if the checkpoint and references align. A qualified final-QSA 16K path is roughly **two to four focused weeks**; wider coverage plus useful-quality/32K qualification is roughly **four to seven weeks total**. This is larger than the first decoding-only Apple port because it adds long-context sparse and recurrent gradient semantics, checkpoint interoperability, and application validation. A two-hour prototype would not satisfy these exit gates. Discovering an already-correct reusable Metal backward could shorten the estimate substantially.

Implement in a separate checkout/environment when implementation is authorized. Avoid changing the active DeepSeek runner, checkpoint schemas, dependency installation, or training outputs. The only artifact created by this research task is this document.

## Acceptance gates

1. **Base parity:** same quantized checkpoint, zero adapters, matching tokenization/positions; compare hidden states, selected routes, PLE hashes/rows, final logits and NLL with independently implemented references. Establish dtype-dependent tolerances before optimization. Treat changed activation arithmetic as a different baseline.
2. **Decoder/adjoint parity:** every supported codebook/window/rate and affine layout; tiny decoded matrices versus converter fixtures; forward and transpose inner-product identity; bounded allocations and unsupported-format rejection.
3. **Architecture gradients:** compare QSA Q/K/V, GDN inputs/gates, HC, PLE projections, routed-weight derivatives, and LoRA gradients against small FP32/FP64 oracles where practical. Compare composed block gradients, not just isolated kernels.
4. **Long dependencies:** loss placed after multiple recurrent chunks must yield the expected gradient on early inputs/adapters. Match an unchunked tiny recurrence. Verify convolution halos and state adjoints; no chunk detach.
5. **Sparse semantics:** reduced-budget binding tests, full-budget threshold crossings, 4-token pool/tail boundaries, invalid/padded rows, ties and route replay. No dense-forward fallback at 16K that changes the model.
6. **Capacity ramp:** 2K → 4K → 8K → 16K → 32K, including non-multiple tail lengths. Record cold/warm forward, backward, update, evaluation, peak memory, bytes read, cache hit rate, system reserve, and swap. A stage passes only at its actually measured length.
7. **Serialization and resumption:** verify exact target coverage through the end of the item, one globally normalized loss, frozen payload identity, optimizer/adapters reload, and fresh uncached evaluation in a new process. Bind model, tokenizer, quant metadata, arithmetic mode, route semantics, adapter names/basis, and dataset identities.
8. **Quality:** split by change lineage. Report rationale/comment NLL, code NLL, and EOS separately. Generate new patches on held-out tasks, then compile, check numerical correctness, and benchmark representative workloads. Teacher forcing and training-fit improvement do not substitute for these outcomes.
9. **Serving:** apply the saved adapter on the exact base in an actual serving path and compare outputs. Evaluate any merge/requantization separately. Until this passes, label export as a private training checkpoint rather than a portable production adapter.

## Adversarial review of the proposed design

Verdict: **proceed as a staged research implementation, not as an announced 16K capability**. The design is plausible because it bounds the known large buffers and minimizes initial adapter scope. The following challenges remain material. “Addressed” below means addressed in the specification; implementation evidence is still required.

| Challenge / attempted falsification | Consequence | Design response and remaining gate |
| --- | --- | --- |
| “Qwen already trains elsewhere, so importing its config should work.” | CUDA/ROCm operators, expert quantization, and PLE offload do not translate automatically to MLX | Explicit backend and architecture port; M1 parity before context scaling |
| “The 40 GiB GGUF result proves 16K fits.” | Its kernels exploit exhaustive 2K selection and its audit rejects longer fixed items | Treat as alternative-quant precedent only; M2 supplies real sparse training |
| “Sparse attention fixes memory automatically.” | Full selected K/V expansion alone exceeds 64 GiB at 16K; dense masks can remain | Query-tiled gathers and bounded backward; inspect all intermediate shapes |
| “A tiled autodiff loop bounds backward memory.” | The graph can retain every tile's gathers/probabilities until backward | Whole-operation custom VJP, per-tile recomputation, and lifetime/peak tests |
| “Layer checkpointing solves recurrence.” | One layer's naive per-token state history is 48 GiB at 16K | Stage A avoids GDN backward; Stage B requires chunk adjoints and lifetime tests |
| “Detaching chunks still trains a 16K item.” | It loses gradients across chunk boundaries and weakens long-dependency learning | Reject silent truncation of gradients; cross-chunk oracle gate |
| “Frozen weights mean no base backward.” | Earlier adapters need `dx` through frozen linears, experts, and selected router weights | Require packed/tiled transpose paths and composed-gradient verification |
| “Window 14 and 16 are almost equivalent.” | They decode different matrices; a fit can train against the wrong base silently | Parse/check window per checkpoint; converter fixture and serving parity |
| “Quant formats with similar bits share one loader.” | Scales, zero/bias terms, codebook folding, expert groups and shard ownership differ | Per-format capabilities, explicit metadata, fail-closed unsupported geometry |
| “CPU offload creates room on the Mac.” | Host and GPU use the same physical memory; cached disk pages still count | Bounded row reads and whole-system memory qualification |
| “All-expert rank 16 remains a small adapter.” | Independent factors can consume 56.25 GiB in FP32 Adam state alone | Begin with final-QSA attention; approve broader targets only with counted state |
| “Cached repeats establish large-corpus throughput.” | Every unique item needs a cold prefix; expert streaming/recompute can reread much of the bank | Report cold and warm separately, total SSD traffic, and unique-item throughput |
| “Prefix caching is cheap for thousands of long examples.” | A thousand 16K FP32 boundaries consume about 625 GiB of disk and may thrash a smaller quota | Count corpus-wide bytes and cache misses across epochs before selecting caching |
| “Final-layer adapters are enough for code optimizations.” | Capacity may be too limited despite lower NLL | Label scope honestly; compare held-out generated patches before claiming usefulness |
| “A lower pooled loss means better code.” | Our completed fit mostly improved comments/headers and regressed EOS | Separate code/rationale/EOS metrics; compile/parity/performance gates |
| “Recomputed top-k will choose the same tokens.” | Precision changes and near ties can change the selected backward computation | Replay actual forward routes and test invalidation across updates |
| “Exporting factors makes a Sushi adapter.” | Names, transposes, scaling, folded norms, and quant base can disagree | Exact-base serving application plus fresh-process parity; no portable-export claim yet |
| “The 32K constant means 32K is implemented.” | A permissive setting can coexist with unqualified gradients, memory, or data truncation | Capability registry distinguishes accepted limits from measured training lengths |
| “The development cap protects other live work.” | Advisory checks are not hard allocation ceilings; temporary overlaps can overshoot | Separate implementation environment, headroom preflight, incremental memory ramp and abort policy |

The critical pass strengthened the design in three places: explicit frozen-prefix execution semantics, a whole-QSA custom VJP instead of an ordinary tiled autodiff loop, and corpus-wide prefix-cache accounting. These are requirements to implement and test, not fixes already present in the runtime.

Residual uncertainties are the exact Apple sparse/recurrent kernel efficiency, unique-item SSD throughput, total loaded Sushi trunk footprint, the useful adapter scope, and integration with a serving engine. None is resolved by model-native context metadata or inference-only benchmarks. Quant selection also needs a matched zero-adapter quality baseline: faster training on a weaker quantized base may not improve the final application's quality.

### Capability claims permitted at each stage

| Claim | Today | After Stage A gates | After Stage B / additional gates |
| --- | --- | --- | --- |
| DeepSeek complete-item fitting beyond 512 tokens | Historical receipt supports roughly 5.8K, three examples | Unchanged evidence boundary | Separate DeepSeek qualification needed for 16K |
| Qwen3.8-Flash-Next in our Apple trainer | Not supported | Supported for the selected exact quant base | More adapter scopes after their qualification |
| Full 16K context with final-QSA LoRA | Not implemented | Supported only at measured length/budget | Still scope-specific |
| Gradients through adapted GDN / earlier layers | Not implemented | Not provided by frozen-prefix Stage A | Supported after cross-chunk and composed-gradient gates |
| 32K items | Setting elsewhere is insufficient evidence | Not implied | Separate capacity/memory qualification |
| GGUF or mixed IQ training on Apple | Mathematical feasibility and external AMD precedent | Not implied by Sushi support | Each implemented format requires its own gates |
| Better generated Metal/kernel optimizations | Not demonstrated | Not implied by a fitting run | Requires held-out compilation, correctness and performance evidence |
| Serving portable adapters | Current Apple export is private/model-specific | Remains limited until serving integration | Exact-base serving and export verification required |

## Source identities and reproducibility

External links above are pinned where practical. Research versions:

| Source | Revision inspected |
| --- | --- |
| Axolotl | `e767fe077c02a087bdf4c7da0cc2566800cddad3` |
| MS-SWIFT | `e132612cb3be4d1c92e00fd7a1be878a9b44b082` |
| NVIDIA NeMo Automodel | `6c70585d941e78b4fc642a0c3c17caa91b0f48fa` |
| Community Transformers/GGUF recipe | `be48b82232b71cd679e687839cf38cfa3fe52f9e` |
| Sushi engine | `bd4b3954f5a42a87effcdf8a68ea91a4b6a1a1b9` |
| Sushi 3-bpw model | `76bd8f55df2bd4ab6297101c57bf86d450ac0aa3` |
| Official Qwen model | `de4b8e4d43b917e7706784d8bb445c9af86a3540` |

SHA-256 fingerprints of key local files at the source-review checkpoint, relative to repository root:

```text
4890c6f9705899d28a0c663c1c8c747830e1c353081f98ed492025db3bb900f2  apple/src/exl3_mlx/deepseek.py
4da811483bf9b358cbbdbcad5db2d11caa70a6e10ccfa101c292f19660f96809  apple/src/exl3_mlx/native.py
e60c77729a89fd5a9a76cd2641b5325c5ccf8bc0ec4b80a19ff949a1579d03df  apple/src/exl3_mlx/streaming.py
50947ab825cf6effa3b481141117905d814732ff97b99d1f3cea9507755a8f16  apple/src/exl3_mlx/storage.py
d23256d346053e5af83020a2ddbdb21ad48a8635e46815709b4d9075934679cf  apple/src/exl3_mlx/recompute.py
b43a2d9d3c23ebf80fe9d908349a517e1fbb61c9a3ec090fc4c90f04de1d677e  apple/src/exl3_mlx/training.py
0be02ee9626bd197960b32a8a57cb257c2d8dc815173a12c3dd98f00622970af  apple/src/exl3_mlx/context.py
d771fcdb564ab55cfc9f5b8939d86996e4c67f33a0a3221383f6890a783ffc81  apple/reports/metal-tuning-result.md
12c99ff24bb54067a081ceb9e6b52154cf811afd231bd30a13847c70d5729f5e  exllamav3/training/native_llama.py
```

Revalidate these conclusions against the tree and checkpoints used for implementation. Acceptance receipts should bind their own complete source identities, actual item lengths, adapter coverage, arithmetic mode, hardware, memory observations, cold/warm timing, and quality split.
