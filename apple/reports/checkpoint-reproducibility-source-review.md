# Checkpoint update reproducibility: source review

Reviewed 2026-10-12 local time. CPU-only, read-only inspection; no model execution, GPU replay, or engine changes. Machine-readable hashes are in `checkpoint-reproducibility-source-review.json`.

The historical checkpoint update was **not reproduced bit for bit**, and this review did **not identify its cause**. The process review reports exact initial A tensors and saved RNG state, exact final-block B tensors, but differences in 78 earlier B tensors and their Adam moments. The pre-update scalar loss matches exactly. The reconstructed B-gradient difference is approximately 0.952% relative L2 with cosine 0.99995485. A matching scalar loss is weaker evidence than matching all forward intermediates and backward cotangents.

## Recorded implementation comparison

All 36 source hashes recorded by the historical project match Git commit `b7668879` exactly. Every corresponding runtime-v2 file also matches its pinned snapshot. The historical and runtime-v2 dependency-lock hashes match, and the recorded relevant package versions agree (MLX 0.32.3; MLX-LM 0.32.0).

`recompute.py`, `materialize.py`, `moe.py`, `sparse_attention.py`, `linear.py`, `streaming.py`, `engram.py`, `hyperconnections.py`, and the Metal decode source are byte-identical between these implementations. AST comparison also confirms unchanged bodies for `batch_for`, `supervised_hidden`, `projected_cross_entropy`, `masked_loss`, `sample_indices`, `optimizer_step`, `save_checkpoint`, and `load_checkpoint` in `training.py`; only its evaluator changed.

The attention/native/deepseek changes add explicit activation-mode routing and frozen-prefix caching. In the new control, activation mode is `checkpoint`, cache capacity is zero, and the earliest adapter is block zero. Inspection finds the original quantizers, original native matmuls, original embedding/backbone path, original block order, and original loss/update formula on this path. The other changed files add configuration, identity, reporting, and UI behavior. This is source-inspection evidence of intended arithmetic equivalence, not proof that every traced MLX execution has an identical schedule.

The historical and control math settings resolve identically, including all 40 adapter blocks, no expert adapters, rank 4, alpha 8, learning rate 1e-5, chunk 8192, seed 8, input length 5808, 811 target tokens, and identical data hash. Resource limits differ: historical 80 GiB / 12 GiB reserve versus control 76 GiB / 8 GiB reserve. Packed cache, decoded cache, allocator cache, and prefix cache are otherwise equivalent. The resource amendment is not evidence of a specific numerical cause.

## Concrete repeatability limitation; unproven causal candidate

In pinned `sparse_attention.py:43–49`, `_scatter` accumulates KV gradients through `array.at[indices].add`. The causal windows of adjacent queries overlap, and selected compressed latents may also repeat across queries; these are real duplicate destination indices. The backward calls occur at lines 119–121, and FP32 accumulators are cast back to each input's dtype at lines 125–126.

The installed MLX Metal header `indexing/scatter.h:57` implements scatter updates through `op.atomic_update`; `reduction/ops.h:130–133` sends `Sum.atomic_update` to `mlx_atomic_fetch_add_explicit`. The implementation does not impose a single cross-thread addition order. FP32 addition is non-associative, so this path does not establish bitwise repeatability for colliding updates. Host materialization fixes when computation finishes, not the order of concurrent atomic additions.

This supplies a plausible mechanism for tiny differences entering earlier block cotangents and later being rounded during BF16 computation. Exact last-block adapter gradients with differences earlier is compatible with such propagation, but it does not localize the divergence or demonstrate that this scatter produced it. Other reduction/gather paths, runtime scheduling, or an unrecognized composed-path issue remain possible. No repeat-run test of fixed intermediates was performed here; attributing the observed 0.952% difference uniquely to nondeterminism would exceed the evidence.

## Claims and next diagnostic boundary

Keep `historical_reproduced=false`. Interpret the fresh runtime-v2 checkpoint-mode and BF16-mode evaluations using each mode's own baseline and the recorded resource exception. A completed matched comparison would establish those two observed outcomes; it would not certify universal descent, deterministic training, or the unique cause of the historical regression.

If further localization is later needed, the bounded discriminator is to save a late block's exact inputs and output cotangents, repeat its VJP, and compare intermediate dQ/dKV/dX against a deterministic fixed-order accumulation reference. That can separate local repeatability from propagation through earlier layers without another full training run. This report does not request or execute that work while the current run is active.
