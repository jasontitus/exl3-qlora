# Adversarial review of the Apple EXL3 trainer

Reviewed 2026-10-12, repository commit `c6f1ca3d86420a329502d37a2a2235ece50cbbcf`. Scope: read-only inspection of `apple/src/exl3_mlx`, its tests, the pinned official DeepSeek inference reference, and an existing independent inference implementation. No GPU jobs, server starts, or repository edits were performed. CPU reproductions use the actual source definitions isolated from MLX imports; the adapter reader reproduction substitutes NumPy safetensors I/O and a minimal model container, so it tests validation logic rather than GPU execution.

The strongest confirmed functional findings are two missing preflight checks in adapter loading. I did **not** establish a numerical bug in the trellis decoder, activation VJP, hyper-connection ordering, loss mask, or recomputation. The absence of an independent full-model comparison remains a release gate, not proof that the math is wrong. The 111-test result is prior evidence; I did not rerun the GPU suite while training was active.

## Confirmed findings

### P2 — Inference and warm start ignore an existing adapter checksum

Location: [train.py:79](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/train.py:79), especially lines 88–94; callers at [workflow.py:175](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/workflow.py:175).

`save_checkpoint` records SHA256 values for adapter weights and adapter configuration in `trainer_state.json`. `load_checkpoint` checks those values, but the public `--adapter` and `--resume-adapter` paths call `load_adapter` directly. That function never reads the available checksum and accepts any values with the expected names, shapes and dtypes. A valid safetensors payload containing changed finite adapter values therefore loads silently, while exact resume rejects the same changed artifact. File truncation is usually caught by the safetensors parser; changed payload values are the important case.

CPU reproduction: a generated adapter file was replaced with a structurally valid payload after writing its original checksum to `trainer_state.json`. The actual `load_adapter` validation accepted it. See [audit_cpu_repro.py](/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/audit_cpu_repro.py) and [results](/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/adversarial-code-repro.json).

Smallest fix: when `trainer_state.json` is present, verify the adapter configuration and weight hashes **before loading anything into the model**. This can preserve current standalone adapter compatibility; explicitly label such legacy directories as lacking a saved integrity receipt rather than silently treating them as verified. A later version can add standalone adapter checksums. Avoid changing the currently running experiment's format just to implement the existing-checkpoint check.

Regression tests: change one finite weight while preserving keys/shapes/dtypes and require `load_adapter` to reject it; change the saved configuration bytes and require rejection; assert every destination adapter value remains unchanged after either failure. Keep successful standalone and training-checkpoint loads covered.

### P2 — Nonfinite adapters are accepted and mutate the model

Location: [train.py:89](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/train.py:89).

The structural preflight checks do not test finiteness. A correctly shaped NaN/Inf adapter is loaded into the model. This remains possible even after checksum validation, for example with a standalone artifact or an artifact whose checksum accurately describes invalid values. Training eventually detects nonfinite loss/gradients and generation detects nonfinite final logits, but only **after** an expensive full-model pass. A reusable load API leaves the caller's model corrupted when it could fail without mutation.

CPU reproduction used real safetensors with all-NaN arrays of the expected float32 shapes. The actual function loaded every array successfully. This is a bounded loader defect, not evidence that the current training run produced invalid weights.

Smallest fix: evaluate `all(isfinite(weight))` for every incoming adapter array before `model.load_weights`, then raise a clear `ValueError`. Check all arrays before mutating any parameter. Also consider nonfinite Adam moments at exact-resume preflight, though the normal save path already rejects nonfinite updated adapter values.

Regression tests: NaN, positive Inf, and negative Inf with matching keys/shapes/dtypes; include a checksum-consistent invalid file; verify failure leaves the model unchanged. Check a large finite value separately so a finiteness guard does not become an arbitrary weight-range policy.

## Confirmed integrity limitations, already partly documented

### Base identity binds metadata, not verified base tensor contents

Locations: [train.py:45](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/train.py:45), [storage.py:97](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/storage.py:97), [storage.py:174](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/storage.py:174), [streaming.py:51](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/streaming.py:51).

The saved `base_identity` hashes configuration/manifests/index/tokenizer metadata. It does not hash native safetensors payloads or verify the expert/Engram payloads against the manifest on normal load. Stat fingerprints are captured anew per process, so a change made before a new process starts becomes that process's accepted baseline. CPU checks confirmed both of these facts:

- Changing native weight payload bytes left `checkpoint_identity` unchanged.
- Flipping an expert-code byte before construction passed `ExpertBank` construction and its normal `array` read. Explicit `verify_record` correctly rejected it.

The recipe accurately says `doctor --verify-records` samples only two records and does not hash the whole checkpoint; this review should not reframe that documented scope as a regression. However, exact reload/resume is conditional on the original base payload still being intact. The current behavior demo does not test that condition independently.

A practical later hardening path is a one-time full content verification receipt bound to expected remote object hashes and local files, plus per-record verification on first expert use and a verification cache invalidated by file stat changes. Native shards need content hashes as well. Do not claim that inode/size/mtime alone proves identity. Run full-bank verification separately from GPU training; the existing native deployment receipt reports an expert-bank hash taking about 665 seconds, and it excludes separately hashing every trunk/Engram file.

### Saved adapter identity omits generation-template metadata

Location: [train.py:46](/Users/jasontitus/experiments/exl3-qlora/apple/src/exl3_mlx/train.py:46). `checkpoint_identity` includes `tokenizer.json` but omits `tokenizer_config.json` and `chat_template.jinja`. Raw-text training resume correctly adds these through `training_data_identity`, so this is **not** a claim that the recently implemented raw-chat resume guard is broken. Inference with a saved adapter has no equivalent template binding. A changed template can produce different prompt IDs while the adapter still loads under the same identity. The CPU reproduction confirms this metadata change is invisible to `checkpoint_identity`.

Record both template files in future adapter/generation provenance. Preserve old-format compatibility explicitly, and report the actual prompt IDs in every behavior-comparison receipt.

## Numerical and test review

The following checks are meaningful existing evidence:

- `test_metal.py` exhaustively checks all 65,536 codebook entries and checks trellis reconstruction/permutation across integer and fractional rates using a separate NumPy oracle.
- `test_gradients.py:28` checks input VJP against float64 effective weights in float32/float16/bfloat16; adapter tests use nonzero B for A/input gradients and acknowledge zero initial A gradient.
- `test_streaming_moe.py:100` tests sparse dispatch against dense dispatch, including routing-weight input gradients; its oracle deliberately avoids top-k ties.
- `test_block_training.py:64` compares a two-block recomputation path against eager autodiff, with nonzero attention adapters and shared-KV gradient propagation.
- `test_workflow.py:70` checks the shifted loss mask and token weighting; lines 84–101 check unequal-length gradient accumulation; resume tests compare adapter and optimizer values.

Blind spots that can allow apparent learning despite an architecture defect:

1. **No independent complete-trunk logits or adapter-gradient oracle.** Existing grouped-MoE and recomputation tests reuse substantial runtime code on both sides. They can establish equivalence between two execution paths while sharing the same architecture mistake. A model with wrong math can still fit a codeword.
2. **Engram forward and activation gradients lack an independent numerical test.** The suite covers hashing and sparse storage pieces, but `Engram.__call__` at `engram.py:159` is absent from a standalone forward/VJP oracle, and the recomputation fixture has no Engram block. Test its per-stream RMS normalization, signed square root near the clamp, key/value split, and upstream attention-adapter gradient through an Engram layer.
3. **Real BF16 block recomputation is less covered than the FP32 toy.** The two-block test uses FP32 inputs and dense toy attention projections. Include BF16 `NativeLinear` MXFP8 projections, streamed EXL3 experts with nonzero native LoRA A/B, source and reuse of compressed KV, and an Engram insertion. Include lengths 1, 2, 3 and boundaries around the 128-token window and 512-token cap, under explicit memory budgets.
4. **Gradient checks need to respect the intentional surrogate derivative.** FP8/FP4 activation quantizers deliberately use identity STE. Numerical finite differences of the quantized forward are generally zero or discontinuous, so they are not a valid oracle for that surrogate. Compare against an independent graph with matching STE, while using finite differences on a smooth dequantized variant for the surrounding math.
5. **Ablation and identity checks answer different questions.** Zero-adapter logits returning exactly to baseline is strong causal evidence for the adapter's effect. Frozen Python object IDs alone do not establish unchanged array contents. For full evidence, hash/sample frozen arrays and record verified external payload identities, while keeping the fast key/shape/freezing checks.
6. **Artifact tests do not cover the inference loader's adverse cases.** Existing tests reject a changed base config and corrupted exact-resume RNG file, but never alter an inference adapter payload or submit a nonfinite adapter. Add the bounded regression cases above first.
7. **Watchdog tests mock checks rather than exercising termination/recovery.** They establish logic and limit restoration; they do not prove a subprocess exits 75 during a long operation and that the last checkpoint remains usable. Use a bounded synthetic subprocess, not real memory exhaustion, to force a watchdog condition and verify its abort receipt/checkpoint recovery.

## Concrete independent parity route

The neighboring native project has reusable instrumentation, but its existing parity suite is not full EXL3 proof:

- [parity.zig:1](/Users/jasontitus/experiments/mlx-stream/src/arch/deepseek_v41/parity.zig:1) explicitly says the routed expert output is a stand-in/injected output. `Runner.runChained` at line 842 also documents stand-in experts. Its stage comparisons are useful for trunk isolation but must not be called a real-bank end-to-end oracle.
- [dump_dsv41_fixtures.py:1](/Users/jasontitus/experiments/mlx-serve-compression-audit/tests/dump_dsv41_fixtures.py:1) runs pinned official DeepSeek code in CPU PyTorch with PipeNetwork kernel stubs. It exports `logits_full`, `logits_decode`, Engram outputs, and QAT rounding fixtures. The synthetic repack uses affine routed experts; adapt it to our EXL3-decoded weights and select a sequence/indexer configuration that matches the trainer's short-context scope. The present trainer loader hard-codes 40 layers, so initially compare individual `TrainingBlock`s instead of weakening the production architecture checks.
- For a real-checkpoint comparison, the native [model.zig:1382](/Users/jasontitus/experiments/mlx-stream/src/arch/deepseek_v41/model.zig:1382) API supports `.logits = .all`, and the graph accepts stage probes. Build a small offline exporter using its **real** expert provider and stock numerical mode; capture exact token IDs, complete logits, routes and selected block intermediates. Export after training exits, with bounded native memory, then run the trainer separately. Starting another server is unnecessary.

Proposed first comparison: several fixed prompt-ID arrays, including lengths that exercise an incomplete/completed compression group, a source/reuse boundary and Engram. Record per-stage max absolute/relative error, cosine similarity, route indices/weights, final top-k agreement, target-token NLL and KL divergence. Choose tolerances from the independent primitive/BF16 rounding envelope before inspecting outcomes. A greedy token difference near tied logits is less informative than the underlying distribution; unexplained early-stage drift is more concerning than one final token mismatch. This is forward validation only; independently matched VJP/optimizer-step fixtures are still required for training-tool validation.

## Suggested order

1. Finish and preserve the active behavior experiment; do not change its implementation mid-run.
2. Harden `load_adapter` checksum/finiteness preflight and add adverse-case tests without changing current artifact layout.
3. Add independent Engram and mixed-precision block/adapter-gradient fixtures.
4. Establish frozen-base logits parity using the independent source/native implementation and exact prompt IDs.
5. Only then use held-out task improvement plus retention/regression evaluation to claim useful tuning quality. Keep mechanical learning, numerical fidelity, and task quality as separately reported gates.

## Prepared hardening patch

After review, a stricter missing-checksum policy was selected: new standalone adapters save their tensor checksum; legacy training checkpoints retain compatibility through their existing `trainer_state.json` checksums; old standalone adapters without either receipt fail with a clear instruction to recover/re-save verified weights. This supersedes the optional permissive legacy-standalone suggestion above.

The [patch](/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/adapter-integrity-hardening.patch) changes only adapter integrity preflight/save metadata and focused I/O tests. It does not change model math, the resume contract, or the repository until applied. Python parsing, Ruff with the repository configuration, and `git apply --check` passed. A [CPU-only staged preflight test](/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/adapter-integrity-staged/verify_cpu.py) passed thirteen cases, with [results](/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments/adapter-integrity-staged/cpu-results.json). Full MLX pytest remains to run after the active training job finishes.

The patch also shares `load_adapter_config(directory)` between adapter loading and `workflow.model_settings`, so a training-checkpoint configuration checksum is checked before a tampered rank/layer selection can drive model allocation. Tests cover all three entry paths (`adapter`, `resume`, and `resume_adapter`) and show that large finite weights remain accepted when their checksum matches.

## Follow-up recorded 2026-10-12

The proposed adapter integrity hardening is now applied. The full native MLX suite passes **142 tests and 27 subtests**, including checksum/configuration tampering and nonfinite rejection before parameter mutation. An independent NumPy float64 Engram forward/VJP fixture now passes with dense FP32 projection and 192 input-coordinate finite differences; quantized STE, clamp-edge and integrated mixed-precision coverage remain open. Independent full-model parity and all-payload identity also remain open. See [current gate status](quality-status.json) and [quality plan](../QUALITY.md). The findings above describe the inspected pre-fix snapshot; this section records the follow-up.
