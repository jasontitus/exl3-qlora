# Final long-context adversarial review

**Later results, 2026-10-12:** the full update completed, but the original adapter
and all three tested derived strengths worsened training-row NLL. See the
[current diagnosis](first-step-diagnosis-update.md) and
[complete strength receipts](ktune-strength-diagnostics.json). The review below
retains its earlier 511-test, pilot-running cutoff.

Date: 2026-10-12. Scope: read-only inspection of the final staged and installed engine while the 5,809-token full-model pilot runs. No GPU workload or source modification was performed for this review. The parent reports that the full suite passed all 511 tests. This review inspected the tests and previously recorded numerical evidence; it did not independently rerun that suite.

No new critical defect was found that warrants interrupting the current fresh-training pilot. Successful end-to-end long-context training and independent full-model numerical parity remain distinct acceptance gates. Unit tests and a changed output alone do not establish either.

## Mathematical and state checks

- `sparse_attention.py:20-44,107-141`: the window contains only positions up to the current query, and compressed visibility is enforced by the caller. The explicit attention VJP differentiates the shared latent through both its key and value roles, sums repeated reads, and includes the denominator-only sink derivative. Tiling changes storage/work scheduling without dropping prompt history. Float64 oracle and finite-difference fixtures cover each input role, nonuniform cotangents, two batches, and duplicate reads.
- `attention.py:131-190,216-255,292-328`: compressed group visibility is `(position + 1) // ratio`; incomplete compression groups do not enter attention. Candidate blocks include the newest reachable partial block. Candidate consumers rerank only reachable entries in those blocks. Functional shared state preserves compressed KV, index keys, selections, and candidates across layers. At the actual target configuration, new KV sources and ratio changes align.
- `recompute.py:15-83`: trainable arrays are explicit primals, integer routing state is excluded from differentiation, and floating shared-KV state receives a gradient. Module parameter references are restored around recomputation. Existing fixtures compare ordinary and recomputed execution with nonzero adapter gradients.
- `training.py:27-81,100-144`: target tokens use the preceding hidden row; prompt positions remain part of the complete backbone computation. Frozen-head projection is bounded to 32 supervised rows. The VJP divides by the total supervised-token count, and microbatch accumulation weights each mean by its supervised-token count before clipping and the Adam update. Adapter parameters are FP32 in `native.py:58-59`.
- `streaming.py:84-177`: both cache paths check frozen source fingerprints before access; decoded hits also check them. The private store token and matrix key reject cross-store/cross-matrix bases. FP16 tile storage matches the decoder's existing FP16 rounding, then restores FP32 for the original matmul order. An oversized decoded tile does not flush useful retained tiles. Cached arrays remain private and are absent from adapter/optimizer serialization.
- `training.py:199-273`, `workflow.py:411-495`: checkpoint publication uses a temporary directory followed by a rename, and exact resume verifies data/settings/arithmetic contract and all four payload checksums. Best selection is recorded before pruning; retained checkpoints' best references are protected. Stop requests are honored after completed updates, and no optimizer step zero is manufactured.

Some reviewed attention, recomputation, MoE, and cache code was authored by this same review agent. This pass is an adversarial reinspection, not a wholly independent implementation review. Separate NumPy/MLX-LM fixtures and the other review agent provide the independent checks described in the test reports.

## Nonblocking follow-ups

1. **P2 — direct adapter loads do not enforce their saved arithmetic revision.** `train.py:99-107` stores the revision as the top-level `engine_semantics` field in `adapter_config.json`, but `load_adapter_config` and `load_adapter` at lines 118-143 do not compare it with the current engine. `workflow.py:173-210` consequently permits direct evaluate/generate to load a correctly checksummed adapter produced before the quantization scale correction under the new arithmetic. SFT project implementation identity and the `trainer_state.json` contract's `engine_semantics` field reject implementation/math drift for standard-project and exact optimizer-resume paths, so the active fresh pilot is unaffected. Standalone legacy-adapter users currently need explicit operator awareness: the tool has no enforced migration acknowledgment. A follow-up should reject incompatible or absent revisions for evaluation by default; an explicit warm-start migration can preserve intentional reuse. Add a checksum-valid old-revision fixture and assert rejection before destination parameter mutation.

2. **P3 — optimizer state validation is structural, not finite-value validation.** `training.py:253-269` validates optimizer keys/shapes/dtypes and step but does not reject nonfinite moment tensors; `save_checkpoint` also does not independently reject them. A coherently saved/checksummed invalid optimizer state could therefore load, even though adapter loading rejects NaN/Inf. Normal clipped FP32 fresh runs should not generate such a state. Add finite moment checks before publication and before optimizer assignment, with an invalid-state test that preserves destination model/optimizer/RNG values. `optimizer.init` currently occurs before later structure/RNG checks, so a failed load can initialize the supplied optimizer even though model weights remain untouched; ordinary workflow resume supplies a fresh optimizer.

These follow-ups do not justify modifying final source during the current pilot.

## Limits on claims

- Full-model logits and gradients have not been matched to an independently verified native implementation for this repacked checkpoint. Isolated EXL3, Engram, attention, router, and optimizer tests are strong component evidence, not an end-to-end golden comparison.
- Index scoring uses MLX-LM's FP32 accumulation convention. The pinned Torch implementation can round intermediate index scores/scales in BF16, changing selections near ties. Low-index tie ordering is explicit in this engine. Native bitwise selection parity is not claimed.
- Activation quantization uses an identity straight-through surrogate. Discrete index/expert selection is frozen. Custom VJPs support first-order training; higher derivatives are not supported.
- The 32,768-token cap is an implementation bound, not a demonstrated training capacity. Candidate pruning beyond 16,384 has selection fixtures, not a full 32K training receipt. Long-context index scoring still performs quadratic work, even though score memory is tiled.
- Base checkpoint identity is metadata/manifests plus runtime inode/size/mtime checks, not a hash of all frozen payload bytes on each run. Accidental rewrites that change normal metadata are detected; corruption predating startup or an adversarial rewrite preserving metadata is outside that guarantee. This is a known scope limit, not a new cache failure.
- Save publication is atomic for process interruption; it does not explicitly fsync every payload and directory for power-loss durability. Exact resume is supported for complete saved states, not an interrupted partial update.
- Cache limits bound retained cache entries. They do not cap every transient cast, graph, OS page cache, or host copy. A memory guard can abort a run; a completed update/checkpoint receipt is needed before claiming training succeeded.

## Performance evidence relevant to the chosen resource profile

The complete one-MoE-layer reports use genuine router/shared/routed weights and 5,808 synthetic normalized BF16 rows. They measure a forward followed by public VJP, which includes another forward and expert recomputation. Same-chunk output and input-gradient hashes match exactly in every recorded condition.

| Column chunk | 128 MiB packed, no decoded | 8 GiB packed, 32 GiB decoded | 8 GiB packed, no decoded |
| --- | ---: | ---: | ---: |
| 512 | 57.90 s | 62.79 s | 53.73 s |
| 8192 | 48.54 s | 54.69 s | 44.10 s |

Sources: `reports/decoded-cache-moe-benchmark.json` and `reports/packed-cache-moe-benchmark.json`, both marked complete. These are one-sequence timings with order/JIT/OS-cache effects. The evidence supports leaving decoded caching off for this workload. It does not establish full-model throughput. Wider column chunks change reduction order and have separately measured BF16 differences; chunk 512 remains the exact-arithmetic default.

## Reviewed source identity

All nine listed staged files matched the installed files byte-for-byte when inspected:

| File | SHA256 |
| --- | --- |
| attention.py | a22ffd2e0fe128949f045eb95dc332367bca6c70fbf96324b959813e2be3e2c4 |
| sparse_attention.py | 620082cc1e12cbb2f66b09054ff9bea08d9dadc6c34bd802ca25ae7aa26a21a7 |
| recompute.py | a7849b0b1b6877b62914b75e21ab7a66305469e2a0c72def94e8b1caa90cd988 |
| deepseek.py | 8ab19c61590f390d0fd1a9e72229b72f35332cb838244dd83c1446acb9f1073b |
| training.py | 684cb6281c9d37ae3803a0437c9d7d946c5fffabae9458b65dbab7ab3bfb9826 |
| train.py | b820893532405ad46a84aedc170242d5ce98f830c9573eaddfc046859d165de1 |
| streaming.py | e60c77729a89fd5a9a76cd2641b5325c5ccf8bc0ec4b80a19ff949a1579d03df |
| moe.py | 88b821f17c0eda364cb136170f2de52285f872ef0a6330b4a280518d6d1828e0 |
| workflow.py | 40371fb3fa90bd5f32c30d78363f02612cc9f0fbba03ccdefb68aed39aead945 |
