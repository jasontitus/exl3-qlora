# Metal optimization library: collection pause and inventory

Snapshot: 2026-10-14, following the user's request to pause new gathering after the in-flight work. The daily `build-metal-optimization-library` automation is paused; no further history mining or new source fetches are authorized until resumed. Existing review, training and evaluation work continues. The corpus and earlier receipts are preserved.

We have a substantial source-backed training corpus spanning all twelve requested repositories, plus a separately held next-round library. We have not exhausted these repositories or demonstrated that the current adapter produces better kernels. Discovery, complete-edit preparation, source review, compilation and measured performance are different stages.

## What is in the current experiment

The running DeepSeek campaign uses **229 training views: 186 exact-edit views and 43 diagnostics**, plus **16 frozen held-out items**. One view is one serialized training item, not necessarily one independent optimization. Related commits and exact/diagnostic views can share a lineage. There are 427,072 supervised training tokens per pass and 47,069 held-out targets. The campaign has two passes / 458 updates, one complete item per update, with no silent truncation.

| Repository | Training exact edits | Training diagnostics | Frozen test items |
|---|---:|---:|---:|
| ds4 | 16 | 4 | 0 |
| TensorFold | 38 | 8 | 0 |
| sushi | 12 | 8 | 0 |
| mlx-serve | 11 | 8 | 0 |
| llama.cpp | 0 | 0 | 16 |
| omlx | 12 | 0 | 0 |
| MLX | 39 | 8 | 0 |
| mlx-lm | 18 | 2 | 0 |
| mlx-vlm | 9 | 0 | 0 |
| splash | 14 | 3 | 0 |
| MTPLX | 4 | 0 | 0 |
| vllm-metal | 13 | 2 | 0 |
| **Total** | **186** | **43** | **16** |

The llama.cpp rows are held out, so its zero training count is intentional. DS4 is a separate engine; it is not assumed to duplicate another repository simply because its operations resemble theirs. Source lineage and patch equivalence must establish any deduplication.

Exact views include complete changes to compute kernels and host dispatch/integration. Diagnostics add source-grounded discussion of correctness, precision, capacity and performance tradeoffs; they are not all new source edits. The corpus should not be described as 229 independently benchmarked, standalone Metal kernels. The frozen campaign retains its original 43 diagnostics; separately corrected future diagnostics do not retroactively change it.

At the latest verified training observation, the M5 had saved update 160/458; independent exposure audit covers checkpoint 150: 150 distinct views (121 exact, 29 diagnostic), 280,642 target presentations. These exposure counts do not prove gradient correctness or useful model improvement.

## What we collected and how it is packaged

Coverage includes packed quantization/dequantization, GEMV/GEMM and split-K selection, attention and masking, reductions/normalization, fusion and expert aggregation, memory layout/precision, and dispatch/cache/streaming policy. Host integration is preserved where it is part of the optimization; such examples are tagged separately from pure-kernel work.

Complete captures retain repository, revision, parent, full before/after files, associated host edits, source links and license evidence. Prepared agent-facing views use a task with baseline context and complete edits in the expected application format. Smaller context views retain changed targets, relevant declarations and dependency contracts; omitted unchanged context is recorded. Token counts are after native chat serialization, with supervised masks and EOS checked. Oversized originals remain archived rather than truncated.

Reasoning views use concise, auditable technical explanations: what changed, reported evidence, why a choice might help, failure domains and what tests are required. Author measurements are attributed; curator inference is labeled. A commit message or a plausible rationale is not a local benchmark receipt.

The collector's dated 2026-10-13 snapshot recorded 2,544 unique inventoried commits and 308 keyword candidate commits across twelve repositories. These are **historical discovery counts**, not today's cumulative library total or training-ready examples. Later bounded pages overlap anchors and earlier batches; their lengths must not be added as unique commits. Keyword matches also include irrelevant CUDA, formatting and build changes.

## Separately held next-round material

The selected `future-reviewed-context-library-20261014-v3` catalog contains **11 views of 7 repo/commit IDs: six historical exact-edit views and five diagnostics**, all within 12,288 serialized tokens. Independent review binds439 inputs. It has **zero training admissions** and is not the entire future library. Multiple views and related matmul/attention families do not become independent optimizations merely by having distinct IDs.

Recent additions are separate from that catalog and from the active corpus:

| Material | What has been verified | Remaining limit |
|---|---|---|
| MTPLX MoE combine | Complete source/replay; compact view 11,398 tokens/4,847 targets; three source-bound BF16 fixtures compiled offline with Metal 4.0 | No local numerical/performance result or historical MLX JIT-wrapper parity; no admission |
| MTPLX masked softmax | Complete source/replay; compact view 16,087 tokens/6,034 targets; corrective follow-up retained | Exceeds active12,288 capacity; later peak-memory/default-on reversal prevents unconditional positive imitation |
| MLX FP4 QMV dequantization, PR2720 | Three complete file pairs, Git blob/license/primary discussion checks and replay; finite overlap screen | Author reports121.29→124.01 generation tokens/s (~2.24%), with unspecified hardware/model/workload/precision. QMM was deliberately unchanged because slower. Exceptional0xFF conversion contract needs caller-domain review; no local performance or admission |
| MLX ordinary attention large-mask indexing | Complete reviewed edit/replay; two ordinary source translation units compiled with 18 instantiated kernels each | NAX branch excluded from compile scope; no local numerical/performance result |
| MLX attention precision | Before/after source closure compiled, including six each of half/BF16/float instantiations per translation unit | Author-reported precision/speed tradeoff not locally reproduced |

For the compact MTPLX views, the reviewer initially reported an omitted import, then retracted the finding after checking exact source line numbers. Original contextv1 contains the imports and remains selected; the unnecessary extension is preserved as superseded. This history does not invalidate the source or justify changing supervised targets.

For FP4, a separate CPU bit oracle found 255 finite conversions agree while the standalone 0xFF conversion changes NaN to infinity. An existing scale helper already used the latter mapping; this is not proof of a new QMV runtime regression. It is a precise domain-contract hold on positive-edit use.

A compiled fixture count is not a count of distinct optimizations. The fixed kernel evaluation harness has independent correctness/timing controls, but those curator-written controls do not validate every training source or demonstrate model-generated improvements.

## What remains, without further gathering

1. Finish independent reviews already underway and consolidate source/lineage/readiness indexes. Keep raw candidates, reviewed source/edit views, compile checks and device-validated cases separate. There is no single current deduplicated count covering every branch of the working library; publish one only after reconciling those overlapping manifests.
2. Resolve or explicitly exclude the known source-contract holds: thin-N geometry, segmented alignment, two-pass attention query offsets, nonfinite cross-entropy behavior, TensorFold prescaling, MTPLX softmax memory policy and FP4 exceptional conversion. Protected Erf and GEMV protection/capacity constraints remain enforced. These can support careful diagnostics without becoming endorsed positive edits.
3. Validate existing eligible kernels against actual dependencies and toolchains, then numerical parity and representative paired timing when GPU ownership permits. Compilation alone does not establish correctness or speed. The source review does not reconstruct every historical dependency installation.
4. Reconcile families, cherry-picks and equivalent patches against protected ktune, the active 229/16 corpus and future captures before any next split. Finite identity screens are not exhaustive semantic-independence certification.
5. Qualify longer complete items separately. A 16K serialization fits a 16K capacity setting; it does not prove that the model's full forward/backward fits memory or has correct long-context gradients. Do not insert16,087-token softmax into the running 12K campaign.
6. Finish the current training and evaluation. Report added-code, rationale/comment and EOS separately, per-item regressions and source strata. Generate held-out edits and check application, compilation, numerical correctness and performance before claiming useful Metal optimization.

If gathering is resumed later, remaining discovery gaps include older history for every repository, selected branches and path-based searches (including inline MSL and mlx.fast), fuller PR/review discussions, changelogs/releases and primary announcements. All twelve sources are represented; **all useful optimizations have not been exhausted**. The latest older MLX cursor is22a5da76c8c3b92595bff2af7b304b66e1581792. Resume from stored cursors rather than refetching or counting overlapping pages twice.

## Evaluation status and evidence limits

At 19:18 UTC, the matched rank-four zero-B baseline had completed all four items / 11,077 targets. Its NLL was 0.1332943271, elapsed time 3,523.64 seconds (about 59 minutes) and peak MLX allocation 22.18 GiB. Controller68761 and runner68766 remain live; the fresh checkpoint-150 evaluation worker11734 has started. No paired quality result is available yet.

The fixed panel has 4,607 added-Metal targets, 261 added-host-code targets and only five rationale targets. Conditional-reference rows account for 8,718 targets; the bounded-reference stratum has 2,359. These must be reported separately.

No paired checkpoint 150 learning result is available yet. The previously completed checkpoint 50 probe used training-pool examples, not held-out data: pooled NLL fell 4.84%, but total correct predictions were unchanged and all-Metal NLL worsened. It did not demonstrate faster inference, fewer generated errors or better reasoning.

The final 458 fresh all 16 evaluation remains planned. Serving baseline 33+4 qualification passed; actual trained-adapter loading/cached-fresh checks remain separate and unqualified. The prospective serving tests are under review; they do not interrupt loss evaluation. The valid-null-metadata parser repair is staged independently because product auditor files are pinned by the live operation.

## Authoritative local records

Paths below are relative to the workspace, except the paused automation.

- Active assembly: `kernel-library/next-round-v5/expanded-candidate-12288-v3/provenance.jsonl`; frozen run data/summary: `two-host-split-v1/data/` under that same next-round directory.
- Discovery snapshot/gaps: `kernel-library/reports/source-coverage-gap-audit-20261013.json`; source registry: `kernel-library/sources.json`.
- Selected future catalog: `kernel-library/next-round-v5/future-reviewed-context-library-20261014-v3/manifest.json` and `independent-review-v1.json`.
- MTPLX source/context/compile: `next-round-v5/mtplx-batch7-v1`, `mtplx-batch7-context-v1`, `mtplx-moe-compile-v1` under `kernel-library/`; independent review hashes8ae3879a…,25d83eb9…,b63b4015… respectively.
- FP4 source/primary review: `kernel-library/next-round-v5/older-mlx-batch8-v1/fp4-dequant-primary`; reviewb1c2faf5…; full values and original evidence are in each receipt.
- Live evaluation: `kernel-library/next-round-v5/intermediate-evaluation150-v1/concrete-execution-v1/root-launch.json` and `future-execution/`; current handoff: `HANDOFF-2026-10-14.md`.
- Paused collection schedule: `/Users/jasontitus/.codex/automations/build-metal-optimization-library/automation.toml`. Training/evaluation follow-up is not paused by this collection stop.

The evaluation phase transition is preserved in `kernel-library/next-round-v5/intermediate-evaluation150-v1/root-status-transition-20261014T1918/` (actual process output, baseline report and controller snapshot). The older layer19 observation is corroborated by an accumulated log-prefix capture under `root-status-observation-20261014T1917/`; it is no longer the current-status claim.
