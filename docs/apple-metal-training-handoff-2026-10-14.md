# EXL3 Apple training and Metal optimization handoff

Snapshot: **2026-10-14 13:39:46 UTC / 06:39:46 PDT**. This document supersedes the 12:02 snapshot, preserved in `HANDOFF-2026-10-14-at-1202UTC.md`. The original 11:18 snapshot is also preserved. The goal remains active; this handoff does not pause the live training run.

## Current outcome and limits

DeepSeek training is live on the M5 at **saved update 101 of 458**. The completed checkpoint-50 comparison improved teacher-forced probabilities, but did not improve total correct token predictions. Generated Metal quality, reasoning improvement and final held-out benefit remain unproven.

The Studio completed independent numerical and timing qualification of the fixed kernel benchmark harness. Those results compare curator-written controls, not model-generated kernels. Serving parser/runtime repairs are built and independently reviewed, but actual serving qualification has not run: the latest read-only memory admission required 72 GiB available and observed approximately 67.7 GiB.

New source captures and complete-edit context tools are available for a future corpus. They have not been inserted into the active training data. Qwen remains isolated CPU research/implementation; no complete Qwen training or 16K capability is qualified.

## Workspaces and non-negotiable operating constraints

Workspace: `/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments`.
Product Git checkout: `publish-exl3`, branch `apple-metal-training-m5`. Experiment paths below are relative to the workspace; most receipts are outside the product checkout.

Preserve the frozen 229-view data, 16-item held-out set, completed baseline, model weights and pinned `runtime-v5`. Never launch a duplicate GPU job, change the active dataset, silently truncate edits, lower the system reserve, delete models, or signal unrelated workloads. Keep **sfideas, WatchAI and worldpolicies** running. M5 trains; Studio collects data, performs CPU work and runs separately qualified evaluation. No 3090 or independent layer campaign has been started. Independently trained adapters are not assumed safe to combine.

Daily collection remains daily. Automation IDs: `continue-exl3-training-and-qwen-work` and `build-metal-optimization-library`. Manual bounded mining was separately authorized. Do not increase schedule cadence.

## Live M5 campaign

Host: `jason-m5-MacBook.local`.
Remote root: `/Volumes/Scratch/exl3-training/2026-10-13`.
Active attempt:

```text
/Volumes/Scratch/exl3-training/2026-10-13/metal-mix229-internal-split-v1/attempts/attempt-1791952639336825000
```

Fresh read-only observation at snapshot time: worker **47890**, parent **47866**, elapsed **09:02:25**, CPU **384:14.50**, RSS **19,018,976 KiB**; `train/latest.txt` = `step-000101`. This verifies liveness and the saved pointer, not an independent audit of all 101 exposures. Independent saved-state/exposure audit remains through **69**.

Fixed campaign: **229 training views = 186 exact edits + 43 diagnostics**, 16 held-out items, two passes / 458 updates. One complete item per update. Only **layer 39 `wq_a` and `wo_b`** have LoRA: rank 4, alpha 8, **78,848 parameters**. Base and experts are frozen. Training views are not necessarily independent optimization families.

Settings: backbone capacity 12,287 / serialized capacity 12,288; LR `5e-5`, clip 1, seed 8, accumulation 1; save every step, retain two; final fit examples 0; no automatic comparison. Process budget 80 GiB, reserve 8 GiB, packed cache 6 GiB, decoded cache 0, prefix RAM 1 GiB, prefix disk quota 64 GiB, allocator cache 256 MiB; TF32 disabled. Do not change the running settings.

Model on M5 internal drive:
`/Users/jasontitus/.mlx-serve/models/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw`.
Studio model: `/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw`.
Legacy matching identity: 120 files / 426,352,806,119 bytes. Fresh Studio full-payload verification also covers 245 incidental files: 365 files / 426,352,827,347 bytes; independent review `exact-generation-qualification-v1/payload-full-verification-independent-v1/review-v1.json`.

Read-only check before any recovery:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 jason-m5-MacBook.local \
  'ps -p 47890 -o pid,ppid,etime,time,rss; cat /Volumes/Scratch/exl3-training/2026-10-13/metal-mix229-internal-split-v1/attempts/attempt-1791952639336825000/train/latest.txt'
```

Inspect actual `train/history.json`, checkpoint `trainer_state.json`, and `training.log`; an SSH timeout does not prove termination. Remote campaign receipt: `reports/split-internal-training-v1.json`. Capture and independently audit saved step 100 or later without touching the running data or trainer.

### Independent exposure and cache evidence

`kernel-library/next-round-v5/reports/training-capture-step69-v1.json`, `training-exposure-continuity-v4.json`, and `independent-exposure69-zero-b-v1/review-v1.json`:
69 distinct views, 53 exact / 16 diagnostic, 132,003 target presentations, maximum serialized length 12,078, historical MLX allocation peak **71.2798 GiB**. Step-55 history/contract remains unchanged. Review SHA `318e58339264d9ce48e2c6bf2902de0a32b21e6c20ee78a06e5141c4ce188c71`.

Strict auxiliary zero-B validation fixes are separately reviewed in `early-learning-probe50-v1/zero-b-audit-v2`; actual prior metrics remain unchanged. Review SHA `2798d6a1e93ec3761825de6fbc750b8ab1ba78a588419ccbe02c7530730f283e`.

`two-host-split-v1/prefix-cache-capacity-assessment-v1.json` projects approximately 46.388 GiB for the fixed corpus under the 64 GiB quota, using actual cache layouts. This is not a measured full-cache allocation. Quota exhaustion raises; it does not evict automatically. Do not invent a larger cache requirement or alter this quota mid-run.

## Completed evaluation

Base directory for these receipts: `kernel-library/next-round-v5`.

**Full held-out baseline:** `two-host-split-v1/studio-baseline-v2` completed 16 items / 47,069 targets, NLL **0.1110763865**, peak MLX **22.6051 GiB**. Independent review `fb536a05fb179b981c54e741351de9cdb57998a3825e94de9ba798e6f4a7f16e`. Lease released and workers absent. Preserve all 16, reporting nine scoped-source and seven disputed-reference strata. Only five rationale targets exist, so this set cannot establish broad reasoning quality. Historic baseline uses a rank-8 zero-B graph; matched final rank-4 comparison needs explicit handling, not an assumption of bitwise equivalence.

**Checkpoint-50 early probe:** `early-learning-probe50-v1`, two fresh matched rank-4 conditions, four complete TRAINPOOL rows / 9,329 targets. Two rows previously seen, two future training rows; these are not held-out tasks. Independent actual review `paired-independent-v1/review-v1.json`, SHA `2148fc2a12eff1aefe1c16ad915113e9827d4c665c3069afa93b2886780dc2bd`.

- Pooled NLL **0.393180734 → 0.374158164**, 4.838% lower; all four case NLLs lower.
- Correct predictions **8,432 → 8,432**: 27 corrected, 27 regressed.
- All Metal code, 539 targets: NLL **0.091318972 → 0.093388437**, worse; correctness unchanged at 522.
- Literal added Metal, 83 targets: NLL **0.074488868 → 0.071259384**, correctness unchanged at 80.
- Rationale, 144 targets: NLL **3.444025 → 3.080689**, correct 60 → 59. EOS unchanged.

This is mixed probability fitting, not proof of fewer coding errors, faster inference, better reasoning or faster generated kernels. Final-458 fresh held-out evaluation remains outstanding.

## Fixed kernel harness: actual numerical and timing qualification complete

Base: `kernel-library/next-round-v5/full-kernel-evaluation-v1`.

Numerical output: `controller-posix-v3/numerical-controls-run-v2`. **896 GPU dispatches**, four variants × 224 cases. Baseline/golden pass all 224; wrong-row fails 120, zero-output fails 181 as expected. No canary/input/nonfinite violations; owned lease released. Independent actual review `numerical-v4-independent-v1/review-v1.json`, SHA `cd862660530ecfb22974fb6495e576c5ccaf6afeca883c08f36cc03e497c6915`. Golden passing raw floats were not retained; reviewed native checks/errors/digests support its receipt, not an independent reconstruction of every passing float.

Timing output: `controller-timing-posix-v5/timing-controls-run-v1`, launch logs in sibling `root-launch-v1`. **840 batches / 84,000 dispatches** over 12 workloads, five warm batches per variant and 30 alternating measured pairs per workload; 100 dispatches per batch. Actual native elapsed 48.834 seconds; all 840 final-buffer checks pass; own lease released. Independent actual review `timing-actual-independent-v1/review-v1.json`, SHA `5715a4ae48b0dea559636ab185ae296d2b93bb8ed3f0cee174928fcd283ab614`; 1,051 pins rehashed by root.

Known scalar/SIMD32 paired median ratios span **5.20–13.80**. These are amortized command-buffer timings of curator controls, not fine-tuned model benefit. Preserve original failed host-name and Foundation numeric-parser attempts. POSIX timing v5 fixes the actual decimal conversion issue without changing tolerance or schedule. Do not repeat completed lease recovery or fixed-control runs.

Generated-candidate consumers must bind actual numerical v4 and timing v5 artifacts. Fixed agent task: 774 tokens, one greedy final-458 and one matched zero-B output, cap 4,096, no repair/retries/best-of-N. One code task is not 224 independent tasks. Rationale rubric in `rationale-rubric-v1` remains unscored and qualitative.

## Serving: repaired build, reviewed operation, no actual serving qualification

Base: `kernel-library/next-round-v5/exact-generation-qualification-v1`.

Native parser now handles actual nested `text_config`; Python accepts actual null adapter metadata and the known approximately 183 MiB runtime metallib under a bounded declared-runtime allowance. Old source/binary/package remain preserved.

New linked binary from `cached-inference-host-build-v6`, SHA **3e0371f60c12b16d6502cbf18dbda4bcdd04e0222330aaeae643949de002defd**. Build-only execution completed; independent artifact review SHA `e1470f0c973a438b80f927254666e00f1ed0298e93ecd06a5aef092ea488231b`; package review is `runtime-package-v2/independent-adversarial-v1/review-v1.json`, SHA `85424db328940a7fc080596c71562fa960052853db5c0aebaa021495dea47cea`.

`short-serving-requests-v4` binds six actual cached requests (33/129 tokens × absent/zero-B/prior-30 adapters), four greedy tokens, context 256. Twelve same-condition fresh-prefix children give **18 maximum phases**. Cached/fresh exact observations, adapter construction and arithmetic are still unqualified. Missing EOS-shortened observations remain incomplete; do not substitute samples or loosen tolerance.

`short-serving-operation-v2` fixes a composed terminal-evidence drift finding in v1. Independent review `independent/review-v1.json`, SHA `762f7382c55d5251c66980cd9ebbaaa6d58ff097b6e1cbbd761ebc643cb565ea`; 92 pins root-rehashed. This approves scoped operation source, not memory fit or GPU execution.

Read-only admission MUST use the exact reviewed interpreter:

```bash
/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
  kernel-library/next-round-v5/exact-generation-qualification-v1/short-serving-operation-v2/operation.py --admission-only
```

A mistaken Python 3.12 invocation was refused before memory/lease/device work and preserved in `short-serving-operation-v2/root-admission-v1/wrong-interpreter.json`. Correct-interpreter session 68789 validated six requests but refused admission: observed **72,690,270,208 bytes** available, required **77,309,411,328 bytes**. No child, GPU or lease was started. Observation is historical; remeasure before action.

Keep separate terms: **40 GiB process / 96 GiB native box / 32 GiB reserve / 128 GiB physical**, native 8 GiB wired margin and unchanged 16K load preflight. The native box ceiling is not a process RSS guard. Worldpolicies Python 72640 was using approximately 20–22 GiB and its index job 99360 approximately 2 GiB; both belong to unrelated work and must remain running. Availability may change naturally. Smaller-cache assessment `short-serving-operation-v1/smaller-profile-assessment.json` is analysis, not an approved replacement fit profile. Do not bypass the current guard or merely rename a smaller budget.

## Future library and reusable tool

All current data remains frozen. A future corrected 245-view corpus includes the same exact/held-out cases and corrected diagnostics; repackaging does not add source optimizations. Known original MinP diagnostic correction is future-only (scheduled active exposures 148/391); record the limitation without silently changing the run.

Product commit **75de7ff2** adds `apple/src/exl3_mlx/agent_context.py`, 51 tests and `docs/complete-edit-context.md`. It retains complete assistant edits, exact Git replay and supervised targets while using explicit edit-location context and optional hash-checked baseline excerpts. No fetched scripts execute. Independent product review: `edit-location-context-independent-v1/product-review-v4.json`, SHA `179001356ad721f459ef19e3e58568e30ea62db95b86d85fa4fbc7751d3b6632`. This is reusable tool code, not a one-off training script or an automatic dataset admission certificate.

Source additions under `kernel-library/next-round-v5`:

| Bundle | Captures and current review tier |
| --- | --- |
| `new-metal-batch2-v1` | Two prior-inventoried IDs: CE and GEMV. CE critical diagnostic source-reviewed; positive held for nonfinite backward. Full GEMV edit oversized/protected-parent hold. |
| `cross-source-batch3-v1` | Two newly captured IDs: TensorFold/oMLX. Source-reviewed diagnostics; TF numerical/domain limitations remain. oMLX complete component draft 6,241 tokens / 1,423 targets approved for scoped source/replay review, not whole-engine speedup. |
| `historical-batch4-v1` | Three newly captured MLX IDs: quantized split-K, segmented NAX, split-K/AddMM. Eighteen complete changed files and three source-reviewed report diagnostics. Full oversized edits archived. |
| `undercovered-batch5-v1` | Two newly captured IDs: MTPLX GDN norm/gate `dbcee1f00058`, vllm-metal sliding tile `a9922fb88cf4`. Full source/parents/license/primary reports/replay preserved; two report diagnostics reviewed. |

Across batches 2–5: **nine captured changes, seven IDs new to prior local inventory**. Multiple views do not add independent optimizations. Exact overlap checks are bounded, not exhaustive semantic independence claims.

New complete-edit context views fit 12,288 without truncating targets:

- `edit-location-contracts-v3`: MLX quant split-K **12,260 / 4,966 targets**, AddMM **10,939 / 2,634**. Review `edit-location-context-independent-v1/contracts-v3-review.json`, SHA `67d14971ea24fb2f7a25855a7c9ec69c07ccc41561918a30b8a8acfc083444b8`; root-rehashed 65 pins.
- `undercovered-batch5-context-v2`: MTPLX **5,811 / 2,325**, vllm-metal **12,064 / 1,601**. Review `undercovered-batch5-independent-v1/context-v2-review.json`, SHA `be107dfc9ec53ad47395b52139ac698a995b94fec0cd5222bfaf1214b2d78e15`; root-rehashed all 74 pins at this handoff. Full 80,165/16,015-token views remain oversized and preserved.
- `segmented-alignment-review-task-v1`: **7,217 / 509**, source-grounded critical reasoning about a supported shape causing an out-of-bounds NAX store. Review `independent/review-v1.json`, SHA `e76aeeb7c1d7951a3882d82122257d7f35a0f67e2d14b3e6c40524ec339f5335`. Positive segmented optimization remains held. Conditional source-contract proof, not a locally reproduced GPU crash.

These are source/context-reviewed future drafts, **not automatically training-ready**. Compilation, numerical parity and local performance remain separate per-case gates. No new future manifest/export or new run was created by this handoff. Next curator should consolidate these approved drafts in a new future-only manifest, preserving lineage, exclusions and all evidence tiers.

`kernel-library/sources.json` tracks all twelve repositories: sushi, TensorFold, mlx-serve, ds4, llama.cpp, omlx, mlx, mlx-lm, mlx-vlm, splash, MTPLX, vllm-metal. Continue older-history backfill and review backlog with explicit coverage gaps. ds4 is a different engine; deduplicate actual patches/lineage rather than assuming overlap by name.

## Qwen and agent status

Isolated `qwen-next`; design `qwen-next/docs/qwen38-flash-next-16k-finetuning-design.md`, product commit `bf180022`. Preserve sealed `project.json` and reviewed source identities. CPU storage/decoder/affine/MCG and QSA-selection work exists. Selection tests cross 2,051/2,052 tokens and sample queries into 4K history. No complete forward, sparse-attention backward, recurrent backward, real checkpoint training, 16K or serving capability follows from those primitive tests. Review `qwen-next/reports/qsa-selector-independent-review-v2.json`, SHA `ad116b5cc22bf9a487a760ec87a4f3aa25c23ddd792e19e95a35e30637423e9e`.

Existing agents completed their assigned work: `/root/adversarial_code` reviewed runtime, context and batch5 repairs; `/root/ingest_engines_b` captured and repaired source contexts; `/root/quality_research` built the segmented critical reasoning example and runtime preparation. Recheck live agent state before delegation. Use `followup_task` to wake an idle agent; `send_message` does not start an idle turn. No agent is authorized to launch an unreviewed GPU job.

## Next operator actions, in order

1. Read this handoff and the actual receipts; verify live M5 worker and saved state. Capture/audit update 100 or later. Never restart an existing job.
2. Recheck serving headroom and shared GPU ownership. If the unchanged profile passes and all concrete reviewed dependencies match, prepare a fresh root-authorized serialized serving qualification using exact Python 3.14. If not, continue CPU work and develop a separately reviewed bounded profile; do not stop unrelated work or bypass guards.
3. Consolidate the four bounded exact-edit drafts, critical segmented task and report diagnostics in a future-only manifest. Retain source-family grouping and explicit unqualified compiler/numerical/performance status. Continue mining and isolated Qwen CPU work.
4. When update **458 actually completes**, validate exact final bundle using `two-host-split-v1/final-transfer/check_final_bundle.py`; preserve data/runtime/model/optimizer identity and transfer only reviewed files. No final bundle exists yet.
5. Fresh-process score all 16 held-out items with explicit matched graph handling, scoped/disputed strata and separate code/rationale/EOS metrics. Use held-out generated edits with compilation, numerical and timing gates; report mixed or negative results honestly. Teacher-forced fit alone does not demonstrate useful kernel optimization.
6. Document concrete results and only mark the larger training/evaluation goal complete once required work is finished. Handoff itself does not complete or pause it.

## Git, publication and durable records

Latest product implementation commit: **75de7ff2**. Earlier handoff commits **b0d1dd54**, **201944ab**; audited exposure report **72d73b7d**; Qwen design **bf180022**; historical Metal compiler **17454054**.

This updated handoff exists at workspace `HANDOFF-2026-10-14.md` and product `docs/apple-metal-training-handoff-2026-10-14.md`. Continuation ledger `kernel-library/next-round-v5/m5-successor/current-run.json` contains older fields; prefer explicitly dated actual observations and receipts, including this snapshot. Do not interpret old fields as live state.

Push remains unperformed: automatic approval review previously rejected public-fork publication because explicit public-sharing consent was absent. Do not retry or bypass that rejection. Local documentation and commits are authorized. All baseline outputs, historical failures and reviewed receipts remain preserved.

## Continuation after the snapshot — 2026-10-14 13:59 UTC

M5 was freshly verified live at 13:51:25 UTC, saved 103/458. Independent audit now covers 101: 80 exact / 21 diagnostic views,195,554 targets, first 69 history and full contract unchanged. Review `reports/independent-exposure101-v1/review-v1.json` SHA134ac17600049e40269c31ab1c96e1999cf8cfa7e8f620a4a382c6d3337caf92; root verified 13 pins. Checkpoint hashes remain remote-observed metadata, not independently read local payloads. Product receipt committed as `78bfd5fe`.

The seven-view future catalog is independently approved as references only: `future-reviewed-context-library-20261014-v1/independent-review-v1.json`, SHAb050634f027e940db10591b9a62b637b549face52cccf29a3875dd6a1facb16a; root verified 212 pins. Five existing source changes; zero admissions. New older MLX backfill captured 100 history entries and three full changes/four files in `older-mlx-batch6-v1`. Two attention changes await curation; exact erf634b148dd420 is already protected and held. No active data changes.

Studio headroom later passed unchanged 72 GiB startup gate (78,851,948,544 bytes available). Actual `short-serving-operation-v2` session 30645 then failed its first absent-adapter phase with `GenerationMetadata`; zero comparisons completed. Native baseline requires `adapter_dir:null` but actual frontend omits the field. Failure/engine/memory/lease logs are preserved. Independently reviewed `short-serving-owned-recovery-v2` released only this task's terminal owned lease, with no signals; `release-result.json` confirms cleanup. Do not repeat release or restart failed v2 outputs.

One-line baseline omitted-or-null compatibility repair is isolated under `short-serving-native-v2`; nonnull baseline fields still reject and adapter validation remains unchanged. Build-v7 source/plan review SHA bb86bff4e8f8d8cbd4bf658788ade44da7c915aa0b6ecf2d1fcb1fffb26a0735, 1,800 pins root verified. Isolated CPU compiler started 13:59 UTC / session 48194 under `cached-inference-host-build-v7`. Check that exact handle/`execution-receipt.json` before any next build. Successful link still needs artifact review, new runtime/request/controller bindings and fresh memory/lease checks; no serving retry authorized by compilation alone.

Qwen selected-attention CPU extension is held under `qwen-next/extensions/qsa-attention-cpu-v1`,23 pins / 12 new tests plus 9 selector tests. Independent review is queued. No native rounding, backward, complete model or 16K claim.

## Continuation — 2026-10-14 15:21 UTC

Latest read-only M5 observation: saved **119/458**, worker47890 live, elapsed10:41:52, CPU454:22.01, RSS18,542,176KiB; layer14 then16/40 logs advanced. Current dataset and runtime-v5 remain unchanged. Independent exposure audit remains101, not119.

Next evaluation target is **update150**, before the final458 evaluation. A concrete CPU-only capture/subset/matched-rank4 plan is being prepared in a new namespace by quality_research and must receive independent review. Checkpoint150 does not exist yet; no capture or evaluation has launched. With keep2 checkpoint retention, preserve the exact complete checkpoint before rotation using a reviewed transfer plan. Use outcome-independent held-out selection, fresh zeroB and trained150 processes, separate code/rationale/EOS metrics, and unchanged final all16 evaluation. A 32GiB process/32GiB reserve profile is a guarded candidate supported by prior observed allocations, not a guaranteed future fit. M5 stays on training; Studio does evaluation only after actual memory and ownership preflight.

The newly collected MLX attention precision change4c86c1e55adc now has **exact offline before/after compilation evidence**: both historical steel_attention.metal translation units compile,18 explicit kernel names each, FP16/BF16/FP32. Independent actual-artifact review616197d8f82fd99cb7aa14a8d631005c5e66c8f0a52caf66935ed2898ce001ae; root rehashed all118 input pins. See older-mlx-batch6-precision-closure-v1/root-compile-tier-confirmation-v1.json. No GPU/pipeline/numerical/performance reproduction or training admission follows. Two-pass attention positive remains held because reviewed source witnesses wrong query addressing for a supported transposed layout; diagnostic task is source-only.

Native serving build-v7 completed and artifact/runtime reviews passed. A fresh operation-v3 failed its preclaim memory check **before operation-start, lease or native child**; it needs no cleanup. Old operation-v2 owned lease was already released after its processes terminated. These failures are in generation/evaluation, not active training, corpus analysis or Qwen. Corrected fixed16K source memory assembly is34.459439GiB plus64MiB plugin margin, with additional scheduler headroom and separate context256 construction arithmetic. Do not use the earlier33.2247GiB generation estimate: it wrongly removed retained prompt embeddings. Proposed bounded allocator-cache/prefetch changes remain isolated; source/metadata CPU plan is under review. Unrelated sfideas/WatchAI/worldpolicies remain running.

Qwen reviewed selected-attention forward CPU references are committed locally as9430844f;22 reference tests plus1 source-binding test passed. A separate FP64 fixed-route attention adjoint extension passed17 author and8 independent checks, but is not yet integrated into production. Neither establishes full-model, native-precision, recurrent backward or16K training support.

Cloud acceleration was discussed, not provisioned. A CUDA numerical/throughput qualification is required for this exact checkpoint and adapter layout; Apple MLX code cannot simply be run on NVIDIA. Streaming or final-block training from compatible cached frozen-prefix states may avoid entire-checkpoint GPU residency; no measured cloud fit or speedup is claimed. No paid resource, model upload or new campaign was started.

## Continuation — 2026-10-14 15:47 UTC

Actual checkpoint150 CPU watcher was launched on M5 at15:29:55UTC, PID78277, while training was at120. Exact reviewed script/contract hashes and canonical fresh namespace were checked; actual remote receipt is intermediate-evaluation150-v1/watcher150-launch.json, local copy root-watcher-launch.json. Latest subsequent ps observation verified both47890 and78277 live: saved123, training backward logs advancing, watcher RSS21,360KiB. Do not launch a duplicate watcher or training job. Require completed capture.json/allfivefilehashes before transferring or deriving control.

Intermediate evaluation preparation is independently approved as preparation only: review8d4e666dbdc67a835381bdf9ed305d4b470514e4e64256174932d1b690933f8a/170pins rootrehashed. Four outcome-independent length-stratified heldout rows:560445bf34c8,3347e6d90468,27ad57a69b85,793c0d7f4638;29,071 IDs/11,077targets. Two bounded and two conditional historical-target references remain included/separatelyreported. Five rationale targets cannot establish general reasoning quality. execution-v1 binder/controller is being implemented with matchedrank4/alpha8 zeroB derived from actual150snapshot; it is not yet launchapproved. Final458/all16 unchanged.

New batch7 captured100older MLX commits and two full source changes: copy116f is CUDAonly/H100 and excluded from Metal intake; attention47d250 is a Metal large-mask address correctness fix with primary issue2894. Exactagentedit4300/719 and critique2171/375 passed independent source/context/frozen serialization/complete3filepatch replay review fefa718201fae8499971dede771a01a45f80b3fd917b57521b640f9c68c3c5ee/49pins rootrehashed. Source-only conditional wide-arithmetic explanation; no device reproduction or speedup. The two views are one source/attention family. Futurecatalogv3 has11views/7recordedrepo-commitIDs/0admissions; catalogreviewpending, originalcatalogsunchanged. Ordinarybefore/after exactcompilationclosure captured34fileinstances in older-mlx-batch7-attention-closure-v1; compiler/sourceplanreviewpending, no compile yet and NAXoutsidecompilation scope.

Actual CPU construction probe passed and independent result review4cc6808aba536475ef44f0282f533c8cdc464789a24f9837d63d0114e2074711/152pins rootrehashed. Proposed cache256MiB/prefetchdepth1: context256 floor27.595549368GiB, source-derived nativeavailable threshold67.595549368GiB; separate fixed16K load+scheduler39.837181413GiB. This is source-derived metadata/phase arithmetic, not actual allocation or fullBank.open/arm.planRows/row-fill validation. Originalruntime preserved. Isolatedprofile source/buildplan staged; no build or GPUlaunch yet. Process40/reserve32/box96/wired8 unchanged.

Read-only live Studio vm_stat/sysctl sample15:42UTC: nativeformula84.797GiB available, outerfree+inactive+spec73.725GiB. Different accounting formulas must not be confused; unchanged72GiB outergate passed only at that observation with1.725GiBmargin. Freshactualstartguards/watchdog required; do not treat the sample as fit or launchauthority. Preserve sfideas/WatchAI/worldpolicies.


## Continuation — 2026-10-14 16:50 UTC

Fresh actual SSH verified M5 worker47890 and checkpoint150 watcher78277 live. Saved140/458, elapsed12:13:12, CPU518:04.07, RSS19,037,344KiB. Exact150 capture directory does not exist yet. No restart or duplicate job. Independent exposure audit remains101; current live progress is not a new exposure audit. Four-row matched-zeroB/150 controller execution-v2 passed independent CPU review8eb35ab493b50c11e8d2cda57a0524ef6003d8f51f367f6c6fc7217afd5045d5. Actual capture, transfer, binding, concrete launch review and GPU evaluation remain pending. Keep Studio available for this evaluation. Final458/all16 and active runtime-v5/dataset unchanged.

The reduced-profile serving diagnostic actually loaded the model and completed33-token prefill without an adapter, then failed generation with AdapterUnsupportedModuleRoute. Source diagnosis: phaseChange fences the device embedding into host rows, while exact-generation checkGenerationState rejects that fenced route. This is an integration lifecycle bug in isolated mlx-serve linked to patched mlx-stream, not a new inference engine, OOM, or M5 training failure. Partial129,280 F32 prefill logits are finite; not base parity or generation-quality evidence.99 advisory samples passed, max aggregateRSS34.951GiB/min outeravailable37.626GiB; reported native phase footprint~41.99GiB prevents claiming hard40GiB physical fit. Failure preserved. Independently reviewed owned recovery released only this task's terminal reservation at16:35:16UTC; no signals or unrelated changes. Never repeat that cleanup. Retained-device-embedding repair is staged; independent lifecycle/ownership review and revised CPU Bill/phase bounds must precede a new build or GPU request.

Reusable evaluation receipt snapshot safeguards were applied and committed a7c5c59d after45 actual CPU pytest cases passed. Fixed-route Qwen attention adjoint reference applied and committed d9159854 after43 actual unittest methods passed. These are tool improvements; neither changes runtime-v5 or establishes Qwen full-model/native-backward/16K support.

Future context catalogv3 is independently reviewed:11views/7recorded source commit identities/0training admissions. Ordinary historical attention47d250 before/after translation units actually compiled offline; independent artifact review6a1c2a00ef719f74d28c018f11aa8269f81ed90b8ef92fa30f407096803c7e6d. NAX, numerical parity and performance remain unvalidated; compile evidence is not training admission.
