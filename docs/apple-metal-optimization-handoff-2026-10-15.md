# Apple EXL3 Metal optimization fine-tuning: handoff

Status: end of October 15, 2026, America/Los_Angeles. Some receipts are dated October 16 UTC. This document supersedes the live-job instructions in the October 14 handoff and old heartbeat messages. Preserve those historical documents as evidence; do not execute their queues.

## User goal and immediate priority

Build reusable Apple-silicon EXL3 fine-tuning tools and an adapter that generates **correct, faster Metal kernels and makes useful optimization decisions from measured profiles**. The user wants the best useful adapter soon, with practical generated-code evaluation. Lower training loss, successful updates, elaborate receipts, and plausible explanations are insufficient outcomes.

The user is frustrated after days of infrastructure work and experiments concentrated on the same small adapter. They explicitly requested this handoff for another agent to move the work forward. Do not spend the next session repeating a learning-rate sweep or another epoch of essentially the same setup. Change a meaningful learning hypothesis and evaluate actual generated code early.

Preferences and constraints that remain in force:

- Prefer cache-compatible experiments when their expected usefulness is comparable. Measure actual cache eligibility; never defeat cache identity checks.
- Use M5 for training and useful independent kernel profiling; Studio for adapter evaluation. Divide work between hosts when useful, with one admitted GPU job per host.
- **Source collection is paused. Qwen is parked. Recurring automation notices are paused. Keep them that way.** Existing-library validation/profiling is distinct from fetching more repositories.
- Keep sfideas, WatchAI, and unrelated workloads running. Do not delete models, lower reserves, or stop unrelated processes.
- No new model downloads, cloud provisioning, or unrelated campaigns are needed for this handoff.
- “38 or so” meant the profile-focused subset plus the full corpus. It did **not** authorize layer 38 or identify a verified 38-item profile dataset. That ambiguity is resolved; do not ask again or use it to hold work.

## Current state: completed, not running

The full-corpus continuation and all its optimization evaluations are **complete and independently audited**. No unfinished training/evaluation job is handed over. The Studio shared lease was observed free after the last benchmark; that is historical evidence, not a reservation. Inspect actual ownership and process liveness before any new launch.

Product repository:

```text
Workspace: /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments
Repository: <workspace>/publish-exl3
Branch: apple-metal-training-m5
Latest evaluation commit: 72dbde57
```

Read these first:

1. [Latest evaluation and decision](../apple/reports/full-corpus-profile238/README.md).
2. [Machine-readable results](../apple/reports/full-corpus-profile238/evaluation-result.json).
3. [Active experiment plan](apple-cache-aware-experiment-plan.md), especially its completed full-corpus continuation section.
4. [Profile-guided training design and importer limitations](apple-profile-guided-kernel-training.md).
5. [Evaluation framework and historical failures](apple-generated-kernel-evaluation.md).

Detailed workspace artifacts live outside the product Git repository. Define:

```text
V = <workspace>/kernel-library/next-round-v5
F = <V>/full229-profile238-run-v1
S = <V>/studio-functional-panel-v1
R = <S>/row_rms-v2
```

The mirrored status files are `<V>/m5-successor/current-run.json` and `apple/reports/m5-successor/current-run.json`. Read **active_authorized_run** for the latest completed run. Older top-level and `active_authorized_pilot` fields retain historical states and are not current launch instructions.

## Adapter lineage and exact scope

| Endpoint | Training | Relationship to latest adapter |
|---|---|---|
| original458 | Two epochs over 229 items, 458 updates | Parent and reference |
| full238 / step-000238 | One additional 238-update pass: original 229 rows plus 5/3/1 exposures of three profile records | Current challenger; 696 updates in this lineage |
| 188-update, code-suffix152, targeted-profile pilot, time-weighted24/48 | Separate experiments | Their weights are **not** incorporated into full238 |

Full238 used original458 weights via `--resume-adapter`, fresh Adam, step/cursor zero, LR 2.5e-5, seed 8, rank 4, alpha 8, layer 39 only. It did not resume optimizer state from a failed pilot. Full details are in `<F>/main-request.json` and captured `main/run_config.json`.

The actual trained tensors are:

```text
blocks.39.attn.linears.wq_a.lora_a  [5120, 4]
blocks.39.attn.linears.wq_a.lora_b  [4, 1280]
blocks.39.attn.linears.wo_b.lora_a  [8192, 4]
blocks.39.attn.linears.wo_b.lora_b  [4, 5120]
```

That is **78,848 trainable parameters**, two attention projections in the final DeepSeek block. It is not all attention projections, multiple layers, routed experts, or full-model training. Limited capacity is a plausible bottleneck, not an experimentally established explanation for every quality result.

The latest dataset has 238 presentations, 232 distinct record variants, 1,151,731 input-token presentations, 439,708 supervised targets, and maximum serialized length 12,173. It contains only **three new profile-focused records from two already represented lineages**, one speed-qualified lineage. Repetition does not create new independent examples. The profile presentations are about 3.8% of updates. The original 229 already includes the former dev41; never label dev41 unseen. Protected16 and current generated evaluation material stay excluded.

Full238 adapter identity:

```text
Payload SHA256: 0a7952fade3e4cc387509f1cf3e3623651b86f2a17d9a4c356c3771e435ca876
Config SHA256:  13bb4563ebb045a97e89df875340daad3d4d3564988588610511ba0c152fee37
Local: <F>/main-capture-v1/files/main/step-000238
M5: /Volumes/Scratch/exl3-training/2026-10-16/full229-profile238-run-v1/main/step-000238
```

Original458 on M5:

```text
/Volumes/Scratch/exl3-training/2026-10-13/metal-mix229-internal-split-v1/attempts/attempt-1791952639336825000/train/step-000458
```

Find the accepted local original458 reference through the existing evaluation registrations; preserve its exact base/factor bindings rather than reconstructing one from a similarly named checkpoint.

## What the latest evaluation actually showed

Four deterministic complete-file generations: row RMS statistic, strided transpose, and rotation with profile present/omitted. All ended with natural EOS, compiled unchanged, and passed all **260 candidate numerical cases**: 224 RMS plus 12 each for the other three conditions. Corresponding baseline checks passed. Four audited timings completed 336,000 dispatches.

The following figures are **ratios of summed measured baseline/candidate GPU durations** across 12 workloads with equal sample counts. They are speedups over each run’s supplied kernel, not factors by which one adapter beats another.

| Evaluation | original458 | full238 |
|---|---:|---:|
| Row RMS statistic | 10.908× | 12.310× |
| Strided transpose | 2.264× | 2.260× |
| Rotation, profile supplied | 2.297× | 2.310× |
| Rotation, profile omitted | 2.343× | 2.324× |

RMS computes `sqrt(mean(x²) + epsilon)`, one output per row. It is **not full elementwise RMSNorm**. Full238 uses an explicit 32-lane shuffle tree; original458 uses `simd_sum`. New candidate median latency was lower on 9/12 workloads, higher on 3/12 by approximately 0.5–1.25%. Its within-run paired median speedups ranged 5.112–13.606×; original458 ranged 5.062–12.194×. This is a positive RMS signal. Separate measurement windows and baseline drift limit attribution.

Rotation and transpose retain the same guarded scalar XY16×16 mechanism as original458. No additional advantage is demonstrated. Rotation chooses essentially the same optimization with or without the real baseline profile. The transpose explanation falsely claims neighboring x-lanes write contiguous output; they write at the output stride. Preserve that reasoning failure separately from passing code correctness.

**Decision:** keep full238 as a promising RMS challenger and original458 as the comparison reference. There is no broad replacement winner or demonstrated profile-guided reasoning improvement. These are repeated development tasks across three families, not an unseen benchmark. Confirmation remains sealed.

The user’s practical standard does not require perfection or winning every workload: seek more correct, faster useful outputs across different tasks, without serious regressions, and grounded decisions when profiles matter. Do not turn uncertainty into endless baseline reruns or avoid choosing a useful next experiment.

## Recommended next work: meaningful experiments

This handoff does not launch another campaign. Use the user’s ongoing mandate for useful fine-tuning work, existing conditional capacity work, and fresh actual evidence. The following are priorities, not implemented capabilities or promised runtimes.

### 1. Make the existing rank-8 route evaluable, then test greater capacity

Start here because it changes capacity while retaining the final-block frozen boundary and potentially existing training prefixes. Training supports a rank argument; native rank-8 serving is **not yet hardware-qualified**.

Existing isolated work:

```text
<V>/exact-generation-qualification-v1/attention-rank8-serving-extension-v1
```

Its linked artifact has passed static review (`80ea4e31…be7d17c`), but source/build success is not device qualification. Read `qualification-plan-v1.json` and the artifact review. Finish the bounded intended-route residual/zero-B qualification. The plan uses small model-free rank 4/rank 8 forward cases and a short same-build absent/zero-B control; do not restart a full41 null baseline or pursue perfect trainer/native cross-mode bitwise equivalence as a launch prerequisite. Preserve the qualified 800b rank-4 engine.

A practical candidate is rank 8/alpha 16 at boundary 39. If expanding a rank-4 adapter, explicitly implement and test the orientation/padding/scaling mapping; alpha/rank remains 2, but a valid schema alone does not establish equivalent predictions. Extra components with zero B can preserve the intended initial function; validate the arithmetic contract. Record the chosen parent and fresh optimizer. A directly expanded rank-4 adapter is not a fresh random rank-8 baseline.

Use a bounded end-to-end train/save/reload/generate pilot first, then a useful training run and actual code tests. Do not spend hours training an adapter that the serving path cannot apply. Rank 8 doubles this adapter’s parameter count; it is still a small intervention, so treat success as a hypothesis.

### 2. Increase useful profile-decision coverage from the existing library

Use M5 for measurements while Studio evaluates. The speed-qualified training pair is row48 Q6_K decode reuse: tiny workloads regressed while larger workloads improved. All its variants remain one lineage. Row94 is restricted precision evidence for three half-involving cases; the float/float case did not pass and must not be relabeled successful. Read `<V>/correctness-pilot-inventory-v1` and the reviewed measured drafts/receipts.

Validate additional feasible independent optimization pairs already collected. Examples should contain baseline code, real workload/precision/dispatch context and baseline measurements, a concise grounded decision, complete correct code, and actual feedback where used. Include workload-dependent regressions/retain-baseline decisions. Do not fabricate profiler counters or put future optimized timings into an initial proposal prompt. The preserved row48 draft-v1 leaked after-results; its corrected v2 is the usable precedent.

This should genuinely change the learning content, rather than repeat the same three profile records more often. Report real lineage counts and gaps. Do not pad a quota or block all progress on an arbitrary large quota. Keep evaluation lineages and confirmation material out of training. Source collection remains paused.

### 3. Assess broader final-block adapter coverage before paying for layer 38

More final-block projections may offer a larger change than another LR tweak while preserving the mathematical boundary 39 input. Current trainer/export/native support for additional targets must be inventoried and implemented; do not claim it exists. Treat shared/routed expert coverage separately and count parameters/optimizer state before selecting it.

Adding layer 38 moves the frozen boundary and requires different cached states, earlier backward, serving support, and a measured cold-cost/fit pilot. It is a legitimate later capacity experiment, not automatically equivalent in cost to cached boundary 39 runs.

For any candidate, start actual generated-code evaluation early. Do not make another full41 zero-B score, repeated baseline training, perfect controls, or a new inference engine the default prerequisite. Reuse accepted references when the full protocol matches. Expand the practical task panel when a specific limitation requires it; a separately declared equal-budget feedback mode can test whether compiler/numerical/timing feedback helps. Do not open sealed confirmation automatically.

## Machines, runtimes, caches, and launch discipline

M5 host: `jason-m5-MacBook.local` (Apple M5 Max, 40 GPU cores, 128 GB). Remote hostname in receipts: `Jason-M5-MacBook.localdomain`. Storage under `/Volumes/Scratch/exl3-training`; the model itself is on internal storage:

```text
/Users/jasontitus/.mlx-serve/models/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw
```

Pinned training runtime and lease wrapper:

```text
/Volumes/Scratch/exl3-training/2026-10-13/runtime-v5
/Volumes/Scratch/exl3-training/2026-10-13/runtime-v5/.venv/bin/python
/Volumes/Scratch/exl3-training/2026-10-13/shared_gpu_lease_v2.py
```

Keep runtime-v5 unchanged. Use an isolated runtime for source changes. Every MLX training/preflight/evaluation launch must bind `MLX_ENABLE_TF32=0`. Existing M5 training profile is **80 GiB process / 8 GiB reserve**, packed cache 6144 MiB, decoded 0, prefix RAM 1024 MiB, allocator 256 MiB, chunk 2048. These are reviewed operating settings, not hard OS partitions. Diagnose actual availability/ownership instead of assuming a fixed amount of free RAM.

Existing prefix disk cache:

```text
/Volumes/Scratch/exl3-training/2026-10-13/metal-mix229-internal-split-v1/prefix-disk-cache
```

It has a 64 GiB quota, about 47.7 GiB stored at the latest run. Full238 took approximately 97 minutes end to end, peak MLX 70.331 GiB, with 238 training prefix-hit log entries plus 2 reload hits. Aggregate disk hits 239, zero misses/writes. Do not describe all 239 as training-only hits.

Cache eligibility binds exact worker/runtime/model/arithmetic, complete tokens, train/eval mode, state layout, boundary and chunk behavior. New text needs new prefixes. Changing runtime source can invalidate implementation bindings even when the intended frozen math is unchanged. A versioned, reviewed compatibility mechanism is needed for any reuse across such changes; never relabel a cache or remove its checks. Rank changes under unchanged runtime may be eligible, subject to actual worker validation.

Studio: Apple M1 Ultra, 128 GB. Local model:

```text
/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw
```

Qualified native rank-4 engine:

```text
<V>/exact-generation-qualification-v1/runtime-package-full-prompt-f32-v1/runtime/host/bin/mlx-serve
```

This is the accepted 800b artifact, layer 39/rank 4/alpha 8 support. It is a private qualified adapter route; exported training checkpoints are not automatically portable to arbitrary mlx-serve/Sushi installations. Avoid building another engine when extending this route suffices.

Studio native serving uses its reviewed **50 GiB process / 104 GiB native box / 32 GiB reserve** profile and clean environments. Small kernel jobs use **2 GiB process / 32 GiB reserve**, plus their concrete tracked-device bounds/deadlines. Do not confuse these profiles with M5 MLX training. Preserve native arithmetic/environment bindings.

Read-only ownership commands, to be used with actual permissions and current host state:

```sh
# M5
ssh jason-m5-MacBook.local '/Volumes/Scratch/exl3-training/2026-10-13/runtime-v5/.venv/bin/python /Volumes/Scratch/exl3-training/2026-10-13/shared_gpu_lease_v2.py status'

# Studio, from the workspace
/Users/jasontitus/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12 kernel-library/next-round-v5/two-host-split-v1/shared_gpu_lease_studio.py status
```

Check full process identity and waiters before launch. Never duplicate a GPU job or bypass another owner. A serving lease is process coordination, not part of the training dataset or model math. Operational failures are not numerical model failures or successes. Preserve failed receipts and diagnose concrete causes.

## Reusable tooling and current limitations

- Product source: `apple/src/exl3_mlx`; use the existing workflow/preparation/training/export/evaluation components. Pinned runtime-v5 is a separate frozen copy.
- [Weighted prepared-data mixing](apple-weighted-dataset-mixing.md), commit `db05057f`: `exl3-mlx mix` preserves explicit repeats, token rows and masks, producing ordinary train-only data. Normal `prepare` deduplicates; passing a weighted mixture through it again erases deliberate repetition. This command is in product code, not installed into runtime-v5.
- [Full-corpus recipe](apple-full-corpus-profile-recipe.md) and `<V>/full229-profile238-candidate-v1` preserve the exact accepted data. Its frozen manifest says “candidate not launched”; current completion/evaluation receipts supersede that historical status.
- The importer supports last-assistant SFT with a single `loss_start`. Native multi-tool observations/all-assistant multi-span targets are not supported merely by changing JSONL. Use clearly labeled feedback as context for one next assistant decision, or implement/test a new serializer separately.
- Standard chat/prompt-completion and complete-file outputs are supported. Raw git diffs alone are weak evidence for profile-driven choices. Preserve host/dispatch/build dependencies and actual correctness/performance status.
- Source/license provenance is retained, including gaps; collection or compile success is not permission to redistribute arbitrary source or proof of training quality.

## Evaluation artifacts to reuse, not rerun

Latest generations: `<F>/evaluation-prep-v1/profile_present`, `profile_omitted`, `<F>/rms-evaluation-prep-v1`, `<F>/transpose-evaluation-prep-v1`. Each contains exact requests, prompts, factor bindings, raw output, actual audit and source acceptance. New adapter generation must use a fresh condition/output, not overwrite these.

Rotation numerical/timing and all-workload comparison: `<F>/candidate-evaluation-prep-v1` and `<F>/rotation-comparison-v1`.

RMS: `<R>/candidate-fullfiles-v4`, `candidate-controller-fullfiles-full238-v1`, `candidate-timing-fullfiles-v4`, `candidate-timing-controller-full238-v1`.

Transpose: `<S>/strided-transpose-candidate-v3` and `strided-transpose-candidate-timing-v3`, each `full238-*` build/operation. Their actual reviews and comparison retain all shapes and regressions.

The policy is fixed per existing comparison: one greedy attempt per condition, up to 4096 generated tokens within 8192 context, natural EOS required. Rotation prompts are 2128/1263 tokens; RMS 702; transpose 789. The common timing policy is 5 warmup pairs, 30 measured pairs, 100 dispatches per batch across 12 workloads, 840 batches/84,000 dispatches per condition. Compile/source/ABI/numerical acceptance precedes timing. Never execute generated shell or repair a candidate into a strict success.

Preserve all old affine failures, completed 229/458, 188, 152 runs, targeted pilots, v2 step24/48 results, RMS/transpose/rotation controls and historical serving diagnostics. Do not restart their transfers, qualifications, generations, or evaluations. Repeat a check only for a concrete new change, failure, or unresolved concern.

## Operational lessons: avoid repeating the time sink

Latest evaluation preparation caught stale provenance and condition bindings. Frozen prompt metadata pinned the product `data.py` before the mixing tool’s lazy-import change. The repair archives the **exact historical bytes**, changes the provenance reference only, and preserves frozen prompt IDs/raw prompts. Files are under `<F>/evaluation-prep-v1/prompt-source-provenance-v1`. Do not retokenize matching references or weaken a hash check to conceal drift.

Inherited helpers also had directory-depth assumptions and condition allowlists that omitted full238. Receipt schemas differed between detailed audits and actual consumers. The failed versions/logs are preserved; corrected siblings passed the real composed CPU validators before launch. A copied source file passing import tests is not proof that its complete request works.

Future tool work should parameterize adapter conditions, stable provenance and endpoint binding in the reusable framework, rather than growing another family of handwritten near-identical controllers. Preserve the existing generic host/math/ownership code. Reuse reviewed unchanged components; focus independent reviews on actual changes and consequential outcomes. Do not make thousands of pin counts or repeated whole-suite tests the project’s progress measure.

## Qwen and source-library status

Qwen is parked at `<workspace>/qwen-next/project.json`. CPU storage/decoder/architecture references and some small synthetic native tests exist. Real Qwen checkpoint training, full 16K sparse backward, broad recurrent backward and production serving are not qualified. Read its project/design documents before making any capability claim. Do not resume it while the current priority is useful DeepSeek Metal fine-tuning.

The library tracks Sushi, TensorFold, mlx-serve, ds4, llama.cpp, omlx, mlx, mlx-lm, mlx-vlm, splash, MTPLX and vllm-metal. Collected commits, reconstructed edits, locally validated kernels and training-ready examples are different tiers. Source collection remains paused; no hourly/daily collector should be re-enabled by this handoff.

## First session checklist

1. Read the latest result and this handoff; inspect actual Git status and host ownership. Do not infer running jobs from historical PIDs or heartbeats.
2. Summarize the next two distinct learning hypotheses in a short concrete plan, then advance the first ready work. Favor rank-8 intended-route qualification plus existing-library profile validation in parallel; wider final-block targets are a separate capacity path. Do not hold on the resolved “38” question.
3. Establish serving/evaluation support and a bounded end-to-end sample before committing hours to a new adapter. Record actual cache hits, cold costs and memory. Preserve earlier adapters.
4. Judge generated kernels and grounded decisions. Report correctness failures and every workload regression, rather than calling lower NLL or format success a win.
5. Put reusable fixes in product code, document actual results, and commit. Keep recurring notices/source collection paused. No automatic duplicate training or confirmation run.

The latest product evaluation is independently publication-reviewed (`386b7bf0…6ddde`) and committed as `72dbde57`. This handoff records priorities and evidence; higher-capacity training and better profile coverage are proposed next experiments, not capabilities already proved.
