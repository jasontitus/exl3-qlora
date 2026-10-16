# Apple EXL3 Metal optimization fine-tuning: handoff

Status: end of October 15, 2026, America/Los_Angeles, plus the October 16 UTC state section below. Some receipts are dated October 16 UTC. This document supersedes the live-job instructions in the October 14 handoff and old heartbeat messages. Preserve those historical documents as evidence; do not execute their queues.

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

## Current state: the full238 lineage is completed and not running

The full-corpus continuation and all its optimization evaluations are **complete and independently audited**. No unfinished training/evaluation job from that lineage is handed over. The Studio shared lease was observed free after the last benchmark; that is historical evidence, not a reservation. Inspect actual ownership and process liveness before any new launch.

**Since that state was written, both hosts have run further work and are now idle.** The M5 ran the Gate 5 rank-8 full-corpus continuation (`rank8-full-corpus-run-v1`, 238 updates) under the shared lease from `2026-10-16T17:31:28Z` to `19:06:02Z` and released it; the row-98 split-K precision pilot's confirmation then took that same lease for 6.0 s at `19:07:04Z` and released it. Both are complete and recorded in the October 16 section below. On the Studio, the rank-8 adapter's generation evaluation is prepared, sealed and queued behind its reviewed occupancy window with a live waiter (item 4) — prepared and queued, **not** running.

Product repository:

```text
Workspace: /Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments
Repository: <workspace>/publish-exl3
Branch: apple-metal-training-m5
Latest evaluation commit: 72dbde57
```

Publication: the handoff and completed work are pushed to `https://github.com/jasontitus/exl3-qlora`, branch `apple-metal-training-m5`. This writable fork was created for the user-requested push; `UnstableLlama/exl3-qlora` is read-only for the authenticated account. Local `origin` points to `/Users/jasontitus/experiments/exl3-qlora`; use the writable fork for GitHub publication, preserving the local origin.


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

## October 16 UTC update: the rank-8 capacity path

This section is the current state. It does not change the priorities above; it records what has actually been observed since they were written. **The generation half now has a trained-adapter result.** Everything below is a training-side pass, an executed native control, or a recorded refusal together with its resolution.

**1. Rank-8 end-to-end pilot — both halves passed (generation half: item 2).** `<V>/rank8-end-to-end-pilot-v1/pilot-result.json` (host `jason-m5-MacBook.local`, observed `2026-10-16T12:55:00Z`) records the bounded train/save/reload pilot on the M5. All three pre-registered acceptance criteria passed: 8 of 8 requested steps completed (`cursor` 8), 157,696 trainable parameters, and the trainer's own post-save check logged `Adapter prediction reload is exact`. The adapter is a **fresh** rank-8 (rank 8, alpha 16, layer 39 only, fresh Adam, LR 2.5e-5, seed 8, no parent and no resumed optimizer — it is not an expansion of the accepted rank-4 adapter), 631,212 bytes, sha256 `113ffa27…b705ef3`, transferred into the v4 lane below. Per-step loss is recorded for all eight updates, but each step is a different single sequence, so the series is not a curve and **no loss-trend claim is made**. The file's own `limits` list states what it does not cover; `generation_half` in the same file still reads `blocked_on_studio_memory_admission`, which was true when it was written and is superseded by item 2's executed result.

**2. The generation half ran and passed both pre-registered criteria.** It finished `2026-10-16T15:28:28Z` with `status: rank8_trained_adapter_native_control_gate_4b_accepted`, after the armed waiter found the box free at `15:25:52Z` and ran `operation.py --preclaim` then `--execute` once. `r8trained-zerob` is bitwise equal to `absent` (both raw logits sha256 `266166f5…`), and `r8trained` differs from `absent` (`b53820c5…`) while the greedy argmax stays 553 and the prefix ids are unchanged in all three rows — so the trained weights do take effect on the native route, with the widening still numerically inert. The lane is a complete, gate-cleared three-case control, not a stub. `<V>/exact-generation-qualification-v1/rank8-trained-adapter-native-control-v4/` holds the three-case native control (`absent` / `r8trained-zerob` / `r8trained`) at position 33 on prompt 0033, run by the widened rank-8 native serving package. `plan.json` (sha256 `1e57d096…`, status `reviewed_pending_independent_source_review`) carries the static independent source review, the host/build context, the license record, and the exact regression assertions that review required. Gate 4b is pre-registered in the plan: `r8trained-zerob` must be bitwise equal to `absent` at position 33, `r8trained` must differ from `absent` (the trained weights must actually take effect), and `absent` must still reproduce the pre-widening engine's published row `266166f5…` bitwise. The plan states that a difference in the first comparison is a retained finding, not something to hide.

**3. Admission passed through the armed waiter; the refusal it replaced is preserved.** `preclaim-observations.json` in that lane records the original refusal: the reviewed `40/80/32/128` GiB memory profile requires 72 GiB available at startup, and the Studio was showing 37–46 GiB. That refusal stood until `2026-10-16T15:25:52Z`, when the waiter described below found the box free and ran the operation once; the preclaim then reported 117.24 GB outer-available and 113.78 GB native-available against the 77.31 GB outer floor and the 89.76 GB native requirement. The identified holder is **operator-owned and not this lane** — `screen` sessions `bonsai-lora-*` running `mlx_align.py train` under `/Users/jasontitus/experiments/worldpolicies`. No process was signalled, no memory profile was edited, and no admission check was bypassed. `rank8-trained-adapter-native-control-v4-admission/wait_for_admission.py` was armed and detached (ppid 1) and has now completed its single admitted run: it polls the same two read-only predicates, additionally requires that no foreign MLX training process is running, and then runs `--preclaim` and `--execute` once. If admission never passes inside its window, no device work happens and every poll is logged to `admission-wait.jsonl`. The operation remains the only thing that can start the engine, and it re-checks admission internally.

**4. The rank-8 full-corpus adapter is trained, captured, served, and its evaluation is queued behind the Studio occupancy window.** `<V>/rank8-full-corpus-run-v1/README.md` records the whole chain.

- Training: a **fresh** rank-8 (rank 8, alpha 16, layer 39, no `--resume-adapter`, fresh Adam, step/cursor 0, LR 2.5e-5, seed 8) over the byte-identical 238-row payload (`sha256 bce31595…f5914`; 1,151,731 input tokens, 439,708 supervised, max serialized 12,173) on the M5, `2026-10-16T17:31:27Z` → `19:06:02Z`: 238/238 steps, `status completed`, `exit_code 0`, `lease.ownership released`, `cleanup_verified true`, `processes_signalled false`, `foreign_objects_removed false`, completion sha256 `2b2b1cd6…1af88`, 5,491.75 s, peak MLX 70.32 GiB, final loss `0.10130806267261505` (gradient norm `0.03996831551194191`), and the trainer's own post-save check logged `Adapter prediction reload is exact`. The only deltas from the accepted full238 run are rank 4 → 8, alpha 8 → 16 and the removed resume.
- Adapter: `main/step-000238/adapter.safetensors`, 631,212 B, sha256 `f6c6fc74…87af6f`, config sha256 `69da06ad…09d3c`; captured read-only into `main-capture-v1/` (12 files, receipt `copied_closed12_terminal_evidence_pending_quality_audit`), with the captured and served bytes re-reading to the same two digests.
- Served: `<V>/exact-generation-qualification-v1/rank8-full-corpus-adapter-native-control-v5/RESULT.md` — gate 4b **accepted** (`19:14:18Z → 19:16:32Z`, three cases serial). `r8trained-zerob` is bitwise equal to `absent` (129,280/129,280 values, max abs 0.0); `r8trained` differs from `absent` on 129,127/129,280 values (max abs 5.453125, mean 1.35487, greedy argmax unchanged); the `absent` row reproduces the published pre-widening-engine row bit-exactly. Scope: servability and non-zero effect only — not quality, not timing.
- Evaluation: `evaluation-prep-v1/` is prepared and sealed (`author-seal-v1.json`, 50 local + 65 input digests; independent Q review `evaluation-prep-v1/independent-q/review-v1.json`) over the frozen development prompts (2,128 profile-present, then 1,263 profile-omitted), 800b runtime, 50/104/32 resources, 8192 context, one attempt each. **No generation has been launched.** `evaluation-admission-v1/wait_for_admission.py` (armed, pid 83089, 300 s poll) re-verifies the seal at startup and would take both generations once each, in order, and verify both receipts.
- Why it is still queued: the reviewed gate `evaluation-prep-v1/profile_{present,omitted}/preclaim_check.py` is an **occupancy window**, not a headroom ceiling — it passes only while `native_available >= 71.65 GiB` **and** `96 GiB - baseline <= 50 GiB`. Latest of 12 refused polls, `2026-10-16T21:56:43Z`: outer available 93.88 GiB and native available 91.02 GiB (both pass), but native baseline 36.98 GiB puts the process upper bound at 59.02 GiB against the 50 GiB budget → `native source-derived headroom refused`, `status cpu_preclaim_refused_no_lease`. This is the reviewed gate behaving as designed, not a bug: the passing precedent (`<F>/evaluation-prep-v1/profile_present`, the full238 rotation evaluation) passed at baseline 48.61 GiB / upper 47.39 GiB, and `time-weighted-pilot-v2/evaluation-prep-v2` passed at 48.27 / 47.73. Do not weaken the predicate; the lane unblocks when other work occupies 46–56 GiB on the Studio. The waiter (pid 83089, polling every 300 s since `21:26:35Z`) logs every poll to `evaluation-admission-v1/admission-wait.jsonl` and has launched nothing.
- Next after the generations: a reviewed read-only capture lane for the two outputs (`postgeneration-prep-v1/`), then numerical acceptance and the 12-workload paired timing comparison against original458 and full238. A rank-8 quality/timing verdict does not exist yet.

**5. Profile-coverage validation is in flight.** Per the "Minimal next preparation" section of `<V>/correctness-pilot-inventory-v1/RECOMMENDATION.md`, source-verified validation cases are being built for the first candidates (row 98, `mlx` split-K quantized matmul partials; row 96, `mx.block_masked_mm` output mask). Each case is source plus closure plus license plus exact regression assertions with a CPU numerical reproduction of the mechanism. These are explicitly **not** device measurements and **not** performance evidence, and the training-ready count in the inventory remains 0 until they and their reviews land. **Status 2026-10-16:** rows 98 and 96 are past source validation **and now have device before/after evidence**; row 98's M5 confirmation exists and its cross-device verdict is **`not-replicated`**.

- Row 98 (split-K quantized-matmul partials), Studio: `<V>/row98-splitk-precision-pilot-v1/RESULT.md` (M1 Ultra). The parent build is bit-exact against the installed wheel 8/8, `err/round` goes 2.19–2.62 → exactly 1.000, and the doubled intermediate bytes cost +1.8…+5.1% on the split-K dispatch (aggregate, two runs). Limits: reduction step emulated, microbenchmark, no model-level claim.
- Row 98, M5: `<V>/row98-splitk-precision-pilot-v1/m5-confirmation-v1/RESULT.md` (M5 Max, launched `2026-10-16T19:07:04Z`, 6.0 s after the lease was taken only once the training released it). `compare_devices.py` verdict **`not-replicated`** (`checks_failed` 117): the dtype effect (2.19–2.62 → 1.000) and the harness's wheel fidelity replicate; the fp16 CPU-oracle bit-exactness (M5 residual 1 ULP) and the cost direction (Studio +5.1%, M5 −0.01%, M5 2.2–3.2× faster per case) do not. Times are never pooled.
- Row 96 (`mx.block_masked_mm` output-mask stride): `<V>/row96-outmask-stride-pilot-v1/RESULT.md` — each pinned revision's real `gemv_t` kernel (pre-fix parent `789be76e27a3`, corrected `77bf1fa01956`) compiled from one include tree differing in exactly `gemv_masked.h` and dispatched in one process on the Studio. Pre-fix ≠ reference on every strided case (64/128 and 36/100 elements, max abs 14.5–20.3), corrected == reference everywhere (≤3.81e-6 f32 rounding, `atol=1e-5`), and pre-fix == corrected bit-identically on the contiguous control; ~8 µs per dispatch, no speed claim. Reproduced on an independent re-run (mismatch counts and max-abs identical). Limits: Studio only, NumPy oracle, `gemv_t` geometry, microbenchmark; the host `mask_strides` packing is read from the PR base `3c40e8f92d0c`, not the exact parent tree.
- Lineage caution for the coverage goal: rows 96, 97 and 51 are **one lineage component** (`lineage_component_rows: [51, 96, 97]`), so they cannot be counted as independent examples of the same lesson. In the `RECOMMENDATION.md` order, the next *independent* candidates are rows 75, 86 and 121.
- The inventory's "training-ready count: 0" and its "no selected item has independently reproduced numerical or paired performance evidence" line were written before these runs. Row 98 now has a matched device pair with paired timing on two devices; whether that clears the training-ready gate is a curator/review decision, and the frozen inventory file is not edited here.

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

Start here because it changes capacity while retaining the final-block frozen boundary and potentially existing training prefixes. Training supports a rank argument; native rank-8 serving is **hardware-qualified as of October 15, 2026** on the Studio (Apple M1 Ultra, MLX 0.32.3, GPU stream): 26/26 model-free cases passed, the device output is bit-exact against the seam including the intermediate `astype(z*2, dt)` rounding, and an independent verifier recomputed every acceptance from the frozen case file. Evidence: `<V>/exact-generation-qualification-v1/attention-rank8-serving-extension-v1/device-qualification-v1/RESULT.md` (`run-v3`, `cases-v2.json` sha256 `3990621e…f28eda`).

**Executed 2026-10-16 UTC — the rank-8 route is now evaluable.** The gated intended-route control ran once on the Studio (93 s wall; cases 45.6 s / 46.0 s, both `completed`, exit 0). The widened rank-8 engine loading a rank-8 **zero-B** adapter produced raw logits **bitwise identical** to the absent-adapter baseline (`zero_b_bitwise_equal_to_absent: true`) and **reproduced the pre-widening engine's published row** (`reproduces_pre_widening_engine_logits: true`; witness engine `800b8a3f…96d2f`): 129280 values, 0 differing, max absolute error 0.0, argmax 553, position 33, row sha256 `266166f5…f4974fa`, greedy argmax equal. Receipts consistent; same prefix and vocab. A second descriptive control from a different engine build (`report80` rank-4, row `ef651a4e…9e3a2f`) does not match, as expected across builds. Result: `<V>/exact-generation-qualification-v1/rank8-zero-b-native-control-v3/operation-result.json` (`status: rank8_zero_b_native_control_bitwise_reproduced`; plan `2774d9ad…c7598`, review `a609374b…95ac9c` `approved_scoped_operation_source_no_launch`). **Scope: selected-row raw-logit comparison only — not trainer parity, not quality, not timing.** So with zero B the widening is numerically inert, and a trained rank-8 adapter can be evaluated without a widening artifact.

Existing isolated work:

```text
<V>/exact-generation-qualification-v1/attention-rank8-serving-extension-v1
```

Its linked artifact has passed static review (`80ea4e31…be7d17c`). Read `qualification-plan-v1.json` and the artifact review. The bounded intended-route residual/zero-B qualification is finished; do not re-run it without a fresh run directory. Preserve the qualified 800b rank-4 engine.

That qualification is a cautionary result about review: the first case-file revision fused the seam's intermediate rounding into the final one (`round_dt(base + z*2)` instead of `s = round_dt(z*2); y = base + s`), and static review passed it. Only executing against the device found it — the device was right, the reference was wrong, by exactly one bf16 ulp on 9373 of 42240 elements. Reusable rule: a generated reference that implements a **chain of roundings** must be cross-checked against a second, independently written formulation of the same chain, and its error bound must budget a full ulp at every rounding step, not half. Both are now enforced in `reference/generate_cases.py` and `run_device.py`.

A practical candidate is rank 8/alpha 16 at boundary 39. If expanding a rank-4 adapter, explicitly implement and test the orientation/padding/scaling mapping; alpha/rank remains 2, but a valid schema alone does not establish equivalent predictions. Extra components with zero B can preserve the intended initial function; validate the arithmetic contract. Record the chosen parent and fresh optimizer. A directly expanded rank-4 adapter is not a fresh random rank-8 baseline.

Use a bounded end-to-end train/save/reload/generate pilot first, then a useful training run and actual code tests. Do not spend hours training an adapter that the serving path cannot apply. Rank 8 doubles this adapter’s parameter count; it is still a small intervention, so treat success as a hypothesis. **Status 2026-10-16:** the bounded pilot, the intended-route control and the useful rank-8 full-corpus run are all complete, and the trained rank-8 adapter is servable through the qualified route (October 16 section, items 1–4). Do not re-run the pilot, the control, or the training run. The rank-8 generation evaluation is prepared, sealed and queued behind its reviewed Studio occupancy window; the rank-8 quality/timing verdict follows those generations.

### 2. Increase useful profile-decision coverage from the existing library

Use M5 for measurements while Studio evaluates. The speed-qualified training pair is row48 Q6_K decode reuse: tiny workloads regressed while larger workloads improved. All its variants remain one lineage. Row94 is restricted precision evidence for three half-involving cases; the float/float case did not pass and must not be relabeled successful. Read `<V>/correctness-pilot-inventory-v1` and the reviewed measured drafts/receipts.

Validate additional feasible independent optimization pairs already collected. Examples should contain baseline code, real workload/precision/dispatch context and baseline measurements, a concise grounded decision, complete correct code, and actual feedback where used. Include workload-dependent regressions/retain-baseline decisions. Do not fabricate profiler counters or put future optimized timings into an initial proposal prompt. The preserved row48 draft-v1 leaked after-results; its corrected v2 is the usable precedent.

This should genuinely change the learning content, rather than repeat the same three profile records more often. Report real lineage counts and gaps. Do not pad a quota or block all progress on an arbitrary large quota. Keep evaluation lineages and confirmation material out of training. Source collection remains paused.

### 3. Assess broader final-block adapter coverage before paying for layer 38

More final-block projections may offer a larger change than another LR tweak while preserving the mathematical boundary 39 input. Current trainer/export/native support for additional targets must be inventoried and implemented; do not claim it exists. Treat shared/routed expert coverage separately and count parameters/optimizer state before selecting it.

**Inventoried 2026-10-16.** `<V>/final-block-target-coverage-v1/{INVENTORY.md,inventory.json}` — 33 stage/target rows across trainer, export/serving-admission and native. End-to-end support today is exactly `blocks.39.attn.linears.wq_a` and `blocks.39.attn.linears.wo_b` (rank 4 or 8, alpha = 2*rank, `adapter_layers == [39]`, `expert_adapter_layers == []`). Layer 38 is trainer-supported and generically exportable, but blocked by the serving-admission frontend (`rank8-widened-identity-v1/frontend/adapter_generate.py:55-71,86-119`) and by the native loader/model (`lora.zig:89-102,143-155,169-185`; `model.zig:250-252,386-395`), which hardcode four block-39 tensors and forbid adapters on layers 0..38. `wq_b`, `wkv`, `wo_a` and final-block norms are unimplemented in both the trainer and the serving route; routed-expert adapters exist in the trainer only as all 384 experts' `w1`/`w2`/`w3` per selected layer and are rejected by the serving route; shared-expert and selective-expert targets are unimplemented everywhere. No trained layer-38 adapter exists in this effort. So layer 38 with the existing `wq_a`/`wo_b` pair is a serving/native implementation job first, not a training-config change, and the cheapest genuinely new capacity is an additional projection in block 39 (also a serving/native implementation job).

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

Transpose: `<S>/strided-transpose-candidate-v3` and `<S>/strided-transpose-candidate-timing-v3`, each `full238-*` build/operation. Their actual reviews and comparison retain all shapes and regressions.

Rank-8 intended-route control: `<V>/exact-generation-qualification-v1/rank8-zero-b-native-control-v3` (operation result, per-case attempts with raw logits and metadata, wrapper and lease receipts, preclaim observation). Its result is recorded under priority 1; do not re-run it, and any repeat needs a fresh run directory.

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
