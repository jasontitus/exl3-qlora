# EXL3 Apple training and Metal optimization handoff

Snapshot: **2026-10-14 12:02 UTC / 05:02 PDT**. Recheck live processes before taking action. Training is still running on the M5. The Studio checkpoint-50 comparison has finished. It shows better teacher-forced probabilities, but no increase in correct predictions or demonstrated improvement in generated kernels.

The goal remains active: build a larger, well reviewed Metal optimization corpus, finish the authorized DeepSeek fine-tune, and evaluate whether it improves useful coding results. This handoff does not pause or complete that goal.

## Latest verified state

The original 11:18 UTC snapshot is preserved in `HANDOFF-2026-10-14-at-1118UTC.md` and Git commit `201944ab`. This document incorporates subsequent completed operations:

- M5 worker 47890 was confirmed live at 12:01:59 UTC with saved step **79 of 458**. The independent exposure audit remains through 69.
- The exact failed-job lease was released through the reviewed copied-state operation after fresh checks. `controller-v2/recovery-numerical-controls-v1/release-result.json` SHA is `8cf369deb6821e89ca1aa10d9af6ac75ed481261ffadd773e0b3739a41aed0dc`. No signals were sent; original failures remain preserved.
- Fresh full-payload verification and current continuity passed independent review: `exact-generation-qualification-v1/payload-full-verification-independent-v1/review-v1.json`, SHA `5b06b7dcd478fe3f9fc21d9df99326fb9621b29edfffd13ae341c7ab69936c11`.
- Corrected numerical controls actually completed **896 GPU dispatches** and released their lease. Output is `full-kernel-evaluation-v1/controller-posix-v3/numerical-controls-run-v2`. Baseline and golden passed all 224 cases each; wrong-row and zero-output controls failed 120 and 181 cases as expected. Independent actual review SHA `cd862660530ecfb22974fb6495e576c5ccaf6afeca883c08f36cc03e497c6915`; native companion SHA `5ebe52f4a22e74bef801d0ec87a8b61f081c13d2fc02afd4901fb98f95e5712c`. Only fixed controls are qualified; no generated model outputs or timing result exists. Golden passing raw floats were not retained, so its qualification relies on the reviewed native checks and recorded errors/digests, not independently reconstructed passing floats.
- Timing and generated-candidate consumers require separate reviewed POSIX-host and v4-qualification rebindings. New drafts fail closed until timing artifacts are compiled and pinned. Do not use the old v3 qualification hash as a substitute.
- Six actual 33/129-token serving requests pass isolated CPU validation. Their execution is disabled: the old `ea5f...` binary incorrectly reads top-level vocabulary/context fields that actually live under `text_config`. The isolated Python revision also handles the actual adapter's null metadata and declared 183 MiB runtime metallib. The native parser needs integration and a new linked binary.
- The old Python serving guard's misleading `memory_cap_gib` only configures a total native box ceiling; it does not enforce process RSS or system reserve. A separate frontend/guard draft distinguishes proposed **40 GiB process / 96 GiB native box / 32 GiB reserve on 128 GiB physical RAM**. Its selected-term estimate is 31.516 GiB, not a measured peak or fit guarantee. Preserve native 16K load preflight, retained draft residents and the fixed 8 GiB wired margin. Bundle: `exact-generation-qualification-v1/short-serving-requests-v1/author-seal-v1.json`, SHA `20348594c63f413d19a1efd1a69f09c5c1ba4f73c31c0d5a5f05d60629157336`; source/resource/native integration review is pending.

Do not repeat completed cleanup or control runs. The next priorities are reviewed timing compilation/rebinding, reviewed native serving parser integration/build, and continued M5 observation/final evaluation.

## Workspace and operating constraints

Workspace root:

```text
/Users/jasontitus/Documents/Codex/2026-10-12/can-we-create-a-new-experiments
```

Paths below are relative to this root unless marked remote. Most experiment receipts are outside the product Git checkout. Product repository: `publish-exl3`, branch `apple-metal-training-m5`.

Preserve the active training data, held-out data, completed baseline, model weights, and pinned `runtime-v5`. Do not launch a duplicate GPU job, silently truncate an edit, change a running dataset, lower the system reserve, delete models, or stop unrelated workloads. **Keep sfideas and WatchAI running.** Tune caches, workspaces, and separately recorded complete-item experiments around them.

M5 trains; Studio handles collection, CPU work, evaluation and qualified kernel tests. No training campaign is authorized or qualified on the 3090 PC. Separate layer experiments are possible, but synchronized training across these machines is not implemented, and independently trained adapters cannot be assumed to combine safely.

The daily source collector remains daily. Manual bounded mining was authorized; do not increase the scheduled cadence. Follow-up automation IDs are `continue-exl3-training-and-qwen-work` and `build-metal-optimization-library`.

## Live DeepSeek training

Host: `jason-m5-MacBook.local`.

Remote root and active attempt:

```text
/Volumes/Scratch/exl3-training/2026-10-13
/Volumes/Scratch/exl3-training/2026-10-13/metal-mix229-internal-split-v1/attempts/attempt-1791952639336825000
```

At 12:01:59 UTC, worker **47890**, parent **47866**, was live at elapsed **07:24:38**, CPU time **315:52.80**, RSS **17,315,872 KiB**. `train/latest.txt` contained **step-000079**. This is a liveness and checkpoint observation; the independently audited exposure count is through step 69.

The fixed campaign has **229 training views**, **16 held-out items**, and **458 planned updates**. Each update consumes one complete item. It trains rank-4, alpha-8 LoRA on **layer 39 only**, targeting `wq_a` and `wo_b`: **78,848 parameters**, with no expert adapters. Base weights are frozen.

Settings: sequence length 12,287 backbone positions, serialized capacity 12,288; learning rate `5e-5`; accumulation 1; clip 1; seed 8; save every update; retain two checkpoints; final fit examples 0; no automatic comparison during training. Process budget **80 GiB**, system reserve **8 GiB**, packed cache **6 GiB**, decoded cache **0**, prefix RAM **1 GiB**, prefix disk quota **64 GiB**, allocator cache **256 MiB**. TF32 is disabled in the model runtime. Do not change these settings mid-campaign.

Remote model is on the internal drive:

```text
/Users/jasontitus/.mlx-serve/models/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw
```

The full base identity previously matched the Studio model: 120 files, 426,352,806,119 bytes. Studio model root is `/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw`.

Check the existing worker before any recovery:

```bash
ssh -o BatchMode=yes jason-m5-MacBook.local \
  'ps -p 47890 -o pid,ppid,etime,time,rss; cat /Volumes/Scratch/exl3-training/2026-10-13/metal-mix229-internal-split-v1/attempts/attempt-1791952639336825000/train/latest.txt'
```

If observation fails, recheck this job; do not infer termination from a timeout or start another worker. `train/history.json`, the saved `trainer_state.json`, and training logs are authoritative for progress. Campaign receipt: remote `reports/split-internal-training-v1.json`.

### Audited training exposures

Step 69 has **53 exact-edit views and 16 diagnostic views**, **132,003 supervised target presentations**, maximum complete serialized length **12,078**, and historical MLX allocation peak **71.2797748595 GiB**. Full step-55 and step-69 contracts are equal; the first 55 exposures and their saved metrics are unchanged. These are view counts, not certified independent optimization-family counts.

Evidence under `kernel-library/next-round-v5/reports/`:

- `training-capture-step69-v1.json`
- `training-exposure-continuity-v4.json`
- `independent-exposure69-zero-b-v1/review-v1.json`, SHA `318e58339264d9ce48e2c6bf2902de0a32b21e6c20ee78a06e5141c4ce188c71`

The auxiliary zero-B audit had two boolean-versus-integer validation gaps. Actual captured values are correctly typed and its metrics were independently recomputed. The original receipts and reproductions are preserved; separate `early-learning-probe50-v1/zero-b-audit-v2` repairs more than 80 integer-field substitutions and awaits its independent review. This does not alter model training or the product evaluator.

## Completed evaluation and what it means

### Full held-out baseline

`kernel-library/next-round-v5/two-host-split-v1/studio-baseline-v2` completed all **16 items / 47,069 targets** and released its GPU lease. Baseline NLL is **0.1110763865**; MLX peak **22.6051 GiB**. Independent completion review SHA: `fb536a05fb179b981c54e741351de9cdb57998a3825e94de9ba798e6f4a7f16e`.

Keep all 16 items: nine have scoped source approval and seven have disputed references. Report both strata alongside the complete set. Only five rationale targets exist in this held-out set, so it cannot establish broad reasoning improvement. The baseline used a rank-8 zero-B graph; the early paired probe below uses matching rank-4 conditions. Do not claim bitwise equality between those graphs.

### Checkpoint 50 paired probe

Both conditions completed in separate fresh processes. Old wrapper/runner/worker PIDs **89117 / 89120 / 24355** were confirmed absent; the original lease receipt records successful cleanup and release.

Base directory: `kernel-library/next-round-v5/early-learning-probe50-v1`.

- Actual completion: `prepared-v5/future-execution-recovery-v2/completion.json`
- Actual lease: `execution/concrete-lease-v2.json`
- Scores: `prepared-v5/future-execution-recovery-v2/{zero_b,checkpoint50}/evaluation.json`
- Product evaluator output: `paired-quality-audit-v1/paired-token-metrics.json`
- Predeclared exposure groups: `paired-quality-audit-v1/exposure-group-summary.json`
- Independent audit: `paired-independent-v1/review-v1.json`, SHA `2148fc2a12eff1aefe1c16ad915113e9827d4c665c3069afa93b2886780dc2bd`

The panel contains four complete **training-pool** items, totaling **9,329 targets**. Two were seen by checkpoint 50 at updates 42 and 39; two were future training items scheduled at 136 and 183. Neither subgroup is held out. The final checkpoint remains 458; do not select a checkpoint or tune hyperparameters on this probe.

| Metric | Matching zero-B control | Checkpoint 50 |
| --- | ---: | ---: |
| Overall target-weighted NLL | 0.393180734 | 0.374158164 |
| Correct target predictions | 8,432 | 8,432 |
| All Metal-labelled targets, 539 tokens, NLL | 0.091318972 | 0.093388437 |
| Literal added Metal code, 83 tokens, NLL | 0.074488868 | 0.071259384 |
| Literal added Metal correct predictions | 80 | 80 |
| Rationale, 144 tokens, NLL | 3.444025014 | 3.080689496 |
| Rationale correct predictions | 60 | 59 |
| EOS correct predictions | 4 / 4 | 4 / 4 |

Overall NLL decreased **4.8381%**, and all four individual item losses decreased. There were **27 corrected and 27 regressed targets**, leaving accuracy unchanged. Seen-item NLL decreased 4.3433%; future-training-item NLL decreased 6.4213%. All Metal-labelled target NLL slightly worsened, despite the narrower added-code improvement. This is a mixed early fitting result, not fewer generated errors, faster inference, better held-out kernels, or demonstrated reasoning improvement.

## Current Studio failure and retained lease

The next queued numerical controls encountered two preserved failures:

1. A launch mistakenly supplied `MLX_ENABLE_TF32=0` to a Metal-only controller that rejects injected `MLX_*` variables. It stopped in verification before any output directory, lease claim or child. Independent review confirmed an environment-only correction was appropriate. Preserve `numeric-request-builder-v1/actual-binding-v1/launch.stderr.log`.
2. The corrected launch started controller **59638** and native child **59660**, then failed the native host check **before device creation**, with zero dispatches. Python/POSIX reports `MacStudio-2.localdomain`; unsandboxed Foundation reports `macstudio-2.local`. The child exited 1 and the controller intentionally retained its own lease. Do not retry the old job or remove the lock manually.

All paths here are under `kernel-library/next-round-v5/full-kernel-evaluation-v1`.

Preserved failed run: `controller-v2/numerical-controls-run-v1/{controller.json,native.json,lease.json,authorization.json,stdout.log,stderr.log}`.

Historical failed-job owner, now released:

```text
ktune-engine-59638-ddf6991af15749f2808f902417f27c6d
/private/tmp/sushi-gpu.lock.d
/private/tmp/sushi-gpu.lock.d.holder.e0d63e4f3a29405d8f69691f9ec791a4
```

Do not treat the native failure alone as cleanup authorization. Confirm current PID births, parent links, process groups, command closure, lock/claim inodes and owner hashes before release. Never signal unrelated processes or remove another owner's lease.

### Completed recovery (historical procedure)

`controller-v2/recovery-numerical-controls-v1` contains the exact operation specification, copied original state, read-only observations, guarded `execute_reviewed_release.py`, and tests. Author seal SHA: `bffc04883f449eb55853ca5ee087cd2a26f4c48051685692ac47ad7b85558142`. Eleven CPU tests and two actual read-only checks passed. **Release completed after review and fresh ownership checks; do not repeat it.**

Independent review is now held at `controller-v2/recovery-independent-v1/review-v1.json`, SHA `e965f0319391d70370795814ae6b896fbcc39f556226c2ed1efcc600fe11ed83`, status `approved_conditional_owned_release`. Eleven author and five independent CPU methods passed; root rehashed all 34 review inputs. Root completed the fresh concrete ownership and process check before release. Specification SHA: `630507b4afffe0745a2e87e42b8737a695ed8869e4ccf81e5dbd19c3b04c40cc`; script SHA: `ee283c73dc49f2f1652cfb129ce770757a5bce7bbaf6c56ebb6823d15ea3e9a6`.

Default invocation is read-only and cannot release:

```bash
/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python \
  kernel-library/next-round-v5/full-kernel-evaluation-v1/controller-v2/recovery-numerical-controls-v1/execute_reviewed_release.py
```

Only after checking the held review, original failure, fresh process closure, unchanged owner and fresh output paths, the reviewed release command is:

```bash
/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python \
  kernel-library/next-round-v5/full-kernel-evaluation-v1/controller-v2/recovery-numerical-controls-v1/execute_reviewed_release.py \
  --execute-reviewed \
  --review kernel-library/next-round-v5/full-kernel-evaluation-v1/controller-v2/recovery-independent-v1/review-v1.json \
  --review-sha256 e965f0319391d70370795814ae6b896fbcc39f556226c2ed1efcc600fe11ed83
```

The script repeats checks and uses the unchanged helper's mutex with only the copied owned state. Preserve the original failed receipts. This reviewed release has now completed successfully; preserve this command as historical recovery documentation and do not run it again. It did not itself authorize the subsequent separately reviewed v4 run.

### Prepared hostname repair

The narrow repair uses checked POSIX `gethostname`, the same contract as Python, with exact case-sensitive equality. It does not accept aliases or case folding. Source review SHA: `1ab2683f50c91bea8f47cca70a1a20a87faba7cf8af13220ec672ef4e4018452`.

- New native bundle: `qualification-v4/author-seal-v1.json`, SHA `51dad98105d8de470fee136558a5b9a1d4cf7261d0dcf49baa2f4c39e394c2f4`
- New fixed job: `qualification-v4/build-01/job.json`, SHA `74f89e4cff4f55f2bc395fd60247d0ce9ed840081520f5d82d9c0c7342d9042f`
- New binary SHA: `595f30f4509b1eab378f94f175cc0d1a62370b0ffabf756cb4bbed604dc23a7e`
- `controller-posix-v3/run_control.py`, SHA `d3995fa72670e35a3a5692e1453e25bd963eceb4df28ccd281711dcc54b15263`
- Original pending binder documentation (actual binding subsequently completed): `numeric-request-builder-posix-v2/{bind_numeric.py,pending-v1.json,review-required-v1.json}`

Native CPU validation passed; the separate controller passed 17 CPU tests. Four control kernels, 224 input cases, source/dispatch hashes, ABI and tolerances remain unchanged. Fresh compiled-source/ABI/controller/rebind reviews and concrete request review completed before the successful v4 run recorded above. The old request `f36192e5...` binds the failing v3 job and must not be reused for v4.

Timing's original `timing-v2/timing_body.swift` has the same Foundation hostname bug and must not execute. Separate `timing-posix-v4` source fixes both hostname selection and explicit numerical-receipt host/API consistency. Independent CPU review: `timing-posix-independent-v1/review-v2.json`, SHA `6197f9d627208b7483467188480b36adb01064cc0bee99faddb8fb5f081f237b`. A compiled timing artifact and concrete execution request remain required.

## Serving and generated kernel evaluation

Fixed-control GPU numerical behavior is independently qualified by the successful v4 run recorded above. Original v3 failures remain preserved. Timing and model-generated kernels are still unqualified. The fixed suite is 28 shapes × 8 families = **224 input instances**, 14,664 output rows per variant. Four curator variants require 896 dispatches. Baseline and golden must pass; intentional negatives must fail as specified.

The generated-candidate controller is separately reviewed at `generated-controller-v2`. A complete numerical rejection can release only after verified cleanup and permit the other candidate; a rejected candidate cannot enter timing. Infrastructure, unsafe, incomplete, or uncertain runs retain ownership. Review: `generated-controller-independent-v2/review-v2.json`, SHA `89317372a3831f643318aeaab51dda9fb7d58f6bbd71049c5dd67f4d09f8df11`. Twenty-one author methods and 12 independent scenarios passed. Preserve v1 reproductions.

The task prompt remains the fixed **774-token** native agent prompt; final evaluation uses one greedy output each from final458 and its matching zero-B control, cap 4,096, with no retries or repairs. The task is a single curator row-sum task, not 224 independent coding tasks. Real model-generated task outputs are not available yet.

Before generation, qualify serving construction, adapter projection, cached/fresh logits, trainer/serving arithmetic differences, and task-length/context capacity. Use `exact-generation-qualification-v1/PLAN.md`; several CPU helpers now exist beyond that older plan's status text. The preserved serving binary is the reviewed host-build-v5 artifact SHA `ea5f55e4e7ff2cf8fd5905151ba8c39466ac05d2a5712b438fea4c8c37e04bd3`, from serving-v4 source. It is not usable with the actual nested text configuration until the isolated parser rebuild is complete. Pin a separately reviewed new build; do not substitute unreviewed binaries or libraries.

### Runtime and full payload preparation

Under `kernel-library/next-round-v5/exact-generation-qualification-v1`:

- `runtime-package-v1/runtime` contains six byte-identical runtime files, 252,477,016 bytes. Independent static review verified 38 dependency edges, including exact external webp/sharpyuv files and their symlink chain. Review SHA `3498037be7892064d27a13a0258047e5a2ff9621763e5d94b905d317f9665898`. Static closure is not dynamic-loader or GPU qualification.
- `payload-full-verification-plan-v1/full-verification-attempt-v1/{full-payload.json,identity-continuity.json}` now exists. The unchanged verifier completed successfully at **11:14:24 UTC**, freshly reading **365 files / 426,352,827,347 bytes**. Session 62242 exited 0. Logs are in the parent directory. **Independent receipt and current-continuity audit passed**, SHA `5b06b7dcd478fe3f9fc21d9df99326fb9621b29edfffd13ae341c7ab69936c11`.
- The legacy 120-file verification remains intact. The new verification adds required actual host/global-before/after fields and 245 incidental files, instead of relabelling old metadata as fresh full verification.

For future launches, measure available memory rather than assuming 80 or 100 GiB is free. The last numerical startup observed about 83.5 billion available bytes, roughly 77.8 GiB; retain explicit units. Do not blindly use an 80 GiB process plus 32 GiB reserve for a Studio serving job. Inventory buffers and caches, prepare a separate reviewed budget that fits, and ramp capacity. No such serving launch is authorized by a static package receipt alone.

Rationale rubric: `full-kernel-evaluation-v1/rationale-rubric-v1`, author SHA `8ece6566dceff86b62e280c611041a3bb2d0e1086f45acdeab1326b7d58152f3`; independent review SHA `078c27ac4f4447d27da766f6b86a327c3fb3a7d530b37794f0d6a26233cdab3b`. Five qualitative criteria, no summed winner or verbosity reward; preserve source-only judgments before numerical outcomes. No worksheet is scored yet.

## Future training library

Active 229-view data is unchanged: 186 exact edits and 43 original diagnostics. Only 37 exact views alter literal `.metal`; others cover inline MSL, host/API/dispatch/backend work. The future portable corpus contains corrected diagnostics and must not be confused with the active version. A known MinP diagnostic correction remains a future-data issue; do not silently swap the running data.

Recent additions are separate library tiers, not automatically training-ready examples:

| Bundle under `kernel-library/next-round-v5` | Captured source changes | Useful views and limits |
| --- | ---: | --- |
| `new-metal-batch2-v1` | 2, previously inventoried IDs | CE diagnostic reviewed; positive CE held for nonfinite-target backward behavior. GEMV whole edit 17,173 tokens held for capacity/protected-parent policy. Three historical Metal units compiled. |
| `cross-source-batch3-v1` | 2 newly captured IDs | TensorFold and oMLX diagnostics source reviewed. TF whole edit 8,729 tokens held for prescale/caller-domain qualification; before/after Metal compiled. oMLX whole edit 64,281 tokens preserved untruncated. |
| `cross-source-batch3-v1/omlx-7b9e49acee5f/derived-single-score-v1` | Same oMLX change | Complete 6,241-token / 1,423-target component API addition; 11 CPU checks and independent source/replay/index review passed. It is not full split-K integration or an isolated measured speedup. Review SHA `5a4531c7b75884322592920c08f53609a055ea5be40311dd6ddfca233d7f1ba3`. |
| `historical-batch4-v1` | 3 newly captured IDs | MLX quantized split-K, segmented NAX and split-K routing/AddMM; 18 complete changed files. Three reasoning drafts fit 12K; full edits at 69,166 / 52,142 / 33,053 tokens remain archived untruncated. Independent semantic/lineage review passed for three scoped report diagnostics, SHA `f455c0832a014072003e0556b8f88750a623109a98f3c6d0181b0c96e27aee40`; related NAX/AddMM lineage is explicitly recorded. No runtime or active-corpus admission. Author seal `c4dcc5cb433eae1e6291ffa061e7b3305384ea6ae82d700da007625151b1bdc7`. |

This is seven complete captures across those three bundles, of which five IDs are new to the prior local inventory. Multiple exact/diagnostic/component views of one change do not create additional independent optimizations. Keep source review, compilation, numerical validation, and measured performance separate. Author-reported timing remains reported timing unless locally reproduced.

Source configuration is `kernel-library/sources.json`, covering sushi, TensorFold, mlx-serve, ds4, llama.cpp, omlx, mlx, mlx-lm, mlx-vlm, splash, MTPLX and vllm-metal. Historical batch4 records older MLX pagination/coverage gaps and the next anchor. Continue bounded backfill and review backlog without modifying protected ktune or frozen train/test data.

## Qwen status

Work remains isolated under `qwen-next`. No Qwen model training or 16K qualification has happened. The updated design is `qwen-next/docs/qwen38-flash-next-16k-finetuning-design.md` and the corresponding product document; local commit `bf180022` records it.

CPU storage/decoder/reference work and QSA selection are available. QSA tests cover binding sparse cases across 2,051/2,052 tokens and a few queries into 4K history; they do not provide full 4K training, sparse attention backward, recurrent backward, or a complete Qwen forward. Preserve the sealed `qwen-next/project.json` and current review `reports/qsa-selector-independent-review-v2.json`, SHA `ad116b5cc22bf9a487a760ec87a4f3aa25c23ddd792e19e95a35e30637423e9e`. Reuse CPU work while DeepSeek owns a host's GPU; do not claim model support from isolated primitives.

## Resume order and completion requirements

1. Verify M5 worker and checkpoint progress without changing the fixed 229-view / 458-update campaign. Current saved step is 79; the independent exposure audit remains through 69.
2. Finish the metadata-only reseal and independent receipt for `short-serving-requests-v2`. Its nested text configuration, null adapter metadata, runtime artifact size and explicit-null validation fixes pass CPU tests. Execution remains disabled. Preserve v1. The old binary cannot parse the actual nested model configuration correctly.
3. Prepare and independently review an isolated native host build using the reviewed build-only workflow, replacing the parser with the staged native patch. Preserve original source, binary, runtime and dependencies. No fetched build scripts or dependency upgrades. Rebind the new binary/runtime package into six actual short requests only after build review.
4. Qualify actual serving construction, adapter projection, cached/fresh logits, arithmetic bridge and context capacity in serialized guarded phases. Initial six cached children plus twelve fresh-prefix children are 18 total; missing observations remain incomplete. Proposed resource terms are 40 GiB process / 96 GiB native box / 32 GiB reserve, not a measured fit guarantee. Preserve native load preflight and memory guards.
5. Build the separately reviewed timing POSIX v4 source in a new namespace, pin its artifact/job and review a concrete request. The fixed schedule remains 840 batches / 84,000 dispatches. Source approval is not execution approval; no timing GPU run has occurred.
6. Continue future library curation using completed historical batch4 and component reviews. Preserve source lineage, explicit reported-versus-local timing and complete-item lengths; do not swap active training data or admit oversized full edits as fitting examples.
7. After update 458 actually completes, use `two-host-split-v1/final-transfer/check_final_bundle.py` to verify and transfer exact final state. Fresh-process evaluate all 16 held-out items, both review strata and code/rationale/EOS categories. Generate fixed final/zero-B samples, compile, check numerical correctness, and time only valid candidates with scalar and direct paired comparisons. Apply the predeclared rationale rubric without rewarding verbosity.
8. Continue isolated Qwen CPU work when practical; complete-model forward, sparse attention backward and recurrent backward remain missing. Do not claim 16K support.

Complete the goal only when the larger training run and final evaluation are finished and benefit or lack of benefit is honestly reported. Lower pooled loss, a changing adapter, or a successful save does not prove useful generated Metal optimization.

## Git and agent handoff

Original handoff commit: **`201944ab`**. Earlier completed product commit: **`72d73b7d`**, documenting audited exposures/evaluation work in `apple/reports/m5-successor/progress-step69.json`. Other useful local commits: `bf180022` Qwen design; `17454054` reusable historical Metal closure compiler; `f9e4250f` prior baseline/training progress.

Push remains unperformed after an earlier automatic review rejected public-fork publication without explicit public-sharing consent. Do not retry or bypass that rejection. Local documentation/commits are authorized.

Continuation ledger: `kernel-library/next-round-v5/m5-successor/current-run.json`. Some historical fields remain old observations; prefer explicitly dated live observations and actual receipts. The product report is a snapshot, not the live ledger.

Available agent tasks at this handoff:

- `/root/adversarial_code`: historical batch4 and timing v4 CPU reviews complete; finish serving v2 reseal review, then review the new native build and concrete requests.
- `/root/ingest_engines_b`: timing host/API repair sealed and independently CPU approved. No timing compilation or GPU action; next work requires the new build plan/artifact review.
- `/root/quality_research`: six serving requests CPU validated; source repair passes 22 author Python, eight independent CPU and four pure Zig methods. Metadata-only reseal pending after separating the extended reviewer test filename. No native rebuild or serving execution.

Use `followup_task` to wake completed agents; `send_message` alone does not start an idle agent. Handoff does not stop the live M5 run. All completed baselines, failed attempts and receipts remain preserved.
