# M5 training and evaluation progress

## Current status — 2026-10-15 UTC

**Training and the full 16-item likelihood evaluation are complete. Generated-kernel quality remains unproven.** The next work is the paired generation experiment; exact-task null previews are complete. The exact-task previews completed on M5 with bitwise-equal baseline/zero-B logits and released leases; independent actual review passed. Source collection and Qwen work remain paused; protected data, baselines, pinned runtime-v5 and unrelated workloads are preserved.

Training completed **458 updates on 229 distinct complete examples**: 186 exact edits and 43 diagnostics, each presented twice. Rank-4 LoRA covers `wq_a` and `wo_b` in final layer 39: 78,848 trainable parameters. Final adapter, optimizer and history audits passed.

The independently audited 16-item evaluation covers 47,069 supervised targets. Overall NLL improved **11.72%** and added Metal code NLL improved **5.08%**. Added-code top-1 correctness fell from 17,859 to 17,856. Formatting contributes 82.62% of the NLL-sum gain; only five rationale tokens were measured. This is a narrow target-likelihood result, not evidence of better generated kernels or reasoning. Historical baseline reuse and mixed cold/prewarmed timing remain explicit in the evaluation receipt.

The **M5 six-phase serving controls completed and passed independent outcome review**. All six leases were released; all 19 observed process IDs were absent from two full terminal tables. Twelve raw logit rows (1,551,360 finite values) and eight comparisons were replayed; root verified all 409 audit bindings. Matched zero-B controls equal baseline bit for bit in cached and fresh modes. Trained B changes logits, with a maximum observed effect of 8.640625, but compared argmax tokens stay unchanged. Strict cached/fresh parity failed: maximum errors are 0.296875 for baseline/zero-B and 0.375 for trained B. These M5 values are separate from the earlier Studio failures (1.59375/1.25). No tolerance was widened and trainer-forward parity remains unqualified.

The **separate matched-endpoint protocol passed independent design review**. It asks whether changing only matched adapter B on this fixed serving setup changes an externally graded kernel outcome. The stricter v4 admission remains unchanged and unadmitted. One common base, runtime, prompt, sampling policy and environment bind two elected null previews—absent and matched zero-B, each 774 prompt tokens and one output token—then exactly one zero-B and one final458 endpoint attempt. Both final slots are frozen before previews; preview outputs cannot select or change the prompt, adapters or sampling. Context is 8,192; endpoint cap is 4,096 including any terminal EOS1. Natural EOS is allowed; retries and repairs are not.

The preview source is sealed and **104 files / 5,151,306 bytes have been copied and hash-verified on M5**. Root reran ten CPU tests successfully, including complete preparation and both frontend/wrapper validators. Independent source review and actual M5 CPU preparation, request validation, memory admission and the trapped composition checker passed. All 13 actual artifacts (348,218 bytes) and the 136-pin plan are captured. Concrete-request review passed with 137 bindings. Both device previews completed exit 0 by 07:09:20 UTC, about 29 seconds per native run. Root independently replayed all 129,280 finite F32 values per condition, raw-byte equality and the shared first token. Both owned leases were released and the shared environment was restored empty. Independent actual preview review passed: both production validators, all raw bits, 64 memory observations, two releases, seven terminal process IDs and archive/fresh-environment chains were replayed. Root accounted for all 351 review pins, including fresh raw global release receipts. The paired endpoint source passed independent review and was copied to M5. Actual preparation, both request validators, memory admission and the composition checker passed without launching a GPU job. The 161-pin plan and all 13 prepared JSON files, including the actual admission, are captured as raw bytes for concrete independent review. The capture/grading chain also passed independent CPU review; root rehashed all 677 review inputs with no mismatches. Concrete independent review accepted all 162 execution bindings, and root verified its 40 local evidence pins. Paired generation launched at 07:27:57 UTC. The base-control native child completed at 07:35:52 UTC (472.32 seconds), producing all 4,096 allowed tokens without EOS. Its registered outcome is a retained cap-without-EOS task failure; it is not a qualified generated kernel. The trained slot was refused before native launch because the shared GPU queue had a waiter. The operation exited 1 and its original failure and preflight receipt remain preserved. This is an infrastructure-unrun trained condition, not evidence of trained-model failure. Two later full process tables showed the owned controller/frontend/native processes absent, and the first raw global receipt records verified release at 07:35:53 UTC. At 07:38:15 UTC queue ticket 71 was marked done and the global lock was absent; that observation is not a reservation. No automatic retry or queue bypass occurred. Independent terminal audit passed and root verified all 206 evidence pins: actual tokenizer decoding, initial preview equality, 468 passing memory samples, 21 recorded IDs absent in both PID/PPID/PGID tables, and the shared environment reset. Peak aggregate RSS was 40.06 GiB and minimum runtime available memory was 59.11 GiB. The trained wrapper has no lease, child or output artifacts. A separate five-file continuation passed independent source review (53 root-verified bindings), eight author tests, five independent tests and actual M5 preparation/default validation. Its prepared request differs in exactly three transport paths; concrete review binds all eight source/prepared files. The trained slot launched at 08:00:11 UTC: controller77980, frontend77994 and native78013 were observed live. Fresh queue/lease/memory guards passed. Original requests, failure receipts and the control cap failure remain intact. A normal timestamped preclaim observation is added under the original observation directory; no existing original file is rewritten. These are experimental evaluation tools, not a qualified portable serving-adapter product.

Every launch retains the reviewed physical128/process60/native82/reserve32/wired8 GiB policy, fresh memory checks, ongoing monitoring and exact-owned cleanup. The completed short controls recorded 130 passing memory observations: peak aggregate RSS 30.32 GiB and minimum available memory 52.80 GiB. That is short-control evidence, not a long-generation fit guarantee. The internal-drive model is bound through accepted full useful-payload hashes and current metadata continuity; the separately audited Scratch full checksum is also complete. No further model transfer is required for this experiment.

Successful generated sources still require complete-source/ABI review, compilation, **all 224 numerical cases and all 12 timing workloads**. Invalid, incomplete, cap-limited or incorrect kernels remain unsuccessful; infrastructure failures retain an explicitly unrun slot. Timing comparisons require both kernels to pass numerical qualification. This is initially one synthetic kernel task, not 224 coding tasks, and cannot establish broad model superiority.

Caching remains the preferred next-trial strategy when expected benefit is comparable. The first 229-item epoch took **20.64 hours**, versus **1.55 hours** for the repeated epoch. Six cross-process/resume updates hit the existing disk cache without misses or writes, qualifying three selected rows; the other 226 entries have metadata/key evidence. The second three-update process took 239.48 seconds wall time versus 89.16 seconds inside its training timer. Rank/alpha/LR changes are source-eligible only with the exact frozen boundary and all binding/key identities preserved; changed-rank hardware qualification has not run. See the [cache plan](apple-cache-aware-experiment-plan.md). The reviewed 41-development/188-training proposal is unactivated and requires fresh trials because checkpoint458 saw all 229 examples.

The [current receipt](../apple/reports/m5-successor/current-run.json) records source namespaces, hashes and execution status. Prior import/checksum failures and their cleanup receipts remain preserved in their original experiment lanes. Sections below are historical observations, not current process status.

## Historical preparation — 2026-10-13

The sealed next run contains 198 training items (159 exact edits and 39 diagnostic items), with 16 protected held-out items and 396 planned updates. The M5 campaign launched on October 13 at 17:25 PDT after full checkpoint verification and independent concrete launch review. Its longest-item preflight is currently initializing; main training updates have not yet been observed. See the latest campaign section below; earlier sections retain historical review milestones.

The diagnostic and prior-adapter NLL launchers passed independent review: 35 CPU test methods and 34 subtests. Concrete evaluation requests still require review against the completed training checkpoint. Comparisons must use fresh M5 base, previous adapter, and new adapter results with TF32 disabled. These small diagnostic probes do not establish general reasoning or generated kernel performance.

The Qwen affine row-span planner passed 77 CPU tests, including 62 independently written tests. It plans bounded, deduplicated byte spans; it is not a reader, decoder, or qualified Qwen training implementation. The historical progress receipt predates this independent review.

Supplemental compiler review verified 11 MLX translation units across five changes and eight DS4 source compositions across four changes. Inputs and compiled artifacts were rehashed against pinned Git sources. Metal 4.0, O3, and no-fast-math were explicit test settings, which may differ from upstream. No GPU numerical or performance qualification is claimed. Bulk captured source and AIR artifacts remain in the experiment workspace; committed receipts preserve their identities.

Expression evaluation wrappers passed independent review after two evidence-validation gaps were fixed (23 CPU methods and 33 subtests). Fresh M5 positive and negative controls and the concrete final-checkpoint request remain execution gates. TensorFold compilation review also verified 38 passing file-sides across three changes with passing units, five failures and one unqualified side across seven inspected changes. Missing generated host definitions mean failures are not upstream regression verdicts. No active dataset or pinned runtime was changed.

## M5 control preparation

Both predeclared expression controls compiled on M5 (two candidate expressions per control job, plus reference kernels). The compiled Swift verifier validated both jobs with `--validate-only`; both report `bindings_verified_no_gpu` and `gpu_executed: false`. This does not establish numerical success. A separate active Sushi optimization experiment holds the shared GPU lease, so numerical controls and training must respect that lease. No unrelated workload was interrupted.

## Newer Metal source review

The bounded discovery scan now retains commits from partial mirrors even when parent objects are missing. Its independently reviewed result contains 348 repository/commit entries, representing 323 distinct commit IDs, with missing-history limits recorded. These are discovery candidates, not qualified training rows.

Five source-backed teaching records cover cooperative-tensor compatibility, shape-aware NAX attention dispatch, Metal address-space compatibility, build/deployment gates, and SIMD work avoidance. The last was already represented in the sealed corpus. Four additional changes have complete before/after source and license captures: 49 changed files total. Independent review verified Git blob identities, modes, content and exact full changed-tree patch replay. Compiler, numerical, performance and training-admission checks remain separate. Bulk source captures stay in the experiment workspace.

## Shared GPU guard

The separate shared-GPU lease wrapper and the two concrete control requests passed independent review (29 CPU tests). It uses the existing Sushi coordination protocol, preserves foreign owners and tickets, verifies declared input hashes, and releases only its own unchanged lease after verified completion. Unknown cleanup retains its own lease for inspection. The two fixed controls ran on M5 and released their leases cleanly; their numerical receipts are under independent result audit. A longer training command requires a separate lifecycle and concrete-request review.

## Independent M5 numerical-control audit

The exact relocated expression plan is now qualified by fresh M5 positive/negative controls. The correct expressions passed15/15 cases; the known-wrong expressions failed10/15, with each probe detecting errors. All reference cases passed, and no nonfinite outputs, input mutations or guard corruption occurred. Independent CPU reconstruction matched all60 GPU result rows, including output hashes and mismatch counts (656 output values per reference/candidate condition). The reviewer also rehashed59 remote identities and verified serial completion and lease cleanup. This qualifies only the fixed harness; no model-generated improvement or performance gain is established.

The separate v2 GPU lease guard passed long-campaign lifecycle review with 45 CPU tests and five subtests. It handles reused child PIDs and unknown surviving process-group members conservatively. The original control-qualified guard is retained. Training still requires the completed payload verification, concrete plan and request review, and live resource preflight; implementation approval is not launch completion.

## Protected newer Metal candidate

Independent admission review excluded SDPA commit `5778a97c04b40753cb5d7c255aec76c2156e916a`: its exact parent is the captured reserved MLX PR4461 head. No training rows were exported. This is the existing finite head/parent identity policy, not an exclusion of all related MLX history. All eight active corpus artifacts still match the sealed release, with 198 training and 16 held-out items. The receipts bind the local source evidence; bulk source remains outside this checkout. Other newer Metal drafts remain separate from this run.

## Reviewed cooperative-tensor compatibility example

Commit `91c83d19bf022d7bc3e906582435f69fdc77ec63` now has an independently reviewed agent-facing example for future compatibility-focused intake. It preserves the complete four-file historical edit and seven complete context groups, with 9,153 serialized tokens and 2,603 supervised tokens. Native edit replay, source identities, tokenization, masking and finite protected/prior overlap checks passed. The actual packer accepted the isolated example; the active 198/16 corpus remains unchanged. Compilation, numerical correctness and performance have not been qualified. Committed receipts bind the local full payloads; they do not make this checkout a standalone copy of those payloads.

## External storage and cache review

The checkpoint transfer completed successfully (365 files, 426,352,827,347 bytes); full destination hashing is running. Scratch is an external USB APFS volume. Two one-second physical-disk samples during verification measured 176–188 MB/s; these are observations, not an isolated training benchmark. Internal storage has only about 101 GiB free, so the checkpoint cannot move there intact.

Independent review checked all 81 pinned runtime files, 214 prepared rows and 31 existing cache headers. Prefix reservations total 46.871 GiB, within the 64 GiB quota. The final layer's packed bank is 4.762 GiB versus a 2 GiB cache; this warrants measurement but does not prove a bottleneck. The fixed runtime fully hashes about 397 GiB at each model construction, and its training timer excludes that startup. Preserve the current runtime, reserve and cache settings until M5 preflight measurements; report wall-clock phase duration separately from training-loop timing. No integrity memo bypass or storage migration was performed.

## Campaign launched; parallel local ingestion

All 365 destination checkpoint files passed full SHA-256 verification against the source manifest (426,352,827,347 bytes). Independent reviews approved the concrete five-phase plan and exact detached request. The M5 wrapper PID 37886, runner 37889 and preflight child 37893 were confirmed live at approximately 17:29 PDT. The model loaded all 40 frozen backbone layers and is hashing payloads for prefix-cache identity. The campaign proceeds through longest training/evaluation item preflights, fresh baseline, 396 updates over 198 items, and fresh final-adapter evaluation. No completed main update or quality benefit is claimed yet.

Preparation exposed a representation difference from archived prepared rows: 22 training prompts have one extra token from pinned-runtime JSON schema key ordering. Independent review reproduced every actual row from the sealed raw data and pinned tokenizer. All 198 supervised target sequences and all 16 held-out full sequences are unchanged. This explicit normalization exception is bound by the concrete review; no raw examples, targets or held-out data were edited.

Mac Studio runs three independent CPU ingestion/review lanes covering all twelve source repositories while M5 owns training and evaluation. Newly reviewed data goes into a separate next-round library. Three vLLM MLA edit examples have passed independent source, replay and tokenization review; two fit 12K and all three fit 16K. They are related changes, not three independent optimization families, and have no local performance qualification. Further complete edits and source-grounded diagnostic views remain under review. The active 198/16 experiment remains fixed.

## Predeclared token comparison audit

The exact16 held-out items reconstruct47,069 supervised tokens, including18,364 added Metal-code tokens,16 EOS tokens and only5 rationale tokens. Independently revalidated lexical categories will keep code, comments, metadata and EOS metrics separate. This set cannot establish reasoning quality from its five rationale tokens. The CPU comparison utility reproduces the historical audited aggregate and all16 per-row metrics. Independent synthetic unequal-length tests and22 malformed-input cases pass, including vocabulary bounds and a concurrent-file-change receipt test. The utility hashes the same bytes it parses. It remains a metric calculator: exact-base, runtime, checkpoint and completed-launch provenance must still pass separately. No new M5 quality result exists yet.

## Next-round exports and adversarial finding

Independent review and export-consumption checks now cover15 core historical edits across the two engine lanes:9 fit12K and all15 fit32K. These are additional next-round library items; the current198/16 experiment is untouched. A separate unmeasured item remains opt-in, and the remaining MLX batch is still under review. These counts establish source/edit/tokenization admission, not local compiler, numerical or performance qualification.

The review found a source-arithmetic counterexample in captured MLX commit9ab977b5649154590d598ea5d545aa1b3c97f883: its small-depth3D convolution dispatch accepts flip=true but the decomposed depth views use the same unflipped index. A scalar16-channel example gives208/416 where the declared flipped contract requires176/352. This is a CPU expansion of pinned source, not an executed GPU regression or a claim about current upstream HEAD. The case is being removed from positive intake and retained for reviewed failure analysis.

## Long-item forward and further library checks

The live M5 preflight finished model payload hashing at1051.6seconds and reached layer8/40 at1336.3seconds. Observed activeMLX allocation was10.7–11.3GiB in the early layers; this is not a peak or completed capacity test. One shared external-disk sample measured183.53MB/s, while swap usage remained3597.88MiB; no isolated I/O benchmark or cache change was performed.

Both captured versions of TensorFold5d7fbf3034e7 qmm6_nax.metal compiled with their exact Git header closures under Metal4.0/O3/no-fast-math. Independent review checked Git bytes, arguments and AIR hashes. Metal32023.921/SDK27.0/compiler hash were observed immediately afterward, not pinned at invocation. No GPU numerical/performance qualification follows.

The reviewed B export now contains5 exact edits plus2 reasoning-only examples;4 exact edits fit12K. Export bytes, IDs, masks and independent approval bindings were rechecked. Deeper MLX review also quarantined a grouped-SSM whole-patch regression in80c217ff96de: source-derived geometry produces a256-wide update for128-wide state. The CPU source counterexample is not a claim about current upstream HEAD or an executedGPU test.

## Expanded successor and completed longest-item preflight

The original 198-item campaign completed its longest-training-item preflight:
12,173 serialized tokens, 9,747 supervised tokens, one full update, saved adapter
and optimizer state, and 67.322 GiB peak MLX allocation. Loss on that update was
0.3779546; this single observation is not an improvement comparison. The phase
verifier checked its completed artifacts. Initial checkpoint hashing took about
17.5 minutes; the logged training interval after setup was 1,312 seconds.

Before baseline or main training, the owned campaign was deliberately stopped
through the independently reviewed identity-bound handoff. The newly started
test preflight was interrupted. A separate release verified terminal process
generations/groups and exact lease ownership. Independent post-action review
confirmed the training preflight and all logs/checkpoints remained intact, no
baseline/main-training/final-adapter phase ran, and the owned lease was released.
This supersedes the earlier expectation that the 198-item campaign would continue.

The successor contains 229 distinct training commits: 186 exact-edit examples and
43 diagnostics, with the same 16 held-out items. It adds 27 edits and four distinct
diagnostics. Independent review replayed all 27 new edits and retokenized all 245
published rows. The review found four prompt-order sidecar mismatches in candidate
v2; v3 corrects them explicitly, preserving every supervised answer and the older
artifacts. Candidate v3 SHA-256 is
`fba890f2a5fb4c362c32db8a2d0f72c0904a1553a890baf254832b56561b6735`.

CPU preparation on M5 produced a separate 458-update plan, two exposures per
training item, with a 6 GiB packed cache, unchanged 80 GiB cap and 8 GiB reserve.
That cache can hold a final-layer expert bank for recomputation; it does not avoid
cold-prefix reads and no speedup is established yet. The new campaign must pass
its own preflights and fresh baseline. At this entry, preparation completed and
concrete launch review remains pending; no successor GPU job has launched.

Reusable tool improvements are tracked in `apple/docs/TOOL_INTEGRATION.md`.
The gated preparation helper now exposes the cache budget with 26 passing CPU
tests; more detailed evaluation checks are being extracted from experiment code.

## Expanded campaign launched

At 2026-10-14 01:10:34 UTC, the independently reviewed 229-item successor
launched on M5. Wrapper 56702, runner 56705 and one preflight child 56709 were
confirmed live. The new 6 GiB-cache longest-item preflight is running; there are
no main updates or successor quality results yet. Actual M5 prepared token IDs
and masks matched all 245 reviewed rows, and the held-out bytes are unchanged.
The plan SHA-256 is `1c51251d4bf4404c3f6e21de6b255d96da5e84b3cac4c264181e1ebff5799f33`.

## Successor longest-item training preflight completed

The new campaign completed its full 12,173-ID training preflight (9,747
supervised targets), including backward and one optimizer update. The runner
verified the checkpoint, optimizer and RNG artifacts, then automatically started
the longest-test evaluation preflight as PID 73629. Main training remains pending.
The completed receipt is retained under `apple/reports/m5-successor/` with SHA-256
`dc6576f24f1f4016d36d3204189c9a5c556059ae205b083edc93133feade6cdd`.
Peak MLX allocation was 71.324 GiB under the unchanged 80 GiB process budget.
This is a capacity observation for that item, not a model-quality result or a
whole-system peak measurement. Larger-cache timing limits are recorded in
`apple/docs/TOOL_INTEGRATION.md`.


## Exact evaluation cache reuse amendment

The running final458 full16 evaluation received 13 missing verified baseline prefix entries without restart; three entries already existed and matched. Root executed the independently reviewed seeder successfully after all63 review pins and18 CPU tests passed. All source payloads were checksummed, complete binding objects matched, and publication used the existing worker cache lock and exclusive directory publication. Cache inventory is5,390,530,134bytes under8GiB. Evaluation timing is now mixed cold/prewarmed; inputs, arithmetic, adapter and held-out coverage are unchanged. An already in-flight cold item continues normally. Final metrics remain pending.

The M5 cross-run qualification controls are installed and three complete original training rows were materialized locally, without exporting token data. Actual CPU preparation observed105.24GiB available, a free shared lease and229 cache entries. Its concrete phase1 request is awaiting independent review; no qualification GPU job has launched.


The independently reviewed actual M5 phase1 request launched through the unchanged shared lease wrapper (root session73103; wrapper40325, worker40328). Fresh prelaunch headroom was105.26GiB and the229-entry metadata inventory was unchanged. The actual worker loaded all40 layers and began full model-payload hashing. Cache hits and phase completion are not yet demonstrated; phase2 remains gated on phase1 outcome. Studio evaluation session44309 remains live. The actual Studio cache-seeding outcome independently passed its68-pin review, including six tampered-evidence negatives.


## Fresh-process cache qualification: first phase measured

M5 phase1 completed three complete-item updates with3 disk hits,0 misses,0 writes,743,277,191 activation bytes read, and unchanged metadata for all229 prior cache entries. Model-identity hashing took about147seconds; the training timer then measured89.004seconds for the three updates (9747,2570,96 supervised targets). Peak MLX allocation was69.6165GiB. The wrapper exited0 and released/verified cleanup. Step3 saved adapter, optimizer, RNG and trainer state; bounded CPU checks found78,848 adapter and157,697 optimizer floating values finite, with contiguous tensor payloads. Resume phase remains pending independent outcome/concrete-request review. This qualifies a first-phase cross-run cache hit, not prediction reload parity, uninterrupted trajectory equality, or generated-kernel quality.


M5 phase2 launched under root session6995 from exact step3 after129 preparation pins,12 CPU tests, full source review and actual request review. Fresh preparation observed103.87GiB available, prior phase1 process/groups absent twice, a free shared lease, and unchanged229-entry metadata. The original three-update resume argv and80/8GiB process/reserve remain unchanged. Actual resume-loader success, hits and step6 completion remain pending. Initial root parsed-observation reserialization changed numeric process-key sort order; exact raw metadata bytes were captured and their original hash verified instead. No runtime/request change or guard relaxation was made.


## Cached resume completed

The fresh phase2 process logged restoration of optimizer step3/cursor3 and completed steps4–6 on the original three complete rows in the approved order. It reported3 disk hits,0 misses,0 writes,743,277,191bytes read,89.155seconds training time and69.661GiB MLX peak. Both fresh processes therefore completed6 cached updates/24,826 supervised target presentations. All229 cache metadata records remained unchanged across both phases. Final step6 adapter/optimizer/RNG/state files passed CPU hash/header/finiteness checks, optimizer step was6 and all second moments were nonnegative. The phase2 wrapper completed0, released/verified cleanup, and fresh read-only observations found its process/groups absent twice, its holder absent, shared lease free and no pending queue waiter. Independent final outcome review is pending. This verifies the observed cache and native resume path; it does not establish an uninterrupted-run trajectory equivalence, prediction reload parity, or generated quality.


Independent final cache/resume outcome review accepted the actual six-update evidence (SHA d241e3946f7da10e06e10db27b82d4226978d2aa375c3a047bd1cd9d0127432c;145 pins root-verified). Its three CPU methods/10 tampered-evidence negatives passed. Final ownership closure includes wrapper42895, worker42898 and transient descendant43554; all were absent by PID/PPID/PGID in two fresh root observations. The earlier principal-only observation remains preserved. Review explicitly excludes quality, inference reload parity and uninterrupted-trajectory equivalence.


## Full16 final458 evaluation completed

Root session44309 exited0; all16 examples/47,069 supervised targets completed. Historical baseline NLL0.11107638648 versus final458 NLL0.09806023820 is an11.7182% relative reduction. Added Metal code (18,364tokens) improved NLL5.0779%, but correct next-token predictions fell17,859→17,856. Boilerplate and patch headers account for82.6163% of the overall NLL-sum gain. EOS remains16/16 correct; five rationale tokens cannot establish reasoning benefit. The nine source-scoped and seven unresolved-reference rows remain separately reported. Historical rank8 zero-B reuse/four exact rank4 overlaps remains explicit; no full16 endpoint-A zero-B rerun was claimed. This is a fixed-target result, not generated-kernel correctness/performance. Independent outcome audit is underway.

Elapsed evaluation time was3521.475seconds with21.541GiB MLX peak, mixed cold/prewarmed; disk counters show12hits/4misses/3writes. The exact owned lease/global released state and absence of old tracked descendant generations were checked after terminal0; shared lease is free. A subsequent CPU serving preclaim refused the original42/32GiB process/reserve requirement at73.31GiB available versus74. Native headroom passed. A separately reviewed stricter40GiB process limit with unchanged32GiB reserve is being prepared; no native GPU smoke has launched.


Independent actual outcome review accepted the full16 fixed-target result (SHA3b7add8f870992d90f735fe2ab2041f9154f246853d1eff0d6df79799c89b2e3;219 pins root-verified). It rebuilt all47,069 raw target/category/item/stratum reductions with exact Fraction sums, reproduced the product audit byte-for-byte, and rejected8 malformed outcome/provenance variants. Every example improved total and added-Metal NLL. Added-Metal predictions corrected36 tokens and regressed39; no code top1 benefit is established. Overall net69 additional correct predictions equals the boilerplate net69 gain. This supports narrow target-likelihood improvement, not generated correctness/performance.

The next-trial development proposal remains unactivated: its105-row component is linked by the broad declared mlx-sushi-connected-training label, leaving no MLX/MLX-LM/Sushi/MLX-VLM/mlx-serve development rows and23 TensorFold rows. Root verified200 proposal pins but requested evidence-based review of actual change lineage versus allocation/history buckets before using that split to select hyperparameters. No current or protected data changed.


## Native adapter smoke running with tighter memory cap

A separate process40/native82/reserve32 profile passed73 root CPU tests and its independent115-pin review (03daeffbf6436d3f241c7592cfd655d27ddcaba1158b06cbab159129d5d931a8). Fresh CPU preclaim observed72.399GiB outer availability against72 required, native82.651GiB against81.8677 required and source-derived filled-process28.651GiB against40 allowed. Reserve, native bank target74GiB, binary, arithmetic, base/adapter and six-phase schedule are unchanged; the process/RSS guard and its source bound are stricter. The original42 refusals are preserved.

Root launched session13668; controller94258 owns the serial lease and first frontend94313 is running the actual native base condition. Runtime fit, controls, adapter participation and all six phases remain pending. This short33+4 / own-prefix+1 smoke does not authorize the separate long774+4096 generated experiment or establish generated-kernel quality.


## Six-phase native smoke completed: controls pass, strict cache comparison fails

Session13668 completed0 with all6 planned native processes. Absent versus matched zero-B raw logits and generated IDs passed exact controls in both cached and fresh schedules. The trained adapter changed selected logits (position33 maximum absolute difference8.640625, same argmax). All3 strict cached/fresh comparisons failed: base and zero-B maximum1.59375, trained maximum1.25. No tolerance was widened; historical strict failure remains failed, trainer-forward parity remains false, and generated-kernel quality remains unverified. This candidate must not be described as a whole-engine precision repair. The predeclared result is short-serving evidence only, pending independent actual-outcome review. Fresh root read-only observations captured all6 global released states/absent own holders and2 full process tables; shared lease is free.

## Long-context placement and split review

CPU construction arithmetic for context8192 (minimum4870 for prompt774+decode4096) gives a39.6534GiB construction floor and93.6534GiB required native availability under box82. Studio current availability around83GiB does not qualify. A fresh authorized M5 CPU check found105.4227GiB native/approximately103.0GiB outer availability and exact matching hashes for both external WebP libraries. A proposed60/82/32 profile is arithmetically plausible there (source-filled upper51.4227GiB), with unchanged32GiB reserve. This is not a reservation, runtime fit, or M5 serving qualification; exact package/library resolution, model/host continuity, controls, capacity and conformance remain pending. No new model download is planned.

Independent lineage review found no identified exact patch/commit/raw/prepared, captured parent/known-family, complete-file or substantive64-token changed-code overlap across original229/protected16 in checked local evidence. It covers198 training+16 protected diffs and220/229 training patch identities;31 diagnostics lack direct diff evidence. Protected16 are all llama.cpp, so this remains a single held-out-engine evaluation. Draft dev41 crosses4 captured ancestry links; no64-token added target span was found in the four serialized training rows. The archived DS4 complete-file equality therefore must not be called full serialized answer leakage. No dataset/split was changed.
