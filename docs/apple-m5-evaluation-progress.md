# M5 training and evaluation progress

## Current status — 2026-10-15 UTC

The completed run uses **229 complete training examples: 186 exact edits and 43 diagnostics**, with all **458/458** planned updates saved. Only layer 39 attention projections `wq_a` and `wo_b` have rank-4 LoRA adapters: 78,848 trainable parameters. M5 training and its owned wrapper exited successfully and released their lease. The final checkpoint, optimizer state and full history were transferred and independently audited. New source collection remains paused.

Full-history pairing covers all 229 examples twice: 228 had lower pre-update loss on their repeat presentation and one had higher loss. Token-weighted loss fell from 0.35035952068519527 to 0.338869637879154 (**3.27945%**). The first pass took **20.6384 hours**, the repeat pass **1.55166 hours**; median example times were 216.412 and 17.347 seconds. This is training-fit evidence, not held-out or generated-kernel quality. Timing includes cache and other warm-up effects rather than an isolated cache experiment.

The earlier update-150 evaluation used four fixed held-out items. Aggregate NLL improved 7.90%, but added-code NLL improved only 4.34%, with equal total correct added-code predictions. Final evaluation uses all 16 frozen held-out items and 47,069 supervised targets, comparing checkpoint458 with a matched zero-B control in separate fresh processes. Its five rationale tokens cannot establish broad reasoning improvement.

The concrete final evaluation plan and request passed independent review (500 pinned inputs). The initial root launch failed before any model execution because the inherited environment omitted `MLX_ENABLE_TF32=0`. The initial receipt remains preserved under `final458-evaluation-process40-v2/concrete-execution-v1/wrapper-receipt.json`. Independently reviewed recovery verified terminal processes and released only that owned lease. A fresh admission passed with the corrected environment. The retry actually launched at 03:21 UTC: root session50021, pair37371, zero-B evaluator37407, directly confirmed live while hashing checkpoint payloads. The trained condition follows sequentially in its own fresh process. No final held-out result exists yet.

Generated-output evaluation remains pending serving qualification. A model-free projection microprobe and independent mathematical replay reproduce the first two BF16 projection discrepancies; using F32 projection then casting to BF16 matches the declared ideal reference on the captured cases. A narrow serving source candidate is staged, not hardware qualified. A separate fixed-engine/fixed-schedule evaluation contract is under independent CPU review; historical strict cache/fresh failure and trainer-parity=false remain explicit. No generated-kernel benefit is claimed.

Current execution update03:37UTC: the redundant zero-B attempt was intentionally stopped with one reviewed SIGINT to its exact owned worker and its own lease released after terminal checks. Four actual intermediate150 zero-B examples reproduce all11077targets, token losses, predictions and supervised-logit hashes from the historical full16 baseline. A separate final-only comparison explicitly reuses that baseline; it does not claim a new endpoint-A control was run. The40GiB attempt refused before model load at71.67GiB available against72required. A separately reviewed36GiB process candidate retains32GiB reserve, all16items, allcache capacities and arithmetic. Root session44309 is now live: pair47229 and checkpoint458 evaluator47242 directly PS-confirmed while hashing payload7/110. No duplicate GPU job or unrelated process stop occurred.

M5 cache metadata inventory finds matching boundary39 training entries for all229examples,49,737,199,860payload bytes(46.32GiB). Root independently replayed all229keys with the actual frozen key function. This is metadata coverage, not fresh tensor-checksum or cross-run-hit qualification. Cache-first experiment policy is in `docs/apple-cache-aware-experiment-plan.md`; a reusable CPU inspector is being staged. F32 serving candidate linked successfully and its artifact passed independent static review; no actual device execution has occurred.

Preserved historical milestones follow. Their statements about pending launches, source collection and the earlier 198-item campaign apply to their recorded stage, not the current run.

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
