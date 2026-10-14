# M5 evaluation preparation — 2026-10-13

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
