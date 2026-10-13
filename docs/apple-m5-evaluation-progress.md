# M5 evaluation preparation — 2026-10-13

The sealed next run contains 198 training items (159 exact edits and 39 diagnostic items), with 16 protected held-out items and 396 planned updates. Training has not started. The existing 397 GiB quantized checkpoint is transferring to the authorized M5 Max, followed by full payload SHA-256 verification.

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
