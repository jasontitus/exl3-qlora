# Standard SFT design, usability and effectiveness review

Date: 2026-10-12. Scope: the new project API, CLI, Studio, import pipeline and
engine stop/retention changes. Three parallel review roles examined importer/
quality, engine/correctness, and process/usability; implementation and review
findings were converted into executable regression cases. The final code pass
found no additional blocking defect within this scope. This is not independent
full-model numerical validation.

## Implemented design

Studio → project CLI → `SFTTrainer` → existing EXL3 engine subprocess. The Python
API goes through the same project configuration, importer, contract and engine.
MLX-LM data processors, tokenization conventions, loss and LoRA forward are reused;
MLX provides autograd and Adam. The custom streamed model remains necessary for
the resident trunk, EXL3 experts and Engram banks. No second browser trainer or
claim of TRL/PEFT drop-in compatibility was introduced.

The standard default uses attention adapters only: rank 4, alpha 8, `wq_a` and
`wo_b` in all 40 layers. EXL3 experts contribute frozen forward/activation-gradient
computation. Advanced engine options still expose expert adapters.

## Findings fixed

| Area | Failure found | Implemented response |
|---|---|---|
| Data import | Common CSV, Alpaca and ShareGPT flows rejected; prepare/train length defaults differed | Strict schema normalization, optional column mapping, one inferred serialized length and no silent truncation |
| Data leakage | Related variants or identical prompts with different answers could cross splits | Union lineage IDs, token duplicates and prompt token identities before group splitting; reject overlap in explicit splits |
| Notebook isolation | Memory watchdog exits its process | Model operations run in child processes; caller retains the project and completed checkpoints |
| Process ownership | Dead caller could leave a GPU child running and allow another run | Persistent file inode with inherited `flock` descriptor; no unlink/unlock race |
| Implementation identity | Parent import could differ from child installed package; Metal changes omitted | Explicit package path in child environment; Python/Metal/UI hashes, runtime versions and dependency lock receipt |
| Checkpoint selection | Pruning could delete best adapter or a retained checkpoint's prior-best dependency | Evaluate before pruning, save improved checkpoints, publish best/latest atomically and pin needed references |
| Wrong experiment | Lower-loss copied checkpoint could win selection | Verify hashes and exact data/training/validation/base/adapter contract before cataloguing candidates |
| Changed input | A new dataset passed to an existing project was silently ignored | Reject fresh dataset arguments for an existing immutable project; explicit `from_project`/resume required |
| Stop timing | Mid-update stop could save wrong counters; immediate UI stop could be cleared | Honor stop at completed-update boundaries; preserve requests newer than the published run start timestamp |
| Recovery | Final checkpoint existed but stale run status skipped comparison | Reconcile verified completed steps and allow same-target resume to finish comparison |
| Test exposure | Failed comparison could reuse test as apparently unseen | Persist exposure before launching evaluation; label repeated accesses |
| Chat evaluation | Multi-turn/system context absent from previews | Generate from the preserved full message prefix; enforce identical prompt IDs and cap |
| False success | Matching capped prefix or invalid EOS receipt could look successful | Require a complete answer with EOS; verify expected prompt IDs, cap, EOS position and decoded text |
| UI results | Opening another project retained old results; refresh lost session token | Clear dependent panels and preserve local session authentication across refresh |
| GUI security | Local web endpoints could expose files/actions to unrelated origins | Loopback binding, random token, Host/Origin checks, bounded uploads, generated upload names, argv-only subprocesses and escaped text |
| Progress polling | Hashing all optimizer payloads repeatedly competes with streamed weights and races pruning | Lightweight UI snapshots/log tails; checkpoint integrity checks occur under the project lock |
| Comparison receipts | Adapter metrics overwrote the selected-checkpoint field, falsely marking a current comparison stale | Keep `selected_checkpoint` separate; read earlier nested receipts correctly; regression covers current and superseded checkpoints |
| Export | Partial optimizer receipt could imply resumability | Export inference weights/config/bundle only; resume remains in the complete project |

## Verification

The integrated suite passes **263 tests and 27 subtests**. It includes small
actual Metal numerical tests, optimizer continuation and best-retention tests,
39 importer cases, 43 project/API boundary cases, 25 CPU engine-control cases and
13 Studio boundary cases. These categories overlap the overall suite. Studio's
actual JavaScript executes in a Node VM for project-switching and refresh tests.
The HTTP handler is exercised with in-memory requests for token, Host, Origin,
upload and action checks. None of those mocks substitute for the real model run.

Manual browser testing confirmed CSV preparation, split counts, context/answer
mask previews, disabled controls during jobs, refresh authentication, a real-model
resume from step 2 to step 3, paired comparison and adapter export. Ruff
passes. An offline wheel build succeeds and includes the HTML interface, SFT
modules and Metal source. Full-model pilot, resume, comparison and export evidence
is recorded separately in `standard-sft-verification.json`. The three-update pilot
used at most 10.04 GiB observed memory. One synthetic held-out row improved from
9.34 to 3.70 NLL, yet both capped greedy prefixes stayed `I can help,`. Neither
answer passed exact-answer-plus-EOS grading. This is a useful counterexample to
calling lower loss a successful behavioral fine-tune. Actual-tokenizer API import
also passed for all three common structured formats in `standard-sft-api-formats.json`.

## Remaining usefulness limits

- The standard interface is usable for short SFT experiments on the supported
  V4.1 streaming repack. It is not a generic EXL3 architecture loader, original
  shard converter, Qwen trainer, or external PEFT/MLX-LM adapter integration.
- The 512-token total limit excludes most complete Metal optimization patches.
  A previous audit found 19/20 candidate patches too long. A friendly interface
  does not resolve that model/backend limitation.
- The project API deliberately exposes a subset: batch one with accumulation,
  last-answer supervision, attention LoRA, fixed Adam settings. Packing,
  schedulers, all-assistant masks, tools/multimodal data and distributed training
  are not implemented by the facade.
- Group metadata improves splits but cannot infer every semantic family. A
  structurally held-out example can still be a near-duplicate conceptually.
- Lower held-out NLL and one matching answer do not establish useful task quality.
  Default comparisons evaluate only a bounded subset and disclose its size.
- Correctness and speed of generated Metal patches still need compiler/runtime,
  numerical and on-device benchmark gates. Those task gates are not yet integrated.
- Native independent full-model parity remains unverified. Activation rounding
  uses the existing straight-through gradient approximation.
- Memory protection is a polling watchdog and advisory MLX limit, not an OS hard
  quota. Longer sequences can exceed a budget within an allocation.
- Source/runtime checks intentionally block exact resume across implementation
  changes; automatic migrations are not supported. Checkpoint retention applies
  per attempt, so repeated resumptions accumulate disk usage.

For a Metal optimization dataset, prioritize verified before/after code and
benchmark evidence, keep commit families together, and score held-out generated
patches by correctness and measured speed/memory. Git success messages alone are
insufficient labels. See `metal-training-assessment.md` and `../SIMPLE_FLOW.md`.

The final report-key correction was applied after the measured pilot. Original
source, project/comparison receipts and HTML are preserved under that project’s
`before-report-fix/`; `reporting-fix.json` records both implementation identities.
Only `evaluate` report keys and `status` receipt reading changed. Existing measured
losses and generations were preserved, not recomputed or relabeled as new results.
The pilot’s project receipt was explicitly updated after that audit; general
automatic implementation migrations remain unsupported.
