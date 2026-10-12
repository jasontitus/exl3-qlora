# Final resource and recovery review

**Later results, 2026-10-12:** the full update completed, but the original adapter
and all three tested derived strengths worsened training-row NLL. See the
[current diagnosis](first-step-diagnosis-update.md) and
[complete strength receipts](ktune-strength-diagnostics.json). The review below
retains its earlier 511-test, pilot-running cutoff.

Fresh read-only adversarial pass over the installed/staged source while the
5,809-token full-model pilot was running. No source edits, GPU work, benchmark
reruns, or modifications to pilot artifacts were performed for this review.
This reviewer previously implemented portions of the resource/UI wiring; this
is a fresh code/process review, not a claim of organizational independence.

**No new concrete blocker requiring the current pilot to stop was identified.**
This is limited to the reviewed contracts and does not certify the unfinished
full-model run or the usefulness of a learned kernel optimization.

## Frozen-source check

The 36 `.py`, `.metal`, and `.html` files directly under installed
`apple/src/exl3_mlx` and the staging tree were byte-identical. The SHA-256 of the
sorted compact JSON filename-to-SHA256 map was
`4330203174b82c8d8944f5e8ec41ff6b7346fa2c8256e7ee48f2a06cdb73012c`.
Bytecode caches were excluded. Dependency/runtime checks remain the project's
separate recorded identity; this source comparison does not re-audit them.

## Contracts checked

- [Resource resolution](../src/exl3_mlx/resources.py#L16) keeps the development
  profile at 80 GiB process, 32 GiB reserve, 8 GiB packed, zero decoded and 256 MiB
  allocator cache. Explicit resource fields override presets; nonfinite,
  noninteger cache and overcommitted cache configurations reject. Cache allowance
  is inside the process budget. [Memory headroom](../src/exl3_mlx/memory.py#L50)
  still requires process budget plus reserve at startup and monitors reserve.
- [Project manifest validation](../src/exl3_mlx/sft.py#L296) exempts only declared
  resource fields from numerical-setting comparison. Chunk size remains an
  immutable numerical field, enters every model child command through
  [engine options](../src/exl3_mlx/sft.py#L515), and is checked against checkpoint
  contracts. Explicit CLI profile selection clears old JSON resource defaults
  before explicit numeric flags are applied; it does not clear chunk size.
- Required and pending provenance block ordinary open/train. The
  [binder](../src/exl3_mlx/sft.py#L346) pins report/file hashes before copies and
  verifies a published bundle before completion; repair access does not itself
  authorize training with incomplete evidence.
- The [project lock](../src/exl3_mlx/sft.py#L210) retains one inode and is inherited
  by the [GPU child](../src/exl3_mlx/sft.py#L443). An orphaned child therefore
  retains exclusion even if the launcher exits. Resume uses a new attempt
  directory, and [recovery](../src/exl3_mlx/sft.py#L832) finds complete checkpoints
  rather than trusting a potentially stale latest pointer.
- [Checkpoint validation](../src/exl3_mlx/sft.py#L757) checks payload hashes, data,
  model, adapter and numerical contracts before reuse. [Optimizer loading](../src/exl3_mlx/training.py#L236)
  checks tensor structure, RNG, step and cursor before continuing. Stops are
  honored at update boundaries; failure before checkpoint publication can lose
  that in-flight update, not silently declare it saved.
- [Pruning](../src/exl3_mlx/workflow.py#L299) is limited to the current attempt and
  protects recent/latest/best checkpoints and the best references of retained
  checkpoints. [Selection](../src/exl3_mlx/sft.py#L836) uses finite saved validation
  loss when available, otherwise latest. The training-only ktune fit check
  [verifies counts and scope](../src/exl3_mlx/sft.py#L851) rather than calling fit
  evidence a held-out quality result.

## Remaining operational limits

1. **Studio does not adopt externally launched jobs.** Its
   [running flag](../src/exl3_mlx/studio.py#L41) follows the process it launched.
   Opening a project with an active CLI job can show its logs while leaving
   Train enabled and Stop disabled. The inherited project lock rejects a second
   trainer. Use `exl3-mlx stop PROJECT` to request a safe stop for the external job;
   the UI's inactive Stop control does not mean no training is running.
2. **Process interruption is distinct from host power loss.** Checkpoints publish
   a complete staging payload by [rename](../src/exl3_mlx/training.py#L199), with
   hashes verified on reuse. They do not explicitly fsync every checkpoint file
   and parent directory. Atomic publication should not be described as a tested
   guarantee against abrupt power loss. Corruption fails closed and may require
   restoring an earlier preserved checkpoint or backup.
3. **Integrity has a defined boundary.** Metadata hashes and in-process shard
   fingerprints do not authenticate every byte of the hundreds-of-GiB frozen
   payload, nor defend against an actor rewriting both a local manifest and its
   checksums. Keep base files immutable. A Studio snapshot is a lightweight saved
   display, not a fresh integrity audit.
4. **No automatic source migration.** Exact project continuation requires the
   recorded source/runtime. Resource overrides cannot bypass that identity.
   Preserve the pinned implementation/environment for existing experiments.
   Standalone exported-adapter engine use must explicitly select its recorded
   chunk size; the project workflow forwards it automatically.
5. **Evidence remains bounded.** The decoded-cache prototype was numerically
   exact at equal chunks but slower in the measured whole-MoE sequence, so it is
   opt-in. A wider chunk's matrix-level tolerance and speed screen are not proof
   of full-model parity or end-to-end speed. Successful full-model save, reload,
   resume, fit checks and execution-based kernel quality are separate gates.

The earlier targeted resource/project/CLI/Studio CPU suite passed 111 tests.
The parent reported 511 full-suite tests passing before the active pilot; this
review did not rerun or independently reproduce that GPU suite. No new broad
quality-pass label follows from this review.
