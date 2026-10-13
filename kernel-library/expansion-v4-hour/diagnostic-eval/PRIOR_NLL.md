# Explicit prior-adapter NLL comparison

`prior_nll.py` is separate from the held diagnostic launcher. It does not run
generation. Preparation and verification are CPU-only; `run` explicitly launches
one fresh evaluator process, serially, with no retry/resume.

The request is `trusted-prior-nll-request-v1` with these absolute execution-host
paths and SHA256 fields:

- `launcher`, `launcher_sha256`: independently reviewed `launch.py`.
- `diagnostic_sealed_plan`, `diagnostic_sealed_sha256`: the CPU-sealed
  base/prior/final diagnostic declaration. It need not have been executed.
- `training_plan`, `training_plan_sha256`: frozen M5 native experiment plan.
- `training_launch`, `training_launch_sha256`: completed five-phase M5 launch.
- `resources`: explicit complete resource map with decoded and prefix RAM/disk
  caches disabled. Other bounded resource settings must be approved in this
  declaration; differences from the baseline are visible and do not change the
  requested mathematical configuration.
- `output`: new canonical directory outside the model/runtime.

```text
python prior_nll.py prepare --request REQUEST --sha256 REQUEST_SHA --output SEALED_PLAN
python prior_nll.py run --plan SEALED_PLAN --sha256 PLAN_SHA --verify-only
python prior_nll.py run --plan SEALED_PLAN --sha256 PLAN_SHA
```

Use the declared Python and `MLX_ENABLE_TF32=0`, as for diagnostics. Full checkpoint,
package, runtime, adapter and numerical bindings come from the diagnostic sealed
plan and are rechecked. This intentionally waits for completed baseline,
training and fixed-final evaluation, rather than accepting an arbitrary partial
project or selecting a checkpoint from its scores.

The pinned project helper revalidates exact data/provenance, fixed final step and
cursor, pre-training baseline, and all 16 prepared held-out rows. The observed
five-phase launch must agree on commands, numerical mode, process IDs, report and
log hashes, and full payload receipt. Its final checkpoint must equal the
predeclared diagnostic final condition. The prior command is derived from the
baseline command, changing only output path, adapter and explicitly declared
resource flags. Sequence length, ordered data, masks, token metrics, activation
mode and chunk size remain identical.

The evaluator runs under the project lock through the SHA-pinned process wrapper.
Its actual PID must match its evaluation report and differ from the baseline PID.
The frozen project validator and `fit_probe.validate_scores` check complete per-row
loss targets, lengths, token NLL, predictions, logits hashes and aggregate loss.
Input/artifact drift aborts; failed receipts and logs remain. Success is
`completed_pending_independent_audit`, not a performance or correctness certificate.

NLL is teacher-forced likelihood on previously inspected held-out examples. A
lower value does not establish generated edit correctness, compiled kernels or
runtime speed. Cache equivalence relies on the already qualified frozen runtime;
no cross-device/default-TF32 cache is reused. The trusted-local-process and
quiescent-checkpoint limitations in `LAUNCHER.md` also apply.
