# Trusted fixed diagnostic campaign

`launch.py prepare` is CPU-only. It seals one declaration containing the fixed,
reviewed eight-task diagnostic plan and **all three conditions** (base, prior,
fixed final). Both trained checkpoints must exist before sealing. `run` then
starts 24 generation processes and three scoring processes, serially. It never
selects a checkpoint, retries a response, or resumes an uncertain campaign.

The request schema is `trusted-diagnostic-generation-request-v1`. Required fields:

- `diagnostic_plan`, `diagnostic_plan_sha256`, `review`, `review_sha256`, `scorer`,
  `scorer_sha256`: the relocated M5 plan, relocation review and pinned scorer.
- `runner`, `runner_sha256`, `helper`, `helper_sha256`: reviewed `run_m5_plan.py`
  and frozen `prepare_agent_corpus_run.py`; reused only for local process lifecycle
  and runtime/payload verification. No model imports during preparation.
- `runtime`, `snapshot_sha256`, `python`, `hostname`, `model`.
- `payload_receipt`, `payload_sha256`, `payload_source_manifest`,
  `payload_source_sha256`: completed full-byte target verification and exact
  source manifest. This is distinct from the engine's six metadata hashes.
- `inference`: `activation_mode: "checkpoint"`, explicit `chunk_size`, exact
  `engine_semantics` string from the frozen runtime.
- `resources`: all resolved resource fields; decoded/prefix RAM/disk caches must
  be disabled. Packed and allocator caches remain explicit bounded resources.
- `conditions`: ordered objects `{name: "base", adapter: null}`, then prior and
  final `{name, adapter, step, settings}`. `settings` is the exact saved rank,
  alpha, attention/expert layer lists and activation mode. Both adapters must
  share these settings and the declared base/engine. The base CLI internally
  allocates zero-B rank16/alpha32 adapters; this is not claimed to be trained
  adapter configuration.
- `output`: a new, canonical, condition-owned directory outside runtime/model.

All paths are absolute paths **on the execution host**. Review/source copies
must be staged there. This tool intentionally has no path rewriting or remote
execution facility. Seal with the planned Python and `MLX_ENABLE_TF32=0`:

```text
python launch.py prepare --request REQUEST --sha256 REQUEST_SHA --output SEALED_PLAN
python launch.py run --plan SEALED_PLAN --sha256 PLAN_SHA --verify-only
python launch.py run --plan SEALED_PLAN --sha256 PLAN_SHA
```

Preparation binds package payloads (including native MLX/Metal), Python,
installed versions, runtime source, plan/messages/source/tokenizer, adapter
weights/config/state and the launcher. It verifies that each model file's size
and change times precede the full verification, then checks inventory and
inode/size/mtime/ctime between jobs. This avoids rereading 426 GB for every short
question. The model must remain quiescent; this is trusted local evidence, not
protection from malicious timestamp spoofing or an adversarial operating system.

Each generation child gets only the explicit TF32-off MLX mode, a controlled
Python path, disabled user site, no prefix cache and eight greedy output tokens.
Actual argv, cwd, environment, PID, start/end, status, stdout/stderr hashes and
report hashes are retained. Failed reports/logs remain, and an existing output
blocks reuse. SIGINT/SIGTERM cleanup is delegated to the pinned reviewed process
wrapper; its unsampled-orphan/SIGKILL limitations remain. Parent orchestration
must keep model jobs exclusive; this launcher does not kill unrelated work.

A condition is scored only after all eight successful generation reports pass
binding checks. Malformed/truncated answers are incorrect under the pinned
scorer. All reports/logs/scores are rechecked at completion. Success means
`completed_pending_independent_audit`, not a quality certification. These eight
correlated decisions from five previously inspected families assess narrow
recognition/evidence interpretation, not free-form reasoning, patch correctness,
kernel performance or untouched benchmark generalization.

Prior-adapter held-out NLL is a separate evaluation, not supplied by this
campaign. Its trusted wrapper must bind the completed training plan, exact 16
prepared rows/masks, baseline receipt, prior adapter and fresh evaluation report.
