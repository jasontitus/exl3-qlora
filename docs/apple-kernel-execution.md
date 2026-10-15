# Running generated-code evaluation stages

`exl3-mlx-kernel-run` executes pinned, trusted backend commands and produces
`kernel-eval-result-v1`, which `exl3-mlx-kernel-eval` reads directly. It never
executes a command taken from a model response. These tools do not import MLX.

The current integration imports already captured responses. Live model generation
is still performed by the existing reviewed native request/controller path; the
new CLI does not automatically create or approve a generation request. Likewise,
task-specific source/ABI acceptance and independent numerical/timing review remain
backend responsibilities. A backend assertion is not independent certification.

## Try the working CPU example

From the repository root, using the project's Python environment:

```sh
PYTHONPATH=apple/src python apple/examples/kernel_eval_replay_example.py --output /tmp/kernel-replay-example
PYTHONPATH=apple/src python apple/examples/kernel_eval_replay_example.py --output /tmp/kernel-replay-bad --malformed
```

Both directories must be absent. The first fixture produces a valid exact edit;
the second retains a hunk-count failure. Each writes the captured fixture,
baseline, pinned replay/backend/execution manifests, actual subprocess logs,
`run/execution/result.json`, `suite.json`, and a JSON/Markdown comparison. These are
explicitly synthetic examples, with source validation, compilation, numerics and
timing unrun. They cannot establish model quality or a performance winner.

## Import a real captured response

The default replay protocol remains exact git-apply edits. For the bounded
complete-file interface, add `"protocol": "kernel-full-files-v1"` inside `replay`.
Its generation file must be the exact verified assistant content, with the
terminal EOS excluded by the capture layer. The importer does not strip EOS,
extract JSON from prose or Markdown, repair code, or validate token provenance.
The capture result must bind that content to the actual model output separately.

The complete-file response is one JSON object with exactly `format`, `rationale`
and `files`. `format` is `kernel-full-files-v1`; `files` contains exactly
`kernel.metal` and `dispatch.json` as strings ending in newlines. Dispatch contains
only integer `threads_per_row`, either 1 or 32. This is a deliberately narrow
row-kernel contract, not a general project-edit format. The task baseline must
contain those same two files. Duplicate keys, extra paths, wrappers and unsupported
dispatches fail. Limits are 128 KiB response, 64 KiB kernel, 256 bytes dispatch and
2,048 bytes rationale. Source/ABI, compiler, numerical and performance gates still
follow parsing.

A complete-file response may retain the baseline unchanged. The replay report
records `unchanged_baseline: true`; that is a valid no-change decision, not an
optimization or speedup. The exact-edit protocol still rejects no-op patches.
Use a distinct protocol identity in the suite, and compare base and adapters
under the same response contract; changing formats is not a training benefit.
The suite's protocol digest remains supplied registration evidence: this bridge
pins the chosen parser but does not interpret a protocol manifest to attest that
the supplied digest names that parser. Keep registration and capture review as
separate requirements when importing real results.

Create `replay-request.json` with this shape. File references use `{path, sha256}`;
relative paths resolve against the request's directory.

```json
{
  "format": "kernel-eval-replay-request-v1",
  "initial_result": {"path": "capture-result.json", "sha256": "<64 hex>"},
  "replay": {
    "generation": {"path": "generation.raw", "sha256": "<64 hex>"},
    "baseline": {
      "kernel.metal": {"path": "task/kernel.metal", "sha256": "<64 hex>"},
      "dispatch.json": {"path": "task/dispatch.json", "sha256": "<64 hex>"}
    },
    "required_paths": ["kernel.metal"]
  }
}
```

`capture-result.json` is the suite's result schema: task/adapter/prompt/protocol
identities, the predeclared budget and attempt, and a `tracks.strict.generation`
stage with the actual capture evidence. An EOS/cap or infrastructure failure must
be recorded truthfully there. Import does not attest those generation assertions.

```sh
exl3-mlx-kernel-run --import-replay replay-request.json --sha256 REQUEST_SHA --output prepared
exl3-mlx-kernel-run --plan prepared/execution-plan.json --sha256 PLAN_SHA --output replay-run --execute
```

The first command prepares and validates without spawning. Add `--execute` to it
to prepare and run replay in one command. Hashes printed in generated references
or calculated from the exact files are the values to pass; the placeholder text
above is not a digest. Outputs are exclusive: repeat execution cannot overwrite a
completed result.

Replay accepts one DeepSeek native `exec_command` call containing a quoted
`git apply [--whitespace=nowarn] [-] <<'DELIMITER'` heredoc. The stdin dash is
optional; standard `diff --git` and `index` headers are supported. It allows edits
to the supplied existing UTF-8 files only, at most 16 files/256 KiB per file and
256 KiB response/patch. Counts, coordinates and context must match exactly. No
shell execution, fuzzy application, recounting, extraction or repair occurs.
Added/deleted files, binary diffs and arbitrary shell wrappers are outside this
version. Historical stricter parser receipts remain separate protocols.

## Connect a trusted compiler or owned numerical controller

An execution plan contains an existing result and ordered steps:

```json
{
  "format": "kernel-eval-execution-v1",
  "initial_result": {"path": "replay-run/result.json", "sha256": "<64 hex>"},
  "track": "strict",
  "steps": [
    {"backend": {"path": "compiler-backend.json", "sha256": "<64 hex>"},
     "input": {"path": "approved-compile-spec.json", "sha256": "<64 hex>"}}
  ]
}
```

A backend manifest binds an executable, its trusted implementation files, argv,
environment, working directory and semantic result predicates. All references
resolve against the execution-plan directory. This compiler example adapts the
existing `compile_reviewed` interface without modifying it:

```json
{
  "format": "kernel-eval-backend-v1",
  "kind": "cpu",
  "executable": {"path": "/canonical/path/to/python", "sha256": "<64 hex>"},
  "code": [
    {"path": "/qualified/harness/build_candidates.py", "sha256": "<64 hex>"},
    {"path": "/qualified/harness/admission.py", "sha256": "<64 hex>"}
  ],
  "argv": ["/qualified/harness/build_candidates.py", "--spec", "{input}",
           "--sha256", "{input_sha256}", "--output", "{output}"],
  "cwd": "{backend_dir}",
  "env": {"PATH": "/usr/bin:/bin", "HOME": "/approved/compiler/home", "PYTHONDONTWRITEBYTECODE": "1"},
  "timeout_seconds": 600,
  "result": {
    "path": "{output}/build/build-receipt.json",
    "bindings": [
      {"path": ["format"], "equals": "generated-kernel-CPU-build-v1"},
      {"path": ["admission", "sha256"], "equals": "{input_sha256}"}
    ],
    "stages": {
      "compile": [{
        "when": [{"path": ["status"], "equals": "compiled_and_linked_not_executed"},
                 {"path": ["GPU_executed"], "equals": false}],
        "exit_codes": [0], "status": "passed", "outcome_class": "unspecified"
      }]
    }
  }
}
```

Replace paths/hashes with the installed qualified harness and its complete trusted
source dependency set. This example starts after a passed source/ABI stage; the
existing builder still consumes its reviewed specification and fixed controls.
The example deliberately does not label every compiler-process failure as a model
failure: a missing SDK or binding rejection is infrastructure. A trusted backend
can provide a separate exact completed-compiler-error predicate for model-output
failure. No pass is inferred from exit zero.

For an existing numerical controller, use `kind: owned_device_controller`,
`timeout_seconds: null`, and its original pinned `--request {input} --sha256
{input_sha256} --execute-reviewed` argv. Bind the controller result's
`request_sha256` and exact completion/release fields. Its result may live at the
exact request-owned output path outside the bridge's log directory; the bridge
requires that result path to be fresh and canonical before launching. The
controller retains its normal queue, source, resource, lease, deadline and
descendant-cleanup checks. Execute requires both `--execute` and
`--allow-owned-device`. Merely setting this kind is not a grant of GPU admission.

The existing generated controller distinguishes a complete candidate rejection
(`candidate_rejected_numerically`) from an incomplete/unsafe execution. Map only
the former to `failed/model_output`. Its successful native result still says
`candidate_passed_all224_pending_independent_audit`; map that to `pending` until
the existing independent numerical gate has passed. A subsequent trusted CPU
audit step may replace the pending numerical stage with a passed or failed stage.
Queue refusal, timeout, missing output, ambiguous predicates or binding mismatch
remain `refused/infrastructure`; they do not become model-quality losses.

Timing must be a separate backend step, with a passed numerical prerequisite.
The existing timing builder/controller must continue to enforce the numerical
gate, matching workload/precision/device, warm-up and paired schedule. A passed
timing mapping requires `measurements_path` pointing to actual
`{workload_sha256, device, cache_state, unit: "ms", samples: [...]}`. The comparison
tool labels its median ratios descriptive; qualified paired inference belongs to
the timing backend's original audit. The bridge does not manufacture timing data.

The same `measurements_path` also accepts the multi-workload list in
[the suite schema](kernel-eval-suite-schema.md). Keep all shapes and their explicit
statuses in that list; do not flatten different shapes into one sample vector.
This requires no change to controller argv or the execution bridge. Numerical
acceptance and complete workload/audit validation remain with the trusted backend.

### Existing transpose controller: data bindings, no controller fork

The reviewed transpose implementation already uses a shared `run.py --run-dir
RUN --job JOB` and a shared lease wrapper. A backend manifest invokes the **wrapper**
with its pinned request; it must not call `run.py --execute` outside that ownership
path. The relevant mapping is:

```json
{
  "kind": "owned_device_controller",
  "argv": ["/qualified/shared_gpu_lease_studio.py", "run",
           "--request", "{input}", "--sha256", "{input_sha256}",
           "--receipt", "/fresh/operation/wrapper.json"],
  "timeout_seconds": null,
  "result": {
    "path": "/fresh/operation/wrapper.json",
    "bindings": [
      {"path": ["schema"], "equals": "shared-gpu-lease-command-receipt-v1"},
      {"path": ["request_sha256"], "equals": "{input_sha256}"}
    ],
    "stages": {"timing": [{
      "when": [
        {"path": ["status"], "equals": "completed"},
        {"path": ["exit_code"], "equals": 0},
        {"path": ["lease", "released"], "equals": true},
        {"path": ["lease", "cleanup_verified"], "equals": true}
      ],
      "exit_codes": [0], "status": "pending", "outcome_class": "unspecified",
      "reason": "Captured and released; independent timing audit remains required."
    }]}
  }
}
```

This excerpt needs the normal format/executable/code/cwd/env fields. It deliberately
maps a completed capture to **pending**, not to a speed result. A missing or
unexpected wrapper outcome remains infrastructure refusal. Subsequent accepted
audit data can populate the 12 workload entries through a trusted CPU backend or
a pinned imported result. Independent review and conversion of that harness's
native records remain explicit; there is no automatic source-safety reviewer or
arbitrary task compiler hidden in this CLI.

An exact local, **default-only** example is saved at
[`kernel-multiworkload-integration-v1`](../../kernel-library/next-round-v5/kernel-multiworkload-integration-v1/README.md).
It pins the real completed base transpose request and validated source, with a
passed numerical initial result. Its default validation ran under a subprocess
trap and created no execution output. The historical wrapper result already
exists: do not execute this example or repurpose its output paths. Future runs
need fresh prepared requests and ordinary concrete admission; they can reuse the
same backend manifest structure without editing Python.

The test fixture `apple/tests/fixtures/kernel_timing_transpose_base.json` retains
all 12 × 30 candidate samples from the accepted base transpose audit, with exact
source hashes and seconds-to-milliseconds conversion. It tests the representation,
not a new GPU run or cross-adapter performance claim. Its workload identities are
specific to that saved manifest; a real cross-adapter importer must bind a shared
source-free workload definition to avoid equating different work by label alone.

Every executed step retains argv, input/backend hashes, stdout/stderr, exit code,
timeout and semantic result evidence. CPU timeouts stop only the newly created
CPU process group; device timeouts and recovery remain exclusively controller
responsibilities. Code/input pins are checked before and after work. Interrupted
or mutated-evidence runs remain preserved and cannot be silently resumed.

## Compare results

Reference each emitted `result.json` in the suite's `results` list, with its hash:

```sh
exl3-mlx-kernel-eval --suite suite.json --output comparison.json --markdown comparison.md
```

Use identical task/prompt/protocol/budget identities for comparable adapters.
Keep normalized diagnostics in the `normalized` track; they never replace raw
strict failures. Unrun infrastructure stays unresolved. The real imported affine
panel is in `apple/reports/kernel-eval-panel-v1`; its replay results and historical
numerical diagnostics are distinct from this synthetic example.
