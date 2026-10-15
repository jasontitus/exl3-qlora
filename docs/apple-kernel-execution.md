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
