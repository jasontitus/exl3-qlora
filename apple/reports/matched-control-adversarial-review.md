# Matched-control audit review — 2026-10-12

The repaired auditor has no remaining blocking finding within its stated scope:
receipt consistency for one previously inspected training example and one fresh
update per activation mode. No GPU work or control training was performed during
this review. The control output directory did not yet exist at the final check.

## Confirmed gap and repair

The initial auditor verified the hash of the prior BF16 audit but did not expand
that audit's checkpoint bindings. It also omitted an explicit check of the
treatment checkpoint's activation mode. A CPU fixture demonstrated acceptance
after changing the treatment's saved mode to `checkpoint` and repairing its local
trainer checksum, while keeping the plan and post-evaluation unchanged. It
incorrectly returned `matched_runtime_and_settings=true`.

The repaired `verify_treatment` requires the complete expected prior input set,
checks all 18 hashes, binds checkpoint/config/run/report/post paths, checks BF16
mode in each applicable artifact, and reruns the pinned prior auditor. The new
result must exactly equal the bound prior audit. The original exploit now fails;
additional tests cover repaired outer hashes, missing bindings, wrong checkpoint
paths, and altered baseline/report metrics. Independent CPU run: **24 passed**.
Revalidation of the actual treatment evidence also passed all 18 bindings and
reproduced the complete prior audit exactly.

## Supported comparison and limits

The plan keeps all 40 attention adapter layers, original seed and first Adam
update, 5,808 input positions, and all 811 supervised targets. It compares each
mode's fresh post-update NLL to that mode's own complete uncached pre-update
training objective. This avoids conflating the different zero-adapter forward
baselines. The absolute direction band is 1e-6; regression, unresolved, and
non-reproducing outcomes remain reportable. Historical tensor/logit reproduction
is separate from the fresh comparison.

Training runtime provenance still needs the separately reviewed launch receipts.
The v2 training report has no source identity, and evaluation source hashes alone
cannot establish which code ran training. The auditor now names its flag
`matched_evaluation_runtime_and_resolved_settings` and states this limit. Preserve
the planned command, working directory, environment, pinned snapshot, start/end
times, and exit status for the new control; inspect the existing BF16 launch
evidence alongside it before making the bounded matched-intervention claim.

The result can describe whether bypassing the specified activation-quantizer
bundle changed this single update from regression to descent. It does not isolate
one quantizer, prove a unique gradient defect, establish generalization, or show
generated Metal code correctness/performance. Routes may differ across modes.
Base identities cover metadata/manifests rather than every frozen payload byte.

Exact reviewed source/test/plan hashes and actual treatment revalidation are in
`matched-control-independent-review.json`; the CPU test receipt is
`matched-control-independent-cpu.xml`.

## Launcher follow-up: identified gates closed

Reviewed frozen `run_matched_control.py` SHA256
`c09e4a0de0f2b5a9e747012ba810f24fb692fcf718cd48c91b7a3bf92a9d15b0`.
Scoped SIGTERM/SIGINT handlers now raise through the existing child cleanup path,
which terminates and waits for the child's process group. Repeat stop signals
are ignored during cleanup, failure receipts are written, and prior handlers are
restored. Fake-child tests exercise both signals and confirm evaluation never
starts afterward. This closes the identified ordinary-signal orphan risk; it is
not a guarantee against an uncatchable SIGKILL or host failure.

Preflight now enforces the exact planned Python executable, Python version, and
every package version recorded by the treatment, using metadata without importing
MLX or launching a model. The same checks run after each phase. Independent run:
**39 CPU tests passed** (24 auditor, 15 launcher), and actual `--verify-only`
preflight passed without creating a receipt or starting a subprocess. No
remaining blocker was found in these two repaired launcher gates.

The launch receipt must still be retained and reviewed after the real run; this
preflight does not claim a control run has occurred. The additional test receipt
is `matched-control-launcher-independent-cpu.xml`.

## Prospective resource amendment — independently reviewed

The resource plan preserves the original 80 GiB plan and config, then declares
76 GiB for new control training and 72 GiB for its evaluation. The treatment used
80 GiB. All four comparison phases retain the 8 GiB reserve. The amended config,
launch arguments, explicit caches, data/mask, optimizer, seed, chunk size, and
pinned runtime are checked against the original; only caps and fresh output paths
may differ. The final audit explicitly reports unequal resources and each phase's
resource map.

No 80 GiB control was launched. The prospective amendment instead binds the
separate standard-fit adapter child's startup failure: 86.48 GiB available versus
88 GiB required. It also binds the historical and BF16 training memory reports.
The historical checkpoint-mode MLX peak was 73.9483 GiB, leaving about 2.0517 GiB
under the new training cap. This is not a guarantee against transient allocation
or process RSS growth. Failed or partial outcomes must remain preserved, with no
silent cap retry or result-based selection.

The pinned-v2 resource code changes the MLX advisory limit and admission/watchdog
thresholds. Its workspace reserve calculation stays at 8,192 MiB for all these
caps, and explicit caches and configured math stay unchanged. Exact resource or
timing equality must not be claimed; numerical reproduction is still a separate
measurement.

An initial redundant-metadata gap was fixed during review: the declared changes
map, original config identity, observed peaks, and computed margin now must agree
with the authoritative settings and verified reports. Independent verification:
**56 CPU tests passed** and actual amended `--verify-only` preflight passed
without creating files or starting subprocesses. Both amended output paths were
absent at review. No remaining blocker was found in this amendment's scope.

Frozen reviewed hashes are recorded in
`matched-control-resource-independent-review.json`; tests are recorded in
`matched-control-resource-independent-cpu.xml`.
