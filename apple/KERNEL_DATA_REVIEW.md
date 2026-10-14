# Review kernel optimization training examples

Exact patch replay proves source reconstruction. It does not establish that an
optimization preserves its accepted input contract, improves performance or is
an appropriate answer to the supplied task. Review those questions separately.

For every proposed positive edit, read the complete selected change, including
host dispatch, build registration and tests. Trace accepted shapes, dtypes,
layouts and parameter ranges through callers. Check boundary values against the
old behavior: an optimization benchmark at the default parameter does not cover
all inputs that validation still accepts. A changed operator may introduce a new
restriction even when its common-case output is correct.

Compare existing tests before and after the edit. Record removed dtypes/shapes,
relaxed tolerances, disabled backends and skipped assertions explicitly. Such
changes need a numerical or contract justification; new tests do not compensate
silently for lost coverage. Preserve associated test edits in source evidence.

Bind performance reports to the measured baseline and candidate revisions,
hardware, precision, shape, timing scope and reported uncertainty. Compare those
revisions with the captured merge. If code differs, keep the report as historical
evidence rather than assigning its speedup to the merged source. Preserve missing
units or benchmark commands as unknown; never fill them by guessing.

Match the supervised answer to the agent-visible prompt. A baseline-only task can
teach a proposed mechanism and validation plan. A report-conditioned review can
teach which claims the report supports. Do not claim achieved speedup or parity
unless the observations are supplied, and distinguish inferred rationale from
upstream statements. Include the complete actionable edit in the selected format.

Keep three distinct outcomes:

- Positive edit: source reconstruction, contract review and evidence scope pass;
  local compile/numerical/performance qualification remains separately recorded.
- Critical review or repair: the prompt includes the relevant changed code and
  evidence, and the target identifies supported limitations or proposes a repair.
  A curator-authored repair is not an upstream successful optimization.
- Pending or rejected admission: a material unresolved defect, insufficient
  context or missing qualification prevents a positive target. Preserve the
  original example and its receipts; do not replace it silently in a running set.

Split and deduplicate all related views by source lineage. A review or repair of
an existing training edit is another view of that change, not a new independent
optimization. An unchanged held-out split does not erase training-source defects;
report confirmed defects alongside the campaign's measured results.
