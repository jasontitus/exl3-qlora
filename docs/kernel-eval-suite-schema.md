# Generated-code suite receipts

`exl3_mlx.kernel_eval_suite` compares existing receipts without loading a model,
running generated text, or repeating evaluation. Trusted execution backends are
separate (`kernel_execution`); their output can be wrapped in this result schema.
The existing `ktune_eval` historical Qwen audits and teacher-forced
`summarize_agent_kernel_eval.py` do not establish generated-code correctness.

```sh
PYTHONPATH=apple/src python -m exl3_mlx.kernel_eval_suite \
  --suite suite.json --output new-report.json --markdown new-report.md
```

Start from `apple/examples/kernel-eval-suite.json`. Its hashes are placeholders,
not actual evidence. Freeze task, adapter identities and budget before sampling.
Task `sha256` identifies the full task package; `prompt_sha256` binds the exact
serialized prompt. `protocol_sha256` must identify a saved protocol binding base
model, runtime, tokenizer, sampler, context, output/EOS budget, allowed attempts,
source/ABI policy, numerical oracle/tolerances and timing procedure. Different
protocols require different task IDs or separate suites. The reporter checks
hash equality; it does not interpret a protocol or independently certify it.

Each `results` entry is `{ "path": "result.json", "sha256": "..." }`.
Relative references resolve from the suite's directory, including evidence inside
result files. Only explicitly named files are read, with bounded regular-file
reads and final identity rechecks; symlinks are refused. Outputs are exclusive.
A result has:

```json
{
  "format": "kernel-eval-result-v1",
  "task_id": "example-task", "adapter_id": "candidate", "attempt": 1,
  "task_sha256": "...", "adapter_sha256": "...",
  "prompt_sha256": "...", "protocol_sha256": "...",
  "budget": {"max_output_tokens": 1024, "attempts": 1},
  "tracks": {
    "strict": {
      "generation": {"status": "passed", "evidence": [{"path": "generation.json", "sha256": "..."}]},
      "raw_format": {"status": "failed", "outcome_class": "model_output", "evidence": [{"path": "parse.json", "sha256": "..."}], "reason": "Invalid hunk counts"}
    }
  }
}
```

Stages are `generation`, `raw_format`, `replay`, `source`, `compile`, `numerical`,
and `timing`. Status is exactly `passed`, `failed`, `pending`, `refused`, or
`not_run`. Passed, failed and refused stages require pinned evidence. Missing
stages/slots remain pending. A measured numerical rejection is failed; an unsafe
source that was never executed is source-refused, not a measured numerical
failure. Infrastructure refusals must not be relabeled model failures. A compiler
exit alone cannot assert numerical success.

The `normalized` track starts at `replay` and requires completed strict generation;
it cannot claim raw-format success. Mechanical normalization must have its own
receipt/policy, retain original refusal and identify exactly what changed. Parser
policies may accept equivalent safe patch wrappers; historical parser restrictions
are not universal requirements. Never run generated shell text.

Passed source/compile/numerical/timing require the preceding stage to pass.
Strict replay additionally requires passed raw format. Timing requires
`measurements` with `workload_sha256`, `device`, `cache_state`, `unit: "ms"`, and
positive finite `samples`. Only correct results on matching task, attempt,
protocol, budget, workload, device and cache state are paired. Ratios are unpaired descriptive per-workload median latency, left divided by
right, with timing_qualified=false. Matching labels alone do not prove warmup or
interleaved measurement. Warm/cold results are not compared.

The report retains all results and stage counts over the entire registered
panel. Pending/refused/not-run are neither pass nor failure. It recommends leaders or ties only by strict usable numerical counts on common
resolved task/attempt slots; reports incomplete coverage and measured failures
separately from refusals; and returns insufficient_evidence when no usable result
exists on common support. It selects no global winner and does not turn
survivor-only speedups into an overall quality claim. Review coverage, leakage/training overlap and task-family
balance separately. Reusing receipts does not establish new hardware qualification.

Failure attribution uses optional `outcome_class`: `model_output`,
`infrastructure`, or `unspecified` (default). Only explicit model-output outcomes
enter common quality support: a completed generation cap failure, or a completed
generation followed by parser/replay/source/compile/numerical failure or refusal.
Queue, memory, transport and execution timeouts remain infrastructure, visible but
not a quality loss. Timing failures do not resolve a quality slot. Evidence authors
must justify this classification; the reporter cannot independently infer it from
an exit code. A safety refusal for unsafe generated source can be a model-output
outcome without pretending that code ran or numerically failed.
