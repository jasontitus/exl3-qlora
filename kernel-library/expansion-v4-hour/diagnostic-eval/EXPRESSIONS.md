# Trusted two-expression comparison

`expression_launch.py` is separate from the fixed eight-question launcher. It reuses the latter's sealed runtime, checkpoint payload, tokenizer, Python/native package, adapter bytes, host and `MLX_ENABLE_TF32=0` identities. The base/prior/final adapter choices must be declared before any output. The concrete request still needs an independent binding to the completed training launch and its fixed final checkpoint; the generic launcher does not choose a training checkpoint.

No expression campaign has been sealed or run by these author tests. There has been no model generation, compiler invocation, GPU dispatch or remote execution. The tests mock process/compile boundaries and exercise the original builder's per-probe path. A trusted operator must explicitly schedule each actual execution.

## Required order

1. Run the original relocated builder and Swift harness for both golden and negative controls on the M5. Preserve all receipts including the negative harness's expected unsuccessful candidate result. All 15 reference cases must pass; both negative expressions must fail at least one numerical case; guards and finiteness must pass. M1 receipts do not satisfy this gate. Controls must report completed_fixed_cases, contain no error field, and have finite ordered start/end timestamps; an incomplete failed dispatch cannot masquerade as the expected negative result.
2. Seal and independently inspect the diagnostic campaign's exact base/prior/final choices. Prepare an expression request with its SHA and the fresh M5 control hashes. `prepare` imports no MLX and launches no child.
3. Run the expression plan explicitly. It launches exactly six greedy, maximum-eight-token generations in base/prior/final order and two-probe order, each in a fresh child. It also runs three CPU-only original-builder `collect` commands. It never compiles or dispatches a kernel. An infrastructure/provenance error stops the campaign with a failure receipt and no retry/resume.
4. Review the completed receipt, including every raw response and rejection. Each condition has an admission file and an explicit `expression_build.py` command. Run that command separately to compile the references and independently admitted candidates; no GPU dispatch occurs in this helper. Keep its `build-admission-receipt.json` together with the unchanged original `job.json`.
5. Independently check those build bindings, then explicitly run the existing Swift harness on each job. The harness retains a rejected probe while still executing the other admitted probe. Preserve per-probe/case results, not just the aggregate `passed` flag. The trusted build-admission receipt is mandatory provenance: directly invoking the original builder would bypass the extra EOS/special-token policy.

## Request

`prepare --request REQUEST.json --sha256 SHA256 --output SEALED.json`

Request fields:

- `schema`: `trusted-expression-generation-request-v1`
- `launcher`, `launcher_sha256`: independently reviewed eight-task `launch.py`
- `diagnostic_sealed_plan`, `diagnostic_sealed_sha256`: its sealed exact identities
- `expression_plan`, `expression_plan_sha256`: relocated fixed two-probe plan
- `review`, `review_sha256`: path-only expression relocation review
- `builder`, `builder_sha256`: unchanged relocated `generated_expression_probes.py`
- `swift_source`, `swift_source_sha256`: relocated Swift with only its plan-SHA literal changed
- `build_wrapper`, `build_wrapper_sha256`: separately reviewed `expression_build.py`
- `controls.golden` and `controls.negative`, each `{receipt, sha256, job, job_sha256}`
- `output`: fresh canonical output directory outside the runtime/model

All paths must refer to reviewed local files on the target host, with current hashes. Use absolute paths. The expression plan itself binds exact messages, prompt token IDs, templates, sources, shapes, reference and negative expressions, and tokenizer files. `run --plan SEALED.json --sha256 SHA256 --verify-only` validates without creating output or starting children. Without `--verify-only`, it launches the six model generations and three collections; it refuses any existing output.

## Fixed failure policy and scope

A final EOS within the eight-token budget is required. Missing EOS (even if the visible text happens to look syntactically valid), unexpected special tokens, empty output and disallowed grammar are per-probe failures. Only surrounding whitespace stripping from the original plan is allowed. There is no repair, substitute, retry, answer selection or best-of-N. Bad prompt/model/adapter/resources/token-text/stop metadata is an infrastructure failure that stops the campaign.

All original responses are collected, including failures. `expression_build.py` rereads the complete campaign and all nine exact process keys/argv/cwd/numerical environments, ordered start/end times, actual child IDs and both log hashes; revalidates the selected condition's generation tokens and admission; then wraps only the unchanged builder's parser with an explicit rejection for failed probes. It never modifies their text or templates. A valid probe remains eligible even if its companion failed. The builder snapshots the original generation receipts and emits its ordinary two-probe job; a separate sidecar binds the gate, entire campaign, original job and source hashes. A compile failure remains a compile failure.

This is bounded code correctness: the generated expression is inserted into a supplied, source-bound optimized kernel and determines the executed kernel behavior. It does not test full agent patches, optimization discovery, performance, broad generalization, or real-world kernel coverage. Whole-kernel correctness is limited to the fixed 15 test cases per candidate/reference and the stated dyadic inputs. Independent review and actual M5 positive/negative controls remain execution prerequisites.
