# Durable evaluation rows (CPU ledger)

`exl3_mlx.evaluation_progress` stores complete evaluation rows and combines them after interruption. This is an opt-in Python building block. The current `SFTTrainer` and evaluation CLI do **not** call it; it does not yet resume model evaluation. Existing runs, reports and default behavior are unchanged.

## Identity and evidence

Call `make_identity(rows, ...)` with the frozen prepared `input_ids` and `loss_start`, project hash, complete base-payload verification references, actual runtime artifacts and dependencies, device/host, tokenizer/template, arithmetic and adapter configuration. `seq_length` counts shifted input positions, so a complete row may contain at most `seq_length + 1` serialized IDs. EOS and both vocabulary dimensions are checked. Truncation is rejected.

For a base condition, use `kind="zero_b"` and bind the actual LoRA rank, alpha, layer placement, A-initialization identity and zero-B verification. A missing adapter path does not mean that the evaluator has no LoRA wrappers. An adapter condition instead binds its configuration, tensor payload and selected checkpoint step. Activation mode, TF32 unset versus explicit values, projection/loss chunk geometry, cache semantics and backend options are part of the identity. Reuse is limited to the same project, selection, host, runtime and declared arithmetic.

The mandatory `verify_context` callback must check those references against the actual frozen inputs and return their semantic identity digest. A callback that merely returns a stored digest supplies **no external verification**. The ledger checks internal receipt consistency, not authenticity of model weights, processes or claimed logits. Full-byte verification and historical full hashes plus current file-identity continuity are separate declared methods. Metadata hashes alone do not establish complete base-payload identity.

## Caller lifecycle

Acquire the project lock first, if applicable, then enter `EvaluationProgress(..., resume=True)` to reuse its namespace. New namespaces can use the default `resume=False`; existing progress requires explicit opt-in. Pass a canonical absolute progress path without symlink components.

1. Validate prepared rows and actual context before entering. The ledger acquires a persistent-inode `.writer.lock` for direct callers too. Never delete that lock file.
2. Inspect `pending_indices`. If empty, call `aggregate` without importing MLX or constructing a model. The caller's context guard must also support this CPU-only path.
3. Otherwise construct one model for the condition and verify its actual graph against the identity. For each missing index, call `begin_row(index, producer)` **before** scoring the whole row, then `commit_row(attempt, score)` after materializing all metrics. Partial-row scores are rejected.
4. Call `aggregate(assembler, output=fresh_output_path)` only when every row is complete. Final output is exclusive; it cannot replace an existing comparison or report.

`producer` and `assembler` contain a distinct run ID, positive process ID, full argv and host ID. Original producers and times remain in immutable row receipts. The assembler is a separate record; a restarted process must not reuse an old run ID with a new PID. Cached results are labeled reused. Existing completed rows generate adoption events when assembled. Interrupted attempts without a complete row remain unknown exposures, including work that may have finished before its receipt was written. These local counters are not a global held-out access registry.

Each score contains exact target IDs, predicted IDs, finite nonnegative per-target NLL, and a logits digest descriptor with scope, two-dimensional shape, dtype and byte order. Model vocabulary padding is permitted for predictions. The ledger cannot derive or independently verify logits from an asserted hash; the scorer must hash the actual materialized bytes. `supervised_logits` and `all_next_token_logits` are distinct scopes.

## Versioned metrics and publication

V2 row loss sums use `math.fsum` over the ordered binary64 JSON per-token NLL values. The global sum uses one `math.fsum` over all ordered token values, **not** a sum of previously rounded row subtotals. Main loss is token-weighted; macro row loss is separately named. Fresh and resumed ledger paths use exactly the same computation. Historical evaluator FP32/blockwise reductions retain their original meaning; this module does not promise bit equality with those reports.

`ordered_row_logits_digest_v2` binds the domain, occurrence index, complete prepared-row hash, scope, shape, dtype, byte order and row digest. It is not the historical SHA256 of concatenated logits, which cannot be reconstructed from individual hashes. Duplicate row content remains separate indexed occurrences.

Every event and row is written to a new temporary file, fsynced, linked exclusively to its final name, and followed by directory fsync. Only `.tmp-<32 lowercase hex digits>.json` files are ignored on restart; malformed complete files, unknown paths, symlinks, changed inputs and inconsistent receipts fail closed. A complete row is itself its completion record, avoiding a second-event crash window. Published rows are never replaced. Readers must not treat an ignored temporary as valid evidence.

The aggregate embeds the declared identity, hashes of the progress files, original per-row producers, assembler and exposure counts. Its numeric rows can be consumed by `evaluation_audit.compare_token_evaluations`; that comparison still says provenance auditing is pending. The v1 project single-child receipt validator is unchanged and must not silently accept this multi-producer format.

## Remaining integration gates

Future hooks must score one complete row, validate real graph/runtime continuity before publication, skip model creation when all rows are reusable, accept multi-producer provenance explicitly, and preserve train-first product sequencing. Tests must cover those composed workflow and project paths. Actual-model fresh-versus-resumed numerical qualification is still required before enabling result reuse by default. This ledger establishes neither generated-kernel quality nor a successful hardware evaluation.
