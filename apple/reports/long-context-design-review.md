# Long-context numerical design review

Date: 2026-10-12. Scope: staged DeepSeek V4.1 full-prefill training backend in `kernel-sft/apple`. This report records an adversarial design review, implementation changes, and independent numerical fixtures. The report's author also implemented the backend; parent-agent review of those changes provides the separate code-review pass. This is not a claim of end-to-end parity with the native DeepSeek engine.

## Why increasing the old limit alone was incorrect

The former 512-token limit hid missing routing semantics. Attention to every causal compressed entry agrees with the model's top-512 rule only while at most 512 entries are visible. The real model begins ratio-1 compression at layer 20, so token 513 can require pruning; ratio-2 layers first require pruning when the input contains more than 1,024 tokens. Layers between configured index sources reuse earlier selections, and layers after candidate-source layer 20 restrict selection to that layer's candidate blocks. Candidate pruning becomes active beyond 16,384 ratio-1 entries (2,048 blocks of eight entries).

Full-prefill compression uses only complete groups. Group j is rotated at position j × ratio and becomes visible at query position (j + 1) × ratio − 1. Index keys are derived from the normalized, unrotated compressed latent before attention modifies the latent with RoPE and quantization. Index queries/keys use FP4 groups of 32 with E8M0 scales; compressed attention KV uses FP4 groups of 16 with E4M3 scales. These formats are not interchangeable.

## Implemented contracts

- `attention.py:170` implements a frozen indexer with query tiles, causal visibility, stable low-position tie selection, candidate-source block selection, forced retention of the newest reachable partial block, and compact reusable candidate bitmaps. `ShortAttention` remains the compatibility name for full-prefill attention.
- `attention.py:114` defines functional shared state: compressed KV, detached index keys, integer selected positions, and packed candidate blocks. The differentiable compressed KV crosses layer boundaries; discrete routing has no gradient. A later index source can rerank the same keys, while intermediate layers reuse the previous selection.
- `sparse_attention.py:52` computes attention in query tiles (default 16), retaining the causal sliding window plus selected compressed entries and a denominator-only sink. Its explicit first-order VJP includes both key and value derivatives of shared latent KV, accumulation for repeated reads, query derivatives, and sink derivatives. It does not retain a full sequence × heads × sequence score graph.
- `recompute.py:15` handles nested state and differentiates only floating leaves. Integer indices and byte bitmaps are passed through and receive zero placeholder gradients. The compressed-KV gradient is preserved across recomputation.
- `deepseek.py:230` exposes full-context normalized hidden states; `project_logits` projects selected hidden rows. The separate chunked-loss implementation can supervise all intended output tokens without retaining full sequence × vocabulary logits.

Full-prefill is supported. This implementation does not establish streaming/decode-cache semantics by splitting a sequence into unrelated short windows. Engram hashes still consume the complete input history. The public 32,768-token limit is an implementation bound, not a promise that a particular Mac can train that length within its memory budget.

## Confirmed defect fixed during verification

**High: negative FP8 scale exponents were lost in a direct floating-to-unsigned conversion.** In the previous `pow2_exact`, Metal converted floating exponents −7, −5, −3, −2, and −1 to unsigned zero. The function returned 1 for every corresponding power of two. This affected existing FP8 activation quantization and the new FP4/E8M0 indexer path. A tiny GPU reproduction confirmed the conversion behavior; it was not an oracle mismatch.

`attention.py:17` now casts the exponent to signed int32, adds the IEEE exponent bias in signed arithmetic, then constructs the float bits using uint32. Regression fixtures test negative floating and integer exponents, FP8 against an independently enumerated E4M3 codebook, and FP4/E8M0 against a NumPy oracle. The change alters base-model forward semantics, including at short lengths: old numerical receipts and exact-resume claims must not silently carry across this engine revision. The parent is adding an engine-semantics identity to the training contract.

The first candidate-bitmap test also found an unsupported Python-int-left bit shift. The left operand is now an explicit uint32 MLX scalar. This was a runtime compatibility defect, not a mathematical selection error.

## Numerical verification and exact scope

Environment: MLX 0.32.3 and MLX-LM 0.32.0 on Apple Metal. `tests/test_long_attention.py` includes:

1. Independent NumPy float64 attention forward values and finite-difference checks for query, window KV, compressed KV, and sink. Fixtures include batches, nonuniform cotangents, duplicate compressed reads, missing slots, and multiple query-tile sizes.
2. Independent causal/candidate-selection fixtures, exact top-512 boundary checks, and the newest partial candidate block at entry 16,385. The latter tests selection for a small number of query rows; it is not a 16K full-model memory test.
3. Exact negative powers of two; independent FP8 codebook rounding; independent FP4/E8M0 rounding; and the declared identity straight-through surrogate. Finite differences of quantization staircases are deliberately not used as a purported STE correctness test.
4. Direct comparison against the separately installed MLX-LM `deepseek_v41.Indexer`, using the same tiny deterministic parameters at sequence lengths 11 and 513. Index keys, sorted valid selected indices, and candidate masks all matched. The fixture avoids score ties where upstream argpartition does not specify a stable tie policy.
5. A two-block fixture comparing ordinary execution with block recomputation, including an indexed shared compressed-KV state and nonzero LoRA gradients.

An initial parent-run focused suite had 23 passes and two failures at assertions requiring a literal empty trainable-parameter dictionary. All independent numerical comparisons preceding those assertions passed. MLX represents some fully frozen modules with nested empty dictionaries; the assertions now require zero flattened trainable leaves. The subsequent review-agent run of `test_bounded_moe.py`, `test_streaming_moe.py`, `test_block_training.py`, and `test_long_attention.py` passed all 25 tests in 3.15 seconds on Metal. Separately, all four targeted quantization regression cases passed after the scale fix. No model or GPU workload is running from this review agent.

Reference sources inspected: the installed MLX-LM implementation at `.venv/lib/python3.12/site-packages/mlx_lm/models/deepseek_v41.py`, and the pinned native reference in `artifacts/reference/model.py` (recorded source revision `5254448103b6254fc646dacbca1cbbf20e9216c1`). These are independently implemented references for the relevant isolated operations; MLX-LM explicitly rejects the full Engram model configuration, so its indexer comparison cannot establish full-model parity.

## Full-length pilot memory failure and bounded MoE fix

The first complete-example pilot (5,809 stored tokens, sequence length 5,808) aborted before a completed update. Its `memory-abort.json` records an MLX peak of 32.5643 GiB against a 32 GiB process budget. A subsequent verbose diagnostic confirmed the abort occurred before the first forward layer completed. This is a failed pilot, not successful long-context training.

The old MoE implementation constructed a lazy chain of full token-by-hidden FP32 scatter-add outputs, one for every selected expert, while nested streamed projections executed eagerly. At this length each full accumulator is approximately 113.4 MiB. This was a separate internal memory-retention risk: block-level checkpointing cannot release an internal chain before that block finishes. Materializing the existing differentiable loop without replacing its VJP would incorrectly drop gradients. **It was not the isolated cause of the first pilot failure:** installing the bounded MoE still produced the same 32.5643 GiB failure, and the subsequent phase trace identified an earlier allocation trigger.

`moe.py` now supplies an explicit first-order MoE VJP. Forward accumulates and materializes each expert contribution immediately. Backward recomputes one selected expert at a time, accumulates FP32 input gradients and routing-weight gradients, then computes the shared-expert and router VJPs. Trainable parameters are explicit primals, including expert LoRA weights; unused experts receive zero gradients. Component parameter state is restored after recomputation. This adds an extra expert forward during backward and changes BF16 gradient accumulation order, so the parent is recording a new engine-semantics revision.

New dense-dispatch fixtures compare forward values, complete input gradients (including a demonstrably nonzero router contribution), all routed/shared expert adapter gradients, zero unused-expert gradients, and unchanged model parameter values. A 64-expert, 4,096-token, 128-hidden fixture reproduces eager streamed-projection boundaries and requires forward plus backward peak allocation above its baseline to remain below 64 MiB; it passed. Sixty-four retained full accumulators alone would occupy 128 MiB in that fixture. Existing real-EXL3 grouped MoE and recomputation checks also pass. A repeat full-model pilot is still needed to measure the fix's real impact.

The additional BF16 fixture uses BF16 input and projection weights during matmuls, matching the real dense trunk path. Output, loss, and every routed/shared LoRA parameter gradient exactly matched dense dispatch in that fixture. Input-gradient maximum absolute difference was 0.0001220703125, against reference maximum magnitude 0.01275634765625 (approximately 0.96% at that tensor scale). This is consistent with BF16 accumulation-order differences. The test bounds error by four BF16 epsilons at each tensor's maximum magnitude plus a small absolute floor and records all observed errors. Its FP32 variant, BF16 variant, and memory regression all passed (three tests in 2.74 seconds). This fixture validates the declared numerics; it does not imply bitwise BF16 input-gradient equivalence.

## Phase diagnosis and bounded activation quantization

The diagnostic in `artifacts/ktune-20261012/phase-diagnostic.log` entered layer-zero sparse attention with 8.151 GiB active MLX memory and returned with 16.883 GiB active (17.604 GiB peak). The subsequent activation quantizer and output projections were still lazy and left the observed allocation unchanged. The abort occurred after entering MoE, as its router forced evaluation of the attention-output path; no routed-expert accumulation had begun. Therefore the original first-layer abort must not be attributed to MoE alone.

The attention-output quantizer processes 190,316,544 BF16 elements at this length, approximately 0.354 GiB. Each full FP32 or int32 rounding intermediate occupies approximately 0.709 GiB. The previous quantizer's custom forward returned a lazy whole-array result without an opaque materialization boundary, allowing its large rounding graph to remain inside the enclosing training trace. This was both a transient-scratch and retained-graph problem.

The quantizer now flattens only complete format-specific groups and evaluates at most 1,048,576 elements per tile, materializes each tile, and materializes the final concatenated output. Last-feature-axis group boundaries, scale types, midpoint rules, output dtype, and the identity straight-through VJP are unchanged. The six combinations of FP8/FP4-E4M3/FP4-E8M0 and FP32/BF16 matched an independently expressed vectorized reference using native FP8 conversions bit-for-bit, including a chunk size deliberately crossing feature-row boundaries. Their activation VJPs matched the supplied cotangents exactly.

A three-stage quantization chain over a 2,048 × 8,192 BF16 input (16 million elements) measured 323.13 MiB of extra peak MLX allocation, below its 512 MiB regression bound. The quantization, long-attention, and existing attention suites passed all 26 tests in 2.54 seconds on Metal. These changes preserve quantization arithmetic under engine semantics v3; the earlier negative-exponent correction remains a separate arithmetic change. The full-model pilot must still be rerun to validate the real memory profile and locate any remaining bottleneck.

## Remaining limits and acceptance gates

- Index scores accumulate in FP32, following MLX-LM. The pinned Torch implementation can round intermediate products/scales in BF16. Selection near rank ties can differ. Stable low-index tie selection is explicit; native Torch bitwise selection parity is not established.
- The attention VJP and activation quantization surrogate support first-order adapter training. Second-order derivatives are not supported or tested.
- Memory now grows linearly for retained layer activations and selected-index state, with bounded attention-score tiles. Index selection still performs quadratic work in sequence length; host materialization and gradient scatter copies add significant overhead. No throughput or maximum-length memory result follows solely from these unit tests.
- A complete long-context full-model pilot, measured memory, finite gradients, checkpoint/reload, and paired baseline/adapter evaluation remain required before claiming usable long-context training on the actual checkpoint. A 32K end-to-end run has not been validated by this report.
- End-to-end logits/gradients versus an independently verified implementation remain an open gate. Loss reduction and changed generation alone cannot close it. Existing independent Engram, EXL3 decoding, VJP, and checkpoint tests should remain part of the broader suite.

Read-only review of the separate chunked supervised cross-entropy path found no new blocker: target rows align with prior hidden positions; the complete backbone keeps prompt context; hidden gradients propagate through the frozen head; and token weighting remains consistent. Its frozen-head and first-order assumptions are part of its contract.
