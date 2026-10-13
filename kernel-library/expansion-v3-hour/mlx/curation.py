"""Manual source review decisions, bounded October 13 library expansion."""

# outcome, task kind, concrete task, architecture summary. Source metrics stay in evidence.
DECISIONS = {
    "39a182fb": (
        "pending",
        "kernel",
        "Reduce pre-M5 D512 SDPA memory",
        "Pending finite review of Gemma-linked attention family; not admitted.",
    ),
    "c0418f6e": (
        "pending",
        "kernel",
        "Reduce NAX D512 SDPA memory",
        "Pending finite review of Gemma-linked attention family; not admitted.",
    ),
    "84ea22f3": (
        "positive",
        "kernel",
        "Reduce pre-M5 head-dimension-256 SDPA memory while retaining masked and unaligned cases",
        "Split each query-row group across two SIMD groups along the head dimension, exchange partial scores through reused KV threadgroup memory, and retain each group's output half. Eligible configurations hold Q fragments in registers; dispatch enables the fused D256 path instead of the larger fallback.",
    ),
    "92951975": (
        "positive",
        "kernel",
        "Add kernel gradients for gated-delta recurrence, including the final state cotangent",
        "Add sequential Metal and chunked NAX gated-delta VJP kernels with explicit q, k, v, decay, beta and initial-state gradients. Preserve the ops fallback for unsupported dimensions and hardware; sequential and chunked kernels have different numerical tolerances.",
    ),
    "83b976ea": (
        "tradeoff",
        "kernel",
        "Reduce SDPA backward memory for supported floating dtypes without attention sinks, without assuming every shape becomes faster",
        "Tile the SDPA backward matrix products and retain bounded intermediates instead of materializing the complete score matrix. Use existing matrix-product machinery; the source reports substantial long-sequence memory savings but a slower grouped-query D256 configuration. The new GPU VJP explicitly rejects attention sinks.",
    ),
    "269e099d": (
        "protected",
        "kernel",
        "Fix layer-norm bias-gradient write-after-read hazard",
        "Exact reserved identity; retained only in the exclusion ledger.",
    ),
    "f14e6d2d": (
        "correctness",
        "integration",
        "Fix float16 gradient clipping that overflows finite gradients to a zero update",
        "Accumulate float16 squared norms in float32, apply the scale in promoted precision, and cast each gradient back to its original dtype. Cover overflow, tiny clip thresholds, mixed dtypes and compiled execution.",
    ),
    "f5cffe94": (
        "correctness",
        "host_dispatch",
        "Prevent exception cleanup from deadlocking on an uncommitted cross-stream GPU dependency",
        "Finalize every open GPU stream before waiting on any stream during exception cleanup. Preserve the original exception and verify subsequent CPU and GPU work still completes.",
    ),
    "0fdcdc0e": (
        "correctness",
        "host_dispatch",
        "Keep layer-norm and boolean SDPA fallback operations on the requested stream",
        "Pass the selected stream through layer-norm multiplication and boolean-mask construction in the SDPA fallback. The source change repairs stream propagation without changing the intended mathematical expression.",
    ),
    "9cb9c06f": (
        "correctness",
        "kernel",
        "Fix reductions whose Metal dispatch or output index exceeds 32-bit limits",
        "Split wide column reductions into bounded dispatches carrying a 64-bit column offset, move output-size buffer binding consistently, and initialize large outputs with a two-dimensional grid and widened flat index.",
    ),
    "0840c42a": (
        "correctness",
        "host_dispatch",
        "Fix reduction offsets for small logical views backed by large allocations",
        "Choose Metal large-index kernels when logical size or backing data span exceeds INT32_MAX, and widen CPU reduction sizes and address offsets. A small view can still require a large physical offset.",
    ),
    "9b954755": (
        "positive",
        "kernel",
        "Make float16 InstanceNorm stable without allocating a full float32 input copy",
        "Transpose and flatten spatial axes per sample and channel, use the fused layer-normalization primitive, then restore layout and apply the existing affine terms. The source reports both repaired overflow behavior and lower measured latency.",
    ),
    "d935276d": (
        "correctness",
        "integration",
        "Fix log-softmax cancellation when logits have a large common offset",
        "Subtract a stopped-gradient row maximum before computing logsumexp and the final difference. This preserves shift invariance while avoiding subtraction of two large nearly equal numbers.",
    ),
    "9dcefa52": (
        "positive",
        "host_dispatch",
        "Prevent short convolution-cache views from retaining full prompt buffers",
        "Request contiguity for the retained convolution tail with mx.contiguous in Kimi Linear, Qwen3.5 and Qwen3-Next. The final diff does not copy the recurrent-state cache; the PR's broader proposed-state description and reported benchmarks are historical evidence, not an exact local reproduction.",
    ),
    "23af8570": (
        "correctness",
        "host_dispatch",
        "Fix lazily extracted per-request caches that all use the final loop index",
        "Construct each lazy cache extractor through a helper argument that binds its request index, avoiding a generator expression closing over a later-mutated loop variable.",
    ),
    "eaf1748e": (
        "correctness",
        "host_dispatch",
        "Enable Qwen3-Next training when a forward-only gated-delta Metal kernel lacks a VJP",
        "Select the compiled differentiable ops recurrence while training and retain the custom forward kernel in inference. Pass the training flag explicitly and preserve recurrent-state initialization; the complete commit also adjusts DWQ wired-memory handling.",
    ),
    "735a43b2": (
        "tradeoff",
        "kernel",
        "Improve recurrent gated-delta numerical stability while preserving input/output dtypes",
        "Keep decay and default recurrent state in float32, make state dtype independent from input dtype in the kernel, and return activations in the query dtype. This spends a small amount of state memory to reduce accumulated low-precision error; reported speed is nearly flat, not a proven speedup.",
    ),
    "a790972f": (
        "correctness",
        "host_dispatch",
        "Fix batched grouped-query quantized attention with four-dimensional masks",
        "Insert a grouped-query repeat axis for masks with more than three dimensions. Preserve existing two- and three-dimensional broadcasting; the final committed condition is stricter than the earlier PR description snippet.",
    ),
    "48093440": (
        "tradeoff",
        "host_dispatch",
        "Bound allocator-cache growth during training and validation with an explicit threshold",
        "Clear cached allocations when cached bytes exceed the threshold, after evaluated updates or validation batches; the captured training caller leaves validation at its separate default-zero threshold. Expose the threshold through training arguments and preserve the distinction between process footprint and live-array peak; the initial PR measurement is not a benchmark of every final threshold.",
    ),
    "416f7446": (
        "correctness",
        "host_dispatch",
        "Allow GPT-OSS router training without differentiating discrete expert indices",
        "Stop gradients through argpartition-selected expert indices while retaining gradients through gathered routing scores and their softmax. This separates discrete selection from the differentiable selected weights.",
    ),
    "66e68ce3": (
        "pending",
        "host_dispatch",
        "Apply discrete-router gradient stops across many VLM models",
        "Broad copied router motif includes protected Gemma and DeepSeek families; pending per-model/source-lineage review and not admitted as an independent example.",
    ),
    "99b01556": (
        "correctness",
        "kernel",
        "Fix float16/bfloat16 bilinear grid sampling precision and compilation",
        "Promote coordinates, interpolation weights and accumulation to float32 inside the Metal sampler and cast once to the output dtype. Retain padding behavior and compare low-precision results against the float32 reference.",
    ),
    "86859618": (
        "correctness",
        "host_dispatch",
        "Remove unsafe identity-based left-padding memoization from mutable batch caches",
        "Read current left-padding values instead of caching them by array-object identity. In-place updates preserve identity while changing values; tests cover preparation and ragged speculative-cache commits. Source throughput is roughly flat, so this is a correctness repair rather than a measured optimization.",
    ),
}
DECISIONS.update(
    {
        "8295d5e5": (
            "correctness",
            "kernel",
            "Fix grouped convolution weight bounds and large-dimension integer overflow",
            "Bound weight loads by per-group output channels after applying the group base offset; widen channel differences, spatial counters and address products before arithmetic. Retain the complete host, loader, Winograd and regression-test changes.",
        ),
        "77bf1fa0": (
            "correctness",
            "kernel",
            "Honor strided output masks in the transposed masked GEMV path",
            "Multiply the output-mask block index by its actual stride, matching the non-transposed path, and test a sliced mask with vector left-hand input.",
        ),
        "789be76e": (
            "correctness",
            "host_dispatch",
            "Keep SDPA launches within pipeline threadgroup limits on M1 and M2",
            "Compile a lower-register-pressure TK=1 variant for the supported dtype/head dimensions, and choose it only when the default TK=4 pipeline cannot support the requested threadgroup. Keep the final pipeline-limit check.",
        ),
        "494c8a6c": (
            "tradeoff",
            "kernel",
            "Avoid rounding each low-precision quantized split-K partial before the final sum",
            "Store split-K partials in float32 and use the existing split-K accumulation kernel to round once to the output dtype. Partial-buffer storage doubles for half and bfloat16; addition grouping can still differ from the unsplit result.",
        ),
        "4f8d9160": (
            "correctness",
            "kernel",
            "Decode E4M3FN NaN bytes without changing finite extrema",
            "Carry the all-ones magnitude into the half NaN exponent field while preserving the sign-bit construction, for CPU and Metal decode paths. Verify 0x7f/0xff become NaN while 0x7e/0xfe remain positive/negative448.",
        ),
        "02adf7b2": (
            "tradeoff",
            "kernel",
            "Prevent MX block maxima from saturating because the power-of-two scale rounds downward",
            "Round the E8M0 block scale upward consistently in CPU and Metal MX quantization, matching the reported CUDA policy. This prevents maximum-value clipping but trades against underflow for small values; it is a numerical-policy change.",
        ),
        "a1316145": (
            "neutral",
            "kernel",
            "Reuse RMSNorm input values across sum-of-squares and output passes",
            "Cache each thread's input values in a float register array and use zero for out-of-bounds lanes. This removes a second source read, but analytical byte savings and an author numerical check are not a measured latency gain.",
        ),
        "b08ec310": (
            "correctness",
            "kernel",
            "Prevent scatter-product CAS retries from spinning on a NaN accumulator",
            "After a failed floating-point compare-exchange, stop retrying if the observed accumulator is already NaN. The historical async regression timeout is not a reliable process-termination guard; withhold the exact-edit target pending test-harness repair and use the reasoning review only.",
        ),
    }
)
