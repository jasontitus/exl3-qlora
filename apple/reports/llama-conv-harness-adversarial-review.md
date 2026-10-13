# Llama convolution qualification: independent review

Reviewed 2026-10-12 local time. CPU source/artifact inspection only; this reviewer did not run a GPU workload.

The harness tests the isolated `conv_transpose_1d` before/after pair from commit `560445bf34c87356ad0f8d80fb03ec5488850b65`, parent `2eb3e6b2428c035dd24f996c7dfc48654dc19a6b`. It compares each variant to an independently written dense CPU convolution loop, as well as to the other variant. It does not validate the full ggml runtime, entire commit, arbitrary input distributions, or claimed speedup.

## Numerical and ABI review

The Swift argument structure has four signed 32-bit fields followed by two unsigned 64-bit strides: 32-byte stride, with byte strides at offsets 16 and 24. This matches the included Metal argument structure. Buffers 1/2/3 are weights/input/output, with the output bound after a 16-float prefix canary. One thread per group matches the shader's use of threadgroup grid positions; the grid's Y dimension supplies the output-channel count used by the weight offset formula.

Weights have shape `[input_channels, output_channels, kernel_size]`, input has `[input_channels, input_length]`, and outputs use explicit byte column/row strides. All fixed indices and products fit the declared integer types. The computed storage includes every logical output plus stride holes, row padding, prefix and trailing canaries. Those non-output slots are compared using exact canary bits. Both input buffers are checked for mutation. Nonfinite outputs, numerical disagreements, canary changes, input changes, GPU command errors and incomplete case counts prevent a passing result.

There are eight shapes and two weight types, giving 16 cases and 29,498 output positions per variant. An independent CPU enumeration confirms that the optimized lower/upper index bounds select exactly the same `(channel,input_position,kernel_position)` contributors as the dense loop for every fixed output. The cases include singleton inputs, overlapping kernels, stride larger than the kernel, right-of-support outputs, odd extents, output column/row padding, and the declared codec geometry.

The exact-oracle claim is justified for these chosen values. Inputs are integer multiples of 1/16 and weights of 1/32, both exactly representable in FP16 and FP32. Products are integer multiples of 1/512. At most 30 products contribute to a fixed output, and the worst possible absolute partial-sum numerator is 4,950, far below the FP32 exact-integer range. Thus reference addition order and FMA contraction cannot introduce rounding differences in these cases. This deliberate restriction tests indexing and preservation, not broad rounding-sensitive behavior.

## Findings and repairs

The first runner accepted any internally consistent source/composition/compile manifest while labeling it with the fixed upstream commit. The repaired runner pins both reviewed ordered translation-unit SHA256 values (`cdd40eec...e118e` before and `d7d32a6f...d2575` after), in addition to verifying full source blobs, exact slice composition, AIR/source compile bindings, ordered distinct libraries, and actual executable bytes. Captured library data is loaded directly, avoiding a hash/load reread.

The first catch path could retain `passed=true` if the final success-receipt write threw. The repair resets `passed=false` and sets numerical status to failed/incomplete before writing the failure receipt. Per-side compile exit/error checks were also added. The output file is exclusively created; validation-only execution returns before creating a Metal device and never labels numerical parity as passed.

The initial builder checked original compiler inputs before and after use. A suggested additional hardening is to link and compile captured build-owned source/AIR snapshots; the final artifact review records whether this was included. Checksums are local provenance bindings, not external authentication of a compiler or upstream author.

Final build identity and GPU status are tracked by separate receipts so that this pre-execution source review is not confused with an observed numerical result.

## Final build-v2 check

The repaired build captures Swift, AIR, Metal source and original compile/composition receipts into build-owned snapshots. Its manifest SHA256 is `37e804c0a9d46ab86d8ed65c941cccd7c651036624a9e84093492e1d430b0b42`; the captured Swift SHA256 is `eb30b3546eb4e60dfa586a03ca4a63cf82644ddf513636324faa229c974d9a67`. Independent file/source/AIR/library/compiler/ordered-link/exact-composition checks all pass. Receipt: `reports/llama-conv-build-v2-independent-check.json`.

No remaining blocker was found for the declared controlled numerical run. GPU numerical outcomes remain pending at this review stage; compiler/link success and the exact CPU fixture analysis do not substitute for executing the shaders.

## Completed GPU result and independent receipt audit

Root subsequently ran build-v2 on Apple M1 Ultra and reported exit zero. The raw receipt `artifacts/loss-fix/llama-conv-parity-v2/gpu-parity.json` has SHA256 `f197f71c75c3518a2b140ae8ebd652cf4ff2c0388e165bbcf1b7a7c20cd81bd2`. CPU-only independent review confirms all 28 artifact, provenance, case-plan, outcome and scope checks, recorded in `reports/llama-conv-gpu-independent-check.json`.

All 16 declared cases are present exactly once and in the planned order. Each variant has 29,498 output positions: 58,996 total values compared to the CPU oracle across before/after, plus 6,376 total checked canary words. The receipt records zero before/after bit mismatches, zero CPU-oracle mismatches, zero nonfinite outputs, zero canary mutations and zero input mutations.

This review additionally reconstructed each complete output buffer on CPU using integer numerators and input-position/kernel-offset accumulation, independently of the Swift per-output loop and optimized shader bounds. All 16 reconstructed buffer SHA256 values match both GPU output hashes exactly, including guards. Regenerated FP32 input and FP32/FP16 weight bytes also match every recorded input hash. The manifest, actual executable, captured Swift, ordered distinct libraries, AIR, exact source composition and upstream pair bindings all remain unchanged and verify against the pre-run review.

The observed result establishes exact correctness for this isolated kernel pair on the 16 finite dyadic cases on M1 Ultra, with the declared memory guards. It establishes no timing or speed result, full upstream-build/commit correctness, arbitrary-distribution accuracy, or quality of model-generated kernels. No GPU workload was executed by this independent reviewer.
