# Independent activation-quantization chunk review

Date: 2026-10-12. Read-only review by the quality-research agent, separate from the implementer. No MLX or GPU execution was performed during this review.

Reviewed `attention.py` SHA-256: `a22ffd2e0fe128949f045eb95dc332367bca6c70fbf96324b959813e2be3e2c4`. Staged and installed files were byte-identical at review completion.

## Result

No blocking numerical, gradient or shape defect was found in the new chunked `quantize_activation` path. The change bounds rounding scratch while preserving the existing group quantization arithmetic and declared identity straight-through derivative. It does not establish that the whole model fits its memory guard.

The previous 5,809-token pilot's failure remained after the bounded-MoE change. The subsequent phase diagnostic located evaluation of the attention-output path at MoE entry, before routed-expert accumulation. Therefore it would be inaccurate to claim that fixing MoE alone resolved the observed full-model allocation. The current bounded-quantization pilot needs its own completion and memory receipt.

## Checks

- `quantize_activation` requires a last feature dimension divisible by the format's scale group: 16 for FP4/E4M3 scales and 32 for FP8 or FP4/E8M0. Reshaping to `[-1, block]` therefore preserves every original group even when a tile crosses a feature-row boundary. Tiles and the final partial tile contain whole groups; no padding, dropping or truncated group is introduced.
- `_quantize_blocks` reduces only the group axis. Splitting the independent outer group axis does not change the amax, scale, rounding rule or output dtype. The final reshape restores the logical input shape. Materializing through the existing BF16-aware helper preserves BF16 bit patterns.
- The forward detaches its implementation graph, but the custom operation returns the original cotangent as its VJP. It therefore preserves the explicitly chosen identity surrogate instead of substituting the zero derivative of rounding or accidentally stopping the activation gradient. This is a surrogate-gradient contract, not a claim that quantization has an ordinary derivative equal to one.
- Empty arrays take the early return before an ambiguous reshape. Scalar and invalid-feature shapes are rejected. Logical flattening also works for non-contiguous views; it may copy them, which affects memory rather than the grouping rule.

The new numerical tests compare all six combinations of format and FP32/BF16 against a separately expressed vectorized reference, with groups that cross feature-row boundaries. They require exact forward equality and exact cotangent equality. Existing independent codebook tests cover format rounding. The reported chain-memory test addresses retained rounding graphs, not just a single inference call.

This reviewer additionally executed **1,092 CPU-only boundary cases** using the actual function's AST with a NumPy-backed dispatch mock and a group-sensitive stand-in for quantization arithmetic. Cases included caps below/above group sizes, partial final tiles, one- through four-dimensional inputs, empty dimensions, reversed non-contiguous views, FP32/FP16 storage, all formats and cotangent object identity. All passed; nine invalid shape cases were rejected. This mock checks control flow, shape and grouping only, not Metal arithmetic, BF16 conversion or MLX autograd.

## Limits and evidence

The 1,048,576-element cap bounds the input elements of one rounding tile. FP4 temporarily broadcasts over seven boundaries, so the cap does not mean one million total scratch elements. All materialized output pieces remain live until concatenation; concatenation and final host materialization also require output-sized storage. Total output/activation memory remains linear in input size. The implementation correctly claims bounded rounding scratch, not constant total memory.

The implementation report records the six parity cases plus the 16-million-element chain regression passing, and the parent reports **439 tests plus 27 subtests**, with Ruff clean, before starting the current real pilot. This reviewer did not rerun those GPU tests. The real pilot must still establish complete update, finite gradients, measured memory and saved-adapter reload/fit. Source review alone cannot establish those outcomes.
