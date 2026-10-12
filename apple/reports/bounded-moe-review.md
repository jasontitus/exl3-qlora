# Independent bounded MoE and recomputation review

Date: 2026-10-12. Reviewer: quality-research agent, separate from the MoE/recomputation implementers. This pass inspected source, existing numerical tests, the pinned DeepSeek reference, and a CPU-only failure-injection reproduction. It did not execute MLX or a model. No Python source was changed by this reviewer.

## Result

No remaining blocking defect was found in the inspected first-order, serialized training path. One concrete exception-restoration defect in `recompute.py` was reported and fixed during review. This is code-review evidence, not proof of full-model numerical equivalence or that the 5,809-token pilot fits its memory budget.

At review completion, these staged files were byte-identical to their installed counterparts:

| File | SHA-256 |
| --- | --- |
| `src/exl3_mlx/moe.py` | `88b821f17c0eda364cb136170f2de52285f872ef0a6330b4a280518d6d1828e0` |
| `src/exl3_mlx/recompute.py` | `a7849b0b1b6877b62914b75e21ab7a66305469e2a0c72def94e8b1caa90cd988` |
| `src/exl3_mlx/materialize.py` | `8dbe3bb9500edc74c679c936aef8dba50d48483154c09b1383188b5171ec4be9` |

## Resolved finding: restoring parameter objects after failure

The earlier recomputation backward installed detached/traced parameter primals through `inner()` without a `finally` around its VJP and evaluation. An exception could therefore leave replacement arrays installed in the module. Restoring equal values is insufficient: later optimizer and trace behavior also depends on preserving the intended parameter objects.

The corrected implementation captures entry parameters and restores them unconditionally around backward VJP/evaluation (`recompute.py:66`). Forward restores the original captured parameter tree (`recompute.py:25`). The parent added injected VJP and evaluation failure regressions that require exact original array identity. An independent CPU-only fake-MLX injection, which threw after parameter installation, confirmed restoration of the exact original array object. The component and whole-MoE paths also have unconditional restoration (`moe.py:32`, `:142`, `:160`).

## Mathematical and state checks

- **Parameter primals:** adapter leaves are explicit custom-function inputs; component backward reconstructs the corresponding parameter tree before evaluating the VJP. Frozen weights remain constants. Returned gradients preserve the input parameter tree, including zero leaves for unused experts. The existing test compares every routed/shared adapter gradient against a dense dispatch and checks zero gradients for a deliberately unused expert.
- **Router derivative:** selected indices are discrete, but the gathered, normalized routing scores remain differentiable. Each expert backward returns a derivative with respect to its routing weight; `dw` accumulates those slot derivatives before a router VJP adds the resulting input gradient (`moe.py:170–187`). Detaching all routing scores would lose this term; the test explicitly verifies that doing so changes the input gradient.
- **Repeated token rows:** a token can occur in several expert groups. Input gradients use scatter-add across those groups. Within one group, top-k expert selection gives each token/expert pair one unique slot; routing-weight derivatives are added to that slot. The shared expert contributes once for every token.
- **Reference semantics:** the pinned DeepSeek `Gate`, `Expert` and `MoE` code agrees on unbiased routing weights, bias used only for selection, selected-score normalization, routing weight applied before the down projection, asymmetric gate clamp, FP32 expert accumulation, and shared expert added after routed experts. Stable tie selection is a declared local policy, not bitwise parity with an unspecified native top-k tie order.
- **Nested recomputation state:** floating shared-KV leaves participate in the VJP; integer indices and bitmaps are excluded and get zero placeholders. Existing two-block tests compare adapter gradients with and without recomputation. This assumes the same tree structure on recomputation, as produced by the supported blocks.

## Memory and numerical limits

Materializing the accumulated output, `dx`, and routing `dw` after each expert removes the growing lazy scatter chain. Materializing component VJP results before continuing bounds the retained derivative graph to a component plus live inputs, accumulated outputs and parameter gradients. The fixed 64-expert regression directly targets the old retained-buffer failure mode.

This does not bound memory independently of sequence length, hidden/intermediate dimensions, active adapter count, or the rest of the model. Host materialization also creates synchronized copies. A 64 MiB synthetic regression is not a 32 GiB full-model proof; the real pilot and its process/RAM guard remain necessary.

FP32 accumulation does not make BF16 component gradients FP32-accurate: component VJPs can already have rounded to BF16 before accumulation. The reported dense-dispatch fixture matched forward, loss and adapter gradients exactly, while input gradients differed by a maximum 0.0001220703125. Its BF16 acceptance threshold is four epsilons at the **maximum tensor magnitude**, plus an absolute floor. That is a global absolute-error bound; it does not imply a 3.125% relative bound for each element or accuracy on near-cancelled entries. The declared first-order numerical scope is appropriate. Higher-order gradients, compilation and concurrent mutation of one module are not established by these tests.

## Evidence and remaining gates

The implementation agent reports all 25 focused numerical/recomputation tests passing, including BF16 and the memory regression. After the restoration fix, the parent reports the full staged suite passed **432 tests plus 27 subtests**, with Ruff clean. These runs were performed by those agents; this reviewer did not rerun a GPU workload during the real pilot.

The next acceptance receipt should bind the above engine version to a complete real-model update, finite gradients, intended adapter changes, memory measurements, checkpoint/reload, and paired baseline/adapter fit on the exact stored row. That establishes training mechanics and conditional edit fit. Generated patch applicability, kernel correctness/performance on the M5/NAX target, held-out task generalization and independent full-model parity remain separate gates described in `ktune-applicability-review.md`.

Useful follow-up coverage, if expanding the supported numerical contract, is a BF16 cancellation case reporting both normalized error and gradient direction, and whole-MoE non-default temperature/top-k/no-normalization fixtures. These are coverage suggestions, not blockers for the currently tested checkpoint configuration. Do not interrupt the frozen pilot solely to add them.
