# First-update diagnosis: component checks pass; fitting still fails

Final bounded-series result: 2026-10-12, after the +0.1, -0.1 and adaptive +10
strength probes and the actual-dimension component checks. **All three tested
derived variants worsened answer NLL; no adapter was promoted as an improvement.**
No source or GPU work was performed by this documentation reviewer; results below
come from saved parent-run receipts. The original optimizer checkpoint and failed
fit are unchanged.

The 80 GiB configuration completed the 5,809-token example in **31.0 minutes**,
peaking at **73.95 GiB MLX**. This establishes operational feasibility for that
example and configuration. It has not established improved fitting or kernel
quality. In the separate single-MoE-layer experiment, 57.902 s became 44.103 s
(**23.8% lower elapsed**) when both packed caching and decoder chunk changed;
there is no completed low-memory full-update baseline for a full-model speed ratio.

## Same-answer fitting results

All completed conditions score the same 811 supervised answer tokens. Lower NLL
is better. Strength scales saved LoRA B while retaining A; the derived artifacts
are inference-only probes, not new optimizer updates or resumed checkpoints.

| Condition | Answer NLL | Interpretation |
| --- | ---: | --- |
| Base, strength 0 | 0.5012578592 | Reference |
| Original step 1, strength +1 | 0.5053724668 | Worse than base; original fitting failure retained |
| Derived strength +0.1 | 0.5090817250 | Worse than base and original; smaller strength did not remedy this example |
| Derived strength -0.1 | 0.5019030712 | Also worse than base; not a useful sign-reversal fix |
| Adaptive derived strength +10 | 0.5074643706 | Also worse than base; added after seeing the smaller probes |

The [original fit receipt](ktune-full-parent-first-update-fit.json) is immutable.
The [final strength-diagnostic aggregate](ktune-strength-diagnostics.json) binds
all five conditions, raw scores, plan/checkpoint/derived-weight hashes and engine
identity. It verifies unchanged A tensors, exact B scaling, unchanged engine
source, and preservation of the original checkpoint and failed-fit receipt.
The +10 evaluation took 566.3 seconds and ended this bounded series without an
improved adapter. These adaptive training-row probes are not held-out evidence,
new optimizer updates or a retroactive success for step 1. No smaller/larger
strength, sign reversal or learning-rate remedy was established by the tested
conditions; improved fitting and kernel quality remain unproved.

## What the new checks establish

| Check | Result and scope |
| --- | --- |
| Actual native projection/head VJPs | [6/6 passed](native-vjp-actual-shapes.json): five layer-0 MXFP8 projection cases at actual feature dimensions, including one wo_a output group, plus the complete BF16 vocabulary head and cross-entropy input VJP. Linear tests use four input rows; the head uses two. |
| Full-prefill-row native VJPs | [5/5 passed](native-vjp-fullm-sampled.json): execute 5,808 rows for the same projection cases; all outputs and dX are finite. Independent NumPy-decoded oracle comparison covers eight fixed rows, not every row. |
| Sparse-attention VJP | [4/4 passed](sparse-attention-real-dim-vjp.json): F32/BF16, 64 heads, dimension 512, 512 selected entries, 128-token window, query tiles 16, sequence lengths 145/160. Synthetic fixed/repeated selections test accumulation; this is not full indexer/RoPE/compression differentiation or a 5,808-token attention oracle. |
| Current test suite | **518 tests plus 27 subtests passed**; [JUnit](kernel-sft-pytest.xml) contains 545 passing case records and no failures/errors. Older 511-test review cutoffs remain historical. |

The native reference decodes actual weights independently in NumPy float64 blocks
and preserves the documented BF16 cast boundaries. Sparse attention is compared
with a separately expressed dense MLX autodiff calculation; it shares the MLX
backend. These are declared-tolerance component passes, not bitwise full-model
parity or proof that the end-to-end surrogate gradient reduces the quantized loss.

The [optimizer audit](first-step-optimizer-diagnostic.json) also found that saved
B follows the configured Adam rule and moves opposite the recorded gradient.
That rules out the tested optimizer-sign/update mismatch; it does not establish
that the recorded full-model gradient is correct. Quantization and discrete
routing sensitivity remain plausible explanations, not a measured diagnosis.
The covered component tests narrow the investigation without explaining away
the failed first step or justifying a training-quality claim.

The tested complete length remains **5,809**, not the 32,768 implementation
ceiling. Fresh full-context scores have taken about **523–570 seconds per
condition**. Efficient long-patch generation and independent compiled-kernel
correctness/performance remain unfinished practical requirements.
