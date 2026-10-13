# Matched activation-mode control: independent final review

The fresh checkpoint-mode control **regressed**, while the BF16 activation-mode treatment **descended against its own baseline**. The CPU rerun of the pinned auditor exactly reproduced the saved audit. This supports the differentiable-path quantizer-bundle hypothesis for this one development case; it does not establish a universal training fix or the unique root cause.

| Mode | Own pre-update NLL | Fresh saved-adapter NLL | Change |
| --- | ---: | ---: | ---: |
| Checkpoint | 0.501257836819 | 0.503300195536 | +0.002042358717 |
| Experimental BF16 | 0.506172358990 | 0.501490603820 | −0.004681755169 |

The difference between within-mode changes is −0.006724113887. The modes have different forward objectives; their baseline or post-update absolute values are not an interchangeable quality scale. All 811 assistant targets on the same previously inspected Metal-change row are scored. The selection, seed, all 40 adapter layers, rank/alpha, optimizer, one-update limit, chunk 8192 and pinned runtime-v2 are fixed. The treatment bypasses the declared differentiable-path explicit activation roundtrips; indexer quantizers, BF16 casts, packed weights and actual routing remain. Route identities may differ.

## Evidence checked

The reviewer reran the immutable CPU audit, including prior treatment input hashes and its pinned audit, and obtained exact report equality. The control launch records actual training PID 79190 and fresh evaluation PID 29338, both exit 0, exact planned argv/cwd/PYTHONPATH, ordered start/finish/runtime checks, matching package versions and unchanged source/config/plan pins. All four stdout/stderr hashes pass; stdout files equal the corresponding JSON reports byte for byte. The post receipt binds the new checkpoint and actual PID, complete selected IDs/mask, finite per-token values, aggregate loss and logits hash. Prefix caching is disabled.

The resource amendment was declared before the control launch: treatment 80 GiB; control training 76 GiB and evaluation 72 GiB; reserve 8 GiB throughout. Explicit caches and arithmetic chunk settings are unchanged. These are different operational limits, so exact resource or timing equality is not claimed. The control training peak MLX allocation was 73.948243 GiB; its 20.991058 GiB end-of-run process observation is not a sampled process peak.

The treatment launch JSON binds its predeclared runtime/config/protocol, but is not the same full actual-process receipt format as the control. Its original execution history is separate supporting evidence; the files alone do not attest identical training-process provenance for both arms. Local hashes establish internal consistency rather than external authenticity.

## Reproduction limit that remains unresolved

The fresh control did **not** reproduce the historical update or post logits bit for bit. Its scalar baseline, all 80 initial A matrices, saved RNG and both final-layer B matrices are exact. The other 78 B matrices and 156 first/second-moment tensors differ. First-step reconstructed B gradients have 0.95174% relative L2 difference, cosine 0.99995485, and 2,674 opposite signs among 1,024,000 elements. The original post NLL was 0.505372466839; the new value is 0.503300195536. Both regress, but that does not erase the mismatch.

Source review found unchanged core VJP/math files and a real repeatability limitation in repeated-index atomic gradient accumulation. That is a plausible mechanism, not a measured localization of this difference. Neither source changes, arithmetic scheduling, resource effects nor nondeterminism is established as the unique cause. Preserve `historical_update_reproduced=false`; do not relax the exact comparison after seeing the result.

## Claim boundary

This is one preselected training row and one update per mode. The fresh control was planned after the treatment outcome was known. There are no repeated trials, independent task samples or statistical uncertainty estimates. The result establishes the two observed outcomes under the recorded recipes and supports further testing of this bundled activation intervention. It does not prove that one quantizer or STE caused the earlier failure, that all-layer training always descends, that gradients are smooth/exact, or that the model learned to generate correct or faster Metal. Held-out generation, compilation, numerical checks and benchmarks remain separate requirements.

Machine-readable evidence, source hashes and launch checks are in `matched-control-final-independent-review.json`; the detailed historical tensor/gradient comparison remains in `matched-control-training-interim-independent-review.json`. No engine, pinned artifact, plan or existing result was modified by this review. No GPU work was performed.
