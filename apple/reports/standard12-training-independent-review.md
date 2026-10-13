# Standard 12-step training evidence review

No blocker found in the completed training/checkpoint/probe receipts. This CPU-only review verified all **12 checkpoint manifests**, finite saved adapter/Adam/RNG tensors, optimizer step and learning rate, embedded state, exact prepared data, source identities, and the final probe's checkpoint and token bindings. The selected checkpoint is step 12. The four layer-39 adapter matrices all changed between steps 1 and 12; 78,848 parameters were trainable.

The sampler visited each of the three rows four times. Its recorded target counts match the deterministic seed-8 sampling order at every update. All 2,892 targets are present in the final probe, and row 1 exactly matches the original 5,809-token example and its mask. All 61 runtime-v3 snapshot files matched their pins. Base metadata and tokenizer/template hashes matched their recorded identities; this does not hash every frozen weight byte.

The final training-mode probe is consistent with the embedded training report and all reported token means/predictions/counts. It reports aggregate NLL **0.3262765812**, with row NLLs **0.2412309419 / 0.4611514947 / 0.2772439499**. It recorded **three additional prefix-cache hits and zero new misses**, retaining 676,312,732 bytes. The training report records peak MLX memory **53.08 GiB** and **1,808.36 seconds** including the final probe.

**Fresh uncached fit and adapter parity remain pending in this review.** These receipts show completed updates and internally consistent training-mode scores. They do not yet establish that a fresh reload reproduces the probe's logits or that the final checkpoint improves every row against a fresh baseline. Do not compare the first history loss directly with the final aggregate: they score different row sets. Training losses are pre-update measurements on changing samples.

There is one source group and no validation/test split, validation-based selection, generated patch, or executed Metal benchmark. These results are training-fit evidence only. CPU checkpoint integrity checks do not execute a resume or prove optimizer recurrence by themselves.

Machine-readable checks and hashes: [standard12-training-independent-audit.json](standard12-training-independent-audit.json). No project evidence, pinned runtime, or engine source was modified.
