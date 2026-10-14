# Training and serving integration

The Apple trainer produces private EXL3 LoRA checkpoints and evaluates their
supervised loss. Neither training nor loss evaluation requires a serving engine.
The trainer also has a generation path that recomputes the prefix; it is not a
cached serving implementation.

Generated-answer evaluation is being integrated with an isolated build of
`mlx-serve` linked to patched `mlx-stream`. This extends existing inference
software to read the private adapter and apply the learned attention-projection
updates. It is not a replacement inference engine. Loading the base model in
unmodified serving software does not establish support for this adapter format.

As of 2026-10-14, the isolated absent-adapter diagnostic loaded the base and
completed a 33-token prefill, then generation failed with
`AdapterUnsupportedModuleRoute`. The exact-generation route rejects host-row
embeddings, but the handover fenced its device embedding into host rows. The
reviewed repair retains the device embedding and includes it in decode memory
accounting. CPU checks are complete; a revised native build and device
qualification remain pending. This failure is separate from training.

Before claiming serving compatibility, verify the absent-adapter route, a
zero-B adapter against that same base, and the trained adapter on the exact
checkpoint. Verify projection names, orientation, scaling and application across
prefill and decode. Compare cached and fresh computation under an explicit
numerical contract. Successful private integration still does not establish
PEFT, unmodified MLX-LM, or general HTTP serving compatibility.

Lower held-out loss is one learning measurement. Useful Metal edits require
generated base/adapter comparisons followed by patch application, compilation,
numerical correctness and representative latency measurements. Compilation or
likelihood alone does not prove faster or more reliable kernels.
