# Experimental activation-mode configuration review

Date: 2026-10-12. Staging implementation and CPU checks only; no GPU run or modification of the active pinned runtime.

`checkpoint` preserves the existing activation arithmetic and is the default. `bf16` bypasses differentiable-path FP8/FP4 activation roundtrips while retaining packed weights, FrozenIndexer quantizers, BF16 casts and normal routing. The zero-adapter forward pass changes. This option requires a new project and a freshly scored baseline in the same mode; it is not a demonstrated remedy for the original failed fit or a quantization-free/smooth model.

## Inspected contracts

- SFTConfig validates exactly `checkpoint` or `bf16`. Standard finetune, ktune finetune and Studio expose the mode for new projects. Resume does not expose it as a resource override.
- The saved project mode is immutable and is forwarded to every engine child. In particular, both base and adapter branches of training-fit/standard comparison inherit the same mode.
- Adapter serialization writes an explicit mode and a mode-specific activation description. Legacy absence means `checkpoint`; explicit null or an unknown string is rejected. Normalization occurs after integrity verification and does not rewrite a saved checkpoint.
- Standalone evaluation, generation, optimizer resume and adapter-only warm start inherit a saved adapter's mode if the caller omits it. An explicit mismatch fails before model allocation. Low-level load also compares mode before replacing model parameters.
- Optimizer contract format is unchanged: the numerical mode is bound through the verified adapter configuration. Existing project source/runtime checks remain strict. Legacy field normalization does not authorize migration to changed source code.
- Export writes the normalized mode, including for a legacy default-mode checkpoint.
- Training, scoring and generation receipts are checked against the saved project mode. A BF16 project rejects a missing/default-mode receipt instead of accepting a potentially mismatched comparison.

## Checks and limits

[CPU test receipt](activation-mode-config-cpu-tests.xml): **178 passed**, covering the activation contracts plus existing project, selected-layer, resource, Studio and ktune boundaries. New I/O tests execute the actual serialization/loading functions with CPU NumPy safetensors standing in for MLX arrays. They cover default/legacy normalization, finite payload round trips, explicit cross-mode rejection before mutation, all three low-level adapter-loading entry paths, fresh matched baseline forwarding, project immutability and export. The Studio JavaScript harness covers saved-mode display, preservation of pending form edits, new-project payload and omission from resume.

Ruff check and formatting pass on the changed Python files. These results do not validate BF16-mode model arithmetic, full-model gradients, memory behavior, loss improvement or generated kernel correctness. Engine tests and actual matched-mode experiments remain separate gates. Independent process-agent review found no concrete configuration blocker. The review concerns the configuration contract, not experimental effectiveness.
