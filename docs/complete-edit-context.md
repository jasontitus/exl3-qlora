# Complete edits with bounded agent context

`exl3_mlx.agent_context` can render a historical edit using its baseline edit locations instead of repeating whole source declarations. It retains every removed line and unchanged patch anchor, a baseline file inventory, and the complete assistant tool call. Full source images remain archived. This is an explicit context selection, not token truncation or splitting the answer into partial edits.

Generate a draft using an existing source capture:

```bash
python -m exl3_mlx.agent_context source-record.json new-draft-directory \
  --reference-raw reviewed-exact-example.json
```

The output directory must be new. `raw.json` contains the native chat and `exec_command` / `git apply` edit. `replay.json` records exact application against all captured after-images and identifies the omitted declaration context. `--reference-raw` preserves a separately reviewed assistant explanation only if its entire tool call matches the reconstructed complete patch. Without it, the source record supplies the explanation.

Use `--baseline-context baseline-contracts.json` to include specific missing definitions. The JSON list contains `path`, `first_line`, `last_line`, and `source_sha256` for each range; the digest binds the complete captured baseline file. Line numbers are one-based and inclusive, counting LF. Ranges must address existing captured files, cannot overlap, and cannot use boolean indices. The receipt records each excerpt's hash. This lets a reviewer supply a necessary stride, loader, epilogue or dispatch contract without repeating unrelated source. Extra context can exceed the training capacity; preparation must reject the resulting complete item rather than truncate it.

The renderer never runs the generated shell, fetched build scripts, or kernels. It uses constrained Git application in a temporary tree to verify the edit. It rejects unsupported deletions, mode changes, binary content, unsafe paths, unknown source roles, and changed files disguised as unchanged context.

Review the draft before export. Old-line context may omit definitions that a model needs to reason about shapes, synchronization, precision, or dispatch. Add independently reviewed declarations or use the original complete-context example when needed. Replay proves reconstruction, not task-context sufficiency, numerical correctness, compilation, speedup, or training eligibility. Source provenance, lineage splits, tokenization and the existing reviewed-corpus admission checks remain required.

Three historical MLX captures exercised this renderer with the actual DeepSeek tokenizer:

| Complete historical edit | Original tokens | Draft tokens | Supervised tokens |
| --- | ---: | ---: | ---: |
| Quantized split-K (`38ad257088fb`) | 69,166 | 8,196 | 4,966 |
| Segmented NAX (`940ba473fe46`) | 52,142 | 8,231 | 4,731 |
| NAX split-K routing / AddMM (`a6222f53d518`) | 33,053 | 5,730 | 2,634 |

Every assistant message and supervised token ID remained identical to its archived exact example, including host changes and tests. These are three initial draft views of existing source changes, not three new discoveries or admitted training items. Review subsequently identified missing baseline contracts and a potential alignment defect in the segmented NAX change; the compact lengths alone do not establish suitability. The current 229-view training run is unchanged. Local numerical and performance validation of these historical edits is still absent.
