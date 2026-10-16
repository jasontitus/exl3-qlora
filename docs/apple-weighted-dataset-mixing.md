# Mix prepared datasets with deliberate repetition

Use `mix` after tokenization when you want explicit example weighting. Ordinary `prepare` still deduplicates examples before splitting. `mix` preserves selected prepared records and their loss masks, including deliberate repeats, and produces a train-only dataset for the regular trainer.

For a base corpus and three profile records, save this plan beside the source files:

```json
{
  "format": "exl3-mlx-mixture-plan-v1",
  "sources": [
    {"path": "base/train.jsonl", "repeat": 1},
    {"path": "profiles.jsonl", "rows": [0], "repeat": 5},
    {"path": "profiles.jsonl", "rows": [1], "repeat": 3},
    {"path": "profiles.jsonl", "rows": [2], "repeat": 1}
  ],
  "protected": ["protected/test.jsonl"]
}
```

```sh
exl3-mlx mix --model /path/to/local/checkpoint \
  --plan mix.json --output weighted-dataset --seq-length 12287
```

The source checkout also supports `python -m exl3_mlx.workflow mix` with the same arguments. This command does not load model weights or run a GPU job. It reads the local tokenizer/configuration to validate prepared IDs and bind the resulting dataset.

Paths are relative to the plan. Input paths, including symlinks, resolve to canonical targets; provenance binds those targets and mixing rejects changes to their captured identities or bytes. `rows` uses zero-based nonblank prepared-record indices; omit it to select all records. Each source repeats its complete selected sequence in order. The trainer's sampling remains unchanged. A 229-record base plus the 5/3/1 additions produces 238 presentations. The command does not shuffle, split, deduplicate or truncate them.

Sources must contain `input_ids` and optional `loss_start`; tokenize raw chat/text separately first. Existing standard dataset manifests are checked. A source without a manifest has no independently verified tokenizer identity; its origin must be established by the source review. The output records the actual source and tokenizer hashes. Protected and excluded inputs are compared by complete token sequence, independently of the loss mask. An overlap, invalid item or existing output directory is refused before publication.

The output contains:

- `train.jsonl`: selected source lines, preserving their content; a missing final newline is added and recorded.
- `dataset.json`: standard trainer-compatible tokenizer/file bindings and counts, with no validation or test rows.
- `mixture.json`: source selection, repeat counts, input/target counts and shares.
- `mixture.jsonl`: the source record and repetition behind every output row.

Pass `weighted-dataset/train.jsonl` directly to the training recipe. Running it through ordinary `prepare` again would deduplicate the repeated examples. Source attribution, lineage grouping and kernel-quality evidence remain separate from this mechanical mixing operation.

Presentation shares describe update shares only with one example per update. Target-token shares describe the data; they do not imply globally token-weighted optimization. Preserved tokens can remain eligible for existing prefix caches, subject to exact worker/model/runtime/mode validation. This command does not qualify a training length, memory budget or cache hit.
