# Revised development split — proposal only

41 development /188 future training rows; 33 exact edits+8 diagnostics in development, covering all11 repositories represented in training. Development length bins <=2048/4096/8192/>8192 contain11/9/12/9 rows. It contains85,791 supervised tokens versus341,281 remaining for future training. These are metadata strata, not measured quality.

The reviewed lineage assessment supplies171 actual identity/known-family/captured-parent components. Its broad curator repository/history labels remain explicit annotations; they no longer merge105 unrelated changes as one bucket. All known specific family, fork/port, patch, duplicate and captured parent edges stay within one partition. The source graph retains its finite evidence limits: ancestry is partial and unique whole-diff IDs do not prove semantic independence.

Fifteen connected components are excluded from development due to known unresolved upstream ports or nine missing patch identities. The two unresolved port annotations with no current229 member remain disclosed; their known members occur outside this training inventory. No unknown relation is declared resolved by repository or keyword matching. Independent review counted 16 train-only rows in these excluded components, including 13 diagnostics and 10 TensorFold rows (overlapping categories). The development partition therefore measures the eligible subset, not an unbiased sample of all 229 rows.

Selection uses a fixed seed, target41 and a deterministic whole-group heuristic with equal repo/kind/length/supervised-token objectives. It retains one best subset per row count after each group, not a globally optimal stratification solution. No model losses, heldout scores or answer preference were used. Input hashes, taxonomy, row hashes, complete edge list and assignments are in proposal.json; split-input.json contains metadata only. All484 reviewed assessment pins and the actual229 row/token/mask hashes were verified before/after. No raw examples are exported.

All229 rows were consumed by endpoint458. This proposal is only for future trials freshly initialized from the base; it cannot assess458 generalization. Protected16 and active229 stay byte-identical. Activation, dataset exports and trial budgets require separate review. Repeated development tuning itself can overfit; keep protected16 out of model selection.

Cache boundary39 remains a priority without changing any row/token/mask or frozen worker implementation. Existing training-mode keys can be reused only when the worker's complete binding and all other key fields match; standard validation mode has different keys. The proposal does not create caches or claim old training entries satisfy validation mode. The product helper is CPU-only. Installing any new .py into the worker package would change its current whole-package implementation identity. Keep frozenruntime-v5 untouched and run proposal tooling outside it.

`src/exl3_mlx/development_split.py` is the reusable pure-standard-library module, byte-identical to split_report.py. Nine self-contained synthetic methods test grouping/exclusions/strict metadata and deterministic selection. A tenth optional evidence test uses DEVELOPMENT_SPLIT_EVIDENCE_DIR to replay the saved229 metadata; all10 ran here. The CLI accepts metadata JSON and creates one exclusive report output, never training JSONL. It has no MLX imports, model reads, network or training controls. The module was installed in the product checkout after independent review `4ba49c52d2c886b7423aed28264b6102ce6eb15c1251192471b28c966ef18635`; the frozen training runtime remains unchanged. Installing the helper does not activate this proposal.

Run the metadata-only CLI in a tooling environment:

```sh
python -m exl3_mlx.development_split metadata.json split-report.json
```

The input contains `rows`, reviewed `edges`, and an optional `target`. Rows identify their repository, commit, kind, complete-item token counts, content hash, patch identities and unresolved-lineage exclusions. The command creates one new report and refuses to overwrite an existing output. It never exports training JSONL, constructs caches or launches training.
