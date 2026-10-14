# Captured evaluation receipts

`SFTTrainer.evaluate()` and `verify_training_fit()` capture report bytes before
validating their provenance. They recheck the same files after validation and
before publishing comparison or fit results. Fit checks also capture the optional
completed training report used for final-probe parity. A changed file causes an
error; the prior published result is preserved.

Comparison retains its existing checks against the token auditor's input hashes,
dataset/category identities and final audit artifact. The snapshot adds bounded
descriptor reads and file identity checks; it does not change loss reduction,
selection, adapter arithmetic, child lifecycle, or held-out exposure accounting.
Fit results add `receipt_inputs_sha256`, a role-to-hash map for the captured reports.
These SFT hooks capture JSON reports; they do not certify or newly bind child logs.

The CPU-only helper is also available to other report consumers:

```python
from exl3_mlx.evaluation_receipts import capture_receipts

# Choose allowed, canonical absolute files after their producer has terminated.
snapshot = capture_receipts({"scores": report_path, "log": log_path})
scores = snapshot.validate_json("scores", verify_expected_provenance)
result = summarize(scores)
snapshot.recheck()  # Repeat after downstream work, immediately before publishing.
result["receipt_inputs_sha256"] = snapshot.hashes
publish_result(result)
```

`validate_json` rejects duplicate keys and nonfinite JSON constants. It passes only
the captured JSON to the supplied validator, ignores the validator's return value,
rejects semantic mutation of that input, rechecks the captured files, and returns
a separate decode. Validators signal failure by raising an exception. Callers must
still validate the required schema and semantic identities.

Snapshots contain immutable raw bytes and local device/inode/size/mtime/ctime
identities. Reads reject symlinks (including parent aliases), nonregular files,
descriptor drift and changes to the current pathname. The default limits are
64 MiB per file, 128 MiB total and 32 files; explicit byte limits may be supplied
to the helper. Limits bound raw captured bytes, not JSON-decoder or process peak
memory. Normal SFT hooks use these defaults and fail rather than truncate.

This is a local consistency check. The caller must restrict allowed paths; the
helper is not a model-payload allowlist and does not authenticate the producer,
independently verify model/runtime identity, prove process termination or grant
GPU execution permission. Finite checks cannot prevent arbitrary same-user edits
after the final check. Portable hashes and local file identities are distinct.
Training-fit likelihood is a wiring check; held-out likelihood is not generated
patch correctness, kernel speed, or general reasoning quality.
