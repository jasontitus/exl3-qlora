# Actual adapter comparison

Read [actual-v3/comparison.md](actual-v3/comparison.md). It compares the base,
old458 and final188 captured outputs under the same safe replay parser. No
strict usable kernel has been demonstrated. Row RMS model attempts are pending.

The suite can be replayed without the original experiment workspace:

```sh
PYTHONPATH=apple/src python -m exl3_mlx.kernel_eval_suite \
  --suite apple/reports/kernel-eval-panel-v1/actual-v3/suite.json \
  --output /tmp/kernel-comparison.json --markdown /tmp/kernel-comparison.md
```

Run from the product repository root; output files must be absent. The report's
direct evidence references are local. Historical receipt contents retain their
original machine paths for provenance; replaying this report does not execute or
independently recertify those historical jobs.

Earlier `actual-v1` and `actual-v2` imports are preserved. `import_evidence.py`
and its archival integration test require the original experiment workspace;
they are not required to read or reproduce the sealed suite. `common-replay`
contains the three actual captured-response parser runs. The RMS control summary
records harness qualification only, not an adapter success or speedup.
