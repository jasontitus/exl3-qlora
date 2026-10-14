# Plan complete-item prefix disk caching

Count unique complete shifted inputs before selecting a disk quota. The cache
stores immutable exact-item states; its current implementation raises when a new
entry plus atomic-write scratch exceeds the quota. It does not evict entries.
Epoch shuffling does not change the set of exact-item cache keys.

For the qualified DeepSeek final-block configuration (BF16 residual streams,
512 indexer routes, compression eight), a conservative tensor-byte bound for
one backbone input of length `T` is:

```text
44304*T + T*ceil(T/64)
```

Here `T = len(input_ids) - 1`, including the full prompt and target context.
The bound includes four BF16 residual streams of width5120, pre-normalization
state, compressed K/V, INT32 route IDs and the packed candidate mask. Below512
tokens, the route buffer is smaller than this bound. This formula is specific
to that layout; inspect checkpoint/runtime metadata for other configurations.

Add metadata and publication headroom, and check free disk space separately.
The reviewed229-item campaign has1,119,780 shifted input tokens; its projected
cache upper bound including a conservative metadata allowance is46.39 GiB.
The64 GiB quota therefore needs no change on this evidence. Five saved entry
layouts were checked against the implementation and an independent review.

This estimates disk capacity, not process RAM, GPU allocation or throughput.
Measure actual stored bytes and cold/warm timing. Avoid changing a running
project to increase caching; prepare a reviewed recovery or a fresh experiment
if measured evidence requires a different resource configuration.

The [independent receipt](reports/prefix-cache-capacity-review.json) binds the
source, data and captured entry metadata used for this assessment. Its original
workspace references may require the preserved campaign archive.
