# Optional decoded EXL3 tile cache

Date: 2026-10-12. Status: implementation and regression fixtures staged; Ruff and Python compilation pass. The first parent GPU suite reported 497 passes and 27 subtests, with two failures confined to a test helper trying to view a uint64 scalar optimizer step as uint32. That helper now compares native bytes without changing the scalar's MLX item size; its two optimizer parity cases await rerun. An initial real-matrix benchmark passed all 54 same-chunk cache bitwise comparisons, but early timings may overlap the test workload and are screening data only. Parent will rerun timings in isolation. No throughput improvement is claimed by this report.

## Contract and implementation

`MatrixStore(..., decoded_cache_bytes=0)` adds an optional global LRU independent of the existing packed-matrix cache. The default disables retention. Keys include the matrix key, codebook, column start, and width. Each retained entry is evaluated FP16 data; its actual `nbytes` counts against the limit. A cached tile remains usable after its packed matrix is evicted.

The decoder's Metal codebook function already returns FP16 before the original FP32 reconstruction output. Retaining this intermediate FP16 value and casting it to FP32 before the existing matmul is therefore intended to be lossless. `_base` accepts an optional decoder callback and preserves the same default 512-column traversal, Hadamard transforms, FP32 matmul/accumulation, and output casts. This cache does not select a different arithmetic mode or change engine semantics.

Decoded hits recheck the frozen source shard fingerprints. A private per-store token and matrix key bind loaded bases to the correct callback, preventing a foreign store or another matrix's base from populating the wrong cache key. Entries that can fit evict old entries before new decoded allocation. Oversized requests bypass retention and preserve useful cached entries. `clear()` empties both caches; private cache state is excluded from module parameter traversal.

Decoded counters are `decoded_loads`, `decoded_hits`, `decoded_evictions`, `decoded_loaded_bytes`, `decoded_cached_bytes`, and `decoded_peak_cached_bytes`. Byte counters describe the FP16 payload, not the transient FP32 matmul cast. Disabled-cache decode requests count as loads. Budgets constrain retained tiles, not all concurrent working allocations, allocator cache, or OS page cache.

Tile population uses `materialize()` to detach the decoded tile from its packed source and enclosing gradient trace. This includes a host copy of the frozen FP16 tile; numerical decoding and matrix arithmetic remain on Metal. The cache can cost time when there is no reuse. The documentation now states this explicitly.

## Authored regression coverage

`tests/test_decoded_cache.py` covers:

- Bitwise forward and activation-VJP equality versus both cache-disabled callbacks and the original direct decoder path, with FP32/BF16 inputs, non-square matrices, and a partial final column tile.
- Cold, warm, evicted, disabled, and undersized cache states; actual FP16 retention counts; eviction before allocation; and independence from packed-cache eviction.
- Source mutation rejection on a decoded hit, rejected foreign-store/wrong-matrix bases, and preservation of useful entries when a requested tile exceeds the cache budget.
- Every codebook entry for all three codebooks, comparing direct FP32 output with FP16 storage cast to FP32; all supported bitrates with a partial final decoded tile.
- Three BF16 LoRA optimizer steps for warm and evicting caches, comparing loss, input gradients, adapter gradients, clipping norm, updated adapters, and AdamW state bitwise.
- Actual `save_checkpoint` export with a populated decoded cache, requiring only the expected adapter and optimizer tensor keys and equal tensor bytes against the uncached model.

The first Metal run passed the cache format, cold/warm/eviction, invalidation, and source-identity cases. The optimizer comparisons reached the first update without a numerical mismatch, then encountered the uint64 test-helper error described above. Completion of all three optimizer steps and their checkpoint export assertions remains pending the corrected rerun; intended coverage is not treated as evidence of a pass.

## Benchmark and interpretation

`scripts/benchmark_decoded_cache.py` accepts a real expert repack directory or EXL3 checkpoint and explicit matrix keys. It defaults to 32/128 token rows, chunk sizes 512/2048/8192, and cache-disabled/enabled comparisons. It records cold store-cache timing, warmed forward timing, public forward-plus-VJP timing, a forward followed by recomputation/VJP sequence, cache counters, source file hashes, numerical metrics, and peak MLX allocation.

For a fixed chunk size the benchmark requires bitwise output and input-gradient equality between cached and uncached runs. Against the 512-column baseline it separately reports absolute/L2 errors and dtype-specific tolerances for larger chunks: those experiments can change matmul and transpose accumulation order and are not declared exact substitutes.

Warm repeated single-matrix reuse is an upper bound on cache benefit. The actual model has 40 layers and approximately 25.3 GiB of FP16 routed weights per layer; a 32 GiB decoded cache cannot retain the entire model. A cold, single-pass evaluation can miss every tile, while repeated expert computations during a training backward pass can reuse tiles. Larger chunks may reduce launch and synchronization overhead, but only measured results can establish the tradeoff. First-use timing may also include JIT compilation and OS file-cache effects. Run the benchmark with no concurrent GPU workload.

Example from the Apple project directory:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_decoded_cache.py \
  --checkpoint /Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw \
  --keys layers.0.ffn.experts.0.w1 layers.0.ffn.experts.0.w2 layers.0.ffn.experts.0.w3 \
  --rows 32 128 --chunk-sizes 512 2048 8192 \
  --output reports/decoded-cache-benchmark.json
```

The standalone benchmark defaults to a 4 GiB process budget because it loads individual expert matrices, not the full model. The separately wired development profile uses the user-authorized 80 GiB process budget with separate packed/decoded cache budgets. That profile still requires a completed full-model training/evaluation run to establish its practical behavior.
