# Remaining porting and validation work

The working short-context text pipeline has real EXL3 forward/backward, complete
routed MoE, hyper-connections, compressed/sliding attention, sparse Engram lookup,
streamed block recomputation, MLX-LM data/loss/LoRA reuse, optimizer resume,
evaluation and generation. See [the recipe](RECIPE.md) and measured reports.
A real update, recovery from a memory-watchdog interruption, and exact full-model
prediction reload have now passed on the local 3.0-bpw repack. The full workflow
also matched resumed adapter/Adam tensors exactly to uninterrupted training and
completed standalone evaluation and greedy generation.

## Independent architecture parity

Obtain logits and layer activations from an independent native V4.1 engine using
the same quantized weights. Component oracles, finite text loss and exact adapter
reload do not establish native end-to-end logit parity. Check compression
boundaries, shared KV owners, both Engram injections and token normalization.
Backward intentionally uses identity STE for FP8/FP4 activation rounding.

## The requested 2.8-bpw checkpoint

Connect its original per-layer EXL3 safetensors/native trunk layout, including
quantized attention, to the complete model loader. Reuse existing Engram banks
only after validating source identity and layout. Do not silently substitute the
local 3.0-bpw repack or create an unlabeled hybrid of the two models.

Target: `jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw`, revision
`915e1d0de8ee0a99944c5bc88ab802a639c63380`. All three projections of expert 0 and
wkv are downloaded (under 12 MiB payload). The requested whole-model weights have
not been downloaded. Storage is tight; check capacity before acquiring them.
No existing model files have been deleted.

## Longer contexts and serving

Implement sparse indexer/candidate selection before lifting the 512-token cap.
The short-prefill path keeps all causally eligible compressed positions while
their count fits `index_topk`. Add KV decode caching for practical generation.
Vision and MTP remain outside the current text-training scope.

Map private adapter names/orientations to the chosen serving engine, then verify
prediction parity there. Do not label artifacts PEFT-compatible or fuse them into
packed EXL3 weights without a tested export/requantization path.
Qwen is a separate architecture port.

## Performance and quality

Replace host activation materialization with a native MLX primitive while retaining
memory-retention and adjoint tests. Evaluate fused Metal kernels from
[PonyExl3](https://github.com/beamivalice/PonyExl3), preserving its license if code
is incorporated; its inspected path is inference-only and cannot supply our VJP
unchanged. Native MLX-LM currently rejects Engram configurations, so replacing the
whole loader with its model does not solve the target architecture.

Benchmark SSD bytes, cold/warm cache, distinct routed experts, sequence lengths,
adapter selections, optimizer state and system RAM. The four-token update is an
operational baseline; it does not predict long-context throughput. Keep process
and system-reserve guards enabled during experiments. Check representative held-out
task quality and long-run numerical stability before a consequential fine-tune.
