# Remaining full-model work

The port now has EXL3 forward/backward, complete routed MoE, hyper-connections,
short-context compressed/sliding attention, sparse Engram lookup, and an
experimental text trainer for the local 3.0-bpw mtplx repack. A real 40-layer
forward/backward succeeded with finite adapter gradients (see reports), but no
full-model optimizer update or trained text quality has been verified.

## Immediate next validation

The user reported memory pressure during development. No further full-model run
was started after that warning. Keep the watchdog and allocator-cache controls
on for any subsequent run. First verify one four-token update at rank 2, check
that only adapters change, save/reload predictions, and record process/system RAM
as well as MLX allocations. Then measure longer sequences and routed-expert
adapters. Do not infer their memory needs from the four-token probe.

The initial Python custom-function implementation retained dense decoded matrices
under outer autodiff even with eval/stop_gradient, and runs were terminated.
Copying activation results through host buffers established a truly opaque
first-order boundary. The successful full-model probe used this mechanism and
peaked at 8.47 GiB MLX allocation. Subsequent per-slice hardening and 16-GiB
fail-fast defaults have only been tested on small fixtures. CLI budget polling
is not an OS hard quota; a single allocation can overshoot.

## Numerical and architecture gates

Obtain full-model logits and layer activations from an independent native V4.1
engine using the same quantized weights. Existing NumPy/component tests and
plausible finite text loss do not establish end-to-end logit parity. Test
compression boundaries, shared KV sources, both Engram injections and token
normalization. Activation FP8/FP4 uses an intentional identity straight-through
estimator in backward, not a derivative of the discrete rounding operation.

Implement the actual indexer and candidate selection before allowing contexts
above 512. The current short-prefill path keeps all eligible compressed positions;
this is justified only while their count fits the configured index_topk. Vision,
MTP and autoregressive decode caches remain outside the first text-training scope.

## The requested 2.8-bpw checkpoint

Connect the original per-layer EXL3 safetensors/native trunk layout to this decoder
and reuse the available Engram banks after validating identity/layout. Do not
silently substitute the local 3.0-bpw repack. The requested model is
jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw at
915e1d0de8ee0a99944c5bc88ab802a639c63380; all three projections of expert 0 and wkv
have been downloaded (under 12 MiB payload). No full 2.8-bpw model download and no
model deletion have occurred. Check storage before any whole-model acquisition.

## Performance and usable training

Replace host activation materialization with a native MLX/C++ primitive and
consider fused Metal trellis GEMM. Preserve independent decode/adjoint tests and
the memory-retention regression. Benchmark actual SSD bytes, cold and warm cache,
selected experts per sequence, token lengths and optimizer state. Optimizing the
matrix kernel alone will not remove the substantial disk traffic.

Add held-out evaluation, assistant-token masks, optimizer-state resume, gradient
accumulation, data shuffling and a tested adapter mapping to an inference engine.
The current JSONL trainer uses causal loss on every token, prefix truncation,
microbatch size one, and private adapter names. Do not label its artifacts PEFT
or serving-compatible without a mapping and prediction-parity test.

Qwen3.8-Flash-Next is a later separate architecture port; it is not supported by
the DeepSeek decoder.
