# Strided-copy task support (CPU contract)

`exl3_mlx.kernel_strided_copy` validates a bounded three-buffer transpose ABI and generates an exact byte-copy reference without MLX, model, or device imports. It preserves signed zero, subnormal and finite extreme bit patterns without floating arithmetic. Input/output allocations include padding and eight trailing guard words. The reference rejects incorrect lengths, unsupported geometry and excessive allocation.

`dispatch()` accepts only `strided-copy-dispatch-v1` with `layout` linear or xy and two explicit threadgroup dimensions, at most256 total threads. Its returned grid/threadgroup values contain no executable expressions. This is a CPU schema helper, not an extension to the existing RMS full-file parser, a source-safety checker or an execution backend. A full-file protocol registration and trusted host adapter must bind this schema explicitly; do not label historical RMS responses with it.

Synthetic development controls and the unactivated standalone host live separately in the experiment workspace. No product command launches a device. Geometry, output bitwise accuracy, padding preservation, source safety and owned resource admission all require validation before candidate timing. No measured optimization or unused confirmation result is implied by this module.

`kernel_strided_response.parse` now provides a separate full-file envelope format, `kernel-strided-copy-full-files-v1`, with the strict dispatch schema above. Optional `protocol` in an existing `--import-replay` request selects it explicitly; the replay backend pins both this parser and `kernel_strided_copy` along with its existing helpers. The RMS `kernel-full-files-v1` format still accepts only its original dispatch schema. No automatic format guessing or cross-protocol relabeling occurs. Unchanged complete source files are valid parser output, not a speedup. Capture/tokenizer/task registration remains external trusted evidence; the parser does not attest those claims. Generated text is never executed as a shell.

To replay a captured response on CPU, start from the request example in
[the execution guide](apple-kernel-execution.md). Set `replay.protocol` to
`kernel-strided-copy-full-files-v1`, set `required_paths` to both `kernel.metal`
and `dispatch.json`, and pin the actual response and baseline files. Keep the
capture result’s registered protocol identity distinct from RMS.

```sh
exl3-mlx-kernel-run --import-replay transpose-request.json --sha256 REQUEST_SHA --output transpose-prepared
exl3-mlx-kernel-run --plan transpose-prepared/execution-plan.json --sha256 PLAN_SHA --output transpose-replay --execute
```

Replace the SHA placeholders with the exact saved request and emitted plan
hashes. This executes only the bounded CPU replay parser. It does not compile or
run candidate Metal, attest capture provenance, or establish correctness. Use a
fresh output directory for each command; generated shell text is never run.
