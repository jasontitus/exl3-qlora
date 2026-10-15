# Strided-copy task support (CPU contract)

`exl3_mlx.kernel_strided_copy` validates a bounded three-buffer transpose ABI and generates an exact byte-copy reference without MLX, model, or device imports. It preserves signed zero, subnormal and finite extreme bit patterns without floating arithmetic. Input/output allocations include padding and eight trailing guard words. The reference rejects incorrect lengths, unsupported geometry and excessive allocation.

`dispatch()` accepts only `strided-copy-dispatch-v1` with `layout` linear or xy and two explicit threadgroup dimensions, at most256 total threads. Its returned grid/threadgroup values contain no executable expressions. This is a CPU schema helper, not an extension to the existing RMS full-file parser, a source-safety checker or an execution backend. A future full-file protocol registration and trusted host adapter must bind this schema explicitly; do not label historical RMS responses with it.

Synthetic development controls and the unactivated standalone host live separately in the experiment workspace. No product command launches a device. Geometry, output bitwise accuracy, padding preservation, source safety and owned resource admission all require validation before candidate timing. No measured optimization or unused confirmation result is implied by this module.
