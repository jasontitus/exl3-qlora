"""CPU-only ABI contract for bounded strided F32 bit-copy tasks.

No generated source execution, device import, or full-file protocol attestation.
"""
from __future__ import annotations

import struct
from typing import Any

MAX_ELEMENTS = 4 * 1024 * 1024
MAX_THREADS = 256


def geometry(params: list[int]) -> dict[str, int]:
    if not isinstance(params, list) or len(params) != 6:
        raise ValueError("Six integer parameters required")
    if any(type(x) is not int or not 0 <= x <= 65535 for x in params):
        raise ValueError("Invalid parameter")
    rows, cols, input_stride, output_stride, input_offset, output_offset = params
    if rows == 0 or cols == 0 or input_stride < cols or output_stride < rows:
        raise ValueError("Invalid strided geometry")
    x = input_offset + rows * input_stride + 8
    y = output_offset + cols * output_stride + 8
    if max(x, y, rows * cols) > MAX_ELEMENTS:
        raise ValueError("Allocation exceeds bounded task contract")
    return {"input_elements": x, "output_elements": y, "active_elements": rows * cols}


def dispatch(spec: dict[str, Any], params: list[int]) -> dict[str, list[int]]:
    geometry(params)
    if not isinstance(spec, dict) or set(spec) != {"schema", "layout", "threads"}:
        raise ValueError("Exact dispatch fields required")
    if spec["schema"] != "strided-copy-dispatch-v1" or spec["layout"] not in ("linear", "xy"):
        raise ValueError("Unsupported dispatch protocol")
    threads = spec["threads"]
    if not isinstance(threads, list) or len(threads) != 2 or any(type(x) is not int or not 1 <= x <= MAX_THREADS for x in threads):
        raise ValueError("Invalid threadgroup")
    if threads[0] * threads[1] > MAX_THREADS:
        raise ValueError("Too many threads")
    if spec["layout"] == "linear" and threads[1] != 1:
        raise ValueError("Linear dispatch must be one dimensional")
    rows, cols = params[:2]
    grid = [rows * cols, 1] if spec["layout"] == "linear" else [cols, rows]
    return {"grid": grid, "threads": threads, "threadgroups": [(n + t - 1) // t for n, t in zip(grid, threads)]}


def reference(params: list[int], input_bytes: bytes, sentinel: int = 0x7FC12345) -> bytes:
    sizes = geometry(params)
    if not isinstance(input_bytes, bytes) or len(input_bytes) != sizes["input_elements"] * 4:
        raise ValueError("Exact input length required")
    if type(sentinel) is not int or not 0 <= sentinel < 2**32:
        raise ValueError("Invalid sentinel")
    # Preserve bits without interpreting floats (including subnormal and signed zero).
    result = bytearray(struct.pack("<I", sentinel) * sizes["output_elements"])
    rows, cols, si, so, io, oo = params
    for row in range(rows):
        for col in range(cols):
            src = (io + row * si + col) * 4
            dst = (oo + col * so + row) * 4
            result[dst:dst + 4] = input_bytes[src:src + 4]
    return bytes(result)
