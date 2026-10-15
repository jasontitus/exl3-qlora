"""Exact two-file response adapter for the distinct strided-copy development task.

Same bounded full-file envelope, separate protocol and dispatch schema. Parsing
neither executes generated text nor qualifies source safety or numerical results.
"""
from . import kernel_full_files as envelope
from .kernel_strided_copy import dispatch

FORMAT = "kernel-strided-copy-full-files-v1"


def parse(raw: bytes) -> dict[str, bytes]:
    envelope.require(type(raw) is bytes and 0 < len(raw) <= envelope.MAX_RESPONSE, "Invalid response size/type")
    obj = envelope.decode(raw)
    envelope.require(type(obj) is dict and set(obj) == {"format", "rationale", "files"}, "Unexpected response fields")
    envelope.require(obj["format"] == FORMAT, "Wrong strided-copy format")
    envelope.text_bytes(obj["rationale"], envelope.MAX_RATIONALE)
    files = obj["files"]
    envelope.require(type(files) is dict and set(files) == {"kernel.metal", "dispatch.json"}, "Exact two files required")
    result = {
        "kernel.metal": envelope.text_bytes(files["kernel.metal"], envelope.MAX_KERNEL),
        "dispatch.json": envelope.text_bytes(files["dispatch.json"], envelope.MAX_DISPATCH),
    }
    envelope.require(all(value.endswith(b"\n") for value in result.values()), "File must end with newline")
    spec = envelope.decode(result["dispatch.json"])
    # Validate static dispatch grammar with the minimum valid geometry. Actual
    # workload geometry is independently validated by the trusted host adapter.
    dispatch(spec, [1, 1, 1, 1, 0, 0])
    return result
