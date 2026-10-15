"""Parse the bounded kernel-full-files-v1 response without executing generated text.

Input is exactly the captured assistant content excluding the tokenizer's EOS
control token. Capture/serialization binding and source/ABI qualification remain
separate requirements. JSON validity is not Metal safety or correctness.
"""

import json

MAX_RESPONSE = 128 * 1024
MAX_KERNEL = 64 * 1024
MAX_DISPATCH = 256
MAX_RATIONALE = 2048


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Nonfinite JSON constant")


def decode(raw):
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject_constant
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds parser capacity") from exc


def text_bytes(value, limit):
    require(type(value) is str, "File/rationale must be text")
    raw = value.encode("utf-8")
    require(
        0 < len(raw) <= limit and b"\x00" not in raw and b"\r" not in raw,
        "Invalid text size or encoding",
    )
    return raw


def parse(raw):
    require(type(raw) is bytes and 0 < len(raw) <= MAX_RESPONSE, "Invalid response size/type")
    obj = decode(raw)
    require(
        type(obj) is dict and set(obj) == {"format", "rationale", "files"},
        "Unexpected response fields",
    )
    require(obj["format"] == "kernel-full-files-v1", "Wrong format")
    text_bytes(obj["rationale"], MAX_RATIONALE)
    files = obj["files"]
    require(
        type(files) is dict and set(files) == {"kernel.metal", "dispatch.json"},
        "Exact two files required",
    )
    result = {
        "kernel.metal": text_bytes(files["kernel.metal"], MAX_KERNEL),
        "dispatch.json": text_bytes(files["dispatch.json"], MAX_DISPATCH),
    }
    for value in result.values():
        require(value.endswith(b"\n"), "File must end with newline")
    dispatch = decode(result["dispatch.json"])
    require(
        type(dispatch) is dict and set(dispatch) == {"threads_per_row"}, "Wrong dispatch schema"
    )
    require(
        type(dispatch["threads_per_row"]) is int and dispatch["threads_per_row"] in (1, 32),
        "Unsupported dispatch",
    )
    return result
