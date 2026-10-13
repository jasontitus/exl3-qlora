"""CPU byte-view planning for uniform Qwen Sushi expert banks, not a decoder.

Only routed gate/up/down banks are accepted. Grouped, ragged and MTP banks
are deliberately rejected. Scales are not interpreted or folded here.
"""

import itertools
import math
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def expert_views(header, *, prefix, expert, config, data_start, file_bytes):
    """Return absolute byte spans for one expert without reading its payload."""
    require(isinstance(header, dict) and isinstance(config, dict), "Expected objects")
    match = re.fullmatch(
        r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj)",
        prefix,
    )
    require(match is not None, "Only uniform routed Qwen banks are supported")
    text = config.get("text_config", {})
    require(
        config.get("model_type") == "qwen4_exp" and isinstance(text, dict),
        "Wrong architecture",
    )
    quant = config.get("expert_quant", {})
    require(
        isinstance(quant, dict)
        and quant.get("format") == "exl3"
        and quant.get("codebook") == "mcg",
        "Only MCG EXL3 metadata supported",
    )
    window = quant.get("window", 16)
    ceiling = quant.get("k")
    require(type(window) is int and 8 <= window <= 16, "Invalid codeword window")
    require(
        type(ceiling) in (int, float) and math.isfinite(ceiling) and 1 <= ceiling <= 8,
        "Invalid rate ceiling",
    )
    dims = [
        text.get(k)
        for k in (
            "num_hidden_layers",
            "num_experts",
            "hidden_size",
            "moe_intermediate_size",
        )
    ]
    require(
        all(type(v) is int and v > 0 for v in dims), "Invalid architecture dimensions"
    )
    layers, experts, hidden, intermediate = dims
    require(
        int(match[1]) < layers and hidden % 128 == 0 and intermediate % 128 == 0,
        "Unsupported layer/geometry",
    )
    require(type(expert) is int and 0 <= expert < experts, "Expert index out of bounds")
    require(
        type(data_start) is int
        and type(file_bytes) is int
        and 8 < data_start <= file_bytes,
        "Invalid file bounds",
    )
    inp, out = (
        (intermediate, hidden) if match[2] == "down_proj" else (hidden, intermediate)
    )
    stamp = header.get("__metadata__", {})
    require(isinstance(stamp, dict), "Invalid shard stamp")
    require(
        stamp.get("format") == "exl3" and stamp.get("codebook") == "mcg",
        "Missing or mismatched shard stamp",
    )
    try:
        stamped_window, stamped_rate = int(stamp["window"]), float(stamp["k"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError("Invalid shard window/rate stamp") from e
    require(
        stamped_window == window and math.isfinite(stamped_rate),
        "Shard window/rate mismatch",
    )
    expected_names = {prefix + "." + name for name in ("trellis", "suh", "svh")}
    require(
        set(header) - {"__metadata__"} == expected_names,
        "Expected one complete uniform projection bank",
    )
    packed = header[prefix + ".trellis"]
    require(isinstance(packed, dict), "Invalid packed entry")
    shape = packed.get("shape")
    require(
        isinstance(shape, list)
        and len(shape) == 4
        and all(type(v) is int and v > 0 for v in shape),
        "Invalid packed shape",
    )
    n = shape[-1]
    require(
        16 <= n <= 128 and n % 2 == 0 and n / 16 <= ceiling and n / 16 == stamped_rate,
        "Unsupported or mismatched packed rate",
    )
    expected = {
        "trellis": ("U16", [experts, inp // 16, out // 16, n]),
        "suh": ("F16", [experts, inp]),
        "svh": ("F16", [experts, out]),
    }
    views, ranges = {}, []
    for name, (dtype, shape) in expected.items():
        spec = header[prefix + "." + name]
        require(
            isinstance(spec, dict)
            and set(spec) == {"dtype", "shape", "data_offsets"}
            and spec.get("dtype") == dtype
            and spec.get("shape") == shape
            and all(type(v) is int for v in spec.get("shape", [])),
            "Bank dtype/shape mismatch: " + name,
        )
        offsets = spec.get("data_offsets")
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(v) is int for v in offsets),
            "Invalid offsets",
        )
        lo, hi = offsets
        per_expert = math.prod(shape[1:]) * 2
        require(
            0 <= lo < hi <= file_bytes - data_start and hi - lo == experts * per_expert,
            "Invalid payload bounds/size",
        )
        ranges.append((lo, hi))
        views[name] = {
            "offset_bytes": data_start + lo + expert * per_expert,
            "length_bytes": per_expert,
            "dtype": dtype,
            "shape": shape[1:],
        }
    ranges.sort()
    require(all(a[1] <= b[0] for a, b in itertools.pairwise(ranges)), "Overlapping banks")
    return {
        "expert": expert,
        "prefix": prefix,
        "window": window,
        "bits_per_weight": n / 16,
        "views": views,
        "total_read_bytes": sum(v["length_bytes"] for v in views.values()),
        "scope": "Byte geometry only; no payload integrity, decoder parity, scale semantics or training support established",
    }
