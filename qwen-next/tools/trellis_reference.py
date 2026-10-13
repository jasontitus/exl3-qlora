"""Small CPU reference for Sushi MCG EXL3 fixtures; no model or GPU imports.

The bitstream/permutation/codebook follow pinned expert_exl3.zig. Known-answer
qualification is limited to the converter fixtures explicitly checked. A public
matrix is a mathematical weight reconstruction; multiplying it does NOT reproduce
serving's activation-rounding sequence in prepareInput/innerGemv/finishOutput.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

TILE = 16
TILE_VALUES = 256
HAD_DIM = 128
MCG_MULT = 0xCBAC1FED
MAX_ELEMENTS = 1 << 20
MAX_FIXTURE_BYTES = 8 << 20
MAX_HEADER_BYTES = 1 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def half_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def half_bits(value):
    return struct.unpack("<H", struct.pack("<e", value))[0]


def validate_rate(n):
    require(
        type(n) is int and 16 <= n <= 128 and n % 2 == 0,
        "Packed rate must be an even halfword count in16..128",
    )
    return n


def validate_window(window):
    require(
        type(window) is int and 8 <= window <= 16,
        "Codeword window must be an integer in8..16",
    )
    return window


def validate_words(words, count):
    require(
        isinstance(words, (list, tuple)) and len(words) == count,
        "Packed halfword count does not match geometry",
    )
    require(
        all(type(w) is int and 0 <= w <= 65535 for w in words),
        "Packed words must be unsigned16-bit integers",
    )


def decode_mcg(codeword):
    require(
        type(codeword) is int and 0 <= codeword <= 65535,
        "MCG codeword must be an unsigned16-bit integer",
    )
    mixed = (codeword * MCG_MULT) & 0xFFFFFFFF
    pair = 0x3B603B60 ^ (mixed & 0x8FFF8FFF)
    # Half inputs are exactly representable in FP32. Preserve the FP32 sum,
    # then IEEE round-to-nearest-even to half as in decodeMcg.
    return half_bits(f32(half_value(pair & 65535) + half_value(pair >> 16)))


def unpack_tile(words, n):
    """Return256 tail-biting codewords, in packed (not matrix) order.

    Halfwords form little-endian U32 words. Each U32 then contributes its bits
    most-significant first to the cyclic stream. This whole-integer formulation
    is equivalent to the source's pairwise64-bit funnel, without its shift cases.
    """
    validate_rate(n)
    validate_words(words, n)
    stream = 0
    for i in range(0, n, 2):
        stream = (stream << 32) | words[i] | (words[i + 1] << 16)
    total = 16 * n
    repeated = (stream << total) | stream
    return [
        (repeated >> (total - ((t + 1) * n // 16))) & 65535 for t in range(TILE_VALUES)
    ]


def tensor_core_permutation():
    result = []
    for thread in range(32):
        row, col = (thread % 4) * 2, thread // 4
        result.extend(
            (
                row * 16 + col,
                (row + 1) * 16 + col,
                (row + 8) * 16 + col,
                (row + 9) * 16 + col,
                row * 16 + col + 8,
                (row + 1) * 16 + col + 8,
                (row + 8) * 16 + col + 8,
                (row + 9) * 16 + col + 8,
            )
        )
    return result


def decode_tile(words, n, *, window):
    validate_window(window)
    mask = (1 << window) - 1
    output = [0] * TILE_VALUES
    for codeword, offset in zip(unpack_tile(words, n), tensor_core_permutation()):
        output[offset] = decode_mcg(codeword & mask)
    return output


def matrix_geometry(in_features, out_features, *, block=TILE):
    require(
        all(
            type(d) is int and d > 0 and d % block == 0
            for d in (in_features, out_features)
        ),
        f"Matrix dimensions must be positive multiples of{block}",
    )
    require(
        in_features * out_features <= MAX_ELEMENTS,
        "CPU fixture matrix exceeds element limit",
    )


def reconstruct_inner(trellis, in_features, out_features, n, *, window):
    """Decode row-major [input, output] FP16 bit patterns, not float values."""
    matrix_geometry(in_features, out_features)
    validate_rate(n)
    validate_window(window)
    in_tiles, out_tiles = in_features // TILE, out_features // TILE
    validate_words(trellis, in_tiles * out_tiles * n)
    result = [0] * (in_features * out_features)
    for tk in range(in_tiles):
        for tn in range(out_tiles):
            offset = (tk * out_tiles + tn) * n
            tile = decode_tile(trellis[offset : offset + n], n, window=window)
            for row in range(TILE):
                dest = (tk * TILE + row) * out_features + tn * TILE
                result[dest : dest + TILE] = tile[row * TILE : (row + 1) * TILE]
    return result


def hadamard128(values):
    """Source-order FP32 normalized H128; intentionally not a fast FWHT."""
    require(len(values) == HAD_DIM, "H128 needs exactly128values")
    require(all(math.isfinite(v) for v in values), "H128 inputs must be finite")
    scale = f32(0.08838834764831845)
    result = []
    for row in range(HAD_DIM):
        acc = 0.0
        for col, value in enumerate(values):
            signed = -scale if (row & col).bit_count() % 2 else scale
            acc = f32(acc + f32(signed * value))
        result.append(acc)
    return result


def reconstruct_public(inner_bits, suh, svh, in_features, out_features):
    """Source-order public FP16 matrix, NOT a serving-project implementation.

    Compose H128 on both axes and the already-folded suh/svh scales, then round
    the final matrix to half. Serving rounds intermediate transformed activations
    and inner products, so its map cannot be replaced by this matrix bitwise.
    """
    matrix_geometry(in_features, out_features, block=HAD_DIM)
    validate_words(inner_bits, in_features * out_features)
    validate_words(suh, in_features)
    validate_words(svh, out_features)
    w = [half_value(b) for b in inner_bits]
    left, right = [half_value(b) for b in suh], [half_value(b) for b in svh]
    require(
        all(math.isfinite(v) for v in w + left + right),
        "Public reconstruction requires finite weights and scales",
    )
    for block in range(0, in_features, HAD_DIM):
        for col in range(out_features):
            values = hadamard128(
                [w[(block + r) * out_features + col] for r in range(HAD_DIM)]
            )
            for r, value in enumerate(values):
                w[(block + r) * out_features + col] = value
    for r in range(in_features):
        for c in range(out_features):
            index = r * out_features + c
            w[index] = f32(w[index] * left[r])
    for block in range(0, out_features, HAD_DIM):
        for r in range(in_features):
            start = r * out_features + block
            w[start : start + HAD_DIM] = hadamard128(w[start : start + HAD_DIM])
    return [half_bits(f32(v * right[i % out_features])) for i, v in enumerate(w)]


def frozen_matmul(x, weights, in_features, out_features, *, transpose=False):
    """Tiny float64 linear oracle on a frozen decoded FP16 matrix.

    This function supplies a linear adjoint sanity check only. It is not a
    production packed matmul, serving emulation, Metal kernel or custom VJP.
    """
    matrix_geometry(in_features, out_features)
    validate_words(weights, in_features * out_features)
    rows = out_features if transpose else in_features
    require(
        isinstance(x, (list, tuple)) and len(x) == rows,
        "Input vector length does not match frozen matrix",
    )
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in x),
        "Input vector must be finite real scalars",
    )
    if transpose:
        return [
            math.fsum(
                x[j] * half_value(weights[i * out_features + j])
                for j in range(out_features)
            )
            for i in range(in_features)
        ]
    return [
        math.fsum(
            x[i] * half_value(weights[i * out_features + j]) for i in range(in_features)
        )
        for j in range(out_features)
    ]


def inventory_module():
    path = Path(__file__).with_name("checkpoint_inventory.py")
    spec = importlib.util.spec_from_file_location("qwen_fixture_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_fixture(path):
    """Read a small explicit fixture payload after bounded container checks."""
    path = Path(path)
    inventory = inventory_module()
    initial = inventory.regular_identity(path)
    require(initial["size_bytes"] <= MAX_FIXTURE_BYTES, "Fixture exceeds byte limit")
    header = inventory.inspect_shard(path, max_header_bytes=MAX_HEADER_BYTES)
    require(header["stat"] == initial, "Fixture changed while inspecting header")
    raw, identity = inventory.read_prefix(path, initial["size_bytes"])
    require(identity == initial, "Fixture changed before payload read")
    tensors = {}
    for entry in header["tensors"]:
        require(entry["dtype"] in ("U16", "F16"), "Fixture requires U16/F16 tensors")
        start, end = entry["data_offsets"]
        blob = raw[
            8 + header["header_bytes"] + start : 8 + header["header_bytes"] + end
        ]
        tensors[entry["name"]] = dict(
            entry, bits=list(struct.unpack(f"<{len(blob) // 2}H", blob))
        )
    require(
        set(tensors) == {"trellis", "inner", "public", "suh", "svh"},
        "Expected the five known-answer fixture tensors",
    )
    require(
        tensors["trellis"]["dtype"] == "U16"
        and all(
            t["dtype"] == "F16" for name, t in tensors.items() if name != "trellis"
        ),
        "Fixture trellis must be U16 and decoded weights/scales F16",
    )
    shape = tensors["trellis"]["shape"]
    require(len(shape) == 3, "Fixture trellis needs [input_tiles,output_tiles,n]")
    inputs, outputs, n = shape[0] * TILE, shape[1] * TILE, shape[2]
    matrix_geometry(inputs, outputs, block=HAD_DIM)
    validate_rate(n)
    require(
        tensors["inner"]["shape"] == [inputs, outputs]
        and tensors["public"]["shape"] == [inputs, outputs]
        and tensors["suh"]["shape"] == [inputs]
        and tensors["svh"]["shape"] == [outputs],
        "Fixture decoded/scales shapes disagree with trellis",
    )
    metadata = header["metadata"]
    require(
        metadata.get("codebook") == "mcg",
        "Only explicit MCG fixture metadata is supported",
    )
    try:
        window = int(metadata.get("window", "16"))
        k = float(metadata["k"])
    except (KeyError, ValueError) as error:
        raise ValueError("Fixture needs valid rate/window metadata") from error
    validate_window(window)
    require(
        math.isfinite(k) and k == n / 16,
        "Fixture rate metadata disagrees with packed dimension",
    )
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "metadata": metadata,
        "in_features": inputs,
        "out_features": outputs,
        "n": n,
        "window": window,
        "tensors": tensors,
    }


def compare_bits(actual, expected):
    require(len(actual) == len(expected), "Known-answer length mismatch")
    count = sum(a != b for a, b in zip(actual, expected))
    return {
        "elements": len(actual),
        "mismatches": count,
        "bitwise_equal": count == 0,
        "actual_sha256": hashlib.sha256(
            struct.pack(f"<{len(actual)}H", *actual)
        ).hexdigest(),
        "expected_sha256": hashlib.sha256(
            struct.pack(f"<{len(expected)}H", *expected)
        ).hexdigest(),
    }


def verify_fixture(path, *, public=False):
    fixture = read_fixture(path)
    tensors = fixture.pop("tensors")
    args = (
        tensors["trellis"]["bits"],
        fixture["in_features"],
        fixture["out_features"],
        fixture["n"],
    )
    decoded = reconstruct_inner(*args, window=fixture["window"])
    comparison = compare_bits(decoded, tensors["inner"]["bits"])
    require(
        comparison["bitwise_equal"],
        f"Inner known-answer mismatch: {comparison['mismatches']}",
    )
    wrong_window = 16 if fixture["window"] != 16 else 12
    wrong = compare_bits(
        reconstruct_inner(*args, window=wrong_window), tensors["inner"]["bits"]
    )
    require(
        not wrong["bitwise_equal"],
        "Wrong-window negative control did not distinguish decoder",
    )
    result = dict(
        fixture,
        inner=comparison,
        wrong_window_control={"window": wrong_window, **wrong},
        qualified_scope="CPU MCG inner decode for this exact fixture",
        production_metal_vjp_qualified=False,
        serving_projection_qualified=False,
        target_qwen_window14_converter_fixture_qualified=False,
    )
    if public:
        weights = reconstruct_public(
            decoded,
            tensors["suh"]["bits"],
            tensors["svh"]["bits"],
            fixture["in_features"],
            fixture["out_features"],
        )
        comparison = compare_bits(weights, tensors["public"]["bits"])
        require(
            comparison["bitwise_equal"],
            f"Public known-answer mismatch: {comparison['mismatches']}",
        )
        result["public_matrix"] = comparison
        result["qualified_scope"] += (
            " and source-order composite public matrix; not serving arithmetic"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument(
        "--public", action="store_true", help="Also qualify CPU composite public matrix"
    )
    parser.add_argument(
        "--output", type=Path, help="New report; never overwrite an existing receipt"
    )
    args = parser.parse_args(argv)
    try:
        report = {
            "schema": "qwen-mcg-cpu-fixture-check-v1",
            "passed": True,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "fixtures": [verify_fixture(p, public=args.public) for p in args.fixtures],
            "scope": "CPU fixture known answers only; no full model, GPU, serving or training qualification",
        }
        if args.output:
            with args.output.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, OverflowError, struct.error) as error:
        print(f"Fixture verification refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
