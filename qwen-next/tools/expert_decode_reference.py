"""Tiny CPU bridge: verified Sushi expert bytes -> MCG public FP16 matrix.

This is a mathematical fixture bridge, not the serving FP16 activation path or
a production whole-expert loader. Python object storage greatly exceeds raw
FP16/F32 tensor bytes. Hard element/work caps intentionally reject the real
2560x640 expert. Window14 source math is supported, but no window14 converter
fixture, GPU parity, full architecture or training qualification is implied.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

MAX_ELEMENTS = 1 << 16
MAX_ESTIMATED_PYTHON_BYTES = 16 << 20
MAX_HADAMARD_TERMS = 2 * 128 * MAX_ELEMENTS
PACK_CONTRACT = Path(__file__).resolve().parents[1] / "sources/pack-format.md"


def _helper(name):
    path = Path(__file__).resolve().with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("expert_decode_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rows = _helper("expert_row_reader")
trellis = _helper("trellis_reference")


class DecodeError(ValueError):
    pass


def require(value, message):
    if not value:
        raise DecodeError(message)


def _sources():
    paths = (
        Path(__file__).resolve(),
        Path(rows.__file__).resolve(),
        Path(rows.inventory.__file__).resolve(),
        Path(rows.sushi.__file__).resolve(),
        Path(trellis.__file__).resolve(),
        PACK_CONTRACT,
    )
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


_LOADED_SOURCES = _sources()


def _source_guard():
    require(
        _sources() == _LOADED_SOURCES,
        "Bridge/helper/format sources changed after import",
    )


def _limit(value, maximum, name):
    require(
        type(value) is int and 0 < value <= maximum, f"{name} must be in1..{maximum}"
    )
    return value


def _halfwords(raw, expected, name):
    require(
        type(raw) is bytes and len(raw) == 2 * expected,
        f"Wrong {name} payload byte count",
    )
    return list(struct.unpack(f"<{expected}H", raw))


def _bits_hash(bits):
    return hashlib.sha256(struct.pack(f"<{len(bits)}H", *bits)).hexdigest()


@dataclass(frozen=True)
class DecodedExpert:
    """Immutable row-major[input,output] half bit patterns, detached from disk."""

    in_features: int
    out_features: int
    public_bits: tuple
    _receipt_json: str

    def receipt(self):
        return json.loads(self._receipt_json)

    def _matmul(self, vector, transpose):
        _source_guard()
        value = trellis.frozen_matmul(
            vector,
            self.public_bits,
            self.in_features,
            self.out_features,
            transpose=transpose,
        )
        require(
            all(math.isfinite(x) for x in value),
            "Mathematical matmul produced nonfinite values",
        )
        return value

    def forward(self, vector):
        """Float64 vector @ public matrix; not serving activation arithmetic."""
        return self._matmul(vector, False)

    def transpose(self, cotangent):
        """Float64 cotangent @ public matrix.T, a linear-adjoint reference."""
        return self._matmul(cotangent, True)


def decode_expert(
    directory,
    identity,
    *,
    expected_sha256,
    expert,
    max_elements=MAX_ELEMENTS,
    max_estimated_python_bytes=MAX_ESTIMATED_PYTHON_BYTES,
    max_hadamard_terms=MAX_HADAMARD_TERMS,
):
    """Read/decode one tiny uniform bank row with explicit scale semantics.

    Stored suh/svh are passed through once, without an additional codebook
    g-scale. That is the pinned Sushi format's folded-suh contract. No attempt
    is made to infer or fix differently normalized data. Real Qwen dimensions
    exceed this fixture-only bridge's hard cap and are refused before payload
    reads, even though their compressed rows may be small enough to load.
    """
    _source_guard()
    _limit(max_elements, MAX_ELEMENTS, "max_elements")
    _limit(
        max_estimated_python_bytes,
        MAX_ESTIMATED_PYTHON_BYTES,
        "max_estimated_python_bytes",
    )
    _limit(max_hadamard_terms, MAX_HADAMARD_TERMS, "max_hadamard_terms")
    require(
        isinstance(identity, dict)
        and rows.identity_sha256(identity) == expected_sha256,
        "Pinned identity digest mismatch",
    )
    bound = json.loads(rows.canonical(identity))
    # Constructor validates config, index, header, all spans and source/stat
    # identities without reading expert payload bytes.
    reader = rows.ExpertRowReader(directory, bound, expected_sha256=expected_sha256)
    require(
        type(expert) is int and 0 <= expert < len(bound["experts"]),
        "Expert index out of bounds",
    )
    plan = bound["experts"][expert]
    views = plan["views"]
    shape = views["trellis"]["shape"]
    require(
        len(shape) == 3 and all(type(x) is int and x > 0 for x in shape),
        "Invalid trellis geometry",
    )
    inputs, outputs, n = shape[0] * 16, shape[1] * 16, shape[2]
    elements = inputs * outputs
    require(
        elements <= max_elements, "Whole-expert matrix exceeds CPU fixture element cap"
    )
    trellis.matrix_geometry(inputs, outputs, block=128)
    trellis.validate_rate(n)
    trellis.validate_window(plan["window"])
    require(
        plan["bits_per_weight"] == n / 16, "Bound rate disagrees with tile geometry"
    )
    require(
        views["suh"]["shape"] == [inputs] and views["svh"]["shape"] == [outputs],
        "Scale vectors disagree with public matrix geometry",
    )
    words = shape[0] * shape[1] * n + inputs + outputs
    # Planning allowance, NOT a measured process-memory cap. Object counts and
    # work are independently hard-bounded above; interpreter peaks may differ.
    estimate = elements * 192 + words * 40 + plan["total_read_bytes"] + 65536
    require(
        estimate <= max_estimated_python_bytes,
        "CPU Python working-set estimate exceeds fixture budget",
    )
    terms = elements * 2 * 128
    require(terms <= max_hadamard_terms, "CPU Hadamard work exceeds fixture budget")
    result = reader.read(expert)
    raw = result["payloads"]
    packed = _halfwords(raw["trellis"], shape[0] * shape[1] * n, "trellis")
    suh = _halfwords(raw["suh"], inputs, "suh")
    svh = _halfwords(raw["svh"], outputs, "svh")
    require(
        all(math.isfinite(trellis.half_value(x)) for x in suh + svh),
        "Stored scale vectors contain nonfinite values",
    )
    inner = trellis.reconstruct_inner(packed, inputs, outputs, n, window=plan["window"])
    try:
        public = trellis.reconstruct_public(inner, suh, svh, inputs, outputs)
    except (OverflowError, struct.error) as error:
        raise DecodeError(
            "Public matrix reconstruction overflowed its FP16/FP32 reference"
        ) from error
    require(
        all(math.isfinite(trellis.half_value(x)) for x in public),
        "Public matrix contains nonfinite values",
    )
    # A long CPU reconstruction does not waive the file/source guards.
    for component in ("config", "index", "shard"):
        item = bound[component]
        path = Path(bound["checkpoint_directory"]) / item["path"]
        require(
            rows.inventory.regular_identity(path) == item["stat"],
            f"Bound {component} changed during decode",
        )
    _source_guard()
    receipt = {
        "format": "sushi-expert-cpu-public-decode-v1",
        "expert": expert,
        "identity_sha256": expected_sha256,
        "read_receipt": result["receipt"],
        "implementation_sha256": _LOADED_SOURCES,
        "matrix_shape": [inputs, outputs],
        "orientation": "row-major[input,output]",
        "codebook": "mcg",
        "window": plan["window"],
        "halfwords_per_tile": n,
        "bits_per_weight": n / 16,
        "scale_contract": "Apply stored F16 suh/svh once in reconstruct_public; g-scale is already folded into suh per the pinned Sushi pack contract. No extra scaling, quantization or normalization.",
        "inner_fp16_bits_sha256": _bits_hash(inner),
        "public_fp16_bits_sha256": _bits_hash(public),
        "bounds": {
            "elements": elements,
            "max_elements": max_elements,
            "raw_payload_bytes": result["receipt"]["payload_bytes_read"],
            "hypothetical_contiguous_fp16_bytes": elements * 2,
            "hypothetical_contiguous_fp32_bytes": elements * 4,
            "python_working_set_estimate_bytes": estimate,
            "max_python_estimate_bytes": max_estimated_python_bytes,
            "hadamard_terms": terms,
            "max_hadamard_terms": max_hadamard_terms,
            "measured_process_peak_bytes": None,
        },
        "matrix_representation": "Immutable tuple of Python integers holding FP16 bit patterns; storage exceeds contiguous FP16/F32 byte counts.",
        "serving_projection_qualified": False,
        "target_qwen_window14_converter_fixture_qualified": False,
        "gpu_parity_qualified": False,
        "full_architecture_qualified": False,
        "training_supported": False,
        "scope": "Bounded synthetic CPU public-matrix reconstruction and float64 linear/adjoint oracle. Not serving's FP16 transformed-activation computation or production memory qualification.",
    }
    return DecodedExpert(
        inputs,
        outputs,
        tuple(public),
        json.dumps(receipt, sort_keys=True, allow_nan=False),
    )
