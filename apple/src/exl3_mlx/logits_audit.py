"""CPU-only comparison of complete, selected vocabulary rows.

The portable manifest binds local files and *asserted* producer identities. This
auditor does not attest a process, verify base checkpoint bytes, run inference,
or certify serving compatibility. Native selected_raw_logits_v1 observations
are supported without importing the experimental serving implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from pathlib import Path

from .evaluation_audit import _canonical, _finite, _json, _require

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_VOCAB = 1 << 20
_MAX_PREFIX = 1 << 20
_SHARED = (
    "base_payload_manifest_sha256",
    "runtime_manifest_sha256",
    "tokenizer_sha256",
    "arithmetic_policy_sha256",
)
_DOMAIN = b"mlx-stream-adapter-transfer-stock-v1\0"


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value, label):
    _require(isinstance(value, str) and _HEX.fullmatch(value), f"Invalid {label} SHA256")
    return value


def _object(value, label):
    _require(isinstance(value, dict), f"{label}: expected object")
    return value


def _int(value, label, low=0, high=(1 << 63) - 1):
    _require(type(value) is int and low <= value <= high, f"Invalid {label}")
    return value


def _same(left, right):
    # JSON equality alone treats bool as an integer.
    return _canonical(left) == _canonical(right)


def _stat(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _capture(path, limit):
    _require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
    before = _stat(path)
    _require(before[2] <= limit, f"Input size limit: {path}")
    raw = path.read_bytes()
    _require(_stat(path) == before and len(raw) == before[2], f"Input drift: {path}")
    return raw, before


class _Inputs:
    def __init__(self, root):
        self.root = root
        self.captured = {}

    def read(self, descriptor, limit):
        d = _object(descriptor, "file descriptor")
        _require(set(d) == {"path", "sha256"}, "File descriptor requires path and sha256")
        rel = d["path"]
        _require(isinstance(rel, str) and rel and "\\" not in rel, "Invalid relative path")
        p = Path(rel)
        _require(
            not p.is_absolute() and all(x not in {"", ".", ".."} for x in rel.split("/")),
            "Artifact path must be portable and contained",
        )
        path = self.root
        for part in p.parts:
            path = path / part
            _require(not path.is_symlink(), f"Symlink artifact component: {path}")
        _require(path.resolve().is_relative_to(self.root), "Escaping artifact path")
        raw, identity = _capture(path, limit)
        digest = _hex(d["sha256"], "file")
        _require(_hash(raw) == digest, f"File hash mismatch: {rel}")
        old = self.captured.get(rel)
        _require(old is None or old == (identity, digest), f"Rebound artifact: {rel}")
        self.captured[rel] = (identity, digest)
        return raw

    def recheck(self):
        for rel, (identity, digest) in self.captured.items():
            path = self.root / rel
            raw = self.read({"path": rel, "sha256": digest}, identity[2])
            _require(_stat(path) == identity and _hash(raw) == digest, f"Input drift: {rel}")


def _tensors(raw):
    """Strict bounded F32 LoRA safetensors parser; never executes model code."""
    _require(len(raw) >= 8, "Truncated adapter")
    header_length = struct.unpack_from("<Q", raw)[0]
    _require(
        2 <= header_length <= (1 << 20) and 8 + header_length < len(raw),
        "Invalid adapter header length",
    )
    header = _object(_json(raw[8 : 8 + header_length]), "adapter header")
    if "__metadata__" in header:
        metadata = _object(header["__metadata__"], "adapter metadata")
        _require(
            all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()),
            "Invalid safetensors metadata",
        )
    tensors = {}
    spans = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        _require(key.endswith((".lora_a", ".lora_b")), "Non-LoRA tensor in adapter")
        value = _object(value, "tensor")
        _require(
            set(value) == {"dtype", "shape", "data_offsets"} and value["dtype"] == "F32",
            "Only explicit F32 LoRA tensor pairs are supported",
        )
        shape, offsets = value["shape"], value["data_offsets"]
        _require(
            isinstance(shape, list)
            and len(shape) == 2
            and all(type(x) is int and x > 0 for x in shape),
            "Invalid LoRA shape",
        )
        _require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(x) is int and x >= 0 for x in offsets),
            "Invalid tensor offsets",
        )
        start, end = offsets
        _require(
            start < end <= len(raw) - 8 - header_length and end - start == math.prod(shape) * 4,
            "Invalid tensor span",
        )
        values = raw[8 + header_length + start : 8 + header_length + end]
        _require(
            all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", values)),
            "Nonfinite adapter value",
        )
        tensors[key] = (shape, values)
        spans.append((start, end))
    _require(tensors and len(tensors) % 2 == 0, "Missing LoRA pairs")
    cursor = 0
    for start, end in sorted(spans):
        _require(start == cursor, "Adapter overlap, hole or trailing payload")
        cursor = end
    _require(cursor == len(raw) - 8 - header_length, "Trailing adapter payload")
    for key, (shape, _) in tensors.items():
        other = key[:-6] + ("lora_b" if key.endswith("lora_a") else "lora_a")
        _require(other in tensors, "Unpaired LoRA tensor")
        if key.endswith("lora_a"):
            _require(shape[1] == tensors[other][0][0], "LoRA rank mismatch")
    return tensors


def _adapter(inputs, value):
    value = _object(value, "adapter")
    _require(set(value) == {"config", "weights"}, "Adapter requires config and weights")
    config_raw = inputs.read(value["config"], 1 << 20)
    weights = inputs.read(value["weights"], 64 << 20)
    config = _object(_json(config_raw), "adapter config")
    _require(config.get("adapter_sha256") == _hash(weights), "Adapter config/payload mismatch")
    tensors = _tensors(weights)
    return config, tensors, _hash(_DOMAIN + hashlib.sha256(weights).digest() + config_raw)


def _zero_control(inputs, condition, config, tensors):
    reference, original, fingerprint = _adapter(inputs, condition.get("zero_b_reference"))
    _require(set(tensors) == set(original), "Zero-B/reference tensor keys differ")
    _require(
        _same(
            {k: v for k, v in config.items() if k != "adapter_sha256"},
            {k: v for k, v in reference.items() if k != "adapter_sha256"},
        ),
        "Zero-B/reference config changed beyond payload hash",
    )
    for key, (shape, raw) in tensors.items():
        _require(shape == original[key][0], "Zero-B/reference tensor shape mismatch")
        if key.endswith(".lora_a"):
            _require(raw == original[key][1], "Zero-B A tensor changed")
        else:
            _require(all(x[0] == 0.0 for x in struct.iter_unpack("<f", raw)), "B is nonzero")
    return {
        "a_unchanged": True,
        "b_all_zero": True,
        "reference_fingerprint": fingerprint,
        "reference_payload_sha256": reference["adapter_sha256"],
        "config_unchanged_except_payload_sha256": True,
    }


def _row(inputs, value, identity, vocab):
    value = _object(value, "observation")
    _require(set(value) == {"metadata", "values", "prefix_ids"}, "Invalid observation fields")
    ids = value["prefix_ids"]
    _require(
        isinstance(ids, list)
        and 1 <= len(ids) <= _MAX_PREFIX
        and all(type(x) is int and 0 <= x < vocab for x in ids),
        "Invalid prefix IDs",
    )
    metadata = _object(_json(inputs.read(value["metadata"], 1 << 20)), "observation metadata")
    raw = inputs.read(value["values"], _MAX_VOCAB * 4)
    digest = _hash(struct.pack("<" + "I" * len(ids), *ids))
    expected = {
        "format": "selected_raw_logits_v1",
        "position": len(ids),
        "prefix_ids_le_u32_sha256": digest,
        "vocab_size": vocab,
        "serialized_dtype": "F32_LE",
        "raw_bytes": vocab * 4,
        "raw_sha256": _hash(raw),
        "request_sha256": identity["request_sha256"],
        "model_instance": identity["model_instance"],
        "adapter_fingerprint": identity["adapter_fingerprint"],
        "all_finite": True,
        "observer_changes_timing": True,
    }
    for key, expected_value in expected.items():
        _require(_same(metadata.get(key), expected_value), f"Observation identity mismatch: {key}")
    _require(len(raw) == vocab * 4, "Incomplete vocabulary row")
    _require(
        metadata.get("source_dtype") in ("float32", "float16", "bfloat16"),
        "Unsupported observed dtype",
    )
    _int(metadata.get("rows_in_forward"), "rows_in_forward", 1, len(ids))
    raw_name = metadata.get("raw_file")
    _require(
        isinstance(raw_name, str) and raw_name == f"logits-{len(ids)}.f32",
        "Unexpected original raw filename",
    )
    argmax, best = 0, -math.inf
    for i, (x,) in enumerate(struct.iter_unpack("<f", raw)):
        _require(math.isfinite(x), "Nonfinite observed value")
        if x > best:
            argmax, best = i, x
    _require(
        type(metadata.get("argmax")) is int and metadata["argmax"] == argmax,
        "Observed argmax mismatch",
    )
    return {
        "position": len(ids),
        "prefix_ids_le_u32_sha256": digest,
        "source_dtype": metadata["source_dtype"],
        "argmax": argmax,
        "values": value["values"],
        "vocab_size": vocab,
    }


def _condition(inputs, condition):
    c = _object(condition, "condition")
    _require(isinstance(c.get("id"), str) and _NAME.fullmatch(c["id"]), "Invalid condition ID")
    _require(c.get("role") in ("baseline", "zero_b", "adapter"), "Invalid condition role")
    identity = _object(c.get("identity"), "condition identity")
    _require(
        set(identity) == set(_SHARED) | {"request_sha256", "model_instance", "adapter_fingerprint"},
        "Incomplete or unknown identity fields",
    )
    for key in (*_SHARED, "request_sha256", "adapter_fingerprint"):
        _hex(identity[key], key)
    _int(identity["model_instance"], "model instance", 1)
    vocab = _int(c.get("vocab_size"), "vocabulary", 1, _MAX_VOCAB)
    control = None
    if c["role"] == "baseline":
        _require(
            identity["adapter_fingerprint"] == "0" * 64
            and "adapter" not in c
            and "zero_b_reference" not in c,
            "Baseline must have no adapter",
        )
    else:
        config, tensors, fingerprint = _adapter(inputs, c.get("adapter"))
        _require(fingerprint == identity["adapter_fingerprint"], "Adapter fingerprint mismatch")
        if c["role"] == "zero_b":
            control = _zero_control(inputs, c, config, tensors)
        else:
            _require("zero_b_reference" not in c, "Reference only allowed for zero-B control")
    evidence = c.get("evidence", [])
    _require(isinstance(evidence, list) and len(evidence) <= 32, "Invalid evidence list")
    for descriptor in evidence:
        inputs.read(descriptor, 16 << 20)
    observations = c.get("observations")
    _require(
        isinstance(observations, list) and 1 <= len(observations) <= 8,
        "Require one to eight explicitly selected observations",
    )
    rows = [_row(inputs, row, identity, vocab) for row in observations]
    positions = [r["position"] for r in rows]
    _require(positions == sorted(set(positions)), "Duplicate or unordered observed positions")
    for earlier, later in zip(observations, observations[1:]):
        prefix = earlier["prefix_ids"]
        _require(
            later["prefix_ids"][: len(prefix)] == prefix,
            "Inconsistent prefix-extension trajectory within one condition/request",
        )
    return {
        "id": c["id"],
        "role": c["role"],
        "identity": identity,
        "rows": rows,
        "zero_b_control": control,
        "vocab_size": vocab,
    }


def compare_logit_values(left, right, *, atol=0.0, rtol=0.0):
    """Compare two complete F32-LE byte rows; no implicit tolerance relaxation.

    The symmetric rule is abs(a-b) <= atol + rtol * max(abs(a), abs(b)).
    Signed zero is numerically equal but its changed bits remain visible.
    """
    for label, value in (("atol", atol), ("rtol", rtol)):
        _require(_finite(value) and value >= 0, f"Invalid {label}")
    _require(
        isinstance(left, bytes)
        and isinstance(right, bytes)
        and 0 < len(left) == len(right) <= _MAX_VOCAB * 4
        and len(left) % 4 == 0,
        "Rows must be equal, nonempty complete F32 vocabularies",
    )
    n = len(left) // 4
    max_abs = max_relative = absolute_sum = 0.0
    numeric_diff = outside = 0
    argmax_left = argmax_right = 0
    best_left = best_right = -math.inf
    for i, ((a,), (b,)) in enumerate(
        zip(struct.iter_unpack("<f", left), struct.iter_unpack("<f", right), strict=True)
    ):
        _require(math.isfinite(a) and math.isfinite(b), "Nonfinite logit")
        delta, scale = abs(a - b), max(abs(a), abs(b))
        max_abs = max(max_abs, delta)
        max_relative = max(max_relative, delta / scale if scale else 0.0)
        absolute_sum += delta
        numeric_diff += a != b
        outside += delta > atol + rtol * scale
        if a > best_left:
            argmax_left, best_left = i, a
        if b > best_right:
            argmax_right, best_right = i, b
    return {
        "values": n,
        "bitwise_equal": left == right,
        "numeric_different_values": numeric_diff,
        "max_absolute_error": max_abs,
        "mean_absolute_error": absolute_sum / n,
        "max_symmetric_relative_error": max_relative,
        "outside_tolerance_values": outside,
        "within_tolerance": outside == 0,
        "argmax_left": argmax_left,
        "argmax_right": argmax_right,
        "argmax_equal": argmax_left == argmax_right,
        "atol": atol,
        "rtol": rtol,
        "tolerance_rule": "abs(a-b) <= atol + rtol * max(abs(a), abs(b))",
    }


def audit_logit_observations(manifest, directory, *, atol=0.0, rtol=0.0):
    """Validate portable source-bound observations and explicitly selected pairs.

    `directory` contains relative, SHA256-bound files. Shared model/runtime/
    tokenizer/arithmetic hashes are assertions, not independent verification.
    Comparisons select every intended position explicitly: mismatched prefixes
    fail instead of silently taking an intersection after generation diverges.
    """
    # Validate tolerances even if an invalid comparison list follows.
    compare_logit_values(struct.pack("<f", 0), struct.pack("<f", 0), atol=atol, rtol=rtol)
    m = _object(manifest, "manifest")
    _require(m.get("format") == "observed-logits-audit-v1", "Unknown logits audit manifest")
    conditions = m.get("conditions")
    comparisons = m.get("comparisons")
    _require(isinstance(conditions, list) and 2 <= len(conditions) <= 16, "Require 2–16 conditions")
    _require(
        isinstance(comparisons, list) and 1 <= len(comparisons) <= 64,
        "Require 1–64 explicit comparisons",
    )
    inputs = _Inputs(Path(directory).resolve(strict=True))
    checked = {}
    for condition in conditions:
        value = _condition(inputs, condition)
        _require(value["id"] not in checked, "Duplicate condition ID")
        checked[value["id"]] = value
    results, used, seen = [], set(), set()
    for pair in comparisons:
        pair = _object(pair, "comparison")
        _require(
            set(pair) == {"left", "right", "purpose", "positions"}, "Invalid comparison fields"
        )
        left_id, right_id, purpose = pair["left"], pair["right"], pair["purpose"]
        _require(
            isinstance(left_id, str)
            and isinstance(right_id, str)
            and left_id in checked
            and right_id in checked
            and left_id != right_id,
            "Invalid comparison condition",
        )
        _require(purpose in ("zero_b_control", "cache_parity", "adapter_effect"), "Invalid purpose")
        left, right = checked[left_id], checked[right_id]
        for key in _SHARED:
            _require(left["identity"][key] == right["identity"][key], f"Different {key}")
        _require(left["vocab_size"] == right["vocab_size"], "Different vocabularies")
        if purpose == "zero_b_control":
            _require(
                {left["role"], right["role"]} == {"baseline", "zero_b"}, "Require baseline/zero-B"
            )
        elif purpose == "cache_parity":
            _require(
                left["role"] == right["role"]
                and left["identity"]["adapter_fingerprint"]
                == right["identity"]["adapter_fingerprint"],
                "Cache comparison must use the same adapter",
            )
        else:
            _require(
                "adapter" in {left["role"], right["role"]}, "Effect comparison requires adapter"
            )
        positions = pair["positions"]
        _require(
            isinstance(positions, list)
            and positions
            and all(type(p) is int and p > 0 for p in positions)
            and positions == sorted(set(positions)),
            "Invalid comparison positions",
        )
        signature = (left_id, right_id, purpose, tuple(positions))
        _require(signature not in seen, "Duplicate comparison")
        seen.add(signature)
        maps = [{r["position"]: r for r in c["rows"]} for c in (left, right)]
        rows = []
        for position in positions:
            _require(position in maps[0] and position in maps[1], "Missing selected observation")
            a, b = maps[0][position], maps[1][position]
            _require(
                a["prefix_ids_le_u32_sha256"] == b["prefix_ids_le_u32_sha256"], "Different prefixes"
            )
            _require(a["source_dtype"] == b["source_dtype"], "Different source dtypes")
            values = compare_logit_values(
                inputs.read(a["values"], _MAX_VOCAB * 4),
                inputs.read(b["values"], _MAX_VOCAB * 4),
                atol=atol,
                rtol=rtol,
            )
            rows.append(
                {
                    "position": position,
                    "prefix_ids_le_u32_sha256": a["prefix_ids_le_u32_sha256"],
                    **values,
                }
            )
            used.update(((left_id, position), (right_id, position)))
        results.append(
            {
                **pair,
                "rows": rows,
                "all_bitwise_equal": all(r["bitwise_equal"] for r in rows),
                "all_within_tolerance": all(r["within_tolerance"] for r in rows),
                "equality_required_by_purpose": purpose != "adapter_effect",
                "improvement_claimed": False,
            }
        )
    inputs.recheck()
    required = [r for r in results if r["equality_required_by_purpose"]]
    required_passed = all(r["all_within_tolerance"] for r in required)
    status = (
        "observations_consistent_required_comparisons_passed"
        if required
        else "observations_consistent_descriptive_only"
    )
    if not required_passed:
        status = "observations_consistent_required_comparisons_failed"
    return {
        "schema": "observed-logits-comparison-v1",
        "status": status,
        "observation_evidence_consistent": True,
        "required_comparisons": len(required),
        "required_comparisons_passed": required_passed,
        "identity_scope": "asserted producer identities checked for equality, not process attestation",
        "conditions": [{k: v for k, v in c.items() if k != "rows"} for c in checked.values()],
        "comparisons": results,
        "atol": atol,
        "rtol": rtol,
        "uncompared_observations": [
            {"condition": c["id"], "position": r["position"]}
            for c in checked.values()
            for r in c["rows"]
            if (c["id"], r["position"]) not in used
        ],
        "files_sha256": {rel: value[1] for rel, value in sorted(inputs.captured.items())},
        "base_payload_independently_verified": False,
        "device_qualified": False,
        "quality_certified": False,
        "trainer_forward_parity": False,
        "limits": [
            "No engine invocation, base-payload read or performance measurement",
            "Copied original receipts are hash-bound evidence, not independently authenticated",
            "Selected full-vocabulary rows only; unobserved positions remain unqualified",
            "A nonzero adapter delta is descriptive, not evidence of improvement",
            "Observation adds synchronization; these files do not establish throughput",
        ],
    }


def audit_logit_files(*, manifest, output, atol=0.0, rtol=0.0):
    """Capture a portable manifest and exclusively publish a complete audit."""
    path = Path(manifest).expanduser().absolute()
    destination = Path(output).expanduser().absolute()
    _require(not destination.exists() and not destination.is_symlink(), "Output already exists")
    raw, identity = _capture(path, 16 << 20)
    result = audit_logit_observations(_json(raw), path.parent, atol=atol, rtol=rtol)
    result["manifest_sha256"] = _hash(raw)
    result["implementation_sha256"] = {
        "logits_audit.py": _hash(Path(__file__).read_bytes()),
        "evaluation_audit.py": _hash(Path(__file__).with_name("evaluation_audit.py").read_bytes()),
    }
    _require(_capture(path, 16 << 20) == (raw, identity), "Manifest changed during audit")
    # Recheck every evidence file after comparison and before publication.
    bound = _Inputs(path.parent.resolve())
    for name, digest in result["files_sha256"].items():
        bound.read({"path": name, "sha256": digest}, 64 << 20)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=".logits-audit-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.link(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        result = audit_logit_files(**vars(args))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Logits audit failed: {error}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparisons": len(result["comparisons"]),
                "required_comparisons_passed": result["required_comparisons_passed"],
                "device_qualified": False,
                "quality_certified": False,
            }
        )
    )
    if not result["required_comparisons_passed"]:
        parser.exit(1, "Required numerical comparison failed; complete audit was published.\n")


if __name__ == "__main__":
    main()
