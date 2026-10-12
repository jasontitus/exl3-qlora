"""Check actual-shape BF16 native projection and head VJPs against NumPy oracles.

Run alone after other GPU jobs finish. Frozen checkpoint files are read only.
The oracle decodes MXFP8 itself and streams head vocabulary rows in float64;
it never uses mx.dequantize, NativeLinear, or the engine quantizer for reference
math. Inputs are deterministic synthetic rows, not captured training activations.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np


def bf16(value):
    """Round finite float32 values to BF16, represented as NumPy float32."""
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("The numerical oracle expects finite values")
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def e4m3_table():
    codes = np.arange(256, dtype=np.uint16)
    exponent, fraction = (codes >> 3) & 15, codes & 7
    magnitude = np.where(
        exponent == 0,
        fraction.astype(np.float64) * 2.0**-9,
        (1 + fraction.astype(np.float64) / 8) * np.exp2(exponent.astype(np.int32) - 7),
    )
    values = np.where(codes & 128, -magnitude, magnitude)
    values[(exponent == 15) & (fraction == 7)] = np.nan
    return values


def decode_mxfp8(packed, scales):
    """Row-major four E4M3 bytes per uint32, one E8M0 scale per 32 values."""
    packed, scales = np.asarray(packed, dtype=np.uint32), np.asarray(scales, dtype=np.uint8)
    if (
        packed.ndim != 2
        or packed.shape[1] % 8
        or scales.shape
        != (
            packed.shape[0],
            packed.shape[1] // 8,
        )
    ):
        raise ValueError("Invalid MXFP8 row layout")
    codes = ((packed[..., None] >> (np.arange(4, dtype=np.uint32) * 8)) & 255).reshape(
        packed.shape[0], -1
    )
    factors = np.exp2(scales.astype(np.int32) - 127).astype(np.float64)
    factors[scales == 255] = np.nan
    values = e4m3_table()[codes] * np.repeat(factors, 32, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite MXFP8 weight in checkpoint")
    return values


def quantize_fp8_reference(value):
    """Independent block-32 activation quantization, nearest-even E4M3."""
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] % 32 or not np.isfinite(value).all():
        raise ValueError("Expected finite complete activation groups of 32")
    groups = value.reshape(-1, 32).astype(np.float64)
    scale = np.exp2(np.ceil(np.log2(np.maximum(np.max(np.abs(groups), axis=1), 1e-4) / 448)))
    normalized = np.abs(groups) / scale[:, None]
    levels = e4m3_table()[:127]
    upper = np.minimum(np.searchsorted(levels, normalized, side="left"), len(levels) - 1)
    lower = np.maximum(upper - 1, 0)
    low_distance, high_distance = normalized - levels[lower], levels[upper] - normalized
    choose_lower = (low_distance < high_distance) | (
        (low_distance == high_distance) & (lower % 2 == 0)
    )
    code = np.where(choose_lower, lower, upper)
    return (np.sign(groups) * levels[code] * scale[:, None]).reshape(value.shape).astype(np.float32)


class NumpyArchive:
    """Independent bounded safetensors reader, with mutation checks on every read."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.headers, self.fingerprints, self.payload_hashes = {}, {}, {}

    @staticmethod
    def fingerprint(path):
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def spec(self, key):
        path = (self.root / self.index[key]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Weight shard path escapes checkpoint")
        if path not in self.headers:
            self.fingerprints[path] = self.fingerprint(path)
            with path.open("rb") as source:
                length = int.from_bytes(source.read(8), "little")
                if not 0 < length <= 16 << 20:
                    raise ValueError("Invalid safetensors header")
                self.headers[path] = 8 + length, json.loads(source.read(length))
        if self.fingerprint(path) != self.fingerprints[path]:
            raise ValueError(f"Frozen source changed: {path}")
        offset, header = self.headers[path]
        return path, offset, header[key]

    def read(self, key, start, end):
        path, offset, spec = self.spec(key)
        dtype = {"U32": "<u4", "U8": "u1", "BF16": "<u2"}[spec["dtype"]]
        shape = spec["shape"]
        if len(shape) != 2 or not 0 <= start < end <= shape[0]:
            raise ValueError("Invalid matrix row slice")
        stride = shape[1] * np.dtype(dtype).itemsize
        lo, hi = spec["data_offsets"]
        if hi - lo != shape[0] * stride:
            raise ValueError("Invalid matrix byte count")
        with path.open("rb") as source:
            source.seek(offset + lo + start * stride)
            raw = source.read((end - start) * stride)
        if len(raw) != (end - start) * stride:
            raise ValueError("Truncated weight shard")
        self.spec(key)
        return np.frombuffer(raw, dtype=dtype).reshape(end - start, shape[1]), raw

    def blocks(self, key, *, block_rows, rows=None, hash_payload=False):
        _, _, spec = self.spec(key + ".weight")
        first, last = rows or (0, spec["shape"][0])
        weight_hash, scale_hash = hashlib.sha256(), hashlib.sha256()
        for start in range(first, last, block_rows):
            end = min(start + block_rows, last)
            packed, raw = self.read(key + ".weight", start, end)
            weight_hash.update(raw)
            if spec["dtype"] == "U32":
                scales, raw_scale = self.read(key + ".scales", start, end)
                scale_hash.update(raw_scale)
                decoded = decode_mxfp8(packed, scales)
            elif spec["dtype"] == "BF16":
                decoded = (packed.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            else:
                raise ValueError("This diagnostic expects MXFP8 or BF16 weights")
            yield start - first, decoded
        if hash_payload:
            self.payload_hashes[key] = {
                "rows": [first, last],
                "weight_sha256": weight_hash.hexdigest(),
                "scales_sha256": scale_hash.hexdigest() if spec["dtype"] == "U32" else None,
            }


def metrics(actual, expected):
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if not finite:
        return {"passed": False, "finite": False}
    delta = actual - expected
    reference_norm, actual_norm = float(np.linalg.norm(expected)), float(np.linalg.norm(actual))
    maximum = float(np.max(np.abs(expected)))
    error = float(np.max(np.abs(delta)))
    relative = float(np.linalg.norm(delta) / max(reference_norm, 1e-30))
    cosine = (
        float(np.sum(actual * expected) / (reference_norm * actual_norm))
        if reference_norm and actual_norm
        else (1.0 if np.array_equal(actual, expected) else 0.0)
    )
    # BF16 casts have 1/128 spacing at unit scale. These fixed, predeclared
    # tolerances allow reduction/rounding changes while testing direction too.
    limits = {"max_abs_per_reference_max": 0.02, "relative_l2": 0.012, "cosine_min": 0.9999}
    return {
        "passed": error <= 0.02 * maximum + 1e-7 and relative <= 0.012 and cosine >= 0.9999,
        "finite": True,
        "max_abs_error": error,
        "reference_max_abs": maximum,
        "relative_l2_error": relative,
        "cosine": cosine,
        "actual_l2": actual_norm,
        "reference_l2": reference_norm,
        "tolerance": limits | {"absolute_floor": 1e-7},
    }


def ce_reference(logits, targets):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    denominator = np.exp(shifted).sum(axis=-1, keepdims=True)
    probability = np.exp(shifted) / denominator
    loss = np.mean(np.log(denominator[:, 0]) - shifted[np.arange(len(targets)), targets])
    probability[np.arange(len(targets)), targets] -= 1
    return float(loss), probability / len(targets)


def oracle_row_indices(total, requested):
    """Fixed sample: endpoints, attention-tile/window boundaries, then coverage."""
    if type(total) is not int or type(requested) is not int or total < 1 or requested < 1:
        raise ValueError("Positive integer row counts required")
    count = min(total, requested)
    if count == total:
        return np.arange(total, dtype=np.int32)
    chosen = []
    proposals = [0, total - 1, 15, 16, 127, 128, 1, total - 2]
    proposals += np.linspace(0, total - 1, count + 2, dtype=np.int64).tolist()
    for row in proposals:
        if 0 <= row < total and row not in chosen:
            chosen.append(row)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        raise AssertionError("Could not construct the requested fixed sample")
    return np.array(sorted(chosen), dtype=np.int32)


def atomic_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    # Deliberately lazy: oracle tests can import this module without initializing Metal.
    import mlx.core as mx
    import mlx.nn as nn

    import exl3_mlx.attention as attention_source
    import exl3_mlx.native as native_source
    from exl3_mlx.memory import MemoryBudget
    from exl3_mlx.native import NativeLinear
    from exl3_mlx.storage import TensorArchive

    root = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Use a new diagnostic output outside the frozen checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "config.json").read_text())["text_config"]
    if (
        not 0 <= args.layer < config["num_hidden_layers"]
        or not 0 <= args.wo_group < config["o_groups"]
    ):
        raise ValueError("Layer/output group is outside this architecture")
    reference = NumpyArchive(root)
    source = TensorArchive(root)
    rng = np.random.default_rng(args.seed)
    report = {
        "scope": "actual checkpoint, full feature dimensions; all executed rows finite, selected rows checked by oracle",
        "diagnostic_revision": "native-vjp-v2-sampled-rows",
        "settings": vars(args),
        "checkpoint": str(root),
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
        "source_sha256": {
            str(Path(module.__file__).resolve()): hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (attention_source, native_source)
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
        },
        "notes": [
            "No full-model gradient or quality claim follows from these component checks.",
            "Small row batches test actual feature dimensions but may select different Metal kernels from 5808-row prefill.",
            "With --rows 5808 the trunk executes the full prefill row count; --oracle-rows controls independent sampled row checks only.",
            "Linear rows are independent: sampled VJP rows use their corresponding nonuniform full-run cotangents.",
            "The head oracle rounds dlogits to BF16 at the cast boundary before its transpose multiply.",
            "Reference weights are independently decoded into NumPy float64 blocks; the full FP64 head is never retained.",
            "Passing tolerances is component numerical evidence, not bitwise equivalence.",
        ],
        "cases": [],
        "complete": False,
    }
    budget = MemoryBudget(args.memory_gib, cache_mib=64, reserve_gib=args.reserve_gib)
    budget.abort_path = output.with_suffix(".memory-abort.json")

    def numpy_value(value):
        mx.eval(value)
        return np.array(value.astype(mx.float32))

    def publish(case):
        case.update(
            peak_mlx_gib=mx.get_peak_memory() / 2**30, process_accounted_gib=budget.usage() / 2**30
        )
        report["cases"].append(case)
        report["tensor_payloads"] = reference.payload_hashes
        atomic_report(output, report)
        print(f"{case['name']}: passed={case['passed']}", flush=True)

    with budget:
        for projection in args.projections:
            begin = time.monotonic()
            key = f"layers.{args.layer}.attn.{projection}"
            rows = None
            if projection == "wo_a":
                rank = config["o_lora_rank"]
                rows = (args.wo_group * rank, (args.wo_group + 1) * rank)
            layer = NativeLinear(source, key, rows=rows, quantize_input=projection != "wo_a")
            if not layer.quantized:
                raise ValueError(f"Expected an actual MXFP8 trunk projection: {key}")
            x = bf16(rng.normal(size=(args.rows, layer.in_features)))
            # Grouped wo_a receives the already quantized attention output.
            if projection == "wo_a":
                for start in range(0, len(x), 128):
                    x[start : start + 128] = bf16(quantize_fp8_reference(x[start : start + 128]))
            dy = bf16(rng.normal(size=(args.rows, layer.out_features)))
            (value,), (gradient,) = mx.vjp(
                layer, [mx.array(x, dtype=mx.bfloat16)], [mx.array(dy, dtype=mx.bfloat16)]
            )
            mx.eval(value, gradient)
            all_forward_finite = bool(mx.all(mx.isfinite(value)).item())
            all_gradient_finite = bool(mx.all(mx.isfinite(gradient)).item())
            sample = oracle_row_indices(args.rows, args.oracle_rows)
            sample_index = mx.array(sample)
            # Only these small slices cross to NumPy. The executed forward/VJP
            # above uses every input/cotangent row, including uninspected rows.
            actual, dx = numpy_value(value[sample_index]), numpy_value(gradient[sample_index])
            sample_x, sample_dy = x[sample], dy[sample]
            quantized_x = sample_x if projection == "wo_a" else quantize_fp8_reference(sample_x)
            expected = np.empty(actual.shape, dtype=np.float64)
            dx_reference = np.zeros(sample_x.shape, dtype=np.float64)
            for start, weight in reference.blocks(
                key, block_rows=args.weight_rows, rows=rows, hash_payload=True
            ):
                end = start + weight.shape[0]
                expected[:, start:end] = quantized_x.astype(np.float64) @ weight.T
                dx_reference += sample_dy[:, start:end].astype(np.float64) @ weight
                budget.check()
            forward, backward = metrics(actual, bf16(expected)), metrics(dx, bf16(dx_reference))
            publish(
                {
                    "name": key,
                    "weight_shape": [layer.out_features, layer.in_features],
                    "weight_rows": rows,
                    "input_shape": list(x.shape),
                    "oracle_row_indices": sample.tolist(),
                    "oracle_rows": len(sample),
                    "oracle_scope": "all_rows" if len(sample) == args.rows else "sampled_rows_only",
                    "all_forward_rows_finite": all_forward_finite,
                    "all_input_gradient_rows_finite": all_gradient_finite,
                    "input_dtype": "bfloat16",
                    "weight_format": "mxfp8-e4m3-e8m0-group32",
                    "hash_array_encoding": "contiguous float32 representation of exact BF16 values",
                    "input_sha256": hashlib.sha256(memoryview(x)).hexdigest(),
                    "cotangent_sha256": hashlib.sha256(memoryview(dy)).hexdigest(),
                    "forward": forward,
                    "input_vjp": backward,
                    "seconds": time.monotonic() - begin,
                    "passed": all_forward_finite
                    and all_gradient_finite
                    and forward["passed"]
                    and backward["passed"],
                }
            )
            del layer, value, gradient, x, dy, actual, dx, expected, dx_reference, weight
            del sample, sample_index, sample_x, sample_dy, quantized_x
            gc.collect()
            mx.clear_cache()

        if not args.skip_head:
            begin = time.monotonic()
            head = NativeLinear(source, "head")
            if head.weight.dtype != mx.bfloat16:
                raise ValueError("Expected actual BF16 LM head")
            x = bf16(rng.normal(size=(args.head_rows, head.in_features)))
            targets = rng.integers(head.out_features, size=args.head_rows, dtype=np.int32)
            target_array = mx.array(targets)

            def objective(value):
                return nn.losses.cross_entropy(
                    head(value).astype(mx.float32), target_array, reduction="mean"
                )

            input_array = mx.array(x, dtype=mx.bfloat16)
            loss, gradient = mx.value_and_grad(objective)(input_array)
            logits = head(input_array).astype(mx.float32)
            loss_value, actual, dx = float(loss.item()), numpy_value(logits), numpy_value(gradient)
            expected = np.empty(actual.shape, dtype=np.float64)
            for start, weight in reference.blocks(
                "head", block_rows=args.weight_rows, hash_payload=True
            ):
                expected[:, start : start + len(weight)] = x.astype(np.float64) @ weight.T
                budget.check()
            expected = bf16(expected)
            expected_loss, cotangent = ce_reference(expected, targets)
            observed_loss, observed_cotangent = ce_reference(actual, targets)
            # Autodiff of head(BF16).astype(F32) casts dlogits back to BF16.
            cotangent, observed_cotangent = bf16(cotangent), bf16(observed_cotangent)
            independent_dx, conditioned_dx = (
                np.zeros(x.shape, np.float64),
                np.zeros(x.shape, np.float64),
            )
            for start, weight in reference.blocks("head", block_rows=args.weight_rows):
                end = start + len(weight)
                independent_dx += cotangent[:, start:end].astype(np.float64) @ weight
                conditioned_dx += observed_cotangent[:, start:end].astype(np.float64) @ weight
                budget.check()
            forward = metrics(actual, expected)
            backward = metrics(dx, bf16(independent_dx))
            conditioned = metrics(dx, bf16(conditioned_dx))
            ce_error, ce_observed_error = (
                abs(loss_value - expected_loss),
                abs(loss_value - observed_loss),
            )
            ce_tolerance = max(2e-3, abs(expected_loss) * 2e-3)
            publish(
                {
                    "name": "head+cross_entropy",
                    "weight_shape": [head.out_features, head.in_features],
                    "input_shape": list(x.shape),
                    "input_dtype": "bfloat16",
                    "weight_format": "bfloat16",
                    "targets": targets.tolist(),
                    "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                    "forward": forward,
                    "input_vjp": backward,
                    "input_vjp_conditioned_on_observed_logits": conditioned,
                    "loss": loss_value,
                    "independent_loss": expected_loss,
                    "independent_loss_abs_error": ce_error,
                    "independent_loss_tolerance": ce_tolerance,
                    "observed_logits_ce_abs_error": ce_observed_error,
                    "observed_logits_ce_tolerance": max(2e-6, abs(observed_loss) * 2e-6),
                    "seconds": time.monotonic() - begin,
                    "passed": forward["passed"]
                    and backward["passed"]
                    and conditioned["passed"]
                    and ce_error <= ce_tolerance
                    and ce_observed_error <= max(2e-6, abs(observed_loss) * 2e-6),
                }
            )
            del head, loss, gradient, logits, input_array, weight
        report["complete"] = True
        report["passed"] = all(case["passed"] for case in report["cases"])
        report["saved_at_unix_seconds"] = time.time()
        atomic_report(output, report)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--wo-group", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument(
        "--oracle-rows",
        type=int,
        default=8,
        help="Fixed trunk sample count; all rows still execute on Metal",
    )
    parser.add_argument("--head-rows", type=int, default=2)
    parser.add_argument("--weight-rows", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument(
        "--projections",
        choices=("wq_a", "wq_b", "wkv", "wo_a", "wo_b"),
        nargs="*",
        default=["wq_a", "wq_b", "wkv", "wo_a", "wo_b"],
    )
    parser.add_argument("--skip-head", action="store_true")
    parser.add_argument("--memory-gib", type=float, default=12)
    parser.add_argument("--reserve-gib", type=float, default=32)
    args = parser.parse_args()
    if (
        not 1 <= args.rows <= 8192
        or not 1 <= args.oracle_rows <= 128
        or not 1 <= args.head_rows <= 32
        or not 1 <= args.weight_rows <= 2048
    ):
        parser.error("Use rows1..8192, oracle-rows1..128, head-rows1..32, weight-rows1..2048")
    if len(set(args.projections)) != len(args.projections) or (
        not args.projections and args.skip_head
    ):
        parser.error("Select unique projections or the head")
    if not all(math.isfinite(v) and v > 0 for v in (args.memory_gib, args.reserve_gib)):
        parser.error("Memory and reserve budgets must be finite and positive")
    raise SystemExit(0 if run(args) else 2)


if __name__ == "__main__":
    main()
