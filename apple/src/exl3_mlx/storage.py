"""Bounded reads of the local mtplx EXL3 expert repack and native MLX trunk."""

import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoint import Checkpoint

DTYPES = {
    "I16": "<i2",
    "F16": "<f2",
    "F32": "<f4",
    "BF16": "<u2",
    "U32": "<u4",
    "U8": "u1",
    "I32": "<i4",
}


def relative_file(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest file must remain inside checkpoint directory")
    return path


def read_range(path, offset, size):
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated checkpoint: {path}")
    return data


class TensorArchive:
    """Read one tensor (or contiguous row range), including BF16, without loading a shard."""

    def __init__(self, path):
        path = Path(path).expanduser()
        files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
        self.tensors = {}
        self.fingerprints = {}
        for file in files:
            stat = file.stat()
            self.fingerprints[file] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            length = int.from_bytes(read_range(file, 0, 8), "little")
            if length > 16 << 20 or length + 8 > stat.st_size:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(read_range(file, 8, length))
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                if key in self.tensors:
                    raise ValueError(f"Duplicate tensor: {key}")
                lo, hi = spec["data_offsets"]
                if lo < 0 or hi < lo or hi + 8 + length > stat.st_size:
                    raise ValueError(f"Invalid tensor offsets: {key}")
                self.tensors[key] = (file, 8 + length + lo, hi - lo, spec)
        if not self.tensors:
            raise ValueError("No safetensors found")

    def array(self, key, rows=None):
        file, offset, size, spec = self.tensors[key]
        stat = file.stat()
        if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != self.fingerprints[file]:
            raise RuntimeError(f"Frozen checkpoint changed: {file}")
        shape, dtype = tuple(spec["shape"]), spec["dtype"]
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported native tensor dtype: {dtype}")
        item = np.dtype(DTYPES[dtype])
        if math.prod(shape) * item.itemsize != size:
            raise ValueError(f"Invalid tensor size: {key}")
        if rows is not None:
            start, end = rows
            if not shape or not 0 <= start <= end <= shape[0]:
                raise ValueError("Invalid row range")
            stride = math.prod(shape[1:]) * item.itemsize
            offset, size = offset + start * stride, (end - start) * stride
            shape = (end - start, *shape[1:])
        data = np.frombuffer(read_range(file, offset, size), dtype=item).reshape(shape)
        result = mx.array(data)
        return result.view(mx.bfloat16) if dtype == "BF16" else result


class ExpertBank(Checkpoint):
    """Expose a validated mtplx-expert-manifest-v2 as individual EXL3 linears.

    Reads only the selected matrix. No 190-GiB mapping, conversion, or duplication.
    The compact layer layouts must agree with every record descriptor.
    """

    def __init__(self, root):
        self.path = Path(root).expanduser().resolve()
        manifest_path = self.path / "expert-manifest-v2.json"
        raw = manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        m = json.loads(raw)
        q = m["quantization"]
        if (
            m["format"] != "mtplx-expert-manifest-v2"
            or q["mode"] != "exl3"
            or q["codebook"] != "mul1"
            or q["codebook_multiplier"] != 0x83DCD12D
            or q["tile"]
            != {"size": 16, "layout": "exl3-tensor-core", "bitstream": "msb-first-swap16"}
            or q["hadamard"] != {"size": 128, "order": "sylvester-natural", "scale": "1/sqrt(128)"}
            or q["scales"] != {"rin": "exl3.suh", "rout": "exl3.svh"}
        ):
            raise ValueError("Unsupported EXL3 expert bank layout")
        file = relative_file(self.path, m["sidecar"]["file"])
        self.total_bytes = file.stat().st_size
        if self.total_bytes != m["sidecar"]["size"]:
            raise ValueError("Expert bank size differs from manifest")
        self.dims, self.source = m["dims"], q["source"]
        self.tensors, self._ranges = {}, {}
        records = {(r["layer"], r["expert"]): r for r in m["records"]}
        if len(records) != self.dims["n_layers"] * self.dims["n_experts"]:
            raise ValueError("Incomplete or duplicate expert records")
        self.records = records
        self.file = file
        if sorted(layer["layer"] for layer in m["layers"]) != list(range(self.dims["n_layers"])):
            raise ValueError("Invalid expert layer coverage")
        names = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}
        suffixes = {"code": "trellis", "rin": "suh", "rout": "svh"}
        expected = {f"{p}.{s}" for p in names for s in suffixes}
        for layer in m["layers"]:
            if {s["component"] for s in layer["segments"]} != expected or len(
                layer["segments"]
            ) != 9:
                raise ValueError("Invalid expert segments")
            for expert in range(self.dims["n_experts"]):
                base = layer["base_offset"] + expert * layer["record_bytes"]
                record = records[layer["layer"], expert]
                if (
                    record["sidecar_offset"] != base
                    or record["record_bytes"] != layer["record_bytes"]
                    or record["logical_bytes"] != layer["logical_bytes"]
                    or base < 0
                    or base + layer["record_bytes"] > self.total_bytes
                ):
                    raise ValueError("Expert record bounds do not match layout")
                for seg in layer["segments"]:
                    proj, suffix = seg["component"].split(".")
                    key = f"layers.{layer['layer']}.ffn.experts.{expert}.{names[proj]}.{suffixes[suffix]}"
                    shape, dtype = seg["shape"], seg["dtype"]
                    k, n = self.dims["hidden"], self.dims["inter"]
                    if proj == "down_proj":
                        k, n = n, k
                    wanted = (
                        [k // 16, n // 16, layer["K"] * 16]
                        if suffix == "code"
                        else [k if suffix == "rin" else n]
                    )
                    if shape != wanted or dtype != ("I16" if suffix == "code" else "F16"):
                        raise ValueError("Invalid expert tensor shape or dtype")
                    size = math.prod(shape) * 2
                    if (
                        seg["length"] != size
                        or seg["offset"] < 0
                        or seg["offset"] + size > layer["logical_bytes"]
                    ):
                        raise ValueError("Invalid expert segment bounds")
                    self.tensors[key] = file, shape, dtype
                    self._ranges[key] = base + seg["offset"], size

    def codebook(self, key):
        return "mul1"

    def array(self, key):
        file, shape, dtype = self.tensors[key]
        offset, size = self._ranges[key]
        return np.frombuffer(read_range(file, offset, size), DTYPES[dtype]).reshape(shape)

    def verify_record(self, layer, expert):
        record = self.records[layer, expert]
        raw = read_range(self.file, record["sidecar_offset"], record["record_bytes"])
        actual = hashlib.sha256(raw).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Expert checksum mismatch at {layer}/{expert}")
        return actual
