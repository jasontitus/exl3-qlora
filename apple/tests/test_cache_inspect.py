import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Staged review only; copied product tests import normally after application.
from exl3_mlx import cache_inspect as c
from exl3_mlx.prefix_cache import FrozenPrefixCache
from exl3_mlx.prefix_disk import FrozenPrefixDisk


def write(path, value):
    path.write_bytes(c.canonical(value))


class Fixture:
    def __init__(self, root):
        self.root = root
        (root / "data").mkdir()
        self.cache = root / "cache"
        self.cache.mkdir()
        self.rows = [
            {"input_ids": [1, 40, 2], "loss_start": 1},
            {"input_ids": [1, 41, 42, 2], "loss_start": 2},
        ]
        data = b"\n".join(c.canonical(r) for r in self.rows) + b"\n"
        (root / "data/train.jsonl").write_bytes(data)
        self.project = {
            "format": "exl3-sft-project-v1",
            "config": {
                "adapter_layers": [39],
                "activation_mode": "checkpoint",
                "chunk_size": 2048,
                "prefix_disk_cache": str(self.cache),
            },
            "data": {"files": {"train.jsonl": hashlib.sha256(data).hexdigest()}},
        }
        write(root / "project.json", self.project)
        self.binding = {
            "frozen_blocks": 39,
            "activation_mode": "checkpoint",
            "chunk_size": 2048,
            "implementation": {},
            "platform": {},
            "model_payloads": {},
            "state_layout": {},
        }
        write(
            self.cache / "binding.json", {"format": c.FORMAT, "binding": self.binding}
        )
        self.entries = []
        for row in self.rows:
            key, metadata = c.key_for_row(row, training=True, checkpoint_blocks=True)
            entry = self.cache / key
            entry.mkdir()
            payload = entry / "state.safetensors"
            payload.write_bytes(
                b"x" * 2048
            )  # Not a real tensor/header; metadata-only fixture.
            size = len(row["input_ids"]) - 1
            tensors = {}
            for name in c.LEAVES:
                shape = [1, size, 1, 2] if name == "h" else [1, size, 2]
                dtype = {"kv2": "int32", "kv3": "uint8"}.get(name, "bfloat16")
                width = {"int32": 4, "uint8": 1}.get(dtype, 2)
                tensors[name] = {
                    "shape": shape,
                    "dtype": "mlx.core." + dtype,
                    "bytes": int(np.prod(shape)) * width,
                }
            receipt = {
                "format": c.FORMAT,
                "binding_sha256": hashlib.sha256(c.canonical(self.binding)).hexdigest(),
                "key": metadata,
                "file_bytes": 2048,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                "tensors": tensors,
            }
            write(entry / "entry.json", receipt)
            self.entries.append(entry)


class Tests(unittest.TestCase):
    @contextlib.contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory() as td:
            yield Fixture(Path(td).resolve())

    def test_keys_match_real_cache_for_modes_and_widths(self):
        for training in (True, False):
            for checkpoint in (True, False):
                row = {"input_ids": [1, 258, 99999, 2], "loss_start": 2}
                expected = c.key_for_row(
                    row, training=training, checkpoint_blocks=checkpoint
                )
                for dtype in (np.int32, np.int64):
                    key = FrozenPrefixCache.key(
                        np.array([row["input_ids"][:-1]], dtype=dtype),
                        training=training,
                        checkpoint_blocks=checkpoint,
                    )
                    self.assertEqual(FrozenPrefixDisk._key(key), expected)

    def test_metadata_only_never_hashes_or_loads_payload(self):
        with (
            self.fixture() as f,
            patch.object(c, "hash_file", side_effect=AssertionError("payload read")),
        ):
            before = sorted(str(p.relative_to(f.root)) for p in f.root.rglob("*"))
            result = c.inspect_project(f.root)
            self.assertEqual(result["metadata_consistent_rows"], 2)
            self.assertEqual(result["unique_matching_payload_bytes"], 4096)
            self.assertFalse(result["compatibility_qualified"])
            self.assertEqual(result["payload_bytes_hashed"], 0)
            self.assertFalse(result["worker_implementation_verified"])
            self.assertFalse(result["inspector_is_worker"])
            self.assertEqual(
                result["inspector_source_sha256"],
                hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
            )
            self.assertEqual(
                before, sorted(str(p.relative_to(f.root)) for p in f.root.rglob("*"))
            )
            self.assertNotIn("mlx.core", sys.modules)

    def test_full_hash_opt_in_and_corruption(self):
        with self.fixture() as f:
            self.assertEqual(
                c.inspect_project(f.root, full_payload_checksums=True)[
                    "payload_bytes_hashed"
                ],
                4096,
            )
            (f.entries[0] / "state.safetensors").write_bytes(b"y" * 2048)
            self.assertEqual(c.inspect_project(f.root)["metadata_consistent_rows"], 2)
            result = c.inspect_project(f.root, full_payload_checksums=True)
            self.assertEqual(result["malformed_rows"], [0])
            self.assertFalse(result["compatibility_qualified"])

    def test_missing_wrong_modes_and_malformed_key(self):
        with self.fixture() as f:
            self.assertEqual(
                c.inspect_project(f.root, training=False)["missing_rows"], [0, 1]
            )
            receipt = json.loads((f.entries[0] / "entry.json").read_bytes())
            receipt["key"]["training"] = False
            write(f.entries[0] / "entry.json", receipt)
            self.assertEqual(c.inspect_project(f.root)["malformed_rows"], [0])

    def test_project_hash_and_row_validation(self):
        with self.fixture() as f:
            (f.root / "data/train.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "project manifest"):
                c.inspect_project(f.root)
        for row in (
            {"input_ids": [True, 2], "loss_start": 1},
            {"input_ids": [1, -1, 2], "loss_start": 1},
            {"input_ids": [1, 2], "loss_start": True},
            {"input_ids": [1], "loss_start": 1},
        ):
            with self.assertRaises(ValueError):
                c.key_for_row(row, training=True, checkpoint_blocks=True)

    def test_binding_mismatch_remains_unqualified(self):
        with self.fixture() as f:
            f.project["config"]["adapter_layers"] = [38]
            write(f.root / "project.json", f.project)
            expected = f.root / "expected.json"
            write(expected, {"format": c.FORMAT, "binding": {"different": True}})
            result = c.inspect_project(f.root, expected_binding=expected)
            self.assertFalse(result["project_binding_checks"]["frozen_boundary"])
            self.assertFalse(result["expected_binding_equal"])
            self.assertFalse(result["compatibility_qualified"])

    def test_symlink_fifo_surplus_and_bad_geometry(self):
        for kind in ("symlink", "fifo", "surplus", "geometry"):
            with self.subTest(kind=kind), self.fixture() as f:
                entry = f.entries[0]
                if kind in ("symlink", "fifo"):
                    p = entry / "entry.json"
                    p.unlink()
                    if kind == "fifo":
                        os.mkfifo(p)
                    else:
                        p.symlink_to(f.cache / "binding.json")
                elif kind == "surplus":
                    (entry / "extra").write_text("x")
                else:
                    value = json.loads((entry / "entry.json").read_bytes())
                    value["tensors"]["h"]["bytes"] = 1
                    write(entry / "entry.json", value)
                self.assertEqual(c.inspect_project(f.root)["malformed_rows"], [0])

    def test_metadata_and_payload_drift_refuse_publication(self):
        for kind in ("metadata", "payload"):
            with self.subTest(kind=kind), self.fixture() as f:
                original = c.hash_file

                def mutate(path, **kw):
                    result = original(path, **kw)
                    target = f.entries[0] / (
                        "entry.json" if kind == "metadata" else "state.safetensors"
                    )
                    raw = target.read_bytes()
                    target.unlink()
                    target.write_bytes(raw)
                    return result

                with (
                    patch.object(c, "hash_file", side_effect=mutate),
                    self.assertRaises((ValueError, RuntimeError)),
                ):
                    c.inspect_project(f.root, full_payload_checksums=True)

    def test_limits_and_duplicate_rows_count_unique_bytes(self):
        with (
            self.fixture() as f,
            patch.object(c, "MAX_DATA", 5),
            self.assertRaises(ValueError),
        ):
            c.inspect_project(f.root)
        with self.fixture() as f:
            data = c.canonical(f.rows[0]) + b"\n" + c.canonical(f.rows[0]) + b"\n"
            (f.root / "data/train.jsonl").write_bytes(data)
            f.project["data"]["files"]["train.jsonl"] = hashlib.sha256(data).hexdigest()
            write(f.root / "project.json", f.project)
            result = c.inspect_project(f.root)
            self.assertEqual(result["metadata_consistent_rows"], 2)
            self.assertEqual(result["unique_expected_keys"], 1)
            self.assertEqual(result["unique_matching_payload_bytes"], 2048)

    def test_duplicate_payload_mutation_after_checksum_refuses(self):
        with self.fixture() as f:
            data = c.canonical(f.rows[0]) + b"\n" + c.canonical(f.rows[0]) + b"\n"
            (f.root / "data/train.jsonl").write_bytes(data)
            f.project["data"]["files"]["train.jsonl"] = hashlib.sha256(data).hexdigest()
            write(f.root / "project.json", f.project)
            original = c.hash_file

            def corrupt(path, **kwargs):
                digest = original(path, **kwargs)
                Path(path).write_bytes(b"y" * 2048)
                return digest

            with (
                patch.object(c, "hash_file", side_effect=corrupt),
                self.assertRaises(ValueError),
            ):
                c.inspect_project(f.root, full_payload_checksums=True)

    def test_receipt_object_and_key_types(self):
        for change in ("object", "training", "shape"):
            with self.subTest(change=change), self.fixture() as f:
                p = f.entries[0] / "entry.json"
                value = json.loads(p.read_bytes())
                if change == "object":
                    value = []
                elif change == "training":
                    value["key"]["training"] = 1
                else:
                    value["key"]["shape"][0] = True
                write(p, value)
                self.assertEqual(c.inspect_project(f.root)["malformed_rows"], [0])

    def test_cli_summary_cpu_only(self):
        with self.fixture() as f, contextlib.redirect_stdout(io.StringIO()) as stream:
            c.main([str(f.root)])
            result = json.loads(stream.getvalue())
            self.assertNotIn("entries", result)
            self.assertFalse(result["device_verified"])
