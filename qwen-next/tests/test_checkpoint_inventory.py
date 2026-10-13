"""Small CPU-only malformed-container fixtures; no model or optional imports."""

import contextlib
import importlib.util
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "checkpoint_inventory.py"
spec = importlib.util.spec_from_file_location("checkpoint_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


def tensor(shape=None, offsets=None, dtype="F32"):
    return {
        "dtype": dtype,
        "shape": [2] if shape is None else shape,
        "data_offsets": [0, 8] if offsets is None else offsets,
    }


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text('{"model_type":"fixture"}')

    def shard(self, header=None, payload=b"\0" * 8, name="model.safetensors", raw=None):
        if raw is None:
            raw = json.dumps(
                {"model.layers.0.weight": tensor()} if header is None else header
            ).encode()
        path = self.root / name
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        return path

    def index(self, mapping=None, size=8):
        path = self.root / "model.safetensors.index.json"
        path.write_text(
            json.dumps(
                {
                    "weight_map": mapping
                    or {"model.layers.0.weight": "model.safetensors"},
                    "metadata": {"total_size": size},
                }
            )
        )
        return path

    def test_valid_indexed_inventory_counts_only_tensor_payload(self):
        path = self.shard()
        self.index()
        report = inventory.inventory(self.root)
        self.assertTrue(report["index_ownership_verified"])
        self.assertEqual(report["physical_tensor_bytes"], 8)
        self.assertEqual(
            report["categories"]["trunk"], {"tensor_count": 1, "physical_bytes": 8}
        )
        self.assertEqual(report["physical_safetensors_file_bytes"], path.stat().st_size)
        self.assertEqual(len(report["files"][0]["header_sha256"]), 64)
        self.assertFalse(report["weight_payloads_read"])
        self.assertFalse(report["payload_integrity_verified"])
        self.assertFalse(report["supported_training"])
        self.assertFalse(report["whole_checkpoint_complete"])

    def test_headers_only_read_exactly_eight_plus_declared_header(self):
        path = self.shard()
        with mock.patch.object(
            inventory, "read_prefix", wraps=inventory.read_prefix
        ) as read:
            report = inventory.inspect_shard(path)
        self.assertEqual(
            [
                (c.args[1], c.args[2] if len(c.args) > 2 else 0)
                for c in read.call_args_list
            ],
            [(8, 0), (report["header_bytes"], 8)],
        )

    def test_multi_file_categories_and_component_precedence(self):
        labels = {
            "language_model.model.layers.1.ple.key_proj.weight": "PLE",
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis": "expert",
            "language_model.model.layers.0.mlp.experts.0.weight": "expert",
            "language_model.lm_head.weight": "head",
            "visual.blocks.0.weight": "vision",
            "language_model.mtp.layers.0.mlp.switch_mlp.weight": "MTP",
            "mystery.weight": "unknown",
        }
        for i, name in enumerate(labels):
            self.shard({name: tensor()}, name=f"part-{i}.safetensors")
        report = inventory.inventory(self.root)
        self.assertEqual(report["physical_tensor_bytes"], 8 * len(labels))
        self.assertEqual(report["categories"]["expert"]["physical_bytes"], 16)
        self.assertEqual(
            {
                t["name"]: t["category_heuristic"]
                for f in report["files"]
                for t in f["tensors"]
            },
            labels,
        )
        self.assertFalse(report["index_ownership_verified"])

    def test_external_ple_is_declared_not_inventoried_even_when_present(self):
        self.shard()
        (self.root / "config.json").write_text(
            json.dumps({"ngram_table": {"file": "ngram_table.bin", "bits": 4}})
        )
        report = inventory.inventory(self.root)
        self.assertEqual(
            report["external_payloads"][0]["status"], "declared_not_inventoried"
        )
        self.assertFalse(report["external_payloads"][0]["present"])
        (self.root / "ngram_table.bin").write_bytes(b"external bytes")
        report = inventory.inventory(self.root)
        self.assertEqual(report["external_payloads"][0]["stat"]["size_bytes"], 14)
        self.assertEqual(report["physical_tensor_bytes"], 8)
        self.assertFalse(report["external_payloads"][0]["payload_read"])
        self.assertIsNone(report["external_payloads"][0]["payload_sha256"])

    def test_scalar_and_empty_tensor_shapes(self):
        self.shard(
            {"scalar": tensor([], [0, 4]), "empty": tensor([0, 2], [4, 4])},
            payload=b"\0" * 4,
        )
        report = inventory.inventory(self.root)
        self.assertEqual(report["tensor_count"], 2)
        self.assertEqual(report["physical_tensor_bytes"], 4)

    def test_truncated_length_prefix(self):
        path = self.root / "model.safetensors"
        path.write_bytes(b"\0" * 7)
        with self.assertRaisesRegex(inventory.InventoryError, "Truncated"):
            inventory.inventory(self.root)

    def test_truncated_header(self):
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", 1024) + b"{}")
        with self.assertRaisesRegex(inventory.InventoryError, "truncated header"):
            inventory.inventory(self.root)

    def test_header_limit_rejects_before_reading_header(self):
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", 2**63))
        with (
            mock.patch.object(
                inventory, "read_prefix", wraps=inventory.read_prefix
            ) as read,
            self.assertRaisesRegex(inventory.InventoryError, "header length"),
        ):
            inventory.inspect_shard(path)
        self.assertEqual(read.call_count, 1)

    def test_zero_header_and_negative_limits_rejected(self):
        (self.root / "model.safetensors").write_bytes(struct.pack("<Q", 0))
        for kwargs in ({}, {"max_header_bytes": -1}, {"max_metadata_bytes": 0}):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises(inventory.InventoryError),
            ):
                inventory.inventory(self.root, **kwargs)

    def test_invalid_header_json_and_duplicate_keys(self):
        for raw in (b"[]", b"{", b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.subTest(raw=raw):
                self.shard(raw=raw)
                with self.assertRaises(inventory.InventoryError):
                    inventory.inventory(self.root)

    def test_invalid_metadata_values(self):
        self.shard({"__metadata__": {"k": 4}, "model.w": tensor()})
        with self.assertRaisesRegex(inventory.InventoryError, "metadata values"):
            inventory.inventory(self.root)

    def test_shape_dtype_and_offsets_validation(self):
        invalid = [
            tensor([-1]),
            tensor([True]),
            tensor([1.0]),
            tensor([2**64]),
            tensor([1] * 33),
            tensor([999999999]),
            tensor(dtype="UNKNOWN"),
            tensor(dtype=[]),
            tensor(offsets=[False, 8]),
            tensor(offsets=[0.0, 8]),
            tensor(offsets=[0]),
            tensor(offsets=[8, 0]),
            tensor(offsets=[-1, 7]),
            tensor(offsets=[0, 9]),
            tensor(offsets=[0, 4]),
        ]
        for item in invalid:
            with self.subTest(item=item):
                self.shard({"model.w": item})
                with self.assertRaises(inventory.InventoryError):
                    inventory.inventory(self.root)

    def test_overlapping_tensor_ranges(self):
        self.shard({"a": tensor(), "b": tensor()}, payload=b"\0" * 8)
        with self.assertRaisesRegex(inventory.InventoryError, "overlapping"):
            inventory.inventory(self.root)

    def test_payload_gap_and_trailing_garbage_rejected(self):
        for offsets in ([4, 12], [0, 8]):
            with self.subTest(offsets=offsets):
                self.shard({"a": tensor(offsets=offsets)}, payload=b"\0" * 12)
                with self.assertRaisesRegex(inventory.InventoryError, "gap|trailing"):
                    inventory.inventory(self.root)

    def test_truncated_payload(self):
        self.shard(payload=b"\0" * 7)
        with self.assertRaisesRegex(inventory.InventoryError, "bounds"):
            inventory.inventory(self.root)

    def test_duplicate_names_across_shards_rejected(self):
        self.shard()
        self.shard(name="extra.safetensors")
        self.index()
        with self.assertRaisesRegex(
            inventory.InventoryError, "Duplicate physical tensor"
        ):
            inventory.inventory(self.root)

    def test_missing_index_owner(self):
        self.shard()
        self.index({"model.layers.0.weight": "missing.safetensors"})
        with self.assertRaisesRegex(inventory.InventoryError, "owner shard missing"):
            inventory.inventory(self.root)

    def test_index_ownership_mismatch(self):
        self.shard()
        self.shard({"other": tensor()}, name="other.safetensors")
        self.index(
            {
                "model.layers.0.weight": "other.safetensors",
                "other": "model.safetensors",
            },
            size=16,
        )
        with self.assertRaisesRegex(inventory.InventoryError, "ownership mismatch"):
            inventory.inventory(self.root)

    def test_index_omitted_extra_tensor(self):
        self.shard()
        self.shard({"extra": tensor()}, name="extra.safetensors")
        self.index()
        with self.assertRaisesRegex(inventory.InventoryError, "ownership mismatch"):
            inventory.inventory(self.root)

    def test_index_declared_total_size_must_match(self):
        self.shard()
        self.index(size=99)
        with self.assertRaisesRegex(inventory.InventoryError, "total_size"):
            inventory.inventory(self.root)

    def test_unsafe_owner_or_external_path(self):
        self.shard()
        for owner in ("../outside", "/outside", "nested/shard.safetensors", "a\\b"):
            with self.subTest(owner=owner):
                self.index({"model.layers.0.weight": owner})
                with self.assertRaisesRegex(inventory.InventoryError, "basenames"):
                    inventory.inventory(self.root)
        (self.root / "model.safetensors.index.json").unlink()
        (self.root / "config.json").write_text('{"ngram_table":{"file":"../outside"}}')
        with self.assertRaisesRegex(inventory.InventoryError, "basenames"):
            inventory.inventory(self.root)

    def test_symlink_shard_and_external_refused(self):
        target = self.shard(name="actual.bin")
        (self.root / "model.safetensors").symlink_to(target)
        with self.assertRaisesRegex(inventory.InventoryError, "regular non-symlink"):
            inventory.inventory(self.root)
        (self.root / "model.safetensors").unlink()
        self.shard()
        (self.root / "ngram_table.bin").symlink_to(target)
        (self.root / "config.json").write_text(
            '{"ngram_table":{"file":"ngram_table.bin"}}'
        )
        with self.assertRaisesRegex(inventory.InventoryError, "regular non-symlink"):
            inventory.inventory(self.root)

    def test_metadata_size_bound(self):
        self.shard()
        with self.assertRaisesRegex(inventory.InventoryError, "Metadata exceeds"):
            inventory.inventory(self.root, max_metadata_bytes=4)

    def test_metadata_only_checkpoint_is_not_inventory(self):
        self.index()
        with self.assertRaisesRegex(inventory.InventoryError, "metadata alone"):
            inventory.inventory(self.root)

    def test_file_mutation_between_reads_rejected(self):
        path = self.shard()
        original = inventory.read_prefix

        def mutate(p, count, offset=0):
            result = original(p, count, offset)
            if offset == 0:
                with p.open("ab") as stream:
                    stream.write(b"x")
            return result

        with (
            mock.patch.object(inventory, "read_prefix", side_effect=mutate),
            self.assertRaisesRegex(inventory.InventoryError, "changed between"),
        ):
            inventory.inspect_shard(path)

    def test_cli_exclusive_output_and_malformed_nonzero(self):
        self.shard()
        output = self.root / "inventory.json"
        self.assertEqual(inventory.main([str(self.root), "--output", str(output)]), 0)
        before = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                inventory.main([str(self.root), "--output", str(output)]), 2
            )
        self.assertEqual(output.read_bytes(), before)
        self.shard(payload=b"x")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(inventory.main([str(self.root)]), 2)


if __name__ == "__main__":
    unittest.main()
