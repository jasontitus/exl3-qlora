"""Small synthetic files only; no model payloads, network or GPU activity."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "expert_row_reader", ROOT / "tools/expert_row_reader.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ExpertRowReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.name = "model-expert.safetensors"
        self.prefix = "language_model.model.layers.1.mlp.switch_mlp.gate_proj"
        self.config = {
            "model_type": "qwen4_exp",
            "text_config": {
                "num_hidden_layers": 2,
                "num_experts": 3,
                "hidden_size": 128,
                "moe_intermediate_size": 256,
            },
            "expert_quant": {"format": "exl3", "codebook": "mcg", "window": 14, "k": 3},
        }
        self.expected = {i: {} for i in range(3)}
        self.header = {
            "__metadata__": {
                "format": "exl3",
                "codebook": "mcg",
                "window": "14",
                "k": "3",
            }
        }
        payload = bytearray()
        # Non-default on-disk order verifies planning rather than assumed order.
        for kind, shape, dtype in (
            ("svh", [3, 256], "F16"),
            ("trellis", [3, 8, 16, 48], "U16"),
            ("suh", [3, 128], "F16"),
        ):
            per_expert = 2
            for dim in shape[1:]:
                per_expert *= dim
            start = len(payload)
            for expert in range(3):
                raw = bytes(
                    (i * 7 + expert * 41 + len(kind)) % 256 for i in range(per_expert)
                )
                self.expected[expert][kind] = raw
                payload.extend(raw)
            self.header[self.prefix + "." + kind] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, len(payload)],
            }
        self.payload = bytes(payload)
        self.write_metadata()
        self.write_shard()

    def write_metadata(self):
        (self.root / "config.json").write_text(json.dumps(self.config))
        self.index = {
            "weight_map": {self.prefix + "." + kind: self.name for kind in m.KINDS}
        }
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))

    def write_shard(self):
        raw = json.dumps(self.header).encode()
        raw += b" " * (-len(raw) % 8)
        (self.root / self.name).write_bytes(
            struct.pack("<Q", len(raw)) + raw + self.payload
        )

    def bind(self, **kwargs):
        return m.bind_projection(
            self.root, shard_name=self.name, prefix=self.prefix, **kwargs
        )

    def reader(self, identity=None, **kwargs):
        identity = self.bind() if identity is None else identity
        return m.ExpertRowReader(
            self.root, identity, expected_sha256=m.identity_sha256(identity), **kwargs
        )

    def test_full_identity_and_first_last_expert_payloads(self):
        identity = self.bind(scan_chunk_bytes=31)
        self.assertEqual(
            identity["shard"]["file_sha256"],
            hashlib.sha256((self.root / self.name).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            identity["binding_scan_bytes"], (self.root / self.name).stat().st_size
        )
        reader = self.reader(identity)
        for expert in (0, 2):
            with self.subTest(expert=expert):
                result = reader.read(expert)
                self.assertEqual(result["payloads"], self.expected[expert])
                receipt = result["receipt"]
                self.assertEqual(
                    receipt["identity_sha256"], m.identity_sha256(identity)
                )
                self.assertEqual(receipt["payload_bytes_read"], 13056)
                self.assertTrue(receipt["selected_payload_checksums_verified"])
                self.assertFalse(receipt["full_file_rehashed_on_read"])
                self.assertFalse(receipt["decoded"])
                for kind, value in result["payloads"].items():
                    self.assertIsInstance(value, bytes)
                    self.assertEqual(
                        hashlib.sha256(value).hexdigest(),
                        receipt["views"][kind]["sha256"],
                    )

    def test_read_only_three_selected_ranges_not_entire_bank(self):
        identity = self.bind()
        reader = self.reader(identity)
        with mock.patch.object(m.os, "pread", wraps=os.pread) as read:
            result = reader.read(1)
        self.assertEqual(read.call_count, 3)
        actual = [(call.args[2], call.args[1]) for call in read.call_args_list]
        expected = [
            (
                identity["experts"][1]["views"][kind]["offset_bytes"],
                identity["experts"][1]["views"][kind]["length_bytes"],
            )
            for kind in m.KINDS
        ]
        self.assertEqual(actual, expected)
        self.assertLess(sum(n for _, n in actual), identity["shard"]["file_bytes"])
        self.assertEqual(result["payloads"], self.expected[1])

    def test_caller_identity_and_returned_receipt_cannot_mutate_reader(self):
        identity = self.bind()
        reader = self.reader(identity)
        identity["experts"][0]["views"]["suh"]["sha256"] = "0" * 64
        result = reader.read(0)
        result["receipt"]["views"]["suh"]["sha256"] = "0" * 64
        self.assertEqual(reader.read(0)["payloads"], self.expected[0])

    def test_invalid_expert_indices(self):
        reader = self.reader()
        for expert in (-1, 3, True, 1.0, "1"):
            with (
                self.subTest(expert=expert),
                self.assertRaisesRegex(ValueError, "bounds"),
            ):
                reader.read(expert)

    def test_digest_pin_and_geometry_tamper_rejected(self):
        original = self.bind()
        changed = copy.deepcopy(original)
        changed["experts"][0]["views"]["trellis"]["offset_bytes"] += 2
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            m.ExpertRowReader(
                self.root, changed, expected_sha256=m.identity_sha256(original)
            )
        with self.assertRaisesRegex(ValueError, "geometry differs"):
            self.reader(changed)
        changed = copy.deepcopy(original)
        changed["experts"][0]["views"]["suh"]["sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "Invalid SHA256"):
            self.reader(changed)

    def test_payload_mutation_rejected_by_stat_guard(self):
        reader = self.reader()
        path = self.root / self.name
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"!")
        with self.assertRaisesRegex(ValueError, "changed|replaced"):
            reader.read(0)

    def test_payload_hash_detects_change_even_if_stat_guards_miss_it(self):
        identity = self.bind()
        reader = self.reader(identity)
        path = self.root / self.name
        offset = identity["experts"][1]["views"]["trellis"]["offset_bytes"]
        with path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(b"!")
        # Simulate an ineffective stat layer; selected byte hash remains decisive.
        with (
            mock.patch.object(m, "_unchanged"),
            mock.patch.object(
                m, "_open_bound", side_effect=lambda *_: os.open(path, os.O_RDONLY)
            ),
            self.assertRaisesRegex(ValueError, "payload checksum mismatch"),
        ):
            reader.read(1)

    def test_truncated_and_replaced_shards_rejected(self):
        identity = self.bind()
        path = self.root / self.name
        old = path.read_bytes()
        path.write_bytes(old[:-1])
        with self.assertRaises(ValueError):
            self.reader(identity)
        path.write_bytes(old)
        with self.assertRaisesRegex(ValueError, "header/stat differs"):
            self.reader(identity)

    def test_mid_read_truncation_or_mutation_fails_closed(self):
        for after_last in (False, True):
            with self.subTest(after_last=after_last):
                self.write_shard()
                reader = self.reader()
                original = os.pread
                count = 0

                def changing_read(
                    fd, size, offset, original=original, after_last=after_last
                ):
                    nonlocal count
                    value = original(fd, size, offset)
                    count += 1
                    if count == (3 if after_last else 1):
                        with (self.root / self.name).open("r+b") as stream:
                            stream.truncate(8)
                    return value

                with (
                    mock.patch.object(m.os, "pread", side_effect=changing_read),
                    self.assertRaisesRegex(ValueError, "Truncated|changed"),
                ):
                    reader.read(0)

    def test_metadata_mutation_after_bind_and_after_open_rejected(self):
        identity = self.bind()
        for filename in ("config.json", "model.safetensors.index.json"):
            with self.subTest(filename=filename):
                reader = self.reader(identity)
                path = self.root / filename
                raw = path.read_bytes()
                path.write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    reader.read(0)
                with self.assertRaises(ValueError):
                    self.reader(identity)
                path.write_bytes(raw)
                identity = self.bind()

    def test_wrong_index_owner_rejected(self):
        self.index["weight_map"][self.prefix + ".suh"] = "another.safetensors"
        (self.root / "model.safetensors.index.json").write_text(json.dumps(self.index))
        with self.assertRaisesRegex(ValueError, "ownership mismatch"):
            self.bind()

    def test_path_escape_and_symlink_refused(self):
        for name in (
            "../outside.safetensors",
            "/outside.safetensors",
            "nested/x.safetensors",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "basenames"),
            ):
                m.bind_projection(self.root, shard_name=name, prefix=self.prefix)
        path = self.root / self.name
        target = path.with_suffix(".bin")
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            self.bind()

    def test_config_or_index_symlink_rejected(self):
        for filename in ("config.json", "model.safetensors.index.json"):
            with self.subTest(filename=filename):
                path = self.root / filename
                target = path.with_suffix(".copy")
                path.rename(target)
                path.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "non-symlink"):
                    self.bind()
                path.unlink()
                target.rename(path)

    def test_invalid_bank_geometry_and_stamp_rejected(self):
        original = copy.deepcopy(self.header)
        for field in ("dtype", "shape", "rate", "window", "offsets"):
            with self.subTest(field=field):
                self.header = copy.deepcopy(original)
                entry = self.header[self.prefix + ".trellis"]
                if field == "dtype":
                    entry["dtype"] = "I16"
                elif field == "shape":
                    entry["shape"] = [3, 16, 8, 48]
                elif field in ("rate", "window"):
                    self.header["__metadata__"][
                        "k" if field == "rate" else "window"
                    ] = "4" if field == "rate" else "16"
                else:
                    entry["data_offsets"][0] += 2
                self.write_shard()
                with self.assertRaises(ValueError):
                    self.bind()

    def test_binding_and_read_byte_caps_fail_before_payload_scan(self):
        for kwargs in (
            {"max_shard_bytes": 8},
            {"max_header_bytes": 8},
            {"max_expert_bytes": 13055},
            {"scan_chunk_bytes": 0},
            {"scan_chunk_bytes": (1 << 20) + 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.bind(**kwargs)
        identity = self.bind()
        with self.assertRaisesRegex(ValueError, "bounded read limit"):
            self.reader(identity, max_expert_bytes=13055)

    def test_identity_loader_and_no_overwrite_cli(self):
        path = self.root / "identity.json"
        args = [
            "bind",
            str(self.root),
            "--shard",
            self.name,
            "--prefix",
            self.prefix,
            "--output",
            str(path),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(args), 0)
        raw = path.read_bytes()
        identity = json.loads(raw)
        pin = m.identity_sha256(identity)
        self.assertEqual(m.load_identity(path, expected_sha256=pin), identity)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(m.main(args), 2)
        self.assertEqual(path.read_bytes(), raw)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(
                m.main(
                    [
                        "read",
                        str(self.root),
                        "--identity",
                        str(path),
                        "--identity-sha256",
                        pin,
                        "--expert",
                        "2",
                    ]
                ),
                0,
            )
        self.assertEqual(json.loads(out.getvalue())["payload_bytes_read"], 13056)
        identity["prefix"] += ".changed"
        path.write_text(json.dumps(identity))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            m.load_identity(path, expected_sha256=pin)

    def test_identity_source_and_directory_binding(self):
        identity = self.bind()
        for field, value in (
            ("implementation_sha256", {}),
            ("checkpoint_directory", "/elsewhere"),
        ):
            changed = copy.deepcopy(identity)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.reader(changed)
        with (
            mock.patch.object(m, "implementation_identity", return_value={}),
            self.assertRaisesRegex(ValueError, "changed after import"),
        ):
            self.bind()


if __name__ == "__main__":
    unittest.main()
