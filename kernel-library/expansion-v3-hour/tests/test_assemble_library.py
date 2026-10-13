import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "assemble_library", ROOT / "tools/assemble_library.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import pack_exact
from exact_edit_renderer import render_command, unified_patch


def blob(text):
    import hashlib

    raw = text.encode()
    return {
        "content": text,
        "sha256": m.sha(raw),
        "blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
        "mode": "100644",
    }


class AssemblyBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.lib = self.root / "kernel-library"
        self.lib.mkdir()
        for mod, attr in ((m, "W"), (pack_exact, "WORK")):
            p = patch.object(mod, attr, self.root)
            p.start()
            self.addCleanup(p.stop)
        self.base = self.lib / "base"
        self.base.mkdir()
        (self.base / "test.jsonl").write_text("frozen test\n")
        self.manifest = {
            "artifacts_sha256": {
                "test.jsonl": m.sha((self.base / "test.jsonl").read_bytes())
            }
        }
        self.pin_base()
        self.src = {
            "files": {
                "kernel.metal": {
                    "before": blob("old\n"),
                    "after": blob("new\n"),
                    "role": "changed",
                }
            },
            "outcome": "source_only",
        }
        patchtext = unified_patch(
            {"kernel.metal": b"old\n"}, {"kernel.metal": b"new\n"}
        )
        self.raw = {
            "tools": [],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "exec_command",
                                "arguments": {"cmd": render_command(patchtext)},
                            }
                        }
                    ],
                }
            ],
        }
        self.prepared = {"input_ids": [1, 2, 3], "loss_start": 1}
        self.replay = {"patch_sha256": m.sha(patchtext.encode())}
        self.entry = {
            "id": "case",
            "paths": {},
            "curriculum_tier": "correctness",
            "review": {"status": "approved"},
        }
        for name, value in [
            ("source", self.src),
            ("raw", self.raw),
            ("prepared", self.prepared),
            ("replay", self.replay),
        ]:
            p = self.lib / (name + ".json")
            p.write_text(json.dumps(value))
            self.entry["paths"][name] = str(p)
        review = {
            "schema": "exact-edit-independent-review-v1",
            "status": "approved",
            "reviewer": "peer",
            "cases": [
                {
                    "id": "case",
                    "inputs_sha256": {
                        p: m.sha(Path(p).read_bytes())
                        for p in self.entry["paths"].values()
                    },
                }
            ],
        }
        rp = self.lib / "review.json"
        rp.write_text(json.dumps(review))
        self.entry["review"].update(receipt=str(rp), sha256=m.sha(rp.read_bytes()))
        self.d = {
            "id": "case",
            "source": self.src,
            "review": self.entry["review"],
            "curriculum_tier": "correctness",
            "replay": m.verify_edit(self.src, self.raw, self.replay),
        }

    def pin_base(self):
        p = self.base / "ktune-import.json"
        p.write_text(json.dumps(self.manifest))
        self.patch_pin = patch.object(
            m, "BASE_RELEASES", {12288: m.sha(p.read_bytes())}
        )
        self.patch_pin.start()
        self.addCleanup(self.patch_pin.stop)

    def check(self, d=None, raw=None, prepared=None, entries=None):
        return m.verify_new_row(
            d or self.d,
            raw or self.raw,
            prepared or self.prepared,
            entries if entries is not None else {"case": self.entry},
            {},
        )

    def test_pinned_release_accepts(self):
        self.assertEqual(m.verify_base(self.base, 12288, {}), self.manifest)

    def test_changed_base_manifest_rejected(self):
        (self.base / "ktune-import.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "release identity"):
            m.verify_base(self.base, 12288, {})

    def test_changed_frozen_test_rejected(self):
        (self.base / "test.jsonl").write_text("still sixteen but different")
        with self.assertRaisesRegex(ValueError, "artifact checksum"):
            m.verify_base(self.base, 12288, {})

    def test_base_path_escape_rejected(self):
        p = self.lib / "escape"
        p.write_text("x")
        self.manifest["artifacts_sha256"] = {"../escape": m.sha(b"x")}
        self.pin_base()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            m.verify_base(self.base, 12288, {})

    def test_signed_exact_row_replayed(self):
        self.check()

    def test_prompt_tamper_even_same_patch_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["messages"].insert(0, {"role": "user", "content": "different task"})
        with self.assertRaisesRegex(ValueError, "reviewed payload"):
            self.check(raw=raw)

    def test_source_provenance_tamper_rejected(self):
        d = copy.deepcopy(self.d)
        d["source"]["outcome"] = "reported_improvement"
        with self.assertRaisesRegex(ValueError, "reviewed payload"):
            self.check(d=d)

    def test_token_ids_not_merely_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed payload"):
            self.check(prepared={"input_ids": [4, 5, 6], "loss_start": 1})

    def test_loss_mask_not_merely_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "reviewed payload"):
            self.check(prepared={"input_ids": [1, 2, 3], "loss_start": 2})

    def test_missing_index_authorization_rejected(self):
        with self.assertRaisesRegex(ValueError, "signed index"):
            self.check(entries={})

    def test_tier_laundering_rejected(self):
        d = copy.deepcopy(self.d)
        d["curriculum_tier"] = "optimization"
        with self.assertRaisesRegex(ValueError, "tier mismatch"):
            self.check(d=d)

    def test_forged_replay_rejected(self):
        d = copy.deepcopy(self.d)
        d["replay"]["exact_scoped_replay"] = False
        with self.assertRaisesRegex(ValueError, "replay receipt"):
            self.check(d=d)

    def test_signed_input_mutation_rejected(self):
        Path(self.entry["paths"]["source"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.check()

    def test_addition_original_inputs_drift_rejected(self):
        p = Path(self.entry["paths"]["raw"])
        with self.assertRaisesRegex(ValueError, "input drift"):
            m.addition_entries({"inputs_sha256": {str(p): "0" * 64}}, {})

    def test_addition_conflicting_existing_binding_rejected(self):
        p = Path(self.entry["paths"]["raw"])
        with self.assertRaisesRegex(ValueError, "input drift"):
            m.addition_entries(
                {"inputs_sha256": {str(p): m.sha(p.read_bytes())}}, {str(p): "0" * 64}
            )

    def test_addition_index_discovery(self):
        p = self.lib / "index.json"
        p.write_text(json.dumps({"cases": [self.entry]}))
        entries = m.addition_entries(
            {"inputs_sha256": {str(p): m.sha(p.read_bytes())}}, {}
        )
        self.assertEqual(entries, {"case": self.entry})

    def test_addition_ambiguous_authorization_rejected(self):
        a = self.lib / "a.json"
        a.write_text(json.dumps({"cases": [self.entry]}))
        entry = copy.deepcopy(self.entry)
        entry["curriculum_tier"] = "optimization"
        b = self.lib / "b.json"
        b.write_text(json.dumps({"cases": [entry]}))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            m.addition_entries(
                {"inputs_sha256": {str(p): m.sha(p.read_bytes()) for p in (a, b)}}, {}
            )


if __name__ == "__main__":
    unittest.main()


class ExactHistoricalTokenTests(unittest.TestCase):
    def test_same_ids_and_mask_accept(self):
        m.require_exact_prepared(
            {"input_ids": [1, 2, 3], "loss_start": 1},
            {"input_ids": [1, 2, 3], "loss_start": 1},
        )

    def test_equal_length_changed_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "token IDs"):
            m.require_exact_prepared(
                {"input_ids": [1, 9, 3], "loss_start": 1},
                {"input_ids": [1, 2, 3], "loss_start": 1},
            )

    def test_same_ids_different_mask_rejected(self):
        with self.assertRaisesRegex(ValueError, "loss mask"):
            m.require_exact_prepared(
                {"input_ids": [1, 2, 3], "loss_start": 2},
                {"input_ids": [1, 2, 3], "loss_start": 1},
            )

    def test_original_serializer_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.py"
            p.write_text("changed")
            with self.assertRaisesRegex(ValueError, "serializer implementation drift"):
                m.frozen_test_reference(
                    {"helper_source_sha256": {str(p): "0" * 64}}, {}
                )

    def test_historical_prepared_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.jsonl"
            p.write_text("{}\n")
            with (
                patch.object(m, "FROZEN_TEST_PREPARED", p),
                self.assertRaisesRegex(ValueError, "prepared identity"),
            ):
                m.frozen_test_reference({"helper_source_sha256": {}}, {})
