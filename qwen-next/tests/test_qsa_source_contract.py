"""Ensure the reference's recorded local source contract remains reviewable."""

import hashlib
import json
import unittest
from pathlib import Path


class SourceContract(unittest.TestCase):
    def test_pinned_source_and_reference_files(self):
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / "qwen-next/reference-contracts/qsa-cpu-v2.json").read_text())
        for path, entry in manifest["sources"].items():
            with self.subTest(source=path):
                raw = (root / path).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
                for span in entry.get("ranges", []):
                    text = b"".join(raw.splitlines(keepends=True)[span["first_line"] - 1:span["last_line"]])
                    self.assertEqual(hashlib.sha256(text).hexdigest(), span["sha256"])
        for path, expected in manifest["reference_files_sha256"].items():
            with self.subTest(reference=path):
                self.assertEqual(hashlib.sha256((root / path).read_bytes()).hexdigest(), expected)
        self.assertFalse(manifest["model_or_training_supported"])
        self.assertFalse(manifest["native_arithmetic_parity"])


if __name__ == "__main__":
    unittest.main()
