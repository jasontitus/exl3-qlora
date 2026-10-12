"""CPU-only checks: import the verifier without loading the MLX runtime."""

import copy
import json
import runpy
import tempfile
import unittest
from pathlib import Path

V = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_workflow.py"))


class VerificationProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "project"
        self.model = Path(self.temporary.name) / "model"
        self.output = Path(self.temporary.name) / "run"
        self.output.mkdir()
        self.runtime = {"python": "fixture", "packages": [["mlx", "fixture"]]}
        for name in (
            "src/exl3_mlx/__init__.py",
            "src/exl3_mlx/training.py",
            "scripts/verify_workflow.py",
            "pyproject.toml",
            "uv.lock",
            "fixture.jsonl",
        ):
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        for name in V["MODEL_METADATA"]:
            path = self.model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.fixture = self.project / "fixture.jsonl"
        self.current = self.report()
        (self.output / "base-eval.stdout.json").write_text('{"loss": 1.0}')
        (self.output / "base-eval.stderr.log").write_text("completed")
        self.old = copy.deepcopy(self.current)
        self.old["phases"]["base-eval"] = {
            "exit_code": 0,
            "evidence": V["phase_evidence"](self.output, "base-eval"),
        }

    def report(self):
        return {
            "model": str(self.model),
            "phases": {},
            "fixture_sha256": V["sha256"](self.fixture),
            "provenance": V["provenance"](
                self.project, self.model, self.fixture, runtime=self.runtime
            ),
        }

    def reject(self, current=None):
        with self.assertRaises((ValueError, OSError)):
            V["validate_continuation"](self.old, current or self.current, self.output)

    def test_unchanged_continuation_and_legacy_rejection(self):
        V["validate_continuation"](self.old, self.current, self.output)
        del self.old["provenance"]
        self.reject()

    def test_changed_source_metadata_template_fixture_and_runtime_rejected(self):
        paths = [
            self.project / "src/exl3_mlx/training.py",
            self.project / "uv.lock",
            self.project / "scripts/verify_workflow.py",
            self.fixture,
            self.model / "config.json",
            self.model / "chat_template.jinja",
        ]
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"modified")
                self.reject(self.report())
                path.write_bytes(original)
        self.runtime["packages"] = [["mlx", "changed"]]
        self.reject(self.report())

    def test_added_or_removed_source_rejected(self):
        added = self.project / "src/exl3_mlx/new.py"
        added.write_text("new runtime code")
        self.reject(self.report())
        added.unlink()
        (self.project / "src/exl3_mlx/training.py").unlink()
        self.reject(self.report())

    def test_phase_stdout_stderr_and_missing_hash_rejected(self):
        for suffix in ("stdout.json", "stderr.log"):
            with self.subTest(suffix=suffix):
                path = self.output / f"base-eval.{suffix}"
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.reject()
                path.write_bytes(original)
        del self.old["phases"]["base-eval"]["evidence"]
        self.reject()

    def test_checkpoint_payload_and_manifest_tampering_rejected(self):
        directory = self.output / "train/step-000001"
        directory.mkdir(parents=True)
        for name in V["CHECKPOINT_FILES"]:
            (directory / name).write_bytes(name.encode())
        state = {
            "format": "exl3-mlx-training-state-v1",
            "step": 1,
            "files": {name: V["sha256"](directory / name) for name in V["CHECKPOINT_FILES"]},
        }
        manifest = directory / "trainer_state.json"
        manifest.write_text(json.dumps(state))
        (self.output / "train.stdout.json").write_text("{}")
        (self.output / "train.stderr.log").write_text("completed")
        self.old["phases"]["train"] = {
            "exit_code": 0,
            "evidence": V["phase_evidence"](self.output, "train"),
        }
        V["validate_continuation"](self.old, self.current, self.output)
        payload = directory / "optimizer.safetensors"
        original = payload.read_bytes()
        payload.write_bytes(b"corrupted")
        self.reject()
        # A coherently rewritten checkpoint manifest still differs from saved evidence.
        state["files"][payload.name] = V["sha256"](payload)
        manifest.write_text(json.dumps(state))
        self.reject()
        payload.write_bytes(original)
        state["files"][payload.name] = V["sha256"](payload)
        state["step"] = 2
        manifest.write_text(json.dumps(state))
        self.reject()


if __name__ == "__main__":
    unittest.main()
