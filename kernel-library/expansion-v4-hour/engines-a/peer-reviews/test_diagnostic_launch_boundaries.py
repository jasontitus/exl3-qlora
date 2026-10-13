"""Independent CPU-only guard tests; no model import or child execution."""

import copy
import datetime
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(ROOT / "diagnostic-eval/launch.py", "independent_diagnostic_launch")
runner = load(ROOT / "remote-m5/run_m5_plan.py", "independent_pinned_m5_runner")


class Guards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def environment(self, values, **overrides):
        request = {"python": str(Path(m.sys.executable).absolute()), "hostname": "test-host"}
        with (
            patch.dict(os.environ, values, clear=True),
            patch.object(m.platform, "python_version", return_value=overrides.get("python", "3.12.13")),
            patch.object(m.socket, "gethostname", return_value=overrides.get("host", "test-host")),
            patch.object(m.importlib.metadata, "version", side_effect=lambda name: overrides.get("versions", m.PACKAGES)[name]),
        ):
            return m.check_environment(request)

    def test_exact_environment_accepted_and_all_drift_rejected(self):
        self.assertEqual(self.environment(m.ENV), m.PACKAGES)
        variants = [({}, {}), ({"MLX_ENABLE_TF32": "1"}, {}),
                    ({**m.ENV, "MLX_EXTRA": "1"}, {}),
                    ({**m.ENV, "PYTHONHOME": "/foreign"}, {}),
                    ({**m.ENV, "DYLD_LIBRARY_PATH": "/foreign"}, {}),
                    ({**m.ENV, "LD_PRELOAD": "/foreign"}, {}),
                    (m.ENV, {"python": "3.12.12"}),
                    (m.ENV, {"host": "other-host"}),
                    (m.ENV, {"versions": {**m.PACKAGES, "mlx": "0.32.2"}})]
        for values, overrides in variants:
            with self.subTest(values=values, overrides=overrides), self.assertRaises(ValueError):
                self.environment(values, **overrides)

    def payload(self):
        model = self.root / "model"
        model.mkdir()
        payload = model / "weight.bin"
        payload.write_bytes(b"pinned fixture")
        files = {"weight.bin": {"size": payload.stat().st_size, "sha256": m.digest(payload)}}
        source = self.root / "source.json"
        source.write_text(json.dumps({"schema": "checkpoint-full-payload-sha256-v1", "files": files,
                                      "total_bytes": payload.stat().st_size}))
        receipt = self.root / "receipt.json"
        started = datetime.datetime.now(datetime.UTC).isoformat()
        value = {"schema": "checkpoint-full-payload-sha256-v1", "root": str(model),
                 "started_utc": started, "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                 "verified_manifest_sha256": m.digest(source), "files": files,
                 "total_bytes": payload.stat().st_size}
        receipt.write_text(json.dumps(value))
        request = {"model": str(model), "payload_receipt": str(receipt),
                   "payload_sha256": m.digest(receipt), "payload_source_manifest": str(source),
                   "payload_source_sha256": m.digest(source)}
        return request, receipt, value, payload

    def test_real_payload_validator_valid_fixture(self):
        request, _path, value, payload = self.payload()
        states, receipt = m.payload_state(request, runner, {})
        self.assertEqual(receipt, value)
        self.assertEqual(states[str(payload)], m.stat_id(payload))

    def test_real_payload_validator_rejects_malformed_or_incomplete_receipts(self):
        request, path, value, _payload = self.payload()
        variants = [
            {**value, "files": {}}, {**value, "total_bytes": 0},
            {**value, "completed_utc": None},
            {**value, "verified_manifest_sha256": "wrong"},
            {**value, "started_utc": value["started_utc"].split("+")[0]},
            {**value, "started_utc": (datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1)).isoformat()},
            {**value, "root": "/another-model"},
        ]
        for changed in variants:
            with self.subTest(changed=changed):
                path.write_text(json.dumps(changed))
                request["payload_sha256"] = m.digest(path)
                with self.assertRaises((ValueError, TypeError)):
                    m.payload_state(request, runner, {})

    def test_real_payload_validator_rejects_future_completion(self):
        request, path, value, _payload = self.payload()
        value["completed_utc"] = (datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=1)).isoformat()
        path.write_text(json.dumps(value))
        request["payload_sha256"] = m.digest(path)
        with self.assertRaisesRegex(ValueError, "Future"):
            m.payload_state(request, runner, {})

    def test_same_length_payload_rewrite_after_verification_rejects(self):
        request, _path, _value, payload = self.payload()
        payload.write_bytes(b"changed fixture"[:payload.stat().st_size])
        with self.assertRaisesRegex(ValueError, "changed since"):
            m.payload_state(request, runner, {})

    def test_symlink_payload_rejected(self):
        request, _path, _value, payload = self.payload()
        external = self.root / "external"
        external.write_bytes(payload.read_bytes())
        payload.unlink()
        payload.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "symlink"):
            m.payload_state(request, runner, {})

    def test_generation_contract_model_adapter_mode_resources_tools_rejected(self):
        task = "conv_interval"
        plan = self.root / "diag.json"
        plan.write_text(json.dumps({"tasks": [{"id": task, "prompt_ids": [1, 2], "messages": []}]}))
        sealed = {"request": {"diagnostic_plan": str(plan), "model": "/model", "conditions":
                              [{"name": "prior", "adapter": "/prior"}],
                              "inference": {"activation_mode": "checkpoint"},
                              "resources": {"prefix_cache_mib": 0}}}
        valid = {"prompt_ids": [1, 2], "prompt_messages": [], "checkpoint": "/model",
                 "adapter": "/prior", "activation_mode": "checkpoint",
                 "resources": {"prefix_cache_mib": 0}, "method": "greedy full-prefix recomputation",
                 "max_new_tokens": 8, "tools": None}
        report = self.root / "generation.json"
        report.write_text(json.dumps(valid))
        m.verify_generation(sealed, "prior", task, report)
        for field, value in [("checkpoint", "/wrong"), ("adapter", None),
                             ("activation_mode", "bf16"), ("resources", {"prefix_cache_mib": 1024}),
                             ("max_new_tokens", 9), ("tools", []), ("prompt_messages", ["extra"]),
                             ("method", "sampled")]:
            changed = copy.deepcopy(valid)
            changed[field] = value
            report.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.verify_generation(sealed, "prior", task, report)


if __name__ == "__main__":
    unittest.main()
