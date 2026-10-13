"""CPU lifecycle tests. No model package imported and no GPU child launched."""

import datetime
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / "launch.py"
spec = importlib.util.spec_from_file_location("diagnostic_launch_tests", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FakeRunner:
    def __init__(self, test):
        self.test = test
        self.calls = []
        self.fail = None
        self.callback = None
        self.handle_stop = lambda *_: None

    def persist(self, path, receipt):
        path.write_text(json.dumps(receipt))

    def run_child(self, key, argv, plan, receipt, receipt_path):
        self.calls.append((key, argv))
        folder = receipt_path.parent
        record = {
            "argv": argv,
            "process_id": len(self.calls),
            "exit_code": 0,
            "status": "completed",
            "log_sha256": {},
        }
        receipt["processes"][key] = record
        for suffix in ("stdout", "stderr"):
            p = folder / (key + "." + suffix)
            p.write_text("log")
            record[suffix] = str(p)
            record["log_sha256"][suffix] = m.digest(p)
        if key == self.fail:
            record.update(exit_code=4, status="failed")
            raise RuntimeError("Mock child failed")
        if key.endswith("-score"):
            role = key.removesuffix("-score")
            out = Path(argv[argv.index("--output") + 1])
            out.write_text(
                json.dumps(
                    {
                        "total": 8,
                        "results": [{"id": t} for t in m.TASKS],
                        "plan_sha256": self.test.sealed["request"][
                            "diagnostic_plan_sha256"
                        ],
                        "adapter": None if role == "base" else "/adapters/" + role,
                    }
                )
            )
        else:
            out = Path(argv[argv.index("--report") + 1])
            role = key.split("-", 1)[0]
            task = key[len(role) + 1 :]
            t = next(x for x in self.test.diagnostic["tasks"] if x["id"] == task)
            out.write_text(
                json.dumps(
                    {
                        "prompt_ids": t["prompt_ids"],
                        "prompt_messages": t["messages"],
                        "checkpoint": "/model",
                        "adapter": None if role == "base" else "/adapters/" + role,
                        "activation_mode": "checkpoint",
                        "resources": self.test.resources,
                        "method": "greedy full-prefix recomputation",
                        "max_new_tokens": 8,
                        "tools": None,
                    }
                )
            )
        if self.callback:
            self.callback(key, out)
        return record


class Tests(unittest.TestCase):
    def test_prepare_paths_reject_relative_aliases_and_duplicate_adapters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            paths = {}
            for name in ("model", "runtime", "prior", "final"):
                (root / name).mkdir()
                paths[name] = str(root / name)
            for name in ("diagnostic_plan", "scorer", "helper", "runner"):
                (root / name).write_text("input")
                paths[name] = str(root / name)
            request = {k: v for k, v in paths.items() if k not in ("prior", "final")}
            request["output"] = str(root / "new-output")
            request["conditions"] = [
                {"name": "base", "adapter": None},
                {"name": "prior", "adapter": paths["prior"]},
                {"name": "final", "adapter": paths["final"]},
            ]
            m.validate_request_paths(request)
            for key in (
                "model",
                "runtime",
                "diagnostic_plan",
                "scorer",
                "helper",
                "runner",
                "output",
            ):
                changed = dict(request, **{key: "relative"})
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ValueError, "canonical"),
                ):
                    m.validate_request_paths(changed)
            alias = root / "alias"
            alias.symlink_to(root / "prior", target_is_directory=True)
            for value in (str(alias), "prior", paths["prior"]):
                changed = json.loads(json.dumps(request))
                changed["conditions"][2]["adapter"] = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    m.validate_request_paths(changed)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output = self.root / "output"
        self.diag_path = self.root / "diagnostic.json"
        self.diagnostic = {
            "tasks": [
                {
                    "id": t,
                    "messages": [{"role": "user", "content": t}],
                    "prompt_ids": [1, i, 2],
                }
                for i, t in enumerate(m.TASKS)
            ]
        }
        self.diag_path.write_text(json.dumps(self.diagnostic))
        self.resources = {
            "profile": "low-memory",
            "memory_gib": 72,
            "reserve_gib": 8,
            "packed_cache_mib": 2048,
            "decoded_cache_mib": 0,
            "prefix_cache_mib": 0,
            "prefix_disk_cache": None,
            "prefix_disk_cache_gib": 0,
            "allocator_cache_mib": 256,
        }
        self.sealed = {
            "schema": "trusted-diagnostic-generation-plan-v1",
            "task_ids": list(m.TASKS),
            "adapter_settings": {"adapter_layers": [39], "expert_adapter_layers": []},
            "request": {
                "output": str(self.output),
                "runtime": "/runtime",
                "runner": "/runner",
                "python": "/python",
                "model": "/model",
                "scorer": "/scorer",
                "diagnostic_plan": str(self.diag_path),
                "diagnostic_plan_sha256": m.digest(self.diag_path),
                "conditions": [
                    {"name": r, "adapter": None if r == "base" else "/adapters/" + r}
                    for r in m.ROLES
                ],
                "inference": {"activation_mode": "checkpoint", "chunk_size": 2048},
                "resources": self.resources,
            },
        }
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps(self.sealed))
        self.runner = FakeRunner(self)

    def execute(self, verify_only=False, guard=None):
        with (
            patch.object(
                m, "guard", side_effect=guard or (lambda _: {"verified": True})
            ),
            patch.object(m, "load_module", return_value=self.runner),
        ):
            return m.run(self.plan, m.digest(self.plan), verify_only)

    def test_complete_serial_fixed_campaign(self):
        result = self.execute()
        self.assertEqual(result["status"], "completed_pending_independent_audit")
        self.assertEqual(len(self.runner.calls), 27)
        for i, role in enumerate(m.ROLES):
            self.assertEqual(
                [k for k, _ in self.runner.calls[i * 9 : (i + 1) * 9]],
                [role + "-" + t for t in m.TASKS] + [role + "-score"],
            )
        self.assertEqual(set(result["scores"]), set(m.ROLES))
        self.assertEqual(result["actual_child_environment"]["MLX_ENABLE_TF32"], "0")
        for key, argv in self.runner.calls:
            self.assertEqual(argv[0], "/python")
            if not key.endswith("-score"):
                self.assertIn("--messages-file", argv)
                self.assertEqual(argv[argv.index("--max-tokens") + 1], "8")
                self.assertNotIn("--prefix-disk-cache", argv)

    def test_verify_only_no_output_or_process(self):
        result = self.execute(True)
        self.assertEqual(result["status"], "verified_only")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.runner.calls)

    def test_child_failure_stops_and_preserves_failed_receipt(self):
        self.runner.fail = "base-" + m.TASKS[2]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.execute()
        self.assertEqual(len(self.runner.calls), 3)
        self.assertEqual(m.read(self.output / "launch.json")["status"], "failed")
        self.assertFalse((self.output / "base-score.json").exists())
        with self.assertRaisesRegex(ValueError, "Existing"):
            self.execute()

    def test_wrong_generation_prompt_rejects_before_scoring(self):
        def corrupt(key, out):
            d = m.read(out)
            d["prompt_ids"] = [999]
            out.write_text(json.dumps(d))

        self.runner.callback = corrupt
        with self.assertRaisesRegex(ValueError, "prompt"):
            self.execute()
        self.assertEqual(len(self.runner.calls), 1)

    def test_report_drift_between_tasks_rejected(self):
        def corrupt(key, out):
            if key == "base-" + m.TASKS[-1]:
                (self.output / ("base-" + m.TASKS[0] + ".json")).write_text("{}")

        self.runner.callback = corrupt
        with self.assertRaisesRegex(ValueError, "report drift"):
            self.execute()
        self.assertEqual(len(self.runner.calls), 8)

    def test_runtime_drift_after_child_stops(self):
        calls = []

        def drift(_):
            calls.append(None)
            if len(calls) == 3:
                raise ValueError("runtime drift")
            return {}

        with self.assertRaisesRegex(ValueError, "runtime drift"):
            self.execute(guard=drift)
        self.assertEqual(len(self.runner.calls), 1)

    def test_wrong_sealed_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "hash"):
            m.run(self.plan, "0" * 64)

    def test_missing_fixed_task_rejected(self):
        self.sealed["task_ids"].pop()
        self.plan.write_text(json.dumps(self.sealed))
        with self.assertRaisesRegex(ValueError, "task order"):
            self.execute()
        self.assertFalse(self.runner.calls)

    def test_environment_restored_on_failure(self):
        import os

        before = dict(os.environ)
        self.runner.fail = "base-" + m.TASKS[0]
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual(dict(os.environ), before)

    def test_postscore_cross_condition_output_drift_fails_final_check(self):
        def corrupt(key, _):
            if key == "final-score":
                (self.output / ("base-" + m.TASKS[0] + ".json")).write_text("{}")

        self.runner.callback = corrupt
        with self.assertRaisesRegex(ValueError, "Final generation drift"):
            self.execute()
        self.assertEqual(m.read(self.output / "launch.json")["status"], "failed")

    def test_payload_requires_quiescent_exact_inventory(self):
        model = self.root / "model"
        model.mkdir()
        weight = model / "weight.bin"
        weight.write_bytes(b"abc")
        source = self.root / "source.json"
        files = {"weight.bin": {"size": 3, "sha256": m.digest(weight)}}
        source.write_text(
            json.dumps(
                {
                    "schema": "checkpoint-full-payload-sha256-v1",
                    "files": files,
                    "total_bytes": 3,
                }
            )
        )
        receipt = self.root / "payload.json"
        start = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=1)
        value = {
            "root": str(model),
            "started_utc": start.isoformat(),
            "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "files": files,
            "total_bytes": 3,
            "verified_manifest_sha256": m.digest(source),
        }
        receipt.write_text(json.dumps(value))
        request = {
            "model": str(model),
            "payload_receipt": str(receipt),
            "payload_sha256": m.digest(receipt),
            "payload_source_manifest": str(source),
            "payload_source_sha256": m.digest(source),
        }
        runner = types.SimpleNamespace(validate_payload_receipt=lambda _: None)
        state, _ = m.payload_state(request, runner, {})
        self.assertEqual(state[str(weight)]["size"], 3)
        (model / "extra").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "inventory"):
            m.payload_state(request, runner, {})
        (model / "extra").unlink()
        value["started_utc"] = (start - datetime.timedelta(days=1)).isoformat()
        receipt.write_text(json.dumps(value))
        request["payload_sha256"] = m.digest(receipt)
        with self.assertRaisesRegex(ValueError, "changed since"):
            m.payload_state(request, runner, {})

    def test_adapter_integrity_and_common_contract(self):
        payload = {"files": {n: {"sha256": "b" * 64} for n in m.BASE_NAMES}}
        settings = {
            "rank": 4,
            "alpha": 8,
            "adapter_layers": [39],
            "expert_adapter_layers": [],
            "activation_mode": "checkpoint",
        }
        conditions = [{"name": "base", "adapter": None}]
        for name in ("prior", "final"):
            folder = self.root / name
            folder.mkdir()
            weights = folder / "adapter.safetensors"
            weights.write_bytes(name.encode())
            config = {
                "format": "exl3-mlx-deepseek-short-text-v1",
                "engine_semantics": "v4",
                **settings,
                "base_identity": {n: "b" * 64 for n in m.BASE_NAMES},
                "adapter_sha256": m.digest(weights),
            }
            cp = folder / "adapter_config.json"
            cp.write_text(json.dumps(config))
            (folder / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "format": "exl3-mlx-training-state-v1",
                        "step": 30,
                        "files": {p.name: m.digest(p) for p in (cp, weights)},
                    }
                )
            )
            conditions.append(
                {"name": name, "adapter": str(folder), "step": 30, "settings": settings}
            )
        r = {
            "conditions": conditions,
            "inference": {
                "activation_mode": "checkpoint",
                "chunk_size": 2048,
                "engine_semantics": "v4",
            },
        }
        bindings = {}
        self.assertEqual(m.adapter_conditions(r, payload, bindings), settings)
        self.assertEqual(len(bindings), 6)
        (self.root / "prior/adapter.safetensors").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            m.adapter_conditions(r, payload, {})

    def test_scoring_failure_prevents_next_condition(self):
        self.runner.fail = "base-score"
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertEqual(len(self.runner.calls), 9)
        self.assertFalse(any(k.startswith("prior-") for k, _ in self.runner.calls))

    def test_signal_style_interruption_restores_handlers(self):
        import signal

        class Stopped(BaseException):
            pass

        old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

        def stop(*_):
            raise Stopped()

        self.runner.callback = stop
        with self.assertRaises(Stopped):
            self.execute()
        self.assertEqual({sig: signal.getsignal(sig) for sig in old}, old)
        self.assertEqual(m.read(self.output / "launch.json")["status"], "failed")

    def test_real_guard_rejects_source_or_model_drift(self):
        model = self.root / "smallmodel"
        model.mkdir()
        weight = model / "weights.bin"
        weight.write_bytes(b"fixed bytes")
        source = self.root / "source.py"
        source.write_text("original")
        sealed = {
            "request": {
                "model": str(model),
                "runtime": "/runtime",
                "snapshot_sha256": "snapshot",
                "helper": "/helper",
            },
            "inputs_sha256": {str(source): m.digest(source)},
            "model_file_stats": {str(weight): m.stat_id(weight)},
        }
        helper = types.SimpleNamespace(
            verify_runtime=lambda _: (Path("/runtime"), "snapshot")
        )
        with (
            patch.object(m, "check_environment"),
            patch.object(m, "load_module", return_value=helper),
        ):
            m.guard(sealed)
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "input drift"):
                m.guard(sealed)
            source.write_text("original")
            weight.write_bytes(b"same length")
            with self.assertRaisesRegex(ValueError, "Model changed"):
                m.guard(sealed)

    def test_conflicting_binding_rejected(self):
        p = self.root / "file"
        p.write_text("first")
        bindings = {}
        m.bind_file(p, m.digest(p), bindings)
        p.write_text("second")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            m.bind_file(p, m.digest(p), bindings)

    def test_duplicate_json_and_nonfinite_rejected(self):
        p = self.root / "bad.json"
        for value in ('{"a":1,"a":2}', '{"a":NaN}'):
            p.write_text(value)
            with self.assertRaises(ValueError):
                m.read(p)

    def test_existing_symlink_output_rejected(self):
        self.output.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(ValueError, "Existing"):
            self.execute()


if __name__ == "__main__":
    unittest.main()
