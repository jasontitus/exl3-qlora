"""Prior-NLL orchestration tests; all execution mocked, score math CPU-only."""

import copy
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


m = module(HERE / "prior_nll.py", "prior_nll_under_test")
d = module(HERE / "launch.py", "diagnostic_for_prior_tests")
FIT = (
    HERE.parents[2]
    / "loss-fix/apple/artifacts/loss-fix/runtime-v5/src/exl3_mlx/fit_probe.py"
)
# Locate the actual frozen CPU-only validation implementation from the workspace.
FIT = next(
    p / "loss-fix/apple/artifacts/loss-fix/runtime-v5/src/exl3_mlx/fit_probe.py"
    for p in HERE.parents
    if (p / "loss-fix").is_dir()
)
fit = module(FIT, "frozen_fit_cpu")


class Tests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name).resolve()
        self.base = [
            "/python",
            "-m",
            "exl3_mlx.workflow",
            "evaluate",
            "--model",
            "/model",
            "--data",
            "/data/test.jsonl",
            "--report",
            "/project/experiment/baseline.json",
            "--max-examples",
            "16",
            "--seq-length",
            "12288",
            "--token-metrics",
            "--chunk-size",
            "2048",
            "--activation-mode",
            "checkpoint",
            "--prefix-cache-mib",
            "1024",
            "--prefix-disk-cache",
            "/oldcache",
            "--prefix-disk-cache-gib",
            "64",
            "--memory-gib",
            "80",
            "--verbose",
        ]
        self.resources = {
            "prefix_cache_mib": 0,
            "prefix_disk_cache": None,
            "prefix_disk_cache_gib": 0,
            "memory_gib": 72,
        }

    def test_command_preserves_all_math_rows_masks(self):
        command = m.build_command(
            self.base, "/prior", self.root / "out.json", self.resources
        )
        self.assertNotIn("/oldcache", command)
        self.assertNotIn("--prefix-disk-cache", command)
        for flag in (
            "--model",
            "--data",
            "--seq-length",
            "--chunk-size",
            "--activation-mode",
            "--max-examples",
        ):
            self.assertEqual(
                command[command.index(flag) + 1], self.base[self.base.index(flag) + 1]
            )
        self.assertIn("--token-metrics", command)
        self.assertEqual(command[-2:], ["--adapter", "/prior"])
        self.assertEqual(command.count("--memory-gib"), 1)

    def test_bad_plan_hash_rejects_before_dynamic_import(self):
        path = self.root / "plan"
        path.write_text(json.dumps({"request": {"launcher": "/malicious"}}))
        with (
            patch.object(m, "launcher") as loader,
            self.assertRaisesRegex(ValueError, "hash"),
        ):
            m.run(path, "0" * 64)
        loader.assert_not_called()

    def test_bad_request_hash_rejects_before_dynamic_import(self):
        path = self.root / "request"
        path.write_text("{}")
        with (
            patch.object(m, "launcher") as loader,
            self.assertRaisesRegex(ValueError, "hash"),
        ):
            m.prepare(path, "0" * 64, self.root / "sealed")
        loader.assert_not_called()

    def setup_run(self):
        out = self.root / "out"
        diag = self.root / "diag.json"
        diag.write_text(
            json.dumps(
                {
                    "request": {
                        "runtime": "/runtime",
                        "conditions": [{"name": "prior", "adapter": "/prior"}],
                    }
                }
            )
        )
        request = {
            "diagnostic_sealed_plan": str(diag),
            "output": str(out),
            "resources": self.resources,
        }
        command = m.build_command(
            self.base, "/prior", out / "prior.json", self.resources
        )
        sealed = {
            "schema": "trusted-prior-nll-plan-v1",
            "request": request,
            "command": command,
        }
        planfile = self.root / "sealed"
        planfile.write_text(json.dumps(sealed))
        baseline = {
            "loss": 1.0,
            "evaluation_context": {
                "process_id": 10,
                "selected_examples_sha256": "rows",
            },
        }
        runner = types.SimpleNamespace(
            handle_stop=lambda *_: None,
            persist=lambda p, x: p.write_text(json.dumps(x)),
        )
        calls = []

        def child(key, argv, plan, receipt, path, **kwargs):
            calls.append(argv)
            record = {
                "process_id": 99,
                "exit_code": 0,
                "status": "completed",
                "log_sha256": {},
            }
            receipt["processes"][key] = record
            (out / "prior.json").write_text(
                json.dumps({"loss": 0.9, "resources": self.resources})
            )
            return record

        runner.run_child = child
        context = (
            None,
            {"commands": {"baseline": self.base}, "project": "/project"},
            None,
            None,
            None,
            runner,
            baseline,
            [{}] * 16,
        )
        return out, planfile, context, calls

    def test_success_one_explicit_child_no_retries(self):
        out, p, ctx, calls = self.setup_run()

        def verified(*args):
            return out / "prior.json", d.read(out / "prior.json")

        with (
            patch.object(m, "launcher", return_value=d),
            patch.object(m, "guard"),
            patch.object(m, "context", return_value=ctx),
            patch.object(m, "verify_result", side_effect=verified),
        ):
            result = m.run(p, d.digest(p))
        self.assertEqual(result["status"], "completed_pending_independent_audit")
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(result["prior_minus_base_loss"], -0.1)

    def test_verify_only_and_existing_refusal(self):
        out, p, ctx, calls = self.setup_run()
        with (
            patch.object(m, "launcher", return_value=d),
            patch.object(m, "guard"),
            patch.object(m, "context", return_value=ctx),
        ):
            self.assertEqual(m.run(p, d.digest(p), True)["status"], "verified_only")
            self.assertFalse(calls)
            self.assertFalse(out.exists())
            out.mkdir()
            with self.assertRaisesRegex(ValueError, "Existing"):
                m.run(p, d.digest(p))

    def test_child_failure_preserved_no_scoring(self):
        out, p, ctx, _calls = self.setup_run()

        def fail(*_, **__):
            raise RuntimeError("failed child")

        ctx[5].run_child = fail
        with (
            patch.object(m, "launcher", return_value=d),
            patch.object(m, "guard"),
            patch.object(m, "context", return_value=ctx),
            patch.object(m, "verify_result") as verify,
        ):
            with self.assertRaises(RuntimeError):
                m.run(p, d.digest(p))
            verify.assert_not_called()
        self.assertEqual(d.read(out / "launch.json")["status"], "failed")

    def test_completed_launch_required(self):
        with self.assertRaisesRegex(ValueError, "not complete"):
            m.check_completed_launch(d, {"status": "failed"}, None, None, None, {})

    def test_completed_phase_mode_and_argv_gates(self):
        p = self.root / "training-plan.json"
        p.write_text("{}")
        plan = {"commands": {"baseline": self.base}}
        log = self.root / "log"
        log.write_text("complete")
        record = {
            "status": "completed",
            "exit_code": 0,
            "environment": {"MLX_ENABLE_TF32": "0"},
            "argv": self.base,
            "process_id": 42,
            "stdout": str(log),
            "log_sha256": {"stdout": d.digest(log)},
        }
        result = {
            "status": "completed_pending_quality_audit",
            "plan": str(p),
            "plan_sha256": d.digest(p),
            "numerical_environment": d.ENV,
            "identity_sha256": {},
            "processes": {"baseline": record},
            "verified_phases": {"baseline": {"artifacts_sha256": {}}},
        }
        runner = types.SimpleNamespace(
            PHASES=("baseline",), verify_completion=lambda *_: None
        )
        m.check_completed_launch(d, result, plan, p, runner, {})
        record["environment"]["MLX_UNREVIEWED"] = "1"
        with self.assertRaisesRegex(ValueError, "wrong-mode"):
            m.check_completed_launch(d, result, plan, p, runner, {})
        del record["environment"]["MLX_UNREVIEWED"]
        record["argv"] = self.base + ["--raw-prompt"]
        with self.assertRaisesRegex(ValueError, "command"):
            m.check_completed_launch(d, result, plan, p, runner, {})

    def test_token_mask_and_observed_pid_checks_use_pinned_validator(self):
        rows = [{"input_ids": [0, 10, 20, 1], "loss_start": 2} for _ in range(16)]
        scores = {
            "examples": 16,
            "supervised_tokens": 32,
            "loss": 0.5,
            "rows": [
                {
                    "target_ids": [20, 1],
                    "predicted_ids": [20, 1],
                    "token_nll": [0.4, 0.6],
                    "supervised_tokens": 2,
                    "loss": 0.5,
                    "logits_sha256": "a" * 64,
                }
                for _ in rows
            ],
            "evaluation_context": {
                "process_id": 22,
                "selected_examples_sha256": "rows",
            },
        }
        baseline = copy.deepcopy(scores)
        baseline["evaluation_context"]["process_id"] = 11
        helper = types.SimpleNamespace(
            verify_evaluation=lambda *_a, **_k: (Path("/receipt"), scores)
        )
        with patch.dict("sys.modules", {"exl3_mlx.fit_probe": fit}):
            m.verify_result(
                d, helper, None, None, None, [], "/prior", 22, baseline, rows
            )
            with self.assertRaisesRegex(ValueError, "PID"):
                m.verify_result(
                    d, helper, None, None, None, [], "/prior", 23, baseline, rows
                )
            scores["rows"][0]["target_ids"] = [21, 1]
            with self.assertRaises(ValueError):
                m.verify_result(
                    d, helper, None, None, None, [], "/prior", 22, baseline, rows
                )


if __name__ == "__main__":
    unittest.main()
