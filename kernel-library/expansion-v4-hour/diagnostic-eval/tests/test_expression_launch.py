"""CPU-only fixed-generation and admission tests; no compiler, model or GPU."""

import copy
import importlib.util
import json
import signal
import tempfile
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

HERE = Path(__file__).parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


m = module(HERE / "expression_launch.py", "expression_launcher_tests")
d = module(HERE / "launch.py", "expression_launch_bindings_tests")
b = module(HERE / "expression_build.py", "expression_build_tests")
BUILDER_PATH = (
    HERE.parent
    / "remote-m5/expression-probes-m5-v1/scripts/generated_expression_probes.py"
)
builder = module(BUILDER_PATH, "unchanged_expression_builder_tests")


class Tokenizer:
    eos_token_id = 99
    all_special_ids = (98, 99)
    tokens: ClassVar = {
        1: "simd_sum(sumf)",
        2: "tpig % nb",
        3: "bad()",
        4: " ",
        98: "<special>",
        99: "<eos>",
    }

    def __len__(self):
        return 100

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            self.tokens.get(i, "?")
            for i in ids
            if not (skip_special_tokens and i in self.all_special_ids)
        )

    def apply_chat_template(self, messages, **kwargs):
        return [10, 20 if messages[0]["content"] == m.PROBES[0] else 21]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.out = self.root / "out"
        self.tok = Tokenizer()
        self.plan = {"probes": []}
        for i, key in enumerate(m.PROBES):
            messages = self.root / (key + ".messages.json")
            messages.write_text(json.dumps([{"role": "user", "content": key}]))
            ids = self.root / (key + ".ids.json")
            ids.write_text(json.dumps(self.tok.apply_chat_template(d.read(messages))))
            self.plan["probes"].append(
                {
                    "id": key,
                    "operation": "sum" if i == 0 else "broadcast_mul",
                    "cases": builder.SUM_CASES if i == 0 else builder.ROW_CASES,
                    "messages": {"path": str(messages), "sha256": d.digest(messages)},
                    "prompt_token_ids": {"path": str(ids), "sha256": d.digest(ids)},
                    "prompt_tokens": 2,
                }
            )
        self.planpath = self.root / "expression-plan.json"
        self.planpath.write_text(json.dumps(self.plan))
        self.resources = {"memory_gib": 72, "prefix_cache_mib": 0}
        self.diagnostic = {
            "request": {
                "runtime": "/runtime",
                "model": "/model",
                "python": "/python",
                "runner": "/runner",
                "diagnostic_plan": "/original/plan.json",
                "inference": {"activation_mode": "checkpoint", "chunk_size": 2048},
                "resources": self.resources,
                "conditions": [
                    {"name": role, "adapter": None if role == "base" else "/" + role}
                    for role in d.ROLES
                ],
            },
            "adapter_settings": {"adapter_layers": [39], "expert_adapter_layers": []},
        }
        self.diagpath = self.root / "diagnostic.json"
        self.diagpath.write_text(json.dumps(self.diagnostic))
        self.sealed = {
            "schema": "trusted-expression-generation-plan-v1",
            "inputs_sha256": {},
            "request": {
                "diagnostic_sealed_plan": str(self.diagpath),
                "expression_plan": str(self.planpath),
                "expression_plan_sha256": d.digest(self.planpath),
                "builder": str(BUILDER_PATH),
                "build_wrapper": str(HERE / "expression_build.py"),
                "build_wrapper_sha256": d.digest(HERE / "expression_build.py"),
                "output": str(self.out),
            },
            "order": [{"condition": r, "probe": p} for r in d.ROLES for p in m.PROBES],
        }
        self.sealedpath = self.root / "sealed.json"
        self.sealedpath.write_text(json.dumps(self.sealed))
        self.calls = []
        self.responses = {}
        self.fail = None
        self.callback = None
        self.runner = types.SimpleNamespace(
            handle_stop=lambda *_: None,
            persist=lambda path, val: path.write_text(json.dumps(val)),
            run_child=self.child,
        )

    def result(self, probe=0, role="base", ids=None):
        p = self.plan["probes"][probe]
        ids = [probe + 1, 99] if ids is None else ids
        return {
            "prompt_ids": d.read(p["prompt_token_ids"]["path"]),
            "prompt_messages": d.read(p["messages"]["path"]),
            "checkpoint": "/model",
            "adapter": None if role == "base" else "/" + role,
            "activation_mode": "checkpoint",
            "resources": self.resources,
            "method": "greedy full-prefix recomputation",
            "max_new_tokens": 8,
            "tools": None,
            "generated_ids": ids,
            "eos_token_id": 99,
            "stop_reason": "eos" if ids[-1] == 99 else "max_tokens",
            "text": self.tok.decode(ids, True),
            "raw_text": self.tok.decode(ids),
        }

    def child(self, key, argv, child, receipt, path):
        self.calls.append((key, argv))
        record = {
            "argv": argv,
            "cwd": child["cwd"],
            "environment": child["environment"] | d.ENV,
            "process_id": 100 + len(self.calls),
            "started_ns": 1000 * len(self.calls),
            "finished_ns": 1000 * len(self.calls) + 1,
            "status": "completed",
            "exit_code": 0,
            "log_sha256": {},
        }
        for name in ("stdout", "stderr"):
            log = self.out / ("launch-" + key + "." + name + ".log")
            log.write_text("child log")
            record[name] = str(log)
            record["log_sha256"][name] = d.digest(log)
        receipt["processes"][key] = record
        if key == self.fail:
            record.update(status="failed", exit_code=1)
            raise RuntimeError("child failed")
        if key.endswith("-collect"):
            items = [argv[i + 1] for i, a in enumerate(argv) if a == "--receipt"]
            builder.collect(
                self.planpath, items, Path(argv[argv.index("--output") + 1])
            )
        else:
            role, probe = key.split("-", 1)
            val = self.responses.get(key, self.result(m.PROBES.index(probe), role))
            Path(argv[argv.index("--report") + 1]).write_text(json.dumps(val))
        if self.callback:
            self.callback(key)
        return record

    def run_mock(self, verify_only=False):
        with (
            patch.object(m, "load_launcher", return_value=d),
            patch.object(m, "guard"),
            patch.object(m, "tokenizer", return_value=self.tok),
            patch.object(
                d,
                "load_module",
                side_effect=lambda path, name: (
                    self.runner if path == "/runner" else builder
                ),
            ),
        ):
            return m.run(self.sealedpath, d.digest(self.sealedpath), verify_only)

    def test_command_exact_budget_prompt_and_condition(self):
        for role in d.ROLES:
            command = m.command(
                d, self.diagnostic, self.plan["probes"][1], role, self.root / "r"
            )
            self.assertEqual(command[command.index("--max-tokens") + 1], "8")
            self.assertEqual(
                command[command.index("--messages-file") + 1],
                self.plan["probes"][1]["messages"]["path"],
            )
            self.assertEqual("--adapter" in command, role != "base")
            self.assertNotIn("--tools-file", command)

    def test_six_generations_three_collections_serial_no_compiler(self):
        old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        result = self.run_mock()
        self.assertEqual(result["accepted_responses"], 6)
        self.assertEqual(len(self.calls), 9)
        self.assertEqual(
            [k for k, _ in self.calls],
            [
                k
                for r in d.ROLES
                for k in [r + "-" + p for p in m.PROBES] + [r + "-collect"]
            ],
        )
        self.assertFalse(result["compiled"])
        self.assertEqual(old, {s: signal.getsignal(s) for s in old})
        for c in result["collections"].values():
            self.assertIn("expression_build.py", c["build_command_not_executed"][1])
            self.assertTrue(Path(c["admission"]["path"]).is_file())
            self.assertFalse(Path(c["build_command_not_executed"][-1]).exists())

    def test_one_truncated_preserves_other_probe_and_all_raw_responses(self):
        self.responses["base-" + m.PROBES[0]] = self.result(ids=[1] + [4] * 7)
        result = self.run_mock()
        self.assertEqual(result["accepted_responses"], 5)
        self.assertEqual(result["rejected_responses"], 1)
        admission = d.read(result["collections"]["base"]["admission"]["path"])
        self.assertEqual(admission["results"][m.PROBES[0]]["category"], "truncated")
        self.assertEqual(
            admission["results"][m.PROBES[1]]["status"], "accepted_for_compile"
        )
        response = d.read(result["collections"]["base"]["path"])
        self.assertEqual(
            response["responses"][m.PROBES[0]]["text"], "simd_sum(sumf)" + " " * 7
        )
        self.assertEqual(len(self.calls), 9)

    def test_special_and_malformed_rejected_individually(self):
        for ids, category in (
            ([98, 1, 99], "unexpected_special_tokens"),
            ([3, 99], "malformed_expression"),
            ([99], "malformed_expression"),
        ):
            with self.subTest(ids=ids):
                outcome = m.validate_generation(
                    d,
                    self.diagnostic,
                    self.plan["probes"][0],
                    "base",
                    self.result(ids=ids),
                    self.tok,
                    builder,
                )
                self.assertEqual(outcome["category"], category)

    def test_wrong_metadata_is_infrastructure_failure(self):
        changes = [
            {"prompt_ids": [5]},
            {"raw_text": "repair"},
            {"stop_reason": "max_tokens"},
            {"adapter": "/final"},
            {"method": "sampled"},
            {"generated_ids": [True]},
            {"eos_token_id": 98},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_generation(
                    d,
                    self.diagnostic,
                    self.plan["probes"][0],
                    "base",
                    self.result() | change,
                    self.tok,
                    builder,
                )
        with self.assertRaisesRegex(ValueError, "EOS/stop"):
            m.validate_generation(
                d,
                self.diagnostic,
                self.plan["probes"][0],
                "base",
                self.result(ids=[1]),
                self.tok,
                builder,
            )

    def test_child_failure_preserved_stops_no_retry(self):
        self.fail = "base-" + m.PROBES[1]
        with self.assertRaisesRegex(RuntimeError, "child"):
            self.run_mock()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(d.read(self.out / "launch.json")["status"], "failed")
        with self.assertRaisesRegex(ValueError, "Existing"):
            self.run_mock()

    def test_cross_condition_report_drift_rejected(self):
        def mutate(key):
            if key == "final-collect":
                (self.out / ("base-" + m.PROBES[0] + ".json")).write_text("{}")

        self.callback = mutate
        with self.assertRaisesRegex(ValueError, "report drift"):
            self.run_mock()
        self.assertEqual(d.read(self.out / "launch.json")["status"], "failed")

    def test_verify_only_never_creates_output(self):
        self.assertEqual(self.run_mock(True)["status"], "verified_only")
        self.assertFalse(self.calls)
        self.assertFalse(self.out.exists())

    def test_wrong_hash_rejected_before_dynamic_import(self):
        with (
            patch.object(m, "load_launcher") as loader,
            self.assertRaisesRegex(ValueError, "hash"),
        ):
            m.run(self.sealedpath, "0" * 64)
        loader.assert_not_called()

    def test_guard_checks_inputs_and_delegates_full_identity(self):
        pinned = self.root / "pinned"
        pinned.write_text("x")
        sealed = {"inputs_sha256": {str(pinned): d.digest(pinned)}}
        with patch.object(d, "guard") as guard:
            m.guard(d, sealed, self.diagnostic)
            guard.assert_called_once_with(self.diagnostic)
            pinned.write_text("y")
            with self.assertRaisesRegex(ValueError, "drift"):
                m.guard(d, sealed, self.diagnostic)

    def controls(self):
        request = {
            "expression_plan_sha256": d.digest(self.planpath),
            "swift_source_sha256": "s",
            "builder_sha256": "b",
            "controls": {},
        }
        for name in ("golden", "negative"):
            origin = (
                "source_reference_not_model"
                if name == "golden"
                else "predeclared_negative_control_not_model"
            )
            # No path here: branch exercises reference hash comparison; real bind_tree tested elsewhere.
            job = {
                "origin": origin,
                "plan_sha256": request["expression_plan_sha256"],
                "plan": {"sha256": request["expression_plan_sha256"]},
                "responses": {"sha256": "response"},
                "harness_source": {"sha256": "s"},
                "builder": {"sha256": "b"},
            }
            jobpath = self.root / (name + "-job")
            jobpath.write_text(json.dumps(job))
            report = {
                "job_sha256": d.digest(jobpath),
                "plan_sha256": request["expression_plan_sha256"],
                "origin": origin,
                "response_sha256": "response",
                "device": "Apple M5 Max",
                "gpu_executed": True,
                "bindings_verified": True,
                "reference_passed": True,
                "numerical": "completed_fixed_cases",
                "started_unix": 100,
                "finished_unix": 101,
                "candidate_passed": name == "golden",
                "passed": name == "golden",
                "cases": [],
            }
            for p in self.plan["probes"]:
                for i, case in enumerate(p["cases"]):
                    for variant in ("reference", "candidate"):
                        ok = name == "golden" or variant == "reference"
                        report["cases"].append(
                            {
                                "probe": p["id"],
                                "case_index": i,
                                "variant": variant,
                                "case": case,
                                "canary_mutations": 0,
                                "input_mutation": False,
                                "nonfinite_values": 0,
                                "passed": ok,
                                "oracle_bit_mismatches": int(not ok),
                            }
                        )
            reportpath = self.root / (name + "-report")
            reportpath.write_text(json.dumps(report))
            request["controls"][name] = {
                "receipt": str(reportpath),
                "sha256": d.digest(reportpath),
                "job": str(jobpath),
                "job_sha256": d.digest(jobpath),
            }
        return request

    def test_controls_require_m5_all_cases_negative_discrimination_and_guards(self):
        request = self.controls()
        m.validate_controls(d, request, self.plan, {})
        for change in ("olddevice", "missingcase", "guard", "negativepasses"):
            req = self.controls()
            item = req["controls"]["negative"]
            val = d.read(item["receipt"])
            if change == "olddevice":
                val["device"] = "Apple M1 Ultra"
            elif change == "missingcase":
                val["cases"].pop()
            elif change == "guard":
                val["cases"][0]["canary_mutations"] = 1
            else:
                for c in val["cases"]:
                    if c["variant"] == "candidate":
                        c.update(passed=True, oracle_bit_mismatches=0)
            Path(item["receipt"]).write_text(json.dumps(val))
            item["sha256"] = d.digest(item["receipt"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_controls(d, req, self.plan, {})

    def test_build_gate_rejects_one_without_changing_other(self):
        fake = types.SimpleNamespace(parse_expression=builder.parse_expression)
        admission = {
            "results": {
                m.PROBES[0]: {"status": "rejected", "category": "truncated"},
                m.PROBES[1]: {
                    "status": "accepted_for_compile",
                    "expression": "tpig % nb",
                },
            }
        }
        b.install_gate(fake, self.plan, admission)
        with self.assertRaisesRegex(ValueError, "truncated"):
            fake.parse_expression("simd_sum(sumf)", "sum", builder.SUM_CASES)
        fake.parse_expression("tpig % nb", "broadcast_mul", builder.ROW_CASES)
        with self.assertRaisesRegex(ValueError, "changed"):
            fake.parse_expression("0", "broadcast_mul", builder.ROW_CASES)

    def test_control_error_incomplete_or_unordered_completion_is_not_negative_success(
        self,
    ):
        for change in (
            {"error": "post-dispatch failure"},
            {"numerical": "incomplete"},
            {"numerical": None},
            {"finished_unix": 99},
            {"started_unix": True},
            {"finished_unix": None},
        ):
            request = self.controls()
            item = request["controls"]["negative"]
            report = d.read(item["receipt"]) | change
            Path(item["receipt"]).write_text(json.dumps(report))
            item["sha256"] = d.digest(item["receipt"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_controls(d, request, self.plan, {})

    def test_build_process_records_bind_exact_argv_environment_keys_and_seriality(self):
        receipt = self.run_mock()
        b.validate_processes(m, d, self.sealed, self.diagnostic, self.plan, receipt)
        for field, value in (
            ("argv", ["wrong-program", "--wrong-mode"]),
            ("cwd", "/unrelated"),
            ("environment", {"MLX_ENABLE_TF32": "1"}),
            ("started_ns", 0),
            ("finished_ns", 1),
            ("stdout", "/other/log"),
            ("log_sha256", {}),
        ):
            wrong = copy.deepcopy(receipt)
            wrong["processes"]["prior-collect"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                b.validate_processes(
                    m, d, self.sealed, self.diagnostic, self.plan, wrong
                )
        wrong = copy.deepcopy(receipt)
        wrong["processes"]["unexpected-collect"] = wrong["processes"].pop(
            "prior-collect"
        )
        with self.assertRaisesRegex(ValueError, "keys"):
            b.validate_processes(m, d, self.sealed, self.diagnostic, self.plan, wrong)
        wrong = copy.deepcopy(receipt)
        wrong["actual_child_environment"]["MLX_ENABLE_TF32"] = "1"
        with self.assertRaisesRegex(ValueError, "environment"):
            b.validate_processes(m, d, self.sealed, self.diagnostic, self.plan, wrong)

    def test_build_gate_rejects_duplicate_operations(self):
        plan = copy.deepcopy(self.plan)
        plan["probes"][1]["operation"] = "sum"
        with self.assertRaisesRegex(ValueError, "Nonunique"):
            b.install_gate(
                types.SimpleNamespace(parse_expression=builder.parse_expression),
                plan,
                {},
            )

    def test_build_refuses_wrong_hash_before_import(self):
        p = self.root / "admission"
        p.write_text("{}")
        with (
            patch.object(b, "load") as loader,
            self.assertRaisesRegex(ValueError, "drift"),
        ):
            b.context(p, "0" * 64)
        loader.assert_not_called()

    def test_build_context_revalidates_admission_and_child_pid(self):
        self.responses["base-" + m.PROBES[0]] = self.result(ids=[1] + [4] * 7)
        result = self.run_mock()
        path = Path(result["collections"]["base"]["admission"]["path"])
        checksum = d.digest(path)
        with (
            patch.object(b, "load", return_value=m),
            patch.object(m, "load_launcher", return_value=d),
            patch.object(m, "guard"),
            patch.object(m, "tokenizer", return_value=self.tok),
            patch.object(d, "load_module", return_value=builder),
        ):
            context = b.context(path, checksum)
            self.assertEqual(
                context[0]["results"][m.PROBES[0]]["category"], "truncated"
            )
            launch_path = self.out / "launch.json"
            launch = d.read(launch_path)
            launch["processes"]["base-" + m.PROBES[0]]["process_id"] += 1
            launch_path.write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError, "PID"):
                b.context(path, checksum)

    def test_build_context_cannot_relabel_truncated_response_accepted(self):
        self.responses["base-" + m.PROBES[0]] = self.result(ids=[1] + [4] * 7)
        result = self.run_mock()
        path = Path(result["collections"]["base"]["admission"]["path"])
        admission = d.read(path)
        result0 = admission["results"][m.PROBES[0]]
        result0.update(
            status="accepted_for_compile",
            category="not_yet_compiled_or_executed",
            expression="simd_sum(sumf)",
        )
        path.write_text(json.dumps(admission))
        checksum = d.digest(path)
        launch_path = self.out / "launch.json"
        launch = d.read(launch_path)
        launch["results"]["base-" + m.PROBES[0]] = result0
        launch["collections"]["base"]["admission"]["sha256"] = checksum
        launch_path.write_text(json.dumps(launch))
        with (
            patch.object(b, "load", return_value=m),
            patch.object(m, "load_launcher", return_value=d),
            patch.object(m, "guard"),
            patch.object(m, "tokenizer", return_value=self.tok),
            patch.object(d, "load_module", return_value=builder),
            self.assertRaisesRegex(ValueError, "revalidation"),
        ):
            b.context(path, checksum)

    def test_actual_builder_mixed_gate_with_compiler_mocked(self):
        """Exercise original per-probe exception handling, not a replacement builder."""
        plan = copy.deepcopy(self.plan)
        plan.update(
            format="heldout-expression-probe-plan-v1",
            model_tokenizer="/model",
            tokenizer_files={},
        )
        for i, probe in enumerate(plan["probes"]):
            golden = ("simd_sum(sumf)", "tpig % nb")[i]
            probe.update(golden=golden, function="kernel_" + str(i))
            for field, data in (
                ("source_record", "{}"),
                ("template", "value = __HOLE__;\n"),
                ("golden_source", "value = " + golden + ";\n"),
            ):
                path = self.root / (probe["id"] + "." + field)
                path.write_text(data)
                probe[field] = {"path": str(path), "sha256": d.digest(path)}
        self.planpath.write_text(json.dumps(plan))
        reports = []
        results = {}
        for i, probe in enumerate(plan["probes"]):
            ids = [1] + [4] * 7 if i == 0 else [2, 99]
            path = self.root / (probe["id"] + ".generation.json")
            path.write_text(json.dumps(self.result(i, ids=ids)))
            reports.append(probe["id"] + "=" + str(path))
            results[probe["id"]] = m.validate_generation(
                d, self.diagnostic, probe, "base", d.read(path), self.tok, builder
            )
        responses = self.root / "responses.json"
        builder.collect(self.planpath, reports, responses)
        admission = {
            "condition": "base",
            "results": results,
            "responses": {"path": str(responses), "sha256": d.digest(responses)},
        }
        output = self.root / "compiled"
        (self.root / "scripts").mkdir()
        (self.root / "scripts/verify_expression_probes.swift").write_text(
            "// mocked compiler fixture"
        )
        compiled = []

        def compile_source(source, folder):
            folder.mkdir(parents=True)
            path = folder / "source.metal"
            path.write_bytes(source)
            library = folder / "out.metallib"
            library.write_bytes(b"mock compiler result")
            compiled.append((folder, source))
            return {
                "passed": True,
                "source": builder.binding(path),
                "library": builder.binding(library),
            }

        def subprocess_run(argv, **kwargs):
            if argv[1] == "swiftc":
                Path(argv[argv.index("-o") + 1]).write_bytes(b"mock swift binary")
            return types.SimpleNamespace(
                returncode=0, stdout="mock toolchain", stderr=""
            )

        ctx = (admission, self.sealed, self.diagnostic, plan, {}, m, d, builder)
        with (
            patch.object(b, "context", return_value=ctx),
            patch.object(m, "guard"),
            patch.object(builder, "tokenizer", return_value=self.tok),
            patch.object(builder, "compile_source", side_effect=compile_source),
            patch.object(builder.subprocess, "run", side_effect=subprocess_run),
            patch.object(builder, "STAGE", self.root),
        ):
            outcome = b.build(self.root / "admission", "mock", output)
        self.assertEqual(outcome["status"], "built_pending_numerical_evaluation")
        self.assertEqual(
            [p["candidate_status"] for p in outcome["probes"]], ["rejected", "compiled"]
        )
        self.assertEqual(len(compiled), 3)  # Both references and exactly one candidate.
        self.assertEqual(compiled[-1][1], b"value = tpig % nb;\n")
        self.assertFalse((output / m.PROBES[0] / "candidate").exists())
        self.assertEqual(
            d.read(responses)["responses"][m.PROBES[0]]["text"],
            "simd_sum(sumf)" + " " * 7,
        )
        self.assertFalse(outcome["gpu_executed"])


if __name__ == "__main__":
    unittest.main()
