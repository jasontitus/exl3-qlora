"""Stdlib-only durable ledger tests; no MLX or model execution."""

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/exl3_mlx/evaluation_progress.py"
spec = importlib.util.spec_from_file_location("evaluation_progress_under_test", SOURCE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
H = "1" * 64
ROWS = [
    {"input_ids": [0, 2, 3, 1], "loss_start": 2},
    {"input_ids": [0, 3, 4, 5, 1], "loss_start": 1},
    {"input_ids": [0, 2, 3, 1], "loss_start": 2},
]


def identity(rows=ROWS, **overrides):
    values = dict(
        project_sha256=H,
        base={
            "payload_manifest_sha256": H,
            "verification_receipt_sha256": H,
            "verification_method": "fresh_full_sha256",
            "continuity_sha256": None,
            "format": "packed",
            "metadata_sha256": H,
            "layer_count": 40,
        },
        runtime={
            "source_manifest_sha256": H,
            "artifact_manifest_sha256": H,
            "dependency_lock_sha256": H,
            "python": "3.12.13",
            "versions": {"mlx": "test-only"},
            "host_id": "CPU-fixture",
            "os": "test",
            "device": "test",
            "backend": "test",
        },
        arithmetic={
            "activation_mode": "checkpoint",
            "tf32": "0",
            "eval_mode": True,
            "preserve_mode": False,
            "activation_dtype": "bfloat16",
            "weight_format": "test-packed",
            "engine_semantics": "test-v2",
            "exl3_chunk_size": 2048,
            "head_chunk_size": 32,
            "cache_semantics_sha256": H,
            "backend_options": {},
        },
        tokenizer={
            "artifact_manifest_sha256": H,
            "template_sha256": H,
            "tokenizer_size": 10,
            "model_vocab_size": 12,
            "eos_token_id": 1,
        },
        condition={
            "kind": "zero_b",
            "rank": 4,
            "alpha": 8.0,
            "adapter_layers": [39],
            "expert_adapter_layers": [],
            "a_initialization_sha256": H,
            "zero_b_verification_sha256": H,
            "adapter_config_sha256": None,
            "adapter_payload_sha256": None,
            "checkpoint_step": None,
        },
        seq_length=16,
    )
    values.update(overrides)
    return p.make_identity(copy.deepcopy(rows), **values)


def origin(run="a", pid=100):
    return {
        "run_id": run,
        "process_id": pid,
        "argv": ["python", "evaluate"],
        "host_id": "CPU-fixture",
    }


def score(row, index=0):
    target = row["input_ids"][row["loss_start"] :]
    return {
        "target_ids": target,
        "predicted_ids": list(target),
        "token_nll": [0.1 * (index + 1)] * len(target),
        "logits": {
            "sha256": hashlib.sha256(bytes([index])).hexdigest(),
            "scope": "supervised_logits",
            "shape": [len(target), 12],
            "dtype": "float32",
            "byte_order": "little",
        },
    }


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "progress"
        self.ident = identity()
        self.guard = lambda: p.digest(self.ident)

    def ledger(self, resume=False, **kwargs):
        return p.EvaluationProgress(
            self.root,
            kwargs.get("identity", self.ident),
            kwargs.get("rows", ROWS),
            verify_context=kwargs.get("guard", self.guard),
            resume=resume,
        )

    def complete(self, ledger, run="a", pid=100):
        for i in ledger.pending_indices:
            ledger.commit_row(ledger.begin_row(i, origin(run, pid)), score(ROWS[i], i))

    def test_interrupt_resume_missing_only_and_original_producers(self):
        with self.ledger() as ledger:
            ledger.commit_row(ledger.begin_row(1, origin()), score(ROWS[1], 1))
            ledger.begin_row(2, origin())  # Interrupted during computation, no row receipt.
        with self.ledger(True) as ledger:
            self.assertEqual(ledger.pending_indices, (0, 2))
            self.complete(ledger, "b", 200)
            result = ledger.aggregate(
                origin("assembler", 300), output=Path(self.tmp.name).resolve() / "result.json"
            )
            self.assertEqual(
                [r["producer"]["process_id"] for r in result["row_provenance"]], [200, 100, 200]
            )
            self.assertEqual(result["exposure"]["attempted_row_computations"], 4)
            self.assertEqual(result["exposure"]["unknown_or_incomplete_attempts"], 1)
            self.assertEqual(result["exposure"]["outcome_adoptions"], 1)
            self.assertTrue(result["row_provenance"][1]["reused"])
            self.assertEqual(result["supervised_tokens"], 8)
            self.assertEqual(
                result["loss"], math.fsum([0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]) / 8
            )

    def test_fresh_and_resumed_v2_exact_reduction_and_digest(self):
        with self.ledger() as ledger:
            self.complete(ledger)
            fresh = ledger.aggregate(origin())
        with self.ledger(True) as ledger:
            self.assertEqual(ledger.pending_indices, ())
            resumed = ledger.aggregate(origin("reuse", 200))
        for key in ("loss", "loss_sum", "macro_row_loss", "ordered_row_logits_digest_v2", "rows"):
            self.assertEqual(fresh[key], resumed[key])
        self.assertEqual(resumed["exposure"]["outcome_adoptions"], 3)
        self.assertNotIn("logits_sha256", resumed)

    def test_all_complete_aggregation_in_fresh_process_rejects_model_imports(self):
        with self.ledger() as ledger:
            self.complete(ledger)
        program = """
import builtins, importlib.util, os, sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx', 'mlx_lm'}:
        raise AssertionError('CPU aggregate attempted model runtime import')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location('cpu_progress_fixture', sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
identity = t.identity()
with t.p.EvaluationProgress(sys.argv[2], identity, t.ROWS,
        verify_context=lambda: t.p.digest(identity), resume=True) as ledger:
    assert ledger.pending_indices == ()
    result = ledger.aggregate(t.origin('separate-cpu-assembler', os.getpid()))
    assert result['examples'] == 3
    assert result['exposure']['outcome_adoptions'] == 3
assert 'mlx' not in sys.modules
assert 'mlx_lm' not in sys.modules
"""
        run = subprocess.run(
            [sys.executable, "-c", program, str(Path(__file__).resolve()), str(self.root)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_all_complete_path_never_invokes_a_scorer_or_factory(self):
        with self.ledger() as ledger:
            self.complete(ledger)
        factory_calls = []
        with self.ledger(True) as ledger:
            if ledger.pending_indices:
                factory_calls.append("would-create-model")
            result = ledger.aggregate(origin("new", 200))
        self.assertEqual(factory_calls, [])
        self.assertEqual(result["examples"], 3)

    def test_duplicate_content_is_distinct_occurrence(self):
        self.assertEqual(
            self.ident["selection"]["row_sha256"][0], self.ident["selection"]["row_sha256"][2]
        )
        with self.ledger() as ledger:
            self.complete(ledger)
            result = ledger.aggregate(origin())
        self.assertEqual([r["index"] for r in result["row_logit_descriptors"]], [0, 1, 2])

    def test_explicit_resume_required(self):
        with self.ledger():
            pass
        with self.assertRaisesRegex(ValueError, "explicit resume"):
            with self.ledger():
                pass

    def test_empty_resume_and_named_crash_temp(self):
        with self.ledger():
            pass
        (self.root / (".tmp-" + "a" * 32 + ".json")).write_text("{partial")
        with self.ledger(True) as ledger:
            self.assertEqual(ledger.pending_indices, (0, 1, 2))

    def test_partial_aggregate_never_publishes(self):
        out = Path(self.tmp.name).resolve() / "result.json"
        with self.ledger() as ledger:
            with self.assertRaises(ValueError):
                ledger.aggregate(origin(), output=out)
        self.assertFalse(out.exists())

    def test_no_overwrite_row_or_output(self):
        out = Path(self.tmp.name).resolve() / "result.json"
        out.write_text("preserved")
        with self.ledger() as ledger:
            self.complete(ledger)
            with self.assertRaises(ValueError):
                ledger.begin_row(0, origin())
            with self.assertRaises(FileExistsError):
                ledger.aggregate(origin(), output=out)
        self.assertEqual(out.read_text(), "preserved")

    def test_real_os_single_writer_lock(self):
        with self.ledger():
            with self.assertRaises(BlockingIOError):
                with self.ledger(True):
                    pass
        with self.ledger(True):
            pass

    def test_lock_inode_replacement_rejected(self):
        with self.ledger() as ledger:
            lock = self.root / ".writer.lock"
            lock.unlink()
            lock.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "inode"):
                ledger.pending_indices

    def test_foreign_entry_and_symlink_rejected(self):
        for name in ("unknown.json", "row-99999999.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve() / "p"
                with p.EvaluationProgress(root, self.ident, ROWS, verify_context=self.guard):
                    pass
                (root / name).write_text("{}")
                with self.assertRaises(ValueError):
                    with p.EvaluationProgress(
                        root, self.ident, ROWS, verify_context=self.guard, resume=True
                    ):
                        pass
        with self.ledger():
            pass
        (self.root / (".tmp-" + "a" * 32 + ".json")).symlink_to("/does-not-exist")
        with self.assertRaises(ValueError):
            with self.ledger(True):
                pass

    def test_symlink_lock_and_progress_root_rejected(self):
        self.root.mkdir()
        (self.root / ".writer.lock").symlink_to("/does-not-exist")
        with self.assertRaises(OSError):
            with self.ledger():
                pass
        alias = Path(self.tmp.name).resolve() / "alias"
        alias.symlink_to(self.root)
        with self.assertRaises(ValueError):
            with p.EvaluationProgress(alias, self.ident, ROWS, verify_context=self.guard):
                pass

    def test_context_drift_before_commit_leaves_unknown_attempt(self):
        current = [p.digest(self.ident)]
        with self.ledger(guard=lambda: current[0]) as ledger:
            attempt = ledger.begin_row(0, origin())
            current[0] = "0" * 64
            with self.assertRaisesRegex(ValueError, "context guard"):
                ledger.commit_row(attempt, score(ROWS[0]))
        with self.ledger(True) as ledger:
            self.assertEqual(ledger.exposure()["unknown_or_incomplete_attempts"], 1)
            self.assertEqual(ledger.pending_indices, (0, 1, 2))

    def test_completed_file_mutation_before_final_link_rejected(self):
        out = Path(self.tmp.name).resolve() / "result.json"
        with self.ledger() as ledger:
            self.complete(ledger)
            real_fsync = p.os.fsync
            once = [False]

            def mutate(fd):
                real_fsync(fd)
                if not once[0]:
                    once[0] = True
                    (self.root / "row-00000000.json").write_text("{}")

            with patch.object(p.os, "fsync", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "drift"):
                    ledger.aggregate(origin(), output=out)
        self.assertFalse(out.exists())

    def test_duplicate_json_nonfinite_and_noncanonical_receipts_rejected(self):
        with self.ledger() as ledger:
            self.complete(ledger)
        path = self.root / "row-00000000.json"
        old = path.read_bytes()
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', json.dumps(json.loads(old), indent=2).encode()):
            with self.subTest(raw=raw[:30]):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    with self.ledger(True):
                        pass
        path.write_bytes(old)

    def test_missing_attempt_or_event_gap_rejected(self):
        with self.ledger() as ledger:
            self.complete(ledger)
        (self.root / "event-00000000.json").unlink()
        with self.assertRaises(ValueError):
            with self.ledger(True):
                pass

    def test_same_run_conflicting_pid_rejected(self):
        with self.ledger() as ledger:
            ledger.begin_row(0, origin())
            with self.assertRaisesRegex(ValueError, "conflicting"):
                ledger.begin_row(1, origin(pid=200))

    def test_closed_or_reentered_object_rejected(self):
        ledger = self.ledger()
        with ledger:
            pass
        with self.assertRaises(ValueError):
            ledger.pending_indices
        with self.assertRaises(ValueError):
            with ledger:
                pass

    def test_bad_scores_do_not_publish_and_preserve_attempt(self):
        for name in (
            "targets",
            "predictions",
            "negative",
            "nan",
            "missing",
            "shape",
            "dtype",
            "scope",
            "byteorder",
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                with p.EvaluationProgress(
                    Path(tmp).resolve() / "p", self.ident, ROWS, verify_context=self.guard
                ) as ledger:
                    attempt = ledger.begin_row(0, origin())
                    value = score(ROWS[0])
                    if name == "targets":
                        value["target_ids"][0] = 9
                    elif name == "predictions":
                        value["predicted_ids"][0] = 12
                    elif name == "negative":
                        value["token_nll"][0] = -0.1
                    elif name == "nan":
                        value["token_nll"][0] = float("nan")
                    elif name == "missing":
                        value["token_nll"].pop()
                    elif name == "shape":
                        value["logits"]["shape"] = [True, 12]
                    elif name == "dtype":
                        value["logits"]["dtype"] = "unknown"
                    elif name == "scope":
                        value["logits"]["scope"] = "selected_tokens_only"
                    else:
                        value["logits"]["byte_order"] = "native"
                    with self.assertRaises(ValueError):
                        ledger.commit_row(attempt, value)
                    self.assertFalse((ledger.root / "row-00000000.json").exists())

    def test_digest_distinguishes_all_logits_and_supervised_scope(self):
        with self.ledger() as ledger:
            self.complete(ledger)
            a = ledger.aggregate(origin())
        with tempfile.TemporaryDirectory() as tmp:
            with p.EvaluationProgress(
                Path(tmp).resolve() / "p", self.ident, ROWS, verify_context=self.guard
            ) as ledger:
                for i, row in enumerate(ROWS):
                    val = score(row, i)
                    val["logits"]["scope"] = "all_next_token_logits"
                    val["logits"]["shape"] = [len(row["input_ids"]) - 1, 12]
                    ledger.commit_row(ledger.begin_row(i, origin()), val)
                b = ledger.aggregate(origin())
        self.assertNotEqual(a["ordered_row_logits_digest_v2"], b["ordered_row_logits_digest_v2"])
        self.assertEqual(a["loss"], b["loss"])

    def test_rounding_sensitive_flat_token_sum_is_used(self):
        with self.ledger() as ledger:
            values = [[1e16, 1.0], [1.0] * 4, [1.0] * 2]
            for i, row in enumerate(ROWS):
                v = score(row, i)
                v["token_nll"] = values[i]
                ledger.commit_row(ledger.begin_row(i, origin()), v)
            out = ledger.aggregate(origin())
        self.assertEqual(out["loss_sum"], math.fsum(x for row in values for x in row))
        self.assertNotEqual(out["loss_sum"], math.fsum(math.fsum(row) for row in values))

    def test_crash_after_row_link_recovers_without_second_completion_event(self):
        with self.ledger() as ledger:
            attempt = ledger.begin_row(0, origin())
            real_link = p.os.link

            def linked_then_crash(source, destination):
                real_link(source, destination)
                raise OSError("injected crash after publication")

            with patch.object(p.os, "link", side_effect=linked_then_crash):
                with self.assertRaisesRegex(OSError, "injected"):
                    ledger.commit_row(attempt, score(ROWS[0]))
        with self.ledger(True) as ledger:
            self.assertEqual(ledger.pending_indices, (1, 2))
            self.assertEqual(ledger.exposure()["unknown_or_incomplete_attempts"], 0)
            self.assertEqual(ledger.receipts[0]["producer"], origin())

    def test_row_producer_or_adoption_timing_tamper_rejects(self):
        with self.ledger() as ledger:
            self.complete(ledger)
        row = self.root / "row-00000000.json"
        original = row.read_bytes()
        value = json.loads(original)
        value["producer"]["process_id"] = 999
        row.write_bytes(p.canonical(value))
        with self.assertRaisesRegex(ValueError, "producer/attempt"):
            with self.ledger(True):
                pass
        row.write_bytes(original)
        with self.ledger(True) as ledger:
            ledger.aggregate(origin("new", 200))
        adoption = self.root / "event-00000003.json"
        value = json.loads(adoption.read_bytes())
        value["utc"] = "2000-01-01T00:00:00+00:00"
        adoption.write_bytes(p.canonical(value))
        with self.assertRaisesRegex(ValueError, "predates"):
            with self.ledger(True):
                pass

    def test_conflicting_assembler_cannot_impersonate_current_producer(self):
        with self.ledger() as ledger:
            self.complete(ledger)
            with self.assertRaisesRegex(ValueError, "conflicting assembler"):
                ledger.aggregate(origin(pid=999))

    def test_fresh_and_resumed_aggregates_compose_with_product_token_auditor(self):
        audit_spec = importlib.util.spec_from_file_location(
            "progress_composed_token_audit", SOURCE.with_name("evaluation_audit.py")
        )
        audit = importlib.util.module_from_spec(audit_spec)
        audit_spec.loader.exec_module(audit)
        categories = {
            "prediction_vocab_size": 12,
            "tokenizer_size": 10,
            "rows": [],
            "counts": {"metal/added/code": 8},
        }
        for i, row in enumerate(ROWS):
            n = len(row["input_ids"]) - row["loss_start"]
            categories["rows"].append(
                {
                    "index": i,
                    "loss_start": row["loss_start"],
                    "input_ids_sha256": p.digest(row["input_ids"]),
                    "supervised_tokens": n,
                    "categories": ["metal/added/code"] * n,
                }
            )
        with self.ledger() as ledger:
            self.complete(ledger)
            fresh = ledger.aggregate(origin())
        with self.ledger(True) as ledger:
            resumed = ledger.aggregate(origin("adopter", 200))
        result = audit.compare_token_evaluations(ROWS, categories, fresh, resumed)
        self.assertEqual(result["aggregate"]["all"]["nll_delta"], 0)
        self.assertEqual(result["aggregate"]["all"]["tokens"], 8)
        self.assertEqual(result["status"], "metrics_consistent_pending_provenance_audit")


class IdentityTests(unittest.TestCase):
    def test_malformed_caller_identity_and_boolean_shift_reject(self):
        for value in (
            None,
            [],
            {},
            {**identity(), "next_token_shift": True},
            {**identity(), "selection": None},
        ):
            with self.subTest(value=str(value)[:40]):
                with self.assertRaises(ValueError):
                    p.EvaluationProgress("unused", value, ROWS, verify_context=lambda: H)

    def test_identity_is_detached(self):
        rows = copy.deepcopy(ROWS)
        v = identity(rows)
        rows[0]["input_ids"][0] = 9
        self.assertNotEqual(v["selection"]["selected_examples_sha256"], p.digest(rows))

    def test_every_bound_semantic_field_prevents_reuse(self):
        first = identity()
        changes = [
            ("base", "payload_manifest_sha256", "2" * 64),
            ("runtime", "artifact_manifest_sha256", "2" * 64),
            ("runtime", "host_id", "other"),
            ("arithmetic", "tf32", "unset"),
            ("arithmetic", "exl3_chunk_size", 512),
            ("arithmetic", "cache_semantics_sha256", "2" * 64),
            ("tokenizer", "template_sha256", "2" * 64),
            ("condition", "rank", 8),
            ("condition", "adapter_layers", [38]),
        ]
        for section, key, value in changes:
            with self.subTest(field=(section, key)), tempfile.TemporaryDirectory() as tmp:
                kwargs = {section: copy.deepcopy(first[section])}
                kwargs[section][key] = value
                new = identity(**kwargs)
                root = Path(tmp).resolve() / "p"
                with p.EvaluationProgress(
                    root, first, ROWS, verify_context=lambda: p.digest(first)
                ):
                    pass
                with self.assertRaisesRegex(ValueError, "semantic identity"):
                    with p.EvaluationProgress(
                        root, new, ROWS, verify_context=lambda: p.digest(new), resume=True
                    ):
                        pass

    def test_missing_proofs_invalid_layers_and_baseline_loaded_adapter_reject(self):
        for kind in (
            "runtime",
            "zero_b",
            "loaded",
            "layers",
            "bool_rank",
            "preserve",
            "continuity",
        ):
            with self.subTest(kind=kind):
                v = identity()
                kwargs = {}
                if kind == "runtime":
                    v["runtime"]["artifact_manifest_sha256"] = None
                    kwargs = {"runtime": v["runtime"]}
                elif kind == "preserve":
                    v["arithmetic"]["preserve_mode"] = True
                    kwargs = {"arithmetic": v["arithmetic"]}
                elif kind == "continuity":
                    v["base"]["verification_method"] = "historical_full_sha256_with_continuity"
                    kwargs = {"base": v["base"]}
                else:
                    c = v["condition"]
                    kwargs = {"condition": c}
                    if kind == "zero_b":
                        c["zero_b_verification_sha256"] = None
                    elif kind == "loaded":
                        c["adapter_payload_sha256"] = H
                    elif kind == "layers":
                        c["adapter_layers"] = [40]
                    else:
                        c["rank"] = True
                with self.assertRaises(ValueError):
                    identity(**kwargs)

    def test_nonzero_adapter_requires_config_payload_step(self):
        c = identity()["condition"]
        c.update(
            kind="adapter",
            a_initialization_sha256=None,
            zero_b_verification_sha256=None,
            adapter_config_sha256=H,
            adapter_payload_sha256=H,
            checkpoint_step=4,
        )
        self.assertEqual(identity(condition=c)["condition"]["checkpoint_step"], 4)
        c["adapter_config_sha256"] = None
        with self.assertRaises(ValueError):
            identity(condition=c)

    def test_selection_order_mask_and_eos_are_bound(self):
        for change in ("order", "mask", "eos", "vocab", "empty", "context"):
            with self.subTest(change=change):
                rows = copy.deepcopy(ROWS)
                if change == "order":
                    rows.reverse()
                    rows[0], rows[1] = rows[1], rows[0]
                elif change == "mask":
                    rows[0]["loss_start"] = 1
                elif change == "eos":
                    rows[0]["input_ids"][-1] = 2
                elif change == "vocab":
                    rows[0]["input_ids"][0] = 10
                elif change == "empty":
                    rows = []
                else:
                    rows[0]["input_ids"] = [0] * 17 + [1]
                if change in ("order", "mask"):
                    self.assertNotEqual(identity(rows)["selection"], identity()["selection"])
                else:
                    with self.assertRaises(ValueError):
                        identity(rows)


if __name__ == "__main__":
    unittest.main()
