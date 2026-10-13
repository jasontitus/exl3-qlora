"""CPU-only local temporary ownership fixtures, never the real shared GPU path."""

import importlib.util
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

P = Path(__file__).resolve().parents[1] / "shared_gpu_lease_v2.py"
s = importlib.util.spec_from_file_location("lease_test", P)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.lock = self.root / "sushi-gpu.lock.d"
        self.owner = "ktune-engine-" + str(os.getpid()) + "-" + uuid.uuid4().hex
        self.state = {
            "owner": self.owner,
            "controller_pid": os.getpid(),
            "lock_dir": str(self.lock),
            "claim": str(self.root / (self.lock.name + ".holder." + uuid.uuid4().hex)),
            "recovery_receipt": str(
                self.root / (self.lock.name + "." + self.owner + ".json")
            ),
        }

    def claim(self):
        return m.claim_locked(self.state, lock=self.lock)

    def test_roundtrip_exact_own_protocol(self):
        s = self.claim()
        self.assertEqual(os.readlink(self.lock), s["claim"])
        self.assertEqual((self.lock / "owner").read_text().split()[0], s["owner"])
        self.assertEqual(m.queue_entries(self.lock)[0][2], "done")
        s["cleanup_verified"] = True
        m.release_locked(s, lock=self.lock)
        self.assertFalse(m.exists(self.lock))
        self.assertFalse(m.read(s["recovery_receipt"])["recovery_required"])

    def test_foreign_owner_never_removed(self):
        foreign = self.root / "foreign"
        foreign.mkdir()
        (foreign / "owner").write_text("other-owner time\n")
        os.symlink(str(foreign), self.lock)
        with self.assertRaises(ValueError):
            self.claim()
        self.assertEqual((foreign / "owner").read_text(), "other-owner time\n")
        self.assertTrue(self.lock.is_symlink())

    def test_dangling_link_blocks(self):
        os.symlink(str(self.root / "missing"), self.lock)
        with self.assertRaises(ValueError):
            self.claim()
        self.assertTrue(self.lock.is_symlink())

    def test_legacy_ownerless_directory_blocks(self):
        self.lock.mkdir()
        with self.assertRaises(ValueError):
            self.claim()
        self.assertTrue(self.lock.is_dir())

    def test_pending_ticket_blocks_even_unknown_dead_pid(self):
        q = Path(str(self.lock) + ".queue")
        q.mkdir()
        os.symlink("999999 old-owner", q / "2")
        with self.assertRaises(ValueError):
            self.claim()
        self.assertEqual(os.readlink(q / "2"), "999999 old-owner")

    def test_malformed_ticket_blocks_and_is_preserved(self):
        q = Path(str(self.lock) + ".queue")
        q.mkdir()
        (q / "2").write_text("not symlink")
        with self.assertRaises(ValueError):
            self.claim()
        self.assertEqual((q / "2").read_text(), "not symlink")

    def test_unresolved_recovery_blocks(self):
        p = self.root / (self.lock.name + ".ktune-engine-42-x.json")
        p.write_text(json.dumps({"recovery_required": True}))
        with self.assertRaises(ValueError):
            self.claim()
        self.assertTrue(p.exists())

    def test_bad_recovery_json_blocks(self):
        p = self.root / (self.lock.name + ".ktune-engine-42-x.json")
        p.write_text("{")
        with self.assertRaises(ValueError):
            self.claim()

    def test_completed_recovery_and_done_ticket_allowed(self):
        p = self.root / (self.lock.name + ".ktune-engine-42-x.json")
        p.write_text(json.dumps({"recovery_required": False}))
        q = Path(str(self.lock) + ".queue")
        q.mkdir()
        os.symlink("done", q / "9")
        self.claim()
        self.assertEqual(
            [v for _, _, v in m.queue_entries(self.lock)], ["done", "done"]
        )
        self.assertTrue((q / "10").is_symlink())

    def test_no_cleanup_proof_no_release(self):
        s = self.claim()
        with self.assertRaises(ValueError):
            m.release_locked(s, lock=self.lock)
        self.assertTrue(self.lock.is_symlink())

    def test_changed_owner_retained(self):
        s = self.claim()
        s["cleanup_verified"] = True
        (self.lock / "owner").write_text("foreign owner\n")
        with self.assertRaises(ValueError):
            m.release_locked(s, lock=self.lock)
        self.assertEqual((self.lock / "owner").read_text(), "foreign owner\n")

    def test_added_claim_content_retained(self):
        s = self.claim()
        s["cleanup_verified"] = True
        (self.lock / "unknown").write_text("data")
        with self.assertRaises(ValueError):
            m.release_locked(s, lock=self.lock)
        self.assertTrue((self.lock / "unknown").exists())

    def test_changed_link_identity_retained(self):
        s = self.claim()
        s["cleanup_verified"] = True
        self.lock.unlink()
        os.symlink(str(self.root / "foreign"), self.lock)
        with self.assertRaises(ValueError):
            m.release_locked(s, lock=self.lock)
        self.assertEqual(os.readlink(self.lock), str(self.root / "foreign"))

    def test_preexisting_claim_not_deleted(self):
        p = Path(self.state["claim"])
        p.mkdir()
        (p / "owner").write_text(self.owner + " never ours\n")
        with self.assertRaises(ValueError):
            self.claim()
        self.assertTrue((p / "owner").exists())

    def test_record_path_escape_rejected(self):
        self.state["claim"] = str(self.root / "foreign")
        with self.assertRaises(ValueError):
            self.claim()

    def test_process_tracking_retains_reparented_and_new_session_child(self):
        tracked = {10: "old"}
        t = {
            10: {"ppid": 1, "pgid": 10, "start": "old"},
            11: {"ppid": 10, "pgid": 11, "start": "child"},
        }
        self.assertEqual(set(m.update_descendants(t, tracked, 10)), {10, 11})
        self.assertEqual(
            set(
                m.update_descendants(
                    {11: {"ppid": 1, "pgid": 11, "start": "child"}}, tracked, 10
                )
            ),
            {11},
        )
        self.assertFalse(
            m.update_descendants(
                {11: {"ppid": 1, "pgid": 11, "start": "new"}}, tracked, 10
            )
        )

    def test_busy_run_never_starts_command(self):
        self.lock.mkdir()
        req = self.root / "request.json"
        req.write_text("{}")
        receipt = self.root / "receipt.json"
        with (
            patch.object(m, "LOCK", self.lock),
            patch.object(m, "validate_request"),
            patch.object(m, "mutate") as mutate,
            patch.object(m.subprocess, "Popen") as popen,
            self.assertRaises(ValueError),
        ):
            m.run(req, m.sha(req), receipt)
        mutate.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(m.read(receipt)["status"], "blocked")
        self.assertTrue(self.lock.is_dir())

    def test_capture_uses_no_signal_cleanup(self):
        with patch.object(m.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("ok", "")
            popen.return_value.returncode = 0
            self.assertEqual(m.capture_command(["readonly"]), (0, "ok", ""))
            popen.return_value.kill.assert_not_called()
            popen.return_value.terminate.assert_not_called()

    def run_fixture(self, code, *, drift=False, snapshots=None, poll_codes=None, completion_status="complete", expected_status="complete"):
        req = self.root / "request.json"
        complete = self.root / "complete.json"
        executable = self.root / "reviewed-command"
        executable.write_text("reviewed source")
        job = self.root / "job.json"
        job.write_text("{}")
        request = {
            "inputs_sha256": {str(executable): m.sha(executable), str(job): m.sha(job)},
            "argv": [str(executable), str(job), str(complete)],
            "cwd": str(self.root),
            "expected_exit_codes": [0],
            "completion_receipt": str(complete),
            "completion_expect": {"status": expected_status},
        }
        req.write_text(json.dumps(request))
        protocol = self.root / "protocol"
        protocol.write_text("fixture protocol")

        def mutation(action, state_path):
            value = m.read(state_path)
            result = (m.claim_locked if action == "claim" else m.release_locked)(
                value, lock=self.lock
            )
            m.save(state_path, result)
            return result

        def start(*args, **kwargs):
            complete.write_text(json.dumps({"status": completion_status}))
            if drift:
                executable.write_text("changed executable during child")
            child.pid = 77
            child.poll.return_value = code
            if poll_codes is not None:
                child.poll.side_effect = poll_codes
            child.returncode = code
            return child

        from unittest.mock import Mock

        child = Mock()
        with (
            patch.object(m, "LOCK", self.lock),
            patch.object(m, "PROTOCOL_PATH", protocol),
            patch.object(m, "PROTOCOL_SHA256", m.sha(protocol)),
            patch.object(m, "validate_request"),
            patch.object(m, "mutate", side_effect=mutation),
            patch.object(m.subprocess, "Popen", side_effect=start),
            patch.object(
                m,
                "processes",
                side_effect=snapshots if snapshots is not None else [{77: {"ppid": 1, "pgid": 77, "start": "child"}}, {}],
            ),
        ):
            try:
                result = m.run(req, m.sha(req), self.root / "run.json")
            finally:
                child.kill.assert_not_called()
                child.terminate.assert_not_called()
        return result

    def test_successful_reviewed_completion_releases_only_own_lease(self):
        result = self.run_fixture(0)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["lease"]["released"])
        self.assertFalse(m.exists(self.lock))

    def test_failed_command_retains_own_lease_without_signalling(self):
        with self.assertRaisesRegex(ValueError, "Unexpected child exit"):
            self.run_fixture(1)
        self.assertTrue(self.lock.is_symlink())
        receipt = m.read(self.root / "run.json")
        self.assertEqual(receipt["status"], "failed_lease_retained")
        self.assertTrue(receipt["lease"]["recovery_required"])

    def test_hash_guard_rejects_changed_command_or_job(self):
        exe = self.root / "exe"
        job = self.root / "job.json"
        exe.write_bytes(b"expected executable")
        job.write_text("{}")
        request = {
            "argv": [str(exe), str(job)],
            "inputs_sha256": {str(exe): m.sha(exe), str(job): m.sha(job)},
        }
        m.verify_inputs(request)
        job.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "drift"):
            m.verify_inputs(request)
        del request["inputs_sha256"][str(job)]
        with self.assertRaisesRegex(ValueError, "file argument"):
            m.verify_inputs(request)

    def test_unpinned_executable_and_stale_observed_harness_rejected(self):
        exe = self.root / "exe"
        exe.write_bytes(b"expected executable")
        request = {
            "argv": [str(exe)],
            "inputs_sha256": {str(exe): m.sha(exe)},
            "observed_harness_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "Harness expectation"):
            m.verify_inputs(request)
        request.pop("observed_harness_sha256")
        request["inputs_sha256"] = {}
        with self.assertRaisesRegex(ValueError, "hashes required"):
            m.verify_inputs(request)

    def test_post_command_input_drift_retains_own_lease(self):
        with self.assertRaisesRegex(ValueError, "Command/input drift"):
            self.run_fixture(0, drift=True)
        self.assertTrue(self.lock.is_symlink())
        receipt = m.read(self.root / "run.json")
        self.assertEqual(receipt["status"], "failed_lease_retained")
        self.assertTrue(receipt["lease"]["recovery_required"])


    @staticmethod
    def row(parent, group, start):
        return {"ppid": parent, "pgid": group, "start": start}

    def test_new_owned_generation_replaces_expired_descendant(self):
        tracked = {10: "root", 11: "old"}
        history = {}
        table = {12: self.row(11, 12, "grandchild"), 11: self.row(10, 11, "new"), 10: self.row(1, 10, "root")}
        self.assertEqual(set(m.update_descendants(table, tracked, 10, history)), {10, 11, 12})
        self.assertEqual(history[11], ["old", "new"])
        self.assertEqual(tracked[11], "new")
        self.assertEqual(set(m.update_descendants({11: self.row(1, 11, "new"), 12: self.row(11, 12, "grandchild")}, tracked, 10, history)), {11, 12})

    def test_unrelated_reused_descendant_not_adopted(self):
        tracked = {10: "root", 11: "old"}
        live = m.update_descendants({10: self.row(1, 10, "root"), 11: self.row(1, 11, "foreign"), 12: self.row(11, 12, "foreign-child")}, tracked, 10)
        self.assertEqual(set(live), {10})
        self.assertEqual(tracked[11], "old")

    def test_reused_root_is_uncertain_even_if_new_root_has_same_group(self):
        for parent, group in [(1, 10), (2, 20)]:
            with self.subTest(parent=parent, group=group), self.assertRaisesRegex(ValueError, "Leader PID generation"):
                m.update_descendants({10: self.row(parent, group, "foreign")}, {10: "root"}, 10)

    def test_live_parent_group_allows_reused_owned_group_member(self):
        tracked = {10: "root", 11: "old"}
        self.assertEqual(set(m.update_descendants({10: self.row(1, 10, "root"), 11: self.row(1, 10, "new")}, tracked, 10)), {10, 11})

    def test_dead_leader_group_alone_is_uncertain_not_proven_foreign(self):
        tracked = {10: "root", 11: "old"}
        with self.assertRaisesRegex(ValueError, "Uncertain late leader-group"):
            m.update_descendants({11: self.row(1, 10, "new")}, tracked, 10)
        self.assertEqual(tracked[11], "old")

    def test_live_generation_and_reparented_children_remain_tracked(self):
        tracked = {10: "root", 11: "child"}
        self.assertEqual(set(m.update_descendants({11: self.row(1, 11, "child"), 12: self.row(11, 12, "grandchild")}, tracked, 10)), {11, 12})

    def test_missing_or_empty_birth_identity_fails_closed(self):
        for value in [None, "", "  "]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "generation metadata"):
                m.update_descendants({11: self.row(10, 11, value)}, {10: "root"}, 10)

    def test_full_serial_five_phase_nested_sessions_release_after_completion(self):
        root = {77: self.row(1, 77, "child")}
        tables = []
        for phase in range(5):
            for role in range(3):
                pid = 100 + phase * 3 + role
                tables.append({**root, pid: self.row(77, pid, str(pid)), 200+pid: self.row(pid, 200+pid, str(200+pid))})
        result = self.run_fixture(0, snapshots=tables+[root, {}], poll_codes=[None]*15+[0], completion_status="completed_pending_quality_audit", expected_status="completed_pending_quality_audit")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["observed_descendants"]), 31)
        self.assertTrue(result["lease"]["released"])

    def test_reused_owned_survivor_retains_lease_after_launcher_exit(self):
        root = {77: self.row(1, 77, "child")}
        tables = [{**root, 88:self.row(77,88,"old")}, {**root, 88:self.row(77,88,"new")}, {88:self.row(1,88,"new")}]
        with self.assertRaisesRegex(ValueError, "descendants remain"):
            self.run_fixture(0, snapshots=tables, poll_codes=[None,0])
        result = m.read(self.root / "run.json")
        self.assertEqual(result["observed_descendant_generations"]["88"], ["old", "new"])
        self.assertTrue(result["lease"]["recovery_required"])
        self.assertTrue(self.lock.is_symlink())

    def test_reused_root_retains_lease_without_foreign_signalling(self):
        tables = [{77:self.row(1,77,"child")}, {77:self.row(1,77,"foreign")}]
        with self.assertRaisesRegex(ValueError, "Leader PID generation"):
            self.run_fixture(0, snapshots=tables)
        self.assertTrue(self.lock.is_symlink())

    def test_incomplete_training_completion_keeps_lease_even_exit_zero(self):
        with self.assertRaisesRegex(ValueError, "Completion criteria"):
            self.run_fixture(0, completion_status="verified_only", expected_status="completed_pending_quality_audit")
        self.assertTrue(self.lock.is_symlink())


    def test_new_same_group_child_after_leader_exit_retains_lease(self):
        tables = [{77: self.row(1, 77, "child")}, {88: self.row(1, 77, "late-child")}]
        with self.assertRaisesRegex(ValueError, "Uncertain late leader-group"):
            self.run_fixture(0, snapshots=tables)
        result = m.read(self.root / "run.json")
        self.assertEqual(result["status"], "failed_lease_retained")
        self.assertTrue(result["lease"]["recovery_required"])
        self.assertTrue(self.lock.is_symlink())

    def test_late_same_group_child_has_live_owned_parent_and_is_tracked(self):
        tracked = {10: "root", 11: "child"}
        table = {11:self.row(1,11,"child"), 12:self.row(11,10,"new-child")}
        self.assertEqual(set(m.update_descendants(table, tracked, 10)), {11,12})


    def test_zero_system_inventory_ids_do_not_become_owned(self):
        tracked = {10: "root"}
        table = {0: self.row(0,0,"kernel"), 1:self.row(0,0,"system"), 10:self.row(1,10,"root"), 11:self.row(10,11,"child")}
        self.assertEqual(set(m.update_descendants(table, tracked, 10)), {10,11})
        self.assertNotIn(0, tracked)
        self.assertNotIn(1, tracked)
