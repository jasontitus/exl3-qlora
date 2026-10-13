"""Launcher lifecycle tests with fake Popen; never starts a child or GPU work."""

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "control_launcher", Path(__file__).parents[1] / "scripts/run_matched_control.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    auditor = launcher.load_auditor()
    fake_path = tmp_path / "auditor.py"
    fake_path.write_text("# fake only\n")
    fake = SimpleNamespace(
        read=auditor.read,
        digest=auditor.digest,
        require=auditor.require,
        verify_plan=lambda _: (SimpleNamespace(read_checkpoint=lambda _: None), {}),
        __file__=str(fake_path),
    )
    monkeypatch.setattr(launcher, "load_auditor", lambda: fake)
    monkeypatch.setattr(launcher, "verify_runtime_versions", lambda *_: {"fixture": True})
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "snapshot.json").write_text("{}\n")
    config = tmp_path / "config.json"
    output = tmp_path / "train-output"
    config.write_text(json.dumps({"output": str(output)}))
    post = tmp_path / "post.json"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "config": str(config),
                "runtime": str(runtime),
                "launch": {
                    "cwd": str(tmp_path),
                    "environment": {"PYTHONPATH": str(runtime / "src")},
                    "train_argv": ["train"],
                    "post_argv": ["evaluate"],
                    "post_report": str(post),
                },
            }
        )
    )
    state = {"calls": [], "train_exit": 0, "mutate": False, "stop_signal": None, "kills": []}

    class Process:
        def __init__(self, argv, *, cwd, env, stdout, stderr, start_new_session):
            state["calls"].append(argv)
            self.pid = len(state["calls"]) + 100
            self.returncode = None
            self.argv = argv
            state["process"] = self
            assert cwd == tmp_path and env["PYTHONPATH"] == str(runtime / "src")
            assert start_new_session is True
            stdout.write("fake stdout\n")
            stderr.write("fake stderr\n")

        def wait(self, **kwargs):
            if self.returncode is not None:
                return self.returncode
            if state["stop_signal"] is not None:
                signum, state["stop_signal"] = state["stop_signal"], None
                signal.getsignal(signum)(signum, None)
            if self.argv == ["train"]:
                self.returncode = state["train_exit"]
                if not self.returncode:
                    output.mkdir()
                    (output / "training.json").write_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "completed_step": 1,
                                "checkpoint": str(output / "step-000001"),
                            }
                        )
                    )
                if state["mutate"]:
                    config.write_text("{}\n")
            else:
                self.returncode = 0
                post.write_text("{}\n")
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)

    def killpg(pid, signum):
        state["kills"].append((pid, signum))
        state["process"].returncode = -signum

    monkeypatch.setattr(launcher.os, "killpg", killpg)
    return plan, tmp_path / "launch.json", state


def test_verify_only_creates_no_evidence_or_subprocess(setup):
    plan, receipt, state = setup
    result = launcher.launch(plan, receipt, verify_only=True)
    assert result["no_subprocesses_started"] is True
    assert state["calls"] == [] and not receipt.exists()


def test_serial_success_records_both_processes_and_before_after_hashes(setup):
    plan, receipt, state = setup
    result = launcher.launch(plan, receipt)
    assert state["calls"] == [["train"], ["evaluate"]]
    assert result["status"] == "completed_pending_cpu_audit"
    assert all(key in result for key in ("before_train", "after_train", "after_evaluate"))
    assert result["phases"]["train"]["process_id"] != result["phases"]["evaluate"]["process_id"]
    assert all(len(p["log_sha256"]) == 2 for p in result["phases"].values())
    assert json.loads(receipt.read_text()) == result


def test_train_failure_preserved_and_evaluation_never_started(setup):
    plan, receipt, state = setup
    state["train_exit"] = 7
    with pytest.raises(RuntimeError, match="status 7"):
        launcher.launch(plan, receipt)
    assert state["calls"] == [["train"]]
    report = json.loads(receipt.read_text())
    assert report["status"] == "failed"
    assert report["phases"]["train"]["exit_code"] == 7


def test_changed_config_after_train_prevents_evaluation(setup):
    plan, receipt, state = setup
    state["mutate"] = True
    with pytest.raises(ValueError, match="Launch input changed"):
        launcher.launch(plan, receipt)
    assert state["calls"] == [["train"]]
    assert json.loads(receipt.read_text())["status"] == "failed"


@pytest.mark.parametrize("kind", ["receipt", "stdout", "stderr", "output", "post"])
def test_existing_outputs_never_overwritten(setup, kind):
    plan, receipt, state = setup
    data = json.loads(plan.read_text())
    paths = {
        "receipt": receipt,
        "stdout": receipt.with_name("launch-train.stdout.log"),
        "stderr": receipt.with_name("launch-train.stderr.log"),
        "output": Path(json.loads(Path(data["config"]).read_text())["output"]),
        "post": Path(data["launch"]["post_report"]),
    }
    paths[kind].write_text("sentinel")
    with pytest.raises(ValueError, match="existing output"):
        launcher.launch(plan, receipt)
    assert paths[kind].read_text() == "sentinel" and state["calls"] == []


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signals_terminate_child_group_preserve_receipt_and_restore_handlers(setup, signum):
    plan, receipt, state = setup
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    state["stop_signal"] = signum
    with pytest.raises(launcher.LaunchInterrupted, match=signal.Signals(signum).name):
        launcher.launch(plan, receipt)
    assert state["calls"] == [["train"]]
    assert state["kills"] == [(101, signal.SIGTERM)]
    report = json.loads(receipt.read_text())
    assert report["status"] == "failed"
    assert report["phases"]["train"]["exit_code"] == -signal.SIGTERM
    assert "LaunchInterrupted" in report["error"]
    assert previous == {s: signal.getsignal(s) for s in previous}


@pytest.mark.parametrize("mismatch", [None, "python", "dependency", "executable"])
def test_installed_runtime_versions_checked_without_importing_model(
    tmp_path, monkeypatch, mismatch
):
    auditor = launcher.load_auditor()
    post = tmp_path / "bf16-all-layers-post.json"
    post.write_text(
        json.dumps(
            {
                "evaluation_context": {
                    "implementation": {
                        "python": "3.12.13",
                        "runtime": {"fixture-dependency": "1.2.3"},
                    }
                }
            }
        )
    )
    plan = {
        "bound_inputs_sha256": {str(post): "unused"},
        "launch": {
            "train_argv": [sys.executable],
            "post_argv": [sys.executable],
        },
    }
    monkeypatch.setattr(
        launcher.platform, "python_version", lambda: "3.99.0" if mismatch == "python" else "3.12.13"
    )
    monkeypatch.setattr(
        launcher.importlib.metadata,
        "version",
        lambda _: "0.0.0" if mismatch == "dependency" else "1.2.3",
    )
    if mismatch == "executable":
        plan["launch"]["post_argv"] = ["/some/other/python"]
    if mismatch:
        with pytest.raises(ValueError, match="differs|versions differ|exact planned"):
            launcher.verify_runtime_versions(auditor, plan)
    else:
        result = launcher.verify_runtime_versions(auditor, plan)
        assert result["runtime"] == {"fixture-dependency": "1.2.3"}
