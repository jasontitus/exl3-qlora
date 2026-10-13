"""CPU lifecycle mocks: no child process, model or GPU is launched."""

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "remote-m5/run_m5_plan.py"
spec = importlib.util.spec_from_file_location("agent_corpus_launcher", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    for key in list(runner.os.environ):
        if key.startswith("MLX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    dump(runtime / "snapshot.json", {})
    helper = tmp_path / "helper.py"
    helper.write_text("# bound test helper only\n")
    project = tmp_path / "project"
    folder = project / "experiment"
    folder.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"test weights")
    dump(project / "project.json", {"status": "prepared", "model": str(model)})
    payload = tmp_path / "payload.json"
    dump(payload, {"schema": "checkpoint-full-payload-sha256-v1", "root": str(model), "started_utc": "2026-10-13T21:00:00+00:00", "completed_utc": "2026-10-13T21:00:01+00:00", "files": {"weights.bin": {"size": 12, "sha256": runner.digest(model / "weights.bin")}}, "total_bytes": 12, "verified_manifest_sha256": "a" * 64})
    plan_path = tmp_path / "plan.json"
    plan = {
        "format": "native-agent-kernel-run-plan-v1",
        "helper": str(helper),
        "helper_sha256": runner.digest(helper),
        "runtime": str(runtime),
        "snapshot_sha256": runner.digest(runtime / "snapshot.json"),
        "python": str(Path(sys.executable).absolute()),
        "cwd": str(runtime),
        "environment": {"PYTHONPATH": str(runtime / "src")},
        "project": str(project),
        "initial_project_sha256": runner.digest(project / "project.json"),
        "implementation": {"python": runner.platform.python_version(), "runtime": {}},
        "order": list(runner.PHASES),
        "config": {"max_steps": 4, "prefix_disk_cache": str(project / "new-cache")},
    }
    plan["commands"] = {
        p: [plan["python"], "-m", "exl3_mlx.workflow", p]
        + (["<FIXED_FINAL_CHECKPOINT>"] if p == "adapter" else [])
        for p in runner.PHASES
    }
    dump(plan_path, plan)
    fake_helper = SimpleNamespace(
        verify_runtime=lambda _: (runtime, runner.digest(runtime / "snapshot.json"))
    )
    monkeypatch.setattr(runner, "load_helper", lambda _: fake_helper)
    monkeypatch.setattr(runner, "observed_children", lambda _: {})
    state = {
        "payload": payload,
        "calls": [],
        "failure": None,
        "mutate": None,
        "bad_emit": None,
        "bad_verify": None,
        "signal": None,
        "killed": [],
        "artifacts": {},
    }

    class Process:
        def __init__(self, argv, *, cwd, env, stdout, stderr, shell, start_new_session, pass_fds):
            assert shell is False and start_new_session is True
            assert {k: v for k, v in env.items() if k.startswith("MLX_")} == {"MLX_ENABLE_TF32": "0"}
            assert cwd == str(runtime) and env["PYTHONPATH"] == str(runtime / "src")
            self.argv, self.stdout, self.returncode = argv, stdout, None
            self.pid = 100 + len(state["calls"])
            state["calls"].append(argv)
            state["process"] = self
            if argv[1] == str(helper):
                self.kind, self.phase = argv[2], argv[-1]
            else:
                self.kind, self.phase = "model", argv[3]
            self.key = (self.kind, self.phase)
            if self.kind == "model" and self.phase != "train":
                assert len(pass_fds) == 1
            else:
                assert not pass_fds

        def wait(self, **kwargs):
            if self.returncode is not None:
                return self.returncode
            if state["signal"]:
                sig, state["signal"] = state["signal"], None
                signal.getsignal(sig)(sig, None)
            if self.key == state["failure"]:
                self.returncode = 7
                return 7
            self.returncode = 0
            if self.kind == "command":
                command = list(plan["commands"][self.phase])
                if self.phase == "adapter":
                    command[-1] = str(project / "attempts/one/train/step-000004")
                if state["bad_emit"] == self.phase:
                    command.append("--unplanned")
                value = {
                    "argv": command,
                    "phase": self.phase,
                    "cwd": plan["cwd"],
                    "environment": plan["environment"],
                    "plan": str(plan_path),
                    "plan_sha256": runner.digest(plan_path),
                }
                self.stdout.write(json.dumps(value))
            elif self.kind == "model":
                if self.phase == "train":
                    (project / "attempts/one/train/step-000004").mkdir(parents=True)
                    dump(project / "project.json", {"status": "completed"})
                path = folder / (self.phase + ".json")
                dump(path, {"evaluation_context": {"process_id": self.pid}})
                state["artifacts"][self.phase] = {str(path): runner.digest(path)}
            else:
                value = {
                    "phase": self.phase,
                    "verified_complete": True,
                    "plan_sha256": runner.digest(plan_path),
                    "artifacts_sha256": state["artifacts"][self.phase],
                }
                if state["bad_verify"] == self.phase:
                    value["verified_complete"] = False
                self.stdout.write(json.dumps(value))
            if state.get("mutate_env") == self.key:
                runner.os.environ["MLX_ENABLE_TF32"] = "1"
            if state["mutate"] == self.key:
                helper.write_text("# drift\n")
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner, "process_table", lambda: {})

    def killpg(pid, sig):
        state["killed"].append((pid, sig))
        state["process"].returncode = -sig

    monkeypatch.setattr(runner.os, "killpg", killpg)
    return plan_path, tmp_path / "launch.json", plan, state


def launch(fixture, **kwargs):
    path, receipt, _, state = fixture
    return runner.launch(path, receipt, expected_sha256=runner.digest(path), payload_receipt=state["payload"], payload_sha256=runner.digest(state["payload"]), **kwargs)


def test_verify_only_never_starts_process_or_claims_project(fixture):
    _, receipt, plan, state = fixture
    result = launch(fixture, verify_only=True)
    assert result["no_subprocesses_started"] is True
    assert not receipt.exists() and not state["calls"]
    assert not (Path(plan["project"]) / ".agent-corpus-launch.json").exists()


def test_all_five_phases_serial_emission_execution_verification(fixture):
    _, receipt, _, state = fixture
    result = launch(fixture)
    assert len(state["calls"]) == 15
    assert list(result["verified_phases"]) == list(runner.PHASES)
    assert result["status"] == "completed_pending_quality_audit"
    assert result["quality_certified"] is False
    for i, phase in enumerate(runner.PHASES):
        assert state["calls"][3 * i][2] == "command"
        assert state["calls"][3 * i + 1][3] == phase
        assert state["calls"][3 * i + 2][2] == "verify"
    assert all(len(p["log_sha256"]) == 2 for p in result["processes"].values())
    assert json.loads(receipt.read_text()) == result


@pytest.mark.parametrize("kind", ["command", "model", "verify"])
def test_nonzero_stops_without_later_phase(fixture, kind):
    _, receipt, _, state = fixture
    state["failure"] = (kind, "preflight-test")
    with pytest.raises(ValueError, match="exited 7"):
        launch(fixture)
    assert not any("baseline" in c for c in state["calls"])
    assert runner.read(receipt)["status"] == "failed"


@pytest.mark.parametrize("failure", ["bad_emit", "bad_verify", "mutate"])
def test_invalid_emission_or_receipt_or_source_drift_stops(fixture, failure):
    _, receipt, _, state = fixture
    state[failure] = ("model", "preflight-train") if failure == "mutate" else "preflight-train"
    with pytest.raises(ValueError):
        launch(fixture)
    assert not any("preflight-test" in c for c in state["calls"])
    assert runner.read(receipt)["status"] == "failed"


@pytest.mark.parametrize("kind", ["receipt", "claim", "attempts", "preflight", "log"])
def test_existing_or_uncertain_outputs_never_reused(fixture, kind):
    _, receipt, plan, state = fixture
    paths = {
        "receipt": receipt,
        "claim": Path(plan["project"]) / ".agent-corpus-launch.json",
        "attempts": Path(plan["project"]) / "attempts",
        "preflight": Path(plan["project"]) / "experiment/preflight-train",
        "log": receipt.with_name("launch-baseline.stdout.log"),
    }
    paths[kind].write_text("sentinel")
    with pytest.raises(ValueError, match="Existing"):
        launch(fixture)
    assert not state["calls"] and paths[kind].read_text() == "sentinel"


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_cleans_child_preserves_failed_receipt_and_handlers(fixture, sig):
    _, receipt, _, state = fixture
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    state["signal"] = sig
    with pytest.raises(runner.LaunchInterrupted):
        launch(fixture)
    assert len(state["calls"]) == 1 and state["killed"] == [(100, signal.SIGTERM)]
    assert runner.read(receipt)["status"] == "failed"
    assert previous == {s: signal.getsignal(s) for s in previous}


def test_plan_pin_and_initial_project_hash_enforced(fixture):
    path, receipt, plan, state = fixture
    with pytest.raises(ValueError, match="pinned SHA256"):
        runner.launch(path, receipt, expected_sha256="0" * 64, payload_receipt=state["payload"], payload_sha256=runner.digest(state["payload"]))
    dump(Path(plan["project"]) / "project.json", {"changed": True, "model": str(Path(plan["project"]).parent / "model")})
    with pytest.raises(ValueError, match="Project changed"):
        launch(fixture)
    assert not state["calls"]


def test_receipt_pid_must_match_actual_launched_process(tmp_path):
    folder = tmp_path / "experiment"
    path = folder / "baseline.json"
    dump(path, {"evaluation_context": {"process_id": 123}})
    value = {
        "phase": "baseline",
        "verified_complete": True,
        "plan_sha256": "p",
        "artifacts_sha256": {str(path): runner.digest(path)},
    }
    with pytest.raises(ValueError, match="PID"):
        runner.verify_completion(value, "baseline", {"project": str(tmp_path)}, "p", 456)


@pytest.mark.parametrize("mode", ["missing", "enabled", "other_mlx"])
def test_numerical_environment_rejected_before_any_child(fixture, monkeypatch, mode):
    if mode == "missing": monkeypatch.delenv("MLX_ENABLE_TF32")
    elif mode == "enabled": monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    else: monkeypatch.setenv("MLX_OTHER_TEST", "1")
    with pytest.raises(ValueError, match="MLX_ENABLE_TF32"):
        launch(fixture)
    assert not fixture[3]["calls"]


def test_numeric_env_bound_on_all_helper_and_model_receipts(fixture):
    result = launch(fixture)
    assert result["numerical_environment"] == {"MLX_ENABLE_TF32": "0"}
    assert result["cache_isolation"]["initially_absent"] is True
    for process in result["processes"].values():
        assert process["environment"]["MLX_ENABLE_TF32"] == "0"


@pytest.mark.parametrize("kind", ["existing", "outside"])
def test_cache_is_fresh_project_owned(fixture, tmp_path, kind):
    path, _, plan, state = fixture
    if kind == "existing": Path(plan["config"]["prefix_disk_cache"]).mkdir()
    else:
        plan["config"]["prefix_disk_cache"] = str(tmp_path / "outside")
        dump(path, plan)
    with pytest.raises(ValueError, match="project-owned"):
        launch(fixture)
    assert not state["calls"]


def test_model_root_mismatch_rejected(fixture, tmp_path):
    payload = fixture[3]["payload"]; value = runner.read(payload); value["root"] = str(tmp_path / "another-model"); dump(payload, value)
    with pytest.raises(ValueError, match="checkpoint differs"):
        launch(fixture)
    assert not fixture[3]["calls"]


@pytest.mark.parametrize("mutation", ["files_bool", "files_list", "bad_digest", "missing_completed", "bad_total", "bad_size", "bad_file_sha", "escaped_path", "source_side_only"])
def test_malformed_payload_receipt_never_launches(fixture, mutation):
    payload = fixture[3]["payload"]; value = runner.read(payload)
    if mutation == "files_bool": value["files"] = True
    elif mutation == "files_list": value["files"] = ["weights.bin"]
    elif mutation == "bad_digest": value["verified_manifest_sha256"] = "not-a-hash"
    elif mutation == "missing_completed": del value["completed_utc"]
    elif mutation == "bad_total": value["total_bytes"] = 99
    elif mutation == "bad_size": value["files"]["weights.bin"]["size"] = True
    elif mutation == "bad_file_sha": value["files"]["weights.bin"]["sha256"] = "z" * 64
    elif mutation == "escaped_path": value["files"] = {"../weights.bin": value["files"]["weights.bin"]}
    else: value["verified_manifest_sha256"] = None
    dump(payload, value)
    with pytest.raises(ValueError):
        launch(fixture, verify_only=True)
    assert not fixture[3]["calls"]


def test_payload_receipt_hash_is_mandatory(fixture):
    path, receipt, _, state = fixture
    with pytest.raises(ValueError, match="receipt drift"):
        runner.launch(path, receipt, expected_sha256=runner.digest(path), payload_receipt=state["payload"], payload_sha256="0" * 64, verify_only=True)
    assert not state["calls"]


@pytest.mark.parametrize("kind", ["naive", "reversed", "relative_root", "zero_total"])
def test_payload_completion_and_root_shape(fixture, kind):
    payload = fixture[3]["payload"]; value = runner.read(payload)
    if kind == "naive": value["completed_utc"] = "2026-10-13T21:00:01"
    elif kind == "reversed": value["completed_utc"] = "2026-10-13T20:59:59+00:00"
    elif kind == "relative_root": value["root"] = "relative/model"
    else: value["total_bytes"] = 0
    dump(payload, value)
    with pytest.raises(ValueError): launch(fixture, verify_only=True)
    assert not fixture[3]["calls"]


def test_numerical_environment_drift_after_emission_blocks_model(fixture):
    fixture[3]["mutate_env"] = ("command", "preflight-train")
    with pytest.raises(ValueError, match="MLX_ENABLE_TF32"):
        launch(fixture)
    assert len(fixture[3]["calls"]) == 1
    assert runner.read(fixture[1])["status"] == "failed"
