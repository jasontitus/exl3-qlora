"""CPU-only Studio boundary tests; no sockets, browser, model, or GPU jobs."""

import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from exl3_mlx import studio
from exl3_mlx.sft import SFTConfig, SFTTrainer


class FakeSocket:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, *_):
        return self.input

    def sendall(self, value):
        self.output.extend(value)


def request(
    monkeypatch, state, method, path, *, token=None, origin=None, host=None, body=b"", extra=None
):
    """Exercise the actual HTTP handler over in-memory streams."""
    server = SimpleNamespace(server_port=32109)

    def factory(address, handler):
        assert address == ("127.0.0.1", 0)
        server.handler = handler
        return server

    monkeypatch.setattr(studio, "ThreadingHTTPServer", factory)
    assert studio.make_server(state) is server
    headers = {"Host": host or "127.0.0.1:32109", "Content-Length": str(len(body))}
    if token is not None:
        headers["X-Studio-Token"] = token
    if origin is not None:
        headers["Origin"] = origin
    headers.update(extra or {})
    wire = f"{method} {path} HTTP/1.0\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
    connection = FakeSocket(wire.encode() + b"\r\n" + body)
    server.handler(connection, ("127.0.0.1", 50001), server)
    head, response = bytes(connection.output).split(b"\r\n\r\n", 1)
    return int(head.split(b" ")[1]), head, response


@pytest.fixture
def state(tmp_path):
    return studio.StudioState(tmp_path / "studio")


def project(path, *, comparison=None):
    path.mkdir()
    (path / "project.json").write_text(
        json.dumps(
            {
                "format": "exl3-sft-project-v1",
                "status": "prepared",
                "config": {},
                "data": {"counts": {"train": 1, "valid": 0, "test": 0}},
            }
        )
    )
    if comparison is not None:
        (path / "comparison.json").write_text(json.dumps(comparison))
    return path


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"token": "wrong"},
        {"token": "valid", "origin": "http://example.com"},
        {"token": "valid", "host": "example.com:32109"},
    ],
)
def test_api_rejects_missing_token_wrong_token_foreign_origin_and_host(
    monkeypatch, state, overrides
):
    if overrides.get("token") == "valid":
        overrides = {**overrides, "token": state.token}
    status, _, _ = request(monkeypatch, state, "GET", "/api/state", **overrides)
    assert status == 403


def test_authenticated_state_is_no_store_and_cpu_only(monkeypatch, state):
    status, headers, body = request(monkeypatch, state, "GET", "/api/state", token=state.token)
    assert status == 200
    assert b"Cache-Control: no-store" in headers
    assert json.loads(body)["running"] is False
    assert state.process is None


def test_authorized_upload_stays_inside_studio_directory(monkeypatch, state):
    payload = b'{"prompt":"Q","completion":"A"}\n'
    status, _, body = request(
        monkeypatch,
        state,
        "POST",
        "/api/upload",
        token=state.token,
        body=payload,
        extra={"X-Filename": "../../outside.jsonl"},
    )
    assert status == 200
    uploaded = Path(json.loads(body)["path"])
    assert uploaded.parent == state.directory
    assert uploaded.suffix == ".jsonl" and uploaded.read_bytes() == payload


def test_upload_cap_and_extension_rejected_without_writes(monkeypatch, state):
    for extra in ({"Content-Length": str(studio.MAX_UPLOAD + 1)}, {"X-Filename": "data.py"}):
        status, _, _ = request(
            monkeypatch,
            state,
            "POST",
            "/api/upload",
            token=state.token,
            body=b"x",
            extra=extra,
        )
        assert status == 400
    assert not list(state.directory.glob("upload-*"))


def test_training_commands_are_argument_vectors_and_duplicate_start_is_blocked(
    monkeypatch, state, tmp_path
):
    state.project = project(tmp_path / "project")
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=1234, poll=lambda: None)

    monkeypatch.setattr(studio.subprocess, "Popen", popen)
    state.action({"action": "train", "steps": 2})
    command = calls[0][0]
    start = command.index("resume")
    assert command[start : start + 5] == [
        "resume",
        str(state.project),
        "--steps",
        "2",
        "--no-compare",
    ]
    request_started = int(command[command.index("--request-start-ns") + 1])
    assert request_started > 0
    assert "shell" not in calls[0][1]
    with pytest.raises(ValueError, match="already running"):
        state.action({"action": "train", "steps": 2})
    assert len(calls) == 1
    state.action({"action": "stop"})
    assert (state.project / "stop.request").exists()
    assert (state.project / "stop.request").stat().st_mtime_ns >= request_started


def test_opening_project_without_comparison_drops_previous_snapshot(monkeypatch, state, tmp_path):
    a = project(tmp_path / "a", comparison={"adapter": "A", "status": "likelihood_improved"})
    b = project(tmp_path / "b")
    state.action({"action": "open", "project": str(a)})
    assert state.snapshot()["comparison"]["adapter"] == "A"
    state.action({"action": "open", "project": str(b)})
    snapshot = state.snapshot()
    assert snapshot["project"] == str(b)
    assert "comparison" not in snapshot
    assert snapshot["job"] is None and snapshot["log"] == ""


def test_snapshot_never_tails_an_active_path_outside_the_project(state, tmp_path):
    state.project = project(tmp_path / "project")
    outside = tmp_path / "outside.log"
    outside.write_text("NOT A PROJECT LOG")
    (state.project / "active.json").write_text(json.dumps({"log": str(outside), "phase": "train"}))
    assert "NOT A PROJECT LOG" not in state.snapshot()["log"]


@pytest.mark.parametrize("request_is_new", [False, True])
def test_stop_during_launch_initialization_survives_but_stale_stop_is_cleared(
    tmp_path, request_is_new
):
    root = project(tmp_path / "project")
    trainer = SFTTrainer(model="/unused-model", args=SFTConfig(output_dir=str(root), max_steps=1))
    trainer.prepare = lambda: None
    trainer._manifest = lambda: {
        "status": "prepared",
        "data": {"seq_length": 8, "counts": {"valid": 0}},
    }
    trainer._recover = lambda: None
    trainer.status = lambda: {"status": "stopped"}
    requested_at = 1_750_000_000_000_000_000
    stop = root / "stop.request"
    stop.touch()
    timestamp = requested_at + (1_000_000 if request_is_new else -1_000_000)
    os.utime(stop, ns=(timestamp, timestamp))
    observed = []

    def child(command, *, label, directory):
        observed.append(stop.exists())
        if label == "training":
            output = Path(directory) / "train"
            output.mkdir()
            (output / "training.json").write_text('{"status":"stopped"}')

    trainer._child = child
    trainer.train(compare=False, request_start_ns=requested_at)
    assert observed == [request_is_new, request_is_new]


def test_browser_script_clears_old_project_results_and_survives_refresh():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the CPU-only JavaScript state test")
    html = Path(studio.__file__).with_name("studio.html").read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const vm = require('node:vm');
const assert = require('node:assert/strict');
const script = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const storage = new Map();
class Element {
  constructor() { this.children=[]; this.value=''; this._text=''; }
  set textContent(value) { this._text=value; this.children=[]; }
  get textContent() { return this._text+this.children.map(c=>c.textContent).join(''); }
  replaceChildren() { this.children=[]; this._text=''; }
  append(...children) { this.children.push(...children); }
}
function page(hash) {
  const nodes=new Map(), calls=[];
  const get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  let snapshot={running:false,project:'A',log:'A log',comparison:{adapter:'A-ADAPTER',status:'likelihood_improved',evaluation_scope:'held_out_test'}};
  const context={
    location:{hash,pathname:'/'},history:{replaceState(){}},
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
    document:{getElementById:get,querySelectorAll:()=>[],createElement:()=>new Element()},
    setInterval(){},
    fetch:async(path,options)=>{calls.push(options.headers);const value=snapshot;return {ok:true,json:async()=>value};}
  };
  vm.createContext(context);vm.runInContext(script,context);
  return {context,get,calls,set:value=>snapshot=value};
}
(async()=>{
  const first=page('#secret');await first.context.poll();
  assert.match(first.get('comparison').textContent,/A-ADAPTER/);
  first.set({running:false,project:'B',log:''});await first.context.poll();
  assert.equal(first.get('projectName').textContent,'B');
  assert.equal(first.get('comparison').textContent,'No comparison yet.');
  assert.equal(first.get('log').textContent,'Job output appears here.');
  assert.equal(first.get('dataset').textContent,'');
  const refreshed=page('');await refreshed.context.poll();
  assert.equal(refreshed.calls[0]['X-Studio-Token'],'secret');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [node, "-e", harness], input=json.dumps(script), text=True, capture_output=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
