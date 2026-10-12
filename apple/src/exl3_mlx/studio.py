"""Small authenticated, loopback-only UI over the same SFT command-line workflow."""

import json
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .sft import write_json

MAX_UPLOAD = 16 << 20


def tail(path, size=12000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, stream.seek(0, 2) - size))
            return stream.read(size).decode("utf-8", errors="replace")
    except OSError:
        return ""


class StudioState:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.process = None
        self.project = None
        self.job = None
        self.log = None

    def running(self):
        return self.process is not None and self.process.poll() is None

    def snapshot(self):
        with self.lock:
            result = {
                "running": self.running(),
                "job": self.job,
                "project": str(self.project) if self.project else None,
            }
            if self.process is not None:
                result["exit_code"] = self.process.poll()
            result["log"] = tail(self.log) if self.log else ""
            if self.project and (self.project / "project.json").exists():
                manifest = json.loads((self.project / "project.json").read_text())
                result["project_status"] = manifest["status"]
                result["data"] = manifest["data"]
                result["config"] = manifest["config"]
                active = self.project / "active.json"
                if active.exists():
                    activity = json.loads(active.read_text())
                    log = Path(activity["log"]).resolve()
                    if log.is_relative_to(self.project):
                        result["phase"] = activity["phase"]
                        result["log"] += "\n" + tail(log)
                comparison = self.project / "comparison.json"
                if comparison.exists():
                    result["comparison"] = json.loads(comparison.read_text())
                    result["comparison_note"] = (
                        "Saved comparison; run Compare again after new training."
                    )
            return result

    def action(self, value):
        with self.lock:
            action = value.get("action")
            if action == "stop":
                if not self.project or not self.running() or self.job != "train":
                    raise ValueError("No active training job to stop")
                (self.project / "stop.request").touch()
                return {
                    "message": "Stop requested. The current update will finish and save a checkpoint."
                }
            if self.running():
                raise ValueError("A job is already running; wait or request a safe stop")
            if action == "open":
                project = Path(value["project"]).expanduser().resolve()
                manifest = json.loads((project / "project.json").read_text())
                if manifest.get("format") != "exl3-sft-project-v1":
                    raise ValueError("Choose an EXL3 SFT project")
                self.project = project
                self.log, self.job, self.process = None, None, None
                return {"message": "Project opened"}
            stamp = str(time.time_ns())
            if action == "prepare":
                from .sft import SFTConfig

                config = SFTConfig(
                    output_dir=value["project"],
                    max_steps=value.get("steps", 2),
                    max_length=value.get("max_length"),
                    chunk_size=value.get("chunk_size", 512),
                    demo=value.get("demo", False),
                    learning_rate=value.get("learning_rate", 1e-4),
                    profile=value.get("profile", "low-memory"),
                    memory_gib=value.get("memory_gib"),
                    reserve_gib=value.get("reserve_gib"),
                    packed_cache_mib=value.get("packed_cache_mib"),
                    decoded_cache_mib=value.get("decoded_cache_mib"),
                    allocator_cache_mib=value.get("allocator_cache_mib"),
                    prompt_column=value.get("prompt_column") or None,
                    response_column=value.get("response_column") or None,
                    group_column=value.get("group_column") or None,
                )
                import dataclasses

                config_file = self.directory / f"config-{stamp}.json"
                write_json(config_file, dataclasses.asdict(config))
                self.project = Path(config.output_dir)
                if self.project.exists():
                    raise ValueError(
                        "New projects need a new directory. Use Open existing for an existing project."
                    )
                command = [
                    "finetune",
                    "--model",
                    value["model"],
                    "--data",
                    value["data"],
                    "--output",
                    str(self.project),
                    "--config",
                    str(config_file),
                    "--prepare-only",
                ]
            elif action in {"train", "compare", "chat", "export"}:
                if self.project is None or not (self.project / "project.json").is_file():
                    raise ValueError("Check data or open a project first")
                if action == "train":
                    steps = value.get("steps", 2)
                    if type(steps) is not int or steps < 1:
                        raise ValueError("Total steps must be a positive integer")
                    command = [
                        "resume",
                        str(self.project),
                        "--steps",
                        str(steps),
                        "--no-compare",
                        "--request-start-ns",
                        stamp,
                    ]
                    if "resources" in value:
                        from .resources import RESOURCE_FIELDS, resolve_resources

                        if not isinstance(value["resources"], dict):
                            raise ValueError("Resource overrides must be an object")
                        if value["resources"]:
                            saved = json.loads((self.project / "project.json").read_text())[
                                "config"
                            ]
                            options = (
                                {}
                                if "profile" in value["resources"]
                                else {key: saved[key] for key in RESOURCE_FIELDS if key in saved}
                            )
                            resources = resolve_resources(**(options | value["resources"]))
                            command += [
                                part
                                for key, item in resources.items()
                                for part in ("--" + key.replace("_", "-"), str(item))
                            ]
                elif action == "compare":
                    command = ["compare", str(self.project)]
                elif action == "chat":
                    if not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
                        raise ValueError("Enter a prompt")
                    command = ["chat", str(self.project), "--prompt", value["prompt"]]
                else:
                    command = ["export-adapter", str(self.project), "--output", value["output"]]
            else:
                raise ValueError("Unknown action")
            if any(not isinstance(part, str) for part in command):
                raise ValueError("Paths and prompts must be text")
            self.log = self.directory / f"job-{stamp}.log"
            with self.log.open("w") as log:
                environment = dict(os.environ)
                environment["PYTHONPATH"] = (
                    str(Path(__file__).resolve().parent.parent)
                    + os.pathsep
                    + environment.get("PYTHONPATH", "")
                )
                self.process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "exl3_mlx.sft_cli", *command],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
            self.job = action
            return {"message": f"{action.title()} started", "pid": self.process.pid}


def make_server(state, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log authentication headers or training content.

        def send(self, status, body, content_type="application/json"):
            payload = (
                body if isinstance(body, bytes) else json.dumps(body, allow_nan=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected_host:
                return False
            if self.headers.get("Origin") not in (None, f"http://{expected_host}"):
                return False
            return secrets.compare_digest(self.headers.get("X-Studio-Token", ""), state.token)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                    return self.send(403, {"error": "Invalid host"})
                return self.send(
                    200,
                    Path(__file__).with_name("studio.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            if not self.authorized():
                return self.send(
                    403, {"error": "Open the complete local Studio URL printed in your terminal"}
                )
            if path != "/api/state":
                return self.send(404, {"error": "Not found"})
            try:
                self.send(200, state.snapshot())
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.send(400, {"error": str(error)})

        def do_POST(self):
            if not self.authorized():
                return self.send(403, {"error": "Unauthorized local request"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_UPLOAD:
                    raise ValueError("Requests must be between 1 byte and 16 MiB")
                body = self.rfile.read(length)
                if urlsplit(self.path).path == "/api/upload":
                    extension = Path(self.headers.get("X-Filename", "")).suffix.lower()
                    if extension not in {".csv", ".jsonl", ".json"}:
                        raise ValueError("Choose CSV, JSONL or JSON")
                    with state.lock:
                        if state.running():
                            raise ValueError("Wait for the current job before uploading")
                        path = state.directory / f"upload-{secrets.token_hex(8)}{extension}"
                        path.write_bytes(body)
                    result = {"path": str(path)}
                elif urlsplit(self.path).path == "/api/action":
                    value = json.loads(body)
                    if not isinstance(value, dict):
                        raise ValueError("Expected a JSON object")
                    result = state.action(value)
                else:
                    return self.send(404, {"error": "Not found"})
                self.send(200, result)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.send(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(*, port=0, open_browser=True):
    if not 0 <= port <= 65535:
        raise ValueError("Port must be in 0..65535")
    state = StudioState(tempfile.mkdtemp(prefix="exl3-studio-"))
    server = make_server(state, port)
    url = f"http://127.0.0.1:{server.server_port}/#{state.token}"
    print(
        f"EXL3 Fine-tuning Studio\n{url}\nLocal files stay on this Mac. Ctrl-C closes Studio; use Stop training before closing an active job.",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if state.running():
            print(
                f"The {state.job} job continues in its own process. Project: {state.project}. Use exl3-mlx stop PROJECT to request a safe training stop.",
                flush=True,
            )
    finally:
        server.server_close()
    return 0
