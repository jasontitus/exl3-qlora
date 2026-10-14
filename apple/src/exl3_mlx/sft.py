"""Project-based supervised fine-tuning over the streamed EXL3 engine.

A deliberately small SFT API, not a drop-in TRL implementation. GPU work runs in
child processes so the memory watchdog cannot terminate the caller's notebook.
"""

import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .activation_config import normalize_adapter_config, validate_activation_mode
from .context import ENGINE_SEMANTICS, MAX_CONTEXT_TOKENS
from .evaluation_audit import _json, audit_files
from .fit_probe import compare_final_fit_probe, validate_final_fit_examples
from .resources import RESOURCE_FIELDS, resolve_resources


@dataclass
class SFTConfig:
    output_dir: str
    max_steps: int = 20  # Total optimizer steps, including resumed steps.
    max_length: int | None = None  # Total serialized tokens, including BOS/EOS.
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    rank: int = 4
    lora_alpha: float = 8.0
    adapter_layers: list[int] | None = None  # Zero-based attention blocks; None means all 40.
    activation_mode: str = "checkpoint"  # Experimental bf16 changes the zero-adapter forward.
    save_steps: int = 1
    eval_steps: int = 5
    save_total_limit: int = 2
    seed: int = 8
    chunk_size: int = 512  # EXL3 column chunk; changes floating-point reduction order.
    profile: str = "low-memory"
    memory_gib: float | None = None
    reserve_gib: float | None = None
    packed_cache_mib: int | None = None
    decoded_cache_mib: int | None = None
    prefix_cache_mib: int = 0
    prefix_disk_cache: str | None = None
    prefix_disk_cache_gib: float = 0.0
    allocator_cache_mib: int | None = None
    demo: bool = False
    prompt_column: str | None = None
    response_column: str | None = None
    group_column: str | None = None
    max_new_tokens: int = 16
    comparison_examples: int = 4
    final_fit_examples: int = 0  # Optional warm training-mode probe; no optimizer update.
    loss_scope: str = "last_assistant"
    packing: bool = False

    def __post_init__(self):
        self.final_fit_examples = validate_final_fit_examples(self.final_fit_examples)
        self.activation_mode = validate_activation_mode(self.activation_mode)
        if self.adapter_layers is None:
            self.adapter_layers = list(range(40))
        elif (
            not isinstance(self.adapter_layers, (list, tuple))
            or not self.adapter_layers
            or any(type(layer) is not int or not 0 <= layer < 40 for layer in self.adapter_layers)
            or len(set(self.adapter_layers)) != len(self.adapter_layers)
        ):
            raise ValueError("adapter_layers must contain unique integer block IDs in 0..39")
        else:
            self.adapter_layers = sorted(self.adapter_layers)
        if (
            type(self.chunk_size) is not int
            or not 128 <= self.chunk_size <= 32768
            or self.chunk_size % 128
        ):
            raise ValueError("chunk_size must be a multiple of 128 in 128..32768")
        for key, value in resolve_resources(
            **{k: getattr(self, k) for k in RESOURCE_FIELDS}
        ).items():
            setattr(self, key, value)
        for key in (
            "max_steps",
            "gradient_accumulation_steps",
            "rank",
            "save_steps",
            "eval_steps",
            "save_total_limit",
            "max_new_tokens",
            "comparison_examples",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "max_grad_norm", "lora_alpha", "memory_gib", "reserve_gib"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.final_fit_examples > self.comparison_examples:
            raise ValueError(
                "final_fit_examples must not exceed comparison_examples for fresh verification"
            )
        if self.max_length is not None and (
            type(self.max_length) is not int or not 3 <= self.max_length <= MAX_CONTEXT_TOKENS
        ):
            raise ValueError("max_length is total serialized tokens and must be in 3..32768")
        if (
            self.per_device_train_batch_size != 1
            or type(self.per_device_train_batch_size) is not int
        ):
            raise ValueError(
                "Use batch size 1 and gradient_accumulation_steps for effective batching"
            )
        if self.packing is not False or self.loss_scope != "last_assistant":
            raise ValueError("Only unpacked last_assistant chat supervision is supported")
        if self.max_new_tokens > MAX_CONTEXT_TOKENS:
            raise ValueError("max_new_tokens must be <=32768")
        if type(self.seed) is not int or type(self.demo) is not bool:
            raise ValueError("seed must be an integer and demo a boolean")
        if bool(self.prompt_column) != bool(self.response_column):
            raise ValueError("Set both prompt_column and response_column")
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def _sync_provenance(path):
    """Persist a provenance publication before advancing its durable state."""
    path = Path(path)
    if path.is_file():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        path = path.parent
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def implementation_identity():
    root = Path(__file__).parent
    return {
        "source": {
            p.name: digest(p)
            for p in sorted(root.iterdir())
            if p.suffix in {".py", ".metal", ".html"}
        },
        "dependency_lock": digest(root.parent.parent / "uv.lock")
        if (root.parent.parent / "uv.lock").exists()
        else None,
        "runtime": {
            p: importlib.metadata.version(p)
            for p in ("mlx", "mlx-lm", "numpy", "transformers", "tokenizers", "safetensors")
        },
        "python": sys.version.split()[0],
    }


def capabilities(model):
    root = Path(model).expanduser().resolve()
    try:
        c = json.loads((root / "config.json").read_text())
        text = c["text_config"]
        supported = (
            c["model_type"] == "deepseek_v41"
            and text["num_hidden_layers"] == 40
            and text["index_topk"] == 512
            and c["quantization"]["mode"] == "mxfp8"
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Choose a local DeepSeek V4.1 streaming repack with config.json"
        ) from error
    required = (
        "expert-manifest-v2.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "engram/engram-manifest.json",
        "engram-token-map.u32.json",
        "engram/engram-residents.safetensors",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if not supported or missing:
        raise ValueError(
            "Unsupported checkpoint. This backend needs the 40-layer DeepSeek V4.1 "
            "MXFP8/EXL3 streaming repack and both Engram banks. "
            + (f"Missing: {', '.join(missing)}" if missing else "Architecture differs.")
        )
    return {
        "model": str(root),
        "backend": "streamed-exl3",
        "max_length": MAX_CONTEXT_TOKENS,
        "loss_scope": "last assistant answer including EOS; prior turns are context",
        "adapter_format": "exl3-mlx-deepseek-short-text-v1",
        "external_adapter_compatibility": False,
        "base_metadata": {
            name: digest(root / name)
            for name in ("config.json", *required)
            if not name.endswith(".safetensors")
        },
    }


@contextlib.contextmanager
def project_lock(root):
    # An inherited descriptor keeps exclusion while a GPU child outlives its parent.
    # Never unlink this inode: doing so would allow concurrent locks on two files.
    path = Path(root) / ".run.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Project is already running: {path.parent}") from None
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid()}).encode())
        yield fd
    finally:
        os.close(fd)


def checkpoint_state(path):
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text())
    required = {
        "adapter.safetensors",
        "adapter_config.json",
        "optimizer.safetensors",
        "random.safetensors",
    }
    if (
        state.get("format") != "exl3-mlx-training-state-v1"
        or set(state.get("files", {})) != required
    ):
        raise ValueError(f"Incomplete checkpoint: {path}")
    if any(digest(path / name) != state["files"][name] for name in required):
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    if type(state.get("step")) is not int or state["step"] < 0:
        raise ValueError(f"Invalid checkpoint step: {path}")
    return state


class SFTTrainer:
    def __init__(
        self,
        *,
        model,
        args,
        train_dataset=None,
        eval_dataset=None,
        test_dataset=None,
        progress=None,
    ):
        if not isinstance(args, SFTConfig):
            raise TypeError("args must be exl3_mlx.SFTConfig")
        self.args = args
        self.root = Path(args.output_dir)
        self.model = str(Path(model).expanduser().resolve())
        self.datasets = (train_dataset, eval_dataset, test_dataset)
        self._bound = False
        self.progress = progress or (lambda message: print(message, flush=True))

    @classmethod
    def from_project(
        cls,
        project,
        *,
        max_steps=None,
        progress=None,
        allow_pending_provenance=False,
        resources=None,
    ):
        """Open a project; the pending option permits evidence repair, never training."""
        root = Path(project).expanduser().resolve()
        manifest = json.loads((root / "project.json").read_text())
        config = dict(manifest["config"])
        config["output_dir"] = str(root)
        if resources:
            if set(resources) - set(RESOURCE_FIELDS):
                raise ValueError("Only resource settings can be overridden during resume")
            if "profile" in resources:
                for key in RESOURCE_FIELDS:
                    config.pop(key, None)
            config.update(resources)
        if max_steps is not None:
            config["max_steps"] = max_steps
        trainer = cls(model=manifest["model"], args=SFTConfig(**config), progress=progress)
        trainer._manifest(allow_pending_provenance=allow_pending_provenance)
        return trainer

    def _manifest(self, *, allow_pending_provenance=False):
        path = self.root / "project.json"
        manifest = json.loads(path.read_text())
        if manifest.get("format") != "exl3-sft-project-v1":
            raise ValueError("Unsupported fine-tuning project")
        immutable = dataclasses.asdict(self.args)
        saved = dict(manifest["config"])
        # Projects created before layer selection implicitly targeted all 40.
        # Source/runtime identity is still checked below; this is not migration.
        if saved.get("adapter_layers") is None:
            saved["adapter_layers"] = list(range(40))
        saved.setdefault("activation_mode", "checkpoint")
        saved.setdefault("final_fit_examples", 0)
        for config in (immutable, saved):
            config.pop("max_steps", None)
            config.pop("output_dir", None)
            for key in RESOURCE_FIELDS:
                config.pop(key, None)
        if immutable != saved or self.model != manifest["model"]:
            raise ValueError(
                "Project settings changed; create a new project for a different experiment"
            )
        if implementation_identity() != manifest["implementation"]:
            raise ValueError(
                "Training implementation/runtime changed; use the recorded environment or create a new project"
            )
        if capabilities(self.model)["base_metadata"] != manifest["capabilities"]["base_metadata"]:
            raise ValueError("Model/tokenizer metadata changed since project creation")
        for name, expected in manifest["data_files"].items():
            if digest(self.root / "data" / name) != expected:
                raise ValueError(f"Project dataset changed: {name}")
        binding = manifest.get("provenance_binding")
        if type(manifest.get("provenance_required", False)) is not bool:
            raise ValueError("Invalid provenance requirement")
        if manifest.get("provenance_required") and not binding and not allow_pending_provenance:
            raise ValueError(
                "Project requires provenance binding before training; reopen with "
                "allow_pending_provenance=True and call bind_provenance with the import report"
            )
        if binding is not None:
            if not isinstance(binding, dict) or binding.get("state") not in {"pending", "complete"}:
                raise ValueError("Invalid project provenance binding state")
            if binding["state"] == "pending" and not allow_pending_provenance:
                raise ValueError(
                    "Project provenance binding is pending; reopen with "
                    "allow_pending_provenance=True and retry bind_provenance with the same "
                    "import report before training"
                )
            if binding["state"] == "complete" and not manifest.get("provenance_files"):
                raise ValueError("Completed provenance binding has no file manifest")
        for name, expected in manifest.get("provenance_files", {}).items():
            path = (self.root / name).resolve()
            if not path.is_relative_to(self.root) or digest(path) != expected:
                raise ValueError(f"Project provenance changed: {name}")
        return manifest

    def bind_provenance(self, source, report):
        """Bind or resume a pinned evidence publication before any optimizer update.

        A durable pending marker blocks all training until the exact same report
        and file hashes have been published. A retry recreates only its recorded
        private stage, or verifies an already published directory after a crash.
        """
        with project_lock(self.root):
            manifest = self._manifest(allow_pending_provenance=True)
            if manifest.get("provenance_files") or self._checkpoints():
                raise ValueError("Provenance can only be bound once, before training")
            source = Path(source).resolve()
            directory = self.root / "ktune-evidence"
            expected = {
                **report["artifacts_sha256"],
                "ktune-import.json": report["manifest_sha256"],
            }
            if not expected or any(
                not isinstance(name, str)
                or Path(name).name != name
                or name in {".", ".."}
                or not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
                for name, checksum in expected.items()
            ):
                raise ValueError("Invalid evidence filenames or checksums")
            report_hash = hashlib.sha256(
                json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            binding = manifest.get("provenance_binding")
            if binding:
                if (
                    binding.get("state") != "pending"
                    or binding.get("files") != expected
                    or binding.get("report_sha256") != report_hash
                ):
                    raise ValueError("Pending provenance must resume with the same import report")
            else:
                if directory.exists() or directory.is_symlink():
                    raise ValueError("Unbound evidence directory already exists; use a new project")
                binding = {
                    "state": "pending",
                    "files": expected,
                    "report_sha256": report_hash,
                    "stage": ".ktune-evidence-" + uuid.uuid4().hex,
                    "previous_status": manifest.get("status", "prepared"),
                }
                manifest.update(provenance_binding=binding, status="provenance_pending")
                write_json(self.root / "project.json", manifest)
                _sync_provenance(self.root / "project.json")
            if (
                not isinstance(binding.get("stage"), str)
                or re.fullmatch(r"\.ktune-evidence-[0-9a-f]{32}", binding["stage"]) is None
            ):
                raise ValueError("Invalid provenance staging identity")
            stage = self.root / binding["stage"]
            if directory.is_symlink() or stage.is_symlink():
                raise ValueError("Provenance publication must not follow symlinks")
            if not directory.exists():
                if stage.exists():
                    if not stage.is_dir():
                        raise ValueError("Provenance staging path is not a directory")
                    shutil.rmtree(stage)
                stage.mkdir()
                for name, checksum in expected.items():
                    path = (source / name).resolve()
                    if not path.is_relative_to(source) or digest(path) != checksum:
                        raise ValueError(f"Import changed before project binding: {name}")
                    target = stage / name
                    shutil.copyfile(path, target)
                    if digest(target) != checksum:
                        raise ValueError(f"Import changed during copy: {name}")
                    _sync_provenance(target)
                stage.rename(directory)
                _sync_provenance(self.root)
            if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(expected):
                raise ValueError(
                    "Published provenance directory does not match its pending manifest"
                )
            hashes = {}
            for name, checksum in expected.items():
                target = directory / name
                if target.is_symlink() or not target.is_file() or digest(target) != checksum:
                    raise ValueError(f"Published provenance changed: {name}")
                hashes[str(target.relative_to(self.root))] = checksum
            write_json(self.root / "ktune-source.json", report)
            _sync_provenance(self.root / "ktune-source.json")
            hashes["ktune-source.json"] = digest(self.root / "ktune-source.json")
            manifest["provenance_files"] = hashes
            manifest["provenance_binding"] = {
                "state": "complete",
                "report_sha256": report_hash,
            }
            manifest["status"] = binding["previous_status"]
            write_json(self.root / "project.json", manifest)
            _sync_provenance(self.root / "project.json")

    def _child(self, command, *, label, directory):
        command = list(command)
        if command and command[0] in {"train", "evaluate"} and "--verbose" not in command:
            command.append("--verbose")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"{label}.log"
        self.progress(f"{label}: starting")
        started = time.monotonic()
        with log_path.open("w") as log:
            environment = dict(os.environ)
            environment["PYTHONPATH"] = (
                str(Path(__file__).resolve().parent.parent)
                + os.pathsep
                + environment.get("PYTHONPATH", "")
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "exl3_mlx.workflow", *map(str, command)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment,
                pass_fds=(self._lock_fd,) if hasattr(self, "_lock_fd") else (),
            )
            with contextlib.suppress(FileNotFoundError):
                write_json(
                    self.root / "active.json",
                    {
                        "phase": label,
                        "pid": process.pid,
                        "log": str(log_path),
                        "started": time.time(),
                        "running": True,
                    },
                )
            try:
                result = process.wait()
            except KeyboardInterrupt:
                if command[0] == "train":
                    self.request_stop()
                    self.progress(
                        "Stopping at the next completed update; waiting for its checkpoint."
                    )
                    while True:
                        try:
                            result = process.wait()
                            break
                        except KeyboardInterrupt:
                            self.progress("Still waiting for the current update to finish safely.")
                else:
                    process.terminate()
                    process.wait()
                    raise
            finally:
                write_json(
                    self.root / "active.json",
                    {"phase": label, "pid": process.pid, "log": str(log_path), "running": False},
                )
        if result:
            tail = log_path.read_text()[-2000:]
            raise RuntimeError(
                f"{label} stopped (exit {result}). Last complete checkpoint is retained.\n{tail}\nLog: {log_path}"
            )
        self.progress(f"{label}: completed in {time.monotonic() - started:.1f}s")

    def _resources(self):
        return [
            value
            for key in RESOURCE_FIELDS
            if getattr(self.args, key) is not None
            for value in ("--" + key.replace("_", "-"), getattr(self.args, key))
        ]

    def _engine_options(self, *, placement=True):
        return [
            *self._resources(),
            *(
                ["--adapter-layers", *self.args.adapter_layers]
                if placement and self.args.prefix_disk_cache
                else []
            ),
            "--chunk-size",
            self.args.chunk_size,
            "--activation-mode",
            self.args.activation_mode,
        ]

    def _verify_activation_receipt(self, receipt):
        mode = validate_activation_mode(receipt.get("activation_mode", "checkpoint"))
        if mode != self.args.activation_mode:
            raise ValueError("Child receipt activation_mode differs from the saved project")

    def _evaluation_rows(self, manifest, split):
        """Snapshot the exact prepared rows whose bytes the project binds."""
        path = self.root / "data" / f"{split}.jsonl"
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["data_files"][path.name]:
            raise ValueError(f"Project dataset changed: {path.name}")
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        if len(rows) != manifest["data"]["counts"][split]:
            raise ValueError("Project split count differs from its prepared rows")
        selected = rows[: self.args.comparison_examples]
        expected = {
            "data": str(path.resolve()),
            "input_identity": {"data_sha256": manifest["data_files"][path.name]},
            "selected_examples_sha256": hashlib.sha256(
                json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "seq_length": manifest["data"]["seq_length"],
            "requested_max_examples": self.args.comparison_examples,
            "implementation": manifest["implementation"],
        }
        return rows, selected, expected

    def _verify_evaluation_receipt(self, receipt, expected, selected, command, adapter):
        """New comparisons require bound child provenance; old reports stay readable."""
        self._verify_activation_receipt(receipt)
        context = receipt.get("evaluation_context")
        argv = list(map(str, command))
        if "--verbose" not in argv:
            argv.append("--verbose")
        if (
            not isinstance(context, dict)
            or any(context.get(key) != value for key, value in expected.items())
            or context.get("argv") != argv
            or type(context.get("process_id")) is not int
            or context["process_id"] <= 0
            or type(context.get("seq_length")) is not int
            or type(context.get("requested_max_examples")) is not int
        ):
            raise ValueError(
                "Child evaluation provenance differs from the immutable project selection"
            )
        if receipt.get("checkpoint") != self.model or receipt.get("adapter") != (
            str(adapter) if adapter is not None else None
        ):
            raise ValueError(
                "Child evaluation model or adapter differs from the requested comparison"
            )
        token_count = sum(len(row["input_ids"]) - row["loss_start"] for row in selected)
        if (
            type(receipt.get("examples")) is not int
            or receipt["examples"] != len(selected)
            or type(receipt.get("supervised_tokens")) is not int
            or receipt["supervised_tokens"] != token_count
            or type(receipt.get("loss")) not in (int, float)
            or not math.isfinite(receipt["loss"])
            or receipt["loss"] < 0
        ):
            raise ValueError("Child evaluation scores do not match the complete selected rows")

    @staticmethod
    def _write_rows(dataset, target):
        if isinstance(dataset, (str, Path)):
            path = Path(dataset).expanduser()
            if path.suffix.lower() not in {".jsonl", ".json", ".csv"}:
                raise ValueError("Explicit split files must be JSONL, JSON or CSV")
            shutil.copyfile(path, target.with_suffix(path.suffix.lower()))
            return
        with target.open("w") as stream:
            count = 0
            for row in dataset:
                if not isinstance(row, dict):
                    raise ValueError("Dataset must yield dictionaries")
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
                if count > 100000:
                    raise ValueError("Import supports at most 100,000 rows per split")

    def prepare(self, *, provenance_required=False):
        if type(provenance_required) is not bool:
            raise ValueError("provenance_required must be a boolean")
        if (self.root / "project.json").exists():
            if any(d is not None for d in self.datasets) and not self._bound:
                raise ValueError(
                    "Project already contains an immutable dataset; use from_project/resume, or choose a new output directory"
                )
            manifest = self._manifest()
            if provenance_required and not manifest.get("provenance_required"):
                raise ValueError("Existing project was not prepared with required provenance")
            return manifest
        if self.root.exists():
            raise FileExistsError(f"Choose a new project directory: {self.root}")
        if self.datasets[0] is None:
            raise ValueError("Provide train_dataset for a new project")
        caps = capabilities(self.model)
        from .sft_data import prepare_sft_data

        self.root.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".sft-project-", dir=self.root.parent))
        try:
            source = self.datasets[0]
            if self.datasets[1] is not None or self.datasets[2] is not None:
                inputs = stage / "inputs"
                inputs.mkdir()
                for name, data in zip(("train", "valid", "test"), self.datasets):
                    if data is not None:
                        self._write_rows(data, inputs / f"{name}.jsonl")
                source = inputs
            elif not isinstance(source, (str, Path)):
                self._write_rows(source, stage / "input.jsonl")
                source = stage / "input.jsonl"
            report = prepare_sft_data(
                source,
                self.model,
                stage / "data",
                max_length=self.args.max_length,
                seed=self.args.seed,
                demo=self.args.demo,
                prompt_column=self.args.prompt_column,
                response_column=self.args.response_column,
                group_column=self.args.group_column,
            )

            # Relocate only provenance paths. User text and metadata stay byte-for-byte intact.
            def relocate(value):
                if value == str(stage) or value.startswith(str(stage) + "/"):
                    return str(self.root) + value[len(str(stage)) :]
                return value

            dataset_manifest = json.loads((stage / "data/dataset.json").read_text())
            dataset_manifest["source"] = relocate(dataset_manifest["source"])
            dataset_manifest["source_files_sha256"] = {
                relocate(k): v for k, v in dataset_manifest["source_files_sha256"].items()
            }
            for preview in dataset_manifest["preview"]:
                preview["source"] = relocate(preview["source"])
            provenance_file = stage / "data/provenance.jsonl"
            provenance = [json.loads(line) for line in provenance_file.read_text().splitlines()]
            for record in provenance:
                for key in ("origin", "source_file"):
                    record[key] = relocate(record[key])
            provenance_file.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                    for row in provenance
                )
            )
            dataset_manifest["artifacts_sha256"] = {
                p.name: digest(p)
                for p in (stage / "data").iterdir()
                if p.is_file() and p.name != "dataset.json"
            }
            # Source path hashes are provenance; the byte hashes of snapshots remain unchanged.
            source_hashes = dataset_manifest["source_files_sha256"]
            dataset_manifest["source_sha256"] = hashlib.sha256(
                json.dumps(
                    source_hashes,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            write_json(stage / "data/dataset.json", dataset_manifest)
            report = {
                **dataset_manifest,
                "output": str(self.root / "data"),
                "manifest_sha256": digest(stage / "data/dataset.json"),
            }
            manifest = {
                "format": "exl3-sft-project-v1",
                "model": self.model,
                "config": dataclasses.asdict(self.args),
                "capabilities": caps,
                "data": report,
                "implementation": implementation_identity(),
                "data_files": {
                    p.name: digest(p) for p in (stage / "data").iterdir() if p.is_file()
                },
                "created": time.time(),
                "status": "prepared",
                "provenance_required": provenance_required,
            }
            write_json(stage / "project.json", manifest)
            if provenance_required:
                _sync_provenance(stage / "project.json")
            stage.rename(self.root)
            if provenance_required:
                _sync_provenance(self.root.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        self._bound = True
        self.progress(
            f"Prepared {report['counts']} examples. Review project.json and data previews before training."
        )
        return manifest

    def request_stop(self):
        (self.root / "stop.request").touch()

    def train(self, resume_from_checkpoint=None, *, compare=True, request_start_ns=None):
        request_start_ns = time.time_ns() if request_start_ns is None else request_start_ns
        self.prepare()
        with project_lock(self.root) as lock_fd:
            self._lock_fd = lock_fd
            manifest = self._manifest()
            resources = {key: getattr(self.args, key) for key in RESOURCE_FIELDS}
            manifest.setdefault("config", {}).update(resources)
            latest = self._recover()
            if resume_from_checkpoint not in (None, True):
                requested = Path(resume_from_checkpoint).expanduser().resolve()
                if not requested.is_relative_to(self.root / "attempts"):
                    raise ValueError("Resume checkpoint must belong to this project")
                if latest is None or requested != latest[1].resolve():
                    raise ValueError(
                        "Resume from the latest complete checkpoint to continue this project"
                    )
            if latest and resume_from_checkpoint is None:
                raise ValueError(
                    "Project already has updates; use train(resume_from_checkpoint=True) or resume PROJECT"
                )
            previous = latest[0] if latest else 0
            remaining = self.args.max_steps - previous
            if remaining < 0:
                raise ValueError(f"max_steps is total steps; project already completed {previous}")
            if not remaining:
                manifest.update(status="completed", target_steps=self.args.max_steps)
                write_json(self.root / "project.json", manifest)
            if remaining:
                stop = self.root / "stop.request"
                if stop.exists() and stop.stat().st_mtime_ns < request_start_ns:
                    stop.unlink()
                attempt = self.root / "attempts" / f"attempt-{time.time_ns()}"
                attempt.mkdir(parents=True)
                write_json(attempt / "resources.json", resources)
                write_json(self.root / "project.json", manifest)
                self._child(
                    ["doctor", "--model", self.model, *self._engine_options(placement=False)],
                    label="preflight",
                    directory=attempt,
                )
                seq = manifest["data"]["seq_length"]
                command = [
                    "train",
                    "--model",
                    self.model,
                    "--data",
                    self.root / "data",
                    "--output",
                    attempt / "train",
                    "--steps",
                    remaining,
                    "--seq-length",
                    seq,
                    "--rank",
                    self.args.rank,
                    "--alpha",
                    self.args.lora_alpha,
                    "--adapter-layers",
                    *self.args.adapter_layers,
                    "--learning-rate",
                    self.args.learning_rate,
                    "--grad-accumulation",
                    self.args.gradient_accumulation_steps,
                    "--clip-grad",
                    self.args.max_grad_norm,
                    "--save-every",
                    self.args.save_steps,
                    "--eval-every",
                    self.args.eval_steps,
                    "--eval-examples",
                    max(1, manifest["data"]["counts"]["valid"]),
                    "--final-fit-examples",
                    self.args.final_fit_examples,
                    "--keep-last",
                    self.args.save_total_limit,
                    "--seed",
                    self.args.seed,
                    "--stop-file",
                    self.root / "stop.request",
                    *self._engine_options(placement=False),
                ]
                if latest:
                    command += ["--resume", latest[1]]
                manifest.update(status="training", target_steps=self.args.max_steps)
                write_json(self.root / "project.json", manifest)
                try:
                    self._child(command, label="training", directory=attempt)
                    report = json.loads((attempt / "train/training.json").read_text())
                    self._verify_activation_receipt(report)
                    manifest.update(
                        status=report.get("status", "completed"),
                        last_training_report=str(attempt / "train/training.json"),
                    )
                except Exception as error:
                    manifest.update(status="interrupted", error=str(error))
                    write_json(self.root / "project.json", manifest)
                    raise
                write_json(self.root / "project.json", manifest)
        if compare and manifest["status"] == "completed":
            self.evaluate()
        return self.status()

    def _checkpoints(self):
        # Checkpoint directories are nested under each immutable attempt's train directory.
        manifest = json.loads((self.root / "project.json").read_text())
        runtime = manifest["implementation"]["runtime"]
        expected = {
            "learning_rate": self.args.learning_rate,
            "grad_accumulation": self.args.gradient_accumulation_steps,
            "clip_grad": self.args.max_grad_norm,
            "seed": self.args.seed,
            "seq_length": manifest["data"]["seq_length"],
            "truncate": False,
            "chunk_size": self.args.chunk_size,
            "no_checkpoint": False,
            "data_sha256": manifest["data_files"]["train.jsonl"],
            "mlx": runtime["mlx"],
            "mlx_lm": runtime["mlx-lm"],
            "engine_semantics": ENGINE_SEMANTICS,
        }
        validation = None
        if manifest["data"]["counts"]["valid"]:
            rows = [
                json.loads(line)
                for line in (self.root / "data/valid.jsonl").read_text().splitlines()
                if line.strip()
            ]
            validation = {
                "data_sha256": manifest["data_files"]["valid.jsonl"],
                "selection": "first examples in immutable dataset order",
                "requested_examples": len(rows),
                "examples": len(rows),
                "selected_sha256": hashlib.sha256(
                    json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        candidates = []
        for path in sorted((self.root / "attempts").glob("*/train/step-*")):
            if path.is_dir():
                state = checkpoint_state(path)
                if (
                    state.get("contract") != expected
                    or state.get("selection", {}).get("validation_identity") != validation
                ):
                    raise ValueError(
                        f"Checkpoint does not belong to this dataset and training configuration: {path}"
                    )
                config = normalize_adapter_config(
                    json.loads((path / "adapter_config.json").read_text())
                )
                expected_base = {
                    k: manifest["capabilities"]["base_metadata"][k]
                    for k in (
                        "config.json",
                        "expert-manifest-v2.json",
                        "engram/engram-manifest.json",
                        "engram-token-map.u32.json",
                        "model.safetensors.index.json",
                        "tokenizer.json",
                    )
                }
                if (
                    config.get("base_identity") != expected_base
                    or config.get("rank") != self.args.rank
                    or config.get("alpha") != self.args.lora_alpha
                    or config.get("adapter_layers") != self.args.adapter_layers
                    or config.get("expert_adapter_layers") != []
                    or config["activation_mode"] != self.args.activation_mode
                ):
                    raise ValueError(
                        f"Checkpoint base model or adapter settings differ from project: {path}"
                    )
                best = state.get("selection", {}).get("best")
                if best and not Path(best["checkpoint"]).resolve().is_relative_to(
                    self.root / "attempts"
                ):
                    raise ValueError("Best checkpoint reference escapes project")
                candidates.append((state["step"], path, state))
        return candidates

    def _recover(self):
        candidates = self._checkpoints()
        return max(candidates, key=lambda item: (item[0], str(item[1]))) if candidates else None

    def selected_checkpoint(self):
        candidates = self._checkpoints()
        if not candidates:
            raise ValueError("No completed update yet; start or resume training")
        scored = []
        for step, path, state in candidates:
            value = state.get("metrics", {}).get("validation", {}).get("loss")
            if type(value) in (int, float) and math.isfinite(value):
                scored.append((value, step, str(path)))
        return (
            Path(min(scored)[2])
            if scored
            else max(candidates, key=lambda item: (item[0], str(item[1])))[1]
        )

    def verify_training_fit(self):
        """Operational fit check on training rows, explicitly never held-out quality."""
        with project_lock(self.root) as lock_fd:
            self._lock_fd = lock_fd
            manifest = self._manifest()
            adapter = self.selected_checkpoint()
            rows, selected_rows, expected = self._evaluation_rows(manifest, "train")
            directory = self.root / "fit-checks" / str(time.time_ns())
            directory.mkdir(parents=True)
            results = {}
            fit_options = self._engine_options()
            # A new child plus disabled prefix reuse gives the probe an actual
            # uncached comparison, independent of this project's train cache.
            fit_options[fit_options.index("--prefix-cache-mib") + 1] = 0
            for flag in ("--prefix-disk-cache", "--prefix-disk-cache-gib"):
                if flag in fit_options:
                    position = fit_options.index(flag)
                    del fit_options[position : position + 2]
            for condition in ("base", "adapter"):
                path = directory / f"{condition}.json"
                command = [
                    "evaluate",
                    "--model",
                    self.model,
                    "--data",
                    self.root / "data/train.jsonl",
                    "--token-metrics",
                    "--seq-length",
                    manifest["data"]["seq_length"],
                    "--max-examples",
                    self.args.comparison_examples,
                    "--report",
                    path,
                    *fit_options,
                ]
                if condition == "adapter":
                    command += ["--adapter", adapter]
                self._child(command, label=f"fit-{condition}", directory=directory)
                results[condition] = _json(path.read_bytes())
                self._manifest()
                self._verify_evaluation_receipt(
                    results[condition],
                    expected,
                    selected_rows,
                    command,
                    adapter if condition == "adapter" else None,
                )
            token_count = sum(len(row["input_ids"]) - row["loss_start"] for row in selected_rows)
            parity = self._final_fit_parity(
                adapter, results["adapter"], rows, selected_rows, expected
            )
            result = {
                "available_examples": len(rows),
                "evaluated_examples": len(selected_rows),
                "supervised_tokens": token_count,
                "selection": "first immutable training rows; no held-out interpretation",
                "selected_examples_sha256": hashlib.sha256(
                    json.dumps(selected_rows, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "format": "exl3-sft-training-fit-v1",
                "evaluation_scope": "training_only",
                "quality_certified": False,
                "note": "Training likelihood is a wiring/fit check, not generalization or kernel speedup.",
                "selected_checkpoint": str(adapter),
                "training_data_sha256": manifest["data_files"]["train.jsonl"],
                "loss_delta": results["adapter"]["loss"] - results["base"]["loss"],
                "training_mode_probe_parity": parity,
                **results,
            }
            write_json(directory / "fit.json", result)
            write_json(self.root / "fit.json", result)
            return result

    def _final_fit_parity(self, adapter, fresh, rows, selected_rows, expected):
        report_path = adapter.parent / "training.json"
        unavailable = {
            "status": "unavailable",
            "real_model_warm_cache_parity_verified": False,
            "quality_certified": False,
        }
        if not report_path.exists():
            return {**unavailable, "reason": "Selected checkpoint has no completed training report"}
        report = json.loads(report_path.read_text())
        probe = report.get("training_fit_probe")
        if probe is None:
            return {**unavailable, "reason": "No final training-mode fit probe was recorded"}
        if report.get("status") != "completed":
            raise ValueError("Training fit probe requires a completed training report")
        if not isinstance(probe, dict):
            raise ValueError("Invalid training fit probe receipt")
        if probe.get("checkpoint") != str(adapter.resolve()):
            return {
                **unavailable,
                "reason": "Final probe belongs to a different checkpoint than the selected adapter",
            }
        result = compare_final_fit_probe(
            probe,
            fresh,
            selected_rows,
            checkpoint=adapter,
            expected={
                **expected,
                "available_examples": len(rows),
                "requested_examples": self.args.final_fit_examples,
                "activation_mode": self.args.activation_mode,
                "chunk_size": self.args.chunk_size,
            },
        )
        result["training_report_sha256"] = digest(report_path)
        return result

    def evaluate(self, *, categories=None):
        with project_lock(self.root) as lock_fd:
            self._lock_fd = lock_fd
            manifest = self._manifest()
            adapter = self.selected_checkpoint()
            counts = manifest["data"]["counts"]
            split = "test" if counts["test"] else "valid" if counts["valid"] else None
            if split is None:
                result = {
                    "status": "training_completed",
                    "evaluation_scope": "none",
                    "quality_certified": False,
                    "reason": "Demonstration has no held-out examples",
                    "selected_checkpoint": str(adapter),
                }
                write_json(self.root / "comparison.json", result)
                self._render_report(result)
                return result
            _, selected_rows, expected = self._evaluation_rows(manifest, split)
            directory = self.root / "comparisons" / str(time.time_ns())
            directory.mkdir(parents=True)
            from .project_audit import snapshot_audit_inputs, verify_category_source

            prepared_path, category_path, category_origin, category_source = snapshot_audit_inputs(
                self.model, selected_rows, directory, categories
            )
            self._manifest()
            old_exposures = manifest.get("test_exposure_count", 0)
            if split == "test":
                manifest["test_exposure_count"] = old_exposures + 1
                write_json(self.root / "project.json", manifest)
            results = {}
            receipt_hashes = {}
            for condition in ("base", "adapter"):
                path = directory / f"{condition}.json"
                command = [
                    "evaluate",
                    "--model",
                    self.model,
                    "--data",
                    self.root / "data" / f"{split}.jsonl",
                    "--seq-length",
                    manifest["data"]["seq_length"],
                    "--max-examples",
                    self.args.comparison_examples,
                    "--token-metrics",
                    "--report",
                    path,
                    *self._engine_options(),
                ]
                if condition == "adapter":
                    command += ["--adapter", adapter]
                self._child(command, label=condition, directory=directory)
                receipt_bytes = path.read_bytes()
                receipt_hashes[condition] = hashlib.sha256(receipt_bytes).hexdigest()
                results[condition] = _json(receipt_bytes)
                self._manifest()
                self._verify_evaluation_receipt(
                    results[condition],
                    expected,
                    selected_rows,
                    command,
                    adapter if condition == "adapter" else None,
                )
            verify_category_source(category_source)
            token_audit = audit_files(
                prepared=prepared_path,
                categories=category_path,
                base=directory / "base.json",
                adapter=directory / "adapter.json",
                output=directory / "token-audit.json",
            )
            # Retain the existing project provenance gate, checked against the same
            # receipt bytes the audit captured, not a later replacement receipt.
            for condition in ("base", "adapter"):
                receipt_path = directory / f"{condition}.json"
                if (
                    token_audit["inputs_sha256"][str(receipt_path.resolve())]
                    != receipt_hashes[condition]
                    or digest(receipt_path) != receipt_hashes[condition]
                ):
                    raise ValueError("Evaluation receipt changed during comparison")
            delta = token_audit["aggregate"]["all"]["nll_delta"]
            result = {
                "status": "likelihood_improved"
                if delta < 0
                else "likelihood_regressed"
                if delta > 0
                else "likelihood_unchanged",
                "evaluation_scope": "held_out_test" if split == "test" else "selected_development",
                "quality_certified": False,
                "note": "Conditional answer likelihood is not task accuracy or Metal speedup.",
                "selected_checkpoint": str(adapter),
                "split": split,
                "available_examples": counts[split],
                "evaluated_examples": results["base"]["examples"],
                "selected_examples_sha256": expected["selected_examples_sha256"],
                "data_sha256": expected["input_identity"]["data_sha256"],
                "loss_delta": delta,
                "token_audit": token_audit,
                "token_audit_path": str(directory / "token-audit.json"),
                "token_audit_sha256": digest(directory / "token-audit.json"),
                "category_origin": category_origin,
                **results,
            }
            raw = self.root / "data" / f"raw-{split}.jsonl"
            if raw.exists():
                row = json.loads(raw.read_text().splitlines()[0])
                if "prompt" in row or "messages" in row:
                    result["preview"] = self._paired_preview(row, adapter, directory)
            verify_category_source(category_source)
            for name, checksum in token_audit["inputs_sha256"].items():
                if digest(name) != checksum:
                    raise ValueError("Audit input changed before comparison publication")
            if digest(result["token_audit_path"]) != result["token_audit_sha256"]:
                raise ValueError("Token audit artifact changed before comparison publication")
            old_comparisons = manifest.get("comparison_count", 0)
            result["test_previously_observed"] = bool(old_exposures and split == "test")
            if result["test_previously_observed"]:
                result["evaluation_scope"] = "previously_observed_test"
            self._manifest()
            write_json(directory / "comparison.json", result)
            write_json(self.root / "comparison.json", result)
            manifest.update(
                comparison_count=old_comparisons + 1,
                selected_adapter=str(adapter),
                comparison=str(directory / "comparison.json"),
            )
            write_json(self.root / "project.json", manifest)
            self._render_report(result)
            return result

    def _paired_preview(self, row, adapter, directory):
        if "tools" in row:
            return {
                "status": "skipped",
                "reason": "Native tool actions require the explicit tools-aware generation and patch-replay evaluation; plain-text preview is not a tool success metric",
                "tool_action_verified": False,
            }
        from .data import load_tokenizer

        tokenizer = load_tokenizer(self.model)
        if "messages" in row:
            history = row["messages"][:-1]
            expected = row["messages"][-1]["content"]
        else:
            history = [{"role": "user", "content": row["prompt"]}]
            expected = row["completion"]
        history_file = directory / "messages.json"
        write_json(history_file, history)
        ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_dict=False)
        if len(ids) + self.args.max_new_tokens > MAX_CONTEXT_TOKENS:
            return {
                "status": "skipped",
                "reason": "Prompt plus requested generation exceeds 32768; data was not truncated",
            }
        values = {}
        for condition in ("base", "adapter"):
            path = directory / f"{condition}-generation.json"
            command = [
                "generate",
                "--model",
                self.model,
                "--messages-file",
                history_file,
                "--max-tokens",
                self.args.max_new_tokens,
                "--report",
                path,
                *self._engine_options(),
            ]
            if condition == "adapter":
                command += ["--adapter", adapter]
            self._child(command, label=f"{condition}-generation", directory=directory)
            value = json.loads(path.read_text())
            self._verify_activation_receipt(value)
            generated = value.get("generated_ids")
            if (
                value.get("prompt_ids") != list(ids)
                or value.get("max_new_tokens") != self.args.max_new_tokens
                or value.get("eos_token_id") != tokenizer.eos_token_id
                or not isinstance(generated, list)
                or not 1 <= len(generated) <= self.args.max_new_tokens
                or any(type(token) is not int for token in generated)
                or tokenizer.eos_token_id in generated[:-1]
                or value.get("stop_reason") not in {"eos", "max_tokens"}
                or (value["stop_reason"] == "eos") != (generated[-1] == tokenizer.eos_token_id)
                or (
                    value["stop_reason"] == "max_tokens"
                    and len(generated) != self.args.max_new_tokens
                )
                or value.get("text") != tokenizer.decode(generated, skip_special_tokens=True)
            ):
                raise ValueError("Generation receipt differs from the paired comparison request")
            values[condition] = {
                "text": value["text"],
                "stop_reason": value["stop_reason"],
                "prompt_ids": value["prompt_ids"],
                "exact_answer_with_eos": value["text"].strip() == expected.strip()
                and value["stop_reason"] == "eos",
            }
        if values["base"]["prompt_ids"] != values["adapter"]["prompt_ids"]:
            raise ValueError("Comparison prompt tokenization differs")
        return {"messages": history, "expected": expected, **values}

    def _render_report(self, result):
        # HTML is intentionally escaped; examples may contain arbitrary source code.
        import html

        body = html.escape(json.dumps(result, indent=2, ensure_ascii=False))
        audit = result.get("token_audit")
        summary = ""
        if audit:
            metrics = audit["aggregate"]["all"]
            summary = (
                "<p>"
                + html.escape(
                    f"Paired token audit: {audit['examples']} examples, {metrics['tokens']} supervised tokens. "
                    f"Token-weighted NLL delta: {metrics['nll_delta']:.6f}. "
                    f"Teacher-forced correct tokens: {metrics['base_correct']} → {metrics['adapter_correct']}; "
                    f"corrected {metrics['corrected_tokens']}, regressed {metrics['regressed_tokens']}. "
                    "These counts do not measure generated-code success or inference speed."
                )
                + "</p>"
            )
        (self.root / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Fine-tuning comparison</title>'
            "<style>body{font:16px system-ui;max-width:960px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;background:#f4f5f7;padding:20px}</style>"
            "<h1>Fine-tuning comparison</h1><p>Completed training is separate from measured task quality. "
            "The comparison below reports observed likelihood and a fixed generation preview.</p>"
            + summary
            + "<pre>"
            + body
            + "</pre>"
        )

    def generate(self, prompt, *, max_new_tokens=None, base=False):
        tokens = self.args.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(tokens) is not int or not 1 <= tokens <= MAX_CONTEXT_TOKENS:
            raise ValueError("max_new_tokens must be in 1..32768")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty text prompt")
        with project_lock(self.root) as lock_fd:
            self._lock_fd = lock_fd
            self._manifest()
            directory = self.root / "previews" / str(time.time_ns())
            directory.mkdir(parents=True)
            path = directory / "generation.json"
            command = [
                "generate",
                "--model",
                self.model,
                "--prompt",
                prompt,
                "--max-tokens",
                tokens,
                "--report",
                path,
                *self._engine_options(),
            ]
            if not base:
                command += ["--adapter", self.selected_checkpoint()]
            self._child(command, label="preview", directory=directory)
            receipt = json.loads(path.read_text())
            self._verify_activation_receipt(receipt)
            return receipt

    def save_model(self, output_dir):
        with project_lock(self.root) as lock_fd:
            self._lock_fd = lock_fd
            self._manifest()
            source = self.selected_checkpoint()
            target = Path(output_dir).expanduser().resolve()
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".sft-export-", dir=target.parent))
            try:
                # Export inference weights only; resume remains in the project.
                for name in ("adapter.safetensors", "adapter_config.json"):
                    shutil.copyfile(source / name, stage / name)
                config = normalize_adapter_config(
                    json.loads((stage / "adapter_config.json").read_text())
                )
                config["adapter_sha256"] = digest(stage / "adapter.safetensors")
                write_json(stage / "adapter_config.json", config)
                write_json(
                    stage / "bundle.json",
                    {
                        "format": "exl3-mlx-adapter-bundle-v1",
                        "model": self.model,
                        "source_checkpoint": str(source),
                        "engine_settings": {"chunk_size": self.args.chunk_size},
                        "external_compatibility": False,
                    },
                )
                stage.rename(target)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            return str(target)

    def status(self):
        manifest = json.loads((self.root / "project.json").read_text())
        try:
            with project_lock(self.root):
                latest = self._recover()
                selected = str(self.selected_checkpoint()) if latest else None
        except RuntimeError as error:
            if "already running" not in str(error):
                raise
            return {
                "project": str(self.root),
                "status": "running",
                "saved_status": manifest["status"],
                "integrity_checked": False,
            }
        comparison = (
            json.loads((self.root / "comparison.json").read_text())
            if (self.root / "comparison.json").exists()
            else None
        )
        compared_checkpoint = None
        if comparison:
            compared_checkpoint = comparison.get("selected_checkpoint")
            if compared_checkpoint is None:
                # Read the first development version's receipts without losing metrics.
                legacy = comparison.get("adapter")
                compared_checkpoint = legacy.get("adapter") if isinstance(legacy, dict) else legacy
        return {
            "project": str(self.root),
            "status": manifest["status"],
            "completed_steps": latest[0] if latest else 0,
            "target_steps": manifest.get("target_steps", self.args.max_steps),
            "adapter_layers": list(self.args.adapter_layers),
            "activation_mode": self.args.activation_mode,
            "latest_checkpoint": str(latest[1]) if latest else None,
            "selected_checkpoint": selected,
            "comparison": comparison,
            "comparison_is_current": bool(comparison and compared_checkpoint == selected),
            "integrity_checked": True,
        }
