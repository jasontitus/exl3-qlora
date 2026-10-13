"""A bounded, local, end-to-end DeepSeek EXL3 adapter workflow."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .activation_config import ACTIVATION_MODES, validate_activation_mode
from .context import ENGINE_SEMANTICS, MAX_CONTEXT_TOKENS
from .data import load_dataset, load_tokenizer, prepare, training_data_identity
from .deepseek import DeepSeekTextModel
from .fit_probe import file_sha256, run_final_fit_probe, validate_final_fit_examples
from .memory import MemoryBudget
from .resources import RESOURCE_FIELDS, add_resource_arguments, resources_from_args
from .train import load_adapter, load_adapter_config
from .training import evaluate, load_checkpoint, optimizer_step, sample_indices, save_checkpoint


def common(parser, *, inference=False):
    parser.add_argument("--model", required=True)
    add_resource_arguments(parser)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument(
        "--activation-mode",
        choices=ACTIVATION_MODES,
        default=None,
        help="checkpoint (default), or experimental bf16 activation bypass; saved adapters inherit their mode; a changed mode requires a fresh matched baseline",
    )
    parser.add_argument("--verbose", action="store_true", help="Log every layer")
    if inference:
        parser.add_argument("--adapter", help="Saved step directory; omit to evaluate the base")
        parser.add_argument("--report", help="Write a new JSON report")
        parser.add_argument(
            "--adapter-layers",
            type=int,
            nargs="*",
            default=None,
            help="Base-model adapter placement for matching frozen-prefix cache scope",
        )
        parser.add_argument("--expert-adapter-layers", type=int, nargs="*", default=None)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Tokenize, validate, deduplicate and split JSONL")
    prep.add_argument("--model", required=True)
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seq-length", type=int, default=128)
    prep.add_argument("--seed", type=int, default=8)
    prep.add_argument("--valid-fraction", type=float, default=0.1)
    prep.add_argument("--test-fraction", type=float, default=0.1)
    prep.add_argument("--truncate", action="store_true", help="Explicitly allow prefix truncation")
    prep.add_argument("--all-tokens", action="store_true", help="Also supervise chat prompts")
    train = sub.add_parser(
        "train", help="Train adapters with streamed EXL3 and exact optimizer resume"
    )
    common(train)
    train.add_argument("--config", help="JSON defaults; explicit CLI flags override")
    train.add_argument("--data", required=True, help="Training JSONL or prepared dataset directory")
    train.add_argument(
        "--valid-data", help="Optional validation JSONL; auto-detected in data directory"
    )
    train.add_argument("--output", required=True, help="New run directory")
    train.add_argument("--steps", type=int, default=10, help="Additional optimizer steps")
    train.add_argument(
        "--stop-file", help="Stop and checkpoint between updates when this file exists"
    )
    train.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Infer from prepared data or observed token lengths",
    )
    train.add_argument("--rank", type=int)
    train.add_argument("--alpha", type=float)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--grad-accumulation", type=int, default=1)
    train.add_argument("--clip-grad", type=float, default=1.0)
    train.add_argument("--adapter-layers", type=int, nargs="*", default=None)
    train.add_argument("--expert-adapter-layers", type=int, nargs="*", default=None)
    train.add_argument("--no-checkpoint", action="store_true")
    train.add_argument("--truncate", action="store_true")
    train.add_argument("--seed", type=int, default=8)
    resume = train.add_mutually_exclusive_group()
    resume.add_argument("--resume", help="Restore adapters, Adam moments, RNG and sample cursor")
    resume.add_argument("--resume-adapter", help="Warm start weights with a fresh optimizer")
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument(
        "--keep-last", type=int, default=2, help="Retain this many checkpoints in this run"
    )
    train.add_argument("--eval-every", type=int, default=10)
    train.add_argument("--eval-examples", type=int, default=4)
    train.add_argument(
        "--final-fit-examples",
        type=int,
        default=0,
        help="Optional final training-mode probe on first N complete rows (0 disables; max32); no update",
    )
    train.add_argument(
        "--verify-reload", action="store_true", help="Check post-training prediction reload"
    )
    ev = sub.add_parser("evaluate", help="Token-weighted NLL/perplexity on held-out examples")
    common(ev, inference=True)
    ev.add_argument("--data", required=True)
    ev.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Infer from prepared data or observed token lengths",
    )
    ev.add_argument("--max-examples", type=int, default=0, help="0 evaluates all rows")
    ev.add_argument(
        "--token-metrics",
        action="store_true",
        help="Include per-answer-token NLL and teacher-forced predicted IDs in the report",
    )
    gen = sub.add_parser(
        "generate", help="Greedy text generation; recomputes the full prefix each token"
    )
    common(gen, inference=True)
    prompt = gen.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument(
        "--messages-file", help="JSON array of plain-text conversation messages ending in user"
    )
    gen.add_argument("--max-tokens", type=int, default=16)
    gen.add_argument(
        "--tools-file",
        help="JSON list of declared function schemas for native tool-call generation",
    )
    gen.add_argument(
        "--raw-prompt", action="store_true", help="Plain text continuation instead of chat"
    )
    doctor = sub.add_parser(
        "doctor", help="Validate local layout and memory headroom without loading weights"
    )
    common(doctor)
    doctor.add_argument(
        "--verify-records", action="store_true", help="Hash first/last expert records"
    )
    return p, train


def parse_args(argv):
    parser, train = build_parser()
    if (
        argv
        and argv[0] == "train"
        and any(a == "--config" or a.startswith("--config=") for a in argv)
    ):
        cp = argparse.ArgumentParser(add_help=False)
        cp.add_argument("--config")
        cp.add_argument("--profile")
        known, _ = cp.parse_known_args(argv[1:])
        config = json.loads(Path(known.config).expanduser().read_text())
        if isinstance(config, dict) and "cache_mib" in config:
            if "packed_cache_mib" in config:
                parser.error("Use only one of cache_mib and packed_cache_mib")
            config["packed_cache_mib"] = config.pop("cache_mib")
        if isinstance(config, dict) and known.profile is not None:
            for key in RESOURCE_FIELDS:
                config.pop(key, None)
        actions = {a.dest: a for a in train._actions}
        if not isinstance(config, dict) or set(config) - (set(actions) - {"help", "config"}):
            parser.error("Config must be an object containing recognized train option names")
        for key, value in config.items():
            action = actions[key]
            # Enforce JSON types instead of letting argparse bypass type checking of defaults.
            if isinstance(action, argparse._StoreTrueAction):
                valid = type(value) is bool
            elif action.nargs == "*":
                valid = isinstance(value, list) and all(type(v) is int for v in value)
            elif action.type is int:
                valid = type(value) is int
            elif action.type is float:
                valid = type(value) in (float, int) and math.isfinite(value)
            else:
                valid = isinstance(value, str)
            if not valid:
                parser.error(f"Invalid config value for {key}")
            action.required = False
            action.default = value
    args = parser.parse_args(argv)
    if args.command != "prepare":
        for key, value in resources_from_args(args).items():
            setattr(args, key, value)
        args.cache_mib = args.packed_cache_mib  # Preserve the existing Python helper attribute.
    return args


def model_settings(args):
    adapter = (
        getattr(args, "adapter", None)
        or getattr(args, "resume", None)
        or getattr(args, "resume_adapter", None)
    )
    keys = ("rank", "alpha", "adapter_layers", "expert_adapter_layers", "activation_mode")
    defaults = dict(
        rank=8,
        alpha=16.0,
        adapter_layers=None,
        expert_adapter_layers=[],
        activation_mode="checkpoint",
    )
    if adapter:
        c = load_adapter_config(Path(adapter).expanduser())
        for key in keys:
            value = getattr(args, key, None)
            if value is not None and value != c[key]:
                raise ValueError(f"{key} differs from saved adapter")
        return {k: c[k] for k in keys}
    settings = {
        k: getattr(args, k, None) if getattr(args, k, None) is not None else defaults[k]
        for k in keys
    }
    validate_activation_mode(settings["activation_mode"])
    return settings


def make_model(args, log):
    if args.cache_mib < 0 or args.chunk_size < 128 or args.chunk_size % 128:
        raise ValueError("Use nonnegative cache_mib and chunk_size divisible by 128")
    settings = model_settings(args)
    log("Loading frozen trunk; EXL3 experts and Engram rows stay streamed")
    model = DeepSeekTextModel(
        Path(args.model).expanduser(),
        **settings,
        cache_mib=args.cache_mib,
        decoded_cache_mib=getattr(args, "decoded_cache_mib", 0),
        prefix_cache_mib=getattr(args, "prefix_cache_mib", 0),
        prefix_disk_cache=getattr(args, "prefix_disk_cache", None),
        prefix_disk_cache_gib=getattr(args, "prefix_disk_cache_gib", 0),
        chunk_size=args.chunk_size,
        checkpoint_blocks=not getattr(args, "no_checkpoint", False),
        progress=log if args.verbose else None,
    )
    adapter = getattr(args, "adapter", None) or getattr(args, "resume_adapter", None)
    if adapter:
        load_adapter(model, Path(adapter).expanduser())
    log(f"Loaded; {model.report()['trainable_parameters']:,} adapter parameters")
    return model


def atomic_text(path, text):
    """Publish pointers/reports only after their complete contents reach disk."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_resolved_datasets(paths, root, *, seq_length=None, truncate=False):
    """Use one explicit or inferred input length for all supplied splits."""
    paths = [Path(path).expanduser() for path in paths]
    if seq_length is not None and (
        type(seq_length) is not int or not 2 <= seq_length <= MAX_CONTEXT_TOKENS
    ):
        raise ValueError("seq_length must be in 2..32768")
    declared = []
    if seq_length is None:
        for path in paths:
            manifest = path.parent / "dataset.json"
            if manifest.exists():
                length = json.loads(manifest.read_text()).get("seq_length")
                if type(length) is not int or not 2 <= length <= MAX_CONTEXT_TOKENS:
                    raise ValueError("Prepared dataset seq_length must be in 2..32768")
                declared.append(length)
    datasets = [
        load_dataset(path, root, seq_length=seq_length or MAX_CONTEXT_TOKENS, truncate=truncate)
        for path in paths
    ]
    if seq_length is None:
        seq_length = max(
            [2, *declared] + [len(row["input_ids"]) - 1 for dataset in datasets for row in dataset]
        )
    return seq_length, datasets


def validation_identity(path, root, examples, count, *, input_identity=None):
    if not examples:
        return None
    selected = examples[:count]
    return {
        **(training_data_identity(path, root) if input_identity is None else input_identity),
        "selection": "first examples in immutable dataset order",
        "requested_examples": count,
        "examples": len(selected),
        "selected_sha256": hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def restored_best(state, identity):
    """Old checkpoints retain their exact optimizer contract and start selection anew."""
    if "selection" not in state:
        return None
    selection = state["selection"]
    if not isinstance(selection, dict) or selection.get("validation_identity") != identity:
        raise ValueError(
            "Resume validation data/subset differ from saved best-checkpoint selection"
        )
    best = selection.get("best")
    if best is None:
        return None
    if (
        not isinstance(best, dict)
        or type(best.get("step")) is not int
        or not 0 <= best["step"] <= state["step"]
        or type(best.get("loss")) not in (int, float)
        or not math.isfinite(best["loss"])
        or not isinstance(best.get("checkpoint"), str)
    ):
        raise ValueError("Invalid saved best-checkpoint selection")
    path = Path(best["checkpoint"])
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("Saved best checkpoint is missing; restore the original attempt directory")
    load_adapter_config(path)
    return dict(best)


def prune_checkpoints(saved, *, keep_last, latest, best):
    """Delete only this attempt's unpinned checkpoints after pointer publication."""
    protected = set(saved[-keep_last:]) | {latest}
    if best is not None:
        protected.add(Path(best["checkpoint"]).resolve())
    # A retained older checkpoint may refer to the previous best. Preserve that
    # dependency too, so every retained checkpoint remains independently resumable.
    for directory in tuple(protected):
        state = json.loads((directory / "trainer_state.json").read_text())
        previous_best = state.get("selection", {}).get("best")
        if previous_best is not None:
            protected.add(Path(previous_best["checkpoint"]).resolve())
    for directory in list(saved):
        if directory not in protected:
            shutil.rmtree(directory)
            saved.remove(directory)


def train_run(args, budget, log):
    final_fit_examples = validate_final_fit_examples(getattr(args, "final_fit_examples", 0))
    if final_fit_examples:
        from .sft import implementation_identity

        probe_implementation = implementation_identity()
    for key in (
        "steps",
        "grad_accumulation",
        "save_every",
        "keep_last",
        "eval_every",
        "eval_examples",
    ):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if any(not math.isfinite(v) or v <= 0 for v in (args.learning_rate, args.clip_grad)):
        raise ValueError("learning_rate and clip_grad must be finite and positive")
    if args.resume and args.resume_adapter:
        raise ValueError("Choose resume or resume_adapter, not both")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    root, data = Path(args.model).expanduser(), Path(args.data).expanduser()
    valid = Path(args.valid_data).expanduser() if args.valid_data else None
    if data.is_dir():
        if (
            valid is None
            and (data / "valid.jsonl").is_file()
            and (data / "valid.jsonl").stat().st_size
        ):
            valid = data / "valid.jsonl"
        data = data / "train.jsonl"
    resume_state = None
    if args.resume:
        resume_state = json.loads(
            (Path(args.resume).expanduser() / "trainer_state.json").read_text()
        )
        # An omitted override must not change an older checkpoint's core contract.
        if args.seq_length is None:
            args.seq_length = resume_state["contract"]["seq_length"]
    train_identity = training_data_identity(data, root)
    valid_identity = training_data_identity(valid, root) if valid else None
    args.seq_length, datasets = load_resolved_datasets(
        [data, *([valid] if valid else [])],
        root,
        seq_length=args.seq_length,
        truncate=args.truncate,
    )
    if training_data_identity(data, root) != train_identity:
        raise ValueError("Training data changed during loading")
    if valid and training_data_identity(valid, root) != valid_identity:
        raise ValueError("Validation data changed during loading")
    if final_fit_examples and implementation_identity() != probe_implementation:
        raise ValueError("Training implementation changed during loading")
    examples, validation = datasets[0], datasets[1] if valid else []
    if validation and {tuple(e["input_ids"]) for e in examples} & {
        tuple(e["input_ids"]) for e in validation
    }:
        raise ValueError("Identical token sequences occur in training and validation")
    selection_identity = validation_identity(
        valid, root, validation, args.eval_examples, input_identity=valid_identity
    )
    best = restored_best(resume_state, selection_identity) if resume_state else None
    contract = {
        k: getattr(args, k)
        for k in (
            "learning_rate",
            "grad_accumulation",
            "clip_grad",
            "seed",
            "seq_length",
            "truncate",
            "chunk_size",
            "no_checkpoint",
        )
    }
    contract.update(
        **train_identity,
        engine_semantics=ENGINE_SEMANTICS,
        mlx=mx.__version__,
        mlx_lm=importlib.metadata.version("mlx-lm"),
    )
    if resume_state is not None and resume_state.get("contract") != contract:
        raise ValueError(
            "Resume settings/data or engine semantics differ; use the recorded implementation for exact continuation"
        )
    mx.random.seed(args.seed)
    model = make_model(args, log)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    step, cursor = 0, 0
    if args.resume:
        state = load_checkpoint(model, optimizer, Path(args.resume).expanduser(), contract)
        step, cursor = state["step"], state["cursor"]
        # Use the state verified by load_checkpoint, not just the preflight read.
        best = restored_best(state, selection_identity)
        log(f"Resumed optimizer step {step}, sample cursor {cursor}")
    output.mkdir(parents=True)
    budget.abort_path = output / "memory-abort.json"
    atomic_text(output / "run_config.json", json.dumps(vars(args), indent=2) + "\n")
    history, saved = [], []
    latest = Path(args.resume).expanduser().resolve() if args.resume else None
    latest_step = step if latest else None
    start, status = time.monotonic(), "completed"

    def stop_requested():
        path = getattr(args, "stop_file", None)
        return bool(path and Path(path).expanduser().exists())

    def persist(record, *, improved=False):
        nonlocal latest, latest_step, best
        need = sum(v.nbytes for _, v in tree_flatten(model.trainable_parameters())) * 3 + (16 << 20)
        if shutil.disk_usage(output).free < need:
            raise OSError(f"Need {need / 2**20:.0f} MiB free to save a recoverable checkpoint")
        directory = output / f"step-{step:06d}"
        candidate = (
            {"step": step, "loss": record["validation"]["loss"], "checkpoint": str(directory)}
            if improved
            else best
        )
        save_checkpoint(
            model,
            optimizer,
            directory,
            {
                "step": step,
                "cursor": cursor,
                "contract": contract,
                "metrics": record,
                "selection": {"validation_identity": selection_identity, "best": candidate},
            },
        )
        latest, latest_step, best = directory, step, candidate
        saved.append(directory)
        atomic_text(output / "latest.txt", directory.name + "\n")
        if best is not None:
            atomic_text(output / "best.txt", best["checkpoint"] + "\n")
        prune_checkpoints(saved, keep_last=args.keep_last, latest=latest, best=best)

    if best is not None:
        atomic_text(output / "best.txt", best["checkpoint"] + "\n")
    for iteration in range(args.steps):
        if stop_requested():
            status = "stopped"
            break
        selected = sample_indices(args.grad_accumulation, cursor, len(examples), args.seed)
        record = optimizer_step(
            model,
            optimizer,
            [examples[i] for i in selected],
            clip=args.clip_grad,
            check=budget.check,
        )
        # Never use a stop callback inside optimizer_step: its final budget check
        # happens after mutation. Advance both counters before honoring a stop.
        step += 1
        cursor += args.grad_accumulation
        record.update(
            step=step,
            cursor=cursor,
            elapsed_seconds=time.monotonic() - start,
            peak_mlx_gib=mx.get_peak_memory() / 2**30,
        )
        improved = False
        stopping = stop_requested()
        if (
            validation
            and not stopping
            and (step % args.eval_every == 0 or iteration == args.steps - 1)
        ):
            record["validation"] = evaluate(
                model, validation[: args.eval_examples], check=budget.check
            )
            if not math.isfinite(record["validation"]["loss"]):
                raise FloatingPointError("Nonfinite validation loss")
            improved = best is None or record["validation"]["loss"] < best["loss"]
            log("Validation: " + json.dumps(record["validation"]))
        log(json.dumps(record))
        history.append(record)
        stopping = stopping or stop_requested()
        if (
            iteration == 0
            or step % args.save_every == 0
            or iteration == args.steps - 1
            or improved
            or stopping
        ):
            persist(record, improved=improved)
        atomic_text(output / "history.json", json.dumps(history, indent=2) + "\n")
        if stopping:
            status = "stopped"
            break
    # Covers a request arriving after the last per-step poll but before the next
    # iteration starts. Only a completed optimizer state is ever checkpointed.
    if status == "stopped" and history and latest_step != step:
        persist(history[-1])
    fit_probe = None
    if final_fit_examples and history and status == "completed" and not stop_requested():
        if training_data_identity(data, root) != train_identity:
            raise ValueError("Training data changed before final fit probe")
        if implementation_identity() != probe_implementation:
            raise ValueError("Training implementation changed before final fit probe")
        log(f"Final training-mode fit probe: first {min(final_fit_examples, len(examples))} rows")
        fit_probe = run_final_fit_probe(
            model,
            examples,
            count=final_fit_examples,
            evaluate=evaluate,
            check=budget.check,
            metadata={
                "checkpoint": str(latest.resolve()),
                "checkpoint_files_sha256": {
                    name: file_sha256(latest / name)
                    for name in ("adapter.safetensors", "adapter_config.json", "trainer_state.json")
                },
                "step": step,
                "data": str(data.resolve()),
                "input_identity": train_identity,
                "implementation": probe_implementation,
                "seq_length": args.seq_length,
                "activation_mode": model.adapter_config.get("activation_mode", "checkpoint"),
                "chunk_size": args.chunk_size,
            },
        )
        if training_data_identity(data, root) != train_identity:
            raise ValueError("Training data changed during final fit probe")
        if implementation_identity() != probe_implementation:
            raise ValueError("Training implementation changed during final fit probe")
        atomic_text(output / "training-fit-probe.json", json.dumps(fit_probe, indent=2) + "\n")
        log("Final training-mode fit probe completed; fresh uncached comparison remains required")
    reload_exact = None
    if args.verify_reload and history and status == "completed":
        probe = [examples[0]]
        before = evaluate(model, probe, check=budget.check)
        for name, value in tree_flatten(model.trainable_parameters()):
            model.load_weights([(name, mx.zeros_like(value))], strict=False)
        mx.eval(model.trainable_parameters())
        load_adapter(model, latest)
        after = evaluate(model, probe, check=budget.check)
        reload_exact = before == after
        if not reload_exact:
            raise AssertionError("Adapter prediction reload mismatch")
        log("Adapter prediction reload is exact")
    report = model.report()
    report.update(
        status=status,
        history=history,
        completed_step=step,
        additional_steps=len(history),
        requested_steps=args.steps,
        elapsed_seconds=time.monotonic() - start,
        adapter_prediction_reload_exact=reload_exact,
        training_fit_probe=fit_probe,
        optimizer_resume_supported=True,
        checkpoint=str(latest) if latest else None,
        best_checkpoint=best["checkpoint"] if best else None,
        best_validation_loss=best["loss"] if best else None,
        selection={"validation_identity": selection_identity, "best": best},
        seq_length=args.seq_length,
        budget_gib=budget.limit / 2**30,
        observed_memory_gib=budget.usage() / 2**30,
        mlx=mx.__version__,
        mlx_lm=importlib.metadata.version("mlx-lm"),
        device=mx.device_info()["device_name"],
        contract=contract,
        resources=resources_from_args(args),
    )
    if final_fit_examples:
        if training_data_identity(data, root) != train_identity:
            raise ValueError("Training data changed before final report")
        if implementation_identity() != probe_implementation:
            raise ValueError("Training implementation changed before final report")
    atomic_text(output / "training.json", json.dumps(report, indent=2) + "\n")
    return report


def doctor(args, budget):
    from .engram import EngramBank
    from .storage import ExpertBank, TensorArchive

    root = Path(args.model).expanduser()
    if not (root / "expert-manifest-v2.json").exists():
        raise ValueError(
            "Full-model training currently requires an mtplx streaming repack; ordinary EXL3 shards support matrix validation only"
        )
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("Full-model trainer currently supports DeepSeek V4.1 only")
    bank, archive = ExpertBank(root), TensorArchive(root)
    engram = EngramBank(root, config["text_config"])
    resident = TensorArchive(root / "engram/engram-residents.safetensors")
    # Header/size checks only; never materialize entire banks.
    files = set(archive.fingerprints) | set(resident.fingerprints) | {bank.file}
    files |= {desc[0] for desc in engram.layers.values()}
    verified = {}
    if args.verify_records:
        for layer, expert in ((0, 0), (bank.dims["n_layers"] - 1, bank.dims["n_experts"] - 1)):
            verified[f"{layer}/{expert}"] = bank.verify_record(layer, expert)
    return {
        "layout_valid": True,
        "architecture": config["model_type"],
        "expert_source": bank.source,
        "expert_bank_gib": bank.total_bytes / 2**30,
        "checkpoint_files_gib": sum(f.stat().st_size for f in files) / 2**30,
        "disk_free_gib": shutil.disk_usage(root).free / 2**30,
        "budget_gib": budget.limit / 2**30,
        "startup_headroom_check_passed": True,
        "verified_expert_records": verified,
        "scope": "headers, bank sizes, token-map hashes and optional sampled record hashes; not full-model numerical parity",
        "context_limit": MAX_CONTEXT_TOKENS,
        "mlx": mx.__version__,
        "mlx_lm": importlib.metadata.version("mlx-lm"),
    }


def generate(args, model, budget):
    if not 1 <= args.max_tokens <= MAX_CONTEXT_TOKENS:
        raise ValueError("max_tokens must be in 1..32768")
    messages_file = getattr(args, "messages_file", None)
    tools_file = getattr(args, "tools_file", None)
    if tools_file and args.raw_prompt:
        raise ValueError("--raw-prompt cannot be combined with --tools-file")
    tools, tools_identity = None, None
    if tools_file:
        from .sft_data import _parse_json
        from .tool_data import validate_tools

        tools_path = Path(tools_file).expanduser()
        raw_tools = tools_path.read_bytes()
        tools_identity = hashlib.sha256(raw_tools).hexdigest()
        tools = validate_tools(_parse_json(raw_tools.decode("utf-8")))
    if messages_file and args.raw_prompt:
        raise ValueError("--raw-prompt cannot be combined with --messages-file")
    messages = (
        json.loads(Path(messages_file).expanduser().read_text())
        if messages_file
        else [{"role": "user", "content": args.prompt}]
    )
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
            for message in messages
        )
        or messages[-1]["role"] != "user"
    ):
        raise ValueError("Use plain-text system/user/assistant messages ending in a user message")
    tokenizer = load_tokenizer(Path(args.model).expanduser())
    if args.raw_prompt:
        ids = [tokenizer.bos_token_id] + tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        ids = tokenizer.apply_chat_template(
            messages,
            **({"tools": tools} if tools is not None else {}),
            add_generation_prompt=True,
            return_dict=False,
        )
    if len(ids) + args.max_tokens > MAX_CONTEXT_TOKENS:
        raise ValueError("Prompt plus generation must fit the 32768-token context limit")
    if 129264 in ids:
        raise ValueError("Image inputs are unsupported")
    prompt_ids = list(ids)
    model.eval()
    answer = []
    for _ in range(args.max_tokens):
        budget.check()
        inputs = mx.array([ids], dtype=mx.int32)
        if hasattr(model, "hidden_states"):
            hidden = model.hidden_states(inputs)
            logits = model.project_logits(hidden[:, -1:])
            del hidden
        else:
            logits = model(inputs)
        if not mx.all(mx.isfinite(logits[:, -1])).item():
            raise FloatingPointError("Nonfinite generation logits")
        token = mx.argmax(logits[0, -1]).item()
        del logits
        ids.append(token)
        answer.append(token)
        print(tokenizer.decode([token]), end="", flush=True, file=sys.stderr)
        if token == tokenizer.eos_token_id:
            break
    print(file=sys.stderr)
    return {
        "text": tokenizer.decode(answer, skip_special_tokens=True),
        "raw_text": tokenizer.decode(answer, skip_special_tokens=False),
        "tools": tools,
        "tools_file_sha256": tools_identity,
        "generated_ids": answer,
        "prompt_tokens": len(ids) - len(answer),
        "prompt_ids": prompt_ids,
        "prompt_messages": None if args.raw_prompt else messages,
        "max_new_tokens": args.max_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "stop_reason": "eos" if answer[-1] == tokenizer.eos_token_id else "max_tokens",
        "method": "greedy full-prefix recomputation",
        **model.report(),
    }


def main(argv=None):
    invocation = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(invocation)
    start = time.monotonic()

    def log(message):
        print(f"[{time.monotonic() - start:.1f}s] {message}", flush=True, file=sys.stderr)

    try:
        if args.command == "prepare":
            with MemoryBudget(4, cache_mib=0, reserve_gib=16):
                report = prepare(args)
        else:
            if getattr(args, "report", None) and Path(args.report).expanduser().exists():
                raise FileExistsError("Report already exists; choose a new path")
            with MemoryBudget(
                args.memory_gib, cache_mib=args.allocator_cache_mib, reserve_gib=args.reserve_gib
            ) as budget:
                if args.command == "doctor":
                    report = doctor(args, budget)
                elif args.command == "train":
                    report = train_run(args, budget, log)
                else:
                    if args.command == "evaluate":
                        if args.max_examples < 0:
                            raise ValueError("max_examples must be nonnegative")
                        data_path = Path(args.data).expanduser()
                        model_root = Path(args.model).expanduser()
                        from .sft import implementation_identity

                        # Bind the files before reading/tokenizing them. A receipt
                        # captured only after loading could attest different bytes.
                        input_identity = training_data_identity(data_path, model_root)
                        evaluated_implementation = implementation_identity()
                        args.seq_length, datasets = load_resolved_datasets(
                            [data_path],
                            model_root,
                            seq_length=args.seq_length,
                        )
                        examples = datasets[0]
                        if args.max_examples:
                            examples = examples[: args.max_examples]
                        if training_data_identity(data_path, model_root) != input_identity:
                            raise ValueError("Evaluation data changed during execution")
                        if implementation_identity() != evaluated_implementation:
                            raise ValueError("Evaluation implementation changed during execution")
                    model = make_model(args, log)
                    report = (
                        evaluate(
                            model,
                            examples,
                            check=budget.check,
                            include_token_metrics=args.token_metrics,
                        )
                        | model.report()
                        if args.command == "evaluate"
                        else generate(args, model, budget)
                    )
                    report.update(
                        adapter=args.adapter,
                        elapsed_seconds=time.monotonic() - start,
                        budget_gib=budget.limit / 2**30,
                        observed_memory_gib=budget.usage() / 2**30,
                    )
                    if args.command == "evaluate":
                        if training_data_identity(data_path, model_root) != input_identity:
                            raise ValueError("Evaluation data changed during execution")
                        if implementation_identity() != evaluated_implementation:
                            raise ValueError("Evaluation implementation changed during execution")
                        report["evaluation_context"] = {
                            "argv": invocation,
                            "process_id": os.getpid(),
                            "data": str(Path(args.data).expanduser().resolve()),
                            "input_identity": input_identity,
                            "selected_examples_sha256": hashlib.sha256(
                                json.dumps(examples, sort_keys=True, separators=(",", ":")).encode()
                            ).hexdigest(),
                            "seq_length": args.seq_length,
                            "requested_max_examples": args.max_examples,
                            "implementation": evaluated_implementation,
                        }
        if args.command != "prepare":
            report["resources"] = resources_from_args(args)
        if getattr(args, "report", None):
            path = Path(args.report).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    except (MemoryError, ValueError, OSError) as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from error


if __name__ == "__main__":
    main()
