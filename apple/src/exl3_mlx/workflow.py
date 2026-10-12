"""A bounded, local, end-to-end DeepSeek EXL3 adapter workflow."""

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .data import load_dataset, load_tokenizer, prepare, training_data_identity
from .deepseek import DeepSeekTextModel
from .memory import MemoryBudget
from .train import load_adapter, load_adapter_config
from .training import evaluate, load_checkpoint, optimizer_step, sample_indices, save_checkpoint


def common(parser, *, inference=False):
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-gib", type=float, default=16.0)
    parser.add_argument("--allocator-cache-mib", type=int, default=64)
    parser.add_argument("--reserve-gib", type=float, default=32.0)
    parser.add_argument("--cache-mib", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--verbose", action="store_true", help="Log every layer")
    if inference:
        parser.add_argument("--adapter", help="Saved step directory; omit to evaluate the base")
        parser.add_argument("--report", help="Write a new JSON report")


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
    train.add_argument("--seq-length", type=int, default=16)
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
        "--verify-reload", action="store_true", help="Check post-training prediction reload"
    )
    ev = sub.add_parser("evaluate", help="Token-weighted NLL/perplexity on held-out examples")
    common(ev, inference=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--seq-length", type=int, default=128)
    ev.add_argument("--max-examples", type=int, default=0, help="0 evaluates all rows")
    gen = sub.add_parser(
        "generate", help="Greedy text generation; recomputes the short prefix each token"
    )
    common(gen, inference=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=int, default=16)
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
        known, _ = cp.parse_known_args(argv[1:])
        config = json.loads(Path(known.config).expanduser().read_text())
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
    return parser.parse_args(argv)


def model_settings(args):
    adapter = (
        getattr(args, "adapter", None)
        or getattr(args, "resume", None)
        or getattr(args, "resume_adapter", None)
    )
    keys = ("rank", "alpha", "adapter_layers", "expert_adapter_layers")
    defaults = dict(rank=8, alpha=16.0, adapter_layers=None, expert_adapter_layers=[])
    if adapter:
        c = load_adapter_config(Path(adapter).expanduser())
        for key in keys:
            value = getattr(args, key, None)
            if value is not None and value != c[key]:
                raise ValueError(f"{key} differs from saved adapter")
        return {k: c[k] for k in keys}
    return {
        k: getattr(args, k, None) if getattr(args, k, None) is not None else defaults[k]
        for k in keys
    }


def make_model(args, log):
    if args.cache_mib < 0 or args.chunk_size < 128 or args.chunk_size % 128:
        raise ValueError("Use nonnegative cache_mib and chunk_size divisible by 128")
    settings = model_settings(args)
    log("Loading frozen trunk; EXL3 experts and Engram rows stay streamed")
    model = DeepSeekTextModel(
        Path(args.model).expanduser(),
        **settings,
        cache_mib=args.cache_mib,
        chunk_size=args.chunk_size,
        checkpoint_blocks=not getattr(args, "no_checkpoint", False),
        progress=log if args.verbose else None,
    )
    adapter = getattr(args, "adapter", None) or getattr(args, "resume_adapter", None)
    if adapter:
        load_adapter(model, Path(adapter).expanduser())
    log(f"Loaded; {model.report()['trainable_parameters']:,} adapter parameters")
    return model


def train_run(args, budget, log):
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
    output = Path(args.output).expanduser()
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
    examples = load_dataset(data, root, seq_length=args.seq_length, truncate=args.truncate)
    validation = (
        load_dataset(valid, root, seq_length=args.seq_length, truncate=args.truncate)
        if valid
        else []
    )
    if validation and {tuple(e["input_ids"]) for e in examples} & {
        tuple(e["input_ids"]) for e in validation
    }:
        raise ValueError("Identical token sequences occur in training and validation")
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
        **training_data_identity(data, root),
        mlx=mx.__version__,
        mlx_lm=importlib.metadata.version("mlx-lm"),
    )
    mx.random.seed(args.seed)
    model = make_model(args, log)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    step, cursor = 0, 0
    if args.resume:
        state = load_checkpoint(model, optimizer, Path(args.resume).expanduser(), contract)
        step, cursor = state["step"], state["cursor"]
        log(f"Resumed optimizer step {step}, sample cursor {cursor}")
    output.mkdir(parents=True)
    budget.abort_path = output / "memory-abort.json"
    (output / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    history, saved = [], []
    start = time.monotonic()
    for iteration in range(args.steps):
        selected = sample_indices(args.grad_accumulation, cursor, len(examples), args.seed)
        record = optimizer_step(
            model,
            optimizer,
            [examples[i] for i in selected],
            clip=args.clip_grad,
            check=budget.check,
        )
        step += 1
        cursor += args.grad_accumulation
        record.update(
            step=step,
            cursor=cursor,
            elapsed_seconds=time.monotonic() - start,
            peak_mlx_gib=mx.get_peak_memory() / 2**30,
        )
        log(json.dumps(record))
        history.append(record)
        if iteration == 0 or step % args.save_every == 0 or iteration == args.steps - 1:
            need = sum(v.nbytes for _, v in tree_flatten(model.trainable_parameters())) * 3 + (
                16 << 20
            )
            if shutil.disk_usage(output).free < need:
                raise OSError(f"Need {need / 2**20:.0f} MiB free to save a recoverable checkpoint")
            directory = output / f"step-{step:06d}"
            save_checkpoint(
                model,
                optimizer,
                directory,
                {"step": step, "cursor": cursor, "contract": contract, "metrics": record},
            )
            saved.append(directory)
            (output / "latest.txt").write_text(directory.name + "\n")
            while len(saved) > args.keep_last:
                shutil.rmtree(saved.pop(0))
        if validation and (step % args.eval_every == 0 or iteration == args.steps - 1):
            record["validation"] = evaluate(
                model, validation[: args.eval_examples], check=budget.check
            )
            log("Validation: " + json.dumps(record["validation"]))
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    reload_exact = None
    if args.verify_reload:
        probe = [examples[0]]
        before = evaluate(model, probe, check=budget.check)
        # Perturb adapters first to ensure the reload actually restores their values.
        for name, value in tree_flatten(model.trainable_parameters()):
            model.load_weights([(name, mx.zeros_like(value))], strict=False)
        mx.eval(model.trainable_parameters())
        load_adapter(model, saved[-1])
        after = evaluate(model, probe, check=budget.check)
        reload_exact = before == after
        if not reload_exact:
            raise AssertionError("Adapter prediction reload mismatch")
        log("Adapter prediction reload is exact")
    report = model.report()
    report.update(
        history=history,
        completed_step=step,
        additional_steps=args.steps,
        elapsed_seconds=time.monotonic() - start,
        adapter_prediction_reload_exact=reload_exact,
        optimizer_resume_supported=True,
        checkpoint=str(saved[-1].resolve()),
        budget_gib=budget.limit / 2**30,
        observed_memory_gib=budget.usage() / 2**30,
        mlx=mx.__version__,
        mlx_lm=importlib.metadata.version("mlx-lm"),
        device=mx.device_info()["device_name"],
        contract=contract,
    )
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
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
        "context_limit": 512,
        "mlx": mx.__version__,
        "mlx_lm": importlib.metadata.version("mlx-lm"),
    }


def generate(args, model, budget):
    if not 1 <= args.max_tokens <= 512:
        raise ValueError("max_tokens must be in 1..512")
    tokenizer = load_tokenizer(Path(args.model).expanduser())
    if args.raw_prompt:
        ids = [tokenizer.bos_token_id] + tokenizer.encode(args.prompt, add_special_tokens=False)
    else:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_dict=False,
        )
    if len(ids) + args.max_tokens > 512:
        raise ValueError("Prompt plus generation must fit the 512-token context limit")
    if 129264 in ids:
        raise ValueError("Image inputs are unsupported")
    prompt_ids = list(ids)
    model.eval()
    answer = []
    for _ in range(args.max_tokens):
        budget.check()
        logits = model(mx.array([ids], dtype=mx.int32))
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
        "generated_ids": answer,
        "prompt_tokens": len(ids) - len(answer),
        "prompt_ids": prompt_ids,
        "max_new_tokens": args.max_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "stop_reason": "eos" if answer[-1] == tokenizer.eos_token_id else "max_tokens",
        "method": "greedy full-prefix recomputation",
        **model.report(),
    }


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
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
                        examples = load_dataset(
                            Path(args.data).expanduser(),
                            Path(args.model).expanduser(),
                            seq_length=args.seq_length,
                        )
                        if args.max_examples:
                            examples = examples[: args.max_examples]
                    model = make_model(args, log)
                    report = (
                        evaluate(model, examples, check=budget.check) | model.report()
                        if args.command == "evaluate"
                        else generate(args, model, budget)
                    )
                    report.update(
                        adapter=args.adapter,
                        elapsed_seconds=time.monotonic() - start,
                        budget_gib=budget.limit / 2**30,
                        observed_memory_gib=budget.usage() / 2**30,
                    )
        if getattr(args, "report", None):
            path = Path(args.report).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    except (MemoryError, ValueError, OSError) as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from error


if __name__ == "__main__":
    main()
