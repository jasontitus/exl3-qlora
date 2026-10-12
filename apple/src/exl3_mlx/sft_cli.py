"""A project workflow for supervised fine-tuning on the supported EXL3 backend."""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from .sft import SFTConfig, SFTTrainer

COMMANDS = {"finetune", "resume", "compare", "chat", "export-adapter", "status", "stop", "studio"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("finetune", help="Import data, train, and compare a new project")
    start.add_argument("--model", help="Local DeepSeek V4.1 streaming repack directory")
    start.add_argument("--data", help="CSV, JSONL, JSON, or directory of named splits")
    start.add_argument("--output", help="New project directory")
    start.add_argument("--config", help="JSON SFTConfig settings; explicit flags override")
    start.add_argument("--steps", type=int, help="Total optimizer steps (default 20)")
    start.add_argument(
        "--max-length", type=int, help="Total serialized tokens; inferred, at most 512"
    )
    start.add_argument("--learning-rate", type=float)
    start.add_argument("--gradient-accumulation-steps", type=int)
    start.add_argument("--rank", type=int)
    start.add_argument("--lora-alpha", type=float)
    start.add_argument("--memory-gib", type=float)
    start.add_argument("--reserve-gib", type=float)
    start.add_argument("--eval-steps", type=int)
    start.add_argument("--save-steps", type=int)
    start.add_argument("--save-total-limit", type=int)
    start.add_argument("--max-new-tokens", type=int)
    start.add_argument("--comparison-examples", type=int)
    start.add_argument("--prompt-column")
    start.add_argument("--response-column")
    start.add_argument("--group-column")
    start.add_argument(
        "--demo",
        action="store_true",
        default=None,
        help="Tiny training-only demonstration; no quality claim",
    )
    start.add_argument(
        "--prepare-only",
        action="store_true",
        help="Inspect token counts, loss masks and splits before training",
    )
    start.add_argument("--no-compare", action="store_true", help="Train now; run compare later")
    resume = sub.add_parser(
        "resume", help="Continue the latest checkpoint, including Adam and data cursor"
    )
    resume.add_argument("project")
    resume.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Total desired updates, including completed updates",
    )
    resume.add_argument("--no-compare", action="store_true")
    resume.add_argument("--request-start-ns", type=int, help=argparse.SUPPRESS)
    for name, help_text in (
        ("compare", "Compare base and selected adapter"),
        ("status", "Show saved project status"),
        ("stop", "Request a checkpoint after the current update"),
    ):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("project")
    chat = sub.add_parser("chat", help="Generate from the project's selected adapter")
    chat.add_argument("project")
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--max-new-tokens", type=int)
    chat.add_argument("--base", action="store_true")
    export = sub.add_parser("export-adapter", help="Export an adapter for this EXL3 engine")
    export.add_argument("project")
    export.add_argument("--output", required=True)
    studio = sub.add_parser("studio", help="Open the local guided browser interface")
    studio.add_argument(
        "--port", type=int, default=0, help="Local port (default chooses an available port)"
    )
    studio.add_argument("--no-browser", action="store_true")
    return parser


def start_trainer(args, parser):
    values = {}
    if args.config:
        values = json.loads(Path(args.config).expanduser().read_text())
        allowed = {f.name for f in dataclasses.fields(SFTConfig)}
        if not isinstance(values, dict) or set(values) - allowed:
            parser.error("Config must contain only SFTConfig fields")
    for name in ("model", "data", "output"):
        if getattr(args, name) is None:
            if sys.stdin.isatty() and not args.config:
                setattr(args, name, input(f"{name.title()} path: ").strip())
            elif name == "output" and "output_dir" in values:
                args.output = values["output_dir"]
            else:
                parser.error(f"--{name} is required")
    for field in dataclasses.fields(SFTConfig):
        flag = "steps" if field.name == "max_steps" else field.name
        value = getattr(args, flag, None)
        if value is not None:
            values[field.name] = value
    values["output_dir"] = args.output
    return SFTTrainer(model=args.model, args=SFTConfig(**values), train_dataset=args.data)


def main(argv=None):
    started = time.time_ns()
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "studio":
            from .studio import serve

            return serve(port=args.port, open_browser=not args.no_browser)
        if args.command == "finetune":
            trainer = start_trainer(args, parser)
            if args.prepare_only:
                manifest = trainer.prepare()
                result = {
                    "project": str(trainer.root),
                    "status": "prepared",
                    "counts": manifest["data"]["counts"],
                    "max_length": manifest["data"]["max_length"],
                    "evaluation_scope": manifest["data"]["evaluation_scope"],
                    "next": f"exl3-mlx resume {trainer.root} --steps {trainer.args.max_steps}",
                }
            else:
                result = trainer.train(compare=not args.no_compare, request_start_ns=started)
        elif args.command == "stop":
            root = Path(args.project).expanduser().resolve()
            if not (root / "project.json").is_file():
                raise ValueError("Not a fine-tuning project")
            (root / "stop.request").touch()
            result = {
                "status": "stop_requested",
                "note": "The current update may take several minutes; the last completed update will be saved.",
            }
        else:
            trainer = SFTTrainer.from_project(args.project, max_steps=getattr(args, "steps", None))
            if args.command == "resume":
                result = trainer.train(
                    resume_from_checkpoint=True,
                    compare=not args.no_compare,
                    request_start_ns=args.request_start_ns or started,
                )
            elif args.command == "compare":
                result = trainer.evaluate()
            elif args.command == "chat":
                result = trainer.generate(
                    args.prompt, max_new_tokens=args.max_new_tokens, base=args.base
                )
            elif args.command == "export-adapter":
                result = {
                    "adapter": trainer.save_model(args.output),
                    "format": "EXL3 engine adapter; not PEFT or mlx_lm",
                }
            else:
                result = trainer.status()
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
