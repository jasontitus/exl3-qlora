"""Import ktune evidence, prepare complete examples and run standard SFT projects."""

import argparse
import json
import sys
from pathlib import Path

from .context import MAX_CONTEXT_TOKENS
from .resources import add_resource_arguments, resources_from_args
from .sft import SFTConfig, SFTTrainer


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot", help="Copy a bounded evidence archive over SSH")
    snapshot.add_argument("--source", required=True, help="host:/absolute/path/to/ktune")
    snapshot.add_argument("--output", required=True)
    for name in ("prepare", "finetune"):
        item = sub.add_parser(
            name,
            help="Audit/import examples" if name == "prepare" else "Import and train a project",
        )
        item.add_argument("--source", required=True, help="Local ktune root or downloaded snapshot")
        item.add_argument("--model", required=True)
        item.add_argument(
            "--output",
            required=True,
            help="New dataset folder" if name == "prepare" else "New training project",
        )
        item.add_argument(
            "--max-length",
            type=int,
            help=f"Infer full example length, ceiling {MAX_CONTEXT_TOKENS}",
        )
        item.add_argument(
            "--max-examples",
            type=int,
            help="Select this many shortest complete eligible examples; selection is recorded",
        )
        item.add_argument("--context-mode", choices=("hunks", "parent_files"), default="hunks")
        item.add_argument(
            "--parent-repo",
            help="Pinned Git repository: local path or host:/absolute/path, for complete pre-change files",
        )
        item.add_argument(
            "--train-until", help="ISO-8601 cutoff; quarantine mixed or undated lineages"
        )
        item.add_argument(
            "--tier",
            action="append",
            dest="tiers",
            help="Explicit evidence tier; repeat to include more",
        )
        item.add_argument(
            "--case-id",
            action="append",
            dest="selected_case_ids",
            help="Exact archive case ID for a reproducible pilot; repeat for more",
        )
        if name == "finetune":
            item.add_argument(
                "--steps", type=int, default=2, help="Total updates (default 2 for an initial run)"
            )
            add_resource_arguments(item)
            item.add_argument(
                "--chunk-size",
                type=int,
                default=512,
                help="EXL3 column chunk; use a new project to change it",
            )
            item.add_argument("--learning-rate", type=float, default=1e-5)
            item.add_argument("--rank", type=int, default=4)
            item.add_argument(
                "--fit-examples",
                type=int,
                default=1,
                help="First training rows to evaluate with --verify-fit (default 1)",
            )
            item.add_argument("--prepare-only", action="store_true")
            item.add_argument(
                "--verify-fit",
                action="store_true",
                help="Run fresh base/adapter training NLL checks; not a quality test",
            )
    verify = sub.add_parser(
        "verify", help="Compare base/adapter likelihood on training examples (not quality)"
    )
    verify.add_argument("project")
    audit = sub.add_parser("audit-evidence", help="Validate archived benchmark receipts")
    audit.add_argument("--kind", choices=("official", "local"), required=True)
    audit.add_argument("--input", required=True)
    audit.add_argument("--output", required=True)
    export = sub.add_parser(
        "export-edit",
        help="Convert a generated unified diff to ktune SEARCH/REPLACE against a pinned parent",
    )
    export.add_argument("--repo", required=True)
    export.add_argument("--parent", required=True, help="Full 40-character commit SHA")
    export.add_argument("--patch", required=True)
    export.add_argument(
        "--output", required=True, help="New .edit file path (receipt written alongside)"
    )
    export.add_argument("--allow-path", action="append", dest="allowed_paths")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "export-edit":
            from .ktune_patch import export_ktune_edit

            report = export_ktune_edit(
                args.repo, args.parent, args.patch, args.output, allowed_paths=args.allowed_paths
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "audit-evidence":
            from .ktune_eval import main as audit_main

            return audit_main(["--kind", args.kind, "--input", args.input, "--output", args.output])
        if args.command == "snapshot":
            from .ktune_data import snapshot_ktune

            report = snapshot_ktune(args.source, args.output)
        elif args.command == "verify":
            report = SFTTrainer.from_project(args.project).verify_training_fit()
        else:
            from .ktune_data import DEFAULT_TIERS, import_ktune

            output = Path(args.output).expanduser().resolve()
            if output.exists():
                raise FileExistsError(f"Choose a new output folder: {output}")
            dataset = (
                output
                if args.command == "prepare"
                else output.with_name(output.name + "-ktune-data")
            )
            report = import_ktune(
                args.source,
                args.model,
                dataset,
                max_length=args.max_length,
                max_examples=args.max_examples,
                tiers=args.tiers or DEFAULT_TIERS,
                train_until=args.train_until,
                context_mode=args.context_mode,
                parent_repo=args.parent_repo,
                selected_case_ids=args.selected_case_ids,
            )
            if args.command == "finetune":
                if not report["ready_for_sft"]:
                    raise ValueError(
                        f"No eligible complete examples; inspect {dataset / 'ktune-import.json'}"
                    )
                trainer = SFTTrainer(
                    model=args.model,
                    args=SFTConfig(
                        output_dir=str(output),
                        max_steps=args.steps,
                        max_length=report["max_length"],
                        learning_rate=args.learning_rate,
                        **resources_from_args(args, default_memory_gib=32.0),
                        rank=args.rank,
                        chunk_size=args.chunk_size,
                        lora_alpha=2.0 * args.rank,
                        save_steps=1,
                        comparison_examples=args.fit_examples,
                    ),
                    train_dataset=dataset,
                )
                manifest = trainer.prepare(provenance_required=True)
                trainer.bind_provenance(dataset, report)
                report = {
                    "project": str(output),
                    "dataset": str(dataset),
                    "max_length": manifest["data"]["max_length"],
                    "evaluation_scope": "training_only",
                    "quality_certified": False,
                    "status": "prepared",
                }
                if not args.prepare_only:
                    report["training"] = trainer.train(compare=False)
                    report["status"] = report["training"]["status"]
                    if args.verify_fit and report["status"] == "completed":
                        report["fit"] = trainer.verify_training_fit()
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
