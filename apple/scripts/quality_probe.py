"""A small regression probe, not quality certification; uses existing EXL3 APIs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

SCOPE = "small regression probe, not quality certification"
FIELDS = {
    "case_id",
    "source_family",
    "category",
    "prompt",
    "expected",
    "max_new_tokens",
}
SETTINGS = {
    "memory_gib": 16,
    "reserve_gib": 32,
    "allocator_cache_mib": 64,
    "cache_mib": 128,
    "chunk_size": 512,
    "seed": 8,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cases(path):
    cases, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if len(line) > 1 << 20:
                raise ValueError(f"Line {number}: fixture row exceeds 1 MiB")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != FIELDS:
                raise ValueError(f"Line {number}: use exactly {sorted(FIELDS)}")
            for key in FIELDS - {"max_new_tokens"}:
                if not isinstance(row[key], str) or not row[key].strip():
                    raise ValueError(f"Line {number}: {key} must be a nonempty string")
            if row["case_id"] != row["case_id"].strip() or row["case_id"] in seen:
                raise ValueError(f"Line {number}: duplicate or whitespace-padded case_id")
            cap = row["max_new_tokens"]
            if type(cap) is not int or not 1 <= cap <= 512:
                raise ValueError(f"Line {number}: max_new_tokens must be an integer in 1..512")
            seen.add(row["case_id"])
            cases.append(row)
    if not cases:
        raise ValueError("Fixture contains no cases")
    return cases


def validate_tokenized_case(case, example, prompt_ids, eos_token_id, vocab_size):
    """Pure validation shared by native execution and CPU tests; never truncate."""
    ids, start = example["input_ids"], example["loss_start"]
    if type(eos_token_id) is not int or not 0 <= eos_token_id < vocab_size:
        raise ValueError("Tokenizer must provide one valid integer EOS token")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            type(t) is not int or not 0 <= t < vocab_size or t == 129264 for t in ids + prompt_ids
        )
    ):
        raise ValueError(f"{case['case_id']}: invalid or unsupported text token IDs")
    if not 1 <= start < len(ids) or ids[:start] != prompt_ids:
        raise ValueError(f"{case['case_id']}: target template differs from generation prefix")
    if len(ids) - 1 > 512 or len(prompt_ids) + case["max_new_tokens"] > 512:
        raise ValueError(f"{case['case_id']}: context exceeds 512; shortening must be explicit")
    if ids[-1] != eos_token_id:
        raise ValueError(f"{case['case_id']}: expected answer must end with tokenizer EOS")
    if len(ids) - start > case["max_new_tokens"]:
        raise ValueError(f"{case['case_id']}: generation cap cannot fit expected answer plus EOS")
    return {
        **case,
        "example": example,
        "prompt_ids": prompt_ids,
        "target_ids": ids[start:],
        "eos_token_id": eos_token_id,
    }


def grade_answer(expected, text, generated_ids, eos_token_id, max_new_tokens):
    if not generated_ids or len(generated_ids) > max_new_tokens:
        raise ValueError("Generation returned an invalid number of tokens")
    ended = generated_ids[-1] == eos_token_id
    if eos_token_id in generated_ids[:-1]:
        raise ValueError("Generation continued after EOS")
    if not ended and len(generated_ids) != max_new_tokens:
        raise ValueError("Generation stopped before the token cap without EOS")
    exact = text.strip() == expected.strip()
    return {
        "exact_text_match": exact,
        "ended_with_eos": ended,
        "truncated": not ended and len(generated_ids) == max_new_tokens,
        "exact_answer_and_eos": exact and ended,
        "grading": "case-sensitive stripped exact text; terminated pass additionally requires EOS",
    }


def validate_generation_evidence(case, generation):
    """Bind grading to the actual inputs used by the public generation function."""
    if generation["prompt_ids"] != case["prompt_ids"]:
        raise ValueError("Generation prompt IDs differ from recorded fixture IDs")
    if generation["prompt_tokens"] != len(case["prompt_ids"]):
        raise ValueError("Generation prompt length differs from recorded fixture IDs")
    if generation["max_new_tokens"] != case["max_new_tokens"]:
        raise ValueError("Generation cap differs from fixture")
    if generation["eos_token_id"] != case["eos_token_id"]:
        raise ValueError("Generation EOS token differs from fixture")
    grade = grade_answer(
        case["expected"],
        generation["text"],
        generation["generated_ids"],
        case["eos_token_id"],
        case["max_new_tokens"],
    )
    expected_stop = "eos" if grade["ended_with_eos"] else "max_tokens"
    if generation["stop_reason"] != expected_stop:
        raise ValueError("Generation stop reason disagrees with output tokens")
    return grade


def paired_summary(cases, records):
    expected_keys = {(phase, c["case_id"]) for phase in ("base", "adapter") for c in cases}
    by_key = {(r["phase"], r["case_id"]): r for r in records}
    if len(by_key) != len(records) or set(by_key) != expected_keys:
        raise ValueError("Paired summary requires exactly one base and adapter result per case")
    pairs = []
    for case in cases:
        base, adapter = (by_key[(p, case["case_id"])] for p in ("base", "adapter"))
        for key in ("prompt_ids", "target_ids", "max_new_tokens", "expected"):
            if base[key] != adapter[key]:
                raise ValueError(f"{case['case_id']}: comparison changed {key}")
        if base["evaluation"]["supervised_tokens"] != adapter["evaluation"]["supervised_tokens"]:
            raise ValueError("Paired target counts differ")
        bp, ap = (r["generation"]["exact_answer_and_eos"] for r in (base, adapter))
        pairs.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "source_family": case["source_family"],
                "expected": case["expected"],
                "base_pass": bp,
                "adapter_pass": ap,
                "base_pass_adapter_fail": bp and not ap,
                "base_fail_adapter_pass": not bp and ap,
                "base_text": base["generation"]["text"],
                "adapter_text": adapter["generation"]["text"],
                "base_truncated": base["generation"]["truncated"],
                "adapter_truncated": adapter["generation"]["truncated"],
                "output_changed": base["generation"]["generated_ids"]
                != adapter["generation"]["generated_ids"],
                "nll_delta_adapter_minus_base": adapter["evaluation"]["loss"]
                - base["evaluation"]["loss"],
                "unexpected_violet": case["expected"].strip() != "violet"
                and adapter["generation"]["text"].strip() == "violet",
            }
        )
    totals = {}
    for phase in ("base", "adapter"):
        selected = [r for r in records if r["phase"] == phase]
        tokens = sum(r["evaluation"]["supervised_tokens"] for r in selected)
        totals[phase] = {
            "cases": len(selected),
            "exact_text_matches": sum(r["generation"]["exact_text_match"] for r in selected),
            "exact_answers_with_eos": sum(
                r["generation"]["exact_answer_and_eos"] for r in selected
            ),
            "truncated_answers": sum(r["generation"]["truncated"] for r in selected),
            "supervised_tokens": tokens,
            "token_weighted_nll": sum(r["evaluation"]["nll_sum"] for r in selected) / tokens,
        }
    return {
        "scope": SCOPE,
        "pairs": pairs,
        "totals": totals,
        "regressions": sum(p["base_pass_adapter_fail"] for p in pairs),
        "improvements": sum(p["base_fail_adapter_pass"] for p in pairs),
        "confidence_claim": "none; descriptive counts on this fixed development probe only",
        "quality_certified": False,
    }


def write_json(path, value):
    path = Path(path)
    stage = path.with_name(path.name + ".tmp")
    with stage.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    stage.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def runtime_metadata(package_root):
    versions = {}
    for package in (
        "mlx",
        "mlx-lm",
        "numpy",
        "transformers",
        "tokenizers",
        "safetensors",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    code = {"package_root": str(package_root), "script_sha256": sha256(__file__)}
    for name, command in (
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("git_dirty_diff_sha256", ["git", "diff", "--binary", "HEAD"]),
        ("git_status", ["git", "status", "--porcelain"]),
    ):
        result = subprocess.run(
            command, cwd=package_root, capture_output=True, timeout=10, check=False
        )
        code[name] = (
            (
                hashlib.sha256(result.stdout).hexdigest()
                if name.endswith("sha256")
                else result.stdout.decode().strip()
            )
            if result.returncode == 0
            else None
        )
    lock = package_root.parents[1] / "uv.lock"
    code["dependency_lock_sha256"] = sha256(lock) if lock.exists() else None
    source_files = sorted(package_root.glob("*.py")) + sorted(package_root.glob("*.metal"))
    code["source_files_sha256"] = {p.name: sha256(p) for p in source_files}
    chip = (
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if sys.platform == "darwin"
        else None
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip.stdout.strip() if chip and chip.returncode == 0 else None,
        "dependencies": versions,
        "code": code,
    }


def run(args):
    dataset_path = Path(args.cases).expanduser().resolve()
    dataset_hash = sha256(dataset_path)
    cases = read_cases(dataset_path)
    if sha256(dataset_path) != dataset_hash:
        raise ValueError("Fixture changed while being read")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema_valid": True,
                    "cases": len(cases),
                    "scope": SCOPE,
                    "note": "CPU-only schema validation; tokenization/context not checked",
                },
                indent=2,
            )
        )
        return
    if not args.model or not args.adapter or not args.output:
        raise ValueError("Execution requires --model, --adapter and --output")
    root, adapter, output = (
        Path(p).expanduser().resolve() for p in (args.model, args.adapter, args.output)
    )
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    config_path = adapter / "adapter_config.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("format") != "exl3-mlx-deepseek-short-text-v1":
        raise ValueError("Unsupported adapter format")
    adapter_hashes = {
        name: sha256(adapter / name) for name in ("adapter_config.json", "adapter.safetensors")
    }
    if adapter_hashes["adapter_config.json"] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError("Adapter configuration changed while being read")
    state_path = adapter / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if any(
            state.get("files", {}).get(name) != digest for name, digest in adapter_hashes.items()
        ):
            raise ValueError("Saved training checkpoint adapter checksum mismatch")

    # Native imports deliberately occur only for execution, not CPU parser/scorer tests.
    import mlx.core as mx
    from mlx.utils import tree_flatten

    import exl3_mlx
    from exl3_mlx.data import (
        load_tokenizer,
        tokenize_row,
        tokenizer_identity,
        validate_example,
    )
    from exl3_mlx.deepseek import DeepSeekTextModel
    from exl3_mlx.memory import MemoryBudget
    from exl3_mlx.train import checkpoint_identity, load_adapter
    from exl3_mlx.training import evaluate
    from exl3_mlx.workflow import generate, model_settings

    base_hashes = checkpoint_identity(root)
    if config.get("base_identity") != base_hashes:
        raise ValueError("Adapter base identity mismatch")
    tokenizer_hashes = tokenizer_identity(root)
    tokenizer = load_tokenizer(root)
    vocab_size = json.loads((root / "config.json").read_text())["text_config"]["vocab_size"]
    prepared = []
    for case in cases:
        example = validate_example(
            tokenize_row({"prompt": case["prompt"], "completion": case["expected"]}, tokenizer),
            seq_length=512,
            vocab_size=vocab_size,
            truncate=False,
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            add_generation_prompt=True,
            return_dict=False,
        )
        prepared.append(
            validate_tokenized_case(case, example, prompt_ids, tokenizer.eos_token_id, vocab_size)
        )
    settings = model_settings(SimpleNamespace(adapter=str(adapter)))
    manifest = {
        "format": "exl3-mlx-paired-probe-v1",
        "scope": SCOPE,
        "created_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "cases": len(cases),
        },
        "adapter": {
            "path": str(adapter),
            "files_sha256": adapter_hashes,
            "settings": settings,
        },
        "base": {
            "path": str(root),
            "metadata_sha256": base_hashes,
            "hash_scope": "metadata only; does not hash all base weight bytes",
        },
        "tokenizer_sha256": tokenizer_hashes,
        "runtime": runtime_metadata(Path(exl3_mlx.__file__).resolve().parent),
        "settings": SETTINGS,
        "objective": "masked final assistant answer including EOS; no training",
        "decoding": "existing workflow.generate: greedy, chat template, full-prefix recomputation",
        "selection": "fixed development/regression fixtures; no checkpoint selection within this run",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "tokenized-cases.json", prepared)
    start, records = time.monotonic(), []
    progress = {
        "scope": SCOPE,
        "status": "running",
        "phase": "loading",
        "completed_results": 0,
    }

    def event(kind, **values):
        entry = {"event": kind, "elapsed_seconds": time.monotonic() - start, **values}
        append_jsonl(output / "events.jsonl", entry)
        progress.update(
            elapsed_seconds=entry["elapsed_seconds"],
            **{k: v for k, v in values.items() if k in ("phase", "case_id")},
        )
        write_json(output / "progress.json", progress)
        print(
            f"[{entry['elapsed_seconds']:.1f}s] {kind} {values.get('phase', '')} {values.get('case_id', '')}",
            flush=True,
        )

    try:
        with MemoryBudget(16, cache_mib=64, reserve_gib=32) as budget:
            budget.abort_path = output / "memory-abort.json"
            mx.random.seed(8)
            event("load_started", phase="loading")
            model = DeepSeekTextModel(
                root, **settings, cache_mib=128, chunk_size=512, checkpoint_blocks=True
            )
            model.eval()
            b_weights = [
                (k, v)
                for k, v in tree_flatten(model.trainable_parameters())
                if k.endswith("lora_b")
            ]
            if not b_weights or any(not mx.all(v == 0).item() for _, v in b_weights):
                raise ValueError("Baseline requires newly initialized zero-B adapters")
            del b_weights
            event("load_complete", phase="base", model=model.report())
            for phase in ("base", "adapter"):
                if phase == "adapter":
                    if any(
                        sha256(adapter / name) != digest for name, digest in adapter_hashes.items()
                    ):
                        raise ValueError("Adapter files changed before loading")
                    load_adapter(model, adapter)
                    event("adapter_loaded", phase=phase)
                for case in prepared:
                    event("case_started", phase=phase, case_id=case["case_id"])
                    case_start = time.monotonic()
                    evaluation = evaluate(model, [case["example"]], check=budget.check)
                    if evaluation["supervised_tokens"] != len(case["target_ids"]):
                        raise ValueError("Evaluation mask token count differs from fixture")
                    evaluation["nll_sum"] = evaluation["loss"] * evaluation["supervised_tokens"]
                    event(
                        "evaluation_complete",
                        phase=phase,
                        case_id=case["case_id"],
                        evaluation=evaluation,
                    )
                    generation = generate(
                        SimpleNamespace(
                            model=str(root),
                            prompt=case["prompt"],
                            max_tokens=case["max_new_tokens"],
                            raw_prompt=False,
                        ),
                        model,
                        budget,
                    )
                    generation.update(validate_generation_evidence(case, generation))
                    record = {
                        **{k: v for k, v in case.items() if k != "example"},
                        "phase": phase,
                        "evaluation": evaluation,
                        "generation": generation,
                        "elapsed_seconds": time.monotonic() - case_start,
                    }
                    records.append(record)
                    append_jsonl(output / "results.jsonl", record)
                    progress["completed_results"] = len(records)
                    event(
                        "case_complete",
                        phase=phase,
                        case_id=case["case_id"],
                        exact_answer_and_eos=generation["exact_answer_and_eos"],
                        text=generation["text"],
                    )
            if (
                sha256(dataset_path) != manifest["dataset"]["sha256"]
                or checkpoint_identity(root) != base_hashes
                or tokenizer_identity(root) != tokenizer_hashes
                or any(sha256(adapter / name) != digest for name, digest in adapter_hashes.items())
            ):
                raise ValueError("Input files changed during the comparison")
            summary = paired_summary(prepared, records)
            summary.update(
                status="completed",
                elapsed_seconds=time.monotonic() - start,
                peak_mlx_gib=mx.get_peak_memory() / 2**30,
                observed_memory_gib=budget.usage() / 2**30,
                model=model.report(),
            )
            write_json(output / "paired.json", summary)
            progress.update(status="completed", phase="completed")
            event("comparison_complete", phase="completed")
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        event("failed", error=progress["error"])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--output", help="A new output directory")
    parser.add_argument(
        "--validate-only", action="store_true", help="CPU-only fixture schema check"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
