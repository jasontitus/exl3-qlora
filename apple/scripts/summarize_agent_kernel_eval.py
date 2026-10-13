"""CPU-only, offset-aligned teacher-forced scoring of native kernel-edit targets.

No code is executed. File/line categories describe the supplied patch, not
correctness or an independently generated solution. Every target token is kept,
including removed/context lines, comments, EOS and native tool serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from exl3_mlx.agent_edits import parse_command
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity
from exl3_mlx.sft_data import _normalize

ATOL = 1e-6
_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)\n\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
HELPER_SOURCES = [
    Path(fn.__code__.co_filename).resolve() for fn in (parse_command, tokenize_row, _normalize)
]
HELPER_SOURCES.append(Path(tokenize_row.__code__.co_filename).resolve().with_name("tool_data.py"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def read_rows(raw):
    rows = [read_json(line) for line in raw.splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL must contain nonempty object rows")
    return rows


def file_category(path):
    """Declared path heuristic, deliberately separate from patch line role."""
    p = PurePosixPath(path)
    if (
        p.is_absolute()
        or str(p) != path
        or any(part in {"..", ".git"} for part in p.parts)
        or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", path)
    ):
        raise ValueError("Unsafe/unsupported patch path")
    parts = [part.lower() for part in p.parts]
    name, extension = parts[-1], p.suffix.lower()
    if (
        any(re.fullmatch(r"tests?|benchmarks?|benches", part) for part in parts[:-1])
        or name.startswith(("test_", "test-", "bench_", "bench-"))
        or name.endswith(("_test.cpp", ".test.cpp"))
    ):
        return "tests_benchmarks"
    if (
        extension in {".md", ".rst", ".txt", ".adoc"}
        or "docs" in parts[:-1]
        or name.startswith(("readme", "changelog", "license"))
    ):
        return "documentation"
    if extension == ".metal" or name.endswith(".metal.h"):
        return "metal"
    if extension in {".cu", ".cuh", ".cl", ".glsl", ".comp", ".hlsl"}:
        return "other_device_source"
    if (
        extension
        in {
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".h",
            ".hh",
            ".hpp",
            ".hxx",
            ".m",
            ".mm",
            ".swift",
            ".py",
            ".rs",
            ".sh",
            ".cmake",
        }
        or name == "cmakelists.txt"
    ):
        return "host_support_source"
    return "other_file"


def patch_spans(patch):
    """Validate standard unified hunks and classify exact character intervals."""
    spans, position, path, pending_old = [], 0, None, None
    old_left = new_left = 0
    has_hunk = False
    seen = set()
    for line in patch.splitlines(keepends=True):
        if not line.endswith("\n"):
            raise ValueError("Patch lines must be newline terminated")
        role, content = "header", "metadata"
        if old_left or new_left:
            marker = line[0]
            if marker not in " +-":
                raise ValueError("Incomplete or malformed unified hunk")
            old_left -= marker != "+"
            new_left -= marker != "-"
            if old_left < 0 or new_left < 0:
                raise ValueError("Unified hunk line counts disagree")
            role = {"+": "added", "-": "removed", " ": "context"}[marker]
            text = line[1:].strip()
            content = (
                "blank"
                if not text
                else "comment_line"
                if text.startswith(("//", "/*", "*", "*/"))
                else "code_or_other"
            )
        elif line.startswith("--- "):
            if pending_old is not None or (path is not None and not has_hunk):
                raise ValueError("File section has no hunks")
            old = line[4:-1]
            if old != "/dev/null" and not old.startswith("a/"):
                raise ValueError("Unsupported old-file header")
            pending_old = old
            path = None
            has_hunk = False
        elif line.startswith("+++ "):
            if pending_old is None or not line[4:-1].startswith("b/"):
                raise ValueError("Unsupported new-file header")
            path = line[6:-1]
            file_category(path)
            if path in seen or pending_old not in {"/dev/null", "a/" + path}:
                raise ValueError("Duplicate or renamed patch file")
            seen.add(path)
            pending_old = None
        elif (match := _HUNK.fullmatch(line)) and path is not None and pending_old is None:
            old_left = int(match[2] or 1)
            new_left = int(match[4] or 1)
            if old_left + new_left == 0:
                raise ValueError("Empty unified hunk")
            has_hunk = True
        else:
            raise ValueError("Unsupported patch syntax")
        # Old-file headers have no current path yet; all headers are syntax.
        family = file_category(path) if path is not None and role != "header" else "patch_metadata"
        spans.append((position, position + len(line), (family, role, content)))
        position += len(line)
    if not spans or pending_old is not None or old_left or new_left or not has_hunk:
        raise ValueError("Incomplete patch")
    return spans


def categorize_row(raw, prepared, tokenizer):
    normalized, metadata, kind = _normalize(raw, None, None)
    if kind != "terminal_tool_call":
        raise ValueError("Only terminal native tool-call rows can be categorized")
    exact = tokenize_row(normalized, tokenizer)
    if prepared != exact:
        raise ValueError("Prepared IDs/mask differ from complete native raw row")
    fn = normalized["messages"][-1]["tool_calls"][0]["function"]
    if fn["name"] != "exec_command" or set(fn["arguments"]) != {"cmd"}:
        raise ValueError("Expected declared exec_command(cmd) edit target")
    command = fn["arguments"]["cmd"]
    patch = parse_command(command)
    rendered = tokenizer.apply_chat_template(
        normalized["messages"], tools=normalized["tools"], tokenize=False
    )
    prefix = tokenizer.apply_chat_template(
        normalized["messages"][:-1],
        tools=normalized["tools"],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoding = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoding["input_ids"] != exact["input_ids"] or not rendered.startswith(prefix):
        raise ValueError("Actual tokenizer offsets do not reproduce native serialization")
    start = len(prefix)
    suffix = rendered[start:]
    if suffix.count(command) != 1:
        raise ValueError("Template must preserve exactly one literal command in target")
    command_start = start + suffix.index(command)
    patch_start = command_start + command.index("\n") + 1
    labels = [("tool_boilerplate", "serialization", "metadata")] * len(rendered)
    for lo, hi, label in patch_spans(patch):
        labels[patch_start + lo : patch_start + hi] = [label] * (hi - lo)
    rationale = normalized["messages"][-1].get("content") or ""
    if rationale:
        before_command = rendered[start:command_start]
        if before_command.count(rationale) != 1:
            raise ValueError("Rationale cannot be unambiguously located in serialized target")
        lo = start + before_command.index(rationale)
        labels[lo : lo + len(rationale)] = [("rationale", "analysis", "text")] * len(rationale)
    offsets = encoding["offset_mapping"]
    if len(offsets) != len(exact["input_ids"]):
        raise ValueError("Incomplete tokenizer offsets")
    categories, component_counts = [], defaultdict(int)
    for token, (lo, hi) in zip(
        exact["input_ids"][exact["loss_start"] :], offsets[exact["loss_start"] :], strict=True
    ):
        if not 0 <= lo <= hi <= len(rendered) or (hi > lo and lo < start):
            raise ValueError("Supervised token offset crosses masked prefix")
        if token == tokenizer.eos_token_id:
            category = ("eos", "terminator", "metadata")
        elif lo == hi:
            # Empty offsets cannot be credited to source code.
            category = ("unmapped_special", "serialization", "metadata")
        else:
            unique = set(labels[lo:hi])
            if len(unique) == 1:
                category = unique.pop()
            else:
                category = ("mixed_boundary", "mixed", "mixed")
                for label in unique:
                    component_counts["/".join(label)] += 1
        categories.append("/".join(category))
    return {
        "case_id": metadata.get("case_id"),
        "source_id": metadata.get("source_id"),
        "prepared_sha256": digest(canonical(exact)),
        "rendered_sha256": digest(rendered.encode()),
        "command_sha256": digest(command.encode()),
        "patch_sha256": digest(patch.encode()),
        "target_ids": exact["input_ids"][exact["loss_start"] :],
        "tokenizer_vocabulary_size": len(tokenizer),
        "categories": categories,
        "mixed_boundary_components": dict(component_counts),
    }


def aligned_categories(raw_rows, prepared, tokenizer):
    lookup = {}
    for row in raw_rows:
        normalized, _, kind = _normalize(row, None, None)
        if kind != "terminal_tool_call":
            raise ValueError("Raw data contains a non-tool row")
        key = canonical(tokenize_row(normalized, tokenizer))
        if key in lookup:
            raise ValueError("Duplicate token-identical raw rows make lineage ambiguous")
        lookup[key] = row
    seen, result = set(), []
    for row in prepared:
        key = canonical(row)
        if key in seen or key not in lookup:
            raise ValueError("Prepared rows must be unique exact complete raw examples")
        seen.add(key)
        result.append(categorize_row(lookup[key], row, tokenizer))
    return result


def validate_score(score, targets, vocabulary_size):
    n = len(targets)
    if (
        type(score.get("supervised_tokens")) is not int
        or score["supervised_tokens"] != n
        or n < 1
        or score.get("target_ids") != targets
        or any(type(x) is not int or x < 0 for x in score["target_ids"])
    ):
        raise ValueError("Score target IDs/count disagree with exact prepared targets")
    predictions, values = score.get("predicted_ids"), score.get("token_nll")
    if (
        not isinstance(predictions, list)
        or len(predictions) != n
        or any(type(x) is not int or not 0 <= x < vocabulary_size for x in predictions)
    ):
        raise ValueError("Invalid/incomplete prediction IDs")
    if (
        not isinstance(values, list)
        or len(values) != n
        or any(type(x) not in (int, float) or not math.isfinite(x) or x < -ATOL for x in values)
    ):
        raise ValueError("Invalid/incomplete per-token NLL")
    loss = score.get("loss")
    if (
        type(loss) not in (int, float)
        or not math.isfinite(loss)
        or abs(loss - math.fsum(values) / n) > ATOL
    ):
        raise ValueError("Scalar score disagrees with token NLL")
    if type(score.get("correct_tokens")) is not int or score["correct_tokens"] != sum(
        a == b for a, b in zip(targets, predictions, strict=True)
    ):
        raise ValueError("Incorrect correct-token count")
    if not isinstance(score.get("logits_sha256"), str) or not _HASH.fullmatch(
        score["logits_sha256"]
    ):
        raise ValueError("Missing logits digest")


def _argument(argv, flag, default=None):
    if argv.count(flag) > 1:
        raise ValueError("Duplicate evaluation argument")
    if flag not in argv:
        return default
    at = argv.index(flag) + 1
    if at >= len(argv):
        raise ValueError("Missing evaluation argument value")
    return argv[at]


def validate_pair(base, adapter, prepared, categories, *, prepared_path, prepared_sha256, model):
    expected_hash = digest(canonical(prepared))
    total = sum(len(row["input_ids"]) - row["loss_start"] for row in prepared)
    contexts = []
    for report in (base, adapter):
        context = report.get("evaluation_context")
        if (
            not isinstance(context, dict)
            or context.get("selected_examples_sha256") != expected_hash
            or context.get("input_identity") != {"data_sha256": prepared_sha256}
        ):
            raise ValueError("Evaluation receipt does not bind exact prepared data selection")
        if Path(context.get("data", "")).resolve() != Path(prepared_path).resolve():
            raise ValueError("Evaluation receipt data path differs")
        if (
            type(context.get("process_id")) is not int
            or context["process_id"] <= 0
            or context["process_id"] == os.getpid()
        ):
            raise ValueError("Invalid evaluation process ID")
        if type(context.get("seq_length")) is not int or context["seq_length"] < max(
            len(r["input_ids"]) - 1 for r in prepared
        ):
            raise ValueError("Evaluation sequence length would truncate examples")
        requested = context.get("requested_max_examples")
        if type(requested) is not int or requested not in (0, len(prepared)):
            raise ValueError("Evaluation must cover complete prepared split")
        argv = context.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or argv[0] != "evaluate"
            or any(not isinstance(x, str) for x in argv)
            or argv.count("--token-metrics") != 1
        ):
            raise ValueError("Expected explicit evaluate --token-metrics invocation")
        if (
            Path(_argument(argv, "--data", "")).resolve() != Path(prepared_path).resolve()
            or Path(_argument(argv, "--model", "")).resolve() != Path(model).resolve()
        ):
            raise ValueError("Evaluation argv data/model differs")
        if int(_argument(argv, "--max-examples", "0")) != requested:
            raise ValueError("Evaluation maximum differs from argv")
        chunk = _argument(argv, "--chunk-size")
        if (
            chunk is None
            or not chunk.isdecimal()
            or not 128 <= int(chunk) <= 32768
            or int(chunk) % 128
        ):
            raise ValueError("Explicit resolved --chunk-size is required on both evaluations")
        declared_mode = _argument(argv, "--activation-mode")
        if declared_mode is not None and declared_mode != report.get("activation_mode"):
            raise ValueError("Activation mode differs from invocation")
        declared_adapter = _argument(argv, "--adapter")
        if (declared_adapter is None) != (report.get("adapter") is None) or (
            declared_adapter is not None
            and Path(declared_adapter).resolve() != Path(report["adapter"]).resolve()
        ):
            raise ValueError("Evaluation adapter differs from argv")
        if report.get("checkpoint") != str(Path(model).resolve()):
            raise ValueError("Report checkpoint root differs from tokenizer/model")
        implementation = context.get("implementation")
        if (
            not isinstance(implementation, dict)
            or not implementation.get("source")
            or not implementation.get("runtime")
            or not implementation.get("python")
        ):
            raise ValueError("Missing evaluation implementation identity")
        rows = report.get("rows")
        if (
            type(report.get("examples")) is not int
            or report["examples"] != len(prepared)
            or not isinstance(rows, list)
            or len(rows) != len(prepared)
        ):
            raise ValueError("Incomplete evaluated row set")
        for score, label in zip(rows, categories, strict=True):
            validate_score(score, label["target_ids"], label["tokenizer_vocabulary_size"])
        if type(report.get("supervised_tokens")) is not int or report["supervised_tokens"] != total:
            raise ValueError("Aggregate target count mismatch")
        if not isinstance(report.get("logits_sha256"), str) or not _HASH.fullmatch(
            report["logits_sha256"]
        ):
            raise ValueError("Missing aggregate logits digest")
        loss = report.get("loss")
        mean = math.fsum(x for row in rows for x in row["token_nll"]) / total
        if type(loss) not in (int, float) or not math.isfinite(loss) or abs(loss - mean) > ATOL:
            raise ValueError("Aggregate loss disagrees with token means")
        contexts.append(context)
    if (
        base.get("adapter") is not None
        or not isinstance(adapter.get("adapter"), str)
        or not adapter["adapter"]
    ):
        raise ValueError("Pair must contain zero-adapter base and saved-adapter evaluation")
    if contexts[0]["process_id"] == contexts[1]["process_id"]:
        raise ValueError("Paired receipts must name distinct fresh evaluation processes")
    if contexts[0]["implementation"] != contexts[1]["implementation"]:
        raise ValueError("Paired evaluation implementations differ")
    if base.get("activation_mode") not in {"checkpoint", "bf16"} or base[
        "activation_mode"
    ] != adapter.get("activation_mode"):
        raise ValueError("Paired activation modes differ")
    if _argument(contexts[0]["argv"], "--chunk-size", "512") != _argument(
        contexts[1]["argv"], "--chunk-size", "512"
    ):
        raise ValueError("Paired EXL3 arithmetic chunk sizes differ")


def _metrics(values):
    n = len(values)
    base_sum = math.fsum(v[0] for v in values)
    adapter_sum = math.fsum(v[1] for v in values)
    return {
        "tokens": n,
        "base_nll_sum": base_sum,
        "adapter_nll_sum": adapter_sum,
        "base_nll": base_sum / n,
        "adapter_nll": adapter_sum / n,
        "nll_delta": (adapter_sum - base_sum) / n,
        "base_correct": sum(v[2] for v in values),
        "adapter_correct": sum(v[3] for v in values),
        "corrected_tokens": sum(not v[2] and v[3] for v in values),
        "regressed_tokens": sum(v[2] and not v[3] for v in values),
    }


def summarize(base, adapter, categories):
    aggregate, rows = defaultdict(list), []
    for index, (before, after, labels) in enumerate(
        zip(base["rows"], adapter["rows"], categories, strict=True)
    ):
        grouped, changes = defaultdict(list), []
        for at, (target, category, b, a, bp, ap) in enumerate(
            zip(
                labels["target_ids"],
                labels["categories"],
                before["token_nll"],
                after["token_nll"],
                before["predicted_ids"],
                after["predicted_ids"],
                strict=True,
            )
        ):
            value = (b, a, bp == target, ap == target)
            for group in ("all", category, "family:" + category.split("/")[0]):
                grouped[group].append(value)
                aggregate[group].append(value)
            changes.append(
                {
                    "target_index": at,
                    "target_id": target,
                    "category": category,
                    "base_nll": b,
                    "adapter_nll": a,
                    "nll_delta": a - b,
                    "base_prediction_id": bp,
                    "adapter_prediction_id": ap,
                }
            )
        rows.append(
            {
                "row_index": index,
                **{k: v for k, v in labels.items() if k not in {"target_ids", "categories"}},
                "metrics": {k: _metrics(v) for k, v in sorted(grouped.items())},
                "tokens": changes,
            }
        )
    return {
        "aggregate": {k: _metrics(v) for k, v in sorted(aggregate.items())},
        "macro_row_nll": {
            name: math.fsum(r["loss"] for r in value["rows"]) / len(rows)
            for name, value in (("base", base), ("adapter", adapter))
        },
        "rows": rows,
    }


def run(args):
    paths = [args.raw, args.prepared, Path(__file__), *HELPER_SOURCES]
    if args.categories_only:
        if args.baseline or args.adapter:
            raise ValueError("Category-only mode cannot accept evaluation reports")
    elif args.baseline and args.adapter:
        paths.extend([args.baseline, args.adapter])
    else:
        raise ValueError("Provide both baseline/adapter reports or --categories-only")
    snapshots = {Path(p).resolve(): Path(p).read_bytes() for p in paths}
    raw = read_rows(snapshots[args.raw.resolve()])
    prepared = read_rows(snapshots[args.prepared.resolve()])
    tokenizer_hashes = tokenizer_identity(args.model)
    tokenizer = load_tokenizer(args.model)
    categories = aligned_categories(raw, prepared, tokenizer)
    result = {
        "format": "agent-kernel-teacher-forced-summary-v1",
        "quality_certified": False,
        "execution_verified": False,
        "scope": "Complete teacher-forced native-tool target tokens; supplied preceding answer tokens, not autoregressive patch generation",
        "category_policy": {
            "file_family": "Path-extension heuristic; tests/benchmark and documentation paths take precedence. Metal means .metal/.metal.h source, including comments and literals.",
            "line_role": "added/removed/context/header from validated unified hunk counts",
            "content_kind": "blank or conservative line-prefix comment heuristic; code_or_other is not proof of executable code",
            "mixed_tokens": "Tokens crossing category boundaries remain mixed_boundary; never arbitrarily assigned to Metal",
            "tool": "Native serialization/heredoc syntax and EOS reported separately",
            "provenance_scope": "Checks raw/prepared tokenizer alignment and paired receipt consistency; does not independently authenticate checkpoint payloads, source correctness, or benchmark performance",
        },
        "scalar_mean_atol": ATOL,
        "tokenizer": tokenizer_hashes,
    }
    if args.categories_only:
        result.update(evaluation_completed=False, categories=categories)
    else:
        base = read_json(snapshots[args.baseline.resolve()])
        adapted = read_json(snapshots[args.adapter.resolve()])
        validate_pair(
            base,
            adapted,
            prepared,
            categories,
            prepared_path=args.prepared,
            prepared_sha256=digest(snapshots[args.prepared.resolve()]),
            model=args.model,
        )
        result.update(
            evaluation_completed=True,
            evaluation_contexts={
                "base": base["evaluation_context"],
                "adapter": adapted["evaluation_context"],
            },
            **summarize(base, adapted, categories),
        )
    if tokenizer_identity(args.model) != tokenizer_hashes or any(
        p.read_bytes() != value for p, value in snapshots.items()
    ):
        raise ValueError("Summary inputs/tokenizer changed during evaluation")
    result["inputs"] = {str(p): digest(value) for p, value in snapshots.items()}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, required=True, help="Native messages/tools JSONL; metadata permitted"
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="Exact input_ids/loss_start JSONL scored by evaluate",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Pinned local tokenizer directory; no weights loaded",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--adapter", type=Path, help="Fresh saved-adapter evaluate JSON, not the adapter directory"
    )
    parser.add_argument("--categories-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, required=True, help="New report path, never overwrite"
    )
    report = run(parser.parse_args())
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k in {"format", "evaluation_completed", "aggregate", "macro_row_nll"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
