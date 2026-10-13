#!/usr/bin/env python3
"""CPU-only audit and diagnostic token categories for one unified-diff SFT row.

The primary training/evaluation mask is unchanged. Categories support secondary
NLL reporting; they do not measure semantic correctness or independent quality.
No MLX import, model execution, or file modification occurs except --output.
"""

import argparse
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_row(path, index):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if index < 0 or index >= len(rows):
        raise ValueError("Row index is outside the data")
    return rows[index]


def line_category(line):
    if line.startswith(("diff --git ", "index ", "--- ", "+++ ", "@@ ")):
        return "diff_header"
    if line.startswith("+"):
        return "added_comment" if line[1:].lstrip().startswith(("//", "/*", "*")) else "added_code"
    if line.startswith("-"):
        return "removed_code"
    if line.startswith(" ") or not line.strip():
        return "unchanged_context"
    return "other_diff_metadata"


def categories_for_offsets(text, offsets):
    chars = [
        category
        for line in text.splitlines(keepends=True)
        for category in [line_category(line)] * len(line)
    ]
    categories, cross_category_tokens = [], []
    for index, (start, end) in enumerate(offsets):
        if not 0 <= start < end <= len(text):
            raise ValueError("Completion token has unsupported empty/out-of-range offsets")
        counts = Counter(chars[start:end])
        categories.append(counts.most_common(1)[0][0])
        if len(counts) > 1:
            cross_category_tokens.append(index)
    return categories, cross_category_tokens


def audit(raw_path, prepared_path, tokenizer_path, row_index):
    from tokenizers import Tokenizer

    raw = read_row(raw_path, row_index)
    prepared = read_row(prepared_path, row_index)
    if set(raw) != {"prompt", "completion"} or not all(isinstance(v, str) for v in raw.values()):
        raise ValueError("Expected an exact prompt/completion row")
    if not raw["completion"].startswith("diff --git "):
        raise ValueError("This diagnostic labels unified-diff completions only")
    ids, start = prepared["input_ids"], prepared["loss_start"]
    if type(start) is not int or not 1 <= start < len(ids):
        raise ValueError("Invalid prepared loss mask")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = tokenizer.encode(raw["completion"], add_special_tokens=False)
    targets = ids[start:]
    eos = tokenizer.token_to_id("<｜end▁of▁sentence｜>")
    if eos is None or targets != encoding.ids + [eos]:
        raise ValueError("Prepared supervision must exactly equal raw completion plus model EOS")
    categories, cross_category = categories_for_offsets(raw["completion"], encoding.offsets)
    categories.append("eos")
    counts = Counter(categories)
    blocks = difflib.SequenceMatcher(
        None, ids[:start], targets, autojunk=False
    ).get_matching_blocks()
    return {
        "schema": "metal-fit-data-audit-v1",
        "source_sha256": {
            "raw": sha256(raw_path),
            "prepared": sha256(prepared_path),
            "tokenizer": sha256(tokenizer_path),
        },
        "row_index": row_index,
        "serialized_tokens": len(ids),
        "loss_start": start,
        "supervised_tokens": len(targets),
        "completion_plus_eos_exact": True,
        "token_category_counts": dict(counts),
        "token_category_fractions": {k: v / len(targets) for k, v in counts.items()},
        "category_rule": "Majority of token characters by diff line class; ties use earliest character class",
        "cross_category_token_indices": cross_category,
        "supervised_token_categories": categories,
        "supervised_token_ids": targets,
        "diagnostic_score_rule": "For each class report sum of per-token NLL, count, and mean; preserve full-mask mean as primary",
        "ordered_prompt_answer_matching_tokens": sum(block.size for block in blocks),
        "ordered_prompt_answer_matching_tokens_in_runs_at_least_four": sum(
            block.size for block in blocks if block.size >= 4
        ),
        "limitations": [
            "A low aggregate NLL is compatible with copied context/header tokens; this audit does not prove why NLL is low",
            "Added-code labels also contain copied code and syntax; they are not semantic novelty labels",
            "Added-comment classification is a lexical heuristic, not a Swift/Metal parser",
            "Ordered prompt matching is one deterministic sequence alignment, not an exhaustive copy estimate",
            "These diagnostic subsets must not replace or be selected after observing losses to redefine training success",
            "Single training-row fitting does not establish generalization, runnable edits, or kernel speedup",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new file")
    report = audit(args.raw, args.prepared, args.tokenizer, args.row)
    report["script_sha256"] = sha256(__file__)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "supervised_tokens",
                    "completion_plus_eos_exact",
                    "token_category_counts",
                    "ordered_prompt_answer_matching_tokens",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
