"""Mix already prepared token rows without retokenization, deduplication or splitting."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from .context import MAX_CONTEXT_TOKENS
from .data import load_dataset, sha256, tokenizer_identity, validate_example
from .evaluation_audit import _json
from .reviewed_corpus import publish_new

FORMAT = "exl3-mlx-mixture-plan-v1"
MAX_INPUT_BYTES = 128 << 20
MAX_OUTPUT_BYTES = 512 << 20
MAX_ROWS = 100_000
MAX_SOURCES = 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), f"Not a regular file: {path}")
        require(before.st_size <= limit, f"Input exceeds byte limit: {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())

    def identity(s):
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(
        identity(before) == identity(after) == identity(path.lstat()),
        f"Input changed while reading: {path}",
    )
    require(len(raw) <= limit, f"Input exceeds byte limit: {path}")
    return raw, identity(before)


class Inputs:
    def __init__(self):
        self.files = {}
        self.absent = set()
        self.size = 0
        self.identities = {}

    def read(self, path, limit=MAX_INPUT_BYTES):
        path = Path(path).resolve()
        if path not in self.files:
            raw, identity = read_regular(path, limit)
            require(
                len(raw) <= limit and self.size + len(raw) <= MAX_INPUT_BYTES,
                "Mixture inputs exceed byte limit",
            )
            self.files[path] = raw
            self.identities[path] = identity
            self.size += len(raw)
        require(len(self.files[path]) <= limit, f"Input exceeds byte limit: {path}")
        return self.files[path]

    def check(self):
        for path, raw in self.files.items():
            current, identity = read_regular(path, len(raw))
            require(
                identity == self.identities[path] and current == raw,
                f"Input changed during mixing: {path}",
            )
        require(not any(p.exists() for p in self.absent), "Dataset manifest appeared during mixing")


def prepared(path, model, seq_length, vocab, inputs, config_sha):
    raw = inputs.read(path)
    manifest_path = path.parent / "dataset.json"
    manifest_sha = None
    if manifest_path.exists():
        manifest_raw = inputs.read(manifest_path, 1 << 20)
        manifest = _json(manifest_raw)
        require(isinstance(manifest, dict), "Dataset manifest must be an object")
        if "model_config_sha256" in manifest:
            require(manifest["model_config_sha256"] == config_sha, "Dataset model config mismatch")
        manifest_sha = digest(manifest_raw)
    else:
        inputs.absent.add(manifest_path)
    rows = []
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        require(len(line) <= 16 << 20, f"{path}:{number}: row exceeds 16 MiB")
        row = _json(line)
        require(
            isinstance(row, dict) and "input_ids" in row, "Mix accepts prepared token rows only"
        )
        example = validate_example(row, seq_length=seq_length, vocab_size=vocab, truncate=False)
        rows.append((number, line, example))
        require(len(rows) <= MAX_ROWS, "Source exceeds row limit")
    require(rows, f"Empty prepared dataset: {path}")
    # Retain the standard dataset tokenizer/file binding checks, after rejecting raw input.
    loaded = load_dataset(path, model, seq_length=seq_length, truncate=False)
    require(loaded == [r[2] for r in rows], "Source changed while loading prepared rows")
    return rows, {
        "path": str(path),
        "sha256": digest(raw),
        "rows": len(rows),
        "dataset_manifest_sha256": manifest_sha,
    }


def mix(plan_path, model, output, *, seq_length):
    """Publish a train-only dataset, preserving each selected JSON line's content."""
    require(
        type(seq_length) is int and 2 <= seq_length <= MAX_CONTEXT_TOKENS,
        "seq_length must be an integer in 2..32768",
    )
    plan_path = Path(plan_path).expanduser().resolve()
    model = Path(model).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new dataset directory: {output}")
    inputs = Inputs()
    plan_raw = inputs.read(plan_path, 1 << 20)
    plan = _json(plan_raw)
    require(
        isinstance(plan, dict) and set(plan) <= {"format", "sources", "protected", "exclude"},
        "Invalid mixture plan fields",
    )
    require(plan.get("format") == FORMAT, "Unsupported mixture plan format")
    sources = plan.get("sources")
    require(
        isinstance(sources, list) and 1 <= len(sources) <= MAX_SOURCES,
        "sources must be a nonempty bounded array",
    )
    config_raw = inputs.read(model / "config.json", 1 << 20)
    config = _json(config_raw)
    require(
        isinstance(config, dict) and isinstance(config.get("text_config"), dict),
        "Model requires text_config.vocab_size",
    )
    vocab = config["text_config"].get("vocab_size")
    require(type(vocab) is int and vocab > 1, "Invalid model vocabulary size")
    tokenizer = tokenizer_identity(model)
    for name, checksum in tokenizer.items():
        require(digest(inputs.read(model / name)) == checksum, "Tokenizer changed during mixing")

    def path(value):
        require(isinstance(value, str) and bool(value.strip()), "Expected a nonempty file path")
        p = Path(value).expanduser()
        return (p if p.is_absolute() else plan_path.parent / p).resolve()

    cache = {}

    def get(p):
        if p not in cache:
            cache[p] = prepared(p, model, seq_length, vocab, inputs, digest(config_raw))
        return cache[p]

    forbidden, exclusions = set(), []
    for key in ("protected", "exclude"):
        entries = plan.get(key, [])
        require(isinstance(entries, list) and len(entries) <= MAX_SOURCES, f"Invalid {key} paths")
        for value in entries:
            rows, info = get(path(value))
            forbidden.update(tuple(r[2]["input_ids"]) for r in rows)
            exclusions.append({"role": key, **info})

    selected, provenance = [], []
    total_bytes = total_targets = total_tokens = 0
    for source_index, source in enumerate(sources):
        require(
            isinstance(source, dict)
            and set(source) <= {"path", "rows", "repeat"}
            and {"path", "repeat"} <= set(source),
            "Invalid source fields",
        )
        repeat = source["repeat"]
        require(
            type(repeat) is int and 1 <= repeat <= MAX_ROWS, "repeat must be a positive integer"
        )
        rows, info = get(path(source["path"]))
        indices = source.get("rows", list(range(len(rows))))
        require(
            isinstance(indices, list)
            and indices
            and all(type(i) is int and 0 <= i < len(rows) for i in indices),
            "Invalid row indices",
        )
        require(len(set(indices)) == len(indices), "Repeated row index: use repeat explicitly")
        count = len(indices) * repeat
        require(len(selected) + count <= MAX_ROWS, "Mixture exceeds output row limit")
        nbytes = targets = tokens = 0
        for i in indices:
            _, line, row = rows[i]
            require(
                tuple(row["input_ids"]) not in forbidden,
                f"Protected/excluded token overlap: source {source_index}, row {i}",
            )
            nbytes += len(line) + (not line.endswith(b"\n"))
            targets += len(row["input_ids"]) - row["loss_start"]
            tokens += len(row["input_ids"])
        total_bytes += nbytes * repeat
        require(total_bytes <= MAX_OUTPUT_BYTES, "Mixture exceeds output byte limit")
        total_targets += targets * repeat
        total_tokens += tokens * repeat
        for repetition in range(repeat):
            selected.extend((source_index, i, repetition, rows[i]) for i in indices)
        provenance.append(
            {
                **info,
                "source_index": source_index,
                "selected_rows": indices,
                "repeat": repeat,
                "output_rows": count,
                "supervised_tokens": targets * repeat,
                "input_tokens": tokens * repeat,
            }
        )
    for info in provenance:
        info["row_share"] = info["output_rows"] / len(selected)
        info["target_share"] = info["supervised_tokens"] / total_targets
    inputs.check()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mixture-", dir=output.parent))
    try:
        with (
            (stage / "train.jsonl").open("xb") as train,
            (stage / "mixture.jsonl").open("x") as trace,
        ):
            for index, (source_index, row_index, repetition, (number, line, row)) in enumerate(
                selected
            ):
                train.write(line if line.endswith(b"\n") else line + b"\n")
                record = {
                    "output_row": index,
                    "source_index": source_index,
                    "source_row": row_index,
                    "source_line": number,
                    "repetition": repetition,
                    "input_tokens": len(row["input_ids"]),
                    "loss_start": row["loss_start"],
                    "supervised_tokens": len(row["input_ids"]) - row["loss_start"],
                    "source_line_sha256": digest(line),
                    "newline_added": not line.endswith(b"\n"),
                }
                trace.write(json.dumps(record, sort_keys=True) + "\n")
        mixture = {
            "format": FORMAT,
            "plan_path": str(plan_path),
            "plan_sha256": digest(plan_raw),
            "sources": provenance,
            "protected_and_excluded": exclusions,
            "rows": len(selected),
            "input_tokens": total_tokens,
            "supervised_tokens": total_targets,
            "order": "source order; repeat whole selected row sequence; no shuffle",
            "row_share_note": "Update share only when one example is used per update; trainer sampling is unchanged",
        }
        (stage / "mixture.json").write_text(json.dumps(mixture, indent=2) + "\n")
        report = {
            "format": "exl3-mlx-dataset-v1",
            "importer": "prepared-mixture-v1",
            "tokenizer": tokenizer,
            "model_config_sha256": digest(config_raw),
            "seq_length": seq_length,
            "counts": {"train": len(selected), "valid": 0, "test": 0},
            "supervised_tokens": {"train": total_targets, "valid": 0, "test": 0},
            "duplicates_removed": 0,
            "truncated_rows": 0,
            "mask": "source loss_start unchanged",
            "files": {"train.jsonl": sha256(stage / "train.jsonl")},
            "artifacts_sha256": {
                name: sha256(stage / name) for name in ("mixture.json", "mixture.jsonl")
            },
            "source_files_sha256": {str(p): digest(raw) for p, raw in inputs.files.items()},
        }
        (stage / "dataset.json").write_text(json.dumps(report, indent=2) + "\n")
        require(
            len(load_dataset(stage / "train.jsonl", model, seq_length=seq_length)) == len(selected),
            "Published loader row count mismatch",
        )
        inputs.check()
        require(tokenizer_identity(model) == tokenizer, "Tokenizer changed during mixing")
        publish_new(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def add_arguments(parser):
    parser.add_argument("--plan", required=True, help="Prepared mixture plan JSON")
    parser.add_argument("--model", required=True, help="Local model tokenizer/config directory")
    parser.add_argument("--output", required=True, help="New train-only dataset directory")
    parser.add_argument("--seq-length", type=int, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)
    try:
        report = mix(args.plan, args.model, args.output, seq_length=args.seq_length)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise SystemExit(f"{type(error).__name__}: {error}") from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
