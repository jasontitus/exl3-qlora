"""Prepare a reviewed exact-edit/diagnostic SFT mix without touching runtime or GPU.

Input releases remain immutable. This exporter does not approve its own output:
the final manifest and concrete training plan still require independent review.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
RUNTIME = WORK / "loss-fix/apple/artifacts/loss-fix/runtime-v5"


def implementation_bindings():
    snapshot_bytes = (RUNTIME / "snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_bytes)
    expected_sources = {str(p.relative_to(RUNTIME)) for p in (RUNTIME / "src/exl3_mlx").iterdir() if p.suffix in {'.py', '.metal', '.html'}} | {'pyproject.toml', 'uv.lock'}
    if not expected_sources <= snapshot.keys():
        raise ValueError('Incomplete pinned runtime snapshot')
    bound = {str(RUNTIME / 'snapshot.json'): hashlib.sha256(snapshot_bytes).hexdigest()}
    for name, expected in snapshot.items():
        path = (RUNTIME / name).resolve(strict=True)
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_relative_to(RUNTIME):
            raise ValueError('Unsafe runtime snapshot path')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned runtime changed before import')
        bound[str(path)] = actual
    for path in [Path(__file__).resolve(), *[WORK / 'kernel-library/expansion-v3-hour/tools' / n for n in ('tokenize_reasoning.py', 'reasoning_data.py')]]:
        bound[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return bound


IMPLEMENTATION_HASHES = implementation_bindings()
sys.path.insert(0, str(RUNTIME / "src"))
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity, validate_example

sys.path.insert(0, str(WORK / "kernel-library/expansion-v3-hour/tools"))
from tokenize_reasoning import verify_export

TEST_SHA = "0720ead3d696e6e9215e99cbf18dd26e532def8c309add06bbbe21efe1841ad0"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bind(path, hashes):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    digest = sha(data)
    if str(path) in hashes and hashes[str(path)] != digest:
        raise ValueError("Input changed during preparation")
    hashes[str(path)] = digest
    return data


def key(record):
    return record["repo"].lower().rstrip("/"), record["commit"]


def select_diagnostics(records, exact_keys, limit):
    """Prefer new commits, then round-robin repositories, one view per commit."""
    chosen, seen, seen_lineages = [], set(), set()
    for overlaps_exact in (False, True):
        groups = defaultdict(list)
        for record in sorted(records, key=lambda r: r["id"]):
            if (key(record) in exact_keys) == overlaps_exact:
                groups[key(record)[0]].append(record)
        while groups and len(chosen) < limit:
            for repo in sorted(list(groups)):
                while groups[repo] and (key(groups[repo][0]) in seen or str(groups[repo][0].get('lineage', key(groups[repo][0]))).lower() in seen_lineages):
                    groups[repo].pop(0)
                if not groups[repo]:
                    del groups[repo]
                    continue
                item = groups[repo].pop(0)
                chosen.append(item)
                seen.add(key(item))
                seen_lineages.add(str(item.get('lineage', key(item))).lower())
                if len(chosen) == limit:
                    break
    return chosen


def build(exact, reasoning, model, output, aliases=None):
    exact, output = Path(exact).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    hashes = dict(IMPLEMENTATION_HASHES)
    audit = json.loads(bind(exact / "audit.json", hashes))
    capacity = audit["capacity"]
    if capacity != 12288 or audit["truncated"] != 0 or not audit["heldout_prepared_exact"]:
        raise ValueError("This experiment requires the reviewed complete 12K exact release")
    for filename, expected in audit["inputs_sha256"].items():
        if sha(bind(filename, hashes)) != expected:
            raise ValueError("Exact release source drift: " + filename)
    required = {'train.raw.jsonl', 'train.jsonl', 'test.raw.jsonl', 'test.jsonl', 'valid.jsonl', 'provenance.jsonl', 'cross-split-code-overlap.json'}
    if not required <= audit['artifacts_sha256'].keys():
        raise ValueError('Exact release omits required artifact bindings')
    captured = {}
    for name, expected in audit["artifacts_sha256"].items():
        if Path(name).name != name:
            raise ValueError('Unsafe exact artifact name')
        captured[name] = bind(exact / name, hashes)
        if sha(captured[name]) != expected:
            raise ValueError("Exact release artifact drift")
    if sha(captured['test.jsonl']) != TEST_SHA:
        raise ValueError("Frozen heldout token IDs or masks changed")
    if captured['valid.jsonl']:
        raise ValueError("Validation split must remain empty")
    def captured_rows(name):
        return [json.loads(line) for line in captured[name].decode().splitlines() if line.strip()]
    metadata = captured_rows('provenance.jsonl')
    train_meta = [r for r in metadata if r["split"] == "train"]
    test_meta = [r for r in metadata if r["split"] == "test"]
    train_raw, train_prepared = captured_rows('train.raw.jsonl'), captured_rows('train.jsonl')
    test_raw, test_prepared = captured_rows('test.raw.jsonl'), captured_rows('test.jsonl')
    if len(train_meta) != len(train_raw) or len(train_meta) != audit["train_examples"] or len(test_meta) != 16:
        raise ValueError("Exact provenance count mismatch")
    identity = tokenizer_identity(model)
    if identity != audit["tokenizer"]:
        raise ValueError("Tokenizer differs from exact release")
    config = json.loads(bind(Path(model) / 'config.json', hashes))
    tokenizer = load_tokenizer(model)
    vocab = config.get("text_config", config)["vocab_size"]
    for raw, prepared, meta in zip(train_raw + test_raw, train_prepared + test_prepared, train_meta + test_meta, strict=True):
        if tokenize_row(raw, tokenizer) != prepared:
            raise ValueError("Pinned runtime tokenization differs from prepared release")
        validate_example(prepared, seq_length=capacity - 1, vocab_size=vocab, truncate=False)
        if sha(canon(raw)) != meta["raw_sha256"] or sha(canon(prepared)) != meta["prepared_sha256"]:
            raise ValueError("Exact row/provenance identity mismatch")
    diagnostic_records, diagnostic_rows = {}, {}
    heldout_lineages = {str(value).lower() for m in test_meta for value in (m.get('change_lineage'), m.get('split_family'), m['id']) if value}
    for directory in reasoning:
        directory = Path(directory).resolve()
        raw, _, bound = verify_export(directory / "train.messages.jsonl")
        for filename, digest in bound.items():
            if filename in hashes and hashes[filename] != digest:
                raise ValueError("Conflicting diagnostic input binding")
            hashes[filename] = digest
        for row, record in zip(raw, rows(directory / "provenance.jsonl"), strict=True):
            rid = record["id"]
            if rid in diagnostic_records and diagnostic_records[rid] != record:
                raise ValueError("Conflicting diagnostic record identity")
            if key(record) in {key(m) for m in test_meta}:
                raise ValueError("Protected heldout diagnostic in training")
            if str(record['lineage']).lower() in heldout_lineages:
                raise ValueError('Known protected lineage in diagnostic training')
            diagnostic_records[rid], diagnostic_rows[rid] = record, row
    if aliases is not None:
        alias_doc = json.loads(bind(aliases, hashes))
        if alias_doc.get('schema') != 'diagnostic-known-family-aliases-v1':
            raise ValueError('Invalid known-family alias schema')
        for filename, expected in alias_doc['inputs_sha256'].items():
            if sha(bind(filename, hashes)) != expected:
                raise ValueError('Known-family source drift')
        alias_map = {}
        for group in alias_doc['groups']:
            family = group['id']
            if not isinstance(family, str) or not family:
                raise ValueError('Missing family identity')
            for member in group['members']:
                member_key = key(member)
                if member_key in alias_map and alias_map[member_key] != family:
                    raise ValueError('Overlapping families must be unioned before selection')
                alias_map[member_key] = family
        protected_families = {alias_map[key(m)] for m in test_meta if key(m) in alias_map}
        for rid, record in diagnostic_records.items():
            family = alias_map.get(key(record))
            if family in protected_families:
                raise ValueError('Known alias connects diagnostic to heldout')
            if family is not None:
                diagnostic_records[rid] = {**record, 'original_lineage': record['lineage'], 'lineage': 'known-family:' + family}
    cap = len(train_raw) // 4
    selected = select_diagnostics(list(diagnostic_records.values()), {key(m) for m in train_meta}, cap)
    selected_meta = []
    for record in selected:
        raw = diagnostic_rows[record["id"]]
        prepared = tokenize_row(raw, tokenizer)
        validate_example(prepared, seq_length=capacity - 1, vocab_size=vocab, truncate=False)
        train_raw.append(raw)
        train_prepared.append(prepared)
        selected_meta.append({
            "id": record["id"], "repo": record["repo"], "commit": record["commit"],
            "lineage": record["lineage"], "kind": "diagnostic", "split": "train",
            "serialized_tokens": len(prepared["input_ids"]),
            "supervised_tokens": len(prepared["input_ids"]) - prepared["loss_start"],
            "raw_sha256": sha(canon(raw)), "prepared_sha256": sha(canon(prepared)),
            "record": record,
        })
    if len({sha(canon(r)) for r in train_raw}) != len(train_raw):
        raise ValueError("Duplicate training payload")
    all_meta = [{**m, "kind": "exact_edit"} for m in train_meta] + selected_meta
    for filename in [__file__, RUNTIME / "snapshot.json", Path(model) / "config.json"]:
        bind(filename, hashes)
    # Recheck every consumed dependency before publication, including the export validators.
    for module in (sys.modules["tokenize_reasoning"], sys.modules["reasoning_data"], sys.modules["exl3_mlx.data"]):
        bind(module.__file__, hashes)
    for filename, digest in hashes.items():
        if sha(Path(filename).read_bytes()) != digest:
            raise ValueError("Input drift before publication")
    if tokenizer_identity(model) != identity:
        raise ValueError("Tokenizer drift")
    output.mkdir(parents=True)
    def write_json(name, obj):
        (output / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    def write_rows(name, values):
        (output / name).write_bytes(b"".join(canon(x) + b"\n" for x in values))
    write_rows("train.jsonl", train_raw)
    write_rows("test.jsonl", test_raw)
    write_rows("valid.jsonl", [])
    write_rows("train.prepared.jsonl", train_prepared)
    # Preserve exact historical bytes, not just equivalent JSON serialization.
    (output / "test.prepared.jsonl").write_bytes(captured['test.jsonl'])
    write_rows("provenance.jsonl", all_meta + test_meta)
    write_json("selection.json", {
        "policy": "Prefer commits without an exact view, then repository round-robin and sorted IDs; at most one diagnostic per commit and declared lineage; floor(exact/4) cap. Known-family aliases apply when a reviewed alias manifest is supplied; unrecorded ancestry remains possible.",
        "selected_diagnostics": [r["id"] for r in selected],
        "unselected_diagnostics": sorted(set(diagnostic_records) - {r["id"] for r in selected}),
        "reason": "Declared update-weight heuristic; not an empirically optimal ratio.",
    })
    report = {
        "format": "exl3-agent-kernel-corpus-v1", "content_schema": "exact-edit-plus-evidence-diagnostics-v1",
        "counts": {"train": len(train_raw), "valid": 0, "test": len(test_raw)},
        "max_length": capacity, "truncated": 0, "tokenizer": identity,
        "exact_rows": len(train_meta), "diagnostic_rows": len(selected),
        "unique_training_commits": len({key(r) for r in all_meta}),
        "by_repository": dict(Counter(key(r)[0] for r in all_meta)),
        "supervised_tokens_by_kind": {kind: sum(r["supervised_tokens"] for r in all_meta if r["kind"] == kind) for kind in ("exact_edit", "diagnostic")},
        "inputs_sha256": hashes, "frozen_heldout_prepared_sha256": TEST_SHA,
        "qualification": "Source-reviewed single-edit imitation and evidence interpretation. Not recorded agent trajectories, locally qualified speedups, or proof of model benefit. Independent final review required before training.",
    }
    report["artifacts_sha256"] = {p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    write_json("ktune-import.json", report)
    return {k: v for k, v in report.items() if k not in {"inputs_sha256", "artifacts_sha256", "tokenizer"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", required=True)
    parser.add_argument("--reasoning", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aliases")
    args = parser.parse_args()
    print(json.dumps(build(args.exact, args.reasoning, args.model, args.output, args.aliases), indent=2))
