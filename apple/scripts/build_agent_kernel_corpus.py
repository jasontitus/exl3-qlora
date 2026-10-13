"""Build a fresh, replay-qualified native-tool corpus from checksummed manifests.

All llama.cpp examples remain protected final test; TensorFold and connected
mlx-serve/Sushi examples train. No candidate code or generated shell is executed.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import load_tokenizer, tokenize_row, tokenizer_identity
from exl3_mlx.upstream_data import FORMAT, _evidence, reconstruct_candidate

_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}\Z")
_SHA256 = re.compile(r"[a-f0-9]{64}\Z")
_ALLOWED = {
    "train": {
        "https://github.com/ddalcu/mlx-serve",
        "https://github.com/beamivalice/sushi",
        "https://github.com/ashhart/tensorfold",
    },
    "test": {"https://github.com/ggml-org/llama.cpp"},
}
_CODE = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]'
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def with_manifest_checksum(manifest):
    """Return a sealed copy; does not mutate/write or silently repair an input file."""
    value = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return {**value, "manifest_sha256": sha(dump(value))}


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _manifests(train, test):
    result, paths_seen, ids_seen = [], set(), {}
    for split, names in (("train", train), ("test", test)):
        for name in names:
            path = Path(name).expanduser().resolve(strict=True)
            if path in paths_seen:
                raise ValueError("A manifest may be supplied only once")
            paths_seen.add(path)
            if path.stat().st_size > 32 << 20:
                raise ValueError("Manifest exceeds32 MiB bound")
            raw = path.read_bytes()
            m = _json(raw)
            if not isinstance(m, dict) or m.get("format") != FORMAT:
                raise ValueError("Unsupported manifest format")
            supplied = m.get("manifest_sha256")
            if (
                not isinstance(supplied, str)
                or supplied != with_manifest_checksum(m)["manifest_sha256"]
            ):
                raise ValueError(
                    "A valid canonical manifest_sha256 is required; seal the reviewed manifest explicitly"
                )
            if not isinstance(m.get("repositories"), dict) or not isinstance(
                m.get("candidates"), list
            ):
                raise ValueError("Manifest requires repositories and candidate list")
            if "split" in m and m["split"] != split:
                raise ValueError("Manifest declared split conflicts with requested split")
            for c in m["candidates"]:
                if (
                    not isinstance(c, dict)
                    or not isinstance(c.get("id"), str)
                    or not _CASE_ID.fullmatch(c["id"])
                ):
                    raise ValueError(
                        "Case IDs must be safe1..120-character alphanumeric/underscore/hyphen names"
                    )
                identity = c["id"].casefold()
                if identity in ids_seen:
                    raise ValueError(f"Duplicate case ID across manifests: {c['id']}")
                ids_seen[identity] = str(path)
                declared = c.get("split")
                permitted = {"train"} if split == "train" else {"test", "final_heldout"}
                if declared is not None and declared not in permitted:
                    raise ValueError(
                        f"Case {c['id']} declared split conflicts with requested split"
                    )
                if type(c.get("protected")) is not bool or c["protected"] != (split == "test"):
                    raise ValueError(
                        "Train must be explicitly unprotected; final test must be explicitly protected"
                    )
            result.append((split, path, raw, m))
    return result


def _documentation_policy(name, manifests):
    """Validate the whole opt-in policy before any candidate processing.

    Source-byte/hash agreement and complete changed-document coverage remain
    per-row checks in agent_example; malformed/foreign policy declarations are
    global errors rather than silently ignored exclusions.
    """
    if name is None:
        return None, None, None
    path = Path(name).expanduser().resolve(strict=True)
    if path.stat().st_size > 32 << 20:
        raise ValueError("Documentation omissions policy exceeds32 MiB bound")
    raw = path.read_bytes()
    policy = _json(raw)
    if not isinstance(policy, dict):
        raise ValueError("Documentation omissions must map case IDs to reviewed paths")
    allowed_ids = set()
    for split, _, _, manifest in manifests:
        for candidate in manifest["candidates"]:
            task = candidate.get("task_kind")
            repository = manifest["repositories"].get(candidate.get("repository"), {})
            url = repository.get("url") if isinstance(repository, dict) else None
            if (
                candidate.get("outcome") in {"reported_improvement", "verified_improvement"}
                and isinstance(task, dict)
                and task.get("value") in {"kernel", "host_dispatch"}
                and isinstance(url, str)
                and url.rstrip("/").lower() in _ALLOWED[split]
            ):
                allowed_ids.add(candidate["id"])
    for case_id, entries in policy.items():
        if case_id not in allowed_ids:
            raise ValueError(
                "Documentation omissions case ID is not a known positive eligible candidate: "
                + case_id
            )
        if not isinstance(entries, dict):
            raise ValueError("Documentation omissions case must map reviewed document paths")
        for document, declaration in entries.items():
            relative = PurePosixPath(document)
            if (
                not document
                or relative.is_absolute()
                or str(relative) != document
                or any(part in {".", ".."} or part.lower() == ".git" for part in relative.parts)
                or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", document)
                or relative.suffix.lower() not in {".md", ".rst"}
            ):
                raise ValueError("Documentation omissions require safe .md/.rst paths")
            if not isinstance(declaration, dict) or set(declaration) != {
                "before_sha256",
                "after_sha256",
                "reason",
            }:
                raise ValueError(
                    "Documentation omission requires exact before/after hashes and reason"
                )
            before, after, reason = (
                declaration["before_sha256"],
                declaration["after_sha256"],
                declaration["reason"],
            )
            if (
                (
                    before is not None
                    and (not isinstance(before, str) or not _SHA256.fullmatch(before))
                )
                or not isinstance(after, str)
                or not _SHA256.fullmatch(after)
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise ValueError("Documentation omission has invalid hash or empty review reason")
    return policy, path, raw


def _tokens(text):
    return [token for token in _CODE.findall(text) if not token.startswith(("//", "/*"))]


def cross_split_code_report(records):
    """Block exact repeated substantive edited units; report other code overlap.

    Common unchanged declarations/helpers are excluded from automatic blocking.
    This reports selected changed units, not exhaustive semantic clone detection.
    """
    edits, shingles = defaultdict(list), defaultdict(set)
    scope, token_counts = {}, {}
    for record in records:
        key = (record["split"], record["id"])
        scope[key], all_shingles = [], set()
        units = record["source"].get("reviewed_context", {}).get("units", {})
        for path, selected in units.items():
            for unit in selected:
                before, after = _tokens(unit["before_content"]), _tokens(unit["after_content"])
                if before == after:
                    continue
                scope[key].append(
                    {
                        "path": path,
                        "kind": unit["kind"],
                        "before_tokens": len(before),
                        "after_tokens": len(after),
                    }
                )
                # Exact same source edit, not merely a common helper or baseline.
                if (
                    unit["kind"] in {"complete_kernel", "complete_function"}
                    and min(len(before), len(after)) >= 128
                    and len(set(before + after)) >= 32
                ):
                    edits[(sha(dump(before)), sha(dump(after)))].append(
                        {"split": key[0], "id": key[1], "path": path}
                    )
                for sequence in (before, after):
                    for offset in range(0, max(0, len(sequence) - 63), 16):
                        window = sequence[offset : offset + 64]
                        if len(set(window)) >= 16:
                            all_shingles.add(sha(dump(window)))
        token_counts[key] = len(all_shingles)
        for shingle in all_shingles:
            shingles[shingle].add(key)
    exact = [
        values for values in edits.values() if {v["split"] for v in values} == {"train", "test"}
    ]
    pairs = defaultdict(int)
    for users in shingles.values():
        train = sorted(key for key in users if key[0] == "train")
        test = sorted(key for key in users if key[0] == "test")
        for left in train:
            for right in test:
                pairs[(left, right)] += 1
    near = []
    for (left, right), shared in sorted(pairs.items()):
        smaller = min(token_counts[left], token_counts[right])
        if shared >= 4:
            near.append(
                {
                    "train_id": left[1],
                    "test_id": right[1],
                    "shared_64_token_shingles": shared,
                    "fraction_of_smaller_shingle_set": shared / smaller,
                    "classification": "review_required_not_automatic_leakage_verdict",
                }
            )
    return {
        "format": "cross-split-code-overlap-v1",
        "exact_repeated_edited_units": exact,
        "near_duplicate_pairs": near,
        "automatic_blocker": bool(exact),
        "scope": "C/C++/Metal-style token normalization of selected changed before/after units;64-token windows stride16. Comments ignored; strings retained. Does not exhaustively inspect unselected full-commit targets or infer semantic equivalence.",
        "policy": "Identical substantive before-and-after complete edited units across splits block publication. Shared shingles alone are review evidence; ubiquitous unchanged helpers never automatically exclude a case.",
        "selected_changed_units": [
            {"split": key[0], "id": key[1], "units": values}
            for key, values in sorted(scope.items())
        ],
    }


def _publish(stage, out):
    """Atomic directory rename which must fail if any destination already exists."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        fn = library.renamex_np
        fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
        result = fn(os.fsencode(stage), os.fsencode(out), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        fn = library.renameat2
        fn.argtypes, fn.restype = (
            [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint],
            ctypes.c_int,
        )
        result = fn(
            -100, os.fsencode(stage), -100, os.fsencode(out), 1
        )  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError(
            "Atomic exclusive directory publication requires macOS or Linux rename support"
        )
    if result:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError("Output appeared before publication; nothing was replaced")
        raise OSError(error, os.strerror(error))


def _write(path, value):
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _helper_source_identity():
    # These are the renderer, serialization/validation and reconstruction
    # helpers used by this builder. Hash local source without importing MLX.
    root = Path(sys.modules["exl3_mlx.agent_edits"].__file__).resolve().parent
    return {
        str(root / name): sha((root / name).read_bytes())
        for name in (
            "agent_edits.py",
            "data.py",
            "tool_data.py",
            "upstream_data.py",
            "ktune_data.py",
            "context.py",
        )
    }


def build_corpus(train, test, model, output, *, max_length=32768, documentation_omissions=None):
    if type(max_length) is not int or not 3 <= max_length <= 32768:
        raise ValueError("max_length must be3..32768")
    builder_identity = sha(Path(__file__).read_bytes())
    helper_source_hashes = _helper_source_identity()
    manifests = _manifests(train, test)
    policy, policy_path, policy_raw = _documentation_policy(documentation_omissions, manifests)
    policy_hashes = {str(policy_path): sha(policy_raw)} if policy is not None else {}
    requested = Path(output).expanduser()
    if not requested.name or requested.name in {".", ".."}:
        raise ValueError("Output must name a fresh directory")
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("Output must be a fresh directory")
    requested.parent.mkdir(parents=True, exist_ok=True)
    out = requested.parent.resolve() / requested.name
    stage = Path(tempfile.mkdtemp(prefix="." + out.name + ".pending-", dir=out.parent))
    try:
        tokenizer = load_tokenizer(model)
        tokenizer_hashes = tokenizer_identity(model)
        records, ledger, manifest_hashes, evidence_hashes = [], [], {}, {}
        for split, path, raw, m in manifests:
            manifest_hashes[str(path)] = sha(raw)
            evidence, sources = _evidence(path.parent, m.get("evidence", {}))
            evidence_hashes.update(
                {str(path.parent / name): value for name, value in sources.items()}
            )
            for c in m["candidates"]:
                try:
                    if c.get("outcome") not in {"reported_improvement", "verified_improvement"}:
                        raise ValueError(
                            "Positive corpus requires a reviewed improvement classification"
                        )
                    if not isinstance(c.get("task_kind"), dict) or c["task_kind"].get(
                        "value"
                    ) not in {"kernel", "host_dispatch"}:
                        raise ValueError("Require explicit reviewed kernel or host-dispatch task")
                    url = m["repositories"][c["repository"]]["url"].rstrip("/").lower()
                    if url not in _ALLOWED[split]:
                        raise ValueError("Repository outside predeclared source-family split")
                    r = reconstruct_candidate(c, m["repositories"], evidence)
                    if policy is None:
                        row, replay = agent_example(r)
                    else:
                        row, replay = agent_example(
                            r, documentation_omissions=policy.get(c["id"], {})
                        )
                    tokens = tokenize_row(row, tokenizer)
                    if len(tokens["input_ids"]) > max_length:
                        raise ValueError("Complete example exceeds context ceiling; no truncation")
                    records.append(
                        {
                            "id": c["id"],
                            "split": split,
                            "source": r,
                            "row": row,
                            "replay": replay,
                            "total_tokens": len(tokens["input_ids"]),
                            "target_tokens": len(tokens["input_ids"]) - tokens["loss_start"],
                        }
                    )
                except Exception as exc:
                    ledger.append({"id": c["id"], "split": split, "reason": str(exc)})
        train_keys, test_keys = set(), set()
        for record in records:
            keys = set(record["source"]["lineage_keys"]) | {
                "commit:" + c for c in record["source"]["commits"]
            }
            (train_keys if record["split"] == "train" else test_keys).update(keys)
        if train_keys & test_keys:
            raise ValueError(
                "Cross-split provenance connection: " + str(sorted(train_keys & test_keys))
            )
        overlap = cross_split_code_report(records)
        if overlap["automatic_blocker"]:
            raise ValueError(
                "Cross-split exact repeated substantive source edit: "
                + json.dumps(overlap["exact_repeated_edited_units"])
            )
        seen, rows, audits, retained = set(), {"train": [], "valid": [], "test": []}, [], []
        for record in records:
            identity = record["source"]["stable_patch_id"]
            if identity in seen:
                ledger.append(
                    {
                        "id": record["id"],
                        "split": record["split"],
                        "reason": "duplicate stable patch",
                    }
                )
                continue
            seen.add(identity)
            group = (
                "mlx-sushi"
                if any(s in record["source"]["repository"] for s in ["mlx-serve", "sushi"])
                else record["source"]["repository"]
            )
            rows[record["split"]].append(
                {**record["row"], "source_id": group, "case_id": record["id"]}
            )
            _write(
                stage / (record["id"] + ".provenance.json"),
                dump({k: v for k, v in record.items() if k != "row"}) + b"\n",
            )
            audits.append(
                {k: record[k] for k in ["id", "split", "total_tokens", "target_tokens", "replay"]}
            )
            retained.append(record)
        if not rows["train"] or not rows["test"]:
            raise ValueError("Need nonempty train and protected test")
        for split, values in rows.items():
            _write(stage / (split + ".jsonl"), b"".join(dump(v) + b"\n" for v in values))
        for name, value in [
            ("exclusions.json", ledger),
            ("audit.json", audits),
            ("cross-split-code-overlap.json", overlap),
        ]:
            _write(stage / name, dump(value) + b"\n")
        if policy is not None:
            _write(stage / "documentation-omissions.json", policy_raw)
        # Inputs cannot drift unnoticed while expensive source/token work runs.
        for name, digest in {
            **manifest_hashes,
            **evidence_hashes,
            **policy_hashes,
            **helper_source_hashes,
        }.items():
            if sha(Path(name).read_bytes()) != digest:
                raise ValueError("Source manifest/evidence changed during build: " + name)
        if sha(Path(__file__).read_bytes()) != builder_identity:
            raise ValueError("Corpus builder changed during build")
        if tokenizer_identity(model) != tokenizer_hashes:
            raise ValueError("Tokenizer changed during build")
        report = {
            "format": "exl3-agent-kernel-corpus-v1",
            "source_manifests": manifest_hashes,
            "source_evidence": evidence_hashes,
            "tokenizer": tokenizer_hashes,
            "counts": {k: len(v) for k, v in rows.items()},
            "max_length": max(r["total_tokens"] for r in retained),
            "context_ceiling": max_length,
            "truncated": 0,
            "split_policy": "All llama.cpp final heldout; TensorFold and connected mlx-serve/Sushi training. No validation selection.",
            "supervision": "One synthetic exec_command action carrying a replay-tested Git unified diff, serialized through actual model tool template. No invented observations.",
            "qualification": "Exact source and patch replay; compilation/numerical/performance receipts are separate and not implied.",
            "code_overlap_review": {
                "report": "cross-split-code-overlap.json",
                "near_duplicate_pairs_requiring_review": len(overlap["near_duplicate_pairs"]),
                "exact_duplicate_blockers": 0,
            },
            "builder_sha256": builder_identity,
            "helper_source_sha256": helper_source_hashes,
            "artifacts_sha256": {
                f.name: sha(f.read_bytes()) for f in stage.iterdir() if f.is_file()
            },
        }
        if policy is not None:
            report["documentation_omissions"] = {
                "source": str(policy_path),
                "sha256": sha(policy_raw),
                "copy": "documentation-omissions.json",
                "declared_cases": len(policy),
                "declared_paths": sum(len(paths) for paths in policy.values()),
                "scope": "Every changed .md/.rst path requires an explicit hash-bound review; only those documentation files may be omitted. Complete source, host, tests and build edits remain required.",
            }
        _write(stage / "ktune-import.json", dump(report) + b"\n")
        _publish(stage, out)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", action="append", required=True)
    parser.add_argument("--test-manifest", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument(
        "--documentation-omissions",
        help="Reviewed case/path/hash JSON: require explicit decisions for all changed .md/.rst files; retain all other commit edits",
    )
    args = parser.parse_args()
    report = build_corpus(
        args.train_manifest,
        args.test_manifest,
        args.model,
        args.output,
        max_length=args.max_length,
        documentation_omissions=args.documentation_omissions,
    )
    print(json.dumps({**report, "artifacts_sha256": "see report"}, indent=2))


if __name__ == "__main__":
    main()
