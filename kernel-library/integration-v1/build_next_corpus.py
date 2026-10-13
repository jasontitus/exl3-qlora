"""Build the next-round native-tool corpus from checksummed manifests.

Explicit source allowlist expands to requested sources. All llama.cpp stays
protected test. No candidate code or generated shell is executed.
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
from urllib.parse import urlsplit

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
        "https://github.com/antirez/ds4",
        "https://github.com/jundot/omlx",
        "https://github.com/ml-explore/mlx",
        "https://github.com/ml-explore/mlx-lm",
        "https://github.com/blaizzy/mlx-vlm",
        "https://github.com/incoai/splash",
        "https://github.com/youssofal/mtplx",
        "https://github.com/vllm-project/vllm-metal",
    },
    "test": {"https://github.com/ggml-org/llama.cpp"},
}
_ORIGINAL_TRAIN = {
    "https://github.com/ddalcu/mlx-serve",
    "https://github.com/beamivalice/sushi",
    "https://github.com/ashhart/tensorfold",
}
_CODE = re.compile(
    r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]'
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
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


class ProtectedLineageError(ValueError):
    pass


def _repository(url):
    if not isinstance(url, str):
        raise ValueError("Repository URL must be text")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", parsed.path)
    ):
        raise ValueError("Expected canonical public GitHub repository URL")
    path = parsed.path.rstrip("/").removesuffix(".git").lower()
    return "https://github.com" + path


def _lineage_key(key):
    if not isinstance(key, str):
        raise ValueError("Protected lineage key must be text")
    kind, sep, value = key.partition(":")
    if kind == "repository" and sep and _repository(value) == value:
        return key
    if kind in {"commit", "stable_patch"} and re.fullmatch(r"[a-f0-9]{40}", value):
        return key
    if (
        kind == "declared_group"
        and value
        and value == value.strip()
        and len(value) <= 256
        and all(32 <= ord(char) < 127 for char in value)
    ):
        return key
    raise ValueError("Invalid or noncanonical protected lineage key: " + key)


def _declared_keys(candidate, repositories):
    repo = _repository(repositories[candidate["repository"]]["url"])
    keys = {"repository:" + repo}
    if repo in {
        "https://github.com/ddalcu/mlx-serve",
        "https://github.com/beamivalice/sushi",
    }:
        keys.add("declared_group:policy-mlx-sushi-connected")
    if candidate.get("source_group"):
        if not isinstance(candidate["source_group"], str):
            raise ValueError("source_group must be text")
        keys.add(_lineage_key("declared_group:" + candidate["source_group"]))
    related = candidate.get("related_commits", [])
    if not isinstance(related, list):
        raise ValueError("related_commits must be a list")
    commits = [candidate.get("commit"), candidate.get("parent"), *related]
    keys.update(_lineage_key("commit:" + c) for c in commits if c is not None)
    return keys


def _record_keys(record):
    return set(record["lineage_keys"]) | {"commit:" + c for c in record["commits"]}


def _protection_keys(manifests):
    protected = {"repository:https://github.com/ggml-org/llama.cpp"}
    for _, _, _, manifest in manifests:
        keys = manifest.get("protected_keys", [])
        if not isinstance(keys, list):
            raise ValueError("protected_keys must be a list")
        protected.update(_lineage_key(key) for key in keys)
    return protected


def _historical_review(candidate, manifest_path, *, required):
    review = candidate.get("historical_protection_review")
    if review is None:
        return ("pending" if required else "preserved_existing_curation"), {}
    if not isinstance(review, dict) or review.get("status") not in {
        "cleared",
        "pending",
        "protected",
    }:
        raise ValueError("Invalid historical_protection_review status")
    for field in ("reviewed_by", "reason"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError("Historical protection review requires " + field)
    bindings = review.get("evidence_sha256", {})
    if not isinstance(bindings, dict) or (review["status"] == "cleared" and not bindings):
        raise ValueError("Cleared historical review requires evidence_sha256 file bindings")
    verified = {}
    root = manifest_path.parent.resolve()
    for name, digest in bindings.items():
        if not isinstance(name, str):
            raise ValueError("Historical evidence path must be text")
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or str(relative) != name
            or any(p in {".", ".."} for p in relative.parts)
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError("Invalid historical evidence path or hash")
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 32 << 20:
            raise ValueError("Historical evidence must be a bounded manifest-local file")
        if sha(path.read_bytes()) != digest:
            raise ValueError("Historical protection evidence checksum mismatch")
        verified[str(path)] = digest
    return review["status"], verified


def _prepare_sources(manifests, protected):
    """Resolve protection before eligibility/length filtering, retaining failures."""
    entries, evidence_hashes = {}, {}
    for split, path, _, manifest in manifests:
        evidence, sources = _evidence(path.parent, manifest.get("evidence", {}))
        evidence_hashes.update({str(path.parent / name): value for name, value in sources.items()})
        for candidate in manifest["candidates"]:
            keys = _declared_keys(candidate, manifest["repositories"])
            repo = _repository(manifest["repositories"][candidate["repository"]]["url"])
            review, bindings = _historical_review(
                candidate,
                path,
                required=split == "train" and repo not in _ORIGINAL_TRAIN,
            )
            evidence_hashes.update(bindings)
            entry = {
                "id": candidate["id"],
                "split": split,
                "keys": keys,
                "protected": split == "test" or review == "protected" or bool(keys & protected),
                "source": None,
                "source_error": None,
                "historical_review": review,
                "outcome": candidate.get("outcome"),
            }
            try:
                entry["source"] = reconstruct_candidate(
                    candidate, manifest["repositories"], evidence
                )
                entry["keys"].update(_record_keys(entry["source"]))
                entry["protected"] |= bool(entry["keys"] & protected)
            except Exception as error:
                if entry["protected"]:
                    raise ProtectedLineageError(
                        "Protected source could not reconstruct: " + candidate["id"]
                    ) from error
                entry["source_error"] = str(error)
            entries[candidate["id"]] = entry
    # Every declared/reconstructed key participates, including rejected and
    # failed rows. This preserves transitive port/alias/family protection.
    owners, parents = {}, {case_id: case_id for case_id in entries}

    def find(case_id):
        while parents[case_id] != case_id:
            parents[case_id] = parents[parents[case_id]]
            case_id = parents[case_id]
        return case_id

    for case_id, entry in entries.items():
        for key in entry["keys"]:
            if key in owners:
                parents[find(case_id)] = find(owners[key])
            else:
                owners[key] = case_id
    groups = defaultdict(list)
    for case_id in entries:
        groups[find(case_id)].append(entries[case_id])
    for members in groups.values():
        all_keys = set().union(*(entry["keys"] for entry in members))
        protected_group = any(entry["protected"] for entry in members)
        family_keys = sorted(
            key for key in all_keys if key.startswith(("repository:", "declared_group:"))
        )
        group_id = "upstream-lineage:" + sha(dump(family_keys))[:24]
        for entry in members:
            entry["source_id"] = group_id
            if protected_group and entry["split"] == "train":
                raise ProtectedLineageError("Protected connected training lineage: " + entry["id"])
    return entries, evidence_hashes


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
                and _repository(url) in _ALLOWED[split]
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
                before, after = (
                    _tokens(unit["before_content"]),
                    _tokens(unit["after_content"]),
                )
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
        fn.argtypes, fn.restype = (
            [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint],
            ctypes.c_int,
        )
        result = fn(os.fsencode(stage), os.fsencode(out), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        fn = library.renameat2
        fn.argtypes, fn.restype = (
            [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ],
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
    protected = _protection_keys(manifests)
    prepared, prepared_evidence_hashes = _prepare_sources(manifests, protected)
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
        records, ledger, manifest_hashes = [], [], {}
        evidence_hashes = dict(prepared_evidence_hashes)
        disputed_patches = {
            entry["source"]["stable_patch_id"]
            for entry in prepared.values()
            if entry["source"] is not None and entry["outcome"] in {"negative", "contradictory"}
        }
        for split, path, raw, m in manifests:
            manifest_hashes[str(path)] = sha(raw)
            for c in m["candidates"]:
                entry = prepared[c["id"]]
                try:
                    if entry["historical_review"] == "pending":
                        raise ValueError(
                            "Quarantined: historical ktune protection mapping is pending"
                        )
                    if c.get("outcome") not in {
                        "reported_improvement",
                        "verified_improvement",
                    }:
                        raise ValueError(
                            "Positive corpus requires a reviewed improvement classification"
                        )
                    if not isinstance(c.get("task_kind"), dict) or c["task_kind"].get(
                        "value"
                    ) not in {"kernel", "host_dispatch"}:
                        raise ValueError("Require explicit reviewed kernel or host-dispatch task")
                    url = _repository(m["repositories"][c["repository"]]["url"])
                    if url not in _ALLOWED[split]:
                        raise ValueError("Repository outside predeclared source-family split")
                    if entry["source_error"] is not None:
                        raise ValueError("Source reconstruction failed: " + entry["source_error"])
                    r = entry["source"]
                    if r["stable_patch_id"] in disputed_patches:
                        raise ValueError(
                            "Equivalent stable patch has negative or contradictory evidence"
                        )
                    if policy is None:
                        row, replay = agent_example(r)
                    else:
                        row, replay = agent_example(
                            r, documentation_omissions=policy.get(c["id"], {})
                        )
                    # The template serializes tool schemas in insertion order. Publish
                    # and tokenize exactly the same canonical JSON representation;
                    # sorting only at publication can change tokens and the ceiling.
                    row = _json(dump(row))
                    tokens = tokenize_row(row, tokenizer)
                    if len(tokens["input_ids"]) > max_length:
                        raise ValueError("Complete example exceeds context ceiling; no truncation")
                    records.append(
                        {
                            "id": c["id"],
                            "split": split,
                            "source_id": entry["source_id"],
                            "source": r,
                            "row": row,
                            "replay": replay,
                            "total_tokens": len(tokens["input_ids"]),
                            "target_tokens": len(tokens["input_ids"]) - tokens["loss_start"],
                        }
                    )
                except ProtectedLineageError:
                    raise
                except Exception as exc:
                    ledger.append(
                        {
                            "id": c["id"],
                            "split": split,
                            "reason": str(exc),
                            "source_id": entry["source_id"],
                            "historical_protection_review": entry["historical_review"],
                            "lineage_keys": sorted(entry["keys"]),
                            "source_error": entry["source_error"],
                        }
                    )
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
        seen, rows, audits, retained = (
            set(),
            {"train": [], "valid": [], "test": []},
            [],
            [],
        )
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
            rows[record["split"]].append(
                {
                    **record["row"],
                    "source_id": record["source_id"],
                    "case_id": record["id"],
                }
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
        source_policy = {
            "format": "exl3-next-corpus-source-policy-v1",
            "allowed_repositories": {split: sorted(urls) for split, urls in _ALLOWED.items()},
            "explicit_protected_keys": sorted(protected),
            "protection": "Connected declared/reconstructed repo, commit, stable-patch and source-group identities include failed/excluded candidates before selection; protected train connections reject the whole build.",
            "historical_review": "Eight newly allowed source repositories require an explicit curator review bound to local evidence files. Missing/pending cases are quarantined; this does not certify independent semantic families or erase prior holdouts.",
            "connected_family_scope": "Build-level source groups conservatively connect entire repositories and declared/shared-key families; mlx-serve and Sushi always connect.",
            "candidate_reviews": {
                case_id: {
                    "status": entry["historical_review"],
                    "source_id": entry["source_id"],
                }
                for case_id, entry in prepared.items()
            },
        }
        _write(stage / "source-policy.json", dump(source_policy) + b"\n")
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
            "split_policy": "All llama.cpp final heldout. Prior three training repositories plus eight explicitly listed sources; new-source rows require historical protection review. No validation selection.",
            "source_policy": "source-policy.json",
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
