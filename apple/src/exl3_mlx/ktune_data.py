"""Auditable ktune archive -> complete, narrowly scoped patch-reconstruction SFT.

This importer executes no candidate code. Evidence classification is independent
of training selection. Historical outcomes never establish current correctness
or speed. All related revisions stay in one training lineage; there is no
synthetic held-out score from random rows of the same optimization campaign.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import selectors
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from .context import MAX_CONTEXT_TOKENS

DEFAULT_TIERS = ("official_promoted", "historical_promoted")
OPTIONAL_TIERS = ("source_only", "correctness_only", "historical_improved_superseded")
_MAX_FILE = 256 << 20
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")
_EDIT = re.compile(
    r"```edit ([^\n]+)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE\n```",
    re.DOTALL,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def _read(path, *, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        raw = stream.read(_MAX_FILE + 1)
    if len(raw) > _MAX_FILE:
        raise ValueError(f"Source exceeds {_MAX_FILE} uncompressed bytes: {path}")
    return raw


def _safe_file(root, relative):
    path = root / relative
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Source path escapes dataset or is not a file: {relative}")
    return resolved


def _helpers():
    from .data import load_tokenizer, tokenize_row, tokenizer_identity, validate_example

    return load_tokenizer, tokenize_row, tokenizer_identity, validate_example


def _official_audit(row):
    from .ktune_eval import validate_official_submission

    return validate_official_submission(row)


def preimage_context(edit, edit_format=None):
    """Return every old-side hunk, never added code, notes or outcome metadata.

    A hunk is intentionally not represented as a complete file. Unknown, binary,
    malformed, rename-only and mode-only patches are rejected rather than guessed.
    Full targets remain byte-identical to their archived UTF-8 edit.
    """
    if not isinstance(edit, str) or not edit.strip():
        raise ValueError("Missing complete text edit")
    if edit_format == "search_replace" or edit.startswith("```edit "):
        matches = list(_EDIT.finditer(edit))
        remainder = _EDIT.sub("", edit).strip()
        if not matches or remainder:
            raise ValueError("Malformed or unsupported search/replace edit")
        blocks = []
        for match in matches:
            name, before, _ = match.groups()
            _safe_patch_name(name)
            if not before:
                raise ValueError("Search/replace insertion lacks pre-change context")
            blocks.append(f"File: {name}\nSearch block (complete):\n{before}")
        return "\n\n".join(blocks), [match.group(1) for match in matches]
    if "GIT binary patch" in edit or "Binary files " in edit:
        raise ValueError("Binary patches cannot be SFT text targets")
    lines = edit.splitlines()
    blocks, paths = [], []
    old_name = None
    section_started, section_hunks = False, 0
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("diff --git "):
            if section_started and section_hunks == 0:
                raise ValueError("A patch file section has no supported textual hunks")
            section_started, section_hunks, old_name = True, 0, None
            index += 1
            continue
        if line.startswith("--- "):
            if old_name is not None:
                if section_hunks == 0:
                    raise ValueError("A patch file section has no supported textual hunks")
                section_hunks = 0
            section_started = True
            old_name = line[4:]
            if old_name != "/dev/null":
                old_name = old_name.removeprefix("a/")
                _safe_patch_name(old_name)
            if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
                raise ValueError("Missing new-side unified diff header")
            new_name = lines[index + 1][4:]
            if new_name != "/dev/null":
                new_name = new_name.removeprefix("b/")
                _safe_patch_name(new_name)
            name = new_name if old_name == "/dev/null" else old_name
            paths.append(name)
            index += 2
            continue
        match = _HUNK.match(line)
        if match:
            if old_name is None:
                raise ValueError("Hunk precedes file headers")
            old_start, old_count, _, new_count = match.groups()
            old_count, new_count = int(old_count or 1), int(new_count or 1)
            before, seen_old, seen_new = [], 0, 0
            index += 1
            while index < len(lines) and (seen_old < old_count or seen_new < new_count):
                item = lines[index]
                if item.startswith("\\ No newline at end of file"):
                    index += 1
                    continue
                if not item or item[0] not in " +-":
                    raise ValueError("Malformed unified diff hunk")
                if item[0] in " -":
                    before.append(item[1:])
                    seen_old += 1
                if item[0] in " +":
                    seen_new += 1
                index += 1
                if seen_old > old_count or seen_new > new_count:
                    raise ValueError("Unified diff hunk counts disagree")
            if (seen_old, seen_new) != (old_count, new_count):
                raise ValueError("Incomplete unified diff hunk")
            label = (
                "New file; empty preimage"
                if old_name == "/dev/null"
                else f"Old lines {old_start}..{int(old_start) + old_count - 1}"
            )
            blocks.append(f"File: {paths[-1]}\n{label}:\n" + "\n".join(before))
            section_hunks += 1
            continue
        if line.startswith(("+", "-", " ")) and not line.startswith(("--- ", "+++ ")):
            raise ValueError("Unparsed code outside unified diff hunk")
        if line.startswith("@@"):
            raise ValueError("Malformed unified diff hunk header")
        if line.startswith(("index ", "new file mode ", "deleted file mode ")):
            if not section_started or old_name is not None:
                raise ValueError("File metadata is outside its diff header")
        elif line == "\\ No newline at end of file":
            if not section_hunks:
                raise ValueError("Newline marker without a hunk")
        elif line:
            raise ValueError("Unsupported metadata or text outside unified diff hunks")
        index += 1
    if not blocks or section_hunks == 0:
        raise ValueError("No textual diff hunks")
    return "\n\n".join(blocks), sorted(set(paths))


def _safe_patch_name(name):
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {"..", ".env", ".git"} for part in path.parts)
        or any(c in name for c in "\t\r\n\x00")
    ):
        raise ValueError("Unsafe or unsupported patch filename")


def _prompt(context, edit_format):
    return (
        "Propose an Apple Silicon inference optimization that preserves numerical behavior. "
        f"Return only a complete {edit_format} edit. "
        "The following is pre-change patch context, not complete source files; "
        "the surrounding build, dispatch and test harness are not provided. "
        "This is a historical patch-reconstruction exercise, not evidence that the result is executable or faster.\n\n"
        + context
    )


def _snapshot_records(root, files):
    path = root / "library/optimizations.json"
    if not path.exists():
        return [], []
    path = _safe_file(root, "library/optimizations.json")
    files[str(path.relative_to(root))] = _file_sha(path)
    snapshot = _json(_read(path))
    payload = {k: v for k, v in snapshot.items() if k != "snapshot_sha256"}
    if (
        snapshot.get("schema_version") != 1
        or snapshot.get("purpose") != "retrieval_only"
        or _sha(_canonical(payload)) != snapshot.get("snapshot_sha256")
    ):
        raise ValueError("Modified or unsupported ktune retrieval snapshot")
    records, ledger = [], []
    for example in snapshot["examples"]:
        payload = {k: v for k, v in example.items() if k != "example_sha256"}
        if _sha(_canonical(payload)) != example.get("example_sha256") or _sha(
            example["diff"].encode()
        ) != example.get("diff_sha256"):
            raise ValueError(f"Modified snapshot example: {example.get('id')}")
        provenance = example.get("provenance", {})
        admission = example.get("admission", {})
        repository = (
            provenance.get("repository") or provenance.get("repo") or provenance.get("track")
        )
        group = provenance.get("source_group") or provenance.get("track") or repository
        records.append(
            {
                "case_id": "snapshot:" + example["id"],
                "source": "checksummed_snapshot",
                "tier": example["outcome"],
                "diff": example["diff"],
                "edit_format": example.get("edit_format"),
                "lineage_keys": ["repository:" + str(repository), "lineage:" + str(group)],
                "commits": [
                    provenance.get(k)
                    for k in (
                        "base_commit",
                        "head_commit",
                        "historical_head_commit",
                        "dependency_commit",
                    )
                    if provenance.get(k)
                ],
                "parent_commit": provenance.get("base_commit"),
                "head_commit": provenance.get("head_commit")
                or provenance.get("historical_head_commit"),
                "created_at": provenance.get("created_at"),
                "protected": admission.get("split") not in (None, "train"),
                "evidence": {
                    "level": example.get("evidence_level"),
                    "receipt": example.get("evidence"),
                    "admission": admission,
                    "limits": example.get("limits"),
                    "provenance": provenance,
                },
                "source_sha256": files[str(path.relative_to(root))],
                "record_sha256": example["example_sha256"],
            }
        )
    for excluded in snapshot.get("excluded", []):
        ledger.append(
            {
                "case_id": "snapshot-excluded:" + excluded["id"],
                "tier": "protected_or_unadmitted",
                "disposition": "excluded",
                "reason": excluded.get("reason"),
                "original": excluded,
            }
        )
    return records, ledger


def _official_records(root, files):
    records, ledger = [], []
    for folder in sorted((root / "library/official").glob("*")):
        if not folder.is_dir():
            continue
        relative = folder.relative_to(root) / "all-submissions.json.gz"
        if not (root / relative).exists():
            continue
        path = _safe_file(root, relative)
        files[str(relative)] = _file_sha(path)
        rows = _json(_read(path, compressed=True))
        if not isinstance(rows, list):
            raise ValueError("Official archive must contain a list of submission records")
        seen, accepted = set(), []
        for row in sorted(rows, key=lambda r: (r["createdAt"], r["id"])):
            identity = f"official:{folder.name}:{row['id']}"
            if row["id"] in seen:
                raise ValueError(f"Duplicate official submission ID: {identity}")
            seen.add(row["id"])
            audit = _official_audit(row) if row.get("status") == "accepted" else None
            provenance = {
                "case_id": identity,
                "source": "official_archive",
                "created_at": row["createdAt"],
                "tier": "official_promoted"
                if audit and audit["valid"]
                else "official_unverified"
                if row.get("status") == "accepted"
                else "official_nonpromoted",
                "source_sha256": files[str(relative)],
                "record_sha256": _sha(_canonical(row)),
                "evidence": {
                    "audit": audit,
                    "status": row.get("status"),
                    "promotion_status": row.get("promotionStatus"),
                    "rejection_reason": row.get("rejectionReason"),
                    "official_metrics": row.get("officialMetrics"),
                    "official_score": row.get("officialScore"),
                },
                "lineage_keys": [f"lineage:{folder.name}"],
                "commits": [
                    row[k]
                    for k in ("submissionCommitSha", "promotedSourceRef", "evaluatedCommitSha")
                    if row.get(k)
                ],
                "protected": False,
                "head_commit": row.get("promotedSourceRef"),
                "parent_commit": None,
            }
            if row.get("status") != "accepted":
                ledger.append(
                    {
                        **provenance,
                        "disposition": "ledger_only",
                        "reason": "No officially promoted complete target; rejection is not automatically a correctness failure",
                    }
                )
                continue
            accepted.append(row)
            stem = f"{len(accepted):02d}-{row['id'][:8]}"
            diff_relative = folder.relative_to(root) / "promoted" / f"{stem}.diff"
            if not (root / diff_relative).exists():
                ledger.append(
                    {
                        **provenance,
                        "disposition": "excluded",
                        "reason": "Complete promoted diff missing",
                    }
                )
                continue
            diff_path = _safe_file(root, diff_relative)
            files[str(diff_relative)] = _file_sha(diff_path)
            provenance.update(
                diff=_read(diff_path).decode("utf-8"),
                edit_format="unified_diff",
                diff_sha256=files[str(diff_relative)],
            )
            records.append(provenance)
    return records, ledger


def _experiment_ledger(root, files):
    relative = "library/experiments/index.jsonl"
    if not (root / relative).exists():
        return []
    path = _safe_file(root, relative)
    files[relative] = _file_sha(path)
    ledger = []
    for line in _read(path).decode().splitlines():
        if not line.strip():
            continue
        row = _json(line)
        ledger.append(
            {
                "case_id": f"experiment:{row['run']}:{row['candidate']}",
                "tier": "local_index_only",
                "disposition": "ledger_only",
                "reason": "Index summary is not a complete edit plus authenticated correctness/performance receipt",
                "original": row,
            }
        )
    return ledger


def _lineages(records):
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, record in enumerate(records):
        keys = (
            record["lineage_keys"]
            + ["commit:" + c for c in record["commits"]]
            + ["diff:" + _sha(record["diff"].encode())]
        )
        for key in keys:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(find(i), []).append(record)
    for members in groups.values():
        identity = "ktune-lineage:" + _sha(_canonical(sorted(r["case_id"] for r in members)))[:24]
        protected = any(r["protected"] for r in members)
        for record in members:
            record["source_id"] = identity
            record["protected"] = protected
    return groups


def _date(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Cutoff and source timestamps must include a timezone")
    return parsed


def _git(repo, *args):
    if not args or args[0] not in {"rev-parse", "diff", "ls-tree", "show"}:
        raise ValueError("Only read-only Git source queries are supported")
    if isinstance(repo, str) and not repo.startswith("/") and ":" in repo:
        host, root = _remote_parts(repo)
        git = ["git", "--no-pager", "-C", root, *args]
        command = ["ssh", "-o", "BatchMode=yes", host, shlex.join(git)]
    else:
        command = ["git", "--no-pager", "-C", str(repo), *args]
    return _bounded_query(command, max_bytes=16 << 20)


def _bounded_query(command, *, max_bytes):
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    output, errors, deadline = bytearray(), bytearray(), time.monotonic() + 30
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pinned source query exceeded 30 seconds")
            for key, _ in selector.select(min(1, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = output if key.data == "stdout" else errors
                target.extend(chunk)
                if len(target) > (max_bytes if key.data == "stdout" else 65536):
                    raise ValueError("Pinned source query exceeds byte bound")
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if process.returncode:
            raise ValueError(
                "Pinned parent source unavailable: " + errors.decode(errors="replace")[:300]
            )
        return bytes(output)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def _parent_context(record, paths, repo):
    head = record.get("head_commit")
    parent = record.get("parent_commit")
    if not head or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("A full pinned head commit is required")
    if parent is None:
        parent = _git(repo, "rev-parse", f"{head}^").decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", parent):
        raise ValueError("A full pinned parent commit is required")
    if record.get("edit_format") != "search_replace":
        actual = _git(
            repo, "diff", "--no-ext-diff", "--no-textconv", "--no-color", parent, head, "--"
        )
        if actual != record["diff"].encode():
            raise ValueError("Archived complete diff differs from pinned parent/head source")
    elif set(_git(repo, "diff", "--name-only", parent, head, "--").decode().splitlines()) != set(
        paths
    ):
        raise ValueError("Search/replace target does not cover exactly the pinned changed files")
    source_pairs = (
        record.get("evidence", {})
        .get("provenance", {})
        .get("source_proof", {})
        .get("source_pairs", {})
    )
    blocks, bindings = [], {}
    for path in paths:
        _safe_patch_name(path)
        tree = _git(repo, "ls-tree", parent, "--", path).decode().strip()
        if not tree:
            content = ""
            if record.get("edit_format") == "search_replace":
                raise ValueError("Search target absent from pinned parent")
        else:
            if not tree.startswith(("100644 blob ", "100755 blob ")):
                raise ValueError("Source must be a regular UTF-8 file")
            content = _git(repo, "show", f"{parent}:{path}").decode("utf-8")
        digest = _sha(content.encode())
        before_hash = source_pairs.get(path, {}).get("before_sha256")
        if before_hash and before_hash != digest:
            raise ValueError("Pinned parent source differs from recorded before hash")
        if record.get("edit_format") == "search_replace":
            matches = [m for m in _EDIT.finditer(record["diff"]) if m.group(1) == path]
            if not matches:
                raise ValueError("Missing search block in pinned parent source")
            after = content
            for match in matches:
                if after.count(match.group(2)) != 1:
                    raise ValueError("Search block is not unique in pinned parent source")
                after = after.replace(match.group(2), match.group(3), 1)
            if _git(repo, "show", f"{head}:{path}") != after.encode():
                raise ValueError("Search/replace result differs from pinned head source")
        bindings[path] = digest
        blocks.append(f"File: {path}\nComplete parent file:\n{content}")
    return "\n\n".join(blocks), {
        "repository": str(repo),
        "parent_commit": parent,
        "head_commit": head,
        "parent_files_sha256": bindings,
    }


def import_ktune(
    source,
    model,
    output,
    *,
    max_length=None,
    tiers=DEFAULT_TIERS,
    train_until=None,
    max_examples=None,
    context_mode="hunks",
    parent_repo=None,
    selected_case_ids=None,
):
    """Write a train-only standard-SFT source directory plus evidence sidecars.

    ``max_length`` is the full serialized prompt+target+BOS/EOS count. Oversize
    examples are explicitly excluded, never truncated or split into partial
    edits. ``train_until`` is optional ISO-8601: unknown dates and every member
    of a lineage containing later evidence are quarantined. It creates no test
    set. Use independently authored kernel tasks for prospective evaluation.
    """
    if max_length is not None and (
        type(max_length) is not int or not 3 <= max_length <= MAX_CONTEXT_TOKENS
    ):
        raise ValueError(f"max_length must be 3..{MAX_CONTEXT_TOKENS} or None")
    if isinstance(tiers, str) or not tiers or set(tiers) - set(DEFAULT_TIERS + OPTIONAL_TIERS):
        raise ValueError(
            "Choose explicit supported positive evidence tiers; negative targets are never SFT positives"
        )
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise ValueError("max_examples must be a positive integer or None")
    if selected_case_ids is not None and (
        isinstance(selected_case_ids, str)
        or not selected_case_ids
        or any(not isinstance(case, str) or not case for case in selected_case_ids)
        or len(set(selected_case_ids)) != len(selected_case_ids)
    ):
        raise ValueError("selected_case_ids must be distinct nonempty candidate identities")
    if context_mode not in ("hunks", "parent_files") or (context_mode == "parent_files") != (
        parent_repo is not None
    ):
        raise ValueError("Use context_mode='parent_files' with parent_repo, or 'hunks' without it")
    if parent_repo is not None:
        if isinstance(parent_repo, str) and not parent_repo.startswith("/") and ":" in parent_repo:
            _remote_parts(parent_repo)
        else:
            parent_repo = Path(parent_repo).expanduser().resolve(strict=True)
    cutoff = _date(train_until) if train_until else None
    root = Path(source).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    files = {}
    records, ledger = _snapshot_records(root, files)
    official, official_ledger = _official_records(root, files)
    records += official
    ledger += official_ledger + _experiment_ledger(root, files)
    if not files:
        raise ValueError(
            "No supported ktune snapshot or official archive found under source/library"
        )
    seen = set()
    for record in records:
        if record["case_id"] in seen:
            raise ValueError(f"Duplicate candidate identity: {record['case_id']}")
        seen.add(record["case_id"])
    if selected_case_ids is not None and set(selected_case_ids) - seen:
        raise ValueError(f"Unknown complete candidate IDs: {sorted(set(selected_case_ids) - seen)}")
    snapshot_receipt = None
    if (root / "snapshot-manifest.json").exists():
        receipt_path = _safe_file(root, "snapshot-manifest.json")
        snapshot_receipt = _json(_read(receipt_path))
        if snapshot_receipt.get("format") != "exl3-mlx-ktune-snapshot-v1" or any(
            snapshot_receipt.get("files_sha256", {}).get(name) != digest
            for name, digest in files.items()
        ):
            raise ValueError("Transferred source differs from its snapshot manifest")
        files["snapshot-manifest.json"] = _file_sha(receipt_path)
    groups = _lineages(records)
    cutoff_groups = set()
    if cutoff:
        for members in groups.values():
            if any(not r.get("created_at") or _date(r["created_at"]) > cutoff for r in members):
                cutoff_groups.add(members[0]["source_id"])
    load_tokenizer, tokenize_row, tokenizer_identity, validate_example = _helpers()
    model = Path(model).expanduser().resolve(strict=True)
    token_identity = tokenizer_identity(model)
    tokenizer = load_tokenizer(model)
    config_sha = _file_sha(model / "config.json")
    config = _json(_read(model / "config.json"))
    vocab_size = config.get("text_config", config)["vocab_size"]
    limit = max_length or MAX_CONTEXT_TOKENS
    rows, audits, provenance, seen_targets = [], [], [], {}
    for record in records:
        # Evidence is always retained separately; post-outcome metadata never enters prompts.
        metadata = {k: v for k, v in record.items() if k != "diff"}
        metadata["diff_sha256"] = _sha(record["diff"].encode())
        metadata["source_payload_verified"] = False
        reasons = []
        if record["tier"] not in tiers:
            reasons.append("evidence_tier_not_selected")
        if record["protected"]:
            reasons.append("protected_lineage")
        if record["source_id"] in cutoff_groups:
            reasons.append("lineage_unknown_or_after_cutoff")
        if selected_case_ids is not None and record["case_id"] not in selected_case_ids:
            reasons.append("not_in_explicit_case_selection")
        if context_mode == "parent_files" and reasons:
            audits.append(
                {
                    "case_id": record["case_id"],
                    "source_id": record["source_id"],
                    "tier": record["tier"],
                    "fits": False,
                    "selected": False,
                    "truncated": False,
                    "tokenization_status": "not_attempted_unselected_parent_source",
                    "exclusion_reasons": reasons,
                }
            )
            metadata.update(disposition="excluded", reasons=reasons)
            provenance.append(metadata)
            ledger.append(metadata)
            continue
        try:
            context, paths = preimage_context(record["diff"], record.get("edit_format"))
            kind = (
                "search/replace"
                if record.get("edit_format") == "search_replace"
                or record["diff"].startswith("```edit ")
                else "unified diff"
            )
            prompt = _prompt(context, kind)
            if context_mode == "parent_files":
                context, binding = _parent_context(record, paths, parent_repo)
                metadata["parent_source_binding"] = binding
                metadata["source_payload_verified"] = True
                prompt = (
                    f"Propose an Apple Silicon inference optimization preserving numerical behavior. Return only a complete {kind} edit. "
                    "Below are complete pinned parent files selected from a historical edit. The surrounding repository, build contracts and performance workload are not included; this is an editing exercise, not an executable optimization benchmark.\n\n"
                    + context
                )
            pair = {"prompt": prompt, "completion": record["diff"]}
            prepared = tokenize_row(pair, tokenizer)
            ids, start = prepared["input_ids"], prepared["loss_start"]
            audit = {
                "case_id": record["case_id"],
                "source_id": record["source_id"],
                "tier": record["tier"],
                "prompt_tokens": start,
                "target_tokens": len(ids) - start,
                "total_tokens": len(ids),
                "limit": limit,
                "fits": len(ids) <= limit,
                "token_ids_sha256": _sha(_canonical(ids)),
                "complete_target_sha256": metadata["diff_sha256"],
                "truncated": False,
            }
            if len(ids) > limit:
                reasons.append("full_serialized_example_exceeds_context")
            else:
                validate_example(
                    prepared, seq_length=max(2, len(ids) - 1), vocab_size=vocab_size, truncate=False
                )
            if not reasons:
                key = _sha(_canonical(prepared))
                if key in seen_targets:
                    reasons.append("duplicate_complete_token_record")
                    metadata["duplicate_of"] = seen_targets[key]
                else:
                    seen_targets[key] = record["case_id"]
                    rows.append(
                        {
                            **pair,
                            "case_id": record["case_id"],
                            "source_id": record["source_id"],
                            "evidence_tier": record["tier"],
                        }
                    )
            metadata["files_changed"] = paths
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            audit = {
                "case_id": record["case_id"],
                "source_id": record["source_id"],
                "tier": record["tier"],
                "fits": False,
                "truncated": False,
                "error": str(error),
            }
            reasons.append("unsupported_edit_or_tokenization")
        audit["selected"] = not reasons
        audit["exclusion_reasons"] = reasons
        audits.append(audit)
        metadata.update(disposition="training" if not reasons else "excluded", reasons=reasons)
        provenance.append(metadata)
        if reasons:
            ledger.append(metadata)
    if max_examples is not None:
        lengths_by_id = {a["case_id"]: a.get("total_tokens", 0) for a in audits}
        rows.sort(key=lambda r: (lengths_by_id[r["case_id"]], r["case_id"]))
        omitted = {r["case_id"] for r in rows[max_examples:]}
        rows = rows[:max_examples]
        for audit in audits:
            if audit["case_id"] in omitted:
                audit["selected"] = False
                audit["exclusion_reasons"].append("explicit_shortest_complete_pilot_limit")
        for metadata in provenance:
            if metadata["case_id"] in omitted:
                metadata.update(
                    disposition="excluded", reasons=["explicit_shortest_complete_pilot_limit"]
                )
                ledger.append(metadata)
    # Verify the source and tokenizer did not change during tokenization.
    if any(_file_sha(_safe_file(root, name)) != value for name, value in files.items()):
        raise ValueError("Source changed during import")
    if (
        tokenizer_identity(model) != token_identity
        or _file_sha(model / "config.json") != config_sha
    ):
        raise ValueError("Tokenizer or model config changed during import")
    lengths = [a["total_tokens"] for a in audits if "total_tokens" in a]
    selected_lengths = [a["total_tokens"] for a in audits if a["selected"]]
    report = {
        "format": "exl3-mlx-ktune-import-v1",
        "source": str(root),
        "model": str(model),
        "task_scope": "historical patch-editing demonstration; not complete executable optimization tasks",
        "prompt_context": "target_selected_preimage_hunks"
        if context_mode == "hunks"
        else "target_selected_complete_parent_files",
        "context_selection_leakage": "Changed files/hunk locations were selected using the historical target; no independent optimization-task claim",
        "source_payload_verified": context_mode == "parent_files",
        "parent_repository": str(parent_repo) if parent_repo is not None else None,
        "evidence_scope": "historical archive receipts, not independently authenticated current kernel correctness or speed",
        "prior_retrieval_only_admission": "preserved in provenance; user-selected training is a new use, not prior full-edit efficacy admission",
        "split_policy": "all selected examples are train; conservative repository/track/commit/diff-connected lineages",
        "evaluation_scope": "training_only",
        "quality_evaluation_available": False,
        "counts": {"train": len(rows), "valid": 0, "test": 0},
        "ready_for_sft": bool(rows),
        "tiers": list(tiers),
        "train_until": train_until,
        "source_files_sha256": files,
        "max_examples": max_examples,
        "selection_policy": "shortest complete eligible token records then case_id"
        if max_examples
        else "all complete eligible records in archive order",
        "selected_case_ids": [r["case_id"] for r in rows],
        "requested_case_ids": list(selected_case_ids) if selected_case_ids is not None else None,
        "snapshot_receipt": snapshot_receipt,
        "tokenizer": token_identity,
        "model_config_sha256": config_sha,
        "max_length": max_length or max(selected_lengths, default=0),
        "context_ceiling": MAX_CONTEXT_TOKENS,
        "length_audit": {
            "examples_with_complete_tokenization": len(lengths),
            "max_total_tokens": max(lengths, default=0),
            "max_selected_tokens": max(selected_lengths, default=0),
            "over_limit": sum(a.get("total_tokens", 0) > limit for a in audits),
            "truncated": 0,
        },
        "evidence_counts": dict(Counter(r["tier"] for r in records)),
        "ledger_entries": len(ledger),
        "selected_lineages": len({row["source_id"] for row in rows}),
        "limitations": [
            "No random row split or held-out generalization claim.",
            "Post-change notes, metrics and mechanism summaries are provenance only.",
            "The large zstd submission corpus is not imported by this bounded importer.",
            "Protection overlap cannot be resolved for excluded/index-only entries without source hashes or lineage metadata; no blanket contamination-free claim.",
            "Archive filename association and self-digests alone do not prove measured source equivalence; parent_files additionally binds source Git objects.",
            "A selected patch still needs source review, apply/build, full correctness and controlled paired timing on the intended target.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ktune-import-", dir=destination.parent))
    try:
        contents = {
            "train.jsonl": rows,
            "valid.jsonl": [],
            "test.jsonl": [],
            "provenance.jsonl": provenance,
            "ledger.jsonl": ledger,
            "length-audit.jsonl": audits,
        }
        for name, values in contents.items():
            (stage / name).write_text("".join(_canonical(v).decode() + "\n" for v in values))
        report["artifacts_sha256"] = {name: _file_sha(stage / name) for name in contents}
        (stage / "ktune-import.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        **report,
        "output": str(destination),
        "manifest_sha256": _file_sha(destination / "ktune-import.json"),
    }


_SNAPSHOT_PATHS = (
    "README.md",
    "LIBRARY.md",
    "library/optimizations.json",
    "library/official",
    "library/experiments/index.jsonl",
    "configs/library-sources.json",
    "scripts/official_results.py",
    "ktune/library.py",
)


def _snapshot_allowed(name):
    path = Path(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and not any(part in {".env", ".git"} for part in path.parts)
        and (name in _SNAPSHOT_PATHS or name.startswith("library/official/"))
    )


def _extract_snapshot(archive, stage, *, max_bytes):
    seen, total, hashes = set(), 0, {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = []
        for member in bundle:
            if len(members) >= 4096:
                raise ValueError("Snapshot exceeds archive entry bound")
            name = member.name.rstrip("/")
            if (
                name in seen
                or not _snapshot_allowed(name)
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe or unexpected snapshot archive member: {name}")
            seen.add(name)
            total += member.size
            if total > max_bytes:
                raise ValueError("Snapshot exceeds uncompressed byte bound")
            members.append(member)
        for member in members:
            target = stage / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1 << 20)
                hashes[member.name] = _file_sha(target)
    return hashes


def _download_snapshot(host, root, archive, *, max_bytes):
    command = (
        "tar -C " + shlex.quote(root) + " -czf - " + " ".join(map(shlex.quote, _SNAPSHOT_PATHS))
    )
    process = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", host, command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline, received, error = time.monotonic() + 60, 0, bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "archive")
    selector.register(process.stderr, selectors.EVENT_READ, "error")
    try:
        with archive.open("xb") as stream:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Snapshot SSH transfer exceeded 60 seconds")
                for key, _ in selector.select(min(remaining, 1)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "archive":
                        received += len(chunk)
                        if received > max_bytes:
                            raise ValueError("Snapshot exceeds compressed byte bound")
                        stream.write(chunk)
                    else:
                        error.extend(chunk)
                        if len(error) > 65536:
                            raise ValueError("Snapshot SSH error output exceeds bound")
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if process.returncode:
            raise RuntimeError("Snapshot SSH failed: " + error.decode(errors="replace"))
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def snapshot_ktune(
    remote, output, *, max_compressed_bytes=64 << 20, max_uncompressed_bytes=256 << 20
):
    """Copy only the bounded evidence allowlist from ``host:/absolute/ktune``.

    Uses existing SSH trust/config and BatchMode; never bypasses host-key checks.
    No remote writes, .env, .git archive, build trees or corpus bundles. The
    resulting file hashes describe transferred bytes, not independent evidence
    authentication or an atomic Git checkout snapshot.
    """
    host, root = _remote_parts(remote)
    for bound in (max_compressed_bytes, max_uncompressed_bytes):
        if type(bound) is not int or not 1 <= bound <= 1 << 30:
            raise ValueError("Snapshot bounds must be positive integers <=1 GiB")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    revision = (
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, "git -C " + shlex.quote(root) + " rev-parse HEAD"],
            capture_output=True,
            check=True,
            timeout=30,
        )
        .stdout.decode()
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Remote revision is not a full Git commit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ktune-snapshot-", dir=destination.parent))
    try:
        archive, stage = temporary / "snapshot.tar.gz", temporary / "source"
        stage.mkdir()
        _download_snapshot(host, root, archive, max_bytes=max_compressed_bytes)
        hashes = _extract_snapshot(archive, stage, max_bytes=max_uncompressed_bytes)
        report = {
            "format": "exl3-mlx-ktune-snapshot-v1",
            "remote": remote,
            "remote_revision": revision,
            "files_sha256": hashes,
            "archive_sha256": _file_sha(archive),
            "archive_bytes": archive.stat().st_size,
            "scope": "bounded read-only evidence snapshot; working files may differ from recorded Git revision",
        }
        (stage / "snapshot-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        stage.rename(destination)
    finally:
        shutil.rmtree(temporary)
    return {**report, "output": str(destination)}


def _remote_parts(remote):
    if not isinstance(remote, str) or ":" not in remote:
        raise ValueError("remote must be host:/absolute/path")
    host, root = remote.split(":", 1)
    if (
        not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host)
        or not root.startswith("/")
        or any(c in root for c in "\n\r\x00")
    ):
        raise ValueError("Invalid SSH host or absolute remote root")
    return host, root
