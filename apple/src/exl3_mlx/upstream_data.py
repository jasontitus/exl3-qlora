"""Offline, complete-commit intake; executes no repository code or evidence commands.

A source reconstruction is not a runnable kernel task. Compilation, numerical
checks and timing remain not_run until a separate trusted harness binds them to
exact source/build/workload/hardware. Source-authored booleans cannot promote a
record's verification stage. See docs/UPSTREAM_INTAKE.md for the manifest schema.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .context import MAX_CONTEXT_TOKENS
from .ktune_data import _canonical, _file_sha, _json, _lineages, _sha

FORMAT = "exl3-mlx-upstream-commits-v1"
_MAX_FILE = 8 << 20
_MAX_TOTAL = 32 << 20
_FIELDS = (
    "task",
    "operation",
    "hardware",
    "bottleneck",
    "constraints",
    "rationale",
    "architecture",
    "dispatch",
)
_PROMPT_FIELDS = frozenset(_FIELDS[:5])
_OUTCOMES = {"reported_improvement", "source_only", "negative", "contradictory"}


def _tokenize_pair(row, tokenizer):
    # Exact standard CompletionsDataset/masked chat-template serialization, but
    # without importing mlx_lm.__init__, which initializes Metal on import.
    if set(row) != {"prompt", "completion"} or any(not isinstance(v, str) for v in row.values()):
        raise ValueError("Expected a text prompt/completion pair")
    messages = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["completion"]},
    ]
    ids = tokenizer.apply_chat_template(messages, tools=None, return_dict=False)
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tools=None, add_generation_prompt=True, return_dict=False
    )
    if ids[: len(prefix)] != prefix:
        raise ValueError("Chat template prefix is inconsistent")
    return {"input_ids": ids, "loss_start": max(1, len(prefix))}


def _validate_tokens(row, *, seq_length, vocab_size, truncate):
    if truncate or not 2 <= seq_length <= MAX_CONTEXT_TOKENS:
        raise ValueError("Complete examples only within context ceiling")
    ids, start = row["input_ids"], row["loss_start"]
    if (
        not isinstance(ids, list)
        or len(ids) < 2
        or any(type(t) is not int or not 0 <= t < vocab_size or t == 129264 for t in ids)
        or type(start) is not int
        or not 1 <= start < len(ids)
        or len(ids) > seq_length + 1
    ):
        raise ValueError("Invalid complete text token record")
    return row


def _helpers():
    from transformers import PreTrainedTokenizerFast

    def load(root):
        return PreTrainedTokenizerFast.from_pretrained(str(root), local_files_only=True)

    def identity(root):
        return {
            name: _file_sha(Path(root) / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        }

    return load, _tokenize_pair, identity, _validate_tokens


def _oid(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("A full lowercase 40-character Git object ID is required")
    return value


def _path(value):
    if not isinstance(value, str):
        raise ValueError("Source path must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or any(p.lower() in {".git", ".env"} for p in path.parts)
        or any(c.isspace() or c in "\\\x00" for c in value)
    ):
        raise ValueError(f"Unsupported source path: {value!r}")
    return value


def _git(repo, *args, max_bytes=_MAX_TOTAL, stdin=None):
    """Bounded read-only Git query, without hooks/filters/network/lazy fetch."""
    if not args or args[0] not in {"cat-file", "diff", "ls-tree", "patch-id"}:
        raise ValueError("Unsupported Git query")
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_NO_LAZY_FETCH="1",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_OPTIONAL_LOCKS="0",
        LC_ALL="C",
    )
    command = [
        "git",
        "--no-pager",
        "--literal-pathspecs",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.attributesFile={os.devnull}",
        "-c",
        "protocol.allow=never",
        "-C",
        str(repo),
        *args,
    ]
    # Only patch-id has stdin. Its source is our already bounded generated diff.
    if stdin is not None:
        if args != ("patch-id", "--stable") or len(stdin) > _MAX_TOTAL:
            raise ValueError("Unsupported Git input")
        result = subprocess.run(command, input=stdin, capture_output=True, env=env, timeout=30)
        if result.returncode or len(result.stdout) > max_bytes:
            raise ValueError("Git patch-ID query failed")
        return result.stdout
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
    )
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "out")
    selector.register(process.stderr, selectors.EVENT_READ, "err")
    output, errors, deadline = bytearray(), bytearray(), time.monotonic() + 30
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("Git query exceeded 30 seconds")
            for key, _ in selector.select(min(1, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = output if key.data == "out" else errors
                target.extend(chunk)
                if len(target) > (max_bytes if key.data == "out" else 65536):
                    raise ValueError("Git query exceeds byte bound")
        process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if process.returncode:
            raise ValueError(
                "Offline Git source unavailable: " + errors.decode(errors="replace")[:300]
            )
        return bytes(output)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()


def _object(repo, kind, oid, bound):
    import hashlib

    size = int(_git(repo, "cat-file", "-s", oid, max_bytes=100))
    if size > bound:
        raise ValueError(f"{kind} exceeds source byte bound")
    raw = _git(repo, "cat-file", kind, oid, max_bytes=bound)
    if hashlib.sha1(f"{kind} {len(raw)}\0".encode() + raw).hexdigest() != oid:
        raise ValueError("Git object bytes do not match pinned identity")
    return raw


def _commit(repo, oid):
    raw = _object(repo, "commit", oid, 1 << 20).decode("utf-8")
    header, message = raw.split("\n\n", 1)
    parents = [line[7:] for line in header.splitlines() if line.startswith("parent ")]
    tree = next(line[5:] for line in header.splitlines() if line.startswith("tree "))
    return {
        "commit": oid,
        "parents": parents,
        "tree": _oid(tree),
        "message": message,
        "raw_commit_sha256": _sha(raw.encode()),
    }


def _file(repo, revision, path):
    raw = _git(repo, "ls-tree", "-z", revision, "--", path, max_bytes=2048)
    if not raw:
        return None
    entries = raw.rstrip(b"\0").split(b"\0")
    if len(entries) != 1:
        raise ValueError("Ambiguous Git tree entry")
    header, name = entries[0].split(b"\t", 1)
    mode, kind, oid = header.decode().split()
    if name.decode() != path or mode not in {"100644", "100755"} or kind != "blob":
        raise ValueError("Only regular source files are supported; no links/submodules")
    content = _object(repo, "blob", _oid(oid), _MAX_FILE)
    text = content.decode("utf-8")
    if "\x00" in text:
        raise ValueError("Binary source file is unsupported")
    return {"mode": mode, "blob": oid, "sha256": _sha(content), "content": text}


def _evidence(root, sources):
    if not isinstance(sources, dict):
        raise ValueError("evidence must be an object")
    result, files = {}, {}
    for name, source in sources.items():
        if not isinstance(name, str) or not name or name == "commit_message":
            raise ValueError("Invalid evidence identity")
        relative = _path(source["path"])
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > _MAX_FILE:
            raise ValueError("Evidence path escapes manifest directory or exceeds bound")
        raw = path.read_bytes()
        if _sha(raw) != source.get("sha256"):
            raise ValueError("Evidence digest mismatch")
        if source.get("phase") not in {"pre_change", "post_change", "unknown"}:
            raise ValueError("Evidence needs explicit pre_change/post_change/unknown phase")
        if urlsplit(source.get("url", "")).scheme not in {"https", "http"}:
            raise ValueError("Evidence needs an HTTP(S) source URL")
        result[name] = {**source, "text": raw.decode("utf-8")}
        files[relative] = _sha(raw)
    return result, files


def _citations(value, sources):
    if not isinstance(value, list) or not value:
        raise ValueError("Grounded annotation needs source citations")
    phases = []
    for citation in value:
        if not isinstance(citation, dict) or set(citation) != {"source", "quote"}:
            raise ValueError("Each citation needs source and exact quote")
        source = sources.get(citation["source"])
        quote = citation["quote"]
        if source is None or not isinstance(quote, str) or not quote or quote not in source["text"]:
            raise ValueError("Annotation quote does not occur in bound evidence")
        phases.append(source["phase"])
    return phases


def _annotations(candidate, sources):
    supplied = candidate.get("annotations", {})
    if not isinstance(supplied, dict) or set(supplied) - set(_FIELDS):
        raise ValueError("Unknown annotation field")
    result = {}
    for field in _FIELDS:
        item = supplied.get(field)
        if item is None:
            result[field] = {"status": "unknown", "text": None, "evidence": []}
            continue
        if (
            not isinstance(item, dict)
            or set(item) != {"text", "basis", "evidence"}
            or item["basis"] not in {"source", "inferred"}
            or not isinstance(item["text"], str)
            or not item["text"].strip()
        ):
            raise ValueError("Annotations require text, source/inferred basis and evidence")
        phases = _citations(item["evidence"], sources)
        result[field] = {
            **item,
            "status": "source_reported" if item["basis"] == "source" else "inferred",
            "phase": "pre_change" if set(phases) == {"pre_change"} else "post_or_unknown",
        }
    if result["rationale"]["status"] == "unknown" and sources["commit_message"]["text"].strip():
        result["rationale"] = {
            "status": "source_reported",
            "text": sources["commit_message"]["text"],
            "phase": "post_or_unknown",
            "basis": "source",
            "evidence": [{"source": "commit_message", "quote": sources["commit_message"]["text"]}],
        }
    kind = candidate.get("task_kind")
    if kind is None:
        kind = {"value": "unknown", "reviewed_by": None, "evidence": []}
    else:
        if (
            set(kind) != {"value", "reviewed_by", "evidence"}
            or kind["value"] not in {"kernel", "host_dispatch", "integration"}
            or not isinstance(kind["reviewed_by"], str)
            or not kind["reviewed_by"].strip()
        ):
            raise ValueError("task_kind must be an explicit reviewed declaration")
        _citations(kind["evidence"], sources)
    return result, kind


def _repository_identity(url):
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("Repository needs a canonical public HTTPS URL")
    path = parsed.path.rstrip("/").removesuffix(".git")
    if not path or path == "/":
        raise ValueError("Repository URL lacks owner/name")
    return f"https://{parsed.hostname.lower()}{path.lower() if parsed.hostname.lower() == 'github.com' else path}"


def reconstruct_candidate(candidate, repositories, evidence):
    """Resolve complete immutable source objects. No builds or candidate execution."""
    repo_config = repositories[candidate["repository"]]
    repo = Path(repo_config["path"]).expanduser().resolve(strict=True)
    identity = _repository_identity(repo_config["url"])
    head = _commit(repo, _oid(candidate["commit"]))
    parent = candidate.get("parent")
    if parent is None and len(head["parents"]) == 1:
        parent = head["parents"][0]
    if _oid(parent) not in head["parents"]:
        raise ValueError("Explicit parent must be a direct parent; merges require parent selection")
    before = _commit(repo, parent)
    names = _git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--name-only",
        "-z",
        parent,
        head["commit"],
        "--",
    )
    paths = [_path(p.decode("utf-8")) for p in names.split(b"\0") if p]
    if not paths or len(paths) > 128 or len({p.casefold() for p in paths}) != len(paths):
        raise ValueError("Require 1..128 distinct non-case-colliding changed files")
    dependencies = candidate.get("dependency_paths", [])
    if not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies)):
        raise ValueError("dependency_paths must be distinct source paths")
    dependencies = [_path(p) for p in dependencies if p not in paths]
    if len(paths) + len(dependencies) > 128:
        raise ValueError("Too many context files")
    files, size = {}, 0
    for path in [*paths, *dependencies]:
        pair = {
            "before": _file(repo, parent, path),
            "after": _file(repo, head["commit"], path),
            "role": "changed" if path in paths else "declared_dependency",
        }
        if pair["before"] is None and pair["after"] is None:
            raise ValueError("Declared dependency is absent at both revisions")
        size += sum(
            len(item["content"].encode()) for item in (pair["before"], pair["after"]) if item
        )
        if size > _MAX_TOTAL:
            raise ValueError("Complete before/after source exceeds record byte bound")
        files[path] = pair
    patch = _git(
        repo,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--no-color",
        "--full-index",
        "--unified=3",
        parent,
        head["commit"],
        "--",
    ).decode("utf-8")
    patch_id = (
        _git(repo, "patch-id", "--stable", stdin=patch.encode(), max_bytes=2048).decode().split()
    )
    if not patch_id:
        raise ValueError("No stable patch identity; metadata-only edits are unsupported")
    sources = {
        **evidence,
        "commit_message": {
            "text": head["message"],
            "phase": "post_change",
            "sha256": _sha(head["message"].encode()),
            "url": identity + "/commit/" + head["commit"],
        },
    }
    annotations, kind = _annotations(candidate, sources)
    outcome = candidate.get("outcome", "source_only")
    if outcome not in _OUTCOMES or type(candidate.get("protected", False)) is not bool:
        raise ValueError("Invalid outcome or protection flag")
    related = candidate.get("related_commits", [])
    if not isinstance(related, list):
        raise ValueError("related_commits must be a list")
    keys = ["repository:" + identity, "stable_patch:" + patch_id[0]]
    if candidate.get("source_group"):
        if not isinstance(candidate["source_group"], str):
            raise ValueError("source_group must be text")
        keys.append("declared_group:" + candidate["source_group"])
    record = {
        "case_id": candidate["id"],
        "source": "pinned_upstream_commit",
        "repository": identity,
        "parent": before,
        "head": head,
        "context_policy": "complete_changed_files",
        "dependency_scope": "declared_files_only; complete build/runtime dependency closure unknown",
        "files": files,
        "diff": patch,
        "diff_sha256": _sha(patch.encode()),
        "stable_patch_id": patch_id[0],
        "task_kind": kind,
        "annotations": annotations,
        "evidence": sources,
        "outcome": outcome,
        "outcome_support": "curator_declared_classification_not_semantically_verified",
        "annotation_grounding": "exact_quote_containment_verified; semantic_paraphrase_grounding_is_curator_asserted",
        "protected": candidate.get("protected", False),
        "lineage_keys": keys,
        "commits": [parent, head["commit"], *head["parents"], *[_oid(c) for c in related]],
        "verification": {
            "source_reconstruction": "verified_git_objects",
            "compile": "not_run",
            "numerical": "not_run",
            "runtime": "not_run",
            "execution_verified": False,
        },
        "input_metadata": candidate,
    }
    conditioning = candidate.get("task_conditioning", "pre_change_only")
    if conditioning not in {"pre_change_only", "historical_source"}:
        raise ValueError("task_conditioning must be pre_change_only or historical_source")
    record["task_conditioning"] = conditioning
    _reviewed_context(record, candidate)
    return record


def _slice(side, binding):
    if not isinstance(binding, dict) or set(binding) != {"start", "end", "sha256"}:
        raise ValueError("Reviewed unit requires byte start/end and SHA-256")
    raw = b"" if side is None else side["content"].encode()
    start, end = binding["start"], binding["end"]
    if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(raw):
        raise ValueError("Reviewed unit byte indices outside pinned file")
    selected = raw[start:end]
    if _sha(selected) != binding["sha256"]:
        raise ValueError("Reviewed unit source hash mismatch")
    return selected.decode("utf-8")


def _reviewed_context(record, candidate):
    policy = candidate.get("context_policy", "complete_changed_files")
    if policy == "complete_changed_files":
        if "operation_scope" in candidate:
            raise ValueError("operation_scope requires reviewed_operation context policy")
        if "reviewed_units" in candidate:
            raise ValueError("reviewed_units requires explicit reviewed_units context policy")
        return
    if policy not in {"reviewed_units", "reviewed_operation"}:
        raise ValueError("Unsupported context policy")
    declaration = candidate.get("reviewed_units")
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"reviewed_by", "dependency_review", "units"}
        or any(
            not isinstance(declaration[k], str) or not declaration[k].strip()
            for k in ("reviewed_by", "dependency_review")
        )
        or not isinstance(declaration["units"], list)
        or not declaration["units"]
    ):
        raise ValueError("Reviewed units need reviewer, dependency review and explicit units")
    selected = {}
    for unit in declaration["units"]:
        if (
            not isinstance(unit, dict)
            or set(unit) != {"path", "kind", "before", "after"}
            or unit["kind"]
            not in {
                "complete_function",
                "complete_kernel",
                "complete_type",
                "complete_declaration",
                "whole_file",
                *(["complete_dispatch_branch"] if policy == "reviewed_operation" else []),
            }
        ):
            raise ValueError("Only explicitly reviewed complete source units are supported")
        path = _path(unit["path"])
        if path not in record["files"]:
            raise ValueError("Reviewed unit path is outside changed/dependency files")
        pair = record["files"][path]
        rendered = {
            **unit,
            "before_content": _slice(pair["before"], unit["before"]),
            "after_content": _slice(pair["after"], unit["after"]),
        }
        if unit["kind"] == "whole_file" and any(
            unit[side]["start"] != 0
            or unit[side]["end"] != len((pair[side] or {"content": ""})["content"].encode())
            for side in ("before", "after")
        ):
            raise ValueError("whole_file declaration must include all bytes")
        selected.setdefault(path, []).append(rendered)
    scoped = policy == "reviewed_operation"
    scope = candidate.get("operation_scope")
    if scoped and (
        not isinstance(scope, dict)
        or set(scope) != {"description", "omitted_path_reasons"}
        or not isinstance(scope["description"], str)
        or not scope["description"].strip()
        or not isinstance(scope["omitted_path_reasons"], dict)
        or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in scope["omitted_path_reasons"].values()
        )
    ):
        raise ValueError("Scoped operation requires description and explicit omission reasons")
    if not scoped and scope is not None:
        raise ValueError("operation_scope requires reviewed_operation context policy")
    omissions = []
    changed_units = 0

    def omitted(path, before, after, b0, b1, a0, a1):
        if before[b0:b1] == after[a0:a1]:
            return
        omissions.append(
            {
                "path": path,
                "before": {"start": b0, "end": b1, "sha256": _sha(before[b0:b1])},
                "after": {"start": a0, "end": a1, "sha256": _sha(after[a0:a1])},
            }
        )

    # Replacing all declared units must reproduce EVERY full changed file. This
    # binds both ranges, their order, all unchanged gaps and every edit. It does
    # not prove language-level completeness or a reviewer's dependency claim.
    for path, pair in record["files"].items():
        units = sorted(
            selected.get(path, []), key=lambda u: (u["before"]["start"], u["after"]["start"])
        )
        if not units and not scoped:
            if pair["role"] == "changed":
                raise ValueError("Reviewed units omit a changed file")
            continue
        before = (pair["before"] or {"content": ""})["content"].encode()
        after = (pair["after"] or {"content": ""})["content"].encode()
        old_cursor = new_cursor = 0
        for unit in units:
            b, a = unit["before"], unit["after"]
            if b["start"] < old_cursor or a["start"] < new_cursor:
                raise ValueError("Reviewed units overlap or change order")
            if scoped:
                omitted(path, before, after, old_cursor, b["start"], new_cursor, a["start"])
            elif before[old_cursor : b["start"]] != after[new_cursor : a["start"]]:
                raise ValueError("Reviewed units omit an edit in an unchanged gap")
            changed_units += before[b["start"] : b["end"]] != after[a["start"] : a["end"]]
            old_cursor, new_cursor = b["end"], a["end"]
        if scoped:
            omitted(path, before, after, old_cursor, len(before), new_cursor, len(after))
        elif before[old_cursor:] != after[new_cursor:]:
            raise ValueError("Reviewed units omit an edit after declared units")
        if units:
            selected[path] = units
    if scoped:
        if not changed_units:
            raise ValueError("Scoped operation must contain an actual changed source unit")
        if {r["path"] for r in omissions} != set(scope["omitted_path_reasons"]):
            raise ValueError("Omission reasons must exactly cover paths with omitted changed bytes")
        record["operation_scope"] = {
            **scope,
            "omitted_changed_regions": omissions,
            "whole_commit_replayed": False,
            "scope_semantics": "Reviewer-selected operation; no claim of full-commit coverage",
        }
    record["context_policy"] = policy
    record["reviewed_context"] = {
        **declaration,
        "units": selected,
        "coverage": "selected units only; omitted changed regions explicitly bound"
        if scoped
        else "all changed file bytes reconstructed exactly",
        "semantic_completeness": "reviewer_declared_not_independently_verified",
    }
    record["dependency_scope"] = (
        "reviewer_declared: "
        + declaration["dependency_review"]
        + "; not independently compiled or executed"
    )


def _render_files(record, side):
    result = {}
    for path, pair in record["files"].items():
        if (
            record.get("context_policy") == "reviewed_operation"
            and path not in record["reviewed_context"]["units"]
            and pair["role"] == "changed"
        ):
            continue
        if side == "after" and pair["role"] != "changed":
            continue
        if pair[side] is None:
            result[path] = None
            continue
        units = record.get("reviewed_context", {}).get("units", {}).get(path)
        if units:
            result[path] = {
                "mode": pair[side]["mode"] if pair[side] else None,
                "source_units": [
                    {"kind": unit["kind"], "content": unit[side + "_content"]} for unit in units
                ],
                "scope": "reviewer-declared complete units; full source retained in provenance",
            }
        else:
            result[path] = (
                None
                if pair[side] is None
                else {"mode": pair[side]["mode"], "content": pair[side]["content"]}
            )
    return result


def structured_pair(record):
    """Pre-change source in prompt, post-change evidence/complete edit in answer."""
    unknown = {"status": "unknown", "text": None, "evidence": []}
    task = {
        field: record["annotations"][field]
        if (
            record["annotations"][field].get("phase") == "pre_change"
            or record["task_conditioning"] == "historical_source"
        )
        and record["annotations"][field]["status"] == "source_reported"
        else unknown
        for field in _FIELDS
        if field in _PROMPT_FIELDS
    }
    prompt = {
        "instruction": "Reconstruct a complete historical optimization edit. Preserve the baseline's intended behavior. Provide grounded analysis, full edited source or reviewed source units (according to context policy) and host dispatch changes. State unknowns; verification is performed separately by a harness.",
        "scope": "Historical target-selected "
        + record["context_policy"]
        + "; not an independently specified executable kernel benchmark.",
        "task_conditioning": record["task_conditioning"],
        "task_conditioning_scope": "Retrospectively authored, target-derived task specification; not prospective optimization discovery"
        if record["task_conditioning"] == "historical_source"
        else "Only bound pre-change source conditions task metadata",
        "task": {
            field: {k: v for k, v in item.items() if k != "evidence"}
            for field, item in task.items()
        },
        "baseline_files": _render_files(record, "before"),
        "dependency_scope": record["dependency_scope"],
    }
    if "operation_scope" in record:
        prompt["operation_scope"] = record["operation_scope"]["description"]
    completion = {
        "analysis": {
            field: {k: v for k, v in item.items() if k != "evidence"}
            for field, item in record["annotations"].items()
        },
        "task_kind": record["task_kind"]["value"],
        "optimized_files": _render_files(record, "after"),
        "verification_metadata": {
            "source_claims": {
                "classification": "curator_declared_" + record["outcome"],
                "independently_measured": False,
            },
            "compile": "not_run",
            "numerical": "not_run",
            "runtime": "not_run",
            "note": "Reported rationale is historical source evidence; model-authored verification status cannot certify a candidate.",
        },
    }
    return {"prompt": _canonical(prompt).decode(), "completion": _canonical(completion).decode()}


def import_upstream(
    source,
    model,
    output,
    *,
    max_length=None,
    max_examples=None,
    minimum_verification="source_reconstructed",
):
    """Import train-only complete examples, preserving every excluded candidate."""
    if max_length is not None and (
        type(max_length) is not int or not 3 <= max_length <= MAX_CONTEXT_TOKENS
    ):
        raise ValueError(f"max_length must be 3..{MAX_CONTEXT_TOKENS} or None")
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise ValueError("max_examples must be a positive integer")
    if minimum_verification not in {"source_reconstructed", "execution_verified"}:
        raise ValueError("Unknown verification admission level")
    source = Path(source).expanduser().resolve(strict=True)
    if source.stat().st_size > _MAX_TOTAL:
        raise ValueError("Manifest exceeds byte bound")
    raw = source.read_bytes()
    manifest = _json(raw)
    supplied = manifest.pop("manifest_sha256", None)
    if manifest.get("format") != FORMAT or supplied != _sha(_canonical(manifest)):
        raise ValueError("Unsupported manifest format or canonical checksum mismatch")
    candidates, repositories = manifest["candidates"], manifest["repositories"]
    if not isinstance(candidates, list) or not isinstance(repositories, dict):
        raise ValueError("candidates/repositories have invalid types")
    ids = [c["id"] for c in candidates]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Candidate identities must be unique nonempty strings")
    evidence, source_files = _evidence(source.parent, manifest.get("evidence", {}))
    source_files[source.name] = _sha(raw)
    protected_keys = manifest.get("protected_keys", [])
    if not isinstance(protected_keys, list) or any(not isinstance(k, str) for k in protected_keys):
        raise ValueError("protected_keys must be a list of lineage identities")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    records, failed = [], []
    for candidate in candidates:
        try:
            record = reconstruct_candidate(candidate, repositories, evidence)
            record["protected"] |= bool(
                set(protected_keys)
                & set(record["lineage_keys"] + ["commit:" + c for c in record["commits"]])
            )
            records.append(record)
        except (
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            OSError,
            subprocess.TimeoutExpired,
        ) as error:
            failed.append(
                {
                    "case_id": candidate["id"],
                    "disposition": "excluded",
                    "reasons": ["source_reconstruction_failed"],
                    "error": str(error),
                    "input_metadata": candidate,
                }
            )
    # Failed candidates still carry protection by declared repo/group/commit identity.
    placeholders = []
    for item in failed:
        c = item["input_metadata"]
        repo = repositories.get(c.get("repository"), {})
        try:
            identity = _repository_identity(repo.get("url", ""))
        except (TypeError, ValueError):
            identity = "unresolved:" + str(c.get("repository"))
        keys = ["repository:" + identity]
        if isinstance(c.get("source_group"), str):
            keys.append("declared_group:" + c["source_group"])
        commits = [
            x
            for x in [c.get("commit"), c.get("parent"), *c.get("related_commits", [])]
            if isinstance(x, str)
        ]
        placeholders.append(
            {
                "case_id": c["id"],
                "lineage_keys": keys,
                "commits": commits,
                "diff": "unresolved:" + c["id"],
                "protected": c.get("protected") is True
                or bool(set(protected_keys) & set(keys + ["commit:" + c for c in commits])),
            }
        )
    _lineages([*records, *placeholders])
    for item, placeholder in zip(failed, placeholders, strict=True):
        item["source_id"] = placeholder["source_id"]
        item["protected"] = placeholder["protected"]
    load_tokenizer, tokenize_row, tokenizer_identity, validate = _helpers()
    model = Path(model).expanduser().resolve(strict=True)
    identity, config_hash = tokenizer_identity(model), _file_sha(model / "config.json")
    config = _json((model / "config.json").read_bytes())
    vocab = config.get("text_config", config)["vocab_size"]
    tokenizer, limit = load_tokenizer(model), max_length or MAX_CONTEXT_TOKENS
    rows, audits, provenance = [], [], []
    dedup = {}
    disputed = {
        r["stable_patch_id"] for r in records if r["outcome"] in {"negative", "contradictory"}
    }
    for record in records:
        reasons = []
        if record["protected"]:
            reasons.append("protected_lineage")
        if record["outcome"] in {"negative", "contradictory"}:
            reasons.append("nonpositive_outcome")
        elif record["stable_patch_id"] in disputed:
            reasons.append("equivalent_patch_has_negative_or_contradictory_evidence")
        if minimum_verification == "execution_verified":
            reasons.append("bound_execution_verifier_not_implemented")
        key = record["stable_patch_id"]
        pair = structured_pair(record)
        audit = {"case_id": record["case_id"], "source_id": record["source_id"], "truncated": False}
        try:
            prepared = tokenize_row(pair, tokenizer)
            ids, start = prepared["input_ids"], prepared["loss_start"]
            if not ids or ids[-1] != tokenizer.eos_token_id:
                raise ValueError("Tokenized assistant answer must end with model EOS")
            audit.update(
                total_tokens=len(ids),
                prompt_tokens=start,
                target_tokens=len(ids) - start,
                token_ids_sha256=_sha(_canonical(ids)),
                complete_target_sha256=_sha(pair["completion"].encode()),
            )
            if len(ids) > limit:
                reasons.append("full_serialized_example_exceeds_context")
            else:
                validate(
                    prepared, seq_length=max(2, len(ids) - 1), vocab_size=vocab, truncate=False
                )
        except (ValueError, TypeError, KeyError) as error:
            reasons.append("tokenization_failed")
            audit["error"] = str(error)
        if key in dedup:
            reasons.append("duplicate_normalized_patch")
            record["duplicate_of"] = dedup[key]
        elif not reasons:
            dedup[key] = record["case_id"]
        audit.update(selected=not reasons, exclusion_reasons=reasons)
        audits.append(audit)
        record.update(
            disposition="excluded" if reasons else "training",
            reasons=reasons,
            record_sha256=_sha(_canonical(record)),
        )
        provenance.append(record)
        if not reasons:
            rows.append(
                {
                    **pair,
                    "case_id": record["case_id"],
                    "source_id": record["source_id"],
                    "evidence_tier": "source_reconstructed",
                }
            )
    if max_examples is not None:
        lengths = {a["case_id"]: a.get("total_tokens", 0) for a in audits}
        rows.sort(key=lambda r: (lengths[r["case_id"]], r["case_id"]))
        omitted = {r["case_id"] for r in rows[max_examples:]}
        rows = rows[:max_examples]
        for audit in audits:
            if audit["case_id"] in omitted:
                audit["selected"] = False
                audit["exclusion_reasons"].append("explicit_shortest_complete_limit")
        for record in provenance:
            if record["case_id"] in omitted:
                record.update(disposition="excluded", reasons=["explicit_shortest_complete_limit"])
    if _file_sha(source) != _sha(raw) or any(
        _file_sha(source.parent / p) != h for p, h in source_files.items()
    ):
        raise ValueError("Source manifest/evidence changed during import")
    if tokenizer_identity(model) != identity or _file_sha(model / "config.json") != config_hash:
        raise ValueError("Model tokenizer/config changed during import")
    for record in provenance:
        record.pop("record_sha256", None)
        record["record_sha256"] = _sha(_canonical(record))
    ledger = [*failed, *[r for r in provenance if r["disposition"] == "excluded"]]
    report = {
        "format": "exl3-mlx-upstream-import-v1",
        "source": str(source),
        "source_files_sha256": source_files,
        "model": str(model),
        "tokenizer": identity,
        "model_config_sha256": config_hash,
        "minimum_verification": minimum_verification,
        "context_policies": dict(Counter(r["context_policy"] for r in records)),
        "counts": {
            "candidates": len(candidates),
            "source_reconstructed": len(records),
            "unique_normalized_patches": len({r["stable_patch_id"] for r in records}),
            "train": len(rows),
            "valid": 0,
            "test": 0,
            "excluded": len(ledger),
        },
        "ready_for_sft": bool(rows),
        "selected_case_ids": [r["case_id"] for r in rows],
        "max_length": max([a["total_tokens"] for a in audits if a["selected"]], default=0),
        "max_examples": max_examples,
        "context_ceiling": limit,
        "truncated": 0,
        "exclusion_counts": dict(Counter(reason for item in ledger for reason in item["reasons"])),
        "evaluation_scope": "training_only",
        "quality_evaluation_available": False,
        "execution_verified_examples": 0,
        "selected_lineages": len({r["source_id"] for r in rows}),
        "limitations": [
            "No compiler, numerical checker or benchmark ran; no executable-kernel quality claim.",
            "Target-selected complete changed files are not complete repository/build/runtime dependencies.",
            "Commit rationale and metrics are source-reported; inferred annotations are explicitly labeled.",
            "All examples train-only; repository/commit/declared-group/normalized-patch connections and explicit protected keys are conservative, not blanket contamination assurance.",
            "No best-of-N generation or independently confirmed performance selection is implemented by this intake.",
            "Unknown metadata stays unknown; large complete examples are excluded without cropping.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".upstream-import-", dir=destination.parent))
    try:
        contents = {
            "train.jsonl": rows,
            "valid.jsonl": [],
            "test.jsonl": [],
            "provenance.jsonl": [*provenance, *failed],
            "ledger.jsonl": ledger,
            "length-audit.jsonl": audits,
        }
        for name, values in contents.items():
            (stage / name).write_bytes(b"".join(_canonical(v) + b"\n" for v in values))
        report["artifacts_sha256"] = {name: _file_sha(stage / name) for name in contents}
        (stage / "ktune-import.json").write_bytes(_canonical(report) + b"\n")
        stage.rename(destination)
    finally:
        if stage.exists():
            import shutil

            shutil.rmtree(stage)
    return {
        **report,
        "output": str(destination),
        "manifest_sha256": _file_sha(destination / "ktune-import.json"),
    }
