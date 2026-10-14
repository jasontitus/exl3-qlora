"""Portable exact-edit admission. No supplied command or repository code executes."""

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from .agent_edits import _path, full_commit_images, parse_command, replay, unified_patch
from .context import MAX_CONTEXT_TOKENS
from .evaluation_audit import _json

FORMAT = "exl3-reviewed-agent-corpus-v1"
MIXED_FORMAT = "exl3-reviewed-agent-corpus-v2"
ROLES = {"source", "raw", "prepared", "replay"}
SPLITS = ("train", "valid", "test")
MAX_FILE = 64 << 20
MAX_TOTAL = 512 << 20


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def mapping(value, label, required=()):
    require(isinstance(value, dict), label + ": expected object")
    require(set(required) <= set(value), label + ": missing required fields")
    return value


def lines(payload):
    return [_json(line) for line in payload.splitlines() if line.strip()]


class Capture:
    def __init__(self, root):
        self.root, self.files = root, {}

    def read(self, binding):
        require(
            isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "Invalid file binding"
        )
        name, checksum = binding["path"], binding["sha256"]
        require(isinstance(name, str), "Invalid path")
        relative = PurePosixPath(name)
        require(
            not relative.is_absolute()
            and str(relative) == name
            and not any(x in ("..", ".") for x in relative.parts),
            "Unsafe bundle path",
        )
        require(
            isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum), "Invalid SHA256"
        )
        path = self.root / name
        require(
            path.resolve().is_relative_to(self.root)
            and all(
                not p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(self.root)
            ),
            "Symlink/path escape",
        )
        require(path.is_file() and path.stat().st_size <= MAX_FILE, "Missing or oversized input")
        if name not in self.files:
            payload = path.read_bytes()
            require(
                len(payload) <= MAX_FILE
                and sum(map(len, self.files.values())) + len(payload) <= MAX_TOTAL,
                "Corpus exceeds captured byte bound",
            )
            self.files[name] = payload
        require(sha(self.files[name]) == checksum, "Input SHA256 mismatch: " + name)
        return self.files[name]

    def check(self):
        for name, payload in self.files.items():
            require(
                self.read({"path": name, "sha256": sha(payload)}) == payload
                and (self.root / name).read_bytes() == payload,
                "Input changed during preparation",
            )


def publish_new(stage, destination):
    """Atomic no-replace directory publication on supported training hosts."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = fn(os.fsencode(stage), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = fn(-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    else:
        raise ValueError("Atomic exclusive publication unsupported on this host")
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(code, os.strerror(code), str(destination))


def patch_id(before, after):
    patch = "".join(
        f"diff --git a/{name} b/{name}\n" + unified_patch({name: before[name]}, {name: after[name]})
        for name in sorted(before)
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    with tempfile.TemporaryDirectory(prefix="reviewed-patch-id-") as directory:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "patch-id", "--stable"],
            input=patch.encode(),
            capture_output=True,
            check=True,
            timeout=30,
            cwd=directory,
            env=env,
        )
    value = result.stdout.decode().split()
    require(
        len(value) == 2 and re.fullmatch(r"[0-9a-f]{40}", value[0]),
        "Invalid normalized patch identity",
    )
    return value[0]


def validate_source_blobs(source):
    mapping(source, "source", {"files", "repository", "head", "commits"})
    require(isinstance(source.get("files"), dict) and source["files"], "Missing source files")
    for name, pair in source["files"].items():
        require(isinstance(name, str), "Source path must be text")
        _path(name)
        mapping(pair, "source file pair", {"role", "before", "after"})
        require(
            isinstance(pair["role"], str) and pair["role"] in {"changed", "context"},
            "Unknown source role",
        )
        for side in ("before", "after"):
            value = pair[side]
            if value is None:
                continue
            mapping(value, "source blob", {"content", "mode", "sha256", "blob"})
            require(isinstance(value["content"], str), "Source content must be text")
            require(
                isinstance(value["mode"], str) and value["mode"] in {"100644", "100755"},
                "Source requires regular text modes",
            )
            require("\0" not in value["content"], "Binary source unsupported")
            content = value["content"].encode()
            require(
                sha(content) == value["sha256"]
                and hashlib.sha1(
                    b"blob " + str(len(content)).encode() + b"\0" + content
                ).hexdigest()
                == value["blob"],
                "Captured source blob mismatch: " + name,
            )

        require(
            pair["role"] != "context" or pair["before"] == pair["after"],
            "Changed source cannot be hidden as context",
        )


def source_images(source):
    validate_source_blobs(source)
    return full_commit_images(source)


def scoped_images(source, before, after, prior, *, allow_scoped):
    scope = source.get("edit_scope")
    if scope is None:
        require(not prior.get("omitted_documentation"), "Omitted edits unsupported")
        return before, after
    require(allow_scoped, "File-scoped edits require mixed v2 admission")
    mapping(scope, "edit scope")
    require(
        set(scope) == {"format", "coverage", "selected_paths", "omitted_paths"}
        and scope["format"] == "reviewed-file-edit-scope-v1"
        and scope["coverage"] == "complete_selected_file_edits",
        "Invalid edit scope",
    )
    selected, omitted = scope["selected_paths"], scope["omitted_paths"]
    require(
        isinstance(selected, list)
        and selected
        and all(isinstance(p, str) for p in selected)
        and selected == sorted(set(selected)),
        "Invalid selected paths",
    )
    require(isinstance(omitted, dict) and omitted, "Explicit nonempty omissions required")
    require(
        set(selected).isdisjoint(omitted) and set(selected) | set(omitted) == set(before),
        "Incomplete changed-file partition",
    )
    for name, declaration in omitted.items():
        pair = source["files"][name]
        mapping(declaration, "omitted documentation")
        require(
            set(declaration) == {"classification", "before_sha256", "after_sha256", "reason"}
            and declaration["classification"] == "documentation"
            and isinstance(declaration["reason"], str)
            and declaration["reason"].strip(),
            "Invalid documentation omission",
        )
        require(
            PurePosixPath(name).suffix == ".md"
            and pair["before"] is not None
            and pair["after"] is not None
            and pair["before"]["mode"] == pair["after"]["mode"],
            "Only existing unchanged-mode Markdown omissions supported",
        )
        require(
            declaration["before_sha256"] == sha(before[name])
            and declaration["after_sha256"] == sha(after[name]),
            "Omission source hash mismatch",
        )
    historical = prior.get("omitted_documentation")
    if historical:
        require(
            isinstance(historical, dict) and set(historical) == set(omitted),
            "Historical omission receipt conflicts with scope",
        )
        for name, declaration in historical.items():
            mapping(declaration, "historical omission", {"before_sha256", "after_sha256", "reason"})
            require(
                declaration["before_sha256"] == omitted[name]["before_sha256"]
                and declaration["after_sha256"] == omitted[name]["after_sha256"]
                and isinstance(declaration["reason"], str)
                and declaration["reason"].strip(),
                "Historical omission receipt conflicts with scope",
            )
    return ({p: before[p] for p in selected}, {p: after[p] for p in selected})


def source_edit(source, raw, prior, *, allow_scoped=False):
    full_before, full_after = source_images(source)
    mapping(raw, "raw", {"messages", "tools"})
    mapping(prior, "replay", {"files", "patch_sha256", "exact_scoped_replay"})
    from .tool_data import normalize_tool_row

    normalized = normalize_tool_row(raw)
    messages = normalized["messages"]
    calls = messages[-1].get("tool_calls")
    require(
        isinstance(calls, list)
        and len(calls) == 1
        and sum(len(m.get("tool_calls", [])) for m in messages) == 1,
        "Require one final edit action",
    )
    function = calls[0]["function"]
    args = function["arguments"]
    args = _json(args) if isinstance(args, str) else args
    require(function["name"] == "exec_command" and set(args) == {"cmd"}, "Unexpected edit tool")
    before, after = scoped_images(source, full_before, full_after, prior, allow_scoped=allow_scoped)
    parse_command(args["cmd"])
    try:
        current = replay(args["cmd"], before, after)
    except subprocess.SubprocessError as error:
        raise ValueError("Exact patch replay failed") from error
    require(
        prior.get("exact_scoped_replay") is True
        and current["patch_sha256"] == prior.get("patch_sha256")
        and current["files"] == prior.get("files"),
        "Replay receipt mismatch",
    )
    stable, payload = source_identity(full_before, full_after)
    selected_stable, selected_payload = source_identity(before, after)
    current.update(
        {
            "coverage": "complete_selected_file_edits"
            if source.get("edit_scope")
            else "complete_commit_text_edits",
            "full_commit_replay": not bool(source.get("edit_scope")),
            "selected_stable_patch_id": selected_stable,
            "selected_payload_sha256": selected_payload,
            "full_stable_patch_id": stable,
            "full_payload_sha256": payload,
        }
    )
    return current, stable, payload


def source_identity(before, after):
    payload = sha(
        canonical(
            {
                name: [sha(before[name]) if before[name] is not None else None, sha(after[name])]
                for name in sorted(before)
            }
        )
    )
    try:
        stable = patch_id(before, after)
    except subprocess.SubprocessError as error:
        raise ValueError("Normalized patch identity failed") from error
    return stable, payload


def prepare_reviewed_corpus(source, model, output, *, project=None):
    """Prepare every explicit row or fail; no truncation, silent exclusions or training."""
    from .data import load_tokenizer, tokenize_row, tokenizer_identity, validate_example
    from .sft_data import _normalize

    source = Path(source).expanduser().absolute()
    require(
        not source.is_symlink() and source.resolve() == source, "Canonical manifest path required"
    )
    require(source.is_file() and source.stat().st_size <= MAX_FILE, "Invalid manifest")
    captured = source.read_bytes()
    manifest = _json(captured)
    require(
        isinstance(manifest, dict)
        and manifest.get("format") in {FORMAT, MIXED_FORMAT}
        and "example_only" not in manifest,
        "Unsupported/example-only manifest",
    )
    expected = manifest.get("manifest_sha256")
    require(
        expected == sha(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})),
        "Manifest checksum mismatch",
    )
    mapping(
        manifest,
        "manifest",
        {"max_length", "entries", "policy", "tokenizer", "model_config_sha256", "frozen_splits"},
    )
    limit = manifest["max_length"]
    require(type(limit) is int and 3 <= limit <= MAX_CONTEXT_TOKENS, "Invalid context capacity")
    entries = manifest["entries"]
    require(
        isinstance(entries, list) and entries and all(isinstance(e, dict) for e in entries),
        "Empty/invalid entries",
    )
    for entry in entries:
        mapping(entry, "entry", {"id", "kind", "split", "inputs", "review"})
    ids = [e["id"] for e in entries]
    require(
        all(isinstance(i, str) and i.strip() for i in ids) and len(set(ids)) == len(ids),
        "Duplicate/invalid IDs",
    )
    mixed = manifest["format"] == MIXED_FORMAT
    require(
        all(
            e.get("kind") in ({"exact_edit", "diagnostic"} if mixed else {"exact_edit"})
            and e.get("split") in SPLITS
            for e in entries
        ),
        "Stage one accepts exact_edit only; diagnostic/mixed corpora are unsupported",
    )
    destination = Path(output).expanduser().absolute()
    require(not destination.exists() and not destination.is_symlink(), "Fresh output required")
    if project is not None:
        require(
            not Path(project).expanduser().exists() and not Path(project).expanduser().is_symlink(),
            "Fresh project required",
        )
    bundle = Capture(source.parent)
    policy = _json(bundle.read(manifest["policy"]))
    required_policy = {
        "protected_commits",
        "protected_patch_ids",
        "protected_lineages",
        "heldout_repositories",
        "lineage_aliases",
    }
    require(
        isinstance(policy, dict) and set(policy) == required_policy,
        "Invalid explicit protection policy",
    )
    require(
        all(
            isinstance(policy[k], list) and all(isinstance(v, str) for v in policy[k])
            for k in required_policy - {"lineage_aliases"}
        ),
        "Invalid protection lists",
    )
    aliases = policy["lineage_aliases"]
    require(
        isinstance(aliases, list)
        and all(
            isinstance(g, list) and len(g) > 1 and all(isinstance(v, str) and v for v in g)
            for g in aliases
        ),
        "Invalid lineage aliases",
    )
    model = Path(model).expanduser().resolve(strict=True)
    identity = tokenizer_identity(model)
    config_bytes = (model / "config.json").read_bytes()
    require(
        identity == manifest["tokenizer"] and sha(config_bytes) == manifest["model_config_sha256"],
        "Tokenizer/config binding mismatch",
    )
    config = mapping(_json(config_bytes), "model configuration")
    text_config = mapping(config.get("text_config", config), "text configuration", {"vocab_size"})
    vocab = text_config["vocab_size"]
    require(type(vocab) is int and vocab > 0, "Invalid model vocabulary size")
    tokenizer = load_tokenizer(model)
    frozen = manifest["frozen_splits"]
    require(
        isinstance(frozen, dict) and {"valid", "test"} <= set(frozen) <= set(SPLITS),
        "Require frozen valid/test files",
    )
    for binding in frozen.values():
        mapping(binding, "frozen split", {"raw", "prepared"})
        require(set(binding) == {"raw", "prepared"}, "Invalid frozen split roles")
    frozen_bytes = {
        split: {kind: bundle.read(binding) for kind, binding in bindings.items()}
        for split, bindings in frozen.items()
    }
    require(
        all(set(v) == {"raw", "prepared"} for v in frozen_bytes.values()),
        "Invalid frozen split binding",
    )
    rows, provenance, lengths, seen, groups = {s: [] for s in SPLITS}, [], [], set(), []
    reviews = {}
    overlap_declarations = manifest.get("multi_views", []) if mixed else []
    require(
        isinstance(overlap_declarations, list)
        and all(
            isinstance(g, list)
            and len(g) > 1
            and all(isinstance(i, str) and i in ids for i in g)
            and len(set(g)) == len(g)
            for g in overlap_declarations
        ),
        "Invalid explicit multi_views",
    )
    require(
        len({tuple(sorted(g)) for g in overlap_declarations}) == len(overlap_declarations),
        "Duplicate multi_views declarations",
    )
    seen_views, seen_prepared_views, view_keys = set(), set(), {}
    for entry in entries:
        bindings = entry["inputs"]
        require(
            isinstance(bindings, dict)
            and set(bindings)
            == (
                ROLES
                if entry["kind"] == "exact_edit"
                else {"source", "raw", "prepared", "evidence"}
            ),
            "Require four reviewed payloads",
        )
        values = {k: _json(bundle.read(v)) for k, v in bindings.items()}
        review_bytes = bundle.read(entry["review"])
        review = mapping(_json(review_bytes), "review", {"format", "status", "reviewer", "cases"})
        require(
            review.get("format")
            == ("exact-edit-review-v1" if entry["kind"] == "exact_edit" else "diagnostic-review-v1")
            and review.get("status") == "approved"
            and isinstance(review.get("reviewer"), str)
            and review["reviewer"].strip(),
            "Unapproved review assertion",
        )
        cases = review["cases"]
        require(isinstance(cases, list), "Review cases must be an array")
        for case in cases:
            mapping(case, "review case", {"id", "inputs"})
            require(isinstance(case["id"], str) and case["id"], "Invalid review case ID")
            mapping(case["inputs"], "review case bindings")
        require(
            isinstance(cases, list) and len({c["id"] for c in cases}) == len(cases),
            "Ambiguous review cases",
        )
        matched = [c for c in cases if c["id"] == entry["id"]]
        require(
            len(matched) == 1 and matched[0]["inputs"] == bindings,
            "Review case/role binding mismatch",
        )
        reviews.setdefault(entry["review"]["path"], [review, set()])[1].add(entry["id"])
        source_record, raw, prepared = values["source"], values["raw"], values["prepared"]
        if entry["kind"] == "exact_edit":
            current, stable, payload = source_edit(
                source_record, raw, values["replay"], allow_scoped=mixed
            )
        else:
            from .reviewed_diagnostics import admit_diagnostic

            current, stable, payload = admit_diagnostic(
                source_record, raw, values["evidence"], bundle
            )
        from .upstream_data import _repository_identity

        repository = _repository_identity(source_record["repository"])
        mapping(source_record["head"], "source head", {"commit"})
        commit = source_record["head"]["commit"]
        require(isinstance(commit, str), "Invalid source commit")
        commits = source_record["commits"]
        require(
            isinstance(commits, list)
            and commit in commits
            and all(isinstance(c, str) and re.fullmatch(r"[0-9a-f]{40}", c) for c in commits),
            "Invalid source commit identities",
        )
        lineage = source_record.get("lineage_keys", [])
        require(
            isinstance(lineage, list) and all(isinstance(k, str) and k for k in lineage),
            "Invalid lineage IDs",
        )
        keys = {"patch:" + stable, "payload:" + payload, "repo-commit:" + repository + ":" + commit}
        if entry["kind"] == "exact_edit":
            keys |= {
                "patch:" + current["selected_stable_patch_id"],
                "payload:" + current["selected_payload_sha256"],
            }
        require(
            mixed or not seen.intersection(keys),
            "Duplicate commit/patch/payload; select explicitly",
        )
        view_keys[entry["id"]] = keys
        view = sha(canonical({"raw": raw, "prepared": prepared}))
        require(view not in seen_views, "Duplicate training view")
        seen_views.add(view)
        seen.update(keys)
        connected = (
            set(lineage)
            | {"commit:" + c for c in commits}
            | {k for k in keys if k.startswith(("patch:", "payload:"))}
        )
        groups.append((entry["split"], connected, entry["id"]))
        normalized, _, _ = _normalize(raw, None, None)
        actual = tokenize_row(normalized, tokenizer)
        validate_example(actual, seq_length=limit - 1, vocab_size=vocab, truncate=False)
        require(
            actual == prepared,
            "Reviewed full token IDs/mask differ from normal product preparation",
        )
        require(len(actual["input_ids"]) <= limit, "Complete item exceeds capacity")
        prepared_view = sha(
            canonical({"input_ids": actual["input_ids"], "loss_start": actual["loss_start"]})
        )
        require(
            not mixed or prepared_view not in seen_prepared_views,
            "Duplicate prepared training view",
        )
        seen_prepared_views.add(prepared_view)
        rows[entry["split"]].append((raw, actual, entry["id"]))
        provenance.append(
            {
                "id": entry["id"],
                "kind": entry["kind"],
                "split": entry["split"],
                "repository": repository,
                "commits": commits,
                "stable_patch_id": stable,
                "source": source_record,
                "review": review,
                "review_binding": entry["review"],
                ("replay" if entry["kind"] == "exact_edit" else "diagnostic_admission"): current,
                "inputs": bindings,
                "reviewed_payloads": values,
                "protected": source_record.get("protected", False),
            }
        )
        lengths.append(
            {
                "id": entry["id"],
                "total_tokens": len(actual["input_ids"]),
                "prompt_tokens": actual["loss_start"],
                "target_tokens": len(actual["input_ids"]) - actual["loss_start"],
                "truncated": False,
            }
        )
    for review, used in reviews.values():
        require(
            {c["id"] for c in review["cases"]} == used, "Review contains unselected/extra cases"
        )
    # Explicit exact-source overlap groups; alias connections still protect splits below.
    exposures = []
    if mixed:
        pending = set(ids)
        while pending:
            first = min(pending)
            pending.remove(first)
            members = {first}
            connected_keys = set().union(*(view_keys[i] for i in members))
            while True:
                adjacent = {i for i in pending if view_keys[i] & connected_keys}
                if not adjacent:
                    break
                members |= adjacent
                pending -= adjacent
                connected_keys |= set().union(*(view_keys[i] for i in adjacent))
            if len(members) > 1:
                require(
                    any(set(g) == members for g in overlap_declarations),
                    "Same-source views require explicit multi_views",
                )
                exposures.append(sorted(members))
        exposures.sort()
        require(
            {tuple(sorted(g)) for g in overlap_declarations} == {tuple(g) for g in exposures},
            "Unmatched multi_views declaration",
        )
    # Transitive finite alias/commit closure; never infer repository equivalence.
    components = [set(k) for _, k, _ in groups] + [set(g) for g in aliases]
    merged = []
    while components:
        group = components.pop()
        while True:
            overlap = [g for g in components if group & g]
            if not overlap:
                break
            for g in overlap:
                group |= g
                components.remove(g)
        merged.append(group)
    protected = (
        set(policy["protected_lineages"])
        | {"commit:" + v for v in policy["protected_commits"]}
        | {"patch:" + v for v in policy["protected_patch_ids"]}
    )
    for group in merged:
        members = [(s, i) for s, keys, i in groups if keys & group]
        require(len({s for s, _ in members}) <= 1, "Declared lineage crosses explicit splits")
        for split, rid in members:
            record = next(v for v in provenance if v["id"] == rid)
            require(
                split != "train"
                or not (
                    group & protected
                    or record["protected"]
                    or record["repository"] in policy["heldout_repositories"]
                ),
                "Protected source in train",
            )
    require(rows["train"], "No training rows")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".reviewed-corpus-", dir=destination.parent))
    try:
        for split in SPLITS:
            raw_values, token_values = [r[0] for r in rows[split]], [r[1] for r in rows[split]]
            if split in frozen_bytes:
                raw_bytes, token_bytes = frozen_bytes[split]["raw"], frozen_bytes[split]["prepared"]
                require(
                    lines(raw_bytes) == raw_values and lines(token_bytes) == token_values,
                    "Frozen split order/content mismatch",
                )
            else:
                raw_bytes = b"".join(
                    json.dumps(r, ensure_ascii=False, allow_nan=False).encode() + b"\n"
                    for r in raw_values
                )
                token_bytes = b"".join(canonical(r) + b"\n" for r in token_values)
            for raw, expected_tokens in zip(lines(raw_bytes), token_values, strict=True):
                normalized, _, _ = _normalize(raw, None, None)
                require(
                    tokenize_row(normalized, tokenizer) == expected_tokens,
                    "Actual serialized tokenization mismatch",
                )
            (stage / f"{split}.jsonl").write_bytes(raw_bytes)
            (stage / f"{split}.prepared.jsonl").write_bytes(token_bytes)
        for name, values in [("provenance.jsonl", provenance), ("length-audit.jsonl", lengths)]:
            (stage / name).write_bytes(b"".join(canonical(v) + b"\n" for v in values))
        (stage / "admission.json").write_bytes(
            canonical(
                {
                    "all_rows_admitted": True,
                    "review_assertions_authenticated": False,
                    "finite_policy": policy,
                }
            )
            + b"\n"
        )
        (stage / "overlap.json").write_bytes(
            canonical(
                {
                    "scope": "Exact patch/payload/repository-commit and explicitly declared finite lineage only; not exhaustive semantic clone detection",
                    "cross_split_overlap": False,
                    "explicit_same_source_views": exposures,
                }
            )
            + b"\n"
        )
        if mixed:
            # Flat content-addressed copies survive project provenance publication.
            captured_index = {}
            for name, payload in bundle.files.items():
                checksum = sha(payload)
                filename = "captured-" + checksum + ".bin"
                (stage / filename).write_bytes(payload)
                captured_index[name] = {"file": filename, "sha256": checksum}
            (stage / "captured-inputs.json").write_bytes(canonical(captured_index) + b"\n")
        bundle.check()
        require(
            source.read_bytes() == captured
            and tokenizer_identity(model) == identity
            and (model / "config.json").read_bytes() == config_bytes,
            "Preparation input drift",
        )
        report = {
            "format": manifest["format"],
            "counts": {s: len(rows[s]) for s in SPLITS},
            "counts_by_kind": {
                kind: sum(e["kind"] == kind for e in entries)
                for kind in ("exact_edit", "diagnostic")
            },
            "explicit_same_source_views": exposures,
            "selected_case_ids": ids,
            "max_length": max(v["total_tokens"] for v in lengths),
            "context_ceiling": limit,
            "tokenizer": identity,
            "model_config_sha256": sha(config_bytes),
            "truncated": 0,
            "ready_for_sft": True,
            "quality_certified": False,
            "verification": "captured source bytes, supplied review assertions, exact edit replay or diagnostic evidence consistency; no semantic completeness/truth/leakage, fresh Git tree/compile/numerical/performance certification",
            "source_files_sha256": {n: sha(v) for n, v in bundle.files.items()},
            "source_manifest_sha256": sha(captured),
            "implementation_sha256": sha(Path(__file__).read_bytes()),
            "diagnostic_implementation_sha256": sha(
                Path(__file__).with_name("reviewed_diagnostics.py").read_bytes()
            )
            if mixed
            else None,
            "artifacts_sha256": {x.name: sha(x.read_bytes()) for x in stage.iterdir()},
        }
        (stage / "ktune-import.json").write_bytes(canonical(report) + b"\n")
        require(
            not destination.exists() and not destination.is_symlink(),
            "Output appeared during preparation",
        )
        publish_new(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    result = {
        **report,
        "output": str(destination),
        "manifest_sha256": sha((destination / "ktune-import.json").read_bytes()),
    }
    if project is not None:
        from .sft import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(output_dir=str(project), max_length=report["max_length"]),
            train_dataset=destination,
        )
        trainer.prepare(provenance_required=True)
        for split in SPLITS:
            require(
                lines((trainer.root / "data" / f"{split}.jsonl").read_bytes())
                == [r[1] for r in rows[split]],
                "Project prepared tokens/order differ; provenance remains pending",
            )
        trainer.bind_provenance(destination, result)
        result["project"] = str(trainer.root)
    return result
