"""Bounded evidence consistency for reviewed diagnostics; never an execution attestation."""

from urllib.parse import urlsplit

from .agent_edits import _path
from .reviewed_corpus import (
    canonical,
    mapping,
    require,
    sha,
    source_identity,
    validate_source_blobs,
)

KINDS = {"before_source", "after_source", "author_report", "local_cpu", "local_gpu"}
CLASSES = {"source_observation", "inferred", "author_reported", "local_cpu", "local_gpu"}


def validate_execution_portability(execution, capture):
    """Bind a single path-only receipt transformation without opening historical paths."""
    if "portability" not in execution:
        return
    from .evaluation_audit import _json

    portable = mapping(execution["portability"], "execution portability")
    require(
        set(portable) == {"original_execution", "path_mapping", "transformation"},
        "Invalid execution portability fields",
    )
    require(
        isinstance(portable["transformation"], str) and portable["transformation"].strip(),
        "Missing portability transformation",
    )
    original = mapping(_json(capture.read(portable["original_execution"])), "original execution")
    require("portability" not in original, "Nested execution portability is unsupported")
    require(
        canonical({k: v for k, v in original.items() if k not in {"inputs", "outputs"}})
        == canonical(
            {k: v for k, v in execution.items() if k not in {"inputs", "outputs", "portability"}}
        ),
        "Execution portability changed non-path metadata",
    )
    require(isinstance(portable["path_mapping"], list), "Invalid execution path mapping")
    paths = {}
    for entry in portable["path_mapping"]:
        mapping(entry, "execution path entry")
        require(set(entry) == {"original_path", "captured"}, "Invalid execution path entry")
        name = entry["original_path"]
        require(
            isinstance(name, str) and name and name not in paths,
            "Invalid or duplicate original execution path",
        )
        capture.read(entry["captured"])
        paths[name] = entry["captured"]
    used = set()
    for role in ("inputs", "outputs"):
        bindings = original.get(role)
        require(isinstance(bindings, list) and bindings, "Missing original execution bindings")
        remapped = []
        for binding in bindings:
            mapping(binding, "original execution binding")
            require(set(binding) == {"path", "sha256"}, "Invalid original execution binding")
            name = binding["path"]
            require(isinstance(name, str) and name in paths, "Unmapped original execution path")
            require(
                binding["sha256"] == paths[name]["sha256"],
                "Execution portability changed file hash",
            )
            used.add(name)
            remapped.append(paths[name])
        require(
            canonical(remapped) == canonical(execution[role]),
            "Execution portability changed role or binding order",
        )
    require(used == set(paths), "Unused execution path mapping")


def span(text, record, label):
    """Ranges are half-open UTF-8 byte offsets; hashes bind exact text, not normalized prose."""
    mapping(record, label, {"start", "end", "sha256"})
    start, end = record["start"], record["end"]
    require(
        type(start) is int and type(end) is int and 0 <= start < end <= len(text),
        label + ": invalid range",
    )
    selected = text[start:end]
    try:
        selected.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(label + ": range cuts a UTF-8 character") from error
    require(sha(selected) == record["sha256"], label + ": range hash mismatch")
    return start, end


def diagnostic_source_images(source):
    """Inspect text transitions without executing or pretending to replay their modes."""
    validate_source_blobs(source)
    before, after, transitions, modes = {}, {}, {}, []
    for name, pair in source["files"].items():
        _path(name)
        for side in ("before", "after"):
            blob = pair[side]
            if blob is not None:
                require(
                    isinstance(blob["mode"], str) and blob["mode"] in {"100644", "100755"},
                    "Diagnostic source requires regular text modes",
                )
                require("\0" not in blob["content"], "Binary diagnostic source unsupported")
        transitions[name] = {
            side: (
                {"mode": pair[side]["mode"], "sha256": pair[side]["sha256"]}
                if pair[side] is not None
                else None
            )
            for side in ("before", "after")
        }
        if pair["role"] != "changed":
            continue
        require(pair["after"] is not None, "Deleted diagnostic files unsupported in this stage")
        old = pair["before"]["content"].encode() if pair["before"] is not None else None
        new = pair["after"]["content"].encode()
        require(old is not None or new, "Empty-file diagnostic additions unsupported in this stage")
        old_mode = pair["before"]["mode"] if pair["before"] is not None else None
        new_mode = pair["after"]["mode"]
        if old_mode != new_mode:
            modes.append({"path": name, "before": old_mode, "after": new_mode})
        if old != new:
            before[name], after[name] = old, new
    require(before, "Pure mode-only diagnostic source unsupported: no text-patch identity")
    return (
        before,
        after,
        {
            "source_transition_sha256": sha(canonical(transitions)),
            "source_transitions": transitions,
            "mode_transitions": modes,
            "text_patch_identity_excludes_modes": True,
        },
    )


def admit_diagnostic(source, raw, evidence, capture):
    before, after, source_metadata = diagnostic_source_images(source)
    mapping(raw, "diagnostic raw", {"messages"})
    require(set(raw) == {"messages"}, "Diagnostic raw must contain only messages")
    messages = raw["messages"]
    require(
        isinstance(messages, list) and len(messages) in {2, 3},
        "Diagnostic requires optional system, one user and one assistant message",
    )
    roles = ("system", "user", "assistant") if len(messages) == 3 else ("user", "assistant")
    for message, role in zip(messages, roles, strict=True):
        mapping(message, "diagnostic message", {"role", "content"})
        require(
            set(message) == {"role", "content"}
            and message["role"] == role
            and isinstance(message["content"], str)
            and message["content"].strip(),
            "Invalid diagnostic message",
        )
    mapping(
        evidence,
        "diagnostic evidence",
        {"format", "task_type", "prompt_mode", "references", "prompt_spans", "claims"},
    )
    require(evidence["format"] == "diagnostic-evidence-v1", "Unsupported diagnostic evidence")
    require(
        isinstance(evidence["task_type"], str)
        and evidence["task_type"]
        in {
            "diagnose_bottleneck",
            "explain_tradeoff",
            "reject_unsafe_change",
            "choose_measurement",
        },
        "Unknown diagnostic task type",
    )
    require(
        isinstance(evidence["prompt_mode"], str)
        and evidence["prompt_mode"] in {"before_only", "supplied_observations"},
        "Unknown prompt information boundary",
    )
    if len(messages) == 3:
        system_mode = evidence.get("system_mode")
        require(
            isinstance(system_mode, str)
            and system_mode in {"task_instruction_only", "supplied_observations"},
            "Explicit system information boundary required",
        )
        require(
            system_mode != "supplied_observations"
            or evidence["prompt_mode"] == "supplied_observations",
            "System observations cannot be a before-only task",
        )
        require(
            system_mode != "task_instruction_only" or "system_spans" not in evidence,
            "Task-only system must not declare factual evidence spans",
        )
    else:
        require(
            "system_mode" not in evidence and "system_spans" not in evidence,
            "System evidence without a system message",
        )
    references = evidence["references"]
    require(isinstance(references, list) and references, "Diagnostic evidence references required")
    by_id = {}
    for reference in references:
        mapping(reference, "evidence reference", {"id", "kind", "file", "start", "end", "sha256"})
        rid, kind = reference["id"], reference["kind"]
        require(
            isinstance(rid, str) and rid and rid not in by_id,
            "Duplicate/invalid evidence reference ID",
        )
        require(isinstance(kind, str) and kind in KINDS, "Unknown evidence kind")
        document = capture.read(reference["file"])
        span(document, reference, "evidence reference")
        if kind in {"before_source", "after_source"}:
            path = reference.get("source_path")
            require(
                isinstance(path, str) and path in source["files"], "Unknown evidence source path"
            )
            blob = source["files"][path]["before" if kind == "before_source" else "after"]
            require(
                blob is not None and document == blob["content"].encode(),
                "Evidence file differs from captured source side",
            )
        elif kind == "author_report":
            origin = mapping(reference.get("origin"), "author report origin", {"url", "identity"})
            require(
                isinstance(origin["url"], str)
                and isinstance(origin["identity"], str)
                and origin["identity"].strip(),
                "Missing primary author report identity",
            )
            url = urlsplit(origin["url"])
            require(
                url.scheme in {"https", "http"}
                and url.hostname
                and not url.username
                and not url.password,
                "Invalid primary author report URL",
            )
        elif kind in {"local_cpu", "local_gpu"}:
            # This binds the supplied execution assertion and all files it names. It does
            # not rerun the command or certify who ran it, the hardware, or its truth.
            from .evaluation_audit import _json

            execution = mapping(
                _json(capture.read(reference.get("execution"))),
                "execution assertion",
                {"executed", "argv", "inputs", "outputs", "environment", "kind", "backend"},
            )
            require(
                execution["executed"] is True
                and isinstance(execution["argv"], list)
                and execution["argv"]
                and all(isinstance(v, str) and v for v in execution["argv"]),
                "Invalid execution assertion",
            )
            require(
                execution["kind"] == kind
                and isinstance(execution["backend"], str)
                and execution["backend"]
                in ({"cpu"} if kind == "local_cpu" else {"metal", "cuda", "rocm", "other_gpu"}),
                "Execution kind/backend disagrees with evidence class",
            )
            require(
                isinstance(execution["environment"], dict) and execution["environment"],
                "Missing execution environment assertion",
            )
            for role in ("inputs", "outputs"):
                require(
                    isinstance(execution[role], list) and execution[role],
                    "Missing execution file bindings",
                )
                for binding in execution[role]:
                    capture.read(binding)
            validate_execution_portability(execution, capture)
            require(
                reference["file"] in execution["outputs"],
                "Local evidence is not a bound execution output",
            )
        by_id[rid] = reference
    used = set()
    texts = [
        ("prompt_spans", messages[-2]["content"].encode()),
        ("claims", messages[-1]["content"].encode()),
    ]
    if len(messages) == 3 and evidence["system_mode"] == "supplied_observations":
        texts.append(("system_spans", messages[0]["content"].encode()))
    for name, text in texts:
        records = evidence.get(name)
        require(isinstance(records, list) and records, "Missing bound " + name)
        intervals = []
        for record in records:
            intervals.append(span(text, record, name))
            refs = record.get("refs")
            require(
                isinstance(refs, list)
                and refs
                and all(isinstance(r, str) and r in by_id for r in refs)
                and len(set(refs)) == len(refs),
                "Unknown/unbound evidence reference",
            )
            used.update(refs)
            kinds = {by_id[r]["kind"] for r in refs}
            if name != "claims":
                require(
                    evidence["prompt_mode"] != "before_only" or kinds <= {"before_source"},
                    "Before-only prompt references post-change/observation evidence",
                )
            else:
                limits = record.get("limits")
                require(
                    isinstance(limits, str) and limits.strip(),
                    "Claim needs explicit limits/caveats",
                )
                classification = record.get("class")
                require(
                    isinstance(classification, str) and classification in CLASSES,
                    "Unknown claim evidence class",
                )
                eligible = {
                    "source_observation": {"before_source", "after_source"},
                    "author_reported": {"author_report"},
                    "local_cpu": {"local_cpu"},
                    "local_gpu": {"local_gpu"},
                }.get(classification)
                require(
                    eligible is None or kinds <= eligible,
                    "Claim class disagrees with evidence kind",
                )
        ordered = sorted(intervals)
        require(
            all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), "Overlapping bound text spans"
        )
        if name == "claims":
            cursor = 0
            for start, end in ordered:
                require(not text[cursor:start].strip(), "Unbound assistant target text")
                cursor = end
            require(not text[cursor:].strip(), "Unbound assistant target text")
    require(used == set(by_id), "Unused evidence reference")
    stable, payload = source_identity(before, after)
    return (
        {
            **source_metadata,
            "evidence_consistency": True,
            "replay_performed": False,
            "semantic_truth_certified": False,
            "semantic_completeness_certified": False,
            "prompt_leakage_absence_certified": False,
            "execution_assertions_authenticated": False,
            "prompt_mode": evidence["prompt_mode"],
            "references": references,
        },
        stable,
        payload,
    )
