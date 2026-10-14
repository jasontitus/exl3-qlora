"""Opt-in, CPU-only durable complete-row evaluation receipts.

This is a local consistency ledger, not model attestation or workflow resume.
Callers must verify actual model/runtime continuity in ``verify_context``. No
MLX import, model construction, scorer, or automatic historical-report adoption
occurs here. Workflow/SFT hooks and actual-model qualification are separate.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import uuid
from pathlib import Path

IDENTITY_FORMAT = "exl3-evaluation-identity-v2"
ROW_FORMAT = "exl3-evaluation-row-v2"
AGGREGATE_FORMAT = "exl3-evaluation-aggregate-v2"
METRIC_ALGORITHM = "ordered-token-binary64-math-fsum-v2"
DIGEST_DOMAIN = "exl3-ordered-row-logit-digests-v2"
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_ROW = re.compile(r"row-([0-9]{8})\.json\Z")
_EVENT = re.compile(r"event-([0-9]{8})\.json\Z")
_TEMP = re.compile(r"\.tmp-[a-f0-9]{32}\.json\Z")
_MAX_BYTES = 64 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError("Evaluation progress: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _json(data):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    return json.loads(
        data,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def _hash(value, name):
    require(isinstance(value, str) and bool(_HEX.fullmatch(value)), name + " must be SHA256")


def _text(value, name):
    require(
        isinstance(value, str) and bool(value.strip()) and "\x00" not in value,
        name + " must be text",
    )


def _fields(value, names, name):
    require(isinstance(value, dict) and set(value) == set(names), name + " fields differ")


def _positive(value):
    return type(value) is int and value > 0


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _time(value):
    require(isinstance(value, str), "timestamp must be text")
    try:
        t = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Evaluation progress: invalid timestamp") from error
    require(t.tzinfo is not None, "timestamp must include timezone")
    return t


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _origin(value, host):
    _fields(value, ("run_id", "process_id", "argv", "host_id"), "producer/assembler")
    _text(value["run_id"], "run_id")
    require(_positive(value["process_id"]), "process ID must be positive")
    require(isinstance(value["argv"], list) and value["argv"], "argv must be nonempty")
    for arg in value["argv"]:
        require(isinstance(arg, str) and "\x00" not in arg, "invalid argv")
    require(value["host_id"] == host, "producer/assembler host differs")


def _rows(rows, vocab, eos, capacity):
    require(
        isinstance(rows, list) and 0 < len(rows) < 100_000_000,
        "nonempty bounded selection required",
    )
    for row in rows:
        _fields(row, ("input_ids", "loss_start"), "prepared row")
        ids, start = row["input_ids"], row["loss_start"]
        require(
            isinstance(ids, list) and 2 <= len(ids) <= capacity + 1, "row context would truncate"
        )
        require(all(type(t) is int and 0 <= t < vocab for t in ids), "input outside vocabulary")
        require(type(start) is int and 1 <= start < len(ids), "invalid or empty target mask")
        require(ids[-1] == eos, "row lacks declared EOS")


def make_identity(
    rows, *, project_sha256, base, runtime, arithmetic, tokenizer, condition, seq_length
):
    """Build a conservative same-project/host identity from explicit caller proofs.

    Hash references are assertions whose actual backing evidence must be checked
    by verify_context; they are not independently verified by this module.
    Cache semantics are keyed until any equivalence is separately qualified.
    """
    _hash(project_sha256, "project")
    _fields(
        base,
        (
            "payload_manifest_sha256",
            "verification_receipt_sha256",
            "verification_method",
            "continuity_sha256",
            "format",
            "metadata_sha256",
            "layer_count",
        ),
        "base",
    )
    for name in ("payload_manifest_sha256", "verification_receipt_sha256", "metadata_sha256"):
        _hash(base[name], name)
    _text(base["format"], "base format")
    require(_positive(base["layer_count"]), "model layer count required")
    require(
        base["verification_method"]
        in ("fresh_full_sha256", "historical_full_sha256_with_continuity"),
        "unknown verification method",
    )
    if base["verification_method"] == "historical_full_sha256_with_continuity":
        _hash(base["continuity_sha256"], "continuity")
    else:
        require(
            base["continuity_sha256"] is None,
            "fresh verification must not masquerade as continuity reuse",
        )
    _fields(
        runtime,
        (
            "source_manifest_sha256",
            "artifact_manifest_sha256",
            "dependency_lock_sha256",
            "python",
            "versions",
            "host_id",
            "os",
            "device",
            "backend",
        ),
        "runtime",
    )
    for key in ("source_manifest_sha256", "artifact_manifest_sha256", "dependency_lock_sha256"):
        _hash(runtime[key], key)
    for key in ("python", "host_id", "os", "device", "backend"):
        _text(runtime[key], key)
    require(
        isinstance(runtime["versions"], dict) and runtime["versions"],
        "dependency versions required",
    )
    for key, val in runtime["versions"].items():
        _text(key, "dependency name")
        _text(val, "dependency version")
    _fields(
        arithmetic,
        (
            "activation_mode",
            "tf32",
            "eval_mode",
            "preserve_mode",
            "activation_dtype",
            "weight_format",
            "engine_semantics",
            "exl3_chunk_size",
            "head_chunk_size",
            "cache_semantics_sha256",
            "backend_options",
        ),
        "arithmetic",
    )
    require(arithmetic["activation_mode"] in ("checkpoint", "bf16"), "invalid activation mode")
    require(arithmetic["tf32"] in ("unset", "0", "1"), "TF32 unset differs from explicit0")
    require(
        arithmetic["eval_mode"] is True and arithmetic["preserve_mode"] is False,
        "requires evaluation mode",
    )
    for key in ("activation_dtype", "weight_format", "engine_semantics"):
        _text(arithmetic[key], key)
    require(
        _positive(arithmetic["exl3_chunk_size"]) and _positive(arithmetic["head_chunk_size"]),
        "invalid chunk geometry",
    )
    _hash(arithmetic["cache_semantics_sha256"], "cache semantics")
    require(isinstance(arithmetic["backend_options"], dict), "backend options required")
    _fields(
        tokenizer,
        (
            "artifact_manifest_sha256",
            "template_sha256",
            "tokenizer_size",
            "model_vocab_size",
            "eos_token_id",
        ),
        "tokenizer",
    )
    for key in ("artifact_manifest_sha256", "template_sha256"):
        _hash(tokenizer[key], key)
    require(
        _positive(tokenizer["tokenizer_size"]) and _positive(tokenizer["model_vocab_size"]),
        "invalid vocab dimensions",
    )
    vocab = min(tokenizer["tokenizer_size"], tokenizer["model_vocab_size"])
    require(
        type(tokenizer["eos_token_id"]) is int and 0 <= tokenizer["eos_token_id"] < vocab,
        "invalid EOS",
    )
    require(type(seq_length) is int and 1 <= seq_length <= 32767, "unsupported sequence capacity")
    _rows(rows, vocab, tokenizer["eos_token_id"], seq_length)
    _fields(
        condition,
        (
            "kind",
            "rank",
            "alpha",
            "adapter_layers",
            "expert_adapter_layers",
            "a_initialization_sha256",
            "zero_b_verification_sha256",
            "adapter_config_sha256",
            "adapter_payload_sha256",
            "checkpoint_step",
        ),
        "condition",
    )
    require(
        condition["kind"] in ("zero_b", "adapter") and _positive(condition["rank"]),
        "invalid adapter graph",
    )
    require(_finite(condition["alpha"]) and condition["alpha"] > 0, "invalid adapter scale")
    for key in ("adapter_layers", "expert_adapter_layers"):
        layers = condition[key]
        require(
            isinstance(layers, list)
            and all(type(i) is int and 0 <= i < base["layer_count"] for i in layers)
            and layers == sorted(set(layers)),
            "invalid adapter placement",
        )
    require(
        condition["adapter_layers"] or condition["expert_adapter_layers"],
        "zero-B wrapper placement must be explicit",
    )
    if condition["kind"] == "zero_b":
        for key in ("a_initialization_sha256", "zero_b_verification_sha256"):
            _hash(condition[key], key)
        require(
            all(
                condition[k] is None
                for k in ("adapter_config_sha256", "adapter_payload_sha256", "checkpoint_step")
            ),
            "baseline cannot load an adapter",
        )
    else:
        for key in ("adapter_config_sha256", "adapter_payload_sha256"):
            _hash(condition[key], key)
        require(
            type(condition["checkpoint_step"]) is int and condition["checkpoint_step"] >= 0,
            "invalid checkpoint step",
        )
        require(
            condition["a_initialization_sha256"] is None
            and condition["zero_b_verification_sha256"] is None,
            "adapter is not zero-B construction",
        )
    result = {
        "format": IDENTITY_FORMAT,
        "project_sha256": project_sha256,
        "base": base,
        "runtime": runtime,
        "arithmetic": arithmetic,
        "tokenizer": tokenizer,
        "condition": condition,
        "selection": {
            "selected_examples_sha256": digest(rows),
            "row_sha256": [digest(r) for r in rows],
            "examples": len(rows),
            "seq_length": seq_length,
        },
        "metric_algorithm": METRIC_ALGORITHM,
        "next_token_shift": 1,
        "argmax_tie_rule": "lowest_index",
        "token_nll_storage": "finite_nonnegative_binary64_json",
        "evidence_scope": "caller-verified identity assertions; local receipt consistency, not process attestation",
    }
    return _json(canonical(result))  # Detached copy; reject non-JSON/nonfinite options.


def _score(score, row, identity):
    _fields(score, ("target_ids", "predicted_ids", "token_nll", "logits"), "row score")
    targets = row["input_ids"][row["loss_start"] :]
    require(
        isinstance(score["target_ids"], list)
        and all(type(t) is int for t in score["target_ids"])
        and score["target_ids"] == targets,
        "target IDs/mask differ",
    )
    predictions, nll = score["predicted_ids"], score["token_nll"]
    vocab = identity["tokenizer"]["model_vocab_size"]
    require(
        isinstance(predictions, list)
        and len(predictions) == len(targets)
        and all(type(t) is int and 0 <= t < vocab for t in predictions),
        "prediction IDs differ",
    )
    require(
        isinstance(nll, list)
        and len(nll) == len(targets)
        and all(_finite(v) and v >= 0 for v in nll),
        "invalid/incomplete NLL",
    )
    loss_sum = math.fsum(nll)
    require(math.isfinite(loss_sum), "loss sum overflow")
    logits = score["logits"]
    _fields(logits, ("sha256", "scope", "shape", "dtype", "byte_order"), "row logits")
    _hash(logits["sha256"], "row logits")
    require(
        logits["scope"] in ("supervised_logits", "all_next_token_logits"),
        "unknown logits digest scope",
    )
    count = len(targets) if logits["scope"] == "supervised_logits" else len(row["input_ids"]) - 1
    require(
        logits["shape"] == [count, vocab] and all(type(x) is int for x in logits["shape"]),
        "logits shape differs",
    )
    require(
        logits["dtype"] in ("float32", "bfloat16", "float16") and logits["byte_order"] == "little",
        "logits dtype/byte order invalid",
    )
    return loss_sum


def _read_file(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(f.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= _MAX_BYTES,
            "invalid/oversized ledger file",
        )
        data = f.read(_MAX_BYTES + 1)
    require(len(data) <= _MAX_BYTES, "oversized ledger file")
    return data


def _publish(path, value, before_link=None):
    require(
        path.parent.resolve() == path.parent and path.parent.is_dir(),
        "noncanonical publication directory",
    )
    temp = path.parent / (".tmp-" + uuid.uuid4().hex + ".json")
    try:
        with temp.open("xb") as f:
            f.write(canonical(value))
            f.flush()
            os.fsync(f.fileno())
        if before_link is not None:
            before_link()
        os.link(temp, path)  # Exclusive atomic publication; never replace a complete file.
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


class EvaluationProgress:
    """Hold one stable-inode exclusive lock for scan, scoring and publication.

    ``resume=True`` is explicit opt-in; it is not hardware qualification. The
    zero-argument verify_context callback must verify actual frozen inputs and
    return the exact semantic identity digest. This module checks that assertion
    at entry and before every durable publication. Caller lock order is project
    first, progress second. Never unlink .writer.lock.
    """

    def __init__(self, root, identity, rows, *, verify_context, resume=False):
        self.root = Path(root).absolute()
        require(callable(verify_context), "actual context guard callback required")
        require(type(resume) is bool, "resume must be boolean")
        _fields(
            identity,
            (
                "format",
                "project_sha256",
                "base",
                "runtime",
                "arithmetic",
                "tokenizer",
                "condition",
                "selection",
                "metric_algorithm",
                "next_token_shift",
                "argmax_tie_rule",
                "token_nll_storage",
                "evidence_scope",
            ),
            "identity",
        )
        _fields(
            identity["selection"],
            ("selected_examples_sha256", "row_sha256", "examples", "seq_length"),
            "selection",
        )
        # Rebuild rather than trusting a caller-crafted identity object.
        rebuilt = make_identity(
            rows,
            project_sha256=identity["project_sha256"],
            base=identity["base"],
            runtime=identity["runtime"],
            arithmetic=identity["arithmetic"],
            tokenizer=identity["tokenizer"],
            condition=identity["condition"],
            seq_length=identity["selection"]["seq_length"],
        )
        require(canonical(rebuilt) == canonical(identity), "identity or ordered selection differs")
        self.identity, self.rows = rebuilt, _json(canonical(rows))
        self.identity_sha256 = digest(rebuilt)
        self.verify_context, self.resume, self.fd = verify_context, resume, None
        self.entered = False
        self.captured, self.receipts, self.events = {}, {}, []
        self.new_indices, self.adopted = set(), set()
        self.attempts = {}

    def __enter__(self):
        require(not self.entered, "create a new ledger object for each invocation")
        self.entered = True
        require(self.root.resolve() == self.root, "progress path contains symlink/alias")
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".writer.lock"
        self.fd = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            require(stat.S_ISREG(os.fstat(self.fd).st_mode), "writer lock is not a regular file")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_identity = (os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
            self._context()
            files = list(self.root.iterdir())
            for p in files:
                require(not p.is_symlink() and p.is_file(), "unexpected directory or symlink")
                require(
                    p.name == ".writer.lock"
                    or p.name == "identity.json"
                    or _ROW.fullmatch(p.name)
                    or _EVENT.fullmatch(p.name)
                    or _TEMP.fullmatch(p.name),
                    "unexpected ledger entry: " + p.name,
                )
            manifest = self.root / "identity.json"
            if manifest.exists():
                require(self.resume, "existing progress requires explicit resume=True")
                require(
                    self._capture(manifest) == self.identity,
                    "semantic identity differs; refuse reuse",
                )
            else:
                require(
                    all(p.name == ".writer.lock" or _TEMP.fullmatch(p.name) for p in files),
                    "rows/events without identity",
                )
                _publish(manifest, self.identity, self._context)
                self._capture(manifest)
            event_paths = sorted(p for p in files if _EVENT.fullmatch(p.name))
            for index, p in enumerate(event_paths):
                require(p.name == f"event-{index:08d}.json", "event sequence gap")
                event = self._capture(p)
                self._validate_event(event, index)
                self.events.append(event)
            for p in files:
                match = _ROW.fullmatch(p.name)
                if match:
                    index = int(match.group(1))
                    record = self._capture(p)
                    self._validate_record(record, index)
                    self.receipts[index] = record
            for event in self.events:
                if event["kind"] == "adoption":
                    record = self.receipts.get(event["index"])
                    require(
                        record is not None and event["row_receipt_sha256"] == digest(record),
                        "adoption references missing/different row",
                    )
                    require(
                        event["sequence"] > record["attempt_sequence"]
                        and _time(event["utc"]) >= _time(record["completed_utc"]),
                        "adoption predates completed row",
                    )
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _context(self):
        require(
            self.verify_context() == self.identity_sha256, "actual semantic context guard differs"
        )

    def _capture(self, path):
        raw = _read_file(path)
        value = _json(raw)
        require(raw == canonical(value), "ledger bytes are not canonical")
        self.captured[path.name] = hashlib.sha256(raw).hexdigest()
        return value

    def _unchanged(self):
        require(self.fd is not None, "ledger is closed")
        info = (self.root / ".writer.lock").lstat()
        require(
            stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) == self.lock_identity,
            "writer lock inode replaced",
        )
        self._context()
        seen = set()
        for p in self.root.iterdir():
            require(not p.is_symlink() and p.is_file(), "ledger path changed type")
            if p.name == ".writer.lock" or _TEMP.fullmatch(p.name):
                continue
            require(p.name in self.captured, "unowned ledger file appeared")
            require(
                hashlib.sha256(_read_file(p)).hexdigest() == self.captured[p.name],
                "completed ledger file drift",
            )
            seen.add(p.name)
        require(seen == set(self.captured), "completed ledger file disappeared")

    def _index(self, index):
        require(type(index) is int and 0 <= index < len(self.rows), "selection index outside range")

    def _validate_event(self, event, sequence):
        common = {"format", "sequence", "identity_sha256", "kind", "index", "origin", "utc"}
        require(isinstance(event, dict), "invalid event")
        expected = common if event.get("kind") == "attempt" else common | {"row_receipt_sha256"}
        _fields(event, expected, "event")
        require(
            event["format"] == "exl3-evaluation-event-v2"
            and event["sequence"] == sequence
            and type(event["sequence"]) is int
            and event["identity_sha256"] == self.identity_sha256,
            "event identity differs",
        )
        require(event["kind"] in ("attempt", "adoption"), "invalid event kind")
        self._index(event["index"])
        _origin(event["origin"], self.identity["runtime"]["host_id"])
        _time(event["utc"])
        for previous in self.events:
            if previous["origin"]["run_id"] == event["origin"]["run_id"]:
                require(
                    previous["origin"] == event["origin"],
                    "same run ID has conflicting producer identity",
                )

    def _validate_record(self, record, index):
        self._index(index)
        _fields(
            record,
            (
                "format",
                "identity_sha256",
                "index",
                "row_sha256",
                "attempt_sequence",
                "attempt_sha256",
                "producer",
                "started_utc",
                "completed_utc",
                "score",
                "supervised_tokens",
                "loss_sum",
                "loss",
                "metric_algorithm",
            ),
            "row receipt",
        )
        require(
            record["format"] == ROW_FORMAT
            and record["identity_sha256"] == self.identity_sha256
            and type(record["index"]) is int
            and record["index"] == index
            and record["row_sha256"] == digest(self.rows[index])
            and record["metric_algorithm"] == METRIC_ALGORITHM,
            "row identity differs",
        )
        _origin(record["producer"], self.identity["runtime"]["host_id"])
        a = record["attempt_sequence"]
        require(type(a) is int and 0 <= a < len(self.events), "missing attempt event")
        event = self.events[a]
        require(
            event["kind"] == "attempt"
            and event["index"] == index
            and record["attempt_sha256"] == digest(event)
            and record["producer"] == event["origin"]
            and record["started_utc"] == event["utc"],
            "row producer/attempt differs",
        )
        require(
            _time(record["completed_utc"]) >= _time(record["started_utc"]),
            "completion predates attempt",
        )
        total = _score(record["score"], self.rows[index], self.identity)
        n = len(self.rows[index]["input_ids"]) - self.rows[index]["loss_start"]
        require(
            type(record["supervised_tokens"]) is int
            and record["supervised_tokens"] == n
            and _finite(record["loss_sum"])
            and record["loss_sum"] == total
            and _finite(record["loss"])
            and record["loss"] == total / n,
            "v2 row reduction differs",
        )

    @property
    def pending_indices(self):
        self._unchanged()
        return tuple(i for i in range(len(self.rows)) if i not in self.receipts)

    def _event(self, kind, index, origin, **extra):
        self._unchanged()
        event = {
            "format": "exl3-evaluation-event-v2",
            "sequence": len(self.events),
            "identity_sha256": self.identity_sha256,
            "kind": kind,
            "index": index,
            "origin": _json(canonical(origin)),
            "utc": _now(),
            **extra,
        }
        self._validate_event(event, len(self.events))
        path = self.root / f"event-{len(self.events):08d}.json"
        _publish(path, event, self._unchanged)
        self._capture(path)
        self.events.append(event)
        return event

    def begin_row(self, index, producer):
        """Persist attempted exposure BEFORE the caller computes this whole row."""
        self._index(index)
        require(index not in self.receipts, "row already complete; never rescore in this namespace")
        event = self._event("attempt", index, producer)
        self.attempts[event["sequence"]] = index
        return event["sequence"]

    def commit_row(self, attempt, score):
        """Validate/materialize a whole row then publish once; never overwrite."""
        self._unchanged()
        require(
            type(attempt) is int and attempt in self.attempts,
            "attempt was not started by this writer",
        )
        index = self.attempts[attempt]
        require(index not in self.receipts, "row already complete")
        event = self.events[attempt]
        score = _json(canonical(score))
        total = _score(score, self.rows[index], self.identity)
        count = len(score["target_ids"])
        record = {
            "format": ROW_FORMAT,
            "identity_sha256": self.identity_sha256,
            "index": index,
            "row_sha256": digest(self.rows[index]),
            "attempt_sequence": attempt,
            "attempt_sha256": digest(event),
            "producer": event["origin"],
            "started_utc": event["utc"],
            "completed_utc": _now(),
            "score": score,
            "supervised_tokens": count,
            "loss_sum": total,
            "loss": total / count,
            "metric_algorithm": METRIC_ALGORITHM,
        }
        self._validate_record(record, index)
        self._unchanged()
        path = self.root / f"row-{index:08d}.json"
        _publish(path, record, self._unchanged)
        self._capture(path)
        self.receipts[index] = record
        self.new_indices.add(index)
        # The durable receipt is the completion event. No second write is needed
        # to recover a crash immediately after publication.
        return _json(canonical(record))

    def exposure(self):
        self._unchanged()
        attempts = [e for e in self.events if e["kind"] == "attempt"]
        completed = {r["attempt_sequence"] for r in self.receipts.values()}
        return {
            "attempted_row_computations": len(attempts),
            "completed_row_receipts": len(self.receipts),
            "unknown_or_incomplete_attempts": sum(e["sequence"] not in completed for e in attempts),
            "outcome_adoptions": sum(e["kind"] == "adoption" for e in self.events),
            "unique_rows_attempted": len({e["index"] for e in attempts}),
        }

    def aggregate(self, assembler, *, output=None):
        """CPU-only complete merge; original producer metadata remains immutable.

        Existing completed rows generate adoption events. An optional fresh
        output path publishes the aggregate atomically and exclusively. No
        current model/scorer callback is accepted or invoked by this method.
        """
        self._unchanged()
        require(len(self.receipts) == len(self.rows), "cannot publish a partial evaluation")
        _origin(assembler, self.identity["runtime"]["host_id"])
        for event in self.events:
            if event["origin"]["run_id"] == assembler["run_id"]:
                require(
                    event["origin"] == assembler, "same run ID has conflicting assembler identity"
                )
        for index in range(len(self.rows)):
            if index not in self.new_indices and index not in self.adopted:
                self._event(
                    "adoption", index, assembler, row_receipt_sha256=digest(self.receipts[index])
                )
                self.adopted.add(index)
        ordered = [self.receipts[i] for i in range(len(self.rows))]
        n = sum(r["supervised_tokens"] for r in ordered)
        total = math.fsum(v for r in ordered for v in r["score"]["token_nll"])
        require(math.isfinite(total), "aggregate NLL overflow")
        descriptors = [
            {"index": r["index"], "row_sha256": r["row_sha256"], **r["score"]["logits"]}
            for r in ordered
        ]
        result = {
            "format": AGGREGATE_FORMAT,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity,
            "progress_files_sha256": dict(self.captured),
            "metric_algorithm": METRIC_ALGORITHM,
            "examples": len(ordered),
            "supervised_tokens": n,
            "loss_sum": total,
            "loss": total / n,
            "macro_row_loss": math.fsum(r["loss"] for r in ordered) / len(ordered),
            "ordered_row_logits_digest_v2": digest({"domain": DIGEST_DOMAIN, "rows": descriptors}),
            "logits_digest_scope": DIGEST_DOMAIN,
            "row_logit_descriptors": descriptors,
            "evaluation_context": {
                "format": AGGREGATE_FORMAT,
                **self.identity["selection"],
                "multi_producer": True,
                "assembler": _json(canonical(assembler)),
            },
            "rows": [
                {
                    **r["score"],
                    "logits_sha256": r["score"]["logits"]["sha256"],
                    "loss": r["loss"],
                    "loss_sum": r["loss_sum"],
                    "supervised_tokens": r["supervised_tokens"],
                    "correct_tokens": sum(
                        a == b
                        for a, b in zip(
                            r["score"]["target_ids"], r["score"]["predicted_ids"], strict=True
                        )
                    ),
                }
                for r in ordered
            ],
            "row_provenance": [
                {
                    "index": r["index"],
                    "receipt_sha256": digest(r),
                    "producer": r["producer"],
                    "started_utc": r["started_utc"],
                    "completed_utc": r["completed_utc"],
                    "reused": r["index"] not in self.new_indices,
                }
                for r in ordered
            ],
            "exposure": self.exposure(),
            "assembled_utc": _now(),
            "quality_certified": False,
            "scope": "Local declared-receipt consistency; not model provenance authentication, generated quality, or hardware resume qualification",
        }
        self._unchanged()
        if output is not None:
            path = Path(output).absolute()
            require(
                path.parent != self.root,
                "publish final aggregate outside mutable progress namespace",
            )
            _publish(path, result, self._unchanged)
        return _json(canonical(result))
