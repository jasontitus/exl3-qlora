"""Convert a generated unified diff into verified ktune SEARCH/REPLACE blocks.

Only existing regular UTF-8 files in an explicit pinned Git parent are supported.
The supplied repository is read-only; patch application uses a disposable bare
repository/index. No hooks, filters, candidate builds or benchmarks run. Whole-file
SEARCH blocks require whole-file editable ranges in the receiving ktune task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

_MAX_PATCH = 2 << 20
_MAX_FILE = 1 << 20
_MAX_EDIT = 250_000  # Actual ktune.edits.apply_response boundary.
_BLOCK = re.compile(
    r"```edit ([^\s]+)\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)>>>>>>> REPLACE\n```",
    re.DOTALL,
)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _git(repo, *args, input=None):
    # Ignore inherited Git overrides, templates and global/system configuration.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", LC_ALL="C"
    )
    result = subprocess.run(
        [
            "git",
            "--no-pager",
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.attributesFile=" + os.devnull,
            "-C",
            str(repo),
            *args,
        ],
        input=input,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
        timeout=30,
    )
    if result.returncode:
        raise ValueError(
            "Git patch validation failed: " + result.stderr.decode(errors="replace")[:500]
        )
    if len(result.stdout) > 4 * _MAX_FILE:
        raise ValueError("Git result exceeds conversion bound")
    return result.stdout


def _name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or str(path) != name
        or ".." in path.parts
        or any(part.lower() == ".git" for part in path.parts)
        or any(c.isspace() or c in "\\\x00" for c in name)
        or (len(path.parts) > 1 and path.parts[0] in {"a", "b"})
    ):
        raise ValueError(f"Unsafe or unsupported ktune path: {name!r}")
    return name


def _patch_paths(text):
    from .ktune_data import preimage_context

    _, paths = preimage_context(text, "unified_diff")
    if not paths or len(paths) > 128:
        raise ValueError("Patch must change 1..128 existing files")
    previous = None
    headers = []
    for line in text.splitlines():
        if line.startswith(
            (
                "new file mode",
                "deleted file mode",
                "old mode",
                "new mode",
                "rename from",
                "rename to",
                "copy from",
                "copy to",
                "similarity index",
                "dissimilarity index",
                "GIT binary patch",
                "Binary files",
                "\\ No newline",
            )
        ):
            raise ValueError("New/deleted/renamed/mode/binary/EOF-newline edits are unsupported")
        if line.startswith("--- "):
            if not line.startswith("--- a/"):
                raise ValueError("Old diff header must name an existing a/ file")
            previous = _name(line[6:])
        elif line.startswith("+++ "):
            if not line.startswith("+++ b/") or previous is None or line[6:] != previous:
                raise ValueError("Renames/new files or unmatched diff paths are unsupported")
            headers.append(previous)
            previous = None
    if sorted(headers) != sorted(paths) or len(set(headers)) != len(headers):
        raise ValueError("Every changed file must have exactly one matching header pair")
    if len({name.casefold() for name in paths}) != len(paths):
        raise ValueError("Case-colliding paths are unsupported")
    return sorted(_name(name) for name in paths)


def _tree_entry(repo, revision, name):
    raw = _git(repo, "ls-tree", "-z", revision, "--", name)
    entries = [entry for entry in raw.split(b"\0") if entry]
    if len(entries) != 1:
        raise ValueError(f"Expected one existing file at pinned parent: {name}")
    header, actual_name = entries[0].split(b"\t", 1)
    mode, kind, oid = header.decode().split()
    if actual_name.decode() != name or mode not in {"100644", "100755"} or kind != "blob":
        raise ValueError(f"Only existing regular files are allowed: {name}")
    size = int(_git(repo, "cat-file", "-s", oid).decode())
    if size > _MAX_FILE:
        raise ValueError(f"Source file exceeds {_MAX_FILE} bytes: {name}")
    data = _git(repo, "cat-file", "blob", oid)
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"Source file is not UTF-8: {name}") from error
    if not text.strip() or "\x00" in text or "\r" in text or not text.endswith("\n"):
        raise ValueError(f"Source must be nonempty UTF-8 LF text with a final newline: {name}")
    return mode, data


def replay_full_file_edit(parent, edit):
    """Strict exact-line replay of emitted blocks, matching ktune's line semantics."""
    matches = list(_BLOCK.finditer(edit.strip()))
    if not matches or _BLOCK.sub("", edit.strip()).strip():
        raise ValueError("Incomplete or ambiguous SEARCH/REPLACE output")
    post, touched = dict(parent), set()
    for match in matches:
        name, search, replace = match.groups()
        if name not in parent or name in touched or not search.strip():
            raise ValueError("Unexpected, duplicate or empty full-file SEARCH")
        if replace and not replace.endswith("\n"):
            raise ValueError("Replacement delimiter must start its own line")
        before = post[name].split("\n")
        needle = search.split("\n")
        positions = [
            i for i in range(len(before) - len(needle) + 1) if before[i : i + len(needle)] == needle
        ]
        if positions != [0] or len(needle) != len(parent[name].splitlines()):
            raise ValueError("SEARCH must uniquely cover the full parent file")
        replacement = replace[:-1].split("\n") if replace else []
        post[name] = "\n".join(replacement + before[len(needle) :])
        touched.add(name)
    if touched != set(parent) or post == parent:
        raise ValueError("Edit must cover every changed file and make a change")
    return post


def export_ktune_edit(repo, parent_revision, patch_path, output, *, allowed_paths=None):
    """Export ``output`` and ``output.receipt.json`` without touching the source repo.

    ``parent_revision`` must be a full 40-character commit hash. ``allowed_paths``
    optionally narrows the editable file set; the receiving task still owns range
    authorization, review and execution checks. Outputs must not already exist.
    """
    if not isinstance(parent_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", parent_revision):
        raise ValueError("parent_revision must be a pinned full 40-character commit hash")
    repo = Path(repo).expanduser().resolve(strict=True)
    patch_path = Path(patch_path).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    receipt_path = output.with_name(output.name + ".receipt.json")
    if output.resolve().is_relative_to(repo) or receipt_path.resolve().is_relative_to(repo):
        raise ValueError("Conversion outputs must be outside the read-only source repository")
    if output.exists() or output.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("Choose new edit and receipt output paths")
    if patch_path.stat().st_size > _MAX_PATCH:
        raise ValueError(f"Patch exceeds {_MAX_PATCH} bytes")
    raw = patch_path.read_bytes()
    text = raw.decode("utf-8")
    if "\x00" in text or "\r" in text:
        raise ValueError("Patch must use UTF-8 LF text")
    paths = _patch_paths(text)
    if allowed_paths is not None:
        if isinstance(allowed_paths, str):
            raise ValueError("allowed_paths must be a collection of exact relative filenames")
        allowed = {_name(name) for name in allowed_paths}
        if set(paths) - allowed:
            raise ValueError("Patch changes files outside allowed_paths")
    actual = _git(repo, "rev-parse", "--verify", parent_revision + "^{commit}").decode().strip()
    if actual != parent_revision:
        raise ValueError("Pinned parent is not the requested commit")
    parent, modes = {}, {}
    for name in paths:
        modes[name], parent[name] = _tree_entry(repo, parent_revision, name)
    with tempfile.TemporaryDirectory(prefix="exl3-ktune-patch-") as directory:
        temp = Path(directory)
        _git(temp, "-c", "init.templateDir=", "init", "--bare", "-q")
        _git(temp, "read-tree", "--empty")
        for name, data in parent.items():
            oid = (
                _git(temp, "hash-object", "--no-filters", "-w", "--stdin", input=data)
                .decode()
                .strip()
            )
            _git(temp, "update-index", "--add", "--cacheinfo", f"{modes[name]},{oid},{name}")
        _git(temp, "apply", "--cached", "--check", "--whitespace=nowarn", "-", input=raw)
        _git(temp, "apply", "--cached", "--whitespace=nowarn", "-", input=raw)
        entries = _git(temp, "ls-files", "--stage", "-z")
        post = {}
        for entry in filter(None, entries.split(b"\0")):
            header, encoded_name = entry.split(b"\t", 1)
            mode, oid, stage = header.decode().split()
            name = encoded_name.decode()
            if name not in parent or mode != modes[name] or stage != "0":
                raise ValueError("Patch changed file membership, mode or index stage")
            size = int(_git(temp, "cat-file", "-s", oid).decode())
            if size > _MAX_FILE:
                raise ValueError("Patched file exceeds conversion bound")
            data = _git(temp, "cat-file", "blob", oid)
            content = data.decode("utf-8")
            if (
                not content.strip()
                or "\x00" in content
                or "\r" in content
                or not content.endswith("\n")
            ):
                raise ValueError("Patched source must be nonempty UTF-8 LF text with final newline")
            post[name] = content
        if set(post) != set(parent):
            raise ValueError("Patch added or removed source files")
    original = {name: data.decode() for name, data in parent.items()}
    # Upstream parse_blocks strips trailing spaces at separators and treats any
    # line starting with the REPLACE sentinel as terminal; mirror that boundary.
    delimiters = {"<<<<<<< SEARCH", "=======", "```"}
    if any(
        line.rstrip() in delimiters or line.startswith(("```edit ", ">>>>>>> REPLACE"))
        for content in (*original.values(), *post.values())
        for line in content.splitlines()
    ):
        raise ValueError("Source contains a SEARCH/REPLACE protocol delimiter collision")
    changed = {name for name in paths if original[name] != post[name]}
    if changed != set(paths):
        raise ValueError("Every declared target must make an actual source change")
    edit = (
        "\n".join(
            f"```edit {name}\n<<<<<<< SEARCH\n{original[name][:-1]}\n=======\n{post[name]}>>>>>>> REPLACE\n```"
            for name in paths
        )
        + "\n"
    )
    encoded = edit.encode()
    if len(encoded) > _MAX_EDIT:
        raise ValueError("Whole-file edit exceeds ktune's 250,000-byte limit")
    if replay_full_file_edit(original, edit) != post:
        raise ValueError("Converted edit does not exactly replay the patched source")
    if patch_path.read_bytes() != raw:
        raise ValueError("Patch changed during conversion")
    receipt = {
        "format": "exl3-ktune-edit-export-v1",
        "parent_revision": parent_revision,
        "source_repository": str(repo),
        "input_patch_sha256": _sha(raw),
        "output_edit_sha256": _sha(encoded),
        "output_bytes": len(encoded),
        "files": {
            name: {
                "mode": modes[name],
                "before_sha256": _sha(parent[name]),
                "after_sha256": _sha(post[name].encode()),
                "required_editable_range": [1, len(original[name].splitlines())],
            }
            for name in paths
        },
        "allowed_paths": sorted(allowed) if allowed_paths is not None else None,
        "source_payload_verified": True,
        "replay_equal": True,
        "compiled": False,
        "correctness_tested": False,
        "performance_tested": False,
        "quality_certified": False,
        "limitations": [
            "Whole-file SEARCH requires whole-file authorized ranges in the receiving ktune task.",
            "The receiving task must independently enforce its source/range and challenge contract.",
            "Exact patch replay is not numerical correctness or an optimization result.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    created = receipt_created = False
    try:
        with output.open("xb") as stream:
            stream.write(encoded)
        created = True
        with receipt_path.open("x", encoding="utf-8") as stream:
            receipt_created = True
            stream.write(json.dumps(receipt, indent=2) + "\n")
    except BaseException:
        if receipt_created:
            receipt_path.unlink()
        if created:
            output.unlink()
        raise
    return {**receipt, "output": str(output), "receipt": str(receipt_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-path", action="append", dest="allowed_paths")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            export_ktune_edit(
                args.repo, args.parent, args.patch, args.output, allowed_paths=args.allowed_paths
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
