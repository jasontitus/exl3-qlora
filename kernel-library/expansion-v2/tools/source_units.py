"""Conservative lexical source-unit boundaries; no semantic/dependency claim.

Lists complete Zig fn/const/var declarations (including nested declarations and
multiline strings), and C/MSL function definitions, struct/enum/union definitions
and preprocessor macros. Exact byte ranges and hashes are returned. Ambiguous or
preprocessor-conditional C units are flagged; reviewers must retain required
macros, templates, dispatch and dependencies. Full edit targets remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from bisect import bisect_left
from pathlib import Path

MAX_BYTES = 16 << 20
IDENT = r"[A-Za-z_][A-Za-z_0-9]*"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def lex(raw, language):
    if language not in {"zig", "c", "metal"}:
        raise ValueError("language must be zig, c or metal")
    if len(raw) > MAX_BYTES:
        raise ValueError("Source exceeds16MiB")
    raw.decode("utf-8")
    text = raw.decode("latin1")
    masked = list(text)
    directives = []

    def hide(a, b):
        for k in range(a, b):
            if masked[k] not in "\r\n":
                masked[k] = " "

    def line_end(a):
        end = text.find("\n", a)
        return len(text) if end < 0 else end + 1

    i = 0
    while i < len(text):
        if language == "zig" and text.startswith("\\\\", i):
            end = line_end(i)
            hide(i, end)
            i = end
        elif text.startswith("//", i):
            end = line_end(i)
            hide(i, end)
            i = end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("Unterminated block comment")
            if language == "zig":
                raise ValueError("Zig has no block comments; uncertain source")
            hide(i, end + 2)
            i = end + 2
        elif language != "zig" and text.startswith('R"', i):
            start = text.find("(", i + 2)
            if start < 0 or start - i > 18 or any(c in " ()\\\t\r\n" for c in text[i + 2 : start]):
                raise ValueError("Invalid C++ raw-string delimiter")
            delimiter = ")" + text[i + 2 : start] + '"'
            end = text.find(delimiter, start + 1)
            if end < 0:
                raise ValueError("Unterminated raw string")
            end += len(delimiter)
            hide(i, end)
            i = end
        elif text[i] in "\"'":
            quote, end = text[i], i + 1
            while end < len(text):
                if text[end] == "\\":
                    end += 2
                elif text[end] == quote:
                    end += 1
                    break
                elif text[end] in "\r\n":
                    raise ValueError("Unterminated quoted literal")
                else:
                    end += 1
            else:
                raise ValueError("Unterminated quoted literal")
            hide(i, end)
            i = end
        elif (
            language != "zig"
            and text[i] == "#"
            and not text[text.rfind("\n", 0, i) + 1 : i].strip()
        ):
            end = line_end(i)
            while text[i:end].rstrip("\r\n").endswith("\\") and end < len(text):
                end = line_end(end)
            match = re.match(r"#\s*([A-Za-z_]+)", text[i:end])
            directives.append(
                {"start": i, "end": end, "directive": match[1] if match else "unknown"}
            )
            hide(i, end)
            i = end
        else:
            i += 1
    code = "".join(masked)
    stack, pairs = [], {}
    for i, char in enumerate(code):
        if char in "([{":
            stack.append((char, i))
        elif char in ")]}":
            if not stack or "([{".index(stack[-1][0]) != ")]}".index(char):
                raise ValueError(
                    "Unbalanced source; use complete file or compiler-backed extraction"
                )
            _, start = stack.pop()
            pairs[start] = i
            pairs[i] = start
    if stack:
        raise ValueError("Unbalanced source; use complete file or compiler-backed extraction")
    return code, pairs, directives


def skip_space(code, position):
    while position < len(code) and code[position].isspace():
        position += 1
    return position


def statement_end(code, pairs, start):
    i = start
    while i < len(code):
        if code[i] == ";":
            return i + 1
        if code[i] in "([{":
            i = pairs[i] + 1
        elif code[i] in ")]}":
            raise ValueError("Declaration escaped its enclosing scope")
        else:
            i += 1
    raise ValueError("Declaration lacks terminating semicolon")


def leading_comments(raw, start):
    """Include adjacent whole-line // docs, preserving byte offsets."""
    first = raw.rfind(b"\n", 0, start) + 1
    if raw[first:start].strip():
        return start
    start = first
    while start:
        end = start - 1
        previous = raw.rfind(b"\n", 0, end) + 1
        if raw[previous:end].lstrip().startswith(b"//"):
            start = previous
        else:
            break
    return start


def annotate_enclosing(units):
    """Sweep active intervals, retaining the original sorted ancestor order.

    Equal-span declarations are peers. Crossing intervals are allowed here:
    they may enclose a later unit even if they do not enclose each other.
    Disjoint declarations expire before the next lookup, avoiding a full
    all-pairs scan of local constants in large generated Zig files.
    """
    units.sort(key=lambda u: (u["start"], -u["end"], u["name"]))
    active, ends = {}, []
    for index, unit in enumerate(units):
        while ends and ends[0][0] <= unit["start"]:
            _, expired = heapq.heappop(ends)
            del active[expired]
        parents = [
            other["name"]
            for other in active.values()
            if unit["end"] <= other["end"]
            and (other["start"], other["end"]) != (unit["start"], unit["end"])
        ]
        unit["enclosing_names"] = parents
        unit["qualified_name"] = ".".join([*parents, unit["name"]])
        active[index] = unit
        heapq.heappush(ends, (unit["end"], index))


def scan_units(raw, language):
    code, pairs, directives = lex(raw, language)
    units, warnings = [], []
    newlines = [match.start() for match in re.finditer(b"\n", raw)]
    conditional_stack, conditional_ranges = [], []
    for directive in directives:
        if directive["directive"] in {"if", "ifdef", "ifndef"}:
            conditional_stack.append(directive)
        elif directive["directive"] == "endif":
            if not conditional_stack:
                raise ValueError("Unmatched preprocessor endif")
            opening = conditional_stack.pop()
            conditional_ranges.append({"start": opening["start"], "end": directive["end"]})
        elif directive["directive"] in {"else", "elif"} and not conditional_stack:
            raise ValueError("Unmatched preprocessor branch")
    if conditional_stack:
        raise ValueError("Unclosed preprocessor condition")

    def add(name, kind, start, end, *, extra=None):
        start = leading_comments(raw, start)
        # Newlines are not necessary for completeness, but retain an immediately
        # adjacent terminal newline without swallowing another declaration.
        if raw[end : end + 2] == b"\r\n":
            end += 2
        elif raw[end : end + 1] == b"\n":
            end += 1
        value = {
            "name": name,
            "kind": kind,
            "start": start,
            "end": end,
            "sha256": digest(raw[start:end]),
            "line_start": bisect_left(newlines, start) + 1,
            "line_end": bisect_left(newlines, end - 1) + 1,
            "boundary_method": "lexical_balanced_declaration",
            "semantic_completeness": "review_required",
            "dependency_closure": "not_established",
        }
        value.update(extra or {})
        enclosing = [r for r in conditional_ranges if r["start"] < start < r["end"]]
        if enclosing:
            value["enclosing_preprocessor_ranges"] = enclosing
            value["requires_conditional_review"] = True
        units.append(value)

    if language == "zig":
        pattern = re.compile(
            r"(?m)^[ \t]*(?:(?:pub|export|extern|inline|noinline|comptime|threadlocal)\s+)*(?P<tag>fn|const|var)\s+(?P<name>"
            + IDENT
            + r")\b"
        )
        for match in pattern.finditer(code):
            start = match.start()
            if match["tag"] != "fn":
                end = statement_end(code, pairs, match.end())
                add(match["name"], "complete_declaration", start, end)
                continue
            paren = skip_space(code, match.end())
            if paren >= len(code) or code[paren] != "(":
                warnings.append(
                    {"name": match["name"], "start": start, "reason": "unrecognized_fn_signature"}
                )
                continue
            i = pairs[paren] + 1
            while i < len(code):
                if code[i] == ";":
                    add(match["name"], "complete_declaration", start, i + 1)
                    break
                if code[i] in "([":
                    i = pairs[i] + 1
                    continue
                if code[i] == "{":
                    prior = re.search(r"([A-Za-z_]+)\s*$", code[pairs[paren] + 1 : i])
                    if prior and prior[1] in {"error", "struct", "union", "enum", "opaque"}:
                        i = pairs[i] + 1
                        continue
                    add(match["name"], "complete_function", start, pairs[i] + 1)
                    break
                if code[i] == "}":
                    warnings.append(
                        {"name": match["name"], "start": start, "reason": "unrecognized_fn_body"}
                    )
                    break
                i += 1
            else:
                warnings.append(
                    {"name": match["name"], "start": start, "reason": "missing_fn_body"}
                )
    else:
        # High precision subset: an explicit return/type prefix plus a named
        # parameter list, followed by a definition body. Constructors, operator
        # overloads and requires expressions intentionally need other review.
        function = re.compile(
            r"(?m)^[ \t]*(?P<prefix>[A-Za-z_][A-Za-z_0-9<>,:*& \t]*[ \t]+|\[\[[^\n]*?\]\][ \t]*[A-Za-z_][A-Za-z_0-9<>,:*& \t]*[ \t]+)(?P<name>"
            + IDENT
            + r")\s*\("
        )
        for match in function.finditer(code):
            if match["name"] in {"if", "for", "while", "switch", "catch", "sizeof", "alignof"}:
                continue
            if re.search(r"\b(return|if|while|for|switch|catch|else|typedef)\b", match["prefix"]):
                continue
            paren = match.end() - 1
            i = skip_space(code, pairs[paren] + 1)
            while i < len(code) and code[i] == "[":
                i = skip_space(code, pairs[i] + 1)
            # const/noexcept member forms are common; other suffixes fail closed.
            suffix = re.match(r"(?:(?:const|noexcept|override|final)\s*)*", code[i:])
            i = skip_space(code, i + suffix.end())
            if i >= len(code) or code[i] != "{":
                continue
            start = match.start()
            # Preserve a preceding multiline template prefix. Reject opaque
            # prefixes instead of silently omitting template parameters.
            prior = code[:start].rstrip()
            if prior.endswith(">"):
                candidate = prior.rfind("template")
                if candidate >= 0 and re.fullmatch(
                    r"template\s*<[^;{}]*>\s*", code[candidate:start]
                ):
                    start = code.rfind("\n", 0, candidate) + 1
            end = pairs[i] + 1
            conditionals = [
                d
                for d in directives
                if d["directive"] in {"if", "ifdef", "ifndef", "elif", "else", "endif"}
                and start <= d["start"] < end
            ]
            add(
                match["name"],
                "complete_function",
                start,
                end,
                extra={
                    "conditional_directives_inside": conditionals,
                    "requires_conditional_review": bool(conditionals),
                },
            )
        type_pattern = re.compile(
            r"(?m)^[ \t]*(?:typedef\s+)?(?P<tag>struct|union|enum)\s*(?P<name>" + IDENT + r")?\s*\{"
        )
        for match in type_pattern.finditer(code):
            brace = match.end() - 1
            after = skip_space(code, pairs[brace] + 1)
            tail = re.match(r"(?:(?:" + IDENT + r")\s*)?;", code[after:])
            if tail:
                name = (
                    match["name"]
                    or code[after : after + tail.end()].strip(" ;")
                    or f"anonymous_{match.start()}"
                )
                add(name, "complete_declaration", match.start(), after + tail.end())
        constants = re.compile(
            r"(?m)^[ \t]*(?:(?:static|inline|constant|constexpr|const)[ \t]+)+"
            r"[A-Za-z_][A-Za-z_0-9<>,:*& \t]*?[ \t]+(?P<name>"
            + IDENT
            + r")[ \t]*(?:\[[^\n]*?\])?[ \t]*="
        )
        for match in constants.finditer(code):
            add(
                match["name"],
                "complete_declaration",
                match.start(),
                statement_end(code, pairs, match.end()),
            )
        for d in directives:
            if d["directive"] == "define":
                name = re.match(
                    rb"#\s*define\s+([A-Za-z_][A-Za-z_0-9]*)", raw[d["start"] : d["end"]]
                )
                if name:
                    add(name[1].decode(), "complete_macro", d["start"], d["end"])
    annotate_enclosing(units)
    return {
        "format": "lexical-source-units-v1",
        "language": language,
        "source_sha256": digest(raw),
        "source_bytes": len(raw),
        "units": units,
        "warnings": warnings,
        "scope": "Exact complete lexical declaration boundaries for a conservative syntax subset, not compiler AST or semantic/dependency qualification. Nested units can overlap; select non-overlapping units for the importer. Unknown syntax requires whole-file/manual/compiler review. Never drops target edits.",
    }


def named_unit(raw, language, name):
    matches = [
        u
        for u in scan_units(raw, language)["units"]
        if u["name"] == name or u["qualified_name"] == name
    ]
    if len(matches) != 1:
        raise ValueError("Named unit missing or ambiguous: " + name)
    unit = matches[0]
    if unit.get("requires_conditional_review"):
        raise ValueError("Conditional source requires explicit complete-branch review")
    return unit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--language", choices=["zig", "c", "metal"], required=True)
    parser.add_argument("--name")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    with Path(args.source).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    result = (
        named_unit(raw, args.language, args.name) if args.name else scan_units(raw, args.language)
    )
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
