"""Strict single-action terminal tool-call SFT; no tool execution or observations.

Serialization always delegates to the actual tokenizer chat template. Optional
call IDs are transport metadata preserved in raw data; a template may omit them.
Supported JSON schemas form a small explicit subset, not arbitrary JSON Schema.
"""

import copy
import math
import re

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def _text(value, label, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} must be a {'string' if empty else 'nonempty string'}")
    if any(marker in value for marker in ("<｜", "</｜", "｜DSML｜")):
        raise ValueError(f"{label} contains a reserved chat/tool control marker")
    return value


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError("Tool/function/parameter names must be simple identifiers")
    return value


def _schema(schema, depth=0):
    if depth > 8 or not isinstance(schema, dict) or schema.get("type") not in _TYPES:
        raise ValueError("Tool schema needs one supported type and at most8 nesting levels")
    kind = schema["type"]
    allowed = {"type", "description", "enum"}
    allowed |= {
        "object": {"properties", "required", "additionalProperties"},
        "array": {"items", "minItems", "maxItems"},
        "string": {"minLength", "maxLength"},
        "integer": {"minimum", "maximum"},
        "number": {"minimum", "maximum"},
    }.get(kind, set())
    if set(schema) - allowed:
        raise ValueError(
            "Unsupported tool schema keyword; unsupported constraints are never ignored"
        )
    if "description" in schema:
        _text(schema["description"], "schema description", empty=True)
    if kind == "object":
        props, required = schema.get("properties"), schema.get("required")
        if (
            not isinstance(props, dict)
            or len(props) > 64
            or not isinstance(required, list)
            or any(not isinstance(k, str) for k in required)
            or len(set(required)) != len(required)
            or not set(required).issubset(props)
            or schema.get("additionalProperties") is not False
        ):
            raise ValueError(
                "Tool object schema needs properties, required and additionalProperties:false"
            )
        for name, child in props.items():
            _name(name)
            _schema(child, depth + 1)
    if kind == "array":
        _schema(schema.get("items"), depth + 1)
    for minimum, maximum in (("minLength", "maxLength"), ("minItems", "maxItems")):
        for name in (minimum, maximum):
            if name in schema and (type(schema[name]) is not int or schema[name] < 0):
                raise ValueError("Tool schema length/item bounds must be nonnegative integers")
        if minimum in schema and maximum in schema and schema[minimum] > schema[maximum]:
            raise ValueError("Tool schema minimum exceeds maximum")
    for name in ("minimum", "maximum"):
        if name in schema and (type(schema[name]) not in (int, float) or not _finite(schema[name])):
            raise ValueError("Tool schema numeric bounds must be finite")
    if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
        raise ValueError("Tool schema minimum exceeds maximum")
    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            raise ValueError("Tool schema enum must be a nonempty list")
        for value in schema["enum"]:
            _value(value, {k: v for k, v in schema.items() if k != "enum"}, depth)


def _finite(value):
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _value(value, schema, depth=0):
    kind = schema["type"]
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float) and _finite(value),
        "boolean": type(value) is bool,
        "null": value is None,
    }[kind]
    if not valid or depth > 8:
        raise ValueError("Tool arguments do not match the declared parameter types")
    if kind == "object":
        if set(value) - set(schema["properties"]) or not set(schema["required"]).issubset(value):
            raise ValueError("Tool arguments have unknown or missing required parameters")
        for name, child in value.items():
            _value(child, schema["properties"][name], depth + 1)
    elif kind == "array":
        for child in value:
            _value(child, schema["items"], depth + 1)
    elif kind == "string":
        _text(value, "tool argument", empty=True)
    if kind in {"string", "array"}:
        lo, hi = ("minLength", "maxLength") if kind == "string" else ("minItems", "maxItems")
        if len(value) < schema.get(lo, 0) or (hi in schema and len(value) > schema[hi]):
            raise ValueError("Tool argument length is outside declared bounds")
    if kind in {"number", "integer"}:
        if ("minimum" in schema and value < schema["minimum"]) or (
            "maximum" in schema and value > schema["maximum"]
        ):
            raise ValueError("Tool argument is outside declared numeric bounds")
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        raise ValueError("Tool argument is outside declared enum")


def validate_tools(tools):
    """Validate tool declarations for training or generation without executing them."""
    _functions(tools)
    return copy.deepcopy(tools)


def _functions(tools):
    if not isinstance(tools, list) or not 1 <= len(tools) <= 8:
        raise ValueError("Tool SFT requires1..8 declared function tools")
    functions = {}
    for tool in tools:
        if (
            not isinstance(tool, dict)
            or set(tool) != {"type", "function"}
            or tool["type"] != "function"
        ):
            raise ValueError("Only standard function tool declarations are supported")
        fn = tool["function"]
        if (
            not isinstance(fn, dict)
            or not {"name", "parameters"}.issubset(fn)
            or set(fn) - {"name", "description", "parameters", "strict"}
        ):
            raise ValueError("Tool function requires name and parameter schema")
        name = _name(fn["name"])
        if name in functions:
            raise ValueError("Duplicate tool function name")
        if "description" in fn:
            _text(fn["description"], "tool description", empty=True)
        if "strict" in fn and type(fn["strict"]) is not bool:
            raise ValueError("Tool strict flag must be boolean")
        _schema(fn["parameters"])
        if fn["parameters"]["type"] != "object":
            raise ValueError("Tool parameters must use an object schema")
        functions[name] = fn["parameters"]
    return functions


def normalize_tool_row(row):
    if not isinstance(row, dict) or set(row) != {"tools", "messages"}:
        raise ValueError(
            "Tool SFT requires exactly tools and messages (metadata is handled separately)"
        )
    tools, messages = row["tools"], row["messages"]
    functions = _functions(tools)
    if not isinstance(messages, list) or not messages:
        raise ValueError("Tool SFT messages must be a nonempty list")
    start = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    if len(messages) - start < 2 or (len(messages) - start) % 2:
        raise ValueError("Tool conversation must alternate user/assistant and end with a tool call")
    for index, message in enumerate(messages):
        expected = (
            "system" if index < start else ("user" if (index - start) % 2 == 0 else "assistant")
        )
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(
                "Tool conversation has invalid roles/order; tool observations are unsupported"
            )
        if index != len(messages) - 1:
            if set(message) != {"role", "content"}:
                raise ValueError("Only the terminal assistant may contain tool calls")
            _text(message["content"], "message content")
            continue
        if not {"role", "tool_calls"}.issubset(message) or set(message) - {
            "role",
            "content",
            "tool_calls",
        }:
            raise ValueError("Terminal assistant requires tool_calls and optional text content")
        if message.get("content") is not None:
            _text(message["content"], "assistant content", empty=True)
        calls = message["tool_calls"]
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Only one terminal tool call is supported per SFT example")
        call = calls[0]
        if (
            not isinstance(call, dict)
            or not {"type", "function"}.issubset(call)
            or set(call) - {"type", "function", "id"}
            or call["type"] != "function"
        ):
            raise ValueError("Terminal tool call requires type:function and function")
        if "id" in call:
            _text(call["id"], "tool call id")
        fn = call["function"]
        if (
            not isinstance(fn, dict)
            or set(fn) != {"name", "arguments"}
            or fn["name"] not in functions
        ):
            raise ValueError("Tool call must name a declared function with object arguments")
        _value(fn["arguments"], functions[fn["name"]])
    return copy.deepcopy(row)


def tokenize_tool_row(row, tokenizer, *, all_tokens=False):
    row = normalize_tool_row(row)
    messages, tools = row["messages"], row["tools"]
    ids = tokenizer.apply_chat_template(messages, tools=tools, return_dict=False)
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tools=tools, add_generation_prompt=True, return_dict=False
    )
    if (
        not isinstance(ids, list)
        or not isinstance(prefix, list)
        or ids[: len(prefix)] != prefix
        or len(ids) <= len(prefix)
    ):
        raise ValueError("Tool chat template prefix is inconsistent or target empty")
    # No synthetic EOS, tool-response or success message is appended. Whatever
    # terminal sequence the pinned actual template emits remains supervised.
    return {"input_ids": ids, "loss_start": 1 if all_tokens else max(1, len(prefix))}
