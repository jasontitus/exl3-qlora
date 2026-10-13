"""Audit character-level terminal tool targets with the pinned Qwen template.

CPU only. This does not tokenize, build training masks, or establish model support.
Uses the installed Transformers Jinja renderer without importing MLX.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from transformers.utils.chat_template_utils import _compile_jinja_template


def audit(config_path, data_path):
    config_path, data_path = Path(config_path), Path(data_path)
    config_raw, data_raw = config_path.read_bytes(), data_path.read_bytes()
    cfg = json.loads(config_raw)
    template = _compile_jinja_template(cfg["chat_template"])
    kwargs = {
        "enable_thinking": False,
        "preserve_thinking": False,
        "reasoning_effort": "low",
    }
    rows = []
    for line in data_raw.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        terminal = messages[-1]
        if terminal["role"] != "assistant" or len(terminal.get("tool_calls", [])) != 1:
            raise ValueError("Audit requires exactly one terminal tool call")
        call = terminal["tool_calls"][0]["function"]
        if call["name"] != "exec_command" or set(call["arguments"]) != {"cmd"}:
            raise ValueError(
                "Audit supports the native corpus exec_command cmd target only"
            )
        cmd = call["arguments"]["cmd"]
        if not isinstance(cmd, str):
            raise TypeError("cmd must be text")
        for reserved in (
            "<|im_start|>",
            "<|im_end|>",
            "</parameter>",
            "</function>",
            "</tool_call>",
        ):
            if reserved in cmd:
                raise ValueError("Tool argument collides with native delimiters")
        full = template.render(
            messages=messages, tools=row["tools"], add_generation_prompt=False, **kwargs
        )
        prompt = template.render(
            messages=messages[:-1],
            tools=row["tools"],
            add_generation_prompt=True,
            **kwargs,
        )
        if not full.startswith(prompt):
            raise ValueError(
                "Generation prompt is not an exact prefix of training serialization"
            )
        target = full[len(prompt) :]
        prose = terminal.get("content", "")
        if not isinstance(prose, str):
            raise TypeError("Assistant prose must be text")
        expected = (
            (prose.strip() + "\n\n" if prose.strip() else "")
            + "<tool_call>\n<function=exec_command>\n<parameter=cmd>\n"
            + cmd
            + "\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        )
        if target != expected:
            raise ValueError(
                "Native target changed argument bytes, wrappers or chat termination"
            )
        rows.append(
            {
                "case_id": row["case_id"],
                "prompt_chars": len(prompt),
                "target_chars": len(target),
                "serialized_sha256": hashlib.sha256(full.encode()).hexdigest(),
                "cmd_sha256": hashlib.sha256(cmd.encode()).hexdigest(),
            }
        )
    return {
        "scope": "Character-level native tool serialization only; tokenizer IDs, assistant token masks, truncation and EOS token coverage are NOT qualified",
        "template_kwargs": kwargs,
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizer_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "data_sha256": hashlib.sha256(data_raw).hexdigest(),
        "rows": rows,
        "passed_rows": len(rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = audit(a.tokenizer_config, a.data)
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Passed character-level template checks for {result['passed_rows']} rows")


if __name__ == "__main__":
    main()
