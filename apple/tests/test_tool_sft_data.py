"""CPU-only native terminal tool-call validation, masking and actual-template parity."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exl3_mlx.data import load_tokenizer, tokenize_row
from exl3_mlx.sft_data import _normalize, prepare_sft_data
from exl3_mlx.tool_data import normalize_tool_row, tokenize_tool_row

MODEL = Path.home() / "models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw"


def row():
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Apply an edit to the source tree.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "Edit the provided kernel. Verification is separate."},
            {"role": "user", "content": "Avoid the redundant multiplication.\nfloat y = x * 1.0f;"},
            {
                "role": "assistant",
                "content": "Remove the redundant multiply.",
                "tool_calls": [
                    {
                        "id": "edit_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": {
                                "cmd": "git apply --whitespace=nowarn - <<'PATCH_a1'\n--- a/kernel.metal\n+++ b/kernel.metal\n@@ -1 +1 @@\n-float y = x * 1.0f;\n+float y = x;\nPATCH_a1"
                            },
                        },
                    }
                ],
            },
        ],
    }


class TinyTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, return_dict=False
    ):
        text = json.dumps(tools, sort_keys=True) + "".join(
            m["role"]
            + ":"
            + (m.get("content") or "")
            + (json.dumps(m["tool_calls"], sort_keys=True) if "tool_calls" in m else "")
            for m in messages
        )
        if add_generation_prompt:
            text += "assistant:"
        # The completion renderer omits content on the masked header, so this
        # exactly mirrors prefix construction above. Distinct tool terminator.
        return [ord(c) + 100 for c in text] + ([] if add_generation_prompt else [77])

    def decode(self, ids):
        return "".join(chr(i - 100) if i >= 100 else "<tool-end>" for i in ids)


def test_terminal_schema_is_retained_and_metadata_separated():
    original = row()
    normalized, metadata, kind = _normalize({**original, "source_id": "family1"}, None, None)
    assert normalized == original
    assert normalized is not original
    assert metadata == {"source_id": "family1"}
    assert kind == "terminal_tool_call"
    assert normalize_tool_row(original) == original


def test_exact_template_mask_and_non_eos_ending_preserved():
    original = row()
    tokenizer = TinyTokenizer()
    actual = tokenize_tool_row(original, tokenizer)
    full = tokenizer.apply_chat_template(original["messages"], tools=original["tools"])
    prefix = tokenizer.apply_chat_template(
        original["messages"][:-1], tools=original["tools"], add_generation_prompt=True
    )
    assert actual == {"input_ids": full, "loss_start": len(prefix)}
    assert actual["input_ids"][-1] == 77 != tokenizer.eos_token_id
    assert (
        "exec_command"
        not in tokenizer.decode(actual["input_ids"][: actual["loss_start"]]).split("assistant:")[-1]
    )
    assert tokenize_row(original, tokenizer) == actual
    assert tokenize_row(original, tokenizer, all_tokens=True)["loss_start"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(tools=[]),
        lambda r: r["tools"].append(copy.deepcopy(r["tools"][0])),
        lambda r: r["tools"][0]["function"]["parameters"].update(additionalProperties=True),
        lambda r: r["tools"][0]["function"]["parameters"].update(unknown="silently ignored"),
        lambda r: r["tools"][0]["function"]["parameters"]["required"].append("missing"),
        lambda r: r["messages"][-1]["tool_calls"].append(
            copy.deepcopy(r["messages"][-1]["tool_calls"][0])
        ),
        lambda r: r["messages"][-1]["tool_calls"][0]["function"].update(name="undeclared"),
        lambda r: r["messages"][-1]["tool_calls"][0]["function"].update(arguments='{"cmd":"x"}'),
        lambda r: r["messages"][-1]["tool_calls"][0]["function"].update(arguments={}),
        lambda r: r["messages"][-1]["tool_calls"][0]["function"].update(arguments={"cmd": 1}),
        lambda r: r["messages"][-1]["tool_calls"][0]["function"].update(
            arguments={"cmd": "x", "extra": "y"}
        ),
        lambda r: r["messages"][-1]["tool_calls"][0]["function"].update(
            arguments={"cmd": "</｜DSML｜ parameter>"}
        ),
        lambda r: r["messages"].append({"role": "tool", "content": "fake success"}),
        lambda r: r["messages"][1].update(tool_calls=[]),
        lambda r: r["messages"][-1].update(reasoning_content="unrequested hidden reasoning"),
    ],
)
def test_invalid_tool_records_fail_before_tokenization(mutation):
    original = row()
    mutation(original)
    with pytest.raises((ValueError, TypeError)):
        normalize_tool_row(original)


def test_object_argument_schema_constraints_are_enforced():
    original = row()
    parameter = original["tools"][0]["function"]["parameters"]["properties"]["cmd"]
    parameter.update(minLength=3, maxLength=8, enum=["allowed"])
    args = original["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    args["cmd"] = "allowed"
    normalize_tool_row(original)
    args["cmd"] = "other"
    with pytest.raises(ValueError, match="enum"):
        normalize_tool_row(original)
    args["cmd"] = "too long string"
    with pytest.raises(ValueError, match="length"):
        normalize_tool_row(original)


def test_prepare_supports_template_tool_terminator_without_invented_eos(tmp_path, monkeypatch):
    from exl3_mlx import sft_data
    from exl3_mlx.data import validate_example

    source, model = tmp_path / "source.jsonl", tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 200000}}))
    source.write_text(json.dumps({**row(), "source_id": "groupA"}) + "\n")
    monkeypatch.setattr(
        sft_data,
        "_helpers",
        lambda: (
            lambda _: TinyTokenizer(),
            tokenize_row,
            lambda _: {"fake": "identity"},
            validate_example,
        ),
    )
    report = prepare_sft_data(source, model, tmp_path / "out", demo=True)
    assert report["terminal_tool_call_rows"] == 1
    prepared = json.loads((tmp_path / "out/train.jsonl").read_text())
    assert prepared["input_ids"][-1] == 77
    assert report["preview"][0]["supervised"].endswith("<tool-end>")
    assert json.loads((tmp_path / "out/raw-train.jsonl").read_text()) == row()


def _installed_dataset_classes():
    import importlib.metadata

    distribution = importlib.metadata.distribution("mlx-lm")
    path = Path(distribution.locate_file("mlx_lm/tuner/datasets.py"))
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in {"ChatDataset", "CompletionsDataset", "TextDataset"}
    ]
    namespace = {"List": list, "Dict": dict, "PreTrainedTokenizer": object}
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def actual_tokenizer():
    if not (MODEL / "tokenizer.json").exists():
        pytest.skip("Actual local DeepSeek tokenizer is not available")
    return load_tokenizer(MODEL)


def test_actual_deepseek_template_tool_call_and_installed_mlx_lm_parity(actual_tokenizer):
    original, tokenizer = row(), actual_tokenizer
    actual = tokenize_row(original, tokenizer)
    classes = _installed_dataset_classes()
    ids, offset = classes.ChatDataset([], tokenizer, mask_prompt=True).process(original)
    assert actual == {"input_ids": ids, "loss_start": offset}
    decoded = tokenizer.decode(ids[offset:])
    assert "Remove the redundant multiply." in decoded
    assert '<｜DSML｜ invoke name="exec_command">' in decoded
    assert "git apply --whitespace=nowarn" in decoded
    assert decoded.endswith("</｜DSML｜ calls>" + tokenizer.eos_token)
    assert ids[-1] == tokenizer.eos_token_id
    assert "fake success" not in decoded
    # The template intentionally omits transport call IDs; changing one cannot
    # be misrepresented as a different trained action.
    changed = copy.deepcopy(original)
    changed["messages"][-1]["tool_calls"][0]["id"] = "different_transport_id"
    assert tokenize_row(changed, tokenizer) == actual


def test_actual_plain_text_formats_bitwise_match_installed_mlx_lm(actual_tokenizer):
    tokenizer, classes = actual_tokenizer, _installed_dataset_classes()
    variants = [
        (
            {"prompt": "Translate café.", "completion": "Café."},
            classes.CompletionsDataset([], tokenizer, "prompt", "completion", True),
        ),
        (
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Again"},
                    {"role": "assistant", "content": "Hello again"},
                ]
            },
            classes.ChatDataset([], tokenizer, mask_prompt=True),
        ),
        ({"text": "Text with π and a newline\n."}, classes.TextDataset([], tokenizer)),
    ]
    for original, dataset in variants:
        ids, offset = dataset.process(original)
        if "text" in original and ids[0] != tokenizer.bos_token_id:
            ids.insert(0, tokenizer.bos_token_id)
        assert tokenize_row(original, tokenizer) == {"input_ids": ids, "loss_start": max(1, offset)}


def _generation_namespace():
    """Compile actual generator with a tiny CPU model/runtime, no MLX import."""
    import hashlib
    import sys

    import numpy as np

    from exl3_mlx.context import MAX_CONTEXT_TOKENS

    source = Path(__file__).parents[1] / "src/exl3_mlx/workflow.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate"]
    namespace = {
        "__name__": "exl3_mlx.generation_cpu_tool_test",
        "__package__": "exl3_mlx",
        "json": json,
        "hashlib": hashlib,
        "Path": Path,
        "sys": sys,
        "MAX_CONTEXT_TOKENS": MAX_CONTEXT_TOKENS,
        "mx": SimpleNamespace(
            array=np.array, int32=np.int32, isfinite=np.isfinite, all=np.all, argmax=np.argmax
        ),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def test_generation_native_tools_exact_prompt_and_raw_dsml_preserved(tmp_path, actual_tokenizer):
    import numpy as np

    original, tokenizer = row(), actual_tokenizer
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(original["tools"]))
    messages = tmp_path / "messages.json"
    messages.write_text(json.dumps(original["messages"][:-1]))
    raw = "<｜DSML｜ calls>\n</｜DSML｜ calls>" + tokenizer.eos_token
    answer = tokenizer.encode(raw, add_special_tokens=False)

    class Model:
        def __init__(self):
            self.calls = []

        def eval(self):
            pass

        def __call__(self, ids):
            target = answer[len(self.calls)]
            self.calls.append(ids.tolist()[0])
            logits = np.zeros((1, 1, len(tokenizer)), np.float32)
            logits[0, 0, target] = 1
            return logits

        def report(self):
            return {}

    namespace = _generation_namespace()
    namespace["load_tokenizer"] = lambda _: tokenizer
    args = SimpleNamespace(
        max_tokens=len(answer),
        messages_file=str(messages),
        tools_file=str(tools),
        raw_prompt=False,
        model=str(MODEL),
        prompt=None,
    )
    model = Model()
    result = namespace["generate"](args, model, SimpleNamespace(check=lambda: None))
    expected = tokenizer.apply_chat_template(
        original["messages"][:-1],
        tools=original["tools"],
        add_generation_prompt=True,
        return_dict=False,
    )
    assert result["prompt_ids"] == model.calls[0] == expected
    assert result["raw_text"] == raw
    assert result["generated_ids"] == answer
    assert result["tools"] == original["tools"]
    assert result["stop_reason"] == "eos"
    args.raw_prompt = True
    with pytest.raises(ValueError, match="tools-file"):
        namespace["generate"](args, model, SimpleNamespace(check=lambda: None))


def test_generation_schema_validation_and_no_tools_default(tmp_path):
    namespace = _generation_namespace()
    tools = tmp_path / "tools.json"
    tools.write_text("[]")
    args = SimpleNamespace(
        max_tokens=1,
        messages_file=None,
        tools_file=str(tools),
        raw_prompt=False,
        model="unused",
        prompt="Q",
    )
    with pytest.raises(ValueError, match="declared"):
        namespace["generate"](args, None, None)
    tools.write_text('[{"type":"function","type":"other"}]')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        namespace["generate"](args, None, None)
