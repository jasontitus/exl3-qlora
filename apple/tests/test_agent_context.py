import copy
import hashlib
import json

import pytest

from exl3_mlx.agent_context import edit_location_example, main
from exl3_mlx.agent_edits import agent_example, parse_command


def record():
    old = "// omitted distant context\n" * 300 + "float result = x;\n" + "// tail\n" * 300
    new = old.replace("float result = x;", "float result = x + y;")
    return {
        "files": {
            "src/kernel.metal": {
                "role": "changed",
                "before": {"mode": "100644", "content": old},
                "after": {"mode": "100644", "content": new},
            },
            "host.cpp": {
                "role": "changed",
                "before": {"mode": "100644", "content": "dispatch(x);\n"},
                "after": {"mode": "100644", "content": "dispatch(x, y);\n"},
            },
            "tests/new.cpp": {
                "role": "changed",
                "before": None,
                "after": {"mode": "100644", "content": "check(x + y);\n"},
            },
        },
        "reviewed_context": {
            "units": {"src/kernel.metal": [{"before": {"start": 0}, "before_content": old}]}
        },
        "annotations": {"operation": {"text": "a Metal operation"}},
    }


def command(row):
    return row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["cmd"]


def test_context_reduction_preserves_complete_author_edit_and_record():
    source = record()
    original = copy.deepcopy(source)
    whole, _ = agent_example(source)
    row, receipt = edit_location_example(source)
    assert source == original
    assert command(row) == command(whole)
    assert receipt["full_commit_replay"]
    assert receipt["complete_changed_paths"] == ["host.cpp", "src/kernel.metal", "tests/new.cpp"]
    assert (
        receipt["target_patch_sha256"]
        == hashlib.sha256(parse_command(command(row)).encode()).hexdigest()
    )
    assert len(row["messages"][1]["content"]) < len(whole["messages"][1]["content"])
    assert receipt["training_admission"] == "not_granted_by_renderer"


def test_every_old_anchor_and_new_file_identity_remain_visible():
    row, receipt = edit_location_example(record())
    prompt = row["messages"][1]["content"]
    assert "float result = x;" in prompt and "dispatch(x);" in prompt
    assert "float result = x + y;" not in prompt
    assert '"tests/new.cpp"' in prompt
    assert receipt["baseline_inventory"]["tests/new.cpp"]["baseline_exists"] is False
    assert "Whole declarations and dependency files are not shown" in prompt


def test_reviewed_rationale_preserved_only_with_identical_complete_tool_call():
    source = record()
    reference = agent_example(source)[0]["messages"][-1]
    reference["content"] = "Separate reviewed explanation, not an empirical claim."
    row, receipt = edit_location_example(source, reference_assistant=reference)
    assert row["messages"][-1] == reference
    assert receipt["reference_assistant_preserved"]
    reference["tool_calls"][0]["function"]["arguments"]["cmd"] += "\necho altered"
    with pytest.raises(ValueError, match="complete exact tool call"):
        edit_location_example(source, reference_assistant=reference)


def test_context_availability_notice_corrected_without_rewriting_primary_report():
    source = record()
    notice = "All complete before-source files and primary report supplied."
    quoted = "<primary_report>" + notice + "</primary_report>"
    source["annotations"]["constraints"] = {
        "status": "source_reported",
        "text": notice + "\n" + quoted,
    }
    original = copy.deepcopy(source)
    row, receipt = edit_location_example(source)
    prompt = row["messages"][1]["content"]
    assert source == original and receipt["context_availability_notice_rewritten"]
    assert "Complete before-source files remain archived" in prompt
    assert quoted in prompt and prompt.count(notice) == 1


@pytest.mark.parametrize("reference", ["", "{}", '{"messages": []}', '{"messages": [null]}'])
def test_explicit_invalid_reference_rejected_before_output(tmp_path, reference):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(record()))
    ref = tmp_path / "reference.json"
    ref.write_text(reference)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        main([str(source), str(output), "--reference-raw", str(ref)])
    assert not output.exists()


def test_exact_additional_contract_context_preserves_target():
    source = record()
    source["files"]["include/types.h"] = {
        "role": "context",
        "before": {"mode": "100644", "content": "header\r\ncontract\u0085still_same_line\nlast"},
        "after": {"mode": "100644", "content": "header\r\ncontract\u0085still_same_line\nlast"},
    }
    spec = {
        "path": "include/types.h",
        "first_line": 2,
        "last_line": 2,
        "source_sha256": hashlib.sha256(
            source["files"]["include/types.h"]["before"]["content"].encode()
        ).hexdigest(),
    }
    original = copy.deepcopy(source)
    row, receipt = edit_location_example(source, baseline_context=[spec])
    assert source == original
    assert command(row) == command(edit_location_example(source)[0])
    assert "contract\u0085still_same_line\n" in row["messages"][1]["content"]
    assert (
        receipt["baseline_contract_excerpts"][0]["excerpt_sha256"]
        == hashlib.sha256(b"contract\xc2\x85still_same_line\n").hexdigest()
    )


@pytest.mark.parametrize("change", ["hash", "bool", "outside", "absent", "overlap", "path_type"])
def test_invalid_additional_context_fails_closed(change):
    source = record()
    spec = {
        "path": "host.cpp",
        "first_line": 1,
        "last_line": 1,
        "source_sha256": hashlib.sha256(b"dispatch(x);\n").hexdigest(),
    }
    ranges = [spec]
    if change == "hash":
        spec["source_sha256"] = "0" * 64
    elif change == "bool":
        spec["first_line"] = True
    elif change == "outside":
        spec["last_line"] = 2
    elif change == "absent":
        spec["path"] = "tests/new.cpp"
    elif change == "path_type":
        spec["path"] = ["host.cpp"]
    else:
        ranges.append(dict(spec))
    with pytest.raises(ValueError):
        edit_location_example(source, baseline_context=ranges)


def test_extra_context_does_not_rewrite_primary_report():
    source = record()
    report = (
        "<primary_report>only baseline edit locations are supplied in this task.</primary_report>"
    )
    source["annotations"]["constraints"] = {
        "status": "source_reported",
        "text": "All complete before-source files and primary report supplied.\n" + report,
    }
    spec = {
        "path": "host.cpp",
        "first_line": 1,
        "last_line": 1,
        "source_sha256": hashlib.sha256(b"dispatch(x);\n").hexdigest(),
    }
    row, _ = edit_location_example(source, baseline_context=[spec])
    assert report in row["messages"][1]["content"]
    assert "Additional declared contract excerpts follow" in row["messages"][1]["content"]


@pytest.mark.parametrize(
    "mutation", ["delete", "mode", "unsafe", "binary", "role", "context", "role_list"]
)
def test_unsupported_complete_edit_is_not_silently_dropped(mutation):
    source = record()
    pair = source["files"]["host.cpp"]
    if mutation == "delete":
        pair["after"] = None
    elif mutation == "mode":
        pair["after"]["mode"] = "100755"
    elif mutation == "unsafe":
        source["files"]["../host.cpp"] = source["files"].pop("host.cpp")
    elif mutation == "binary":
        pair["after"]["content"] = "bad\0source\n"
    elif mutation == "role_list":
        pair["role"] = ["changed"]
    else:
        pair["role"] = "unknown" if mutation == "role" else "context"
    with pytest.raises(ValueError):
        edit_location_example(source)
