import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "source_units", Path(__file__).parents[1] / "tools/source_units.py"
)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def text(raw, unit):
    result = raw[unit["start"] : unit["end"]]
    assert m.digest(result) == unit["sha256"]
    return result.decode()


def test_zig_multiline_msl_and_nested_named_functions():
    raw = b"""const metal_source =
    \\\\kernel void fake() { " } ( \\\" weird
    \\\\} // should not close Zig
;
pub const Thing = struct {
    /// doc comment
    pub fn run(x: u32) u32 {
        const s = "not } a body";
        _ = s;
        return x + 1;
    }
};
"""
    units = m.scan_units(raw, "zig")["units"]
    source = m.named_unit(raw, "zig", "metal_source")
    assert text(raw, source).endswith(";\n")
    run = m.named_unit(raw, "zig", "run")
    assert run["qualified_name"] == "Thing.run"
    assert "/// doc comment" in text(raw, run)
    assert text(raw, run).endswith("    }\n")
    assert not any(u["name"] == "fake" for u in units)
    assert m.named_unit(raw, "zig", "Thing.run") == run


def test_zig_error_return_type_and_callconv_are_signature():
    raw = b"pub fn x(a: u32) callconv(.c) error{Oops}!u32 { return a; }\n"
    unit = m.named_unit(raw, "zig", "x")
    assert text(raw, unit) == raw.decode()


def test_zig_nested_constant_body_not_cut_at_first_semicolon():
    raw = b"const cfg = .{ .a = blk: { const x = 1; break :blk x; }, .b = 2 };\n"
    assert text(raw, m.named_unit(raw, "zig", "cfg")) == raw.decode()


def test_zig_duplicate_names_reject_unqualified():
    raw = b"const A = struct {\n fn run() void {}\n};\nconst B = struct {\n fn run() void {}\n};\n"
    with pytest.raises(ValueError, match="ambiguous"):
        m.named_unit(raw, "zig", "run")
    assert m.named_unit(raw, "zig", "A.run")["qualified_name"] == "A.run"


@pytest.mark.parametrize("language", ["metal", "c"])
def test_complete_template_and_attributes(language):
    raw = b"""template <typename T,
          int N>
[[kernel]] void project(device T *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
#pragma unroll
 for (int n = 0; n < N; n++) { x[i] += T(n); }
}
"""
    unit = m.named_unit(raw, language, "project")
    assert text(raw, unit) == raw.decode()
    assert not unit["requires_conditional_review"]


def test_c_literals_comments_and_raw_string_do_not_end_body():
    raw = b"""static void run() {
 const char *x = R"tag({ " /* )tag";
 const char *y = "} \\\" ";
 char z = '}'; /* { */
 // }
 consume(x,y,z);
}
"""
    assert text(raw, m.named_unit(raw, "c", "run")) == raw.decode()


def test_complete_macro_handles_continuation_and_unbalanced_macro_braces():
    raw = b"#define BEGIN(x) { \\\n  x;\nvoid run() { int z = 1; }\n"
    units = m.scan_units(raw, "c")["units"]
    macro = next(u for u in units if u["name"] == "BEGIN")
    assert text(raw, macro) == "#define BEGIN(x) { \\\n  x;\n"
    assert m.named_unit(raw, "c", "run")


def test_conditional_wrapped_function_requires_review():
    raw = b"#ifdef USE\nvoid run() {}\n#endif\n"
    u = m.scan_units(raw, "metal")["units"][0]
    assert u["requires_conditional_review"]
    assert u["enclosing_preprocessor_ranges"] == [{"start": 0, "end": len(raw)}]
    with pytest.raises(ValueError, match="Conditional"):
        m.named_unit(raw, "metal", "run")


def test_internal_conditional_requires_review():
    raw = b"void run() {\n#if USE\n x();\n#else\n y();\n#endif\n}\n"
    with pytest.raises(ValueError, match="Conditional"):
        m.named_unit(raw, "metal", "run")


def test_unclosed_preprocessor_rejects():
    with pytest.raises(ValueError, match="Unclosed preprocessor"):
        m.scan_units(b"#ifdef X\nvoid run() {}\n", "c")


@pytest.mark.parametrize(
    "raw",
    [b"fn x() void {", b"fn x() void { ] }", b'const x = "oops;', b"const x = 1; /* unclosed"],
)
def test_unbalanced_or_unterminated_source_fails(raw):
    with pytest.raises(ValueError):
        m.scan_units(raw, "zig")


def test_utf8_offsets_are_bytes():
    raw = '// café\npub fn run() void { const text = "日本語"; }\n'.encode()
    unit = m.named_unit(raw, "zig", "run")
    assert text(raw, unit) == raw.decode()
    assert unit["end"] == len(raw)


def test_struct_complete_declaration_includes_typedef_alias():
    raw = b"typedef struct { uint a; uint b; } Params;\n"
    unit = m.named_unit(raw, "metal", "Params")
    assert text(raw, unit) == raw.decode()


def test_missing_unknown_signature_and_input_bounds_fail():
    with pytest.raises(ValueError, match="missing"):
        m.named_unit(b"void declared();\n", "c", "declared")
    with pytest.raises(ValueError, match="Source exceeds"):
        m.scan_units(b" " * (m.MAX_BYTES + 1), "zig")
    with pytest.raises(ValueError, match="language"):
        m.scan_units(b"", "python")


def test_msl_unroll_macro_does_not_make_loop_a_function():
    raw = b"""inline frag<bfloat> ix_frag(const device bfloat *p) {
 TF_UNROLL
 for (short e = 0; e < 8; e++) { use(p[e]); }
 return frag<bfloat>(0);
}
"""
    functions = [u for u in m.scan_units(raw, "metal")["units"] if u["kind"] == "complete_function"]
    assert [u["name"] for u in functions] == ["ix_frag"]
    assert text(raw, functions[0]) == raw.decode()


def test_msl_complete_constant_list_retains_all_declarators():
    raw = b"constant constexpr int S = 4, F = S * TF_D, MIX = (2 + S) * S;\n"
    unit = m.named_unit(raw, "metal", "S")
    assert unit["kind"] == "complete_declaration"
    assert text(raw, unit) == raw.decode()


def test_interval_sweep_matches_old_all_pairs_including_crossing_and_equal_spans():
    units = [
        {"name": name, "start": start, "end": end}
        for name, start, end in [
            ("outer", 0, 100),
            ("same_a", 3, 20),
            ("same_b", 3, 20),
            ("inner", 5, 10),
            ("cross", 15, 30),
            ("cross_child", 16, 19),
            ("after", 30, 40),
            ("sibling", 101, 110),
        ]
    ]
    expected = {}
    for unit in sorted(units, key=lambda u: (u["start"], -u["end"], u["name"])):
        expected[unit["name"]] = [
            other["name"]
            for other in sorted(units, key=lambda u: (u["start"], -u["end"], u["name"]))
            if other["start"] <= unit["start"]
            and unit["end"] <= other["end"]
            and (other["start"], other["end"]) != (unit["start"], unit["end"])
        ]
    m.annotate_enclosing(units)
    assert {u["name"]: u["enclosing_names"] for u in units} == expected


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_many_sibling_declarations_preserve_parent_names_and_old_line_metadata(newline):
    lines = [b"const Outer = struct {", b"  fn run() void {"]
    lines += [f"    const v{i} = {i};".encode() for i in range(4000)]
    lines += [b"  }", b"};", b"const last = 3;"]
    raw = newline.join(lines)  # Last declaration deliberately has no terminal newline.
    units = m.scan_units(raw, "zig")["units"]
    assert len(units) == 4003
    for unit in units:
        start, end = unit["start"], unit["end"]
        assert unit["line_start"] == raw[:start].count(b"\n") + 1
        assert unit["line_end"] == raw[:end].count(b"\n") + (not raw[:end].endswith(b"\n"))
        expected = [] if unit["name"] in {"Outer", "last"} else ["Outer"]
        if unit["name"].startswith("v"):
            expected.append("run")
        assert unit["enclosing_names"] == expected
