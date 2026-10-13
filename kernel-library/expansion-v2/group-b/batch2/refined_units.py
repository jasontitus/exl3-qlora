"""Complete changed Python methods and explicitly reviewed C++ function units.

No edit can disappear: upstream reconstruction must replay these units to the full
after-image. Failed extraction is an exclusion, not permission to trim a patch.
"""
import ast
import difflib
import hashlib
import importlib.util
import re
from pathlib import Path

_SOURCE_UNITS_PATH = Path(__file__).resolve().parents[2] / "tools/source_units.py"
_spec = importlib.util.spec_from_file_location("expansion_source_units", _SOURCE_UNITS_PATH)
_source_units = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source_units)


def bind(raw, a, b):
    return {"start": a, "end": b, "sha256": hashlib.sha256(raw[a:b]).hexdigest()}


def python_units(path, pair):
    old = pair["before"]["content"].encode()
    new = pair["after"]["content"].encode()

    def parse(raw):
        tree = ast.parse(raw.decode())
        offsets = [0]
        for line in raw.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return tree, offsets

    bt, bo = parse(old)
    at, ao = parse(new)

    def key(node):
        if hasattr(node, "name"):
            return type(node).__name__ + ":" + node.name
        return type(node).__name__ + ":" + ast.dump(node, include_attributes=False)

    def parts(nodes, offsets, start, end):
        out = []
        for node in nodes:
            stop = offsets[node.end_lineno]
            out.append((key(node), start, stop, node))
            start = stop
        if out:
            a, b, _, node = out[-1]
            out[-1] = (a, b, end, node)
        return out

    out = []

    def compare(bparts, aparts):
        for tag, i, j, k, l in difflib.SequenceMatcher(None, [x[0] for x in bparts], [x[0] for x in aparts], autojunk=False).get_opcodes():
            spans = [(x, x + 1, y, y + 1) for x, y in zip(range(i, j), range(k, l))] if tag == "equal" else [(i, j, k, l)]
            for i, j, k, l in spans:
                b0 = bparts[i][1] if i < len(bparts) else bparts[-1][2]
                b1 = bparts[j - 1][2] if j > i else b0
                a0 = aparts[k][1] if k < len(aparts) else aparts[-1][2]
                a1 = aparts[l - 1][2] if l > k else a0
                if old[b0:b1] == new[a0:a1]:
                    continue
                if j - i == l - k == 1:
                    bn, an = bparts[i][3], aparts[k][3]
                    if isinstance(bn, ast.ClassDef) and isinstance(an, ast.ClassDef) and bn.body and an.body:
                        bhead = bo[min([bn.body[0].lineno, *[x.lineno for x in getattr(bn.body[0], "decorator_list", [])]]) - 1]
                        ahead = ao[min([an.body[0].lineno, *[x.lineno for x in getattr(an.body[0], "decorator_list", [])]]) - 1]
                        if old[b0:bhead] == new[a0:ahead]:
                            compare(parts(bn.body, bo, bhead, b1), parts(an.body, ao, ahead, a1))
                            continue
                    kind = "complete_function" if isinstance(bn, (ast.FunctionDef, ast.AsyncFunctionDef)) else "complete_type" if isinstance(bn, ast.ClassDef) else "complete_declaration"
                else:
                    kind = "complete_declaration"
                out.append({"path": path, "kind": kind, "before": bind(old, b0, b1), "after": bind(new, a0, a1)})

    compare(parts(bt.body, bo, 0, len(old)), parts(at.body, ao, 0, len(new)))
    return out


def function_range(text, needle, *, include_conditional=False):
    """Explicit function signature, byte-safe complete body; no conditional crop."""
    raw = text.encode("utf-8")
    code, pairs, directives = _source_units.lex(raw, "c")
    if code.count(needle) != 1:
        raise ValueError("Reviewed signature must occur exactly once in code: " + needle)
    at = code.index(needle)
    start = code.rfind("\n", 0, at) + 1
    paren = code.index("(", at, at + len(needle))
    body = pairs[paren] + 1
    while body < len(code) and code[body].isspace():
        body += 1
    if body >= len(code) or code[body] != "{":
        raise ValueError("Unsupported reviewed function suffix: " + needle)
    end = pairs[body] + 1
    if raw[end:end + 1] == b";":
        end += 1
    if raw[end:end + 2] == b"\r\n":
        end += 2
    elif raw[end:end + 1] == b"\n":
        end += 1
    depth = 0
    enclosing_start = None
    for directive in directives:
        tag = directive["directive"]
        if directive["start"] < start:
            if tag in {"if", "ifdef", "ifndef"}:
                if depth == 0:
                    enclosing_start = directive["start"]
                depth += 1
            elif tag == "endif":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unmatched preprocessor condition")
        elif directive["start"] < end and tag in {"if", "ifdef", "ifndef", "elif", "else", "endif"}:
            raise ValueError("Conditional function requires complete-file context")
    if depth:
        if not include_conditional:
            raise ValueError("Conditional function requires complete-file context")
        # Explicit declaration-group extraction retains the WHOLE enclosing
        # preprocessor block, including all branches and unchanged declarations.
        for directive in directives:
            if directive["start"] < end:
                continue
            if directive["directive"] in {"if", "ifdef", "ifndef"}:
                depth += 1
            elif directive["directive"] == "endif":
                depth -= 1
                if depth == 0:
                    return enclosing_start, directive["end"]
        raise ValueError("Unclosed enclosing preprocessor condition")
    return start, end


C_FUNCTIONS = {
    ("ea7e3973", "mlx/backend/metal/quantized.cpp"): ["inline int get_qmv_batch_limit("],
    ("d9add9d1", "mlx/backend/metal/quantized.cpp"): ["void GatherQQMM::eval_gpu("],
    ("d9add9d1", "mlx/ops.cpp"): ["array gather_qqmm("],
}

C_GROUPS = {
    ("3c40e8f9", "mlx/backend/metal/jit_kernels.cpp"): [
        ("const char* steel_gemm_segmented_nax(", "const char* steel_gemm_segmented_nax(", "const char* steel_gemm_segmented_nax(", "const char* steel_gemm_thin_nax("),
        ("MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(", "MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(", "MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(", "MTL::ComputePipelineState* get_steel_gemm_thin_nax_kernel("),
    ],
    ("3c40e8f9", "mlx/backend/metal/nojit_kernels.cpp"): [
        ("MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(", "MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(", "MTL::ComputePipelineState* get_steel_gemm_segmented_nax_kernel(", "MTL::ComputePipelineState* get_steel_gemm_thin_nax_kernel("),
    ],
    ("02ce1fb6", "mlx/backend/metal/jit_kernels.cpp"): [
        ("MTL::ComputePipelineState* get_logsumexp_kernel(", "MTL::ComputePipelineState* get_logsumexp_kernel(", "MTL::ComputePipelineState* get_logsumexp_kernel(", "MTL::ComputePipelineState* get_cross_entropy_kernel("),
    ],
    ("02ce1fb6", "mlx/backend/metal/nojit_kernels.cpp"): [
        ("MTL::ComputePipelineState* get_logsumexp_kernel(", "MTL::ComputePipelineState* get_logsumexp_kernel(", "MTL::ComputePipelineState* get_logsumexp_kernel(", "MTL::ComputePipelineState* get_cross_entropy_kernel("),
    ],
}


BATCH_FUNCTIONS = {
    ("a87691d2", "mlx/backend/metal/matmul.cpp"): ["void steel_matmul_axpby("],
    ("bbf8a7a5", "mlx/backend/metal/scaled_dot_product_attention.cpp"): ["void sdpa_vector_2pass("],
    ("a082cb91", "mlx/backend/metal/quantized.cpp"): ["void qmm_nax("],
    ("3eb822ca", "mlx/backend/metal/quantized.cpp"): ["void qmm_nax(", "void qmm(", "void qmm_splitk(", "void QuantizedMatmul::eval_gpu(", "void QQMatmul::eval_gpu("],
    ("3eb822ca", "mlx/backend/metal/kernels/fp_quantized.h"): ["[[kernel]] void fp_qmm_t("],
    ("3eb822ca", "mlx/backend/metal/kernels/fp_quantized_nax.h"): ["[[kernel]] void fp_qmm_t_nax("],
    ("3eb822ca", "mlx/backend/metal/kernels/quantized.h"): ["[[kernel]] void affine_qmm_t("],
    ("3eb822ca", "mlx/backend/metal/kernels/quantized_nax.h"): ["[[kernel]] void affine_qmm_t_nax("],
}


def template_function(raw, needle):
    a, b = function_range(raw.decode(), needle)
    code, _, _ = _source_units.lex(raw, "c")
    prefix = code[:a].rstrip()
    if prefix.endswith(">"):
        start = prefix.rfind("template")
        if start < 0 or (start and (code[start - 1].isalnum() or code[start - 1] == "_")) or not re.fullmatch(r"template\s*<[^;{}]*>\s*", code[start:a]):
            raise ValueError("Ambiguous template prefix")
        a = code.rfind("\n", 0, start) + 1
    return a, b


def units(key, path, pair, fallback):
    names = BATCH_FUNCTIONS.get((key, path))
    if names:
        old = pair["before"]["content"].encode()
        new = pair["after"]["content"].encode()
        result = [{"path": path, "kind": "complete_function", "before": bind(old, *template_function(old, name)), "after": bind(new, *template_function(new, name))} for name in names]
        if key == "3eb822ca" and path.endswith("fp_quantized_nax.h"):
            # Complete include prelude retains the copyright edit too.
            before = old.index(b"template")
            after = new.index(b"template")
            result.insert(0, {"path": path, "kind": "complete_declaration", "before": bind(old, 0, before), "after": bind(new, 0, after)})
        return result
    if pair["before"] and pair["after"] and path.endswith(".py"):
        return python_units(path, pair)
    groups = C_GROUPS.get((key, path))
    if groups:
        old = pair["before"]["content"].encode()
        new = pair["after"]["content"].encode()
        return [{"path": path, "kind": "complete_declaration", "before": bind(old, function_range(old.decode(), bs, include_conditional=True)[0], function_range(old.decode(), be, include_conditional=True)[1]), "after": bind(new, function_range(new.decode(), ns, include_conditional=True)[0], function_range(new.decode(), ne, include_conditional=True)[1])} for bs, be, ns, ne in groups]
    if (key, path) == ("3c40e8f9", "mlx/backend/metal/matmul.cpp"):
        old = pair["before"]["content"].encode()
        new = pair["after"]["content"].encode()
        # Whole include prelude plus the complete dispatch function.
        ranges = [[], []]
        for raw, target in ((old, ranges[0]), (new, ranges[1])):
            text = raw.decode()
            target.append((0, len(text[:text.index("namespace mlx::core")].encode())))
            target.append(function_range(text, "void steel_matmul_axpby("))
        return [{"path": path, "kind": "complete_declaration" if i == 0 else "complete_function", "before": bind(old, *ranges[0][i]), "after": bind(new, *ranges[1][i])} for i in range(2)]
    names = C_FUNCTIONS.get((key, path))
    if names:
        old = pair["before"]["content"].encode()
        new = pair["after"]["content"].encode()
        return [{"path": path, "kind": "complete_function", "before": bind(old, *function_range(old.decode(), name)), "after": bind(new, *function_range(new.decode(), name))} for name in names]
    return fallback(path, pair)
