"""Read-only local-source audit; writes only evidence review, no admission/execution."""

import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent / "expansion-v4-hour"
REPO = "https://github.com/ml-explore/mlx"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    inventory = read(ROOT / "inventory.json")
    previous_inventory = read(ROOT / "inventory-message-refs-only.json")
    summary = read(ROOT / "discovery-summary.json")
    groups = collections.defaultdict(list)
    for row in inventory["candidates"]:
        groups[row["commit"]].append(row)
    assert summary["inventory_sha256"] == sha(ROOT / "inventory.json")
    assert summary["candidate_repository_commit_rows"] == len(inventory["candidates"])
    assert summary["unique_exact_commit_ids"] == len(groups)
    assert summary["exact_commit_ids_seen_in_multiple_mirrors"] == sum(
        len(v) > 1 for v in groups.values()
    )
    assert summary["counts_by_mirror"] == dict(
        collections.Counter(
            Path(c["repository_mirror"]).name for c in inventory["candidates"]
        )
    )
    assert len(summary["groups"]) == len(groups)
    for group in summary["groups"]:
        rows = groups[group["commit"]]
        assert group["mirrors"] == sorted(r["repository_mirror"] for r in rows)
        assert group["tags"] == sorted({t for r in rows for t in r["discovery_tags"]})
        assert group["title"] == rows[0]["message"].splitlines()[0]
        assert group["status"] == "requires_source_review_and_lineage_checks"
        assert group["scope"] == (
            "hardware_mention_only"
            if set(rows[0]["discovery_tags"]) <= {"m5"}
            else "feature_message_match"
        )
    captures = (
        read(ROOT / "capture.json")["cases"]
        + read(ROOT / "capture-retry.json")["cases"]
    )
    selected = {c["commit"]: c for c in captures if c["exit_code"] == 0}
    paths = {s: Path(c["path"]).resolve() for s, c in selected.items()}
    pins = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            ROOT / "inventory.py",
            ROOT / "inventory.json",
            ROOT / "inventory-message-refs-only.json",
            ROOT / "inventory-metadata-roots-v1.json",
            ROOT / "discovery-summary-metadata-roots-v1.json",
            ROOT / "tests/test_inventory.py",
            ROOT / "summarize.py",
            ROOT / "discovery-summary.json",
            ROOT / "capture.json",
            ROOT / "capture-retry.json",
            *paths.values(),
        ]
    }
    for s, c in selected.items():
        assert (
            sha(paths[s]) == c["patch_sha256"] and paths[s].stat().st_size == c["bytes"]
        )
    indices = [
        WORK / "mlx/case-index.json",
        WORK / "mlx/batch2/case-index.json",
        WORK / "engines-a/case-index-reviewed-v2.json",
        WORK / "engines-b/case-index-reviewed-v2.json",
    ]
    v4 = []
    for p in indices:
        pins[str(p)] = sha(p)
        for c in read(p)["cases"]:
            if "commit" not in c:
                assert c["exact_edit_eligible"] is False
                continue
            v4.append(
                {
                    "id": c["id"],
                    "commit": c["commit"],
                    "parent": c.get("parent"),
                    "index": str(p),
                    "eligible": c.get("exact_edit_eligible"),
                    "core": c.get("recommended_core"),
                    "tier": c.get("curriculum_tier"),
                }
            )
    metadata = {}
    for c in inventory["candidates"]:
        if c["commit"] in selected:
            raw = subprocess.check_output(
                [
                    "git",
                    "--git-dir",
                    c["repository_mirror"],
                    "show",
                    "--no-patch",
                    "--format=%H%n%P%n%B",
                    c["commit"],
                ],
                env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
                text=True,
            )
            lines = raw.splitlines()
            assert lines[0] == c["commit"] and lines[1].split() == c["parents"]
            metadata[c["commit"]] = c

    def record(
        prefix,
        title,
        points,
        qualifications,
        tests,
        available=None,
        extra_evidence=None,
    ):
        commit = next(s for s in selected if s.startswith(prefix))
        path = paths[commit]
        text = path.read_text()
        teachings = []
        for statement, quote, basis in points:
            assert quote in text, (prefix, quote)
            teachings.append(
                {
                    "statement": statement,
                    "basis": basis,
                    "evidence": {
                        "path": str(path),
                        "sha256": sha(path),
                        "quote": quote,
                        "line": text[: text.index(quote)].count("\n") + 1,
                    },
                }
            )
        return {
            "commit": commit,
            "parent": metadata[commit]["parents"][0],
            "repository": REPO,
            "url": REPO + "/commit/" + commit,
            "title": title,
            "source_patch": {"path": str(path), "sha256": sha(path)},
            "teaching_points": teachings,
            "availability": {
                "minimum_metal_version": None,
                "minimum_macos_version": None,
                "supported_gpu_families": None,
                "scoped_code_or_author_conditions": available or [],
                "scope": "Unknown fields deliberately remain unknown. Repository build predicates and runtime branch conditions do not establish a universal API/device support matrix.",
            },
            "qualification_limits": qualifications,
            "proposed_checks_not_run": tests,
            "sealed_v4_exact_head_matches": [c for c in v4 if c["commit"] == commit],
            "sealed_v4_parent_matches": [
                c for c in v4 if c["commit"] == metadata[commit]["parents"][0]
            ],
            "additional_source_evidence": extra_evidence or [],
            "validation": {
                "local_compile": "not_run",
                "local_numerical": "not_run",
                "local_performance": "not_run",
                "dataset_admission": "not_requested_not_performed",
            },
        }

    records = []
    records.append(
        record(
            "91c83d",
            "Cooperative-tensor template-type compatibility",
            [
                (
                    "The three mma/mman helpers name operand types from the factory-expression decltype, then pass those aliases to get_destination_cooperative_tensor instead of decltype of the local ct_a/ct_b objects. This is a concrete compiler/template compatibility technique, not a claimed algorithmic speedup.",
                    "+  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<",
                    "inspected_patch",
                ),
                (
                    "The stated motivating environment is a Mac OS 27 build failure. This title does not imply that macOS27 is the minimum supported runtime for the resulting kernel.",
                    "Fixing cooperative tensor build issues on Mac OS 27 (#4594)",
                    "author_reported_problem",
                ),
                (
                    "The same commit removes operand-typed SDPA helper macros and a tile-cast macro. A faithful edit task must retain this scope or explain it; the patch alone does not establish why these removals were necessary to the build fix.",
                    "-#define CAST_NAX(DST, SRC)",
                    "inspected_patch_scope",
                ),
            ],
            [
                "No compiler diagnostic, compile receipt or measurement is captured for this selected revision.",
                "Do not infer the precise compiler type-system root cause or removed-macro deadness without instantiation/use-site evidence.",
            ],
            [
                "Compile the real GDN forward and VJP translation units with all used A/B/accumulator types under the affected and a prior supported toolchain.",
                "Check helper use sites before isolating the alias change from the macro cleanup; retain complete before/after code and dependency closure.",
            ],
            [
                "Commit title names Mac OS27; actual supported OS/GPU minimum remains unestablished."
            ],
        )
    )
    records.append(
        record(
            "5778a9",
            "Shape-aware SDPA dispatch by zero padding",
            [
                (
                    "For D72/D80, the change zero-pads Q/K/V to96, retains the caller-provided scale, calls NAX attention, and copies only the original output width back. Changing the scale to one derived from96 would alter the attention contract.",
                    "+  // are zero and the caller's scale is retained.",
                    "inspected_patch",
                ),
                (
                    "Default admission requires both qL and kL at least512, no causal mode, mask or sinks; outer gates require NAX availability and FP16/BF16. The per-call MLX_SDPA_PAD_HEAD_DIM override can broaden shape/mask admission while retaining the outer feature/dtype gates.",
                    "+  bool pad_default = qL >= 512 && kL >= 512 && !do_causal_ && !mask && !sinks;",
                    "inspected_patch",
                ),
                (
                    "The copies carry explicit source/destination strides instead of falsely marking a narrow view into padded storage contiguous. Temporary lifetimes are registered with the command encoder.",
                    "+      // non-contiguous with span > size, so it cannot carry xp's flags.",
                    "inspected_patch",
                ),
                (
                    "The added tests cover masked/causal and sink cases under opt-in, plus FP32 fallback cases. They check GPU availability, not NAX execution itself; these are source tests, not executed evidence in this review.",
                    "+            self.test_sdpa_head_dim_72_80_sinks()",
                    "inspected_test_scope",
                ),
            ],
            [
                "No timing measurements were captured here; amortization is the code comment rationale, not a measured result.",
                "GPU-only tests can pass via fallback on hardware without NAX. Added short-shape tests do not alone demonstrate the new default long-shape branch.",
                "Padding/copy allocation overhead and noncontiguous input/output paths need independent checks.",
            ],
            [
                "Instrument the actual dispatch and test511/512 boundary pairs with overrideunset/0/1 on NAX-capable hardware.",
                "Compare to independent attention reference forD72/D80, FP16/BF16, GQA, strides, masked/sinks override and FP32 fallback; retain originalscale.",
                "Benchmark totalcall time including padding/copy and memory for both short and long shapes; report regressions and scope.",
            ],
            [
                "Runtime predicate metal::is_nax_available()",
                "D in{72,80}; dtypeFP16/BF16; defaultqL/kL>=512 and unmasked/noncausal/no-sinks; explicit override policy as captured.",
            ],
        )
    )
    pr = WORK / "mlx/evidence/mlx-pr-4352.txt"
    pins[str(pr)] = sha(pr)
    prtext = pr.read_bytes().decode()
    measured_quote = "The benchmarks were run on M5 Pro (24GB) and M5 max (128GB) using `mlx_lm.benchmark`"
    assert (
        measured_quote in prtext
        and "Decode speeds are unchanged, as expected." in prtext
    )
    records.append(
        record(
            "c7ff35",
            "Skip discarded SIMD-group MoE work without skipping barriers",
            [
                (
                    "Move expert-row overlap bounds before the K loop and form sg_active from a nonempty intersection. Only an active SIMD group loads its A/B fragments and performs its local matrix work.",
                    "+    const bool sg_active = m_hi_lim > m_lo_lim;",
                    "inspected_patch",
                ),
                (
                    "The surrounding threadgroup barriers remain outside sg_active. Teaching the distinction between SIMD-group-local work and threadgroup-cooperative synchronization is more important than applying a blanket early return.",
                    "         threadgroup_barrier(mem_flags::mem_threadgroup);",
                    "inspected_patch_and_inferred_safety_invariant",
                ),
                (
                    "Both affine and floating-format NAX gather kernels are changed; the accumulator is still initialized and original row-range stores retained.",
                    " mlx/backend/metal/kernels/fp_quantized_nax.h | 109",
                    "inspected_patch_scope",
                ),
            ],
            [
                "This exact commit is already admitted in sealed v4 and is not a new independent example.",
                "The captured primary PR reports model-level prefill improvement2.23–6.97% and unchanged decode on specified M5Pro/Max models/prompts. These are author results, not a local isolated-kernel benchmark or universal support statement.",
            ],
            [
                "Use mixed active/inactive groups in one threadgroup and partial expert boundary/tail tiles; detect hangs and untouched/incorrect stores.",
                "Cover affine and FP quantization, transpose/alignment branches, skewed expert routing; compare before/after outputs and end-to-end prefill separately from decode.",
            ],
            [
                "Primary PR benchmarks M5 Pro24GB and M5 Max128GB; full GPU-family support unknown."
            ],
            [
                {
                    "path": str(pr),
                    "sha256": sha(pr),
                    "url": "https://github.com/ml-explore/mlx/pull/4352",
                    "quote": measured_quote,
                    "basis": "author_reported_benchmark_scope",
                }
            ],
        )
    )
    records.append(
        record(
            "ef5fc0",
            "Explicit Metal member-function receiver address spaces",
            [
                (
                    "Repository documentation explains that Metal4.1 changes the implicit this address space from thread to __metal_generic. Returning a thread-qualified reference can then fail; a trailing thread qualifier constrains the member-function receiver.",
                    "+    is ``__metal_generic`` rather than ``thread``. Member functions in Metal",
                    "author_documented_language_contract",
                ),
                (
                    "The repair propagates receiver qualifiers through member operators, constructors, accessors and declarations/definitions throughout kernel headers; it is not merely a return-type annotation.",
                    "+  T operator()(T x, T y) thread {",
                    "inspected_patch",
                ),
                (
                    "Both built-in and extension compilation add -Wmetal-addr-spaces so missing receiver qualifiers can be detected before they become errors on affected toolchains.",
                    "+                             -Wmetal-addr-spaces)",
                    "inspected_patch",
                ),
            ],
            [
                "The captured primary source is MLX documentation, not a locally executed compiler reproduction or independently checked Apple specification.",
                "No broad minimum OS/GPU requirement follows from a source-language compatibility repair.",
                "The36-file patch should not be misrepresented as a standalone complete kernel; dependency/declaration closure matters.",
            ],
            [
                "Compile a receiver/accessor reproduction and actual affected extension/kernel header closures with the pinned affected toolchain, then earlier supported versions.",
                "Check warning-flag availability separately and declaration/definition consistency; run numerical regression after compilation without claiming arithmetic changes were measured.",
            ],
            [
                "MLX documentation explicitly attributes receiver-address-space behavior to Metal4.1; universal OS/GPU minimum remainsunknown."
            ],
        )
    )
    records.append(
        record(
            "ae3db5",
            "Keep SDK, language and deployment target separate",
            [
                (
                    "Building with a new SDK does not alone opt a binary into NAX: the captured MLX build gate requires Metal>=400, SDK>=26.2 and deploymenttarget>=26.2. This is an explicit repository build policy at this revision.",
                    "+     AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER_EQUAL 26.2)",
                    "inspected_build_policy",
                ),
                (
                    "If no deployment target is supplied, the patch sets it from the host OS, then passes -mmacosx-version-min into the Metal preprocessor version probe. The host OS, SDK and effective language version are printed separately.",
                    '+    set(XCRUN_FLAGS "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")',
                    "inspected_patch",
                ),
                (
                    "The release matrix changes its new-OS package from26.0 to26.2 while retaining14/15 targets; JACCL gets its own SDK/deployment gates. This is not a blanket claim that all MLX needs macOS26.2.",
                    '+        macos-target: ["14.0", "15.0", "26.2"]',
                    "inspected_build_scope",
                ),
            ],
            [
                "A build predicate establishes what this source enables, not a universal runtime support matrix for all Metal devices or all MLX wheels.",
                "No builds or deployment tests have run in this review; source CI definitions are not successful CI receipts.",
                "New-target performance and compatibility with older systems require distinct qualification.",
            ],
            [
                "Configure the actual project with SDK/deployment combinations around26.2 and inspect which NAX/JACCL targets are compiled; verify version-aware comparisons.",
                "Retain old-target fallback build checks and run resulting binaries on declared target systems.",
            ],
            [
                "At this revision, NAX precompiled-kernel gate: MLX_METAL_VERSION>=400 AND macOS SDK>=26.2 AND deploymenttarget>=26.2.",
                "MLX minimumgeneralSDK/deployment checks remain14.0 in this patch; that is scopedbuildcode, not a current product recommendation.",
            ],
        )
    )
    seed_files = (
        sorted(WORK.glob("*/case-index*.json"))
        + sorted(WORK.glob("*/inventory*.json"))
        + sorted(WORK.glob("*/selection.json"))
        + sorted(WORK.glob("*/metadata-summaries.json"))
    )
    seeds = set()

    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and re.fullmatch("[0-9a-f]{40}", value):
            seeds.add(value)

    for path in seed_files:
        if path.stat().st_size <= 16 * 1024 * 1024:
            pins[str(path)] = sha(path)
            visit(read(path))
    coverage = []
    for c in inventory["coverage"]:
        refs = subprocess.check_output(
            ["git", "--git-dir", c["mirror"], "for-each-ref", "--format=%(refname)"],
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
            text=True,
        ).splitlines()
        checked = subprocess.run(
            [
                "git",
                "--git-dir",
                c["mirror"],
                "cat-file",
                "--batch-check=%(objectname) %(objecttype)",
            ],
            input="\n".join(sorted(seeds)) + "\n",
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        roots = [
            x.split()[0] for x in checked.stdout.splitlines() if x.endswith(" commit")
        ]
        walk = subprocess.run(
            [
                "git",
                "--git-dir",
                c["mirror"],
                "log",
                "--all",
                *roots,
                "-5000",
                "--format=%H",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        actual = set(walk.stdout.splitlines())
        if walk.returncode:
            actual.update(roots)
        assert len(actual) == c["commits_scanned"]
        assert len(roots) == c["metadata_commit_roots"]
        assert walk.returncode == c["traversal_exit_code"]
        # Independent raw-object check for every emitted candidate. This does
        # not traverse parents or invoke the discovery implementation.
        for candidate in inventory["candidates"]:
            if candidate["repository_mirror"] != c["mirror"]:
                continue
            raw = subprocess.check_output(
                [
                    "git",
                    "--git-dir",
                    c["mirror"],
                    "cat-file",
                    "-p",
                    candidate["commit"],
                ],
                env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
                text=True,
            )
            head, _, body = raw.partition("\n\n")
            assert [
                x[7:] for x in head.splitlines() if x.startswith("parent ")
            ] == candidate["parents"]
            assert body.strip() == candidate["message"]
        coverage.append(
            c
            | {
                "current_local_ref_count": len(refs),
                "independent_local_count_matches": True,
            }
        )
    result = {
        "schema": "new-metal-source-teaching-review-v1",
        "status": "reviewed_evidence_only_not_dataset_admission",
        "reviewer": "adversarial_code",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "inputs_sha256": pins,
        "counts": {
            "selected_commits": len(records),
            "captured_successfully": len(selected),
            "v4_exact_duplicate_commits": sum(
                bool(c["sealed_v4_exact_head_matches"]) for c in records
            ),
            "not_in_checked_v4_case_indices": sum(
                not c["sealed_v4_exact_head_matches"] for c in records
            ),
            "candidate_inventory": len(inventory["candidates"]),
            "unique_candidate_commit_ids": len(
                {c["commit"] for c in inventory["candidates"]}
            ),
            "reachable_commits_scanned": sum(c["commits_scanned"] for c in coverage),
            "nonempty_log_mirrors": sum(c["commits_scanned"] > 0 for c in coverage),
        },
        "coverage": coverage,
        "summary_review": {
            "counts_grouped_attribution_tags_and_statuses_independently_verified": True,
            "status": "no_false_qualification_claim_found",
            "wording_limit": "hardware_mention_only describes the keyword match, not proof that actual code uses no newer feature. Source review remains required. Exact duplicate grouping does not clear stable-patch/semantic lineage or protected tasks.",
        },
        "superseded_ref_only_scan": {
            "candidate_inventory": len(previous_inventory["candidates"]),
            "reachable_commits_scanned": sum(
                c["commits_scanned"] for c in previous_inventory["coverage"]
            ),
            "nonempty_log_mirrors": sum(
                c["commits_scanned"] > 0 for c in previous_inventory["coverage"]
            ),
            "finding": "Initial --all-only scan omitted ref-less local commit objects. The updated metadata-root scan addresses this without fetching history; old receipt preserved.",
        },
        "discovery_limitations": [
            "The scanner inspects commit messages only. It misses relevant source/API changes with no selected keywords, and long multi-change commit messages can mix an API mention with unrelated edits. Mentions do not establish that a final changed unit uses that feature. Discovery tags are not availability evidence.",
            "Metadata-root discovery includes selection and metadata-summary files. Mirrors with missing parents retain partial traversal output and direct verified commit-object roots; their nonzero traversal statuses and errors remain explicit. Neither this fallback nor nonzero counts establish complete history.",
            "Inventory row and exact-commit counts above precede stable-patch or conceptual-lineage deduplication. Shared commit objects can appear in multiple mirrors; do not count these as independent training examples. The earlier335/310 snapshot remains preserved separately.",
            "Metadata seeding examines only direct lane case-index/inventory/selection/metadata-summaries files under expansion-v4-hour, excluding deeper supplement metadata and all llama.cpp history outside this root. Commit-object type checking is conservative, but does not make discovery exhaustive.",
            "The MLX mirror is shallow; max5000 is a per-mirror upper bound, not demonstrated historical coverage. There is no fresh remote inventory claim.",
            "One of five selected commits is already a sealed v4 exact optimization example. Other four lack matching exact heads in the four bound canonical case indices; this is not semantic-family/protected-task clearance.",
            "Initial four zero-byte failed patch captures are preserved. Successful retry receipts and all five actual patch hashes were checked; no missing patch is silently counted as reviewed.",
            "The four new-by-v4-head records are teaching evidence, not trainable/compiled/validated rows. Full before/after reconstruction, licenses, protected identities/finite-operation screening, exact replay and canonical token lengths remain admission gates.",
        ],
        "records": records,
        "execution": {
            "upstream_scripts_executed": False,
            "network_used_by_review": False,
            "compiler_executed": False,
            "gpu_executed": False,
            "dataset_modified": False,
        },
        "source_review_scope": "Read five selected patches; inspected relevant compatibility/dispatch/arithmetic/synchronization changes and existing primary PR4352. Did not independently audit every declaration in the large address-space migration or prove complete runtime hardware support.",
    }
    for p, h in pins.items():
        assert sha(p) == h
    out = ROOT / "reviewed-teaching-evidence.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"output": str(out), "sha256": sha(out), "counts": result["counts"]})
    )


if __name__ == "__main__":
    main()
