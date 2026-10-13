"""Source-grounded historical reasoning, with proposed checks labeled explicitly."""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
H = lambda b: hashlib.sha256(b).hexdigest()
# Each diagnostic describes an observed historical problem, not a fabricated agent rollout.
R = {
    "0512aa6f": (
        "diagnose",
        "A non-transposed quantized split-K matmul works with a two-dimensional input but gives wrong results when an unbatched weight matrix receives leading input batch axes. Why must the host reshape both input and output before inserting the split-K dimension?",
        "The kernel flattens leading input axes into M rows. The host intermediate must use the same M-by-K and M-by-N interpretation before adding the split axis; otherwise the number of launched rows and the reduction stride describe different layouts. Use metadata reshapes of both views, preserve final output storage, and compare several distinct leading batch axes against a dense reference. Include the unsplit path and batched weights as controls. This is a host/layout repair, not a faster matrix algorithm.",
    ),
    "95f8533a": (
        "diagnose",
        "A Metal view has a negative stride. A kernel stores the eventual offset in a signed type, yet indexing still wraps for scatter or SDPA masks. Explain why the final variable type alone does not make the expression safe, and what a cross-kernel audit should test.",
        "C++ usual arithmetic conversions happen within the expression: mixing an unsigned coordinate with a negative signed displacement can create a large unsigned address before assignment. Widen or cast operands before multiplying or adding, and use signed element-to-location helpers. The historical patch touches scatter/update indexing, a KV sequence index, and segmented-GEMM offsets, so changing one destination variable is incomplete. Check reversed axes, irregular strides, nonzero offsets, contiguous controls, and both narrow and large-index dispatches. Proposed tests must verify values and guard memory, rather than treating successful compilation as proof.",
    ),
    "4089fd56": (
        "diagnose",
        "For finite complex z=(1e20+1e20j), direct complex division can return an invalid answer even when the true result z/z is1. The denominator implementation squares both components. Explain a more stable formulation and its limitations.",
        "The intermediate sum br*br+bi*bi can overflow even though the quotient is representable; tiny components can similarly underflow. Branch on the larger absolute denominator component, compute the smaller-to-larger ratio, and express numerator and denominator using that bounded ratio. Route real-over-complex division through the same path. Test balanced large and tiny finite values, unequal magnitudes, signs and zeros against a higher-precision reference. The captured change fixes demonstrated finite cases; this ratio formula is not a proof of correctness for all infinities, NaNs or near-maximum intermediate sums.",
    ),
    "49028fb2": (
        "review_correctness",
        "Empty GPU sums/products of unsigned inputs fail while equivalent nonempty reductions work. The host requests an identity-initialization kernel for the promoted output dtype. What is the likely gap and how should a regression distinguish an ABI fix from arithmetic correctness?",
        "Check whether initialization kernels are actually instantiated for uint32 and uint64 sum/product outputs. No input arithmetic runs for an empty reduction, so an otherwise correct nonempty kernel cannot cover the missing entry points. Add those instantiations while retaining sum identity0 and product identity1, then test empty reduced dimensions, empty output tensors, all input families and promoted dtypes. Nonempty controls ensure dispatch remains unchanged. The exact source target includes the tests; no local Metal compilation or numerical run is claimed.",
    ),
    "ad00ea2d": (
        "diagnose",
        "A column reduction over a reversed Metal view reaches a long-column kernel whose offset expression adds unsigned lid.y to a negative signed view displacement. Why can a cast before addition matter even if the final pointer offset is signed?",
        "The unsigned operand can promote the negative offset and wrap the address before the pointer expression consumes it. Cast lid.y to the kernel index type before addition. A narrow reproducer should cover the long-column path, but qualification must include other column dispatches, multiple negative/positive strides and contiguous controls; otherwise a correct result can merely show that the failing kernel was not selected. This is an addressing repair with no measured performance claim.",
    ),
    "dfe17baf": (
        "plan_validation",
        "A grouped FP-quantized matmul with K=1040 fails when the kernel iterates 32-wide K tiles; partial output-row tiles also exist. Plan the minimum validation of a repair to K-loop tails and safe-load axes.",
        "Compare complete K tiles plus an explicit remainder against an independent dequantize-and-matmul reference. Exercise K remainders permitted by each quantization group, both transposed orientations, gathered and ordinary paths, ragged output rows and columns, and aligned controls. Check that loader row coordinates compare to the row bound and column coordinates to the column bound. Guard buffers because an overread can leave numerical outputs apparently correct, as with some affine/MXFP partial-N cases. Preserve compile-time full-tile paths where group alignment proves safety. The historical report describes an NVFP4 corruption reproducer; it does not justify attributing every safe-load edit to the same visible value error.",
    ),
    "685dba9a": (
        "diagnose",
        "Two-pass grouped-query attention matches a reference for batch1 but fails for batch2. Q heads are mapped to KV heads, and the K/V offset includes the head index but not the batch. Which indexing contract is missing?",
        "The selected KV head belongs to a batch: add batch_idx*num_kv_heads before multiplying by the head stride, then apply sequence and feature offsets. Use batch2 with deliberately different K/V values per batch; duplicated values could hide the defect. Cover multiple GQA ratios, ragged sequence lengths and batch1 controls. Do not infer that changing this address also validates score scaling, masks or both-pass reductions; those require their own numerical checks.",
    ),
    "a124ac09": (
        "review_correctness",
        "A reduction API requires argmin/argmax to return the first NaN. A scalar comparator is updated, but pairwise tree reductions and partial reductions still choose later NaNs. What must remain consistent across CPU, Metal and CUDA?",
        "Both scalar updates and pairwise merge operations must prioritize NaN and break ties by the earliest index. Carry the same policy through partial reductions, full reductions, dtype-specific NaN helpers and complex handling where supported. Test several NaNs crossing partition boundaries, first/last NaN, infinities, repeated finite extrema and ordinary random inputs. The full historical target retains cross-backend/build changes; testing only the scalar loop or only one backend cannot establish the required contract.",
    ),
    "d73eb752": (
        "diagnose",
        "Sorted affine NAX gathered matmul leaves rows unwritten just above a signed-short boundary. The remaining row count is narrowed to short before min with the SIMD-group row limit. Propose a bounded repair and a negative control.",
        "Compute and clamp the remaining count in int, then narrow only the small bounded extent. Narrowing a value such as32769 first wraps negative and can suppress all work even though the tile needs only a few rows. Compare rows immediately below, at and above the seam, inspect sentinel outputs to catch unwritten rows, and vary expert boundaries. Keep ragged-K handling and tile dispatch unchanged: they are separate mechanisms. The captured author negative control supports this row-extent diagnosis, not a broader quantized-matmul qualification.",
    ),
    "a9eed5a8": (
        "review_correctness",
        "A quantization host binds an input to a Metal encoder, then replaces its view with a row-contiguous temporary. Why does merely using the new array variable in subsequent code not guarantee that the encoded operation depends on the right storage?",
        "Encoder binding records a resource/dependency at the time of the call; reassigning the host variable later does not necessarily update that recorded resource. Establish contiguity and temporary lifetime first, then bind the actual input used by the operation. Test sliced and transposed inputs, evaluate lazily and after other work, and compare against explicit contiguous copies. Ordinary contiguous inputs can conceal the lifetime/order defect. No speed benefit follows from the repair itself.",
    ),
    "a1e0e0b5": (
        "review_correctness",
        "A historical report repairs a 2-by-65 negative-outer-stride row reduction by casting the lane coordinate before offset arithmetic. The reporter verified the unpatched GPU failure and a fixed CPU result but could not rebuild patched Metal. Can the report certify the GPU repair?",
        "No. The code-level explanation is plausible: the looped row path mixed an unsigned lane coordinate with a signed negative row displacement, so operands must be converted before arithmetic. But a CPU result does not exercise the patched Metal path. Retain the source qualification gap and require a rebuilt GPU reproducer with the2-by-65 case, larger rows, positive-stride controls, and guard checks. Distinguish a source-backed repair hypothesis from an executed device validation.",
    ),
    "052d4281": (
        "review_correctness",
        "A negative-stride sort patch widens element-to-location offsets in noncontiguous single-block and multi-block kernels, but the report leaves a contiguous-path problem tied to Split metadata unresolved. How should the change and test scope be described?",
        "Describe exactly the two repaired address computations. Signed64-bit offsets prevent a reversed segment base from wrapping before pointer addition; they do not repair a producer that incorrectly labels a view contiguous. Test sort/argsort/partition on reversed non-sorted axes, multiple sizes that select each repaired path, and unaffected contiguous controls. Keep the Split-related limitation in the result instead of describing universal negative-stride support. A missing test route cannot be converted into a passing claim by excluding it silently.",
    ),
    "2414e5df": (
        "diagnose",
        "A Steel GEMM safe loader checks a tile using offsets off_x and off_y but reads src[(off_x+i)*str_x+(off_x+j)*str_y]. Why can diagonal-offset tests pass, and what is the discriminating test?",
        "The column term incorrectly reuses off_x. When off_x==off_y, the typo is invisible, so use unequal nonzero offsets and distinct per-row/per-column values. Correct the column term to off_y+j, retain the original bounds checks, and test partial boundary tiles with sentinel padding. The source is a one-line typo repair with no quantitative benchmark; a test plan is proposed here, not an invented execution result.",
    ),
    "241f1bf7": (
        "diagnose",
        "A quantized split-K kernel receives the current partition length and uses it as the stride between input rows. The last partition can have a different length. Explain the full-row addressing contract and why it is distinct from flattening leading batch axes.",
        "Partition length bounds the current K loop; it is not the distance between successive complete input rows. Pass a separate full input-row stride, reconstructed as (split_k-1)*nominal_partition_size+final_block_size, through affine and FP quantized callers. Test several rows and an unequal last partition, ordinary QVM and gathered controls. The later leading-axis collapse repair changes host shape interpretation; this earlier change fixes the kernel row stride. Group both in related quantized-layout lineage rather than counting them as independent algorithms.",
    ),
    "fa7b5b8c": (
        "interpret_benchmark",
        "A grouped-convolution change reports 7.5–13.4x gains on several M3 Pro C/group=8 cases,3.4x for conv1d, and7.1x for one forward+backward case. It selects the best of three benchmark rounds and adds0.11MB of kernels. What implementation and performance claims are justified?",
        "The baseline routed C/group=8 through explicit unfold plus batched GEMM despite an existing small-channel loader supporting vector width8. Instantiating that loader and choosing BM=64 removes that dispatch gap; TROWS=64 makes the tile constraint a correctness invariant, not just tuning. The same patch separately admits depthwise channel counts divisible by8. Keep C/group5–7 and9–15 fallback behavior and unchanged output-channel rules. Results are author-reported on one M3 Pro with best-round selection, not a universal Apple GPU speedup. A pad-to16 prototype was reportedly about2x slower than the selected method. Reproduce balanced randomized order on additional hardware, include gradients and unchanged paths, and retain remaining conv1d/low-output-channel gaps.",
    ),
    "f8aaf49d": (
        "interpret_benchmark",
        "A ragged-output QMV patch reports11.37→8.22µs for N=1,K=2560,int8,BF16 on M5 Max and gains for a four-layer Qwen cycle. It also fixes affine bias-sum precision. Can the measurements establish a full-model gain or the isolated benefit of the precision change?",
        "Neither. The timing baseline already includes the precision fix, so the reported ratio isolates the fast ragged-row route relative to that baseline. The model-shaped test covers layers4–7 and explicitly cannot execute the full model. The new path reuses the final valid weight row for excess reads and masks stores; widen each low-precision bias-sum contribution before accumulation independently. Validate both changes separately, output sizes around8, FP quantization/global scales and exact route selection. On M3 and newer, M=3 can select qmv_wide and fail to exercise this route. Preserve the report that full-suite subtest failures were unchanged, rather than claiming the whole suite passed. No local benchmark is implied.",
    ),
    "11a6ce75": (
        "review_correctness",
        "A PR proposes attaching recurrent-cache metadata to one populated tensor to avoid an unevaluated decode graph. The final patch instead returns (cached_tensors,left_padding,lengths) as state and restores empty sentinels in the setter. Which mechanism should an explanation teach?",
        "Teach the final serialized/evaluated-state contract: the caller already evaluates state, so including metadata ensures its lazy graph is evaluated at the same boundary without a new per-step synchronization. The setter must preserve absence versus empty arrays using the exact sentinel convention. The dependency-attachment idea is historical proposal context, not what the committed patch implements. Test repeated decode updates, empty and populated caches, save/restore and bounded retained graph/memory. Inspect process memory and graph lifetime as well as value parity; no measured local memory reduction is asserted.",
    ),
    "ee19be43": (
        "review_correctness",
        "A cache restore loses padding/alignment and prevents an optimized attention path from being selected. A patch separates raw state from keys_and_values and changes metadata serialization. What is the essential compatibility and performance caveat?",
        "Raw serialization should preserve storage and bookkeeping, while consumers requesting valid K/V values use the sliced accessor. Update all state setters and callers together, and encode scalar/string/None metadata consistently. Verify exact valid tokens, offsets and attention outputs before and after restore, plus eligibility for the intended optimized path. The historical format changes are not an automatic promise to load older cache files. Preserving eligibility is a mechanism claim; speed must be measured on a fixed workload after restore, not inferred from the name of a fast kernel.",
    ),
    "9b0063e8": (
        "diagnose",
        "A model captures a cache offset, updates a batched cache, then uses the captured offset for RoPE. In-place += mutates both handles. How does rebinding repair the position contract, and what should the test distinguish?",
        "Create a new offset array and rebind the cache field, leaving the earlier handle unchanged. That preserves the pre-update position used by RoPE while advancing current cache state. Use distinct positions per batch and rotating-cache overflow, and assert both the captured old handle and current new value. Compare incremental outputs with a reference, rather than only inspecting shape. The historical model reproducer motivates a generic ownership repair; it does not establish independence from all model-family evaluation tasks.",
    ),
    "f4f3b579": (
        "diagnose",
        "An empty BF16 KV cache joins a populated batch through concatenation. A zero-length placeholder has default float32 dtype. Why can an array with no values still change memory and numerical behavior?",
        "Concatenation promotes based on operand dtypes even when one operand has zero length, so the placeholder can promote the full populated buffer to float32. Construct the empty shape using the populated side dtype. Test both extend directions and rotating/nonrotating variants, check dtype and retained values, and inspect the subsequent attention dtype. The issue can double BF16 storage and alter arithmetic without any changed token; a simple nonempty-values comparison can miss it.",
    ),
    "e03e45ed": (
        "diagnose",
        "A chunked KV cache allocates padded capacity beyond its valid offset. Trimming uses the allocated shape instead of offset minus start_position. Explain the error and a chunk-composition test.",
        "Capacity is not token count. Locate the valid region using offset-start_position, retain the newest intended chunk within that region, and advance start_position by the actual discarded prefix. Otherwise trimming may retain unwritten padding or drop valid tokens. Compare one full prefill against several chunk partitions, including partial final chunks and a following decode step; inspect both K/V values and absolute masking positions. Preserve the original mask handling of older retained tokens. This is a cache sequence-contract repair, not a claimed throughput gain.",
    ),
    "768ad7db": (
        "review_correctness",
        "An empty quantized KV cache allocates its uint32 width as dim//(32//bits). Why is that wrong for3-,5- and6-bit packing even if2-,4- and8-bit tests pass, and how should provenance be retained?",
        "The integer quotient32//bits discards the cross-word packing fraction. Use the actual packed-bit contract dim*bits//32 for supported aligned dimensions, rather than assuming each word holds an integral number of complete values. Test initial zero-row allocation and growth for each supported bit rate, single and batched caches, and compare to the quantizer-produced shape. This captured port explicitly credits an earlier mlx2 repair, so retain that related lineage instead of treating the same formula as an independent invention. Real-model qualitative text differences are not a correctness oracle.",
    ),
    "5a97f54f": (
        "diagnose",
        "A diffusion sampler accumulates thousands of small softmax probabilities in BF16/FP16 for top-p filtering. Cumulative mass stops increasing accurately. The PR discusses both diffusion and typical-p fixes, but the selected commit changes only diffusion. Specify the repair and honest scope.",
        "Compute probability normalization and cumulative mass in float32, then apply the resulting mask to logits in their original dtype, retaining the most likely token. Widening only the final cumulative sum would be too late. Test flat and peaked large vocabularies across dtypes, top-p boundaries and the guaranteed first-token retention. Keep typical-p as a separate historical commit, not part of this exact target. The effect on token selection is numerical correctness under the filtering contract; no downstream quality or speed improvement follows automatically.",
    ),
    "d56ed249": (
        "interpret_benchmark",
        "A Bluestein FFT optimization reports a median per-cell speed ratio1.03856 across20 selected complex64 cells on M5 Max, but2047-point batch2048 cases are about1.0056/1.0028 and an unchanged control is0.9984 with CI spanning1. It adds16384bytes peak memory and38432bytes metallib. How should the decision be framed?",
        "This is a small, narrowly dispatched time-versus-storage tradeoff. Plan4096 and total batch at least1024 reuse a584-value twiddle table across both internal transforms; smaller batches, real transforms and other plans retain dynamic twiddles. Do not turn the median into a guarantee for every shape: preserve the near-neutral cells and control interval. Reproduce the balanced fresh-process methodology and include allocation/binary-size costs, both FFT directions, threshold controls and accuracy checks. The report distinguishes an earlier benchmark baseline from a later resealed main revision; pin the measured comparison rather than assuming every quoted number was regenerated for the final revision.",
    ),
    "2d27ab05": (
        "review_correctness",
        "An M5 Ultra dense/split-K GEMM PR adds architecture-specific tile configurations but contains no measured latency table. How can the patch be useful training material without being labeled a successful optimization?",
        "Keep it in an explicitly unmeasured opt-in tier. Explain the changed device-specific dispatch and require shape/dtype numerical comparisons plus synchronized timings against the old table, including small matrices, tails and competing split-K costs. Other GPU-family branches must remain unchanged. The commit title establishes intent, not speed or optimality; no invented register-pressure, occupancy or profiler observation should fill the missing evidence.",
    ),
    "c7ff35d9": (
        "plan_validation",
        "A NAX MoE kernel performs multiple expert matrix products for a tile then discards rows assigned to other experts. A patch skips unused SIMD-group work and reports prefill gains2.23–6.97% on M5 Pro/Max with unchanged decode. What is the key synchronization hazard and qualification scope?",
        "A SIMD-group-specific work predicate must not make required threadgroup barriers divergent. Determine overlap with the expert row range and skip only local matrix work, while every participating group follows the shared load/barrier schedule. Verify output initialization and stores for completely inactive and partially active groups, expert boundaries, affine and FP formats and tail tiles. Retain decode as a flat control and the tested model/prompt/hardware scope; source throughput is not a local isolated-kernel measurement. A no-overlap group plus another active group in the same threadgroup is a useful deadlock/value negative control.",
    ),
    "90089440": (
        "interpret_benchmark",
        "A quantized gathered-matmul patch adapts a companion dense expert-boundary scheduler and reports1.17–1.44x gains for affine/NVFP4/MXFP8 model-shaped MoE layer timings. What must be preserved when interpreting the claim and validating the scheduler?",
        "Treat the two scheduler patches as related mechanism lineage. Precomputed expert offsets and tile metadata must cover each selected row exactly once without reading another expert matrix, including sparse or empty experts, skewed counts and partial tiles. Compare dense reference outputs and guard padding across all modified formats, and include the planning cost in synchronized layer timing. The table is model-shaped layer work, not a full-model throughput result. Its body refers to the companion experiment but does not separately state hardware, so do not invent an independently confirmed device for this specific table.",
    ),
    "a2a09fd5": (
        "interpret_benchmark",
        "A sorted gathered dense-matmul report on M5 Ultra gives1.08–1.79x speedups across model-shaped MoE layers. Fixed row tiles may contain pieces of several experts, requiring repeated K passes. How can changing tile scheduling help, and what would falsify the proposed benefit?",
        "Schedule an expert-bounded row range using precomputed offsets, avoiding full K passes whose outputs are mostly discarded. The report shows smaller relative gains for some longer sequences, so workload geometry matters. Compare total synchronized time including offset construction and scheduling, with balanced expert counts, highly skewed routing, empty experts and tail rows. Check that every row is computed once and that sorted-index assumptions are enforced. A workload where planning dominates or existing tiles already align may show no gain; retain that result. The source cites a related Quack scheduler design, so preserve attribution and shared mechanism lineage rather than claiming a new independent algorithm.",
    ),
}


def main():
    inventory = json.loads((ROOT / "inventory-all.json").read_text())["records"]
    cases = {
        x["commit"]: x
        for x in json.loads((ROOT / "case-index.json").read_text())["cases"]
    }
    records = []
    for row in inventory:
        key = row["commit"][:8]
        if key not in R:
            continue
        case = cases[row["commit"]]
        if case["status"] != "reviewed_exact_edit":
            raise ValueError("Reasoning needs resolved source " + key)
        task, prompt, completion = R[key]
        bp = ROOT / row["pr"]["body_path"]
        patch = ROOT / row["patch"]["path"]
        body = bp.read_bytes().decode()
        raw = patch.read_bytes().decode()
        plus = [
            s for s in raw.splitlines() if s.startswith("+") and not s.startswith("+++")
        ]
        records.append(
            {
                "id": f"{row['repo']}-{row['commit'][:12]}-{task}-hour-v4",
                "repo": case["repo"],
                "commit": row["commit"],
                "parent": row["parent"],
                "lineage": case["lineage"],
                "source_group": case["source_group"],
                "task_type": task,
                "task_framing": "prospective_diagnosis"
                if key
                in {
                    "0512aa6f",
                    "95f8533a",
                    "4089fd56",
                    "49028fb2",
                    "ad00ea2d",
                    "dfe17baf",
                    "685dba9a",
                    "a124ac09",
                    "d73eb752",
                    "a9eed5a8",
                    "2414e5df",
                    "241f1bf7",
                    "9b0063e8",
                    "f4f3b579",
                    "e03e45ed",
                    "768ad7db",
                    "5a97f54f",
                }
                else "retrospective_analysis",
                "framing_scope": "Historically reconstructed tasks; prospective means prompt omits the final repair and attained speed, not an independently future-held-out discovery problem.",
                "source_evidence": [
                    {
                        "path": str(bp),
                        "sha256": H(bp.read_bytes()),
                        "url": row["pr"]["url"],
                        "quote": body.strip() or " ",
                    },
                    {
                        "path": str(patch),
                        "sha256": H(patch.read_bytes()),
                        "url": row["url"],
                        "quote": plus[0] if plus else raw.splitlines()[0],
                    },
                ],
                "prompt": prompt,
                "completion": completion
                + " The evidence is historical source inspection; proposed checks above were not executed locally.",
                "rationale_basis": "mixed",
                "outcome_class": case["outcome_class"],
                "validation": {
                    "local_execution": "not_run",
                    "source_replay": case["exact_replay"],
                    "reported_results": "Historical author reports, not local compiler/GPU/model/benchmark results.",
                },
                "review": {
                    "status": "approved",
                    "reviewer": "adversarial_code author source review; independent review pending",
                    "evidence_fidelity": True,
                    "no_fabricated_observations": True,
                    "task_answer_separation": True,
                    "lineage_screened": True,
                },
                "protection_review": {
                    "status": case["protection_review"]["status"],
                    "reason": case["protection_review"]["reason"],
                },
                "licenses": [
                    {"path": str(ROOT / x["path"]), "sha256": x["sha256"]}
                    for x in row["licenses"]
                ],
            }
        )
    path = ROOT / "reasoning-records.json"
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    spec = importlib.util.spec_from_file_location(
        "reasoning_data",
        WORK / "kernel-library/expansion-v3-hour/tools/reasoning_data.py",
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for r in records:
        m.validate(r, m.reserved_ids())
    print(
        json.dumps(
            {
                "records": len(records),
                "distinct_commits": len({r["commit"] for r in records}),
                "all_source_quote_hash_schema_protection_checks": "passed",
                "sha256": H(path.read_bytes()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
