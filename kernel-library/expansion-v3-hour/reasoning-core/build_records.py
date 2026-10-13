"""Hand-curated source-grounded benchmark/decision cases; no synthetic execution."""
import json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[2]
MANIFESTS=[W/'kernel-library/expansion-v2/group-b/manifest-reviewed-v2.json',W/'kernel-library/expansion-v2/group-a/manifest-reviewed-v2.json']
loaded=[(p,json.loads(p.read_text())) for p in MANIFESTS]
rows=[]

def add(prefix,suffix,task,question,answer,outcome='tradeoff',extra=None):
    p,m,c=next((p,m,c) for p,m in loaded for c in m['candidates'] if prefix in c['id'])
    # Quote the full hash-bound source description as provenance; prompt contains only
    # the deliberately chosen observations below, not the target explanation.
    evidence=[]
    for field in ['rationale','architecture','operation']:
        for e in c['annotations'].get(field,{}).get('evidence',[]):
            sid=e['source']
            if sid=='commit_message':
                repo=Path(m['repositories'][c['repository']]['path'])
                message=subprocess.check_output(['git','--git-dir='+str(repo),'show','-s','--format=%B',c['commit']])
                assert e['quote'] in message.decode(), 'Manifest quote differs from actual Git message'
                dest=R/'evidence'/('message-'+c['commit']+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(message)
                item={'path':str(dest.resolve()),'sha256':hashlib.sha256(message).hexdigest(),'url':m['repositories'][c['repository']]['url']+'/commit/'+c['commit'],'quote':e['quote']}
            else:
                source=m['evidence'][sid];item={**source,'path':str((p.parent/source['path']).resolve()),'quote':e['quote']}
            if not any(v['quote']==item['quote'] and v['path']==item['path'] for v in evidence):evidence.append(item)
    licenses=[]
    for f in (p.parent/'evidence/licenses').glob('*'+c['commit'][:12]+'*'):
        licenses.append({'path':str(f.resolve()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    if not licenses:
        repo=Path(m['repositories'][c['repository']]['path'])
        raw=subprocess.check_output(['git','--git-dir='+str(repo),'show',c['commit']+':NOTICE'])
        dest=R/'evidence'/('license-'+c['commit']+'.txt');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        licenses.append({'path':str(dest.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'url':m['repositories'][c['repository']]['url']+'/blob/'+c['commit']+'/NOTICE','scope':'Exact revision NOTICE; states MIT for Sushi; referenced LICENSE absent from this revision tree'})
    rows.append({'id':'reason-'+prefix+'-'+suffix,'repo':m['repositories'][c['repository']]['url'],'commit':c['commit'],'parent':c['parent'],
      'lineage':c['repository']+':'+c['commit'],'related_exact_example':c['id'],'task_type':task,'outcome_class':outcome,
      'prompt':'You are reviewing an Apple Silicon inference optimization. The measurements below are historical author reports, not observations from your own execution. Give a concise technical decision, its limits, and the next useful check. Do not invent profiler readings or claim you ran commands.\n\n'+question,
      'completion':answer,'rationale_basis':'mixed','source_evidence':evidence,
      'inference_disclosure':'The target synthesizes a decision and proposed validation from cited historical observations. Recommendations and arithmetic are curator inferences; quoted timings and reported outcomes are source claims.',
      'validation':{'local_execution':'not_run','source_review':'parent curated task/answer against captured sources; independent review pending'},
      'protection_review':c['historical_protection_review'],'licenses':licenses,
      'review':{'status':'pending','reviewer':'pending','evidence_fidelity':True,'no_fabricated_observations':True,'task_answer_separation':True,'lineage_screened':True}})

add('ea7e39730931','architecture-boundary','select_dispatch',
'''An M1 Max report compares affine 4-bit/group-64 multiplication at K=5376, N=8192. At M=6, direct execution takes 1.9305ms and a chunked-QMV control takes 1.6750ms. At M=5, direct/control are 1.4492/1.4515ms. After moving the matrix-dispatch boundary from 6 to 7 for applegpu_g13s and large D/O, M=6 direct takes 1.6987ms. Should we change the boundary for every Apple GPU?''',
'''Keep the change architecture- and shape-specific. The five-row control is effectively equal, while the six-row matrix route is slower than QMV in the reported case; this supports correcting that dispatch boundary rather than rewriting the arithmetic. The patched six-row direct latency is about 12% lower than baseline. It does not establish the same crossover on M3/M5 or smaller matrices. Preserve the other thresholds, verify quantized output tolerances, and run paired repeated sweeps around M=5,6,7,8 on each additional device before expanding the gate. These are primitive timings, not an end-to-end throughput result.''')
add('ea7e39730931','calibration','interpret_benchmark',
'''Five independent 40-trial runs report chunked-QMV/direct latency ratios at six rows of 0.8717,0.8930,0.8947,0.8767,0.9152. The five-row same-route calibration median is 1.0052. Across seven 200-trial runs, the six-row median changes from 0.8807 before a dispatch fix to 1.0118 after it. Why is the calibration useful, and what can we conclude?''',
'''The same-route five-row case estimates timing bias introduced by the comparison method: its ratio is close to one. The repeated six-row ratio below one therefore provides stronger evidence of a route-specific disadvantage than a single best time. After the fix, the ratio near one is consistent with direct execution using the intended QMV route. This supports the measured boundary change, not a universal claim about every matrix size. Retain paired/interleaved measurements and report distributions; do not interpret small deviations around one as another speedup without uncertainty analysis.''')
add('1a4aada280fc','padding-rejection','reject_optimization',
'''A proposed absorbed-MLA change batches all decode rows into one padded gather and SDPA. Author measurements: B128×context512 is 1.4–2.8× as fast; B16×context8k is 0.5–0.7×; a mixed batch of 96 rows×512 plus 32 rows×8k is 0.16×. Is batching everything a sound default?''',
'''Reject unconditional batching. Padding to the longest context expands every short row, and the reported mixed workload is a severe regression. Keep a bounded short-context subset batched and let long rows use the existing path. The source uses at least 16 eligible rows, a per-row context limit 1024, and n×max_context≤65536. Also exclude multi-token segments that need their causal-mask semantics. Validate row order and padding masks against the per-segment reference, then benchmark homogeneous and heterogeneous batches. The reported gains are layer-level; serving throughput remains a separate measurement.''','rejected')
add('1a4aada280fc','volume-gate','select_dispatch',
'''Use this documented candidate-selection policy: single-token absorbed decode only; at least 16 eligible rows; each context≤1024; eligible_count×largest_eligible_context≤65536. Hypothetical requests:A = 128 rows, all context 512; B = 80 rows, all context 1024; C = 16 rows context 512 plus 16 rows context 8192. Determine which rows qualify without pretending these are measured workloads.''',
'''A qualifies: 128×512=65536, exactly the volume ceiling. B fails the volume gate: 80×1024=81920, despite meeting the row-count and individual-context limits. C can batch the 16 short rows: 16×512=8192; the 16 long rows exceed the individual cap and stay on the existing path. Compute volume from the eligible subset, not the longest excluded row, and preserve packed output ordering when merging both paths. These are policy calculations only; they predict dispatch selection, not latency or correctness.''')
add('0ff8b4c40e89','dtype-promotion','diagnose',
'''An M1 Ultra Q6_K path decodes weights to float16 for GEMM when batch exceeds 4. With bfloat16 activations, the report says MLX promotes that GEMM to float32. Separately, the small-batch kernel decodes all weights again for each row. The proposed changes decode into the activation dtype and reuse decoded groups across batch rows. Explain the two mechanisms and required checks.''',
'''There are two distinct costs. Matching the decoded-weight dtype to bfloat16 activations avoids the reported float32 GEMM promotion; that is a dtype-dispatch correction. Reusing each decoded quantization group across rows reduces repeated decoding in the matrix-vector path. Do not attribute both benefits to one mechanism. Check dispatch/dtype selection and compare outputs for FP16 and BF16 separately, across batch boundaries and representative projection/head shapes. The reported 73-token top-1 agreement is useful but is not a numerical error bound or broad model-quality evaluation.''')
add('0ff8b4c40e89','mixed-regressions','review_correctness',
'''A Q6_K optimization improves reported BF16 batch 8 end-to-end latency from 15.5 s to 10.6 s. But isolated FP16 batch 5–8 cases remain 7–13% slower at 4096×4096 and 12–21% slower for a 151936×1024 tied head. Should a single BF16 headline justify raising the QMV limit to 8 for all cases?''',
'''The BF16 result supports that measured configuration, but it does not erase the FP16 regressions. Evaluate the dispatch threshold by dtype and matrix shape rather than treating batch count as the only variable. Preserve the positive measurements and the negative cases in the benchmark suite; compare QMV and GEMM at the boundary with equal warmup and repeated timing. Also separate the dtype-promotion fix from the decoded-group-reuse change in an ablation. A universal threshold needs evidence across those regimes, not just the strongest headline.''')
add('f79dba78325b','micro-versus-serving','interpret_benchmark',
'''A min-p sampler replaces full-vocabulary sorting with max+threshold, using argpartition when more minimum tokens must be retained. At vocabulary 128256 over 500 iterations, time per call falls 0.627 ms → 0.211 ms. A separate model benchmark reports generation 51.126 → 51.450 tokens/s; its temperature-zero path does not exercise min-p. Can the 3× microbenchmark be advertised as 3× faster generation?''',
'''No. The microbenchmark measures the sampling operation: 0.627/0.211≈2.97×, saving 0.416 ms per call in that setup. The reported generation figures differ by only about 0.63%, and a temperature-zero benchmark does not exercise the changed min-p path, so it is mainly a regression control. Run a matched nonzero-temperature workload that actually uses min-p and report full-token latency separately. Avoid extrapolating an isolated operation's speedup to the entire model.''')
add('f79dba78325b','unnecessary-sorting','diagnose',
'''The source states that min-p sampling sorts the entire vocabulary only to obtain the highest log-probability and then map a threshold mask back into the original order. No consumer needs the complete sorted order. What algorithmic replacement is justified, and what behavior must remain?''',
'''Use a maximum reduction and threshold the original-order logits directly. Full sorting provides information this path does not consume, and removing the sort/gather/inverse-map also avoids the reported temporary arrays. When min_tokens_to_keep exceeds one, retain the required top candidates using selection rather than assuming a single maximum is sufficient. Test the keep-count boundary, tied scores, filtering behavior and original index mapping against the existing reference. Complexity improvement is a reason to try the change; measured latency on the target vocabulary and device is still required.''','positive')
add('3ba7272b3305','attribution-correction','interpret_benchmark',
'''An initial Sushi commit credits a one-term f16 probability operand for an 18–25% global-attention layer speedup. A later source correction says that measurement also included lockstep/clamped-load changes; interleaved qL4096/kv8 measurements isolate only 3–8% for f16 P alone. How should the training record and performance explanation be corrected?''',
'''Attribute only the isolated 3–8% reported gain to the f16 P change in that workload. Preserve the 18–25% figure as a combined result that includes other changes, with the later correction linked to the original claim. Do not silently discard the correction or add component percentages as though their effects were independent. Explain that changing the probability representation removes a second probability-value pass, but separate that mechanism from gains due to scheduling and loads. All of these are upstream measurements; local reproduction remains outstanding.''')
add('3ba7272b3305','precision-contract','review_correctness',
'''Sushi replaces a bf16 high-plus-remainder probability operand with one f16 term in P@V. The author reports unchanged cooperative layout and allows additional probability rounding of 2^-11×max|V| per element, with a matching RMS allowance. Served-pack KLD was 0.07768 versus 0.07783. Should this be labeled bitwise-equivalent and should failed old tolerances simply be loosened?''',
'''Do not label it bitwise-equivalent: the representation and rounding contract intentionally change. A revised tolerance needs an analytical justification tied to normalized nonnegative attention weights and the actual low-precision computation, not merely a value large enough to pass failures. Preserve the old-test failure and the revised contract as evidence, then compare against a high-precision oracle over supported masks, shapes and adverse value distributions. The reported KLD check concerns one served pack; it does not establish every model or shape. Separate numerical acceptability from the corrected, isolated performance attribution.''')

rubrics=json.loads((R/'rubrics.json').read_text())
for row in rows:
    suffix=row['id'].split('-',2)[2]
    row['rubric']={k:v for k,v in rubrics[suffix].items() if k!='synthetic_rejected_answer'}
    row['verifier_negative']={'text':rubrics[suffix]['synthetic_rejected_answer'],'origin':'curator_authored_counterexample_not_observed_model_output','label':'unsupported_or_incorrect','training_role':'verifier_or_preference_only_never_positive_SFT'}
R.mkdir(parents=True,exist_ok=True)
(R/'records-draft.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
print(len(rows),'diagnostic records from',len({r['commit'] for r in rows}),'existing change lineages; independent review pending')
