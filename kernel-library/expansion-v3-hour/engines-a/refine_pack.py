from pathlib import Path
import json,hashlib,sys,copy
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path[:0]=[str(W/'loss-fix/apple/src'),str(W/'kernel-library/expansion-v2/tools')]
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
from source_units import lex
sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((R/'draft-manifest.json').read_text());options={x['case_id']:x['options']for x in json.loads((R/'context-options.json').read_text())}
choices={'9a5467375008':['qsaSparseAttn','qsaSparseAttnServes'],'987c40030406':['GEMM_NAX_SOURCE','GEMM_NAX_HEADER','getGemmNaxKernel'],'3dc1dab54e7b':['K1S_SOURCE','recurSeq'],'17c6726e8388':['moe_expert_gguf_gate_up','MoeGgufExpertParams','MoeTileDescriptor'],'5ecf1fe88676':['moe_group_routes','addGgufExperts','ComputeDispatch'],'c2600a40739a':['SOURCE','step','Inputs'],'3adb8002840a':['ModelConfig.gdnStateDtype','ModelConfig.cacheLayoutNamespace','ModelConfig.ssmCheckpointBytes']}
constraints={
'dce06d208cbc':'Offset BF16 KV views can have aligned row strides but an unaligned bound base. Preserve an aligned fast path and safe handling for unsupported layouts; lazy views require explicit treatment. The historical failure involved a six-byte base offset.',
'846428239de2':'A four-row prefill tail has routing-sensitive nearly tied logits despite a correct standalone attention formula. Preserve large-prefill behavior and validate all tail widths through eight.',
'fb2abb29e4ef':'Cover arbitrary context-length residues, compact cache formats and invalid selected rows without changing attention arithmetic or precision.',
'0658dac84541':'Repair the captured long-context prefill/decode correctness behavior, retaining host/kernel contracts and explicit testing limits. Source-reported performance is unavailable for this change.',
'8fcd61d6bf0d':'The scratch buffer must cover both stages of the dense kernel’s double-buffered use; preserve output dimensions and element precision.',
'540e6ae59a58':'The current one-pass gate/up layout depends on one simdgroup staging exactly one32-column tile in32-input steps. Host and kernel must agree on geometry.',
'9402155ae6ed':'Q4 prefill output and output-sum variants fail only with Metal shader validation on the reported macOS26.5/M5 Pro environment. Preserve both validation-enabled and ordinary behavior; no speedup is required.',
'f3a511e50aa3':'Partial GGUF prefill tiles must remain correct under concurrent GPU load. Preserve barrier convergence for all participating simdgroups, accepting the source-reported device-specific cost if required for correctness.',
'ab1256ccd614':'Keep the supported packed-width set and host/kernel argument layout synchronized. Preserve the existing floating-point operation order; no measured elapsed-time gain is supplied.',
'63ceff800643':'Remove only intermediates unused by serving and retain the state layout used by replay. The modified oracle has an explicitly reported BF16 rounding tolerance, not a bitwise identity claim.',
'7d60b965630a':'Respect the distinction between successful host command completion and intra-kernel ordering. Preserve live output/state checks and idle-lane memory checks.',
'7161da96eac6':'Preserve top-k/top-p tie and nucleus-boundary behavior. Shared-GPU microbench results are provisional and do not establish model-generation speed.'}
# Restore the complete templated Q4 helper whose changed attribute defeated automatic name matching.
blocked='splash-9402155ae6ed-hour-v3';rec=json.loads((R/'sourcepacks'/blocked/'complete-record.json').read_text());pin=next(x for x in json.loads((R/'source-pins-batch2.json').read_text())if x['head'].startswith('9402155'))
msg=(R/'sourcepacks'/blocked/'commit-message.txt').read_text();ann=lambda t:{'text':t,'basis':'source','evidence':[{'source':'commit_message','quote':msg}]}
c={'id':blocked,'repository':'splash','commit':pin['head'],'parent':pin['parent'],'source_group':'splash-related-optimization-training','protected':False,'outcome':'source_only','curation_outcome':'correctness','split':'train','task_conditioning':'historical_source','context_policy':'reviewed_operation','task_kind':{'value':'kernel','reviewed_by':'Codex complete template and host-ABI source review','evidence':[{'source':'commit_message','quote':msg.splitlines()[0]}]},'annotations':{'operation':ann('Prefill Q4 behavior under Metal shader validation'),'architecture':ann('Force the complete Q4 prefill helper inline while retaining its template parameters and kernel ABI. The author reports restored shader-validation behavior and neutral timing within the measured spread; this library has not compiled or executed it.'),'hardware':ann('M5 Pro/macOS26.5 and M5 Max/macOS27, as reported by the author'),'rationale':ann(msg)}}
p='runtime/metal/kernels/prefill/linear_q4.metal';pair=rec['files'][p];u={'path':p,'kind':'complete_function'}
for side in ['before','after']:
 b=pair[side]['content'].encode();start=b.index(b'template <ushort TileN, ushort Simdgroups, PrefillQ4Epilogue Epilogue>');tail=b[start:];code,pairs,_=lex(tail,'c');brace=code.index('{');end=start+pairs[brace]+1
 if b[end:end+1]==b'\n':end+=1
 u[side]={'start':start,'end':end,'sha256':sha(b[start:end])}
c['reviewed_units']={'reviewed_by':'Codex exact template/function body reviewed2026-10-13','dependency_review':'Complete templated q4_prefill helper; enum/type/helper definitions remain external repository context. No standalone compilation guarantee. All non-documentation edits replay.','units':[u]};m['candidates'].append(c)
for c in m['candidates']:
 short=c['commit'][:12];rec=json.loads((R/'sourcepacks'/c['id']/'complete-record.json').read_text());msg=(R/'sourcepacks'/c['id']/'commit-message.txt').read_text()
 if short in choices:
  selected=[o for o in options[c['id']]if o['name']in choices[short]];assert len(selected)==len(choices[short]);c['reviewed_units']['units']=[{k:o[k]for k in ['path','kind','before','after']}for o in selected]
 if short in ['080a62e2f2d7','9a5467375008','3dc1dab54e7b']:c['curation_outcome']='source_only';c['outcome']='source_only'
 # Keep data and claims connected to complete captured code as well as commit text.
 ev='diff-'+short;fp=R/'sourcepacks'/c['id']/'commit.diff';m['evidence'][ev]={'path':str(fp.relative_to(R)),'sha256':sha(fp.read_bytes()),'url':m['repositories'][c['repository']]['url']+'/commit/'+c['commit'],'phase':'post_change'}
 c['annotations']['architecture']['evidence'].append({'source':ev,'quote':rec['diff']})
 if short in constraints:c['annotations']['constraints']={'text':constraints[short],'basis':'source','evidence':[{'source':'commit_message','quote':msg},{'source':ev,'quote':rec['diff']}]}
 c['reviewed_units']['reviewed_by']='Codex quality_research complete-unit source inspection2026-10-13'
 c['reviewed_units']['dependency_review']='Complete named baseline functions/declarations reviewed at exact byte ranges. Additional patch anchors disclose all required old lines. External imported helpers, typedefs, state structs and toolchain remain repository dependencies. Full non-documentation edits are retained and replayed; no standalone compile or semantic parity claim.'
 units=c['reviewed_units']['units'];units.sort(key=lambda u:(u['path'],u['before']['start'],u['after']['start']));omitted=[]
 for path,pair in rec['files'].items():
  b=(pair['before']or{'content':''})['content'].encode();a=(pair['after']or{'content':''})['content'].encode();bc=ac=0
  for u in [x for x in units if x['path']==path]:
   if b[bc:u['before']['start']]!=a[ac:u['after']['start']]:omitted.append(path)
   bc=u['before']['end'];ac=u['after']['end']
  if b[bc:]!=a[ac:]:omitted.append(path)
 c['operation_scope']={'description':'Retrospective task with complete selected baseline definitions and target-selected old edit anchors; all non-documentation code/build/test changes remain in the answer. This is historical edit imitation, not a no-leakage autonomous optimization benchmark.','omitted_path_reasons':{p:'Other repository context is not expanded; all old patch anchors and all non-documentation target changes are retained.'for p in sorted(set(omitted))}}
 c['task_kind']['reviewed_by']='Codex quality_research source inspection2026-10-13'
 c['curriculum_eligibility']='reasoning_or_archive_only'if c['curation_outcome']in['neutral','tradeoff','rejected','exploratory']else'exact_edit_candidate'
 reconstruct_candidate(c,m['repositories'],_evidence(R,m['evidence'])[0])
(R/'manifest-review-candidate.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
print(len(m['candidates']),'reviewed source-context candidates')
