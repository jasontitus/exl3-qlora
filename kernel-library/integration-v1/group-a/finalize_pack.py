"""Offline reviewed source-unit selection and exact full-edit replay input manifest."""
from pathlib import Path
import ast, hashlib, json, re, subprocess, sys
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
sha=lambda b:hashlib.sha256(b).hexdigest()
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
# Reuse locator definitions only, without the earlier draft creation loop.
ns={};code=(ROOT/'review_units.py').read_text();exec(code[code.index('def body('):code.index("for c in m['candidates']:")],{'re':re,'ast':ast,'sha':sha},ns)
body=ns['body'];unit=ns['unit']
def pyunit(src,name):
 node=next(n for n in ast.walk(ast.parse(src)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 lines=src.splitlines(keepends=True); start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
 return len(''.join(lines[:start]).encode()),len(''.join(lines[:node.end_lineno]).encode())
def ann(text,quote):return {'text':text,'basis':'source','evidence':[{'source':'commit_message','quote':quote}]}
m=json.loads((ROOT/'manifest-scoped-draft.json').read_text());m.pop('manifest_sha256');m['candidates']=[c for c in m['candidates'] if c['repository']!='vllm-metal']
# Remove only an unchanged complete wrapper from prompt expansion, not any target edit.
om=next(c for c in m['candidates'] if c['repository']=='omlx');om['reviewed_units']['units']=[u for u in om['reviewed_units']['units'] if u['before']['start']!=7884]
for ident in ['splash','vllm-metal']:
 pin=json.loads((ROOT/f'{ident}-compact-source-pin.json').read_text()); msg=subprocess.check_output(['git','--git-dir='+str(ROOT/'mirrors'/f'{ident}.git'),'show','-s','--format=%B',pin['head']],text=True).rstrip('\n'); title=msg.splitlines()[0]
 c={'id':f'{ident}-{pin["head"][:12]}-integration-v1','repository':ident,'commit':pin['head'],'parent':pin['parent'],'source_group':ident+'-upstream-optimization-lineage','protected':False,'outcome':'reported_improvement','split':'train','task_conditioning':'historical_source','context_policy':'complete_changed_files','task_kind':{'value':'host_dispatch' if ident=='vllm-metal' else 'kernel','reviewed_by':'Codex source review 2026-10-13','evidence':[{'source':'commit_message','quote':title}]},'annotations':{'operation':ann(title,title),'rationale':ann(msg,msg)}}
 if ident=='vllm-metal':
  c['annotations'].update(hardware=ann('Apple M5 Pro48GB, source-reported serving and layer benchmarks; not locally reproduced.','Apple M5 Pro 48 GB'),constraints=ann('Shared latent K/V heads, causal masks, dense/4-bit projections. Preserve per-request fallback; source reports neutral low-concurrency controls and largest batched output difference2e-4.','On the\nbatched path the largest output difference from main is 2e-4.'),architecture=ann('Fold query heads into the query row dimension for one shared SDPA head, including PE score matmul/mask; reshape output back.','the query and the PE score mask as `[b, 1, nheads * q_len, ...]`'))
 else:
  c['annotations'].update(hardware=ann('M5 Max,544-row27B prefill; source-reported1.7%/0.4% improvement with unchanged outputs, not independently reproduced.','On the M5 Max that 544-row prefill'),architecture=ann('Group up to three fused-projection segments by quantization format into one prefill dispatch; retain epilogue and plane-input paths.','Linear runs one dispatch per format of a projection'))
 m['candidates'].append(c)
e=_evidence(ROOT,m.get('evidence',{}))[0]
for c in m['candidates']:
 # Reconstruct old candidates without context for stable full files; refresh gap declarations below.
 bare={k:v for k,v in c.items() if k not in ['reviewed_units','operation_scope']};bare['context_policy']='complete_changed_files'
 rec=reconstruct_candidate(bare,m['repositories'],e); units=c.get('reviewed_units',{}).get('units',[])
 if c['commit']==json.loads((ROOT/'splash-compact-source-pin.json').read_text())['head']:
  units=[];path='runtime/metal/abi/Gguf.h'
  # Complete ABI header; macro-family section begins at a complete #define and ends after its #undefs.
  units.append(unit(rec,path,'whole_file',[(0,len(rec['files'][path][s]['content'].encode())) for s in ['before','after']]))
  path='runtime/metal/kernels/shared/gguf_linear.metal';bounds=[]
  for s in ['before','after']:
   src=rec['files'][path][s]['content'];start=len(src[:src.index('#define GGUF_SEGMENT(i')].encode());bounds.append((start,len(src.encode())))
  units.append(unit(rec,path,'complete_declaration',bounds))
  path='runtime/ops/LinearGguf.cpp';units.append(unit(rec,path,'complete_function',[body(rec['files'][path][s]['content'],'void Linear::addGgufPrefill(') for s in ['before','after']]))
 elif c['repository']=='vllm-metal':
  units=[];path='vllm_metal/attention/impls/mla.py';units.append(unit(rec,path,'complete_function',[pyunit(rec['files'][path][s]['content'],'_apply_absorbed_mla_attention') for s in ['before','after']]))
  path='tests/test_mla_materialized_prefill.py';before=rec['files'][path]['before']['content'];after=rec['files'][path]['after']['content'];units.append(unit(rec,path,'complete_function',[(len(before.encode()),len(before.encode())),pyunit(after,'test_absorbed_attention_folds_query_heads')]))
  # Existing test factory is explicit dependency context; its imports/constants remain repository context.
  units.insert(0,unit(rec,path,'complete_function',[pyunit(rec['files'][path][s]['content'],'_make') for s in ['before','after']]))
 omitted=[]
 for path,pair in rec['files'].items():
  b=(pair['before'] or {'content':''})['content'].encode();a=(pair['after'] or {'content':''})['content'].encode();bc=ac=0
  for u in sorted([u for u in units if u['path']==path],key=lambda u:(u['before']['start'],u['after']['start'])):
   ub,ua=u['before'],u['after']
   if b[bc:ub['start']]!=a[ac:ua['start']]:omitted.append(path)
   bc,ac=ub['end'],ua['end']
  if b[bc:]!=a[ac:]:omitted.append(path)
 c['context_policy']='reviewed_operation';c['reviewed_units']={'reviewed_by':'Codex group-a complete named-unit review 2026-10-13','dependency_review':'Complete named baseline declarations, ABI/macro family or Python method/test factory. Surrounding imports/constants, type implementations and integration context remain external repository dependencies. This is a synthetic historical full-edit example, not a standalone compilation unit or genuine recorded agent trajectory.','units':units}
 c['operation_scope']={'description':'Complete named operation context with exact source ranges. The answer retains every non-documentation commit edit, including host dispatch, tests and build changes. Full repository files and raw patch remain in provenance.','omitted_path_reasons':{p:'Additional repository declarations, surrounding comments or registration/build/test context are not expanded in the prompt; exact baseline anchors and every non-documentation edit remain in the answer.' for p in sorted(set(omitted))}}
 rec=reconstruct_candidate(c,m['repositories'],e)
 out=ROOT/'sourcepacks'/c['id'];out.mkdir(exist_ok=True,parents=True);(out/'complete-record.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');(out/'commit.diff').write_text(rec['diff'])
 for path,pair in rec['files'].items():
  for s in ['before','after']:
   if pair[s]:p=out/s/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(pair[s]['content'])
# Preserve metadata and its limits, rather than interpreting old retrieval admission as a train authorization rule.
protection_sources=[WORK/'kernel-sft/ktune-snapshot/library/optimizations.json',WORK/'loss-fix/apple/artifacts/loss-fix/upstream-inventory/ktune_protection_metadata.json',ROOT/'evidence/reserved-metadata.json']
protection={'format':'group-a-historical-protection-review-v1','reviewed_by':'Codex group-a source reviewer','sources':[],'limits':['Finite metadata/source-identity review, not proof of semantic or pretraining independence.','Whole llama.cpp held-out protection remains. DS4 Q8 batch versus reserved Q8-prefill lineage is unresolved and quarantined.','Existing protected assignments are not cleared or changed by this intake.']}
for p in protection_sources:
 protection['sources'].append({'source_path':str(p),'sha256':sha(p.read_bytes()),'metadata':json.loads(p.read_text())})
pp=ROOT/'evidence/protection.json';pp.write_text(json.dumps(protection,indent=2,ensure_ascii=False)+'\n');ph=sha(pp.read_bytes())
review=json.loads((ROOT/'evidence/reserved-metadata.json').read_text())['review'];metadata_text=json.dumps(review,sort_keys=True)
for c in m['candidates']:
 hit=c['commit'] in metadata_text or c['parent'] in metadata_text
 if hit:raise ValueError('Explicit reserved commit match: '+c['id'])
 reason={'splash':'Known reserved metadata has no commit/parent identity match. Reviewed vision attention and GGUF segmented-dispatch edits are repository-specific; no copied reserved unit is asserted. Matrix attention resemblance alone is not independent-family proof.','omlx':'Known reserved metadata has no commit/parent identity match. Reviewed TurboQuant MSE packed-KV matrix attention implementation/patch routing is distinct from listed Q8-prefill, HC, residual, fp8-rounding, MoE and indexer task contracts. Finite source review does not certify semantic independence.','vllm-metal':'Known reserved metadata has no commit/parent identity match. This Python absorbed-MLA head-axis reshape and tests use existing SDPA; no protected Metal kernel implementation is copied. Source report references same-repo908/913/927, not reserved MLX/llama PR lineages.','ds4':'Reserved metadata explicitly includes Q8-prefill. This new Q8 batch GEMM may differ, but shared operation/lineage equivalence is not resolved from current metadata; row remains quarantined.'}[c['repository']]
 c['historical_protection_review']={'status':'pending' if c['repository']=='ds4' else 'cleared','reviewed_by':'Codex group-a metadata and exact-edit reviewer 2026-10-13','reason':reason,'evidence_sha256':{'evidence/protection.json':ph}}
policy={}
for c in m['candidates']:
 rec=json.loads((ROOT/'sourcepacks'/c['id']/'complete-record.json').read_text())
 for path,pair in rec['files'].items():
  if Path(path).suffix.lower() in ['.md','.rst']:
   policy.setdefault(c['id'],{})[path]={'before_sha256':sha(pair['before']['content'].encode()) if pair['before'] else None,'after_sha256':sha(pair['after']['content'].encode()),'reason':'Reviewer-confirmed documentation-only prose describing dispatch behavior; code, ABI, host dispatch, tests and build edits remain complete. Raw document edit remains in provenance.'}
(ROOT/'documentation-omissions.json').write_text(json.dumps(policy,indent=2)+'\n')
m['notes'].append('Protection status is an explicit curator assertion with finite captured metadata. Pending DS4 is not eligible for training. Source-reported gains remain unverified locally.')
m['manifest_sha256']=sha(canon(m));(ROOT/'manifest-reviewed.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
print('SEALED',m['manifest_sha256'],len(m['candidates']))
