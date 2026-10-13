from pathlib import Path
import json,hashlib,sys,ast,re,shutil
R=Path(__file__).resolve().parent;B=R/'batch2';B.mkdir(exist_ok=True);W=R.parents[2];sys.path.insert(0,str(W/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
sha=lambda b:hashlib.sha256(b).hexdigest()
draft=json.loads((R/'draft-manifest.json').read_text());selection=json.loads((R/'selection-batch2.json').read_text());heads={x['commit']for x in selection['candidates']};m={k:v for k,v in draft.items()if k!='candidates'};m['candidates']=[c for c in draft['candidates']if c['commit']in heads];m['evidence']={};opts={x['case_id']:x['options']for x in json.loads((R/'scanned-context-options-full.json').read_text())};pin={x['head']:x for x in json.loads((R/'source-pins-batch2.json').read_text())};(B/'evidence').mkdir(exist_ok=True)
for f in ['protection.json','reserved-task-contracts.json']:shutil.copyfile(R/'evidence'/f,B/'evidence'/f)
selection_map={'a338ca2d2bef':['QKV_MPP_PAGE','QKV_MPP_KERNEL_SOURCE','qkvMppConfig'],'748fab8a2e3a':['QSA_ATTN_Q_SOURCE','QSA_ATTN_MERGE_SOURCE','qsaSparseAttn'],'7de9af0f3023':['Generator.mtpGroupVerify','runBatchedMtpHeadTick','Transformer.flushDeferredPleGroup'],'96578478da83':['Transformer.qsaMaskFromQk','Transformer.qsaMaskBatched'],'fde5bcfc26e6':['ldlq_group_mlx'],'376c23c90e5c':['launch_tiles_for','_search_mlx'],'a939818fb48f':['quantize_prepared_bank','weighted_rel_err'],'0382aee97906':['public_from_mxfp4','dequant_mxfp4','e2m1_value','e8m0_value']}
aux={'fde5bcfc26e6','376c23c90e5c','a939818fb48f','0382aee97906'}
def pyunits(text):
 raw=text.encode();lines=raw.splitlines(keepends=True);off=[0]
 for l in lines:off.append(off[-1]+len(l))
 return {n.name:{'start':off[min([n.lineno]+[d.lineno for d in n.decorator_list])-1],'end':off[n.end_lineno],'sha256':sha(raw[off[min([n.lineno]+[d.lineno for d in n.decorator_list])-1]:off[n.end_lineno]])}for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
reviews=[];policy={}
for c in m['candidates']:
 cid=c['id'];h=c['commit'];short=h[:12];rec=json.loads((R/'sourcepacks'/cid/'complete-record.json').read_text());units=[]
 for name in selection_map[short]:
  hits=[o for o in opts[cid]if o.get('name')==name]
  if hits:
   assert len(hits)==1;u={k:hits[0][k]for k in ['path','kind','before','after']}
  else:
   matches=[]
   for path,pair in rec['files'].items():
    if not path.endswith('.py') or not pair['before'] or not pair['after']:continue
    b=pyunits(pair['before']['content']);a=pyunits(pair['after']['content'])
    if name in b and name in a:matches.append({'path':path,'kind':'complete_function','before':b[name],'after':a[name]})
   assert len(matches)==1,(cid,name);u=matches[0]
  units.append(u)
 units.sort(key=lambda u:(u['path'],u['before']['start'],u['after']['start']));omitted=[]
 for path,pair in rec['files'].items():
  b=(pair['before']or{'content':''})['content'].encode();a=(pair['after']or{'content':''})['content'].encode();bc=ac=0
  for u in [v for v in units if v['path']==path]:
   if b[bc:u['before']['start']]!=a[ac:u['after']['start']]:omitted.append(path)
   bc=u['before']['end'];ac=u['after']['end']
  if b[bc:]!=a[ac:]:omitted.append(path)
  if Path(path).suffix.lower() in ['.md','.rst']:
   assert pair['after'];policy.setdefault(cid,{})[path]={'before_sha256':sha(b)if pair['before']else None,'after_sha256':sha(a),'reason':'Documentation-only source file individually reviewed; raw historical changes retained as evidence. All compute/build/test and host edits remain in target.'}
 c['reviewed_units']={'reviewed_by':'Codex quality_research complete source-unit review 2026-10-13','dependency_review':'Complete named baseline functions/declarations, Python AST boundaries or exact native source units. Imports, generated templates, constants and external modules are repository dependencies; not a standalone compilation claim. Full non-documentation target edits include tests and dispatch.','units':units}
 c['operation_scope']={'description':'Historical source-conditioned one-edit optimization task. Complete operation-bearing baseline units; all non-documentation commit edits retained in exact replay.','omitted_path_reasons':{p:'Other repository definitions, registration, tests or documentation are not expanded in operation context; full files and every non-documentation target edit remain available.'for p in sorted(set(omitted))}}
 c['task_kind']['value']='integration' if short in aux else 'host_dispatch' if short=='7de9af0f3023' else 'kernel';c['task_kind']['reviewed_by']='Codex quality_research actual implementation review 2026-10-13'
 c['curation_role']='auxiliary_conversion_compute' if short in aux else 'native_kernel_or_dispatch'
 protected=(B/'evidence/protection.json').read_text();hits=[v for v in [h,c['parent']]if v in protected or v[:12]in protected]
 c['historical_protection_review']={'status':'pending'if hits else 'cleared','reviewed_by':'Codex quality_research metadata and operation reviewer 2026-10-13','reason':'Finite metadata head/parent identity and exact operation review. EXL3 conversion, MiMo packed attention and MTP command dispatch remain related source training lineages. No independent-family or prior protected Q8/HC/attention evaluation credit. No exact protected identity match.'if not hits else 'Identity match needs further review; not admitted.','evidence_sha256':{'evidence/'+f:sha((B/'evidence'/f).read_bytes())for f in ['protection.json','reserved-task-contracts.json']}}
 for i,lic in enumerate(pin[h]['licenses']):
  src=R/lic['path'];dst=B/'evidence'/Path(lic['path']).name;shutil.copyfile(src,dst);assert sha(dst.read_bytes())==lic['sha256'];m['evidence']['license_'+short+'_'+str(i)]={'path':'evidence/'+dst.name,'sha256':lic['sha256'],'url':pin[h]['url']+'/blob/'+h+'/'+lic['source_path'],'phase':'unknown'}
 rec2=reconstruct_candidate(c,m['repositories'],_evidence(B,m['evidence'])[0]);reviews.append({'case_id':cid,'role':c['curation_role'],'stable_patch_id':rec2['stable_patch_id'],'selected_units':units,'historical_identity_hits':hits,'source_record_sha256':sha((R/'sourcepacks'/cid/'complete-record.json').read_bytes()),'source_record_path':str((R/'sourcepacks'/cid/'complete-record.json').resolve()),'source_reported_only':True})
m['notes']=['Second separate source pack; first25 held artifacts remain unchanged. Four entries are auxiliary conversion compute/host tasks, not native Metal-kernel examples.','Retrospective source-conditioned imitation, not recorded agent trajectories. Preserve mixed outcomes/neutral windows in full source messages; no local candidate execution or performance reproduction.'];m['protected_keys']=['repository:https://github.com/ggml-org/llama.cpp'];m.pop('manifest_sha256',None);m['manifest_sha256']=sha(json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
for name,x in [('manifest-reviewed.json',m),('documentation-omissions.json',policy),('source-unit-review.json',{'format':'expansion-v2-group-a-batch2-source-review-v1','cases':reviews})]:(B/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print('batch2 reviewed',len(reviews))
