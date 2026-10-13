from pathlib import Path
import json,hashlib,sys,importlib.util
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path[:0]=[str(W/'loss-fix/apple/src'),str(W/'kernel-library/expansion-v2/tools')]
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
from source_units import lex,scan_units
sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((R/'draft-manifest.json').read_text());options={x['case_id']:x['options']for x in json.loads((R/'context-options.json').read_text())}
choices={'497d71d94eed':['buildConfig','project'],'fa76a4b50b3f':['Transformer.conv1dWithCache','KVCache.evalState','prefillStreamBytesPerToken'],'84f2a6d74241':['Transformer.mimoAttnWith','fusedSdpa256Masked','prefillHeadDimFused'],'9c511c4710d9':['K1_SOURCE','gdnGateChain','ModelConfig.ssmStateDtype']}
# A test-quality task deliberately selects complete test routines, not production-only defaults.
short='348d6e4b76dd';case='splash-'+short+'-hour-v4';notes=json.loads((R/'curation-notes.json').read_text());pin=next(x for x in json.loads((R/'source-pins-batch2.json').read_text())if x['head'].startswith(short));rec=json.loads((R/'sourcepacks'/case/'complete-record.json').read_text());msg=(R/'sourcepacks'/case/'commit-message.txt').read_text();ann=lambda text:{'text':text,'basis':'source','evidence':[{'source':'commit_message','quote':msg}]}
if not any(c['id']==case for c in m['candidates']):
 selected=[o for o in options[case]if o['name']in ['splitCase','runSplitPlan']];assert len(selected)==2
 c={'id':case,'repository':'splash','commit':pin['head'],'parent':pin['parent'],'source_group':'splash-related-optimization-training','protected':False,'outcome':'source_only','curation_outcome':'correctness','split':'train','task_conditioning':'historical_source','context_policy':'reviewed_operation','task_kind':{'value':'integration','reviewed_by':'Codex source test-oracle inspection','evidence':[{'source':'commit_message','quote':msg.splitlines()[0]}]},'annotations':{k:ann(notes[short][v])for k,v in [('operation','operation'),('task','task'),('architecture','solution')]},'reviewed_units':{'units':[{k:o[k]for k in ['path','kind','before','after']}for o in selected]},'lineage':'patch:'+rec['stable_patch_id']}
 c['annotations']['rationale']=ann(msg);m['candidates'].append(c)
for c in m['candidates']:
 short=c['commit'][:12];rec=json.loads((R/'sourcepacks'/c['id']/'complete-record.json').read_text());msg=(R/'sourcepacks'/c['id']/'commit-message.txt').read_text()
 if short in choices:
  selected=[o for o in options[c['id']]if o['name']in choices[short]];assert len(selected)==len(choices[short]),(short,len(selected));c['reviewed_units']['units']=[{k:o[k]for k in ['path','kind','before','after']}for o in selected]
 if short=='41896655a01a':
  path='src/transformer.zig';u={'path':path,'kind':'complete_function'}
  for side in ['before','after']:
   b=rec['files'][path][side]['content'].encode();matches=[x for x in scan_units(b,'zig')['units']if x.get('qualified_name')=='Transformer.gatedDeltaNetProjected'];assert len(matches)==1;u[side]={k:matches[0][k]for k in ['start','end','sha256']}
  c['reviewed_units']['units']=[u]
 if short in ['f26aaa4ed836','9cf2ea3dbde2']:
  units=[]
  manual=[('runtime/ops/GDN.cpp','PreparedInput GDN::addDecode('),('runtime/ops/PagedAttention.cpp','PreparedInput PagedAttention::addVerifyGate('),('runtime/ops/Linear.cpp','LinearInput LinearPlan::input()')] if short=='f26aaa4ed836' else [('runtime/ops/Linear.cpp','LinearScratchSize LinearPlan::scratchSize()'),('runtime/ops/Linear.cpp','uint32_t LinearPlan::partialSums()'),('runtime/ops/Linear.cpp','std::optional<LinearSimdgroups> fixedSimdgroups(')]
  for path,marker in manual:
   u={'path':path,'kind':'complete_function'}
   for side in ['before','after']:
    b=rec['files'][path][side]['content'].encode();assert b.count(marker.encode())==1;start=b.index(marker.encode());code,pairs,_=lex(b,'c');brace=code.index('{',start);end=pairs[brace]+1
    if b[end:end+1]==b'\n':end+=1
    u[side]={'start':start,'end':end,'sha256':sha(b[start:end])}
   units.append(u)
  c['reviewed_units']['units']=units
 ev='diff-'+short;fp=R/'sourcepacks'/c['id']/'commit.diff';m['evidence'][ev]={'path':str(fp.relative_to(R)),'sha256':sha(fp.read_bytes()),'url':m['repositories'][c['repository']]['url']+'/commit/'+c['commit'],'phase':'post_change'}
 c['annotations']['architecture']['evidence'].append({'source':ev,'quote':rec['diff']})
 c['annotations']['constraints']={'text':c['annotations']['task']['text'],'basis':'source','evidence':[{'source':'commit_message','quote':msg},{'source':ev,'quote':rec['diff']}]}
 c['reviewed_units']['reviewed_by']='Codex quality_research complete source-boundary and task-context review2026-10-13'
 c['reviewed_units']['dependency_review']='Complete named baseline functions/declarations at exact byte ranges. Additional target-selected patch anchors show every required old line. Imported helpers, typedefs, surrounding state and toolchain remain repository dependencies. Complete non-documentation edits are replayed, with no standalone compilation or semantic-parity claim.'
 units=c['reviewed_units']['units'];units.sort(key=lambda u:(u['path'],u['before']['start'],u['after']['start']));omitted=[]
 for path,pair in rec['files'].items():
  b=(pair['before']or{'content':''})['content'].encode();a=(pair['after']or{'content':''})['content'].encode();bc=ac=0
  for u in [x for x in units if x['path']==path]:
   if b[bc:u['before']['start']]!=a[ac:u['after']['start']]:omitted.append(path)
   bc=u['before']['end'];ac=u['after']['end']
  if b[bc:]!=a[ac:]:omitted.append(path)
 c['operation_scope']={'description':'Historical task reconstruction: complete selected baseline definitions plus target-selected old edit anchors. Every non-documentation code/build/test change remains in the answer. This is edit imitation with hindsight-selected locations, not a captured agent rollout or independent autonomous optimization benchmark.','omitted_path_reasons':{p:'Other repository context not expanded; all old patch anchors and every non-documentation target change remain.'for p in sorted(set(omitted))}}
 c['task_kind']['reviewed_by']='Codex quality_research source inspection2026-10-13'
 c['curriculum_eligibility']='reasoning_or_archive_only'if c['curation_outcome']in['neutral','tradeoff','rejected','exploratory']else'exact_edit_candidate'
 reconstruct_candidate(c,m['repositories'],_evidence(R,m['evidence'])[0])
(R/'manifest-review-candidate.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
print(len(m['candidates']),'reviewed source-context candidates')
