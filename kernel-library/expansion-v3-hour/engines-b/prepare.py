"""CPU-only exact historical edits; no upstream execution. Separate curriculum tiers."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'loss-fix/apple/src'))
from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import load_tokenizer,tokenize_row,tokenizer_identity
from exl3_mlx.upstream_data import FORMAT,_evidence,reconstruct_candidate

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prior=module('prior',WORK/'kernel-library/integration-v1/group-b/prepare_group_b.py')
units=module('refined',WORK/'kernel-library/expansion-v2/group-b/refined_units.py')
source_units=module('source_units',WORK/'kernel-library/expansion-v2/tools/source_units.py')
protect=module('exact',WORK/'kernel-library/expansion-v2/group-b/batch2/exact_protection.py')
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda v:json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
URLS={'omlx':'https://github.com/jundot/omlx','mtplx':'https://github.com/youssofal/MTPLX','vllm-metal':'https://github.com/vllm-project/vllm-metal','tensorfold':'https://github.com/ashhart/TensorFold'}
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def git(r,*args):
 return subprocess.run(['git','-c','core.hooksPath=/dev/null','-C',str(ROOT/'mirrors'/(r+'.git')),*args],capture_output=True,check=True).stdout

def main():
 decisions=json.loads((ROOT/'curation.json').read_text());rows=json.loads((ROOT/'selection.json').read_text())
 ep=WORK/'kernel-library/integration-v1/group-a/evidence/reserved-metadata.json';shutil.copyfile(ep,ROOT/'evidence/reserved-metadata.json');idx=protect.reserved_index(json.loads(ep.read_text()))
 cp=WORK/'kernel-library/integration-v1/group-a/evidence/reserved-task-contracts.json';shutil.copyfile(cp,ROOT/'evidence/reserved-task-contracts.json')
 binding={'evidence/reserved-metadata.json':H(ep.read_bytes()),'evidence/reserved-task-contracts.json':H(cp.read_bytes())}
 evidence={}
 for r in rows:
  p=ROOT/'evidence'/f"{r['repo']}-{r['sha'][:12]}-message.txt";actual=git(r['repo'],'show','-s','--format=%B',r['sha']).decode();p.write_text(actual)
  evidence[r['sha'][:12]]={'path':str(p.relative_to(ROOT)),'sha256':H(p.read_bytes()),'phase':'post_change','url':URLS[r['repo']]+'/commit/'+r['sha']}
 for p in (ROOT/'evidence').glob('*-pr-*.txt'):
  repo,n=p.stem.rsplit('-pr-',1);evidence[p.stem]={'path':str(p.relative_to(ROOT)),'sha256':H(p.read_bytes()),'phase':'post_change','url':URLS[repo]+'/pull/'+n}
 sources,_=_evidence(ROOT,evidence);repos={r:{'path':str(ROOT/'mirrors'/(r+'.git')),'url':url} for r,url in URLS.items()}
 tokpath='/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw';tok=load_tokenizer(tokpath)
 candidates=[];audit=[];docs={};typed=collections.defaultdict(list)
 for r in rows:
  key=r['sha'][:8];tier,kind,operation,rationale,status=decisions[key];ev=r['sha'][:12];quote=(ROOT/evidence[ev]['path']).read_text()
  # Evidence helper returns exact decoded text; do not manufacture a benchmark.
  quote=(ROOT/evidence[ev]['path']).read_text()
  ann=lambda text:{'text':text,'basis':'source','evidence':[{'source':ev,'quote':quote}]}
  c={'id':f"{r['repo']}-{r['sha'][:12]}-hour-v3",'repository':r['repo'],'commit':r['sha'],'parent':r['parents'][0],'outcome':'reported_improvement' if tier=='measured_tradeoff' else 'source_only','protected':False,'source_group':'tensorfold-engine-history' if r['repo']=='tensorfold' else 'mlx-derived-engine-history','task_conditioning':'historical_source','task_kind':{'value':kind,'reviewed_by':'adversarial_process','evidence':[{'source':ev,'quote':quote}]},'annotations':{'operation':ann(operation),'architecture':ann(rationale)}}
  hits=protect.identity_matches(c,idx)
  if hits:status='protected'
  reason='Exact protected commit/parent identity gate precedes manual clearance. No exact match. Finite task review: this edit concerns '+operation+'; no reserved HC gate/mix, DeepSeek FP8 rounding/index radix/down-reduction, or Gemma/Laguna implementation is intentionally admitted. Common framework and source ancestry remain; no independent-lineage efficacy claim.'
  if status=='pending':reason='Pending finite operation/lineage review; retained as quarantined source, not training admission: '+rationale
  c['historical_protection_review']={'status':status,'reviewed_by':'adversarial_process','reason':reason,'evidence_sha256':binding}
  case={'id':c['id'],'repo':r['repo'],'commit':r['sha'],'parent':r['parents'][0],'curriculum_tier':tier,'protection_status':status,'protected_exact_hits':hits,'execution_verified':False}
  if hits:audit.append(case);continue
  try:
   rec=reconstruct_candidate(c,repos,sources)
   uu=[]
   for p,pair in rec['files'].items():
    uu.extend(units.python_units(p,pair) if p.endswith('.py') and pair['before'] and pair['after'] else prior.units(p,pair))
   c['context_policy']='reviewed_units';c['reviewed_units']={'reviewed_by':'adversarial_process','dependency_review':'Complete changed Python methods/functions or whole changed native/host/build files. Exact whole-file replay requires every edit. Dependencies remain pinned in repository; no standalone compilation closure.','units':uu}
   if key == 'c0bb87da':
    selected={'zig/kernels/metal/nax.h','zig/kernels/metal/prefill/fn_attn.metal','zig/src/families/flashnext/frags.zig'}
    c['context_policy']='reviewed_operation'
    c['reviewed_units']['units']=[u for u in uu if u['path'] in selected]
    c['reviewed_units']['dependency_review']='Complete nax fragment-layout header and sparse attention kernel, plus the new standalone fragment-layout checker. Every other native/host/test edit remains in exact target and original anchors; full sources are sidecars. Not standalone compilation closure.'
    c['operation_scope']={'description':'Fragment-layout repair with all original implementation and test edits preserved.', 'omitted_path_reasons':{p:'Other changed host/test/kernel context is represented by exact baseline anchors; complete code and full target edits retained.' for p in rec['files'] if p not in selected}}
   if key in {'14641c80','6128a0c1'}:
    selected=({'zig/kernels/metal/train.metal','zig/kernels/metal/train_mixers.metal'} if key=='14641c80' else {'zig/kernels/metal/nemotron_sample.metal','zig/src/core/lanes/gpu_full.zig','zig/src/core/lanes/gpu_full_test.zig','zig/tests/nemotron_sample_check.zig'})
    c['context_policy']='reviewed_operation';c['reviewed_units']['units']=[u for u in uu if u['path'] in selected]
    c['reviewed_units']['dependency_review']='Complete core Metal backward kernels, or complete new full-vocabulary sampler/host/reference/checker units. Every other native/host/test/build edit remains in the target with original baseline anchors and full source sidecars. Not standalone compilation closure.'
    c['operation_scope']={'description':'Complete core operation context with all implementation and test changes retained.', 'omitted_path_reasons':{p:'Remaining changed host/build/profile sections use original edit anchors with full target edits and full before/after sidecars.' for p in rec['files'] if p not in selected}}
   if key in {'5768a522','e0df8e31'}:
    specs={'5768a522':{'zig/tests/flashnext_run.zig':['glue_source','GSelect','DepthRule','Model.pleIdsGpu']},'e0df8e31':{'zig/kernels/metal/prefill/qmm_nax.metal':['qmm_dequant','qmm_k_loop','expert_tile','gather_qmm_rhs_nax'],'zig/src/families/nemotron/prefill_mixers.zig':['moe'],'zig/src/families/nemotron/prefill_launch.zig':['Launch.gatherQmm']}}[key]
    chosen=[]
    for name,names in specs.items():
     pair=rec['files'][name]
     for unit_name in names:
      unit={'path':name,'kind':'whole_file' if unit_name is None else 'complete_declaration'}
      for side in ('before','after'):
       data=(pair[side] or {'content':''})['content'].encode()
       found=source_units.named_unit(data,'c' if name.endswith('.metal') else 'zig',unit_name) if unit_name else {'start':0,'end':len(data)}
       a,b=found['start'],found['end'];unit[side]={'start':a,'end':b,'sha256':H(data[a:b])}
      chosen.append(unit)
    chosen.sort(key=lambda u:(u['path'],u['before']['start']))
    c['context_policy']='reviewed_operation';c['reviewed_units']['units']=chosen
    c['reviewed_units']['dependency_review']='Explicit complete core declarations: embedded Metal glue/GSelect/DepthRule/PLE dispatch for GPU rounds, or full gather kernel plus complete MoE/gather host functions. Every other change remains in exact target with original anchors and complete sidecars; not standalone compilation closure.'
    omitted=[]
    for name,pair in rec['files'].items():
     old=(pair['before'] or {'content':''})['content'].encode();new=(pair['after'] or {'content':''})['content'].encode();bc=ac=0
     for unit in [u for u in chosen if u['path']==name]:
      b,a=unit['before'],unit['after']
      if old[bc:b['start']] != new[ac:a['start']]:omitted.append(name)
      bc,ac=b['end'],a['end']
     if old[bc:] != new[ac:]:omitted.append(name)
    c['operation_scope']={'description':'Complete core operation context for a full historical edit, preserving every source/test/build target change.', 'omitted_path_reasons':{name:'Non-expanded sections retain original edit anchors and exact full target edits; full before/after sidecars remain available.' for name in sorted(set(omitted))}}
   rec=reconstruct_candidate(c,repos,sources)
   docs[c['id']]={p:{'before_sha256':v['before']['sha256'] if v['before'] else None,'after_sha256':v['after']['sha256'] if v['after'] else None,'reason':'Documentation claims retained as source evidence, not taught as executable edits; all code/test/build edits preserved.'} for p,v in rec['files'].items() if Path(p).suffix.lower() in {'.md','.rst'}}
   dest=ROOT/'drafts'/c['id'];dest.mkdir(parents=True,exist_ok=True);save(dest/'source.json',rec)
   licenses=[]
   for name in ('LICENSE','LICENSE.md','LICENSE.txt','COPYING'):
    try:b=git(r['repo'],'show',f"{r['sha']}:{name}")
    except subprocess.CalledProcessError:continue
    lp=ROOT/'evidence/licenses'/f"{c['id']}-{name}";lp.parent.mkdir(exist_ok=True);lp.write_bytes(b);licenses.append({'path':str(lp.relative_to(ROOT)),'sha256':H(b),'commit':r['sha'],'upstream_path':name})
   raw,replay=agent_example(rec,docs[c['id']])
   if tier == 'correctness':
    first=raw['messages'][1]['content'].split('\n\n',1)
    first[0]='Repair the correctness issue in '+operation+' and make the supporting repository changes. Preserve the intended operation semantics and existing supported cases, and reject or fall back for unsupported geometry. Retain the necessary host plumbing and regression tests using exec_command and git apply. Do not claim compilation, numerical parity, or performance measurements that were not run.'
    raw['messages'][1]['content']='\n\n'.join(first)
   raw=json.loads(C(raw));tokens=tokenize_row(raw,tok)
   for name,value in [('raw',raw),('prepared',tokens),('replay',replay)]:save(dest/(name+'.json'),value)
   n=len(tokens['input_ids']);case.update(serialized_tokens=n,supervised_tokens=n-tokens['loss_start'],exact_replay=True,licenses=licenses,files=list(rec['files']),stable_patch_id=rec.get('stable_patch_id'),raw_sha256=H((dest/'raw.json').read_bytes()),prepared_sha256=H((dest/'prepared.json').read_bytes()),source_sha256=H((dest/'source.json').read_bytes()),replay_sha256=H((dest/'replay.json').read_bytes()))
   eligible=status=='cleared' and bool(licenses) and tier!='rejected_optimization'
   case['status']='admitted_12k' if eligible and n<=12288 else 'admitted_32k_only' if eligible and n<=32768 else 'oversized' if eligible else 'reasoning_only_rejected' if tier=='rejected_optimization' else 'quarantined'
   if eligible:
    for bound in (12288,32768):
     if n<=bound:
      typed[(tier,bound,'raw')].append(raw);typed[(tier,bound,'prepared')].append(tokens)
  except Exception as e:case.update(status='error',error=type(e).__name__+': '+str(e))
  candidates.append(c);audit.append(case);save(ROOT/'progress.json',audit);print(c['id'],case['status'],case.get('serialized_tokens',case.get('error')),flush=True)
 manifest={'format':FORMAT,'repositories':repos,'evidence':evidence,'protected_keys':['commit:c793734eb715dbcfdb1ced58e348ec53c2d7ed85','stable_patch:a64f05b89099f719065fd2f432583df50af6d489','commit:bbf8a7a54a4436b09e32fd9aa9fb9f63db03dc99','stable_patch:66bf86cc01909076ba600a8e92d2c4adb0fb58f3'],'candidates':candidates};manifest['manifest_sha256']=H(C(manifest));save(ROOT/'manifest-draft.json',manifest);save(ROOT/'documentation-omissions.json',docs)
 for (tier,bound,kind),values in typed.items():(ROOT/f'{tier}-{bound}.{kind}.jsonl').write_bytes(b''.join(C(v)+b'\n' for v in values))
 save(ROOT/'integration-draft.json',{'schema':'engine-hour-intake-v1','scope':'CPU exact historical source and replay; all upstream correctness/performance claims remain source-reported, not locally executed. Correctness is a separate curriculum, never counted as measured speed winners.','tokenizer':tokenizer_identity(tokpath),'cases':audit,'attrition':dict(collections.Counter(c['status'] for c in audit)),'typed_counts':{f'{t}-{b}-{k}':len(v) for (t,b,k),v in typed.items()}})
if __name__=='__main__':main()
