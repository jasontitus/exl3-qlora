"""CPU validation of the new lane; separate independent semantic review required."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[2];V3=W/'kernel-library/expansion-v3-hour';sys.path[:0]=[str(V3/'tools'),str(W/'loss-fix/apple/src')]
import pack_exact as pack
from exl3_mlx.data import load_tokenizer,tokenize_row
H=lambda b:hashlib.sha256(b).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 report=json.loads((R/'integration-draft.json').read_text());manifest=json.loads((R/'manifest-draft.json').read_text());tok=load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw');reserved=pack.reserved_ids();prior_heads=set();prior_patches=set();bound={};checks=[];entries=[];exclusions=[];sources=[]
 for name in ['corpus-32768-v1/provenance.jsonl','reasoning-library-v1/provenance.jsonl']:
  p=V3/name;bound[str(p)]=H(p.read_bytes())
  for line in p.read_text().splitlines():
   d=json.loads(line);prior_heads.add(d['commit'])
   if 'provenance_file' in d:
    q=Path(d['provenance_file']);v=json.loads(q.read_text());prior_patches.add(v['source']['stable_patch_id'])
 for p in (V3/'optional-unmeasured-32768').glob('*.provenance.json'):
  d=json.loads(p.read_text());prior_heads.add(d['source']['head']['commit']);prior_patches.add(d['source']['stable_patch_id']);bound[str(p)]=H(p.read_bytes())
 for c in report['cases']:
  dest=R/'drafts'/c['id'];paths={k:str(dest/(k+'.json')) for k in ('source','raw','prepared','replay')}
  if not all(Path(p).is_file() for p in paths.values()):exclusions.append({'id':c['id'],'reason':'source_or_preparation_failure','detail':c.get('error')});continue
  d={k:json.loads(Path(p).read_text()) for k,p in paths.items()};s=d['source'];pack.verify_source(s);pack.verify_edit(s,d['raw'],d['replay']);require(tokenize_row(d['raw'],tok)==d['prepared'],'raw/prepared mismatch')
  require(set(d['raw'])=={'tools','messages'},'native raw contract')
  require(s['head']['commit']==c['commit'] and s['parent']['commit']==c['parent'],'source identities')
  require(not set(s['commits'])&reserved,'reserved identity');require(s['head']['commit']not in prior_heads,'previously released head');require(s['stable_patch_id']not in prior_patches,'previously released stable patch')
  repo=R/'mirrors'/(c['repo']+'.git')
  for path,pair in s['files'].items():
   for side in ('before','after'):
    v=pair[side]
    if v is None:continue
    b=subprocess.check_output(['git','-c','core.hooksPath=/dev/null','-C',str(repo),'show',s['parent' if side=='before' else 'head']['commit']+':'+path]);require(b==v['content'].encode(),'Git payload mismatch')
  for lic in c['licenses']:require(H((R/lic['path']).read_bytes())==lic['sha256'],'license binding')
  hashes={p:H(Path(p).read_bytes()) for p in paths.values()};checks.append({'id':c['id'],'inputs_sha256':hashes,'complete_code_replay':True,'published_raw_token_ids_and_mask_equal':True,'git_before_after_bytes_equal':True,'prior_exact_or_stable_duplicate':False,'protected_identity_overlap':False})
  eligible=c['protection_status']=='cleared' and c['curriculum_tier']!='rejected_optimization'
  if not eligible:exclusions.append({'id':c['id'],'reason':'negative_reasoning_only' if c['curriculum_tier']=='rejected_optimization' else 'protection_review_pending'});continue
  n=len(d['prepared']['input_ids'])
  if n>32768:exclusions.append({'id':c['id'],'reason':'complete_item_over_32k','serialized_tokens':n});continue
  entries.append({'id':c['id'],'repo':s['repository'],'commit':c['commit'],'parent':c['parent'],'paths':paths,'curriculum_tier':c['curriculum_tier'],'task_kind':s['input_metadata']['task_kind']['value'],'outcome':s['outcome'],'lineage':s['repository'].lower()+':'+c['commit'],'change_lineage':s['repository'].lower()+':'+c['commit'],'split_family':s['input_metadata']['source_group'],'serialized_tokens':n,'supervised_tokens':n-d['prepared']['loss_start'],'licenses':[{'path':str(R/x['path']),'sha256':x['sha256']} for x in c['licenses']],'review':{'status':'pending'}})
  sources.append({'id':c['id'],'split':'train','source':s})
 for p in (W/'kernel-library/expansion-v2/corpus-12288-v3').glob('*.provenance.json'):
  d=json.loads(p.read_text())
  if d['split']=='test':sources.append(d)
 overlap=pack.checks.cross_split_code_report(sources);require(not overlap['automatic_blocker'],'cross-split automatic blocker');require(not overlap['near_duplicate_pairs'],'cross-split near-code requires review')
 save(R/'case-index-draft.json',{'cases':entries});save(R/'cpu-materialization-audit.json',{'schema':'new-engine-cpu-materialization-v1','reviewer':'adversarial_process (producer; not independent approval)','status':'materialization_checks_passed_independent_review_pending','inputs_sha256':bound,'cases':checks,'excluded':exclusions,'selected_candidates':len(report['cases']),'eligible_12k':sum(c['serialized_tokens']<=12288 for c in entries),'eligible_32k':len(entries),'cross_split_code':overlap,'limitations':['Historical source claims only; no upstream compiler, test suite or GPU execution.','Complete source/host/test/build edits are retained, but selected prompt context may omit unchanged dependencies and use exact edit anchors.','Repository/declared-family grouping is conservative; no proof of semantic independence.']})
 print('CPU PASS',len(checks),'cases;',sum(c['serialized_tokens']<=12288 for c in entries),'12K;',len(entries),'32K')
if __name__=='__main__':main()
