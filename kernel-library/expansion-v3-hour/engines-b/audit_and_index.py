"""CPU evidence/replay/token audit. Semantic approval is supplied independently."""
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[2]
sys.path[:0]=[str(W/'loss-fix/apple/src'),str(R.parent/'tools')]
import pack_exact as pack
from exl3_mlx.data import load_tokenizer,tokenize_row
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((R/'integration-draft.json').read_text());reserved=pack.reserved_ids();prior_commits={};prior_patches={};protected_patches=set()
for root in [W/'kernel-library/expansion-v2/corpus-12288-v3',W/'kernel-library/expansion-v2/corpus-32768-v3']:
 for p in root.glob('*.provenance.json'):
  d=json.loads(p.read_text());s=d['source'];prior_commits[s['head']['commit']]=d['id'];prior_patches[s['stable_patch_id']]=d['id']
  if d['split']=='test':protected_patches.add(s['stable_patch_id'])
tok=load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw');checks=[];entries=[];seen=set();hashes={}
for c in report['cases']:
 if not c['status'].startswith('admitted'):continue
 folder=R/'drafts'/c['id'];paths={n:str(folder/(n+'.json')) for n in ('source','raw','prepared','replay')};d={n:json.loads(Path(p).read_text()) for n,p in paths.items()};s=d['source'];assert not(set(s['commits'])&reserved)
 assert s['stable_patch_id'] not in protected_patches;assert s['input_metadata']['historical_protection_review']['status']=='cleared';assert c['curriculum_tier']!='rejected_optimization';assert s['head']['commit'] not in seen;seen.add(s['head']['commit'])
 pack.verify_source(s);v=pack.verify_edit(s,d['raw'],d['replay']);assert v['exact_scoped_replay'];assert tokenize_row(d['raw'],tok)==d['prepared']
 assert len(d['prepared']['input_ids'])==c['serialized_tokens'];assert c['serialized_tokens']<=32768
 for name,p in paths.items():assert H(Path(p))==c[name+'_sha256'];hashes[p]=H(Path(p))
 for lic in c['licenses']:assert H(R/lic['path'])==lic['sha256']
 old=prior_commits.get(c['commit']) or prior_patches.get(c['stable_patch_id']);checks.append({'id':c['id'],'tokens':c['serialized_tokens'],'exact_replay':True,'canonical_tokens_equal':True,'prior_duplicate':old,'protected_identity_check':'pass'})
 if old:continue
 entries.append({'id':c['id'],'repo':s['repository'],'commit':c['commit'],'parent':c['parent'],'paths':paths,'curriculum_tier':c['curriculum_tier'],'task_kind':s['task_kind']['value'],'outcome':s['outcome'],'lineage':s['repository']+':'+c['commit'],'change_lineage':s['repository']+':'+c['commit'],'split_family':s['input_metadata']['source_group'],'serialized_tokens':c['serialized_tokens'],'supervised_tokens':c['supervised_tokens'],'review':{'status':'pending','receipt':None,'sha256':None}})
result={'schema':'engine-hour-intake-cpu-audit-v1','checks':checks,'exact_new_unique_12k':sum(x['serialized_tokens']<=12288 for x in entries),'exact_new_unique_32k':len(entries),'protected_ids':len(reserved),'prior_duplicate_count':sum(x['prior_duplicate'] is not None for x in checks),'cpu_only':True,'source_claims_not_executed':True,'independent_semantic_review':'pending','inputs_sha256':hashes,'helper_sha256':{str(Path(__file__)):H(Path(__file__)),str(Path(pack.__file__)):H(Path(pack.__file__))},'limitations':['Full code/build/test edit replay does not prove compiled or correct kernels','Source-derived selected context does not establish dependency closure or absence of historical target conditioning','Prior source metadata and exact heldout scans do not prove semantic independence']}
for name,value in [('case-index-draft.json',{'cases':entries}),('cpu-audit.json',result)]:
 (R/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('checks','inputs_sha256','helper_sha256')},indent=2))
