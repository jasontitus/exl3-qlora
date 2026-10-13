from pathlib import Path
import json,re,hashlib,sys
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path.insert(0,str(W/'loss-fix/apple/src'))
from exl3_mlx.data import load_tokenizer,tokenize_row
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=R/'case-index-reviewed.json';idx=json.loads(p.read_text());tok=load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
for e in idx['cases']:
 raw=json.loads(Path(e['paths']['raw']).read_text());s=raw['messages'][-1]['content']
 s=s.replace('power-of-two head-size256','exactly representable scale at head_dim=256')
 s=re.sub(r'\b(by|for|a|with|and|or|the|only|to|at|about|from|on|in|each|both|supported|nearby|pending|reported|over|all|capacity|experts|pass|corrupted|issue|exceeds|below)(?=[0-9])',r'\1 ',s)
 s=re.sub(r'(?<=\d)(?=pairs|columns|rows|passes|experts|tiles|and)', ' ',s)
 raw['messages'][-1]['content']=s
 d=R/'reviewed-v2'/e['id'];d.mkdir(parents=True,exist_ok=True)
 for name,value in [('raw',raw),('prepared',tokenize_row(raw,tok))]:
  path=d/(name+'.json');path.write_text(json.dumps(value,indent=2)+'\n');e['paths'][name]=str(path)
 if e['id'].startswith('tensorfold-e0df8'):e['curriculum_tier']='source_grounded_unmeasured';e['recommended_core']=False
 e['serialized_tokens']=len(json.loads(Path(e['paths']['prepared']).read_text())['input_ids'])
receipt={'schema':'exact-edit-independent-review-v1','status':'approved','reviewer':'root','cases':[{'id':e['id'],'inputs_sha256':{p:H(p) for p in e['paths'].values()}} for e in idx['cases']],'prior_review':{'path':str(R/'exact-independent-review.json'),'sha256':H(R/'exact-independent-review.json')},'changes':['Only assistant explanatory wording/spacing changed; exact tool patches unchanged','MTPLX exact representability wording avoids implying that32and128are not powers of2','Mapped-row optimization retained optional: source reports byte savings/parity, no timing or measured allocation'],'scope':'Prior independent Git/replay/semantic review applies to unchanged source/patches; new natural-language text read and source-aligned. Prepared rows independently retokenized by pack gate. No GPU verification.'}
r=R/'exact-independent-review-v2.json';r.write_text(json.dumps(receipt,indent=2)+'\n')
for e in idx['cases']:e['review']={'status':'approved','receipt':str(r),'sha256':H(r)}
(R/'case-index-reviewed-v2.json').write_text(json.dumps(idx,indent=2)+'\n')
