from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parent;W=R.parents[3];sys.path.insert(0,str(W/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import load_tokenizer,tokenize_row,tokenizer_identity
m=json.loads((R/'manifest-reviewed.json').read_text());policy=json.loads((R/'documentation-omissions.json').read_text());path='/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw';t=load_tokenizer(path);rows=[]
for c in m['candidates']:
 out=R/'reviewed'/c['id'];out.mkdir(parents=True,exist_ok=True)
 try:
  r=reconstruct_candidate(c,m['repositories'],_evidence(R,m['evidence'])[0]);row,receipt=agent_example(r,documentation_omissions=policy.get(c['id'],{}));tok=tokenize_row(row,t)
  for fn,v in [('raw.json',row),('prepared.json',tok),('replay.json',receipt)]: (out/fn).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
  n=len(tok['input_ids']);item={'case_id':c['id'],'serialized_tokens':n,'supervised_tokens':n-tok['loss_start'],'fits12288':n<=12288,'fits32768':n<=32768,'exact_replay':receipt['exact_selected_file_replay'],'raw_sha256':hashlib.sha256((out/'raw.json').read_bytes()).hexdigest()};print(c['id'],n,n-tok['loss_start'],flush=True)
 except Exception as e:item={'case_id':c['id'],'error':str(e)};print(c['id'],'ERROR',str(e),flush=True)
 rows.append(item);(R/'reviewed-length-audit.json').write_text(json.dumps({'tokenizer':tokenizer_identity(path),'cases':rows},indent=2)+'\n')
assert 'mlx.core' not in sys.modules
