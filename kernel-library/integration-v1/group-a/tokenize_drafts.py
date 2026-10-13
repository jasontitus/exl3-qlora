from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import load_tokenizer,tokenize_row,tokenizer_identity
name=sys.argv[1] if len(sys.argv)>1 else 'manifest-fullfiles.json';m=json.loads((ROOT/name).read_bytes())
e=_evidence(ROOT,m.get('evidence',{}))[0];tokenizer=load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
out=ROOT/('drafts-'+Path(name).stem);out.mkdir(exist_ok=True);report=[]
for c in m['candidates']:
 try:
  r=reconstruct_candidate(c,m['repositories'],e);policy_path=ROOT/'documentation-omissions.json';policy=json.loads(policy_path.read_text()) if policy_path.exists() and name=='manifest-reviewed.json' else None;row,receipt=agent_example(r, documentation_omissions=policy.get(c['id'],{}) if policy is not None else None)
  tokens=tokenize_row(row,tokenizer);count=len(tokens['input_ids']);start=tokens['loss_start']
  dest=out/c['id'];dest.mkdir(exist_ok=True)
  (dest/'agent.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
  (dest/'tokens.json').write_text(json.dumps(tokens)+'\n');(dest/'replay.json').write_text(json.dumps(receipt,indent=2)+'\n')
  report.append({'case_id':c['id'],'serialized_tokens':count,'prompt_tokens':start,'supervised_tokens':count-start,'fits32768':count<=32768,'replay':receipt,'row_sha256':hashlib.sha256((dest/'agent.json').read_bytes()).hexdigest()})
  print(c['repository'],count,start,count-start,flush=True)
 except Exception as err:report.append({'case_id':c['id'],'error':str(err)});print(c['repository'],'ERROR',str(err),flush=True)
(out/'length-audit.json').write_text(json.dumps({'manifest':name,'tokenizer':tokenizer_identity('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw'),'rows':report},indent=2)+'\n')
assert 'mlx.core' not in sys.modules,'GPU module imported'
