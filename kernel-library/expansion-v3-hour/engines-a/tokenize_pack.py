from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path.insert(0,str(W/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
sys.path.insert(0,str(W/'kernel-library/expansion-v3-hour/tools'))
from exact_edit_renderer import agent_example
from exl3_mlx.data import load_tokenizer,tokenize_row,tokenizer_identity
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'manifest-review-candidate.json').read_text()) if (R/'manifest-review-candidate.json').exists() else json.loads((R/'draft-manifest.json').read_text())
policy=json.loads((R/'documentation-omissions-draft.json').read_text());path='/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw';t=load_tokenizer(path);rows=[]
for c in m['candidates']:
 out=R/'reviewed'/c['id'];out.mkdir(parents=True,exist_ok=True)
 item={'case_id':c['id'],'repository':c['repository'],'commit':c['commit'],'parent':c['parent'],'lineage':c['lineage'],'source_group':c['source_group'],'curation_outcome':c['curation_outcome'],'task_kind':c['task_kind']['value']}
 try:
  r=reconstruct_candidate(c,m['repositories'],_evidence(R,m.get('evidence',{}))[0]);row,receipt=agent_example(r,documentation_omissions=policy.get(c['id'],{}))
  # Repair/neutral curricula must not falsely request a guaranteed optimization.
  cls=c['curation_outcome'];verb={'correctness':'Repair the reported correctness or compatibility issue in','neutral':'Implement and explain the historically requested dispatch-consolidation change in','tradeoff':'Apply the historically chosen precision/storage tradeoff in','rejected':'Inspect the historical rejected proposal for','exploratory':'Implement the historically proposed, provisionally measured kernel change in','source_only':'Implement the captured kernel or dispatch change in'}.get(cls,'Optimize')
  for message in row['messages']:
   if message['role']=='user':
    assert message['content'].startswith('Optimize ')
    message['content']=verb+' '+message['content'][len('Optimize '):]
    if cls!='source_reported_improvement':message['content']+='\nThis is a historical '+cls+' task. Do not imply a reproduced speed improvement. Missing measurements must remain unknown.'
    break
  tok=tokenize_row(row,t)
  for fn,v in [('source.json',r),('raw.json',row),('prepared.json',tok),('replay.json',receipt)]: (out/fn).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
  n=len(tok['input_ids']);item.update({'serialized_tokens':n,'supervised_tokens':n-tok['loss_start'],'fits12288':n<=12288,'fits32768':n<=32768,'exact_replay':receipt['exact_selected_file_replay'],'paths':{fn:str(out/fn)for fn in ['source.json','raw.json','prepared.json','replay.json']},'sha256':{fn:sha(out/fn)for fn in ['source.json','raw.json','prepared.json','replay.json']},'source_record':str(R/'sourcepacks'/c['id']/'complete-record.json'),'source_record_sha256':sha(R/'sourcepacks'/c['id']/'complete-record.json')});print(c['id'],n,n-tok['loss_start'],flush=True)
 except Exception as e:item['error']=str(e);print(c['id'],'ERROR',str(e),flush=True)
 rows.append(item);(R/'reviewed-length-audit.json').write_text(json.dumps({'tokenizer':tokenizer_identity(path),'cases':rows},indent=2)+'\n')
assert 'mlx.core' not in sys.modules
