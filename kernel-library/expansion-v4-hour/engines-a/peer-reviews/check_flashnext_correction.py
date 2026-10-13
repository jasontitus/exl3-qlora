import hashlib,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(W/'kernel-library/expansion-v3-hour/tools'),str(W/'loss-fix/apple/src')]
import pack_exact as p
R=W/'kernel-library/expansion-v4-hour'
h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
i=R/'engines-b/case-index-revised-v2.json';d=json.loads(i.read_text());c=next(c for c in d['cases']if'ff81a4'in c['id'])
v={k:json.loads(Path(x).read_text())for k,x in c['paths'].items()}
s=v['source'];assert set(s['files'])=={'zig/src/families/flashnext/'+x+'.zig'for x in['engine','replay','split','tp']}
assert 'Nemotron' not in json.dumps(s['annotations'])
assert 'Nemotron' not in v['raw']['messages'][1]['content']
assert 'Flash Next DeltaNet' in v['raw']['messages'][1]['content']
p.verify_source(s);r=p.verify_edit(s,v['raw'],v['replay']);assert r['exact_scoped_replay']
t=p.tokenize_row(v['raw'],p.load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw'));assert t==v['prepared'];assert len(t['input_ids'])==10506
# Original record retains the same actual source bytes, target patch and replay.
original=R/'engines-b/drafts'/c['id']
os=json.loads((original/'source.json').read_text());orr=json.loads((original/'raw.json').read_text())
assert s['files']==os['files'] and s['diff']==os['diff']
assert v['raw']['messages'][-1]['tool_calls']==orr['messages'][-1]['tool_calls']
assert v['replay']==json.loads((original/'replay.json').read_text())
assert 'mlx.core' not in sys.modules
out={'schema':'exact-edit-independent-review-v1','status':'approved','reviewer':'quality_research','scope':'Independent review of the parent Flash Next/DeltaNet attribution correction only; rest of engines-b reviewed separately. Source commit and changed paths identify Flash Next, not Nemotron. Source blobs, target tool action and replay unchanged; corrected row freshly retokenized. No local kernel execution.', 'cases':[{'id':c['id'],'inputs_sha256':{str(Path(x).resolve()):h(Path(x))for x in c['paths'].values()}}], 'checks':{'source_hashes':True,'exact_replay':True,'actual_tokenizer':True,'token_count':10506,'source_blobs_unchanged':True,'tool_action_unchanged':True,'replay_unchanged':True,'architecture_correction_source_supported':True},'index_sha256':h(i),'script_sha256':h(Path(__file__))}
o=R/'engines-a/peer-reviews/flashnext-correction-independent-review.json';o.write_text(json.dumps(out,indent=2)+'\n');print(str(o),h(o))
