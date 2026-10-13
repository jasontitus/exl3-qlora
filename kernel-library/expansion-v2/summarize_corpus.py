"""Count published rows and preserved receipts; never count discovery as training."""
import argparse,collections,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('corpus');a=p.parse_args();r=Path(a.corpus).resolve()
meta=json.loads((r/'ktune-import.json').read_text());audit=json.loads((r/'audit.json').read_text())
train=[x for x in audit if x['split']=='train'];test=[x for x in audit if x['split']=='test']
rows=[]
for x in train:
 x=json.loads((r/(x['id']+'.provenance.json')).read_text());source=x['source'];rows.append({'id':x['id'],'repository':source['repository'],'kind':source['task_kind']['value'],'commit':source['head'],'parent':source['parent'],'serialized_tokens':x['total_tokens'],'supervised_tokens':x['target_tokens'],'stable_patch_id':source['stable_patch_id'],'local_compile':x['replay']['compile'],'local_numerical':x['replay']['numerical'],'local_performance':x['replay']['performance']})
assert len(rows)==meta['counts']['train'];assert len({x['stable_patch_id'] for x in rows})==len(rows)
old=r.parents[1]/'integration-v1/corpus-12288-v3/test.jsonl';assert (r/'test.jsonl').read_bytes()==old.read_bytes()
lengths=sorted(x['serialized_tokens'] for x in rows)
result={'format':'metal-optimization-published-corpus-summary-v1','counts':meta['counts'],'by_repository':dict(collections.Counter(x['repository'] for x in rows)),'by_task_kind':dict(collections.Counter(x['kind'] for x in rows)),'total_supervised_tokens':sum(x['supervised_tokens'] for x in rows),'lengths':{'min':min(lengths),'median':lengths[len(lengths)//2],'max':max(lengths)},'frozen_heldout_byte_identical':True,'test_sha256':hashlib.sha256((r/'test.jsonl').read_bytes()).hexdigest(),'qualification':'Source-grounded retrospective agent edit examples with exact replay. Author-reported outcomes are not independently reproduced compilation, numerical parity or performance. Distinct commits are not independent optimization families. 32K preparation does not qualify 32K training.','cross_split':meta['code_overlap_review'],'cases':rows}
(r/'published-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
