"""Precompute heldout token categories without reading evaluation results."""
import hashlib,importlib.util,json
from collections import Counter
from pathlib import Path
from transformers import PreTrainedTokenizerFast
WORK=Path(__file__).resolve().parents[3]
SOURCE=WORK/'loss-fix/apple/reports/audit_final30_independent.py'
CORPUS=WORK/'kernel-library/expansion-v4-hour/training-mix-12288-v3'
TOKENIZER=Path('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('historical_category_reference',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
tok=PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER),local_files_only=True)
raw=[json.loads(v) for v in (CORPUS/'test.jsonl').read_text().splitlines()]
prepared=[json.loads(v) for v in (CORPUS/'test.prepared.jsonl').read_text().splitlines()]
assert len(raw)==len(prepared)==16
rows=[];counts=Counter()
for i,(r,p) in enumerate(zip(raw,prepared,strict=True)):
 reconstructed,categories=m.labels(r,tok)
 assert reconstructed==p, f'Item{i}: tokenizer/template reconstruction differs'
 assert len(categories)==len(p['input_ids'])-p['loss_start']
 counts.update(categories)
 rows.append({'index':i,'input_ids_sha256':hashlib.sha256(json.dumps(p['input_ids'],separators=(',',':')).encode()).hexdigest(),'loss_start':p['loss_start'],'supervised_tokens':len(categories),'categories':categories})
config=json.loads((TOKENIZER/'config.json').read_text())
prediction_vocab_size=config.get('text_config',config)['vocab_size']
assert type(prediction_vocab_size) is int and prediction_vocab_size>0
inputs=[TOKENIZER/'config.json',SOURCE,Path(__file__),CORPUS/'test.jsonl',CORPUS/'test.prepared.jsonl',TOKENIZER/'tokenizer.json',TOKENIZER/'tokenizer_config.json']
result={'schema':'heldout-token-categories-v1','status':'all16_reconstructed_exactly','tokenizer_size':len(tok),'prediction_vocab_size':prediction_vocab_size,'prediction_vocab_source':'checkpoint config text_config.vocab_size; verify actual evaluated head dimension in provenance audit','inputs_sha256':{str(p):digest(p) for p in inputs},'counts':dict(sorted(counts.items())),'rows':rows,'limits':['Categories reuse historical lexical classifier, not semantic proof of kernel logic','Mixed boundary tokens remain separate','Labels created without consulting new model results','No new model quality evidence']}
out=Path(__file__).with_name('heldout-token-categories-v1.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'supervised_tokens':sum(counts.values()),'counts':result['counts']}))
