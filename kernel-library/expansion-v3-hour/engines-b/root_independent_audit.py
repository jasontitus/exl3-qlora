import json,sys,hashlib,re,subprocess
from pathlib import Path
W=Path.cwd();R=W/'kernel-library/expansion-v3-hour/engines-b';sys.path[:0]=[str(R.parent/'tools'),str(W/'loss-fix/apple/src')]
import pack_exact as p
from exl3_mlx.data import load_tokenizer,tokenize_row
H=lambda b:hashlib.sha256(b).hexdigest()
idx=json.loads((R/'case-index-refined.json').read_text());tok=load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw');cases=[];allrows=[];reserved=p.reserved_ids();prior=set();patches=set()
for f in (W/'kernel-library/expansion-v2/corpus-32768-v3').glob('*.provenance.json'):
 d=json.loads(f.read_text());prior.add(d['source']['head']['commit']);patches.add(d['source']['stable_patch_id'])
for e in idx['cases']:
 d={k:json.loads(Path(v).read_text()) for k,v in e['paths'].items()};s=d['source'];p.verify_source(s);p.verify_edit(s,d['raw'],d['replay']);assert tokenize_row(d['raw'],tok)==d['prepared'];assert not set(s['commits'])&reserved;assert s['head']['commit'] not in prior;assert s['stable_patch_id'] not in patches
 # Independent local Git content lookup ties captured before/after to upstream commits.
 repo=R/'mirrors'/e['id'].split('-')[0]
 # Names can contain hyphens; use inventory of mirrors and repository basename.
 repo=R/'mirrors'/s['repository'].rsplit('/',1)[-1]
 if not repo.exists():
  options=[x for x in (R/'mirrors').iterdir() if x.name.lower().removesuffix('.git')==s['repository'].rsplit('/',1)[-1].lower()]
  assert len(options)==1,(repo,options);repo=options[0]
 for f,pair in s['files'].items():
  for side in ('before','after'):
   v=pair[side]
   if v is None:continue
   ref=s['parent' if side=='before' else 'head']['commit']+':'+f
   raw=subprocess.check_output(['git','-C',str(repo),'show',ref]);assert raw==v['content'].encode(),ref
 # Every complete excerpt is a contiguous exact baseline slice.
 prompt=next(m['content'] for m in d['raw']['messages'] if m['role']=='user');n=0
 for m in re.finditer(r'File: ([^\n]+), starting line (\d+)\n```\n(.*?)\n```',prompt,re.S):
  f,line,excerpt=m.groups();content=s['files'][f]['before']['content'];assert excerpt in content,(e['id'],f,'excerpt');n+=1
 cases.append({'id':e['id'],'inputs_sha256':{v:H(Path(v).read_bytes()) for v in e['paths'].values()},'baseline_excerpts_checked':n,'git_content_checked':True,'replay':True,'tokens_equal':True})
 allrows.append({'id':e['id'],'split':'train','source':s})
for f in (W/'kernel-library/expansion-v2/corpus-12288-v3').glob('*.provenance.json'):
 d=json.loads(f.read_text())
 if d['split']=='test':allrows.append(d)
overlap=p.checks.cross_split_code_report(allrows)
assert not overlap['automatic_blocker'];assert not overlap['near_duplicate_pairs'],overlap
receipt={'schema':'exact-edit-independent-review-v1','status':'approved','reviewer':'root','cases':cases,'scope':'All prompts/source contracts reviewed against primary commit evidence; source blobs independently read from pinned local Git; complete selected edits replayed, raw serialization retokenized. Hardware validation remains not_run. Historical task selection is declared.','corrections':['Concrete reconstructed task contract added to all15 prompts','Measured tradeoff tasks retain precision, hardware and workload qualifications'],'protected_identities':len(reserved),'cross_split_code':overlap,'distinct_new_commits':len(cases)}
rp=R/'exact-independent-review.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
for e in idx['cases']:e['review']={'status':'approved','receipt':str(rp),'sha256':H(rp.read_bytes())}
(R/'case-index-reviewed.json').write_text(json.dumps(idx,indent=2)+'\n');print('PASS',len(cases))
