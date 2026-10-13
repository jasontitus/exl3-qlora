from pathlib import Path
import json,sys,urllib.request
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
# Reuse only our local bounded Git transport definitions; do not execute initial fetch loop.
prefix=(ROOT/'fetch_pinned.py').read_text().split('for ident,slug,head,parent in PINS:')[0]
ns={'__file__':str(ROOT/'fetch_pinned.py')};exec(compile(prefix,'local_fetch_transport','exec'),ns)
ns['LOG']=json.loads((ROOT/'fetch-log.json').read_text());run=ns['run']
for ident,pre in [('splash','4bd9c'),('vllm-metal','d2149'),('omlx','00dec')]:
 p=next(p for p in (WORK/'kernel-library/store/candidates').glob('*/*/summary.json') if p.parent.name.startswith(pre));r=json.loads(p.read_text());head=r['sha'];parent=r['parents'][0];repo=ROOT/'mirrors'/(ident+'.git')
 run(repo,'fetch','--no-tags','--depth=2','--filter=blob:none','origin',head)
 paths=run(repo,'diff','--no-ext-diff','--no-textconv','--no-renames','--name-only','-z',parent,head).decode().split('\0');paths=[p for p in paths if p];objects=[]
 for rev in [parent,head]:
  for path in paths:
   for entry in run(repo,'ls-tree','-z',rev,'--',path).split(b'\0'):
    if entry:objects.append(entry.split(b'\t')[0].split()[2].decode())
 for oid in dict.fromkeys(objects):
  b=run(repo,'cat-file','blob',oid)
  if len(b)>8<<20:raise ValueError('blob exceeds bound')
 (ROOT/(ident+'-compact-source-pin.json')).write_text(json.dumps({'id':ident,'repository':r['repository'],'head':head,'parent':parent,'paths':paths,'blob_objects':list(dict.fromkeys(objects))},indent=2)+'\n')
 print(ident,head[:12],len(paths),flush=True)
url='https://api.github.com/repos/jundot/omlx/pulls/4351'
req=urllib.request.Request(url,headers={'User-Agent':'metal-library-source-intake-v1','Accept':'application/vnd.github+json'})
with urllib.request.urlopen(req,timeout=30) as f:raw=f.read(2<<20)
(ROOT/'evidence/omlx-pr4351.json').write_bytes(raw);r=json.loads(raw);(ROOT/'evidence/omlx-pr4351-body.md').write_text(r.get('body') or '')
print('PR4351',r.get('merge_commit_sha'),(r.get('body') or '')[:15000])
