"""Bounded public Git source acquisition. No checkout, hooks, filters or upstream code execution."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent;W=R.parents[2]
items=json.loads((R/'selection-batch1.json').read_text())['candidates'];log=json.loads((R/'fetch-log-batch1.json').read_text()) if (R/'fetch-log-batch1.json').exists() else [];pins=json.loads((R/'source-pins-batch1.json').read_text()) if (R/'source-pins-batch1.json').exists() else [];total=0
ENV={**os.environ,'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null','GIT_TERMINAL_PROMPT':'0','GIT_NO_REPLACE_OBJECTS':'1'}
def run(repo,*args,check=True,lazy=False):
 argv=['git','-c','core.hooksPath=/dev/null','-c','diff.external=','-c','filter.lfs.required=false','--git-dir='+str(repo),*args];t=time.monotonic();p=subprocess.run(argv,env={**ENV,**({} if lazy else {'GIT_NO_LAZY_FETCH':'1'})},capture_output=True,timeout=90)
 log.append({'argv':argv,'exit_code':p.returncode,'seconds':time.monotonic()-t,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr':p.stderr.decode(errors='replace')[-2000:]});(R/'fetch-log-batch1.json').write_text(json.dumps(log,indent=2)+'\n')
 if check and p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
 return p
for repo_id in dict.fromkeys(x['repo'] for x in items):
 rows=[x for x in items if x['repo']==repo_id];repo=R/'mirrors'/(repo_id+'.git');repo.parent.mkdir(exist_ok=True)
 old=W/'kernel-library/expansion-v3-hour/engines-a/mirrors'/f'{repo_id}.git'
 if not repo.exists():
  run(repo,'init','--bare');(repo/'objects/info/alternates').write_text(str(old/'objects')+'\n');run(repo,'remote','add','origin',rows[0]['source_url']+'.git');run(repo,'config','remote.origin.promisor','true');run(repo,'config','remote.origin.partialclonefilter','blob:none')
 missing=[x['commit'] for x in rows if run(repo,'cat-file','-e',x['commit']+'^{commit}',check=False).returncode]
 if missing:run(repo,'fetch','--no-tags','--depth=2','--filter=blob:none','origin',*missing,lazy=True)
 for row in rows:
  h=row['commit']
  if any(p['head']==h for p in pins):continue
  parents=run(repo,'show','-s','--format=%P',h).stdout.decode().split()
  if len(parents)!=1:raise ValueError('Not singleparent '+h)
  parent=parents[0]
  if run(repo,'cat-file','-e',parent+'^{commit}',check=False).returncode:run(repo,'fetch','--no-tags','--depth=1','--filter=blob:none','origin',parent,lazy=True)
  paths=run(repo,'diff','--no-ext-diff','--no-textconv','--no-renames','--name-only','-z',parent,h).stdout.decode().rstrip('\0').split('\0')
  if len(paths)>40:raise ValueError('Too many paths '+h)
  blobs=[];unsupported=[]
  for rev in [parent,h]:
   for path in paths:
    for e in run(repo,'ls-tree','-z',rev,'--',path).stdout.split(b'\0'):
     if e:
      mode,kind,oid=e.split(b'\t')[0].split()
      if kind==b'blob':blobs.append(oid.decode())
      else:unsupported.append({'revision':rev,'path':path,'mode':mode.decode(),'kind':kind.decode(),'oid':oid.decode()})
  for oid in dict.fromkeys(blobs):
   b=run(repo,'cat-file','blob',oid,lazy=True).stdout
   if len(b)>12<<20:raise ValueError('Blob >8MiB')
   total+=len(b)
   if total>768<<20:raise ValueError('Batch source bound')
  rootpaths=run(repo,'ls-tree','--name-only',h).stdout.decode().splitlines();licenses=[]
  for name in rootpaths:
   if name.lower() in ['license','license.md','license.txt','copying','notice','notice.md']:
    b=run(repo,'show',h+':'+name,lazy=True).stdout
    if len(b)>131072:raise ValueError('License >128KiB')
    dest=R/'evidence/licenses'/f'{repo_id}-{h[:12]}-{name}';dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(b);licenses.append({'source_path':name,'path':str(dest.relative_to(R)),'sha256':hashlib.sha256(b).hexdigest()})
  pin={'repository':repo_id,'url':row['source_url'],'head':h,'parent':parent,'paths':paths,'unsupported_entries':unsupported,'blobs':list(dict.fromkeys(blobs)),'licenses':licenses};pins.append(pin);(R/'source-pins-batch1.json').write_text(json.dumps(pins,indent=2)+'\n');print(repo_id,h[:12],len(paths),'paths',len(set(blobs)),'blobs',flush=True)
print('materialized_read_bytes',total)
