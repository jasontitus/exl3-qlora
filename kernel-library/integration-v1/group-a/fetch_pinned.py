"""Fetch only four fixed commit/parent pairs into new isolated partial bare repos.
Never check out/execute upstream source; log every Git command and captured output.
"""
from pathlib import Path
import subprocess,os,json,hashlib,time
ROOT=Path(__file__).resolve().parent
PINS=[('splash','incoai/splash','086f2c5c36584878a16af59e92a5da4265f9c6ef','e766526e0abd5db7230a9044750e19f8cac20889'),('ds4','antirez/ds4','8378039eb4709fa64a26478f60764035804744f7','fe457518a69c6e41603d9f6b3c444cfb251cf8c0'),('vllm-metal','vllm-project/vllm-metal','d8cb131f02592f63d55f378ad680474324a87363','4afabdf86f6e4eb40aed816edcf02e00f855170c'),('omlx','jundot/omlx','10e675bef8ef5dce4d7ed37b09ab2be69b526936','0c6fd953d3476cefc76fd55170edfd897b57edd3')]
LOG=[]
def run(repo,*args,timeout=120):
 argv=['git','-c','core.hooksPath=/dev/null','-c','diff.external=','-c','filter.lfs.required=false','--git-dir='+str(repo),*args]
 env={**os.environ,'GIT_TERMINAL_PROMPT':'0','GIT_NO_REPLACE_OBJECTS':'1','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
 t=time.monotonic();p=subprocess.run(argv,env=env,capture_output=True,timeout=timeout)
 LOG.append({'argv':argv,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-t,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr':p.stderr.decode(errors='replace')[-5000:]})
 (ROOT/'fetch-log.json').write_text(json.dumps(LOG,indent=2)+'\n')
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
 return p.stdout
for ident,slug,head,parent in PINS:
 repo=ROOT/'mirrors'/(ident+'.git');repo.parent.mkdir(exist_ok=True)
 if not repo.exists():run(repo,'init','--bare');run(repo,'remote','add','origin','https://github.com/'+slug+'.git')
 run(repo,'fetch','--no-tags','--depth=2','--filter=blob:none','origin',head)
 assert parent in run(repo,'show','-s','--format=%P',head).decode().split()
 paths=run(repo,'diff','--no-ext-diff','--no-textconv','--no-renames','--name-only','-z',parent,head).decode().split('\0');paths=[p for p in paths if p]
 if len(paths)>128:raise ValueError('Too many changed paths')
 objects=[]
 for rev in [parent,head]:
  for path in paths:
   entries=run(repo,'ls-tree','-z',rev,'--',path).split(b'\0')
   for entry in entries:
    if entry:
     mode,kind,oid=entry.split(b'\t')[0].split()
     if kind!=b'blob':raise ValueError('Nonblob source')
     objects.append(oid.decode())
 # Explicitly materialize blobs; later importer is offline and refuses lazy fetch.
 for oid in dict.fromkeys(objects):
  b=run(repo,'cat-file','blob',oid)
  if len(b)>8<<20:raise ValueError('Changed blob exceeds8MiB')
 entry={'id':ident,'repository':slug,'head':head,'parent':parent,'paths':paths,'blob_objects':list(dict.fromkeys(objects))}
 (ROOT/(ident+'-source-pin.json')).write_text(json.dumps(entry,indent=2)+'\n')
 print(ident,len(paths),'files',len(set(objects)),'materialized blobs',flush=True)
