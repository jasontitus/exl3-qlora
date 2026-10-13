from pathlib import Path
import os,subprocess,json,hashlib,time
R=Path(__file__).resolve().parent;W=R.parents[2];env={**os.environ,'GIT_CONFIG_GLOBAL':'/dev/null','GIT_CONFIG_NOSYSTEM':'1','GIT_TERMINAL_PROMPT':'0','GIT_NO_REPLACE_OBJECTS':'1'};log=[]
def run(repo,*args):
 cmd=['git','-c','core.hooksPath=/dev/null','--git-dir='+str(repo),*args];t=time.monotonic();p=subprocess.run(cmd,capture_output=True,env=env,timeout=180);log.append({'argv':cmd,'elapsed':time.monotonic()-t,'exit':p.returncode,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr':p.stderr.decode(errors='replace')[-1000:]});(R/'history-acquisition.json').write_text(json.dumps(log,indent=2)+'\n');p.check_returncode();return p.stdout
for rid,url in [('ds4','https://github.com/antirez/ds4.git'),('splash','https://github.com/incoai/splash.git')]:
 repo=R/'mirrors'/(rid+'.git');repo.parent.mkdir(exist_ok=True)
 if not repo.exists():
  run(repo,'init','--bare');(repo/'objects/info/alternates').write_text(str(W/'kernel-library/expansion-v2/group-a/mirrors'/f'{rid}.git'/'objects')+'\n');run(repo,'remote','add','origin',url);run(repo,'config','remote.origin.promisor','true');run(repo,'config','remote.origin.partialclonefilter','blob:none')
 run(repo,'fetch','--depth=600','--no-tags','--filter=blob:none','origin','HEAD');head=run(repo,'rev-parse','FETCH_HEAD').decode().strip();raw=run(repo,'log','--max-count=600','--no-merges','--format=%H%x00%P%x00%aI%x00%B%x00','--name-only','-z',head)
 (R/(rid+'-history.raw')).write_bytes(raw);(R/(rid+'-head.json')).write_text(json.dumps({'repository':url[:-4],'head':head,'history_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest()},indent=2)+'\n');print(rid,head,len(raw),flush=True)
