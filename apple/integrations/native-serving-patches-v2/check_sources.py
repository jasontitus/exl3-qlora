import argparse, hashlib, json, os, stat
from pathlib import Path
HERE=Path(__file__).resolve().parent
def require(v,m):
 if not v:raise ValueError(m)
def capture(path,limit=16<<20):
 path=Path(path);require(type(limit) is int and 0<limit<=512<<20,'read limit')
 identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 before=path.lstat();require(stat.S_ISREG(before.st_mode) and before.st_size<=limit,'bounded regular file')
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as stream:
  opened=os.fstat(stream.fileno());require(stat.S_ISREG(opened.st_mode) and identity(opened)==identity(before),'input replaced before read')
  raw=stream.read(limit+1);after=os.fstat(stream.fileno())
 require(len(raw)==before.st_size and len(raw)<=limit and identity(before)==identity(after)==identity(path.lstat()),'input changed during read');return raw
def check(root,state):
 root=Path(root)
 require(root.is_absolute() and root.resolve()==root and root.is_dir(),'canonical existing root required')
 m=json.loads(capture(HERE/'manifest.json'));require(hashlib.sha256(capture(HERE/m['patch'])).hexdigest()==m['patch_sha256'],'patch changed')
 actual={}
 for rel,row in m['files'].items():
  part=Path(rel);require(not part.is_absolute() and '..' not in part.parts,'unsafe path');p=root/part
  require(p.resolve()==p,'aliased target')
  expected=row[state+'_sha256']
  if expected is None:
   require(not p.exists() and not p.is_symlink(),'new target already exists');actual[rel]=None
  else:
   raw=capture(p);require(len(raw)==row[state+'_bytes'],'target size changed');actual[rel]=hashlib.sha256(raw).hexdigest();require(actual[rel]==expected,'target bytes changed')
 fp=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest();require(fp==m[state+'_target_file_fingerprint'],'fingerprint changed')
 return {'state':state,'files':len(actual),'target_file_fingerprint':fp,'source_written':False,'scope':'target-only identity, not numerical/build/dependency qualification'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--state',choices=['before','after'],required=True);a=p.parse_args();print(json.dumps(check(a.root,a.state),indent=2))
