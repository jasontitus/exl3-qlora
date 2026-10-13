"""Stream full checkpoint bytes into a portable integrity manifest; no tensor decoding."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
if a.output.exists():raise ValueError('Refuse overwrite of integrity receipt')
root=a.root.resolve();files=sorted(v for v in root.rglob('*') if v.is_file());expected=json.loads(a.verify.read_text()) if a.verify else None;records={};started=datetime.datetime.now(datetime.timezone.utc).isoformat()
if any(v.is_symlink() for v in root.rglob('*')):raise ValueError('Symlinks require explicit handling')
if expected and {str(v.relative_to(root)) for v in files}!=set(expected['files']):raise ValueError('File inventory differs')
for path in files:
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed while hashing: '+str(path))
 rel=str(path.relative_to(root));record={'size':after.st_size,'sha256':h.hexdigest()}
 if expected and expected['files'][rel]!=record:raise ValueError('Payload mismatch: '+rel)
 records[rel]=record;print(rel,after.st_size,flush=True)
result={'schema':'checkpoint-full-payload-sha256-v1','root':str(root),'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'total_bytes':sum(v['size'] for v in records.values()),'verified_manifest_sha256':hashlib.sha256(a.verify.read_bytes()).hexdigest() if a.verify else None}
a.output.write_text(json.dumps(result,indent=2)+'\n')
