"""Stream full checkpoint bytes into a portable integrity manifest; no tensor decoding."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
if a.output.exists():raise ValueError('Refuse overwrite of integrity receipt')
root=a.root.resolve()
if not root.is_dir():raise ValueError('Checkpoint root must be an existing directory')
files=sorted(v for v in root.rglob('*') if v.is_file())
if not files:raise ValueError('Empty checkpoint inventory')
expected_bytes=a.verify.read_bytes() if a.verify is not None else None
expected=json.loads(expected_bytes) if expected_bytes is not None else None
if a.verify is not None:
 if not isinstance(expected,dict) or expected.get('schema')!='checkpoint-full-payload-sha256-v1' or not isinstance(expected.get('files'),dict) or not expected['files']:raise ValueError('Invalid verification manifest')
 for name,record in expected['files'].items():
  if not isinstance(record,dict) or set(record)!={'size','sha256'} or type(record['size']) is not int or record['size']<0 or not isinstance(record['sha256'],str) or len(record['sha256'])!=64 or any(c not in '0123456789abcdef' for c in record['sha256']):raise ValueError('Invalid file identity')
records={};started=datetime.datetime.now(datetime.timezone.utc).isoformat()
if any(v.is_symlink() for v in root.rglob('*')):raise ValueError('Symlinks require explicit handling')
if expected is not None and {str(v.relative_to(root)) for v in files}!=set(expected['files']):raise ValueError('File inventory differs')
for path in files:
 before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 after=path.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed while hashing: '+str(path))
 rel=str(path.relative_to(root));record={'size':after.st_size,'sha256':h.hexdigest()}
 if expected is not None and expected['files'][rel]!=record:raise ValueError('Payload mismatch: '+rel)
 records[rel]=record;print(rel,after.st_size,flush=True)
if files!=sorted(v for v in root.rglob('*') if v.is_file()) or any(v.is_symlink() for v in root.rglob('*')):raise ValueError('Inventory changed while hashing')
result={'schema':'checkpoint-full-payload-sha256-v1','root':str(root),'started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':records,'total_bytes':sum(v['size'] for v in records.values()),'verified_manifest_sha256':hashlib.sha256(expected_bytes).hexdigest() if expected_bytes is not None else None}
a.output.write_text(json.dumps(result,indent=2)+'\n')
