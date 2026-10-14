"""Read-only exact target-file/patch check. Does not apply, build, fetch or execute source."""
import argparse
import hashlib
import json
from pathlib import Path
import stat

HERE=Path(__file__).resolve().parent

def check(lane,root,state):
    root=Path(root)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir():
        raise ValueError('Use an existing canonical absolute source root')
    manifest=json.loads((HERE/'manifest.json').read_text())
    row=manifest['lanes'][lane]
    patch=HERE/row['patch']
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=row['patch_sha256']:
        raise ValueError('Patch bytes changed')
    actual={}
    for rel,expected in row['files'].items():
        part=Path(rel)
        if part.is_absolute() or '..' in part.parts:
            raise ValueError('Noncanonical patch path')
        p=root/part
        if p.resolve()!=p or not stat.S_ISREG(p.lstat().st_mode):
            raise ValueError('Aliased or nonregular source: '+rel)
        if p.stat().st_size!=expected[state+'_bytes']:
            raise ValueError('Source length differs: '+rel)
        actual[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual[rel]!=expected[state+'_sha256']:
            raise ValueError('Source content differs: '+rel)
    fingerprint=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if fingerprint!=row[state+'_target_file_fingerprint']:
        raise ValueError('Target-file fingerprint differs')
    return {'lane':lane,'state':state,'files':len(actual),'target_file_fingerprint':fingerprint,'source_written':False,'scope':'Targeted file identity only; no build/dependency compatibility or execution qualification'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',choices=['mlx-stream','mlx-serve-host'],required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--state',choices=['before','after'],required=True)
    a=p.parse_args();print(json.dumps(check(a.lane,a.root,a.state),indent=2))
