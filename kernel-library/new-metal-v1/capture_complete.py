"""Capture complete changed files for reviewed commits; never execute repository code."""
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

BASE = Path(__file__).resolve().parent
MIRROR = BASE.parent/'expansion-v4-hour/mlx/mirrors/mlx.git'

def git(*args):
    return subprocess.check_output(['git','--git-dir',str(MIRROR),*args],timeout=90)

def safe(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in path:
        raise ValueError('Unsafe source path')
    return path

def main():
    review = json.loads((BASE/'reviewed-teaching-evidence.json').read_text())
    results=[]
    for row in review['records']:
        if row['sealed_v4_exact_head_matches']:
            continue
        head,parent=row['commit'],row['parent']
        folder=BASE/'complete-source'/head
        folder.mkdir(parents=True,exist_ok=False)
        paths=git('diff','--name-only','--no-renames',parent,head).decode().splitlines()
        if len(paths)>128:raise ValueError('Too many files')
        records=[]
        for name in paths+['LICENSE']:
            safe(name)
            for side,rev in [('before',parent),('after',head)]:
                entries=git('ls-tree',rev,'--',name).decode().splitlines()
                if not entries:
                    records.append({'side':side,'path':name,'absent':True});continue
                mode,kind,oid=entries[0].split('\t')[0].split()
                if kind!='blob' or mode not in ('100644','100755'):raise ValueError('Unsupported source type')
                size=int(git('cat-file','-s',oid))
                if size>4*1024*1024:raise ValueError('File capture exceeds bound')
                content=git('cat-file','blob',oid)
                dest=folder/side/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
                records.append({'side':side,'path':name,'git_blob':oid,'git_mode':mode,'bytes':size,'sha256':hashlib.sha256(content).hexdigest()})
        patch=git('diff','--binary','--no-renames',parent,head)
        (folder/'edit.patch').write_bytes(patch)
        result={'repository':row['repository'],'commit':head,'parent':parent,'files':records,'patch_sha256':hashlib.sha256(patch).hexdigest(),'status':'complete_changed_sources_captured_pending_replay_and_review','executed_upstream_code':False,'training_admission':False}
        (folder/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');results.append({'commit':head,'changed_files':len(paths),'captured_bytes':sum(x.get('bytes',0) for x in records)})
        print(json.dumps(results[-1]),flush=True)
    (BASE/'complete-source-summary.json').write_text(json.dumps({'cases':results},indent=2)+'\n')

if __name__=='__main__':main()
