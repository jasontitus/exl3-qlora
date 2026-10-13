"""Capture historical MLX header closure and compile; no build scripts/GPU."""
import datetime,hashlib,json,posixpath,re,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'mlx-closure-v2'
REPO=R/'mlx/mirrors/mlx.git'
sha=lambda b:hashlib.sha256(b).hexdigest()

def fetch(rev,path):
    if not re.fullmatch('[0-9a-f]{40}',rev) or path.startswith('/') or '..' in Path(path).parts:
        raise ValueError('Unsafe revision/path')
    p=subprocess.run(['git','-c','core.hooksPath=/dev/null','-C',str(REPO),'show',rev+':'+path],capture_output=True,timeout=15)
    if p.returncode:raise ValueError('Git dependency read failed '+path+': '+p.stderr.decode(errors='replace')[:500])
    if len(p.stdout)>2<<20:raise ValueError('Source file too large')
    return p.stdout

def capture(rev,name,folder,expected):
    files={};todo=[name]
    while todo:
        path=todo.pop()
        if path in files:continue
        raw=fetch(rev,path)
        if path==name and sha(raw)!=expected:raise ValueError('Changed primary source')
        if len(files)>=128 or sum(len(v) for v in files.values())+len(raw)>16<<20:raise ValueError('Closure exceeds budget')
        files[path]=raw
        text=raw.decode().replace('\\\n','')
        for kind,arg in re.findall(r'^\s*#\s*(include|import|include_next)\s+([^\n]+)',text,re.M):
            arg=re.sub(r'/\*.*?\*/','',arg).split('//')[0].strip()
            if kind!='include':raise ValueError('Unsupported include form')
            if re.fullmatch(r'<metal[A-Za-z0-9_]*>',arg):continue
            m=re.fullmatch(r'"([^"\n]+)"',arg)
            if not m:raise ValueError('Unresolved non-Metal include '+arg)
            target=m[1]
            if target.startswith('/') or '\\' in target:raise ValueError('Unsafe include')
            options=[posixpath.normpath(posixpath.join(posixpath.dirname(path),target)),posixpath.normpath(target)]
            found=None
            for option in dict.fromkeys(options):
                try:fetch(rev,option);found=option;break
                except ValueError:continue
            if found is None:raise ValueError('Missing quoted include '+target)
            todo.append(found)
    for path,raw in files.items():
        p=folder/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    return {k:sha(v) for k,v in files.items()}

def main():
    OUT.mkdir(exist_ok=False);results=[]
    for index in ['mlx/case-index.json','mlx/batch2/case-index.json']:
        for c in json.loads((R/index).read_text())['cases']:
            if c.get('repo','').lower().rstrip('/')!='https://github.com/ml-explore/mlx' or not c.get('recommended_core',True) or 'source' not in c.get('paths',{}):continue
            path=Path(c['paths']['source']);source=json.loads(path.read_text())
            for name,pair in source['files'].items():
                if not name.endswith('.metal'):continue
                for side in ['before','after']:
                    if pair.get(side) is None:continue
                    rev=source['parent' if side=='before' else 'head']['commit']
                    rec={'case_id':c['id'],'side':side,'file':name,'revision':rev,'source_record_sha256':sha(path.read_bytes())}
                    folder=OUT/sha((c['id']+side+name).encode())[:20]
                    try:
                        hashes=capture(rev,name,folder,sha(pair[side]['content'].encode()));target=folder/'output.air'
                        argv=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',str(folder),'-fmodules-cache-path='+str(OUT/'module-cache'),'-c',str(folder/name),'-o',str(target)]
                        p=subprocess.run(argv,capture_output=True,text=True,timeout=90)
                        rec.update(status='compiled' if p.returncode==0 else 'compile_failed',files_sha256=hashes,argv=argv,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,air_sha256=sha(target.read_bytes()) if target.exists() else None)
                    except (ValueError,subprocess.TimeoutExpired) as e:rec.update(status='unqualified',reason=str(e))
                    results.append(rec);print(c['id'],side,rec['status'],flush=True)
    report={'schema':'mlx-historical-closure-compiler-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results,'qualification':'Exact historical source and recursively captured quoted headers only. No project build scripts executed; no GPU/linking/numerical/performance proof; no training-data changes. Default explicit compiler flags may differ from upstream.'}
    (OUT/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
