"""Reconstruct pinned DS4 runtime Metal source without executing host code."""
import ast,datetime,hashlib,json,re,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'ds4-runtime-v1';REPO=R/'engines-a/mirrors/ds4.git'
sha=lambda b:hashlib.sha256(b).hexdigest()

def git(rev,name):
    assert re.fullmatch('[0-9a-f]{40}',rev) and not name.startswith('/') and '..' not in Path(name).parts
    p=subprocess.run(['git','-c','core.hooksPath=/dev/null','-C',str(REPO),'show',rev+':'+name],capture_output=True,timeout=30)
    if p.returncode:raise ValueError(p.stderr.decode(errors='replace')[:500])
    if len(p.stdout)>4<<20:raise ValueError('Source exceeds capture budget')
    return p.stdout

def main():
    OUT.mkdir(exist_ok=False);results=[]
    cases=json.loads((R/'engines-a/case-index-reviewed-v2.json').read_text())['cases']
    for c in cases:
        if not c['id'].startswith('ds4-') or not c.get('recommended_core',False):continue
        srcpath=Path(c['paths']['source']);record=json.loads(srcpath.read_text())
        if not any(p.endswith('.metal') for p in record['files']):continue
        for side in ['before','after']:
            rev=record['parent' if side=='before' else 'head']['commit'];folder=OUT/(c['id']+'-'+side);folder.mkdir();rec={'case_id':c['id'],'side':side,'revision':rev,'source_record_sha256':sha(srcpath.read_bytes())}
            try:
                host=git(rev,'ds4_metal.m');text=host.decode();body=text.split('static const char *ds4_gpu_source =',1)[1].split('\nstatic NSString *ds4_gpu_full_source',1)[0]
                parts=[]
                for line in body.strip().splitlines():
                    literal=line.strip().rstrip(';')
                    if not literal:continue
                    if not literal.startswith('"') or not literal.endswith('"'):raise ValueError('Unsupported prelude syntax')
                    value=ast.literal_eval(literal)
                    if not isinstance(value,str):raise ValueError('Nonstring prelude')
                    parts.append(value)
                prelude=''.join(parts)
                names=re.findall(r'@"(metal/[^"\n]+\.metal)"',text.split('required_sources = @[',1)[1].split('];',1)[0])
                if not 10<len(names)<=64 or len(set(names))!=len(names):raise ValueError('Invalid source sequence')
                chunks=[prelude];hashes={}
                for name in names:
                    raw=git(rev,name)
                    if name in record['files'] and record['files'][name].get(side) is not None:
                        assert raw==record['files'][name][side]['content'].encode(),'Changed primary source'
                    dest=folder/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);hashes[name]=sha(raw);chunks.append('\n// appended '+name+'\n'+raw.decode()+'\n')
                (folder/'host.m.txt').write_bytes(host);(folder/'prelude.metal').write_text(prelude);tu=folder/'runtime.metal';tu.write_text(''.join(chunks));air=folder/'runtime.air'
                cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-fmodules-cache-path='+str(OUT/'module-cache'),'-c',str(tu),'-o',str(air)]
                p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
                rec.update(status='compiled' if p.returncode==0 else 'compile_failed',source_order=names,files_sha256=hashes,host_sha256=sha(host),prelude_sha256=sha(prelude.encode()),translation_unit_sha256=sha(tu.read_bytes()),argv=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,air_sha256=sha(air.read_bytes()) if air.exists() else None)
            except (ValueError,IndexError,SyntaxError,subprocess.TimeoutExpired) as e:rec.update(status='unqualified',reason=str(e))
            results.append(rec);print(c['id'],side,rec['status'],flush=True)
    (OUT/'receipt.json').write_text(json.dumps({'schema':'ds4-historical-runtime-compiler-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results,'qualification':'Recorded default runtime prelude and ordered source composition. No upstream scripts, host execution/linking, function constant variants, GPU/numerical/performance verification. Sealed training data unchanged.'},indent=2)+'\n')
if __name__=='__main__':main()
