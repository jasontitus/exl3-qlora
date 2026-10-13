"""Compile captured complete Metal sources; never execute GPU kernels."""
from pathlib import Path
import json,hashlib,subprocess,datetime
R=Path(__file__).resolve().parent;B=R.parent/'engines-b';rows=[]
for name in ['tensorfold-14641c806791-hour-v3','tensorfold-6128a0c131f9-hour-v3','tensorfold-c0bb87da2444-hour-v3','tensorfold-e0df8e314bfb-hour-v3']:
 record_path=B/'drafts'/name/'source.json';record=json.loads(record_path.read_text())
 for side in ['before','after']:
  folder=R/name/side
  for filename,pair in record['files'].items():
   if not filename.endswith(('.metal','.h')) or pair[side] is None:continue
   dest=folder/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(pair[side]['content'].encode())
  for filename,pair in record['files'].items():
   if not filename.endswith('.metal') or pair[side] is None:continue
   file=folder/filename
   flags=['default']
   if name.startswith('tensorfold-c0bb'):flags+=['pre_m5_fragments']
   for flag in flags:
    target=file.with_suffix('.'+flag+'.air')
    cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-fmodules-cache-path='+str(R/'module-cache')]
    if flag=='pre_m5_fragments':cmd+=['-DTF_SIMD_FRAGS=1']
    cmd+=['-c',str(file),'-o',str(target)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    rows.append({'case_id':name,'commit':record['head']['commit'],'parent':record['parent']['commit'],'source_record_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest(),'file':filename,'side':side,'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'configuration':flag,'argv':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'artifact_sha256':hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None})
    print(name,side,filename,flag,r.returncode,flush=True)
report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compiler':subprocess.run(['xcrun','-sdk','macosx','metal','--version'],capture_output=True,text=True).stdout,'qualification':'CPU Metal4 compiler acceptance only. No host linking, GPU dispatch, numerical test or timing. Passing source files do not qualify a complete engine.','results':rows}
(R/'compiler-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
