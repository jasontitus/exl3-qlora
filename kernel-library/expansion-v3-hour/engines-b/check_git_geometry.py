"""Independently bind existence, modes and changed-path coverage to pinned Git trees."""
from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent;idx=json.loads((R/'case-index-reviewed-v2.json').read_text());rows=[]
for e in idx['cases']:
 s=json.loads(Path(e['paths']['source']).read_text());repo=R/'mirrors'/(s['repository'].rsplit('/',1)[-1].lower()+'.git');checked=0
 changed=set(subprocess.check_output(['git','-C',str(repo),'diff','--name-only',s['parent']['commit'],s['head']['commit']]).decode().splitlines())
 assert changed=={p for p,v in s['files'].items() if v.get('role','changed')=='changed'},e['id']
 for filename,pair in s['files'].items():
  for side in ['before','after']:
   ref=s['parent' if side=='before' else 'head']['commit'];line=subprocess.check_output(['git','-C',str(repo),'ls-tree',ref,'--',filename]).decode().strip()
   if pair[side] is None:assert not line,(filename,side,'falselymissing');continue
   mode,kind,rest=line.split(' ',2);oid,path=rest.split('\t',1)
   assert mode==pair[side]['mode'] and kind=='blob' and oid==pair[side]['blob'] and path==filename
   checked+=1
 rows.append({'id':e['id'],'exact_changed_path_coverage':True,'file_existence_mode_blob_checks':checked,'source_sha256':hashlib.sha256(Path(e['paths']['source']).read_bytes()).hexdigest()})
(R/'git-geometry-independent-review.json').write_text(json.dumps({'passed':True,'reviewer':'root','cases':rows,'scope':'Exact changed-path inventory, file absence/presence and Git tree mode/blob checked against parent/head. Complements prior content/read/replay/semantic review.'},indent=2)+'\n');print('PASS',len(rows))
