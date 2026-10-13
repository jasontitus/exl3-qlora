"""CPU compile scoped DS4 translation units using exact pinned runtime prelude."""
from pathlib import Path
import ast,json,re,subprocess,hashlib,datetime
R=Path(__file__).resolve().parent;W=R.parents[2];repo=R.parent/'engines-a/mirrors/ds4.git';rows=[]
for cid,file in [('ds4-8cbdd7624f27-hour-v3','metal/cpy.metal'),('ds4-0658dac84541-hour-v3','metal/dsv4_rope.metal')]:
 record_path=R.parent/'new-core-32768'/(cid+'.provenance.json');s=json.loads(record_path.read_text())['source']
 for side in ['before','after']:
  ref=s['parent' if side=='before' else 'head']['commit'];host=subprocess.check_output(['git','-C',str(repo),'show',ref+':ds4_metal.m']).decode()
  marker='static const char *ds4_gpu_source =';body=host.split(marker,1)[1].split('\nstatic NSString *ds4_gpu_full_source',1)[0];lines=body.strip().splitlines();parts=[]
  for line in lines:
   if not line.strip():continue
   literal=line.rstrip(';');assert literal.startswith('"') and literal.endswith('"'),literal
   value=ast.literal_eval(literal);assert isinstance(value,str);parts.append(value)
  prelude=''.join(parts);shader=s['files'][file][side]['content'];assert subprocess.check_output(['git','-C',str(repo),'show',ref+':'+file])==shader.encode()
  folder=R/'ds4-scoped'/cid/side;folder.mkdir(parents=True,exist_ok=True)
  (folder/'runtime-host.m.txt').write_text(host);(folder/'runtime-prelude.metal').write_text(prelude);(folder/'kernel-source.metal').write_text(shader);tu=folder/'scoped.metal';tu.write_text(prelude+'\n'+shader)
  cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-fmodules-cache-path='+str(R/'module-cache'),'-c',str(tu),'-o',str(folder/'scoped.air')];r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
  rows.append({'case_id':cid,'revision':ref,'side':side,'file':file,'host_sha256':hashlib.sha256(host.encode()).hexdigest(),'prelude_sha256':hashlib.sha256(prelude.encode()).hexdigest(),'source_sha256':hashlib.sha256(shader.encode()).hexdigest(),'translation_unit_sha256':hashlib.sha256(tu.read_bytes()).hexdigest(),'argv':cmd,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr});print(cid,side,r.returncode,flush=True)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exact runtime prelude plus one complete captured source file, default preprocessor configuration. Scoped translation-unit compile only; not full concatenated engine, tensor variant, GPU execution, numerical test or performance proof. Failed scoped compilation may reflect missing dependency context.','results':rows}
(R/'ds4-compiler-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
