"""CPU compile exact DS4 runtime concatenation; never run upstream scripts or GPU work."""
from pathlib import Path
import json,re,subprocess,hashlib,datetime
R=Path(__file__).resolve().parent;repo=R.parent/'engines-a/mirrors/ds4.git';rows=[];cid='ds4-0658dac84541-hour-v3'
s=json.loads((R.parent/'new-core-32768'/(cid+'.provenance.json')).read_text())['source']
for side in ['before','after']:
 ref=s['parent' if side=='before' else 'head']['commit'];scoped=R/'ds4-scoped'/cid/side;host=(scoped/'runtime-host.m.txt').read_text();prelude=(scoped/'runtime-prelude.metal').read_text();array=host.split('required_sources = @[',1)[1].split('];',1)[0];names=re.findall(r'@"(metal/[^"\n]+\.metal)"',array);assert len(names)>10 and len(names)==len(set(names));folder=R/'ds4-complete'/cid/side;folder.mkdir(parents=True,exist_ok=True);parts=[prelude];bindings={}
 for name in names:
  raw=subprocess.check_output(['git','-C',str(repo),'show',ref+':'+name]);bindings[name]=hashlib.sha256(raw).hexdigest();p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);parts.append('\n// appended '+name+'\n'+raw.decode()+'\n')
 tu=folder/'full-runtime.metal';tu.write_text(''.join(parts));cmd=['xcrun','-sdk','macosx','metal','-std=metal4.0','-fmodules-cache-path='+str(R/'module-cache'),'-c',str(tu),'-o',str(folder/'runtime.air')]
 try:
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);exit_code=r.returncode;stdout=r.stdout;stderr=r.stderr
 except subprocess.TimeoutExpired as e:exit_code=None;stdout=str(e.stdout or '');stderr='Compiler exceeded60seconds; no qualification.'
 rows.append({'case_id':cid,'revision':ref,'side':side,'exact_source_order':names,'source_sha256':bindings,'host_sha256':hashlib.sha256(host.encode()).hexdigest(),'prelude_sha256':hashlib.sha256(prelude.encode()).hexdigest(),'translation_unit_sha256':hashlib.sha256(tu.read_bytes()).hexdigest(),'argv':cmd,'exit_code':exit_code,'stdout':stdout,'stderr':stderr});print(side,exit_code,flush=True)
(R/'ds4-complete-compiler-receipt.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Exact default runtime prelude and ordered source concatenation reconstructed from pinned ds4_metal.m, no environment source overrides. Metal4 default preprocessor configuration only; no tensor-feature variant, function-constant pipeline creation, host linkage, GPU execution, numerical or performance qualification.','results':rows},indent=2)+'\n')
