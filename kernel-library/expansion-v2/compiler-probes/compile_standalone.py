"""CPU compiler checks of two pinned standalone source pairs, no GPU execution."""
from pathlib import Path
import hashlib,json,subprocess,datetime
R=Path(__file__).resolve().parent;src=R.parent/'tensorfold/sourcepacks';rows=[]
for prefix,filename in [('22f41','exlr8_dense.metal'),('9b1a','exlr8_projection.metal')]:
 record=json.loads(next(src.glob('tensorfold-'+prefix+'*/complete-record.json')).read_text());pair=record['files']['zig/kernels/metal/core/'+filename]
 for side in ['before','after']:
  raw=pair[side]['content'].encode();file=R/(filename[:-6]+'-'+side+'.metal');file.write_bytes(raw)
  argv=['xcrun','-sdk','macosx','metal','-std=metal4.0','-fmodules-cache-path='+str(R/'module-cache'),'-c',str(file),'-o',str(file.with_suffix('.air'))]
  run=subprocess.run(argv,capture_output=True,text=True,timeout=60)
  rows.append({'commit':record['head'],'side':side,'source_sha256':hashlib.sha256(raw).hexdigest(),'argv':argv,'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
  print(filename,side,run.returncode,flush=True)
receipt={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compiler':subprocess.run(['xcrun','-sdk','macosx','metal','--version'],capture_output=True,text=True).stdout,'qualification':'Standalone Metal4 syntax/compiler gate only, default preprocessor configuration. No linked host engine, runtime correctness or numerical/performance qualification.','earlier_environment_failure':'First default-cache compile failed because sandbox cannot write ~/.cache/clang; reran with explicit workspace module cache.','results':rows}
(R/'compiler-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
