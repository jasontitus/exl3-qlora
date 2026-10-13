"""Bounded peer-review corrections; preserves the original held reasoning payload."""
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path.insert(0,str(W/'kernel-library/expansion-v2/tools'))
from source_units import named_unit
H=lambda b:hashlib.sha256(b).hexdigest()
d=json.loads((R/'reasoning-records.json').read_text());old=H((R/'reasoning-records.json').read_bytes());changes=[]
for r in d['records']:
 if r['commit'].startswith('a1d2a02dfb7c'):
  before='Source-reported KLD and throughput results are evidence for that captured workload, not proof for every model.'
  after='The source explicitly says this is not byte-identical on qwen4 and still owes KLD and benchmark measurements there. Those measurements are required and unreported for that route; numerical range tests alone do not establish its model-quality or throughput impact.'
  assert before in r['completion'];r['completion']=r['completion'].replace(before,after);changes.append({'id':r['id'],'change':'Correct unsupported KLD/throughput implication: the source explicitly leaves qwen4 measurement outstanding.'})
 if r['id']=='mlx-serve-ff7f359bd024-hour-v4-next-test':
  s=R/'sourcepacks/mlx-serve-ff7f359bd024-hour-v4/complete-record.json';record=json.loads(s.read_text());src=record['files']['src/gather_qmm_nax.zig']['before']['content'].encode();helper=R/'evidence/ff7-parent-mlx.zig';hb=helper.read_bytes();repo=R/'mirrors/mlx-serve.git';parent=r['parent'];blob=subprocess.run(['git','--git-dir',str(repo),'rev-parse',parent+':src/mlx.zig'],capture_output=True,check=True).stdout.decode().strip();actual=subprocess.run(['git','--git-dir',str(repo),'show',parent+':src/mlx.zig'],capture_output=True,check=True).stdout;assert actual==hb
  selected=[];snippets=[]
  for path,body,names,sourcepath in [('src/gather_qmm_nax.zig',src,['canary','armed'],s),('src/mlx.zig',hb,['mlx_error_latched','mlx_error_buf','mlx_error_len','mlx_error_mtx','lockErrBuf','unlockErrBuf','latchMlxError','errorPending','takeError','dropLatchedErrorUnless'],helper)]:
   for name in names:
    u=named_unit(body,'zig',name);a,b=u['start'],u['end'];text=body[a:b].decode();selected.append({'path':path,'name':name,'kind':'complete_declaration','source_path':str(sourcepath),'source_sha256':H(sourcepath.read_bytes()),'before':{'start':a,'end':b,'sha256':H(body[a:b])}});snippets.append(f'Baseline {path} (exact pinned byte range {a}:{b}):\n```\n{text}\n```')
  q=r['prompt'].split('\n\nBaseline ',1)[0];r['prompt']=q+'\n\nThe declarations below show the actual baseline probe and the existing error-latch ownership helpers. Unrelated launch/layout code is omitted; no post-change body is supplied.\n\n'+'\n\n'.join(snippets)
  r['baseline_context_receipt']={'source_path':str(s),'sha256':H(s.read_bytes()),'selected_units':selected,'helper_git_pin':{'repository':r['repo'],'parent':parent,'path':'src/mlx.zig','blob':blob,'captured_path':str(helper),'sha256':H(hb)}}
  r['source_evidence'].append({'path':str(helper),'sha256':H(hb),'url':r['repo']+'/blob/'+parent+'/src/mlx.zig','quote':hb[named_unit(hb,'zig','takeError')['start']:named_unit(hb,'zig','takeError')['end']].decode()})
  changes.append({'id':r['id'],'change':'Replace unrelated launchMapped baseline with complete canary/armed functions and pinned parent error-latch declarations/helpers.'})
out=R/'reasoning-records-v2.json';out.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');receipt={'schema':'reasoning-peer-correction-v1','original_path':str(R/'reasoning-records.json'),'original_sha256':old,'revised_path':str(out),'revised_sha256':H(out.read_bytes()),'changes':changes,'exact_payloads_changed':False,'source_execution':'none','script_sha256':H(Path(__file__).read_bytes())};(R/'reasoning-correction-v2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
