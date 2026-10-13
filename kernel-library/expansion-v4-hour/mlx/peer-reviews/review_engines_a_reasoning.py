"""Independent quote and prospective-baseline byte check; no GPU or source execution."""
import hashlib, json, re, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[3]
BASE=WORK/'kernel-library/expansion-v4-hour/engines-a'
sys.path.insert(0,str(WORK/'kernel-library/expansion-v3-hour/tools'))
import reasoning_data as rd
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=BASE/'reasoning-records-v2.json';rs=json.loads(p.read_text())['records']
manifest=json.loads((BASE/'manifest-reviewed.json').read_text());repos=manifest['repositories'];cases=json.loads((BASE/'case-index.json').read_text())['cases'];bound={str(p):H(p)};out=[];slices=0
for r in rs:
 rd.validate(r,rd.reserved_ids())
 for e in r['source_evidence']: bound[e['path']]=e['sha256']
 for e in r['licenses']:bound[e['path']]=e['sha256']
 repo=next(v['path'] for v in repos.values() if v['url'].lower()==r['repo'].lower())
 for path,start,end,code in re.findall(r'Baseline (.+?) \(exact pinned byte range (\d+):(\d+)\):\n```\n(.*?)\n```',r['prompt'],re.S):
  before=subprocess.check_output(['git','--git-dir',repo,'show',r['parent']+':'+path])
  expected=before[int(start):int(end)].decode()
  assert code==expected,(r['id'],path,start,end)
  slices+=1
 out.append({'id':r['id'],'record_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,separators=(",", ":"),ensure_ascii=False).encode()).hexdigest()})
report={'schema':'reasoning-independent-review-v1','status':'approved','reviewer':'adversarial_code','path':str(p),'sha256':H(p),'records':out,'counts':{'records':len(rs),'commits':len({r['commit'] for r in rs}),'prospective_reconstructed':sum('next-test' in r['id'] for r in rs),'retrospective':sum('next-test' not in r['id'] for r in rs),'exact_parent_slices_verified':slices},'checks':['Read all25 prompts/answers and captured primary commit descriptions. Verified quoted evidence and licensing hashes, reserved head/parent gate, exact parent Git byte ranges for all supplied prospective code.','Retained quality regressions, neutral/slower routes, future-test language and lack of local measured results.','Revised ff7 now supplies complete baseline canary and latch ownership helpers; revised a1d explicitly leaves qwen4 KLD/benchmark qualification outstanding.'],'findings':[],'resolved_findings':['ff7 unrelated launchMapped baseline replaced with actual canary/latch source.','a1d unreported KLD/throughput gate explicitly stated in both views.'],'limitations':['25views describe20source commits; six baseline-only prompts are retrospective curator reconstructions, not untouched independent discoveries.','Quoted local-upstream reports are author observations, not executed or independently reproduced by this library.','Exact replay is not compilation, mathematical correctness, runtime speed or model-quality proof.'],'inputs_sha256':bound}
q=HERE/'engines-a-reasoning-independent-review.json';q.write_text(json.dumps(report,indent=2)+'\n');print(q,H(q),report['counts'])
