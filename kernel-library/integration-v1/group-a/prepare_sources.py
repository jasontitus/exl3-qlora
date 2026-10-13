"""CPU-only pinned reconstruction; no upstream build or model imports."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
SHA=lambda b:hashlib.sha256(b).hexdigest()
CANON=lambda v:json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
pr=ROOT/'evidence/omlx-pr4408-body.md'
evidence={'omlx_pr4408':{'path':str(pr.relative_to(ROOT)),'sha256':SHA(pr.read_bytes()),'phase':'post_change','url':'https://github.com/jundot/omlx/pull/4408'}}
repos={};candidates=[]
for ident in ['splash','ds4','vllm-metal','omlx']:
 pin=json.loads((ROOT/(ident+'-source-pin.json')).read_text());repos[ident]={'path':str(ROOT/'mirrors'/(ident+'.git')),'url':'https://github.com/'+pin['repository']}
 src=next((WORK/'kernel-library/store/candidates').glob('*/'+pin['head']+'/summary.json'));saved=json.loads(src.read_text());msg=saved['message'];title=msg.splitlines()[0]
 def ann(text,quote,source='commit_message'):return {'text':text,'basis':'source','evidence':[{'source':source,'quote':quote}]}
 candidate={'id':ident+'-'+pin['head'][:12]+'-integration-v1','repository':ident,'commit':pin['head'],'parent':pin['parent'],'source_group':ident+'-upstream-optimization-lineage','protected':False,'outcome':'reported_improvement','split':'train','context_policy':'complete_changed_files','task_conditioning':'historical_source','task_kind':{'value':'kernel','reviewed_by':'Codex group-a source review 2026-10-13','evidence':[{'source':'commit_message','quote':title}]},'annotations':{'operation':ann(title,title),'rationale':ann(msg,msg)}}
 if ident=='splash':
  candidate['annotations'].update(hardware=ann('M5 Max (40-core GPU) and M3 Max; source-reported workloads, not this training host.','On the M5 Max (40-core GPU)'),constraints=ann('Preserve vision-attention extent/mask/rescale semantics. Summation order changes require the recorded parity tolerances, not a bitwise claim.','Summation order moves the embeddings within the parity bar'),architecture=ann('Use32-query-row tiles, separate score/probability/statistics storage, in-place PV accumulation and four-lane softmax reductions.','A threadgroup now takes 32 query rows on four simdgroups.'))
 elif ident=='ds4':
  candidate['annotations'].update(hardware=ann('M5 Max, C=16 decode workload, as source-reported.','+10% on the C=16 step of an M5 Max (paired A/B).'),constraints=ann('Retain Q8 decode supported batch/matrix shapes and an A/B fallback; fp32 matrix summation changes need double-reference and matvec tolerances.','The sums stay within 2e-6 of the matvec'),architecture=ann('Use fp32 simdgroup batch GEMM with one weight read per batch, staged Q8 blocks and split-K reduction for narrow projections.','The batch GEMM gives each\nsimdgroup 32 weight rows and every token'))
 elif ident=='vllm-metal':
  candidate['annotations'].update(hardware=ann('Source reports M5 Max/M5 Pro/M3 attention-primitive measurements; production routing remains unchanged.','M5 Max'),constraints=ann('Preserve supported FP16/BF16 geometry, page layout, split-KV reduction, tails and memory bounds; this commit adds kernels/tests, not production routing.','Production routing\nstays on the existing kernels.'),architecture=ann('Add P256/P512 grouped-query paged-decode kernels sharing KV locality and four-token online-softmax updates.','Query heads sharing a KV head run in one\nthreadgroup to improve KV cache locality'))
 elif ident=='omlx':
  body=pr.read_text();candidate['annotations'].update(hardware=ann('M4 Max32-core GPU36GB; Qwen3.8-27B MTP with4-bit TurboQuant KV. Reported in-process observations only.','M4 Max (32-core GPU) 36 GB','omlx_pr4408'),constraints=ann('One-sequence MSE decode/causal verify L<=15; preserve fallback for unsupported layouts/codecs/straddling widths and the reported numerical tolerance.','Everything outside the envelope keeps the existing kernels.','omlx_pr4408'),architecture=ann('Use a two-pass simdgroup-matrix path to share each KV unpack across query rows and read step-allocated buffers in place.','A threadgroup owns 8 query rows of one KV head','omlx_pr4408'))
 record=reconstruct_candidate(candidate,repos,_evidence(ROOT,evidence)[0])
 out=ROOT/'sourcepacks'/candidate['id'];out.mkdir(parents=True,exist_ok=True)
 (out/'complete-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 (out/'commit.diff').write_text(record['diff'])
 for path,pair in record['files'].items():
  for side in ['before','after']:
   if pair[side]:
    dest=out/side/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(pair[side]['content'])
 units=[]
 for path,pair in record['files'].items():
  item={'path':path,'kind':'whole_file'}
  for side in ['before','after']:
   b=(pair[side] or {'content':''})['content'].encode();item[side]={'start':0,'end':len(b),'sha256':SHA(b)}
  units.append(item)
 candidate['context_policy']='reviewed_units';candidate['reviewed_units']={'reviewed_by':'Codex group-a complete file coverage review','dependency_review':'All changed text files retained including host/build/tests. Complete repository dependency closure and compilation are not established.','units':units};candidates.append(candidate)
 print(ident,len(record['files']),sum(len((x[s] or {'content':''})['content'].encode()) for x in record['files'].values() for s in ['before','after']),flush=True)
manifest={'format':'exl3-mlx-upstream-commits-v1','repositories':repos,'evidence':evidence,'candidates':candidates,'protected_keys':['repository:https://github.com/ggml-org/llama.cpp'],'notes':['New source integration; current dataset and runtime untouched.','Source-reported gains are not reproduced. Existing ktune exclusions retained separately; unresolved protected-operation overlap blocks final promotion.','All source files and fullcommit patches retained; source snippets are not proof of dependency closure.']}
manifest['manifest_sha256']=SHA(CANON(manifest));(ROOT/'manifest-fullfiles.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
