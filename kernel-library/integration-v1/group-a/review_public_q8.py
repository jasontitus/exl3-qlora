"""Finite local comparison of public pinned source and candidate; no model/kernel execution."""
from pathlib import Path
import difflib,hashlib,json,re
R=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def span(src,name):
 start=src.index(name);begin=src.rfind('\n',0,start)+1;depth=0;seen=False
 for t in re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',src[start:]):
  if t.group()=='{':depth+=1;seen=True
  elif t.group()=='}':
   depth-=1
   if seen and depth==0:return begin,start+t.end()
 raise ValueError(name)
def lexical(src):
 src=re.sub(r'//[^\n]*|/\*[\s\S]*?\*/',' ',src)
 return re.findall(r'[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]',src)
public=R/'evidence/public-source/ds4-b85a6174-dense.metal';s=public.read_text();recordpath=R/'sourcepacks/ds4-8378039eb470-integration-v1/complete-record.json';rec=json.loads(recordpath.read_text())
ref_names=['void dequantize_q8_0(','kernel void kernel_q8_prefill_unpack(','kernel void kernel_mul_mm(','void dequantize_f16(']
ref={}
for name in ref_names:
 a,b=span(s,name);ref[name]={'source':s[a:b],'byte_range':[len(s[:a].encode()),len(s[:b].encode())],'sha256':sha(s[a:b].encode())}
new_names={'metal/qwen4.metal':['kernel void kernel_qwen4_batch_mm_q8('],'ds4_metal.m':['int ds4_gpu_qwen4_batch_mm_q8_tensor('],'tests/test_qwen4_kernels.c':['static void test_batch_mm_q8(']}
units=[];comparisons=[]
for p,names in new_names.items():
 for name in names:
  c=rec['files'][p]['after']['content'];a,b=span(c,name);src=c[a:b];t=lexical(src);units.append({'path':p,'name':name,'byte_range':[len(c[:a].encode()),len(c[:b].encode())],'sha256':sha(src.encode()),'lexical_tokens':len(t)})
  for rn,rv in ref.items():
   rt=lexical(rv['source']);match=difflib.SequenceMatcher(None,rt,t,autojunk=False).find_longest_match(0,len(rt),0,len(t));row={'candidate':name,'reference':rn,'longest_contiguous_lexical_tokens':match.size,'reference_offset':match.a,'candidate_offset':match.b,'common_token_text':' '.join(rt[match.a:match.a+match.size])}
   for w in [32,64]:row[f'shared_unique_{w}grams']=len({tuple(rt[i:i+w]) for i in range(max(0,len(rt)-w+1))}&{tuple(t[i:i+w]) for i in range(max(0,len(t)-w+1))})
   comparisons.append(row)
# Audit the actual full model-visible prompt and complete added source patch too.
rawpath=R/'drafts-manifest-reviewed/ds4-8378039eb470-integration-v1/agent.json'
raw=json.loads(rawpath.read_text());visible={'full_model_prompt':raw['messages'][1]['content'],'all_changed_added_lines':'\n'.join(x[1:] for x in rec['diff'].splitlines() if x.startswith('+') and not x.startswith('+++'))}
visible_comparisons=[]
for name,src in visible.items():
 t=lexical(src)
 for rn,rv in ref.items():
  rt=lexical(rv['source']);match=difflib.SequenceMatcher(None,rt,t,autojunk=False).find_longest_match(0,len(rt),0,len(t))
  visible_comparisons.append({'candidate_part':name,'reference':rn,'longest_contiguous_lexical_tokens':match.size,'shared_unique_32grams':len({tuple(rt[i:i+32]) for i in range(max(0,len(rt)-31))}&{tuple(t[i:i+32]) for i in range(max(0,len(t)-31))})})
for v in ref.values():v.pop('source')
contracts=R/'evidence/reserved-task-contracts.json';metadata=json.loads(contracts.read_text());oldpins=json.loads((R/'evidence/reserved-metadata.json').read_text())['review']['pins']
for row in metadata['rows']:
 suffix=row['path'].split('/data/processed/',1)[1];pin=next(v for k,v in oldpins.items() if k.endswith('/data/processed/'+suffix));assert row['sha256']==pin
out={'format':'ds4-public-q8-contract-comparison-v1','candidate_commit':rec['head']['commit'],'candidate_parent':rec['parent']['commit'],'reference_public_commit':'b85a6174da6d0ea3139b48194a2ca108097657b1','reference_public_repository':'https://github.com/ivanfioravanti/ds4-metal','public_source_matches_original_task_source_pin':True,'inputs_sha256':{str(p.relative_to(R)):sha(p.read_bytes()) for p in [public,recordpath,rawpath,contracts,R/'evidence/public-source/receipt.json',Path(__file__).resolve()]},'reference_complete_units':ref,'candidate_complete_units':units,'lexical_comparisons':comparisons,'complete_visible_text_comparisons':visible_comparisons,'contract_comparison':{'protected_operation':'Generic Q8 prefill: tiled half-rounded Q8 versus unpack→F16 matmul with scratch allocation/reuse; bitwise preservation; explicitly excludes decode.','candidate_operation':'Decode-only Q8_0 batch8/16: staged FP32 dequantized weights, FP32 simdgroup GEMM and optional split-K/reduction; rows/K multiples32; double-reference and matvec tolerance2e-6 plus1e-6.','shared_mechanisms':['Same Q8_0 scale+signed-byte packed format and mathematical matrix product.','Metal simdgroup matrix operations and familiar staged tiling vocabulary.'],'material_differences':['Different kernel/host entrypoints and dispatch route (decode8/16 versus generic prefill).','Different precision contract: FP32 staging and tolerant matvec comparison versus required half rounding and output-bit equality.','Different work partition:32weightrows per SIMD group/all8or16tokens, split-K; reference64x32 tiled matmul or separate full unpack plusF16 GEMM.','Candidate does not modify or include the protected task host/harness/config, existing q8-prefill unpack entrypoint, or generic tiled kernel.']},'recommendation':'Operation-scope clearance is supportable for new supervised training if independent reviewer agrees. Do not claim repository independence, absence of shared lineage/pretraining, or use this as held-out Q8-family efficacy evidence. Preserve existing task assignments.','limitations':['Public source pin matches task provenance, but derived protected standalone source/winner text was not retrieved due approval constraints.','Lexical overlap counts are finite evidence, not proof of semantic independence or absence of cherry-pick/transitive ancestry.','No compilation, numerical test, runtime benchmark or GPU/model computation was run.','Original sealed manifest remains unchanged pending an explicit versioned admission amendment.']}
(R/'evidence/ds4-public-contract-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print('Longest matches',[(x['candidate'],x['reference'],x['longest_contiguous_lexical_tokens']) for x in comparisons]);print('REPORT',sha((R/'evidence/ds4-public-contract-comparison.json').read_bytes()))
