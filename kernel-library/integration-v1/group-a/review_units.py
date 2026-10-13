"""Reviewer-selected complete declarations with exact byte-boundary checks.
This helper locates named complete units; it does not certify dependency closure.
"""
from pathlib import Path
import sys,json,hashlib,re,ast
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2];sys.path.insert(0,str(WORK/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate,_evidence
sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((ROOT/'manifest-fullfiles.json').read_text());ev=_evidence(ROOT,m['evidence'])[0]
# Lexical braces skip comments and quoted strings; named ranges are subsequently reviewed.
def body(src,needle,start=None):
 n=src.index(needle);s=src.rfind('\n',0,n)+1 if start is None else src.index(start)
 tokens=re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]',src[n:]);depth=0;seen=False
 for t in tokens:
  x=t.group()
  if x=='{':depth+=1;seen=True
  elif x=='}':
   depth-=1
   if seen and depth==0:
    end=n+t.end()
    if src[end:end+1]==';':end+=1
    if src[end:end+1]=='\n':end+=1
    return len(src[:s].encode()),len(src[:end].encode())
 raise ValueError('Missing complete body '+needle)
def function_py(src,name):
 tree=ast.parse(src)
 node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
 lines=src.splitlines(keepends=True);start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
 return len(''.join(lines[:start]).encode()),len(''.join(lines[:node.end_lineno]).encode())
def unit(record,path,kind,bounds):
 d={'path':path,'kind':kind}
 for side,lim in zip(['before','after'],bounds):
  data=(record['files'][path][side] or {'content':''})['content'].encode();start,end=lim;d[side]={'start':start,'end':end,'sha256':sha(data[start:end])}
 return d
for c in m['candidates']:
 rec=reconstruct_candidate(c,m['repositories'],ev);units=[]
 def addbody(path,needle,start=None,kind='complete_function'):
  b=[]
  for side in ['before','after']:b.append(body(rec['files'][path][side]['content'],needle,start))
  units.append(unit(rec,path,kind,b))
 if c['repository']=='splash':
  addbody('runtime/metal/kernels/shared/vision.metal','kernel void vision_attention(','kernel void vision_attention(',kind='complete_kernel')
  addbody('runtime/ops/Vision.cpp','void Vision::encode(','void Vision::encode(')
  for path in ['runtime/metal/abi/Vision.h','dev/tests/engine/vision_attention_metal_test.mm','dev/native.mk']:
   if path=='dev/native.mk':continue
   units.append(unit(rec,path,'whole_file',[(0,len((rec['files'][path][s] or {'content':''})['content'].encode())) for s in ['before','after']]))
 elif c['repository']=='ds4':
  addbody('ds4.c','static bool qwen4_gemv_rows(','static bool qwen4_gemv_rows(')
  addbody('ds4_metal.m','int ds4_gpu_qwen4_dense_mm_tensor(','int ds4_gpu_qwen4_dense_mm_tensor(')
  addbody('metal/qwen4.metal','kernel void kernel_qwen4_dense_mm_reduce(','kernel void kernel_qwen4_dense_mm_reduce(',kind='complete_kernel')
  addbody('tests/test_qwen4_kernels.c','static void test_dense_mm(','static void test_dense_mm(')
  # A changed complete host caller is present; newly-added Q8 kernel and all ancillary
  # changes remain full-commit targets with exact original anchors, not fabricated baseline code.
 elif c['repository']=='omlx':
  path='omlx/patches/turboquant_attention.py'
  for name in ['_fused_mse_multirow_2pass1_kernel','_fused_multirow_mse_attention','_decode_multirow_attention','apply_turboquant_attention_patch']:
   units.append(unit(rec,path,'complete_function',[function_py(rec['files'][path][s]['content'],name) for s in ['before','after']]))
 elif c['repository']=='vllm-metal':
  #715 primitive's existing kernel is >1100lines; preserve whole-file draft and
  # length exclusion until a separately bounded contract is reviewed.
  continue
 # Compute exact gap changes to declare omissions; renderer still replays wholecommit.
 omitted=[]
 for path,pair in rec['files'].items():
  before=(pair['before'] or {'content':''})['content'].encode();after=(pair['after'] or {'content':''})['content'].encode();bc=ac=0
  for u in sorted([u for u in units if u['path']==path],key=lambda u:(u['before']['start'],u['after']['start'])):
   b,a=u['before'],u['after']
   if before[bc:b['start']]!=after[ac:a['start']]:omitted.append(path)
   bc,ac=b['end'],a['end']
  if before[bc:]!=after[ac:]:omitted.append(path)
 c['context_policy']='reviewed_operation';c['reviewed_units']={'reviewed_by':'Codex group-a complete named-declaration review2026-10-13','dependency_review':'Complete named baseline kernels/functions and stated ABI context; omitted repository units are external context. Full target still contains every commit edit and exact old anchors. No standalone build or full dependency closure claim.','units':units}
 c['operation_scope']={'description':'Named complete baseline operation declarations for a synthetic historical full-commit edit. Other changed units are present in the exact target and baseline edit anchors; full before/after files are preserved in sidecars.','omitted_path_reasons':{p:'Other complete repository declarations/build/test registration changes are not expanded as full prompt units. They remain exact whole-commit target edits and baseline anchors, with complete source in provenance.' for p in sorted(set(omitted))}}
 reconstruct_candidate(c,m['repositories'],ev)
 print(c['repository'],'units',len(units),'before bytes',sum(u['before']['end']-u['before']['start'] for u in units))
m.pop('manifest_sha256');m['manifest_sha256']=sha(json.dumps(m,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode());(ROOT/'manifest-scoped-draft.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
