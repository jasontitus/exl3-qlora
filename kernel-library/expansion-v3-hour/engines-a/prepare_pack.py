from pathlib import Path
import json,hashlib,sys,re,ast,difflib
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path[:0]=[str(W/'loss-fix/apple/src'),str(W/'kernel-library/expansion-v2/tools')]
from exl3_mlx.upstream_data import reconstruct_candidate
from source_units import scan_units,lex
sha=lambda b:hashlib.sha256(b).hexdigest()
ops={
'cb477466a2a1':'Quantized MoE decode and down-projection reduction at3-bit expert width','e92459219ce3':'MoE projections for single-slot multi-row verification on Ultra GPUs','080a62e2f2d7':'Causal attention for short grouped-query verification sequences','9a5467375008':'Sparse attention verification over dense BF16 KV cache views','3dc1dab54e7b':'GatedDeltaNet multi-row verification recurrence and rollback state','9a5c98126fb8':'Sorted expert matrix multiplication for MoE prefill','6281442d97bd':'Fused add-and-normalize calls with equal row count but different batch layouts','dce06d208cbc':'Vectorized attention staging for offset and strided KV views','c24c70ab8ead':'GatedDeltaNet decode and verification prework/recurrence dispatch',
'987c40030406':'EXL3 decode and prefill on macOS27 Metal toolchains','a05d15f77302':'TINY-codebook n48 EXL3 decode and prefill','6755ff2426b7':'Fused prefill attention over a quantized long-context KV cache','b89991a4e6ab':'Sparse attention over a packed KV cache at wider query-row counts','25a4994db9cc':'Sliding-window prefill attention with learned sink logits','2d58ab608455':'Packed-cache global attention for speculative verification rows','c2600a40739a':'Single-token KDA BF16 gate projections','3adb8002840a':'Qwen GatedDeltaNet recurrent-state storage precision and checkpoint identity',
'582d809fd300':'GLM Metal decode attention over padded selected rows','846428239de2':'Qwen attention at short prefill tails with routing-sensitive logits','fb2abb29e4ef':'Metal GLM attention scratch allocation at arbitrary context lengths','8fcd61d6bf0d':'Double-buffered Metal dense-projection scratch allocation','8cbdd7624f27':'Compressor position-table expansion during Metal prefill','0658dac84541':'Long-context Metal RoPE prefill and decode correctness','b64d3e0689a6':'Compressor row finalization at emit tokens on M5 decode','71d8c2a072c4':'Q8 query/KV and compressor projections on M5 decode','81f8534e3b85':'Routed MoE prefill dispatch geometry',
'17c6726e8388':'Single-tile MoE routed-row input handling','540e6ae59a58':'One-pass quantized gate/up kernel geometry and host ABI','5ecf1fe88676':'Live expert-tile dispatch for block MoE','7d7cadd19113':'Staged quantized-tile metadata prefetch occupancy tradeoffs','9402155ae6ed':'Prefill Q4 kernel behavior under Metal shader validation','dff5dfc57308':'MoE tile padding and gather extents','d2b331112d7f':'Selecting the matmul height for sparse live expert rows','4828e1637469':'Draft-attention query preparation storage','ab1256ccd614':'GatedDeltaNet fixed-shape kernel parameters and dispatch ABI','63ceff800643':'GatedDeltaNet decode intermediate writes and reference checks','7d60b965630a':'GatedDeltaNet decode completion synchronization','7161da96eac6':'Top-k/top-p sampling search over vocabulary logits','f3a511e50aa3':'GGUF prefill barrier convergence on partial tiles'}
architecture={
'cb477466a2a1':'Support3-bit packing with the common quantized loader and use8-lane groups for down+reduce. The reduction order changes, so preserve the stated FP32-truth RMS acceptance rather than claiming bitwise equality to the composed path.',
'e92459219ce3':'Route supported single-slot verification widths to the existing per-row-equivalent fused MoE rows arm on Ultra devices; retain the explicit enable/disable override.',
'080a62e2f2d7':'Split causal query rows according to the GQA-dependent vector-kernel limit, preserving each row’s causal key range. Yield to the preferred NAX fused path and restrict the default-on geometry to the measured head dimension.',
'9a5467375008':'Add a dense-BF16 load variant to the split-K sparse-attention path while preserving quantized source identity. Check actual pointer/stride alignment and use element loads for unaligned views; keep selector, caller and admission routing consistent.',
'3dc1dab54e7b':'Fold recurrence, exact norm-gate reduction and rollback input copies into one supported per-head threadgroup. Validate input shapes and GPU stream, probe the per-pipeline thread limit, and decline cleanly to the unfused chain on the specific resource refusal.',
'9a5c98126fb8':'Segment sorted expert rows into single-expert tiles, add row-mapped input access and pair gate/up with the specified SwiGLU epilogue. Preserve the original accumulation-order-dependent small-call fallback and per-configuration correctness canary.',
'6281442d97bd':'Key the cached add+norm output configuration by the full batch/sequence layout, not their equal product. Preserve each caller’s output shape and validate batched state ownership.',
'dce06d208cbc':'Guard the vectorized K/V staging kernels using the evaluated bound base address as well as geometry, declining misaligned views to the existing fallback. The source helper accepts unevaluated arrays, so this source edit alone does not prove all lazy-view cases safe.',
'c24c70ab8ead':'Reuse the fused GDN prework-and-recurrence kernel for non-Hadamard decode and2–8-row verification, retaining norm-gate and the exact captured rollback state.',
'987c40030406':'Remove address-space qualifiers from the NAX destination template type and probe optional kernels at their first evaluated build. Keep a named fallback for rejected toolchains; use the source’s dispatch/staging improvements with its explicit reduction-tree and near-tie limits.',
'a05d15f77302':'Apply the captured TINY n48 kernel changes. The cited timings came from AC power under desktop load and are exploratory author evidence, not controlled end-to-end validation by this library.',
'6755ff2426b7':'Separate absolute causal cache coordinates from a dispatch-local KV window. Dequantize only the bounded packed-key slice required for that dispatch and preserve the FP32 online-softmax carry and dense fallback.',
'b89991a4e6ab':'Choose split-K, packed gather or one dense rebuild from query width and cache-row reuse. Retain the source’s crossover limits, numerical tolerance and documented near-tie divergence instead of claiming universal bitwise equality.',
'25a4994db9cc':'Seed online softmax with the learned sink logit but no value row, then use the fused sliding-window path over ringed cache views. Couple memory accounting to the same dispatch predicate and preserve sink-free behavior.',
'2d58ab608455':'Share each packed K/V page walk across pairs of verification rows, with a final group of three. Engage only when all rows share the last row’s split partition and the existing decode templates pass their probe; keep each row’s math in decode order.',
'c2600a40739a':'Compute dense-BF16 f_b/g_b projections inside the single-token KDA kernel in the source GEMV accumulation order. Keep affine, FP8 and covariance-tap inputs on their existing path. The reported0.2% timing difference is inside noise, so this is not an established speed win.',
'3adb8002840a':'Use FP32 recurrent-state storage and a distinct persistent checkpoint namespace; reject the removed CLI switch and update admission accounting. The upstream record reports a throughput/memory cost and no resolved KLD gain, not a performance improvement.',
'582d809fd300':'Track the selected rows’ guaranteed valid prefix and use the bounds-checked path for padding. Upstream only reported build/diff validation; no GPU runtime or performance rerun is claimed.',
'846428239de2':'Keep tails of at most8 rows on FP32-query decode-style attention while retaining large-prefill kernels. Preserve coverage of routing-sensitive full logits; a correct double-reference matrix formula does not eliminate low-precision downstream sensitivity.',
'fb2abb29e4ef':'Round scratch allocations up to the required16-byte granularity while leaving indexed extents, arithmetic and precision unchanged. Cover all context-length residues, cache formats and invalid selected rows.',
'8fcd61d6bf0d':'Correct the host scratch allocation for the double-buffered dense-kernel storage and retain the added targeted regression. The commit message supplies no performance measurement.',
'8cbdd7624f27':'Expand the complete compressor position-table batch with one tiled-row copy kernel instead of one encoder per ratio-row segment, preserving conversion values and the explicit opt-out.',
'0658dac84541':'Apply the captured long-context Metal prefill/decode correctness repair. Source inspection and the exact replay establish the edit contents; this library has not executed the resulting model.',
'b64d3e0689a6':'Fuse the emit-path attention/indexer compressor row finalization and state shifts while matching each phase’s original reduction and quantization layout. Threads outside a phase’s virtual width still participate in every barrier.',
'71d8c2a072c4':'Combine the Q8 query/KV pair and four compressor projections into one M5 dispatch with disjoint output regions and the original per-lane accumulation trees. Keep other devices behind the explicit opt-in and preserve the rollback switch.',
'81f8534e3b85':'Apply the captured routed-MoE dispatch compaction. Preserve the source’s input/output indexing and complete host/kernel edits; the commit message alone does not establish a measured speed gain.',
'17c6726e8388':'Let supported one-tile plans read their rows in place and skip the redundant gather. Keep multi-tile plans and the Apple9 register tile on grouped storage with the existing arithmetic.',
'540e6ae59a58':'Assert the32-column/32-input-step requirement beside the one-pass gate/up layout and share its thread-count constant between host dispatch and kernel table fill. This pins a correctness contract, not a measured speed gain.',
'5ecf1fe88676':'Have grouping emit each expert pass’s live-tile grid and launch through validated indirect dispatch. Preserve the maximum grid bound and verify exact outputs against the old full-grid route.',
'7d7cadd19113':'Historical proposal: overlap next-step metadata loads with the current matmul. Later evidence found increased register usage and prefill regression; this proposal is retained as a rejected reasoning case, not a positive optimization target.',
'9402155ae6ed':'Force the prefill Q4 tile helper inline to avoid the source-reported macOS26.5 shader-validation misbehavior. The timing geomean was neutral within its spread; this is a correctness/toolchain compatibility fix.',
'dff5dfc57308':'Use one live-row matmul-height rule for route padding and gathering, writing only rows the selected matmul reads. Preserve untouched-tail canaries and the unchanged8-row path.',
'd2b331112d7f':'Share the smallest-fitting8/16/32-row matmul selector and admit8-row work for sparse GGUF tiles. Reported measurements used gather trimming in both arms, so they are conditional on that companion change.',
'4828e1637469':'Remove the unread duplicate query store, keep the grouped-query destination, and make the proposal input immutable. Preserve host binding order and test that the preparation leaves the input unchanged.',
'ab1256ccd614':'Compile the supported packed widths and derived strides into each GDN variant, removing unreachable guards and unused fields. Update host/kernel ABI together while preserving floating-point operation order.',
'63ceff800643':'Remove recurrent-row and q device writes unused by serving, preserving the k/v layout required by state replay. Update the test oracle to reconstruct the recurrent result from state and retain the explicitly documented BF16 rounding tolerance.',
'7d60b965630a':'Remove device completion counters read only by a host check redundant with successful command completion. Keep idle-lane memory checks and update all affected buffer/argument bindings.',
'7161da96eac6':'Use count-only pivot work for top-k and reuse ranked selected logits for eligible top-p cuts, with the original search for wide/tied cases. Preserve tie handling and disclose the rounding-sensitive nucleus boundary and absence of real-model timing.',
'f3a511e50aa3':'Run one barrier-convergent staged loop for all simdgroups and predicate only matmul/store work. Fixing the data race takes priority over the source-reported M3 Q4K slowdown; isolated branch probes did not identify the branch as its cause.'}
classes={p:'correctness'for p in ['6281442d97bd','dce06d208cbc','582d809fd300','846428239de2','fb2abb29e4ef','8fcd61d6bf0d','0658dac84541','540e6ae59a58','9402155ae6ed','f3a511e50aa3']};classes.update({'c2600a40739a':'neutral','3adb8002840a':'tradeoff','7d7cadd19113':'rejected','a05d15f77302':'exploratory','4828e1637469':'source_only','ab1256ccd614':'source_only','63ceff800643':'source_only','7d60b965630a':'source_only','81f8534e3b85':'source_only','c24c70ab8ead':'source_only','e92459219ce3':'source_only'})

def c_fallback(raw):
 # Independent exact named function spans when whole ObjC/C file has mutually exclusive preprocessor brace layouts.
 text=raw.decode();lines=text.splitlines(keepends=True);offsets=[0]
 for l in lines:offsets.append(offsets[-1]+len(l.encode()))
 out=[]
 pat=re.compile(r'(?m)^(?:(?:static|inline|extern|constexpr|__attribute__\([^\n]*\))\s+)*[A-Za-z_][A-Za-z_0-9:<>*& \t]*\b([A-Za-z_]\w*(?:::\w+)*)\s*\(')
 for m in pat.finditer(text):
  if m[1]in{'if','for','while','switch'}:continue
  start=len(text[:m.start()].encode());tail=raw[start:]
  # Limit only candidate search work; never return a clipped function.
  lines2=tail.splitlines(keepends=True);end=0
  for line in lines2[:1500]:
   end+=len(line)
   if line.rstrip() not in [b'}',b'};']:continue
   try:code,pairs,_=lex(tail[:end],'c')
   except ValueError:continue
   brace=code.find('{');semi=code.find(';')
   if brace<0 or 0<=semi<brace:break
   if pairs.get(brace)!=len(code.rstrip())-1:continue
   out.append({'name':m[1],'qualified_name':m[1],'kind':'complete_function','start':start,'end':start+end,'sha256':sha(raw[start:start+end]),'boundary_method':'individual_balanced_c_function','requires_conditional_review':'#if'in tail[:end].decode()});break
 return out

def scan(raw,path):
 suffix=Path(path).suffix.lower();lang='zig'if suffix=='.zig'else'metal'if suffix=='.metal'else'c'
 if suffix not in ['.zig','.metal','.m','.mm','.c','.cc','.cpp','.h','.hpp']:return[]
 try:return scan_units(raw,lang)['units']
 except ValueError:return c_fallback(raw)if lang!='zig'else[]

pins=[];metadata={}
for n in [1,2]:
 p=R/f'source-pins-batch{n}.json'
 if p.exists():pins+=json.loads(p.read_text())
 p=R/f'selection-batch{n}.json'
 if p.exists():metadata.update({x['commit']:x for x in json.loads(p.read_text())['candidates']})
m={'format':'exl3-mlx-upstream-commits-v1','repositories':{},'candidates':[],'evidence':{},'notes':['Draft source-grounded historical tasks, not captured agent rollouts. Correctness/neutral/rejected outcomes are not speed wins.']};audit=[];policy={};options_out=[]
for pin in pins:
 h=pin['head'];short=h[:12];repo=pin['repository'];case=repo+'-'+short+'-hour-v3';md=metadata[h];msg=md['message'];title=msg.splitlines()[0];ann=lambda t,q:{'text':t,'basis':'source','evidence':[{'source':'commit_message','quote':q}]};cl=classes.get(short,'source_reported_improvement');m['repositories'][repo]={'path':str(R/'mirrors'/f'{repo}.git'),'url':pin['url']}
 c={'id':case,'repository':repo,'commit':h,'parent':pin['parent'],'source_group':'mlx-sushi-connected-training'if repo in['mlx-serve','sushi']else repo+'-related-optimization-training','protected':False,'outcome':'negative'if cl=='rejected'else'reported_improvement'if cl=='source_reported_improvement'else'source_only','split':'train','task_conditioning':'historical_source','context_policy':'complete_changed_files','curation_outcome':cl,'task_kind':{'value':'kernel','reviewed_by':'Codex source review draft 2026-10-13','evidence':[{'source':'commit_message','quote':title}]},'annotations':{'operation':ann(ops[short],title),'rationale':ann(msg,msg),'architecture':ann(architecture[short],msg)}}
 hw=re.search(r'(?:Apple\s+)?M[1-6](?:[–-]M[1-6])?(?:\s+(?:Max|Pro|Ultra))?',msg)
 if hw:c['annotations']['hardware']=ann(hw.group()+' (historical source report; no portability measurement here)',hw.group())
 try:
  rec=reconstruct_candidate(c,m['repositories'],{});out=R/'sourcepacks'/case;out.mkdir(exist_ok=True,parents=True);(out/'complete-record.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');(out/'commit.diff').write_text(rec['diff']);(out/'commit-message.txt').write_text(msg)
  for path,pair in rec['files'].items():
   for side in ['before','after']:
    if pair[side]:p=out/side/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(pair[side]['content'])
  options=[]
  for path,pair in rec['files'].items():
   b=(pair['before']or{'content':''})['content'].encode();a=(pair['after']or{'content':''})['content'].encode()
   if Path(path).suffix.lower()in['.md','.rst']:
    if pair['after']:policy.setdefault(case,{})[path]={'before_sha256':sha(b)if pair['before']else None,'after_sha256':sha(a),'reason':'Documentation-only file reviewed separately; every non-documentation code/build/test edit remains in target and full raw patch retained.'}
    continue
   if not b or len(b)<2500:
    options.append({'path':path,'name':'whole_file','kind':'whole_file','before':{'start':0,'end':len(b),'sha256':sha(b)},'after':{'start':0,'end':len(a),'sha256':sha(a)},'before_bytes':len(b),'native':'.metal'in path or'/kernels/'in path,'test':any(x in path for x in['/test','/bench'])});continue
   bs=scan(b,path);ass=scan(a,path);ad={}
   for v in ass:ad.setdefault(v.get('qualified_name',v['name']),[]).append(v)
   for u in bs:
    name=u.get('qualified_name',u['name']);vs=ad.get(name,[])
    if len(vs)!=1:continue
    v=vs[0]
    if b[u['start']:u['end']]==a[v['start']:v['end']]:continue
    if u['kind']=='complete_declaration'and u.get('enclosing_names'):continue
    if u['end']-u['start']>30000:continue
    native='.metal'in path or'/kernels/'in path or bool(re.search('SOURCE|KERNEL|source',name))
    options.append({'path':path,'name':name,'kind':'complete_declaration'if u['kind']=='complete_macro'else u['kind'],'before':{k:u[k]for k in['start','end','sha256']},'after':{k:v[k]for k in['start','end','sha256']},'before_bytes':u['end']-u['start'],'native':native,'test':any(x in path.lower()for x in['/test','/bench'])or bool(re.search(r'(?i)test|parity|case|probe|assert|expect',name)),'conditional':u.get('requires_conditional_review',False)})
  options_out.append({'case_id':case,'options':options});units=[];names=[];budget=14000
  options.sort(key=lambda o:(o['test'],not o['native'],o['kind']=='complete_declaration'and o['before_bytes']<200,-min(o['before_bytes'],10000)))
  for o in options:
   if o['before_bytes']>budget or o['test']:continue
   if o['kind']=='complete_declaration'and o['before_bytes']<200 and units:continue
   u={k:o[k]for k in['path','kind','before','after']}
   if any(x['path']==u['path']and not(u['before']['end']<=x['before']['start']or u['before']['start']>=x['before']['end'])for x in units):continue
   units.append(u);names.append(o['name']);budget-=o['before_bytes']
   if len(units)>=3:break
  if not units:raise ValueError('No complete bounded operational baseline unit; needs manual review')
  units.sort(key=lambda u:(u['path'],u['before']['start'],u['after']['start']));omitted=[]
  for path,pair in rec['files'].items():
   b=(pair['before']or{'content':''})['content'].encode();a=(pair['after']or{'content':''})['content'].encode();bc=ac=0
   for u in[x for x in units if x['path']==path]:
    if b[bc:u['before']['start']]!=a[ac:u['after']['start']]:omitted.append(path)
    bc=u['before']['end'];ac=u['after']['end']
   if b[bc:]!=a[ac:]:omitted.append(path)
  c['context_policy']='reviewed_operation';c['reviewed_units']={'reviewed_by':'Codex lexical complete-unit candidate; semantic review pending','dependency_review':'Exact source-unit boundaries; external imports/templates/ABI/dispatch dependencies remain repository context. No standalone compilation guarantee. Full non-documentation edit target preserved.','units':units};c['operation_scope']={'description':'Historical operation context with complete named baseline definitions. Answer retains every non-documentation code/build/test edit.','omitted_path_reasons':{p:'Other repository context is not expanded; exact baseline anchors and all non-documentation target edits remain available.'for p in sorted(set(omitted))}}
  reconstruct_candidate(c,m['repositories'],{});m['candidates'].append(c);audit.append({'case_id':case,'status':'draft_ready','class':cl,'selected_names':names,'context_bytes':14000-budget,'stable_patch_id':rec['stable_patch_id'],'licenses':pin['licenses'],'changed_paths':pin['paths']});print(case,cl,14000-budget,names,flush=True)
 except Exception as e:audit.append({'case_id':case,'status':'blocked','class':cl,'error':str(e),'licenses':pin['licenses']});print(case,'BLOCKED',e,flush=True)
 for name,x in [('draft-manifest.json',m),('draft-attrition.json',audit),('context-options.json',options_out),('documentation-omissions-draft.json',policy)]:(R/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
