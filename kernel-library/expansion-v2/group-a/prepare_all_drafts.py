"""CPU source reconstruction and draft complete-unit selection; final review is separate."""
from pathlib import Path
import hashlib,json,re,sys,difflib
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path.insert(0,str(W/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate
sha=lambda b:hashlib.sha256(b).hexdigest()
selection=json.loads((R/'selection-merged.json').read_text())['candidates'];pins=json.loads((R/'source-pins-merged.json').read_text());metadata={x['commit']:x for x in selection};manifest={'format':'exl3-mlx-upstream-commits-v1','repositories':{},'candidates':[],'evidence':{},'notes':['Draft complete named-source-unit selection for independent human/agent review. Native tool target preserves every code/build/test edit; only individually reviewed markdown/rst docs omitted.','Retrospective one-edit imitation, not collected agent trajectories. Upstream numerical/timing statements are source reports, not newly executed measurements.']};audit=[];policy={}
operations={'89283f810986':'Batched quantized matrix multiplication for8–16 decode rows','340016fcb017':'MoE down projection and gated reduction','59d77c06fb9c':'Affine-4 MoE expert weight binding and command-buffer dispatch','a3b04e3ef804':'Gated DeltaNet blocked prefill recurrence','bfe518f77e4d':'Sparse Qwen attention prompt processing','499a08c6c80d':'Qwen sparse-attention decode key-mask construction','242c0a50f1ad':'One-row router softmax and shared-expert gate','b95fbb4afe9e':'Single-query sparse attention over selected keys','364b9fd0760b':'Pooled-key maintenance for sparse-attention indexing','55d834b0460a':'Bonsai ternary matrix-vector decode','f11faec370ab':'EXL3 lane decode and batched expert verification across packing windows','26ef54dc1070':'EXL3 lane bitstream loading in quantized decode','27e81cfc2069':'MiMo global prefill attention matrix-fragment loading','8c16b2bb6a33':'EXL3 sorted expert prefill matrix multiplication on pre-M5 GPUs','3ba7272b3305':'MiMo prefill attention probability-value matrix multiplication','4bf284554f14':'Grouped expert weight reuse during MiMo multi-row verification','9942e8ea6c3d':'EXL3 quantized decode matrix-vector tiling','dd9c7cce4822':'EXL3 n40 decode bitstream funnels','e69d8873c54d':'MiMo sorted expert prefill routing metadata construction','74f0fee245fe':'Staged quantized projection metadata loading','20593c49f48f':'Staged quantized decode projection K traversal','07142b12bc86':'Staged routed-expert gate/up projection','8b4e1a58f222':'Dense quantized gate/up projection on Apple10 GPUs','6c00e2d53519':'DeepSeekV4.1 Engram random-row reads during decode','a154af774b40':'Grouped MoE down-projection block accumulation'}
operations.update({'a338ca2d2bef':'Packed-KV global decode attention at long context','748fab8a2e3a':'Packed global-attention decode without NAX matrix units','fde5bcfc26e6':'EXL3 independent LDLQ block processing at window eight','376c23c90e5c':'EXL3 trellis quantization launch sizing at window eight','a939818fb48f':'Expert-weight reconstruction error scoring during quantization','0382aee97906':'MXFP4 source-weight decoding during quantization','7de9af0f3023':'Grouped MTP draft and verification command dispatch','96578478da83':'Sparse-attention selected-key decode at long context'})
def functions(src):
 if re.search(r'(?m)^(?:async )?def ',src):
  import ast
  tree=ast.parse(src);lines=src.splitlines(keepends=True);off=[0]
  for line in lines:off.append(off[-1]+len(line))
  return [{'name':n.name,'start':off[min([n.lineno]+[d.lineno for d in n.decorator_list])-1],'end':off[n.end_lineno]} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]

 # Lexical candidates include native Metal definitions embedded as Zig multiline text.
 pattern=re.compile(r'(?m)^([ \t]*(?:\\\\[ \t]*)?)(?:(?:pub |inline |static |extern |export |noinline |constexpr |virtual |__attribute__[^\n]* )*)(?:fn\s+([A-Za-z_]\w*)|(?:kernel\s+void|void|bool|int|float|uint32_t|uint64_t|size_t|metal::[\w:<>]+)\s+([A-Za-z_]\w*(?:::\w+)*))\s*\(')
 rows=[]
 for match in pattern.finditer(src):
  start=match.start();name=match.group(2) or match.group(3);depth=0;seen=False
  for t in re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{};]',src[match.end():]):
   token=t.group()
   if token==';' and not seen:break
   if token=='{':depth+=1;seen=True
   elif token=='}':
    depth-=1
    if seen and depth==0:
     end=match.end()+t.end()
     if src[end:end+1]==';':end+=1
     if src[end:end+1]=='\n':end+=1
     # Preserve immediately preceding template declaration when it belongs to this definition.
     prev=src.rfind('\n',0,max(0,start-1))+1
     if src[prev:start].lstrip().startswith(('template','\\\\template')):start=prev
     rows.append({'name':name,'start':start,'end':end});break
 return rows
def changes(before,after):
 a=before.splitlines(keepends=True);b=after.splitlines(keepends=True);offsets=[0]
 for x in a:offsets.append(offsets[-1]+len(x))
 ops=difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes();return [(offsets[i],offsets[j]) for tag,i,j,k,l in ops if tag!='equal'],ops

def makeunit(path,kind,b,a,bs,be,as_,ae):return {'path':path,'kind':kind,'before':{'start':len(b[:bs].encode()),'end':len(b[:be].encode()),'sha256':sha(b[bs:be].encode())},'after':{'start':len(a[:as_].encode()),'end':len(a[:ae].encode()),'sha256':sha(a[as_:ae].encode())}}
for pin in pins:
 repo=pin['repository'];head=pin['head']
 if repo=='tensorfold':continue
 msg=metadata[head]['message'];title=msg.splitlines()[0];case=repo+'-'+head[:12]+'-expansion-v2';manifest['repositories'][repo]={'path':str(R/'mirrors'/f'{repo}.git'),'url':pin['url']}
 ann=lambda text,quote:{'text':text,'basis':'source','evidence':[{'source':'commit_message','quote':quote}]}
 paras=[x.strip() for x in msg.split('\n\n')[1:] if x.strip()];hw=re.search(r'(?:Apple\s+)?M[1-6](?:[–-]M[1-6])?(?:\s+(?:Max|Pro|Ultra))?',msg)
 c={'id':case,'repository':repo,'commit':head,'parent':pin['parent'],'source_group':'mlx-sushi-connected-training' if repo in ['mlx-serve','sushi'] else repo+'-related-optimization-training','protected':False,'outcome':'reported_improvement','split':'train','task_conditioning':'historical_source','context_policy':'complete_changed_files','task_kind':{'value':'host_dispatch' if head.startswith(('59d77','6c00')) else 'kernel','reviewed_by':'quality_research draft subject classification','evidence':[{'source':'commit_message','quote':title}]},'annotations':{'operation':ann(operations[head[:12]],title),'rationale':ann(msg,msg),'architecture':ann(title+'\n'+(paras[0] if paras else ''),msg)}}
 if hw:c['annotations']['hardware']=ann(hw.group()+' (upstream report; current training host is different).',hw.group())
 try:
  rec=reconstruct_candidate(c,manifest['repositories'],{})
  out=R/'sourcepacks'/case;out.mkdir(exist_ok=True,parents=True);(out/'complete-record.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');(out/'commit.diff').write_text(rec['diff'])
  for path,pair in rec['files'].items():
   for side in ['before','after']:
    if pair[side]:p=out/side/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(pair[side]['content'])
  units=[];desc=[];options=[]
  for path,pair in rec['files'].items():
   b=(pair['before'] or {'content':''})['content'];a=(pair['after'] or {'content':''})['content']
   if Path(path).suffix.lower() in ['.md','.rst']:
    if pair['after']:policy.setdefault(case,{})[path]={'before_sha256':sha(b.encode()) if pair['before'] else None,'after_sha256':sha(a.encode()),'reason':'Reviewable documentation-only file; full raw patch retained as provenance, all executable/build/test source edits preserved.'}
    continue
   if not b or len(b.encode())<3500:
    u=makeunit(path,'whole_file',b,a,0,len(b),0,len(a));options.append((100,0,len(b),u,'whole file'));continue
   spans,ops=changes(b,a);bf=functions(b);af=functions(a)
   for f in bf:
    overlap=sum(max(1,min(f['end'],y)-max(f['start'],x)) for x,y in spans if f['start']<=y and f['end']>=x)
    if not overlap:continue
    if f['end']-f['start']>22000:continue
    matches=[g for g in af if g['name']==f['name']]
    if len(matches)!=1:continue
    g=matches[0];u=makeunit(path,'complete_kernel' if 'kernel' in b[f['start']:f['start']+100] else 'complete_function',b,a,f['start'],f['end'],g['start'],g['end'])
    options.append((overlap,1,f['end']-f['start'],u,f['name']))
  # Prefer operation-bearing kernel declarations, then small full support files. Pick bounded complete units, never cropped spans.
  options.sort(key=lambda x:(0 if x[3]['path'].endswith('.metal') else 1,-x[0],x[2]));budget=24000
  for score,kind,size,u,name in options:
   if size>budget:continue
   others=[x for x in units if x['path']==u['path']]
   if any(not(u['before']['end']<=x['before']['start'] or u['before']['start']>=x['before']['end']) for x in others):continue
   units.append(u);desc.append({'path':u['path'],'name':name,'before_bytes':size});budget-=size
   if len(units)>=5:break
  if not units:raise ValueError('No bounded complete changed operation unit found; manual scope required')
  # Repository order within a path, required for exact source-gap accounting.
  units.sort(key=lambda u:(u['path'],u['before']['start'],u['after']['start']))
  omitted=[]
  for path,pair in rec['files'].items():
   b=(pair['before'] or {'content':''})['content'].encode();a=(pair['after'] or {'content':''})['content'].encode();bc=ac=0
   for u in [x for x in units if x['path']==path]:
    if b[bc:u['before']['start']]!=a[ac:u['after']['start']]:omitted.append(path)
    bc=u['before']['end'];ac=u['after']['end']
   if b[bc:]!=a[ac:]:omitted.append(path)
  c['context_policy']='reviewed_operation';c['reviewed_units']={'reviewed_by':'quality_research named-unit review pending final audit','dependency_review':'Complete named definitions or small complete support files. Embedded Metal remains exact source text in its host-language wrapper. External imports, generated substitutions, constants and other definitions are repository context; no standalone compile/dependency-closure claim.','units':units};c['operation_scope']={'description':'Historical operation-edit example with complete named baseline definitions. Full answer keeps all source/build/test edits; omitted prompt definitions remain in full source provenance and exact patch anchors.','omitted_path_reasons':{p:'Other complete repository definitions or documentation are outside the displayed operation context. All non-documentation target edits and original anchors are retained.' for p in sorted(set(omitted))}}
  reconstruct_candidate(c,manifest['repositories'],{})
  manifest['candidates'].append(c);audit.append({'case_id':case,'status':'draft_ready','stable_patch_id':rec['stable_patch_id'],'selected_context':desc,'diff_bytes':len(rec['diff'].encode()),'changed_paths':pin['paths'],'licenses':pin['licenses']});print(case,'units',len(units),'context',24000-budget,'diff',len(rec['diff']),flush=True)
 except Exception as e:audit.append({'case_id':case,'status':'blocked','error':str(e),'licenses':pin['licenses']});print(case,'BLOCKED',str(e),flush=True)
 (R/'draft-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');(R/'draft-attrition.json').write_text(json.dumps(audit,indent=2)+'\n');(R/'documentation-omissions-draft.json').write_text(json.dumps(policy,indent=2)+'\n')
