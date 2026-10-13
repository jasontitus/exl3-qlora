from pathlib import Path
import json,hashlib,sys,re,ast,difflib
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path[:0]=[str(W/'loss-fix/apple/src'),str(W/'kernel-library/expansion-v2/tools')]
from exl3_mlx.upstream_data import reconstruct_candidate
from source_units import scan_units,lex
sha=lambda b:hashlib.sha256(b).hexdigest()
notes=json.loads((R/'curation-notes.json').read_text());ops={k:v['operation'] for k,v in notes.items()};architecture={k:v['solution'] for k,v in notes.items()};classes={k:('source_reported_improvement' if v['tier']=='optimization' else v['tier']) for k,v in notes.items()}
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
 suffix=Path(path).suffix.lower()
 if suffix=='.py':
  try:
   tree=ast.parse(raw);lines=raw.splitlines(keepends=True);offsets=[0]
   for line in lines:offsets.append(offsets[-1]+len(line))
   out=[]
   for node in ast.walk(tree):
    if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
     start=offsets[min([node.lineno]+[d.lineno for d in node.decorator_list])-1];end=offsets[node.end_lineno];out.append({'name':node.name,'qualified_name':node.name,'kind':'complete_function','start':start,'end':end,'sha256':sha(raw[start:end])})
   return out
  except (SyntaxError,UnicodeDecodeError):return []
 lang='zig'if suffix=='.zig'else'metal'if suffix=='.metal'else'c'
 if suffix not in ['.zig','.metal','.m','.mm','.c','.cc','.cpp','.h','.hpp']:return[]
 try:return scan_units(raw,lang)['units']
 except ValueError:return c_fallback(raw)if lang!='zig'else[]

pins=[];metadata={}
for n in [2]:
 p=R/f'source-pins-batch{n}.json'
 if p.exists():pins+=json.loads(p.read_text())
 p=R/f'selection-batch{n}.json'
 if p.exists():metadata.update({x['commit']:x for x in json.loads(p.read_text())['candidates']})
m={'format':'exl3-mlx-upstream-commits-v1','repositories':{},'candidates':[],'evidence':{},'notes':['Draft source-grounded historical tasks, not captured agent rollouts. Correctness/neutral/rejected outcomes are not speed wins.']};audit=[];policy={};options_out=[]
m=json.loads((R/'draft-manifest.json').read_text());audit=json.loads((R/'draft-attrition.json').read_text());policy=json.loads((R/'documentation-omissions-draft.json').read_text());options_out=json.loads((R/'context-options.json').read_text())
for pin in pins:
 h=pin['head'];short=h[:12];repo=pin['repository'];case=repo+'-'+short+'-hour-v4';md=metadata[h];msg=md['message'];title=msg.splitlines()[0];ann=lambda t,q:{'text':t,'basis':'source','evidence':[{'source':'commit_message','quote':q}]};cl=classes.get(short,'source_reported_improvement');m['repositories'][repo]={'path':str(R/'mirrors'/f'{repo}.git'),'url':pin['url']}
 c={'id':case,'repository':repo,'commit':h,'parent':pin['parent'],'source_group':'mlx-sushi-connected-training'if repo in['mlx-serve','sushi']else repo+'-related-optimization-training','protected':False,'outcome':'negative'if cl=='rejected'else'reported_improvement'if cl=='source_reported_improvement'else'source_only','split':'train','task_conditioning':'historical_source','context_policy':'complete_changed_files','curation_outcome':cl,'task_kind':{'value':'kernel','reviewed_by':'Codex source review draft 2026-10-13','evidence':[{'source':'commit_message','quote':title}]},'annotations':{'task':ann(notes[short]['task'],msg),'operation':ann(ops[short],title),'rationale':ann(msg,msg),'architecture':ann(architecture[short],msg)}}
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
  c['lineage']='patch:'+rec['stable_patch_id'];c['context_policy']='reviewed_operation';c['reviewed_units']={'reviewed_by':'Codex lexical complete-unit candidate; semantic review pending','dependency_review':'Exact source-unit boundaries; external imports/templates/ABI/dispatch dependencies remain repository context. No standalone compilation guarantee. Full non-documentation edit target preserved.','units':units};c['operation_scope']={'description':'Historical operation context with complete named baseline definitions. Answer retains every non-documentation code/build/test edit.','omitted_path_reasons':{p:'Other repository context is not expanded; exact baseline anchors and all non-documentation target edits remain available.'for p in sorted(set(omitted))}}
  reconstruct_candidate(c,m['repositories'],{});m['candidates'].append(c);audit.append({'case_id':case,'status':'draft_ready','class':cl,'selected_names':names,'context_bytes':14000-budget,'stable_patch_id':rec['stable_patch_id'],'licenses':pin['licenses'],'changed_paths':pin['paths']});print(case,cl,14000-budget,names,flush=True)
 except Exception as e:audit.append({'case_id':case,'status':'blocked','class':cl,'error':str(e),'licenses':pin['licenses']});print(case,'BLOCKED',e,flush=True)
 for name,x in [('draft-manifest.json',m),('draft-attrition.json',audit),('context-options.json',options_out),('documentation-omissions-draft.json',policy)]:(R/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
