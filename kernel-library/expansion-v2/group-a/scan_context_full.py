from pathlib import Path
import json,sys,difflib,hashlib,re
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'tools'))
from source_units import scan_units,lex,statement_end,skip_space,leading_comments
def top_zig(raw):
 code,pairs,_=lex(raw,"zig");units=[]
 for m in re.finditer(r"(?m)^(?:(?:pub|export|extern|inline|noinline)\s+)*(fn|const|var)\s+([A-Za-z_]\w*)\b",code):
  start=m.start();end=None
  if m[1]!="fn":end=statement_end(code,pairs,m.end());kind="complete_declaration"
  else:
   par=skip_space(code,m.end())
   if code[par]!="(":continue
   i=pairs[par]+1
   while i<len(code):
    if code[i] in "([":i=pairs[i]+1;continue
    if code[i]==";":break
    if code[i]=="{":
     prior=re.search(r"([A-Za-z_]+)\s*$",code[pairs[par]+1:i])
     if prior and prior[1] in {"error","struct","union","enum","opaque"}:i=pairs[i]+1;continue
     end=pairs[i]+1;break
    i+=1
   kind="complete_function"
  if end is not None:
   start=leading_comments(raw,start)
   if raw[end:end+1]==b"\n":end+=1
   units.append({"name":m[2],"kind":kind,"start":start,"end":end,"sha256":hashlib.sha256(raw[start:end]).hexdigest()})
  # Also expose complete MSL definitions inside Zig multiline string blocks.
 lines=raw.splitlines(keepends=True);off=[];n=0
 for line in lines:off.append(n);n+=len(line)
 i=0
 while i<len(lines):
  if not re.match(rb"[ \t]*\\\\",lines[i]):i+=1;continue
  first=i;stripped=[];mapping=[];z=0
  while i<len(lines) and re.match(rb"[ \t]*\\\\",lines[i]):
   m=re.match(rb"[ \t]*\\\\",lines[i]);part=lines[i][m.end():];mapping.append((z,off[i],len(lines[i])));stripped.append(part);z+=len(part);i+=1
  try:embedded=scan_units(b''.join(stripped),'metal')['units']
  except ValueError:continue
  for u in embedded:
   if u.get('enclosing_names') or u['kind']=='complete_declaration':continue
   indices=[j for j,(at,_,_) in enumerate(mapping) if at<u['end'] and (mapping[j+1][0] if j+1<len(mapping) else z)>u['start']]
   if not indices:continue
   start=mapping[indices[0]][1];last=mapping[indices[-1]];end=last[1]+last[2]
   units.append({'name':'embedded.'+u['name'],'kind':'complete_kernel' if u['kind']=='complete_function' else 'complete_declaration','start':start,'end':end,'sha256':hashlib.sha256(raw[start:end]).hexdigest(),'embedded_metal':True})
 return {"units":units}

rows=[]
for recpath in sorted((R/'sourcepacks').glob('*/complete-record.json')):
 rec=json.loads(recpath.read_text());case=recpath.parent.name;opts=[]
 for path,p in rec['files'].items():
  if not p['before'] or not p['after']:continue
  ext=Path(path).suffix
  if ext not in ['.zig','.c','.h','.cpp','.hpp','.metal','.mm']:continue
  lang='zig' if ext=='.zig' else 'metal' if ext=='.metal' else 'c'
  b=p['before']['content'].encode();a=p['after']['content'].encode()
  try:bs=scan_units(b,lang)['units'];ass=scan_units(a,lang)['units']
  except Exception as e:opts.append({'path':path,'error':str(e)});continue
  byname={}
  for v in ass:byname.setdefault(v.get('qualified_name',v['name']),[]).append(v)
  for u in bs:
   name=u.get('qualified_name',u['name']);matches=byname.get(name,[])
   if len(matches)!=1:continue
   v=matches[0]
   if b[u['start']:u['end']]==a[v['start']:v['end']]:continue
   if u['kind']=='complete_declaration' and u.get('enclosing_names'):continue
   if u['end']-u['start']>40000:continue
   opts.append({'path':path,'name':name,'kind':u['kind'],'before_bytes':u['end']-u['start'],'after_bytes':v['end']-v['start'],'before':{k:u[k] for k in ['start','end','sha256']},'after':{k:v[k] for k in ['start','end','sha256']},'conditional':u.get('requires_conditional_review',False),'preview':b[u['start']:min(u['end'],u['start']+200)].decode()})
 rows.append({'case_id':case,'options':opts});print(case,len(opts),flush=True)
 (R/'scanned-context-options-full.json').write_text(json.dumps(rows,indent=2)+'\n')
