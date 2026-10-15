"""Deterministic metadata-only whole-family split; no training or score inputs."""
import collections,hashlib,json,re,os,stat
SEED='cache-dev-v2-preregistered-metadata-only-20261015'
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def length_bin(n):return '0-2048' if n<=2048 else '2049-4096' if n<=4096 else '4097-8192' if n<=8192 else '8193+'
def groups(rows,edges):
 n=len(rows)
 if not 1<=n<=512 or len(edges)>32768:raise ValueError('metadata bounds')
 if [r['train_row'] for r in rows]!=list(range(n)) or any(type(r['train_row']) is not int for r in rows):raise ValueError('row ordering')
 for r in rows:
  if not isinstance(r,dict):raise ValueError('row object')
  for k in ['id','repo','commit','kind','row_sha256']:
   if not isinstance(r.get(k),str) or not r[k]:raise ValueError('row identity')
  if not re.fullmatch('[0-9a-f]{64}',r['row_sha256']):raise ValueError('row hash')
  if r['kind'] not in ['exact_edit','diagnostic']:raise ValueError('kind')
  for k in ['allocation_labels','patch_ids','development_exclusion_reasons']:
   if not isinstance(r.get(k),list) or any(not isinstance(v,str) or not v for v in r[k]):raise ValueError('annotation type')
  if set(r)-{'train_row','id','repo','commit','kind','row_sha256','serialized_tokens','supervised_tokens','allocation_labels','patch_ids','development_exclusion_reasons'}:raise ValueError('unexpected metadata/score field')
  for k in ['serialized_tokens','supervised_tokens']:
   if type(r[k]) is not int or r[k]<=0:raise ValueError('token count')
  if r['supervised_tokens']>r['serialized_tokens']:raise ValueError('mask count')
 parent=list(range(n))
 def find(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 seen={}
 for r in rows:
  keys=['row:'+r['row_sha256'],'commit:'+r['repo']+':'+r['commit']]+['patch:'+v for v in r['patch_ids']]
  for key in keys:
   if key in seen:parent[find(r['train_row'])]=find(seen[key])
   else:seen[key]=r['train_row']
 for e in edges:
  a,b=e['rows']
  if any(type(i) is not int or not 0<=i<n for i in [a,b]) or not isinstance(e['reason'],str):raise ValueError('edge')
  parent[find(b)]=find(a)
 buckets=collections.defaultdict(list)
 for i in range(n):buckets[find(i)].append(i)
 out=[]
 for ids in buckets.values():
  gid=digest(sorted(rows[i]['row_sha256'] for i in ids));reasons=sorted({v for i in ids for v in rows[i]['development_exclusion_reasons']})
  out.append({'group_id':gid,'rows':ids,'development_eligible':not reasons,'exclusion_reasons':reasons})
 return sorted(out,key=lambda x:hashlib.sha256((SEED+x['group_id']).encode()).hexdigest())
def metrics(rows,indices):
 selected=[rows[i] for i in indices]
 return {'rows':len(selected),'by_repo':dict(collections.Counter(r['repo'] for r in selected)),'by_kind':dict(collections.Counter(r['kind'] for r in selected)),'by_length':dict(collections.Counter(length_bin(r['serialized_tokens']) for r in selected)),'supervised_tokens':sum(r['supervised_tokens'] for r in selected),'serialized_tokens':sum(r['serialized_tokens'] for r in selected)}
def score(rows,indices,target):
 total=metrics(rows,range(len(rows)));part=metrics(rows,indices);fraction=target/len(rows);value=0.0
 # Four equally weighted metadata objectives. No model loss or item answer is used.
 for dimension in ['by_repo','by_kind','by_length']:
  terms=[((part[dimension].get(k,0)-v*fraction)/max(1,v*fraction))**2 for k,v in total[dimension].items()]
  value+=sum(terms)/len(terms)
 value+=((part['supervised_tokens']-total['supervised_tokens']*fraction)/(total['supervised_tokens']*fraction))**2
 return value

def propose(rows,edges,target=41):
 if type(target) is not int or not 0<target<min(129,len(rows)):raise ValueError('target')
 gs=groups(rows,edges);dp={0:()}
 # Deterministic bounded heuristic: retain one lowest-objective subset per count after each group.
 # This is not an exhaustive global stratification optimizer.
 for g in gs:
  if not g['development_eligible']:continue
  for count,chosen in list(dp.items()):
   new=count+len(g['rows'])
   if new>target:continue
   ids=tuple(sorted(chosen+tuple(g['rows'])))
   if new not in dp or (score(rows,ids,target),ids)<(score(rows,dp[new],target),dp[new]):dp[new]=ids
 if target not in dp:raise ValueError('target unattainable without splitting/excluded family')
 dev=list(dp[target]);devset=set(dev);train=sorted(set(range(len(rows)))-devset)
 for e in edges:
  if (e['rows'][0] in devset)!=(e['rows'][1] in devset):raise ValueError('crossing family')
 for g in gs:g['assignment']='development' if g['rows'][0] in devset else 'future_train'
 return {'schema':'development-whole-family-proposal-v2','status':'planning_only_not_activated','seed':SEED,'target':target,'selection_algorithm':'One best metadata-objective subset per count in fixed hash group order; deterministic heuristic, not global optimum. Four equally weighted repo/kind/length/supervised-token objectives.','development_rows':dev,'future_train_rows':train,'groups':gs,'rows':rows,'edges':edges,'summary':{'development':metrics(rows,dev),'future_train':metrics(rows,train),'all':metrics(rows,range(len(rows))),'objective':score(rows,dev,target),'components':len(gs),'excluded_components':sum(not g['development_eligible'] for g in gs)},'crossing_known_edges':0,'fresh_initialization_required':True,'endpoint458_generalization_eligible':False,'activation_authorized':False}
if __name__=='__main__':
 import argparse,pathlib
 ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('output');args=ap.parse_args()
 fd=os.open(args.input,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as inp:
  st=os.fstat(inp.fileno())
  if not stat.S_ISREG(st.st_mode) or st.st_size>8*1024*1024:raise ValueError('metadata size/type')
  raw=inp.read(8*1024*1024+1)
  def identity(v):return(v.st_dev,v.st_ino,v.st_mode,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
  if len(raw)!=st.st_size or len(raw)>8*1024*1024 or identity(os.fstat(inp.fileno()))!=identity(st) or identity(os.lstat(args.input))!=identity(st):raise ValueError('metadata drift')
 x=json.loads(raw);result=propose(x['rows'],x['edges'],x.get('target',41));result['metadata_input_sha256']=hashlib.sha256(raw).hexdigest()
 with pathlib.Path(args.output).open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
