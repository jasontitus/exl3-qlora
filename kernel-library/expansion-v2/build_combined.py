"""Combine sealed reviewed batches without modifying running or prior corpora."""
import argparse,hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent; W=R.parents[1]
sys.path[:0]=[str(W/'loss-fix/apple/src'), str(R.parent/'integration-v1')]
from build_next_corpus import build_corpus
sys.path.insert(0,str(R/'tools'))
from reserved_identity import check_manifests, verify_snapshot
p=argparse.ArgumentParser();p.add_argument('--batch',action='append',required=True);p.add_argument('--output',required=True);p.add_argument('--max-length',type=int,required=True);a=p.parse_args()
old=R.parent/'integration-v1'
prior=json.loads((old/'corpus-12288-v3/ktune-import.json').read_text())
train=[s for s in prior['source_manifests'] if '/preserved/llama/' not in s]
test=[s for s in prior['source_manifests'] if '/preserved/llama/' in s]
for name,sha in prior['source_manifests'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError('Prior manifest changed: '+name)
policy=json.loads((old/'combined-documentation-omissions-v3.json').read_text())
for batch in a.batch:
 item=Path(batch).resolve();manifest=item if item.is_file() else item/'manifest-reviewed.json';b=manifest.parent;train.append(str(manifest))
 m=json.loads(manifest.read_text());ids={c['id'] for c in m['candidates'] if c['outcome'] in {'reported_improvement','verified_improvement'} and c['task_kind']['value'] in {'kernel','host_dispatch'}}
 d=b/('documentation-omissions-v2.json' if manifest.name=='manifest-reviewed-v2.json' else 'documentation-omissions-reviewed.json')
 if not d.exists():d=b/'documentation-omissions.json'
 for k,v in json.loads(d.read_text()).items():
  if k not in ids:continue
  if k in policy:raise ValueError('Duplicate policy case '+k)
  policy[k]=v
identity_review=check_manifests(train,old/'group-a/evidence/reserved-metadata.json',expected_sha256='f4eef3821f220de8903d19de9452bdd524648215cc99c0af8ee3106b48aab92b')
out=Path(a.output).resolve();pol=out.with_name(out.name+'-documentation-policy.json')
raw=json.dumps(policy,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()+b'\n'
if pol.exists() and pol.read_bytes()!=raw:raise ValueError('Refuse overwrite changed policy')
pol.write_bytes(raw)
result=build_corpus(train,test,'/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw',str(out),max_length=a.max_length,documentation_omissions=str(pol))
try:
 verify_snapshot(identity_review)
except Exception as error:
 (out/'REJECTED.json').write_text(json.dumps({'reason':str(error)})+'\n')
 raise
(out/'reserved-identity-preflight.json').write_text(json.dumps(identity_review,indent=2)+'\n')
print(json.dumps(result,indent=2))
