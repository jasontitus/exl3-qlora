"""Seal reviewed TensorFold sources, preserving exact license and review evidence."""
import hashlib,json,sys
from pathlib import Path
R=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
m=json.loads((R/'manifest-scoped-draft.json').read_text())
review_path=R/'evidence/admission-review.json'
if not review_path.exists():raise SystemExit('Independent admission review is not available; refusing to seal')
review=json.loads(review_path.read_text())
# Review must explicitly name each candidate, with a reason and disposition.
byid={r['id']:r for r in review['candidates']}
if set(byid)!={c['id'] for c in m['candidates']}:raise ValueError('Review/candidate identity mismatch')
licenses={}
if (R/'license-receipts.json').exists():
 for r in json.loads((R/'license-receipts.json').read_text()):
  if r['status']=='captured':licenses[r['commit']]=r
else:
 for p in json.loads((R/'source-pins.json').read_text()):
  if len(p['licenses'])!=1:raise ValueError('Expected exact one captured license')
  licenses[p['head']]=p['licenses'][0]
for c in m['candidates']:
 r=byid[c['id']]
 if r['status'] not in {'cleared','pending','protected'}:raise ValueError('Invalid review')
 c['historical_protection_review']={'status':r['status'],'reason':r['reason'],'reviewed_by':review['reviewed_by'],'evidence_sha256':{'evidence/admission-review.json':H(review_path.read_bytes())}}
 lic=licenses[c['commit']];path=R/lic['path'];assert H(path.read_bytes())==lic['sha256']
 m['evidence']['license_'+c['commit'][:12]]={'path':lic['path'],'sha256':lic['sha256'],'phase':'post_change','url':'https://github.com/ashhart/TensorFold/blob/'+c['commit']+'/LICENSE'}
 c['license_evidence']='license_'+c['commit'][:12]
 c['validation_status']={'full_non_documentation_patch_replay':'passed','source_performance':'author_reported','local_compilation':'not_run','local_numerical_parity':'not_run','local_performance':'not_run','agent_trajectory':'retrospective_single_edit','independent_family_generalization':'not_established'}
 c['reviewed_units']['dependency_review']='Complete named operation units or complete core files, with all original target edits and baseline anchors retained. Full source sidecars carry dependencies; this is a repository editing task, not a standalone kernel compilation closure.'
m.pop('manifest_sha256',None);m['manifest_sha256']=H(C(m));(R/'manifest-reviewed.json').write_text(json.dumps(m,indent=2)+'\n')
d=json.loads((R/'documentation-omissions.json').read_text());ids={c['id'] for c in m['candidates'] if c['outcome']=='reported_improvement' and c['task_kind']['value'] in {'kernel','host_dispatch'}}
(R/'documentation-omissions-reviewed.json').write_text(json.dumps({k:v for k,v in d.items() if k in ids},indent=2)+'\n')
print(m['manifest_sha256'])
