"""Attach an external peer receipt without changing reviewed payloads."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=ROOT/'case-index.json';d=json.loads(p.read_text());receipt=WORK/'kernel-library/expansion-v4-hour/engines-a/peer-reviews/mlx-independent-review.json';r=json.loads(receipt.read_text());assert r['schema']=='exact-edit-independent-review-v1'and r['status']=='approved'and r['reviewer']=='quality_research';by={x['id']:x for x in r['cases']}
for c in d['cases']:
 if not c.get('exact_edit_eligible'):continue
 signed=by[c['id']]['inputs_sha256'];expected={str(Path(v).resolve()):H(Path(v))for v in c['paths'].values()};assert signed==expected
 c['review']={'status':'approved','receipt':str(receipt),'sha256':H(receipt)}
d['status']='held_independently_reviewed';d['held_payloads']=True;d['reasoning_records']={'path':str(ROOT/'reasoning-records.json'),'sha256':H(ROOT/'reasoning-records.json'),'independent_review':str(receipt)}
p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');files=[p,ROOT/'reasoning-records.json',ROOT/'manifest-reviewed.json',ROOT/'documentation-omissions.json',receipt,ROOT/'inventory-all.json',ROOT/'evidence/finite-operation-review.json']
(ROOT/'sealed-lane.json').write_text(json.dumps({'status':'held_independently_reviewed','core32k':27,'core12k':19,'core_outcomes':{'source_reported_optimization':6,'compute_correctness':21},'opt_in_unmeasured':1,'quarantined':2,'reasoning_views':28,'prospective_diagnosis':17,'retrospective_analysis':11,'qualification':'Historical source reconstruction with byte-exact complete non-documentation edits and actual tokenizer masks. No local compiler/GPU/performance evidence.','inputs_sha256':{str(x):H(x)for x in files}},indent=2)+'\n')
