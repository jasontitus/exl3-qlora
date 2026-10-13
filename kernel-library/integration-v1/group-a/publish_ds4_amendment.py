"""Versioned metadata-only admission amendment after independent operation review."""
from pathlib import Path
import hashlib,json,shutil,sys
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
if len(sys.argv)!=2:raise SystemExit('Provide finalized independent review JSON path')
review_path=Path(sys.argv[1]).resolve();review=json.loads(review_path.read_text())
# Publication requires a finalized independent reviewer receipt, reviewed outside this script.
m=json.loads((R/'manifest-reviewed.json').read_text());oldseal=m.pop('manifest_sha256');assert hashlib.sha256(canon(m)).hexdigest()==oldseal
final=R/'evidence/ds4-independent-operation-review.json'
if final.exists():raise FileExistsError(final)
shutil.copyfile(review_path,final)
comparison=R/'evidence/ds4-public-contract-comparison.json';public=R/'evidence/public-source/ds4-b85a6174-dense.metal';task=R/'evidence/reserved-task-contracts.json'
c=next(c for c in m['candidates'] if c['repository']=='ds4');prior=c['historical_protection_review'];assert prior['status']=='pending'
c['historical_protection_review']={'status':'cleared','reviewed_by':'quality_research curator; independent adversarial_code review','reason':'Finite operation-scope training clearance after comparing exact task-pinned public source and concrete contracts. Reserved generic Q8 prefill requires half-rounded tiled/unpack-to-F16 bitwise behavior and explicitly excludes decode. This candidate adds8/16-row decode FP32 simdgroup GEMM, split-K and matvec tolerance, distinct entrypoints/dispatch and tests. No32-token contiguous lexical match in complete visible prompt/added target against relevant public reference units. Preserve shared DS4/Q8 source lineage; no independent-family, unseen-code or Q8-heldout efficacy claim. Derived protected standalone/winner source was not retrieved; this is finite evidence, not universal ancestry proof.','evidence_sha256':dict(prior['evidence_sha256'],**{str(p.relative_to(R)):sha(p) for p in [comparison,public,task,final,R/'evidence/public-source/receipt.json']})}
m['notes']=[('Prior v1 decision, superseded for DS4 by v3: '+x) if 'Pending DS4' in x else x for x in m['notes']]
m['notes'].append('V3 admits DS4 at finite operation scope after public-source and independent code review; originals/pending decision preserved. No model input/target bytes changed. All new rows remain source-reported, not locally execution verified.')
m['manifest_sha256']=hashlib.sha256(canon(m)).hexdigest();dest=R/'manifest-reviewed-v3.json'
with dest.open('x') as f:f.write(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
rows=[json.loads(s) for s in (R/'train.raw.jsonl').read_text().splitlines()];tok=[json.loads(s) for s in (R/'train.prepared.jsonl').read_text().splitlines()]
draw=json.loads((R/'quarantine.raw.jsonl').read_text());dtok=json.loads((R/'quarantine.prepared.jsonl').read_text());assert draw['case_id']==c['id']==dtok['case_id']
for x in [draw,dtok]:x['historical_protection_status']='cleared'
rows.append(draw);tok.append(dtok)
for path,data in [('train-v3.raw.jsonl',rows),('train-v3.prepared.jsonl',tok)]:
 with (R/path).open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in data))
r=json.loads((R/'integration-report.json').read_text());r['format']='group-a-upstream-integration-v3';r['prior_report_sha256']=sha(R/'integration-report.json');r['manifest_sha256']=m['manifest_sha256'];r['manifest_file_sha256']=sha(dest);r['counts']['ready_current12288']=4;r['counts']['protection_pending_quarantine']=0
rc=next(x for x in r['cases'] if x['case_id']==c['id']);rc['status']='fits_current_12288_operation_review_cleared';rc['historical_protection_status']='cleared'
r['limitations']=[x.replace('DS4Q8 batch remains quarantined because reservedQ8-prefill equivalence is unresolved.','DS4 Q8 decode is admitted only at finite reviewed operation scope; source-family independence and old-Q8 heldout efficacy claims remain disallowed.') for x in r['limitations']]
r['admission_amendment']={'case_id':c['id'],'old_status':'pending','new_status':'cleared','review_scope':'Distinct optimization operation; no model scores considered','independent_review':str(final.relative_to(R)),'input_ids_loss_masks_and_native_tool_edits_unchanged':True,'old_manifests_and_reports_preserved':True}
r['files_sha256'].update({str(p.relative_to(R)):sha(p) for p in [dest,comparison,public,task,final,R/'train-v3.raw.jsonl',R/'train-v3.prepared.jsonl',Path(__file__).resolve()]})
with (R/'integration-report-v3.json').open('x') as f:f.write(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
print('HELD',dest,m['manifest_sha256']);print('report',sha(R/'integration-report-v3.json'))
