from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;G=R.parent;W=R.parents[3];sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((R/'manifest-reviewed.json').read_text());audit=json.loads((R/'reviewed-length-audit.json').read_text());rows={x['case_id']:x for x in audit['cases']};units=json.loads((R/'source-unit-review.json').read_text())['cases'];prior={}
for p in (W/'kernel-library/integration-v1/corpus-32768-v3').glob('*.provenance.json'):
 x=json.loads(p.read_text());prior[x['source']['stable_patch_id']]=x['id']
for x in json.loads((G/'source-unit-review.json').read_text())['cases']:prior[x['stable_patch_id']]=x['case_id']
assert len(rows)==len(m['candidates'])==8
for x in units:assert x['stable_patch_id']not in prior,(x['case_id'],prior.get(x['stable_patch_id']));prior[x['stable_patch_id']]=x['case_id']
scopes={}
for cap in [12288,32768]:
 cases=[c for c in m['candidates']if rows[c['id']]['serialized_tokens']<=cap and c['historical_protection_review']['status']=='cleared'];role={}
 for c in cases:role[c['curation_role']]=role.get(c['curation_role'],0)+1
 for kind,fn in [('raw','raw.json'),('prepared','prepared.json')]:
  (R/f'train-{cap}.{kind}.jsonl').write_text(''.join(json.dumps(json.loads((R/'reviewed'/c['id']/fn).read_text()),ensure_ascii=False,separators=(',',':'))+'\n'for c in cases))
 for x in rows.values():assert x['exact_replay'] and 'error'not in x
 scopes[str(cap)]={'count':len(cases),'roles':role,'serialized_tokens':sum(rows[c['id']]['serialized_tokens']for c in cases),'supervised_tokens':sum(rows[c['id']]['supervised_tokens']for c in cases),'case_ids':[c['id']for c in cases]}
report={'format':'expansion-v2-group-a-batch2-v1','status':'held_source_replayed_not_execution_verified','candidates':8,'scopes':scopes,'lengths':audit['cases'],'tokenizer':audit['tokenizer'],'duplicates_against_first_pack_and_prior_archival':[],'source_claims_only':True,'execution_verified':0,'no_truncation':True,'all_non_documentation_edits_replayed':True,'limits':['Four conversion-compute examples are auxiliary integration tasks, not native Metal-kernel examples.','The MXFP4 decode tile change is CPU NumPy; trellis launch/LDLQ batching and metric readback affect Metal/MLX conversion workloads.','Related source revisions share training lineage. No new independent family or held-out quality claim.','Mixed/neutral cases and timing limitations are retained in exact upstream rationale; no candidate execution here.'],'artifacts_sha256':{},'helper_source_sha256':{str(p.relative_to(W)):sha(p.read_bytes())for p in [W/'loss-fix/apple/src/exl3_mlx/upstream_data.py',W/'loss-fix/apple/src/exl3_mlx/agent_edits.py',W/'loss-fix/apple/src/exl3_mlx/data.py',W/'loss-fix/apple/src/exl3_mlx/tool_data.py',G/'review_batch2.py',R/'tokenize_batch.py',R/'seal_report.py']}}
for p in [R/'manifest-reviewed.json',R/'documentation-omissions.json',R/'source-unit-review.json',R/'reviewed-length-audit.json',*R.glob('train-*.jsonl')]:report['artifacts_sha256'][p.name]=sha(p.read_bytes())
(R/'integration-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'scopes':scopes,'manifest_sha256':report['artifacts_sha256']['manifest-reviewed.json']},indent=2))
