from pathlib import Path
import json,hashlib,re,shutil
R=Path(__file__).resolve().parent;W=R.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'manifest-review-candidate.json').read_text());pins={x['head']:x for n in [1,2]for x in json.loads((R/f'source-pins-batch{n}.json').read_text())}
for name in ['reserved-metadata.json','reserved-task-contracts.json']:
 shutil.copyfile(W/'kernel-library/integration-v1/group-a/evidence'/name,R/'evidence'/name)
reserved=json.loads((R/'evidence/reserved-metadata.json').read_text())['review'];ids={x[k]for x in reserved['coverage']for k in ['head','base']if x.get(k)}
for p in (W/'kernel-library/expansion-v2/corpus-12288-v3').glob('*.provenance.json'):
 x=json.loads(p.read_text())
 if x['split']=='test':ids.update(x['source']['commits'])
prior={};priorfiles={}
for root in [W/'kernel-library/integration-v1/corpus-32768-v3',W/'kernel-library/expansion-v2']:
 for p in root.glob('**/*.provenance.json'):
  x=json.loads(p.read_text());v=x.get('source',{}).get('stable_patch_id')
  if v:prior[v]=str(p);priorfiles[str(p)]=sha(p)
for root in [W/'kernel-library/expansion-v2/group-a',W/'kernel-library/expansion-v3-hour']:
 for p in root.glob('**/complete-record.json'):
  raw=p.read_bytes();v=re.search(rb'"stable_patch_id"\s*:\s*"([0-9a-f]+)"',raw)
  if v:prior[v[1].decode()]=str(p);priorfiles[str(p)]=hashlib.sha256(raw).hexdigest()
 for ip in root.glob('**/case-index.json'):
  obj=json.loads(ip.read_text());cases=obj.get('cases',[])if isinstance(obj,dict)else obj
  for c in cases:
   v=c.get('lineage','')
   if v.startswith('patch:'):prior[v[6:]]=str(ip);priorfiles[str(ip)]=sha(ip)
rows=[];seen={}
for c in m['candidates']:
 rec=json.loads((R/'sourcepacks'/c['id']/'complete-record.json').read_text());stable=rec['stable_patch_id'];refs=set(re.findall(r'\b[0-9a-f]{40}\b',rec['diff']+(R/'sourcepacks'/c['id']/'commit-message.txt').read_text()));matches=sorted(({c['commit'],c['parent']}|refs)&ids);dup=prior.get(stable)or seen.get(stable);seen[stable]=c['id'];c['lineage']='patch:'+stable;c['lineage_scope']='same exact historical patch; related revisions also retain source_group and supplemental linked-family labels'
 status='protected'if matches else'cleared';reason='No captured protected commit/base or explicit full-hash source reference matches. Related engine forks/changes remain grouped; this finite review does not prove semantic independence or inspect unavailable protected winners.'
 ev={f'evidence/{n}':sha(R/'evidence'/n)for n in ['reserved-metadata.json','reserved-task-contracts.json']}
 if c['commit'].startswith('b4da'):
  ev['evidence/ds4-b4da-contract/review.json']=sha(R/'evidence/ds4-b4da-contract/review.json');ev['evidence/ds4-b4da-independent.json']=sha(R/'evidence/ds4-b4da-independent.json');reason='Public before/after host correction and underlying FP32-staging Qwen4 kernel compared with reserved generic half-rounding Q8 contract. Implementation-scope admission; related Q8 exposure remains disclosed, not independent Q8 evaluation.'
 if c['commit'].startswith('909913'):
  status='pending';reason='Large multi-operation residual/normalization fusion bundle not yet resolved against reserved residual operation contract; excluded from this training release independently of context length.'
 c['historical_protection_review']={'status':status,'reviewed_by':'quality_research public-source/metadata finite review2026-10-13','reason':reason,'evidence_sha256':ev}
 c['license_provenance']=pins[c['commit']]['licenses'];c['source_only_evidence']=True;c['execution_validation']='not_run'
 if c['repository']=='sushi'and not any(l['source_path'].lower().startswith('license')for l in c['license_provenance']):c['license_limitation']='Captured pinned NOTICE/README licensing and third-party attribution; referenced root LICENSE absent. No independent legal audit.'
 if matches or dup or status!='cleared':c['curriculum_eligibility']='excluded_protection_or_duplicate'
 if c['commit'].startswith('3d14'):
  c['related_commits']=[x['head']for x in pins.values()if x['head'].startswith('d4fba')];c['curriculum_eligibility']='reasoning_or_archive_only'
 rows.append({'case_id':c['id'],'stable_patch_id':stable,'protected_matches':matches,'previous_stable_patch_match':dup,'protection_status':status,'source_group':c['source_group'],'lineage':c['lineage'],'candidate_eligibility':c['curriculum_eligibility'],'evidence_tier':c['curation_outcome']})
(R/'protection-dedup-review.json').write_text(json.dumps({'scope':'Finite metadata/source review; no universal semantic-independence or exhaustive negative-outcome claim','protected_identity_count':len(ids),'prior_files_sha256':priorfiles,'cases':rows},indent=2)+'\n')
m.pop('manifest_sha256',None);m['manifest_sha256']=hashlib.sha256(json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();(R/'manifest-reviewed.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n');shutil.copyfile(R/'documentation-omissions-draft.json',R/'documentation-omissions.json')
print('sealed candidate manifest',len(rows),'protected/pending',sum(x['protection_status']!='cleared'for x in rows),'duplicate',sum(bool(x['previous_stable_patch_match'])for x in rows))
