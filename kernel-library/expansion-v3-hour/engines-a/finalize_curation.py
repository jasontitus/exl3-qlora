from pathlib import Path
import json,hashlib,re,shutil,collections
R=Path(__file__).resolve().parent;W=R.parents[2];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'manifest-review-candidate.json').read_text())
for name in ['reserved-metadata.json','reserved-task-contracts.json']:
 shutil.copyfile(W/'kernel-library/integration-v1/group-a/evidence'/name,R/'evidence'/name)
reserved=json.loads((R/'evidence/reserved-metadata.json').read_text())['review'];ids={x[k]for x in reserved['coverage']for k in ['head','base']if x.get(k)}
for p in (W/'kernel-library/expansion-v2/corpus-12288-v3').glob('*.provenance.json'):
 x=json.loads(p.read_text())
 if x['split']=='test':ids.update(x['source']['commits'])
prior={};priorfiles=[]
for root in [W/'kernel-library/integration-v1/corpus-32768-v3',W/'kernel-library/expansion-v2']:
 for p in root.glob('**/*.provenance.json'):
  x=json.loads(p.read_text());s=x.get('source',{});v=s.get('stable_patch_id')
  if v:prior[v]=str(p);priorfiles.append(str(p))
# Include our prior source packs, even if omitted from a parent's context-limited export.
for root in [W/'kernel-library/expansion-v2/group-a/sourcepacks',W/'kernel-library/expansion-v2/group-a/batch2/sourcepacks']:
 for p in root.glob('*/complete-record.json'):
  x=json.loads(p.read_text());prior[x['stable_patch_id']]=str(p);priorfiles.append(str(p))
rows=[];seen={}
def tidy(s):
 s=re.sub(r'\b(at|to|through|from|about|around|reported|its|of|by|with|a|an|and|for|in|the|all|on|as|then|only|every|per)(?=\d)',r'\1 ',s)
 s=re.sub(r'(\d)(?=(rows|threads|lanes|tokens|bytes|GiB|GB|MB|ms)\b)',r'\1 ',s)
 return s
pins={x['head']:x for n in [1,2]for x in json.loads((R/f'source-pins-batch{n}.json').read_text())}
for c in m['candidates']:
 rec=json.loads((R/'sourcepacks'/c['id']/'complete-record.json').read_text());stable=rec['stable_patch_id'];refs=set(re.findall(r'\b[0-9a-f]{40}\b',rec['diff']+(R/'sourcepacks'/c['id']/'commit-message.txt').read_text()));matches=sorted(({c['commit'],c['parent']}|refs)&ids);dup=prior.get(stable)or seen.get(stable);seen[stable]=c['id']
 c['lineage']='patch:'+stable;c['lineage_scope']='same historical Git patch; related revisions also share conservative source_group'
 c['historical_protection_review']={'status':'protected'if matches else'cleared','reviewed_by':'Codex quality_research finite public-source/metadata review2026-10-13','reason':'No captured protected head/base or explicit40-hex reference match. Related source revisions/forks stay in one training family; operation-family overlaps remain disclosed and must not be treated as independent evaluation. No unavailable protected winner source was accessed.','evidence_sha256':{'evidence/reserved-metadata.json':sha(R/'evidence/reserved-metadata.json'),'evidence/reserved-task-contracts.json':sha(R/'evidence/reserved-task-contracts.json')}}
 c['license_provenance']=pins[c['commit']]['licenses'];c['source_only_evidence']=True;c['execution_validation']='not_run'
 if c['repository']=='sushi'and not any(l['source_path']=='LICENSE'for l in c['license_provenance']):c['license_limitation']='Pinned NOTICE declares licensing and third-party attribution but referenced root LICENSE file is absent; no independent redistribution audit.'
 if c['commit'].startswith('b64d3e'):c['historical_protection_review']['operation_family_exposure']='Existing FP8/FP4-rounding phases are reproduced inside a different compressor-finalization fusion. Old FP8-rounding task is not an independent quality evaluation for this sample.'
 if c['commit'].startswith('7d7cadd'):
  eid='later-prefetch-regression';p=R/'evidence/splash-later-metadata-prefetch-regression.txt';m['evidence'][eid]={'path':str(p.relative_to(R)),'sha256':sha(p),'url':'https://github.com/incoai/splash/commit/74f0fee245fe9d7db6d0253f508bd96fcfb711d0','phase':'post_change'};c['annotations']['architecture']['evidence'].append({'source':eid,'quote':p.read_text()});c['related_commits']=['74f0fee245fe9d7db6d0253f508bd96fcfb711d0']
 if c['commit'].startswith('dce06'):
  c['annotations']['operation']['text']='evaluated offset BF16 KV views in vectorized attention staging'
  c['annotations']['constraints']['text']='Repair the captured six-byte-offset failure for evaluated K/V views while retaining the aligned fast path. The captured host helper accepts unevaluated arrays; that unresolved broader lazy-view case is outside this historical repair, not proven safe.'
 if c['commit'].startswith('0658'):
  c['annotations']['architecture']['text']='Track whether the current layer actually armed inverse-RoPE fusion before consuming its process-global completion flag; retain standalone inverse RoPE otherwise. Keep the nondeterministic streaming top-k path opt-in, and add the device-memory fence needed when FP8-store lanes read other lanes’ RoPE writes. These source-level repairs have not been executed by this library.'
 for a in c['annotations'].values():a['text']=tidy(a['text'])
 if matches or dup:c['curriculum_eligibility']='excluded_protection_or_duplicate'
 rows.append({'case_id':c['id'],'stable_patch_id':stable,'protected_matches':matches,'previous_stable_patch_match':dup,'source_group':c['source_group'],'lineage':c['lineage'],'candidate_eligibility':c['curriculum_eligibility'],'evidence_tier':c['curation_outcome']})
(R/'protection-dedup-review.json').write_text(json.dumps({'scope':'Finite public source/metadata screening, not proof of semantic independence or an exhaustive upstream negative-result search','protected_identities':len(ids),'prior_files_compared':priorfiles,'cases':rows},indent=2)+'\n')
# This remains a candidate name until reciprocal review; sealing preserves these exact fields.
(R/'manifest-review-candidate.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
obj=json.loads((R/'reasoning-records-v2.json').read_text())
for row in obj['records']:
 short=row['commit'][:12];case=next(p for p in (R/'sourcepacks').glob('*'+short+'*'));rec=json.loads((case/'complete-record.json').read_text());row['source_group']=row['lineage'];row['lineage']='patch:'+rec['stable_patch_id'];row['prompt']=tidy(row['prompt']);row['completion']=tidy(row['completion'])
 if short=='7d7cadd19113':row['related_commits']=['74f0fee245fe9d7db6d0253f508bd96fcfb711d0']
(R/'reasoning-records-v3.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
print('candidates',len(rows),'protected_matches',sum(bool(x['protected_matches'])for x in rows),'prior_patch_duplicates',sum(bool(x['previous_stable_patch_match'])for x in rows))
