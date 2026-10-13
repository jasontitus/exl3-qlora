from pathlib import Path
import json,hashlib,sys
R=Path(__file__).resolve().parent;W=R.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
m=json.loads((R/'manifest-review-candidate.json').read_text());units=json.loads((R/'source-unit-review.json').read_text());audit=json.loads((R/'reviewed-length-audit.json').read_text());byid={x['case_id']:x for x in units['cases']};rows={x['case_id']:x for x in audit['cases']};ids={c['id']for c in m['candidates']};assert len(ids)==25==len(rows)
prior={}
for p in (W/'kernel-library/integration-v1/corpus-32768-v3').glob('*.provenance.json'):
 x=json.loads(p.read_text());prior[x['source']['stable_patch_id']]=x['id']
seen=set();duplicates=[]
for c in m['candidates']:
 cid=c['id'];x=rows[cid];assert 'error' not in x and x['exact_replay'] and x['fits32768']
 patch=byid[cid]['stable_patch_id'];assert patch not in seen;seen.add(patch)
 if patch in prior:duplicates.append({'case_id':cid,'prior_case_id':prior[patch]})
 for i,lic in enumerate(byid[cid]['licenses']):
  p=R/lic['path'];assert sha(p.read_bytes())==lic['sha256'];k='license_'+c['repository'].replace('-','_')+'_'+c['commit'][:12]+'_'+str(i)
  m['evidence'][k]={'path':lic['path'],'sha256':lic['sha256'],'url':m['repositories'][c['repository']]['url']+'/blob/'+c['commit']+'/'+lic['source_path'],'phase':'unknown'}
 c['source_license_evidence']=[k for k in m['evidence'] if k.startswith('license_'+c['repository'].replace('-','_')+'_'+c['commit'][:12]+'_')]
assert not duplicates,duplicates
m['manifest_sha256']=sha(json.dumps(m,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode());write(R/'manifest-reviewed.json',m)
policy={k:v for k,v in json.loads((R/'documentation-omissions-draft.json').read_text()).items() if k in ids};write(R/'documentation-omissions.json',policy)
scopes={}
for cap in [12288,32768]:
 cases=[c for c in m['candidates'] if rows[c['id']]['serialized_tokens']<=cap and c['historical_protection_review']['status']=='cleared'];count={}
 for c in cases:count[c['repository']]=count.get(c['repository'],0)+1
 for kind,filename in [('raw','raw.json'),('prepared','prepared.json')]:
  out=R/f'train-{cap}.{kind}.jsonl';out.write_text(''.join(json.dumps(json.loads((R/'reviewed'/c['id']/filename).read_text()),ensure_ascii=False,separators=(',',':'))+'\n' for c in cases))
 scopes[str(cap)]={'count':len(cases),'repositories':count,'serialized_tokens':sum(rows[c['id']]['serialized_tokens']for c in cases),'supervised_tokens':sum(rows[c['id']]['supervised_tokens']for c in cases),'case_ids':[c['id']for c in cases]}
report={'format':'expansion-v2-group-a-first-pack-v1','status':'held_source_replayed_not_execution_verified','selected_candidates':25,'scopes':scopes,'source_reported_performance_only':True,'execution_verified':0,'truncation':False,'all_code_build_test_edits_retained':True,'omitted_documentation_files':sum(len(v)for v in policy.values()),'duplicates_against_prior_archival_corpus':duplicates,'prior_provenance_files_compared':len(prior),'candidate_length_rows':audit['cases'],'tokenizer':audit['tokenizer'],'historical_protection_scope':'Finite exact metadata identity and operation-source review; no universal semantic independence claim. Whole llama.cpp remains reserved. Related source revisions/forks stay connected training families.','limitations':['Retrospective source-conditioned imitation, not real agent trajectories.','Complete selected named context units do not imply standalone compile/dependency closure; full repository files, parent/head and raw patch remain available.','Upstream timing, parity and neutral/regression claims are not independently reproduced by this import.','The source-specific later PV attribution correction is included. No hardware-portability or generated-kernel win claim.'],'artifacts_sha256':{},'helper_source_sha256':{str(p.relative_to(W)):sha(p.read_bytes())for p in [W/'loss-fix/apple/src/exl3_mlx/agent_edits.py',W/'loss-fix/apple/src/exl3_mlx/data.py',W/'loss-fix/apple/src/exl3_mlx/tool_data.py',W/'loss-fix/apple/src/exl3_mlx/upstream_data.py',R/'review_first_pack.py',R/'tokenize_reviewed.py',R/'seal_first_pack.py']}}
for fn in ['manifest-reviewed.json','documentation-omissions.json','source-unit-review.json','reviewed-length-audit.json','source-pins.json','selection.json','train-12288.raw.jsonl','train-12288.prepared.jsonl','train-32768.raw.jsonl','train-32768.prepared.jsonl']:
 report['artifacts_sha256'][fn]=sha((R/fn).read_bytes())
write(R/'integration-report.json',report);print(json.dumps({'scopes':scopes,'manifest_file_sha256':report['artifacts_sha256']['manifest-reviewed.json'],'canonical_manifest_sha256':m['manifest_sha256']},indent=2))
