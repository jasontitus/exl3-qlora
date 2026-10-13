"""Optional verifier/preference corpus; synthetic rejected answers are never SFT targets."""
import argparse,hashlib,json
from pathlib import Path
from reasoning_data import validate,reserved_ids

def export(records,output):
 p=Path(records).resolve();payload=p.read_bytes();rows=json.loads(payload);out=Path(output)
 if out.exists():raise ValueError('Refuse existing destination')
 pairs=[];provenance=[];hashes={str(p):hashlib.sha256(payload).hexdigest()};seen=set()
 for r in rows:
  hashes.update(validate(r,reserved_ids()))
  if r['id'] in seen:raise ValueError('Duplicate identity')
  seen.add(r['id'])
  if not r['review'].get('negative_rubric_review'):raise ValueError('Negative/rubric review required')
  n=r['verifier_negative'];rubric=r['rubric']
  if n['origin']!='curator_authored_counterexample_not_observed_model_output' or n['training_role']!='verifier_or_preference_only_never_positive_SFT':raise ValueError('Unsupported negative provenance')
  if not isinstance(n['text'],str) or not n['text'].strip() or n['text']==r['completion']:raise ValueError('Invalid rejected answer')
  if any(x in n['text'] for x in ('<｜','</｜','｜DSML｜')):raise ValueError('Control marker in negative')
  if not rubric.get('required') or not rubric.get('reject_if'):raise ValueError('Rubric required')
  pairs.append({'prompt':[{'role':'user','content':r['prompt']}],'chosen':[{'role':'assistant','content':r['completion']}],'rejected':[{'role':'assistant','content':n['text']}]})
  provenance.append({'row':len(pairs)-1,'id':r['id'],'repo':r['repo'],'commit':r['commit'],'change_lineage':r.get('change_lineage',r['lineage']),'split_family':r.get('split_family'),'rubric':rubric,'negative_origin':n['origin'],'review':r['review']})
 for path,digest in hashes.items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input drift')
 out.mkdir(parents=True)
 for name,values in [('preferences.jsonl',pairs),('provenance.jsonl',provenance)]:
  (out/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
 report={'pairs':len(pairs),'distinct_changes':len({(r['repo'],r['commit']) for r in rows}),'inputs_sha256':hashes,'scope':'Optional curator-labeled diagnostic preference pairs. Rejected text is a synthetic counterexample, not an observed model output. Rubrics are semantic review guides, not validated automatic graders. Current Apple trainer has no qualified preference-training path. Never concatenate rejected answers into positive SFT.','pair_file_sha256':hashlib.sha256((out/'preferences.jsonl').read_bytes()).hexdigest()}
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--records',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(export(a.records,a.output),indent=2))
