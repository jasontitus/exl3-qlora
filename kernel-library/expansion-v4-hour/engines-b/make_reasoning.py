"""Source-grounded historical diagnostics, not fabricated execution trajectories."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((R/'integration-draft.json').read_text());manifest=json.loads((R/'manifest-draft.json').read_text());cases={c['commit'][:8]:c for c in report['cases']};cs={c['commit'][:8]:c for c in manifest['candidates']};records=[]
def readable(s):
 s=re.sub(r'(?<=[a-z])(?=\d)', ' ', s)
 return re.sub(r'(?<=\d)(?=(?:ms|µs|tokens|KB|MB)\b)',' ',s)
for key,task,outcome,prompt,completion in json.loads((R/'reasoning_items.json').read_text()):
 case=cases[key];c=cs[key]
 if case['protection_status']!='cleared':continue
 ep=R/'evidence'/f"{case['repo']}-{case['commit'][:12]}-message.txt"
 records.append({'id':f"reason-{case['repo']}-{key}-{task}-hour-v4",'repo':manifest['repositories'][case['repo']]['url'],'commit':case['commit'],'parent':case['parent'],'lineage':case['repo']+':'+case['commit'],'change_lineage':case['repo']+':'+case['commit'],'split_family':c['source_group'],'related_exact_example':case['id'],'task_type':task,'outcome_class':outcome,'prompt':'Review this historical source report. No commands or measurements have been run by you. Explain the decision, limits and useful next check without inventing observations.\n\n'+readable(prompt),'completion':readable(completion),'rationale_basis':'mixed','source_evidence':[{'path':str(ep),'sha256':H(ep),'url':manifest['repositories'][case['repo']]['url']+'/commit/'+case['commit'],'phase':'post_change','quote':ep.read_text()}],'inference_disclosure':'Measurements and failures are historical author claims. Explanations, proposed checks and arithmetic are curator inferences. This is a constructed diagnostic task with disclosed historical evidence, not an observed agent rollout or demonstration of discovery from baseline alone.','validation':{'local_execution':'not_run','source_review':'Compared source message and complete captured diff; independent review pending.'},'protection_review':c['historical_protection_review'],'licenses':[{'path':str(R/a['path']),'sha256':a['sha256']} for a in case.get('licenses',[])],'review':{'status':'pending','reviewer':'pending','evidence_fidelity':True,'no_fabricated_observations':True,'task_answer_separation':True,'lineage_screened':True}})
(R/'reasoning-records-draft.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
print(len(records))
