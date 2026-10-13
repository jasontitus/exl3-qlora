from pathlib import Path
import json,hashlib,sys,shutil,re
R=Path(__file__).resolve().parent;W=R.parents[2];sys.path.insert(0,str(W/'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate
sha=lambda b:hashlib.sha256(b).hexdigest()
m=json.loads((R/'draft-manifest.json').read_text());original=json.loads((R/'selection.json').read_text())['candidates'];wanted={x['commit'] for x in original if x['repo']!='tensorfold'};m['candidates']=[c for c in m['candidates'] if c['commit'] in wanted];opts={x['case_id']:x['options'] for x in json.loads((R/'scanned-context-options.json').read_text())};attrs={x['case_id']:x for x in json.loads((R/'draft-attrition.json').read_text())}
# Deliberately selected complete operation-bearing definitions. All other code edits stay in target.
selection={
 '89283f810986':['qmatmulBits'], '340016fcb017':['gatherQmvDownReduce','gatherQmvDownReduceRows'],
 '59d77c06fb9c':['decode','MoeMlpWeights'], 'a3b04e3ef804':['gdnRunYState'],
 'bfe518f77e4d':['gatherQsa256','qsaWantsBlocks'], '499a08c6c80d':['qsaMaskFromBlocks'],
 '242c0a50f1ad':['moeRouterSource'], 'b95fbb4afe9e':['qsaSlotAttn'],
 '364b9fd0760b':['old:qsaMaskFromQk'], '55d834b0460a':['ROWS_SOURCE','rowsConfig'],
 'f11faec370ab':['moeSwigluClamped','laneClampedPair','apply'], '26ef54dc1070':['codebookHelpers'],
 '27e81cfc2069':['whole:src/kernels/attn_pd_nax_header.metal','getAttnPdKernel'],
 '8c16b2bb6a33':['innerGemmSortedTable','codebookKernel'],
 '3ba7272b3305':['whole:src/kernels/attn_pd_nax.metal'],
 '4bf284554f14':['pairGemv','moeSwigluFused','downGemvPreparedMid'],
 '9942e8ea6c3d':['PAIR_GEMV_SOURCE','downGemvPreparedMid'],
 'dd9c7cce4822':['PAIR_GEMV_SOURCE','codebookHelpers','pairGemv'], 'e69d8873c54d':['moePrefill'],
 '74f0fee245fe':['gguf_staged_steps'],
 '20593c49f48f':['gguf_staged_steps','staged_accumulate','GgufDecodeParams','GgufDecodeFusedParams'],
 '07142b12bc86':['whole:runtime/metal/kernels/shared/moe_gguf.metal','addGgufExperts'],
 '8b4e1a58f222':['old:Linear::addGgufStaged','old:LinearPlan::requireConfiguration','old:LinearPlan::gateScratchBytes'],
 '6c00e2d53519':['ds4_engram_read_batch'], 'a154af774b40':['qwen4_moe_down_mxfp4_pass']}
review=[]
for c in m['candidates']:
 cid=c['id'];record=json.loads((R/'sourcepacks'/cid/'complete-record.json').read_text());units=[];selected=[]
 old_units=c['reviewed_units']['units'];old_names=attrs[cid]['selected_context']
 for name in selection[c['commit'][:12]]:
  if name.startswith('whole:'):
   path=name[6:];pair=record['files'][path];u={'path':path,'kind':'whole_file'}
   for side in ['before','after']:
    b=(pair[side]or{'content':''})['content'].encode();u[side]={'start':0,'end':len(b),'sha256':sha(b)}
  elif name.startswith('old:'):
   n=name[4:];loc=[d for d in old_names if d['name']==n];assert len(loc)==1,(cid,name)
   hits=[u for u in old_units if u['path']==loc[0]['path'] and re.search(r'\b'+re.escape(n)+r'\s*\(', record['files'][u['path']]['before']['content'].encode()[u['before']['start']:u['before']['end']].decode()[:500])];assert len(hits)==1,(cid,name,hits);u=hits[0]
  else:
   hits=[o for o in opts[cid] if o.get('name')==name];assert len(hits)==1,(cid,name)
   o=hits[0];u={k:o[k] for k in ['path','kind','before','after']}
   if u['kind']=='complete_macro':u['kind']='complete_declaration'
  units.append(u);selected.append({'name':name,'path':u['path'],'before_sha256':u['before']['sha256'],'after_sha256':u['after']['sha256'],'before_bytes':u['before']['end']-u['before']['start']})
 units.sort(key=lambda u:(u['path'],u['before']['start'],u['after']['start']));omitted=[]
 for path,p in record['files'].items():
  b=(p['before']or{'content':''})['content'].encode();a=(p['after']or{'content':''})['content'].encode();bc=ac=0
  for u in [x for x in units if x['path']==path]:
   if b[bc:u['before']['start']]!=a[ac:u['after']['start']]:omitted.append(path)
   bc=u['before']['end'];ac=u['after']['end']
  if b[bc:]!=a[ac:]:omitted.append(path)
 c['reviewed_units']={'reviewed_by':'Codex quality_research complete source-unit review 2026-10-13','dependency_review':'Exact complete function/declaration/kernel-source-file boundaries reviewed against the full before/after source. Shader bodies may be include/JIT fragments with external wrapper, template, imports, generated substitutions, ABI and dispatch dependencies. Repository context remains necessary; no standalone compilation or dependency-closure guarantee. The answer preserves all non-documentation commit edits, including host integration and tests.','units':units}
 c['operation_scope']={'description':'Historical source-conditioned one-edit task showing complete operation-bearing baseline definitions, with every non-documentation commit edit retained in the answer. Omitted prompt context is available in hash-bound full-source provenance; not a captured agent trajectory.','omitted_path_reasons':{p:'Other repository definitions, test scaffolding, registration or documentation are not expanded in the operation prompt. Exact baseline anchors and every non-documentation code/build/test edit remain in the replayed answer.' for p in sorted(set(omitted))}}
 c['task_kind']['reviewed_by']='Codex quality_research source classification 2026-10-13'
 reconstruct_candidate(c,m['repositories'],{})
 review.append({'case_id':cid,'commit':c['commit'],'parent':c['parent'],'selected_units':selected,'task_kind':c['task_kind']['value'],'source_reported_only':True,'independently_compiled':False,'independently_benchmarked':False,'complete_target_replay_required':True,'licenses':attrs[cid]['licenses'],'stable_patch_id':attrs[cid]['stable_patch_id']})
# Bind captured protection metadata; old exact-source protected implementations are not copied.
e=R/'evidence';e.mkdir(exist_ok=True);shutil.copyfile(W/'kernel-library/integration-v1/group-a/evidence/protection.json',e/'protection.json');shutil.copyfile(W/'kernel-library/integration-v1/group-a/evidence/reserved-task-contracts.json',e/'reserved-task-contracts.json')
metadata=json.loads((e/'protection.json').read_text());protected=json.dumps(metadata,sort_keys=True)
# Identity hits require manual review; repository names or broad operation nouns alone are not an exclusion proof.
identity_hits=[]
for c in m['candidates']:
 hits=[v for v in [c['commit'],c['parent']] if v in protected or v[:12] in protected]
 if hits:identity_hits.append({'case_id':c['id'],'identity_hits':hits})
 reason=('Captured metadata identity review plus complete commit/source operation review: no exact protected commit/parent match. EXL3/Sushi/MLX-serving, Qwen sparse attention, GGUF projection and DS4 Engram/MXFP4 down-projection implementation tasks remain in related training lineages; they are not declared independent of prior Q8/HC/attention development tasks. Reserved Q8 half-rounded bitwise prefill, HC mixing and residual-fusion targets are not the changed operations here. Finite review, not proof of universal semantic/ancestry independence.')
 c['historical_protection_review']={'status':'pending' if hits else 'cleared','reviewed_by':'Codex quality_research metadata and operation reviewer 2026-10-13','reason':reason if not hits else 'Exact historical identity requires further review; retained as pending, not training-admitted.','evidence_sha256':{'evidence/protection.json':sha((e/'protection.json').read_bytes()),'evidence/reserved-task-contracts.json':sha((e/'reserved-task-contracts.json').read_bytes())}}
# A later public correction changes attribution, not the original source bytes or patch.
hist=W/'loss-fix/apple/artifacts/loss-fix/upstream-inventory/sushi-history.jsonl'
later=next(json.loads(l) for l in hist.open() if json.loads(l)['commit'].startswith('1dd27c838a07'))
(e/'sushi-pv-attribution-correction.txt').write_text(later['message']);m['evidence']['pv_attribution_correction']={'path':'evidence/sushi-pv-attribution-correction.txt','sha256':sha((e/'sushi-pv-attribution-correction.txt').read_bytes()),'url':'https://github.com/beamivalice/sushi/commit/'+later['commit'],'phase':'post_change'}
for c in m['candidates']:
 if c['commit'].startswith('3ba7272'):
  c['annotations']['constraints']={'text':'Later upstream attribution correction is authoritative for interpreting speed claims: the original combined gain must not be attributed to the f16 PV change alone. Arithmetic precision changes are intentional and require the stated tolerance/KLD contract. '+later['message'],'basis':'source','evidence':[{'source':'pv_attribution_correction','quote':later['message']}]}
 m['notes']=['Historical source-conditioned synthetic one-edit tasks, not real agent trajectories. Complete non-documentation code/build/test edits are preserved. Source performance/correctness claims were not rerun.','Source forks and related revisions form shared training lineages; no new independent-family or held-out performance credit. Neutral cases, regressions and later timing-attribution corrections remain in source-grounded targets/evidence.']
m['protected_keys']=['repository:https://github.com/ggml-org/llama.cpp'];m.pop('manifest_sha256',None)
# Save unsealed review candidate until final tokenize/replay verification.
(R/'manifest-review-candidate.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n');(R/'source-unit-review.json').write_text(json.dumps({'format':'expansion-source-unit-review-v1','cases':review,'historical_identity_hits':identity_hits},indent=2)+'\n')
print('reviewed',len(m['candidates']),'identity_hits',identity_hits)
