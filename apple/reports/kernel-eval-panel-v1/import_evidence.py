"""Import the fixed affine observations; never run inference/compiler/device code.

Run from the product checkout with PYTHONPATH=apple/src. --workspace points to
archived experiment evidence, --replays is the three-entry common-parser map.
The destination must be fresh. Archived receipt internal absolute locators are
retained verbatim; all references consumed by the portable suite are local.
"""
import argparse
import hashlib
import json
from pathlib import Path
from exl3_mlx.evaluation_receipts import _read
from exl3_mlx.kernel_eval_suite import Reader, build_report, render_markdown


def digest(raw): return hashlib.sha256(raw).hexdigest()
def encode(obj): return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def shared_identity(request):
    pins = request['inputs_sha256']
    tokenizer = {p: h for p, h in pins.items()
                 if Path(p).name in ('tokenizer.json', 'tokenizer_config.json')}
    runtime = {p: h for p, h in pins.items() if p.endswith(('.dylib', '.metallib'))}
    if len(tokenizer) != 2 or not runtime:
        raise ValueError('Missing shared tokenizer/runtime identity')
    if len({Path(p).name for p in runtime}) != len(runtime):
        raise ValueError('Ambiguous runtime basename')
    return {'payload_manifest_sha256': pins[request['payload_manifest']],
            'tokenizer': tokenizer, 'runtime': runtime}


def check_shared_identities(requests):
    expected = shared_identity(requests['old458'])
    if any(shared_identity(r) != expected for r in requests.values()):
        raise ValueError('Model manifest/tokenizer/runtime identity differs')
    return expected


def run(workspace, replays, output):
    n = Path(workspace).resolve()/'kernel-library/next-round-v5'
    l = n/'exact-generation-qualification-v1'
    s = n/'studio-functional-panel-v1'; t = s/'affine_relu_sum'
    old = l/'studio-affine458-generation-box96-v1'
    final = l/'studio-affine-lr25-step188-profile50-box104-v1'
    out = Path(output).absolute(); out.mkdir()
    (out/'evidence').mkdir(); snapshots = {}
    def copy(path, expected=None):
        path = Path(path).absolute(); raw, identity = _read(path, 8<<20)
        h = digest(raw)
        if expected is not None and h != expected: raise ValueError('Source SHA mismatch: '+str(path))
        if path in snapshots and snapshots[path] != (raw,identity): raise ValueError('Repeated source drift')
        snapshots[path] = raw,identity
        name = 'evidence/'+h[:16]+'-'+path.name
        dest=out/name
        if dest.exists():
            if dest.read_bytes()!=raw: raise ValueError('Destination collision')
        else: dest.write_bytes(raw)
        return {'path':name,'sha256':h}
    def save(name,obj):
        raw=encode(obj);(out/name).write_bytes(raw);return {'path':name,'sha256':digest(raw)}
    replay_map_ref=copy(replays)
    replay_map=json.loads(snapshots[Path(replays).absolute()][0])
    assert set(replay_map)=={'basezeroB','old458','final188'}
    captures={
      'basezeroB':(old/'requests/0729-zeroB458.json',old/'independent-actual-q-v1/review-v1.json',old/'independent-actual-q-v1/zeroB458/generation.raw'),
      'old458':(old/'requests/0729-checkpoint458.json',old/'independent-actual-q-v1/review-v1.json',old/'independent-actual-q-v1/checkpoint458/generation.raw'),
      'final188':(final/'request.json',final/'actual-capture-audit-v1/review-v1.json',final/'actual-capture-audit-v1/generation.raw')}
    requests={}; provenance={}; adapters=[]
    for name,(request,review,rawpath) in captures.items():
        request_ref=copy(request);review_ref=copy(review); raw_ref=copy(rawpath)
        q=json.loads(snapshots[request.absolute()][0]);requests[name]=q
        review_obj=json.loads(snapshots[review.absolute()][0])
        if name=='final188':
            assert review_obj['request_sha256']==request_ref['sha256'] and review_obj['raw_sha256']==raw_ref['sha256']
        else:
            cap=old/'independent-actual-q-v1'/('zeroB458' if name=='basezeroB' else 'checkpoint458')/'capture.json'
            capref=copy(cap);c=json.loads(snapshots[cap.absolute()][0])
            assert c['request_sha256']==request_ref['sha256'] and c['generation_sha256']==raw_ref['sha256']
            assert c['actual_execution_review']['sha256']==review_ref['sha256']
        hashes={Path(p).name:h for p,h in q['inputs_sha256'].items() if p.startswith(q['adapter_dir']+'/')}
        assert 'adapter.safetensors' in hashes and 'adapter_config.json' in hashes
        adapterref=save(name+'-identity.json',{'adapter_files_sha256':hashes,'generation_request':request_ref})
        adapters.append({'id':name,'sha256':adapterref['sha256']})
        provenance[name]={'request':request_ref,'capture_review':review_ref,'raw':raw_ref,'adapter':adapterref,
                          'resource_profile':{k:q['execution'][k] for k in ('process_memory_gib','native_box_ceiling_gib','reserve_gib','physical_memory_gib')}}
    q=requests['old458']
    shared_keys=('policy','serving_profile','arithmetic_profile','tf32','eos_token_ids','max_new_tokens','max_context_tokens')
    assert all(all(r[k]==q[k] for k in shared_keys) for r in requests.values())
    shared=check_shared_identities(requests)
    common={k:q[k] for k in shared_keys}
    common['engine_sha256']=q['inputs_sha256'][q['engine']]
    assert all(r['inputs_sha256'][r['engine']]==common['engine_sha256'] for r in requests.values())
    common['model_manifest']=copy(q['payload_manifest'],q['inputs_sha256'][q['payload_manifest']])
    common['tokenizer_sha256']={Path(p).name:h for p,h in shared['tokenizer'].items()}
    common['runtime_files_sha256']={Path(p).name:h for p,h in shared['runtime'].items()}
    common['observed_profiles']={a:p['resource_profile'] for a,p in provenance.items()}
    common['limits']=['Retrospective evidence reuse, not a newly preregistered matched campaign.',
        'Resource profiles differ for final188; adapter-only causal attribution is not established.',
        'RMS has no generation results. No performance ranking or successful candidate is inferred.',
        'Historical receipt internal locators remain archival; this bundle is not their entire provenance closure.']
    tasks=[]; protocols={}
    for name,root in [('affine',t),('rms',s/'row_rms-v2')]:
        baseline={x:copy(root/'task'/x) for x in ('kernel.metal','dispatch.json')}
        task=save(name+'-task.json',{'design':copy(root/'design-v2.json'),'baseline':baseline,'prompt':copy(root/'prompt.txt'),'serialized_prompt':copy(root/'prompt-prepared.json')})
        proto=save(name+'-protocol.json',dict(common,task=task,parser_protocol='safe-common-git-apply-exact-v1',
                    baseline=baseline,attempts_per_adapter=1,raw_replay='No offsets, recounting or code repair',
                    grading='Task design fixed numerical oracle/tolerance; source safety before compile; timing only after correctness'))
        protocols[name]=proto
        tasks.append(dict(id=name,sha256=task['sha256'],prompt_sha256=copy(root/'prompt-prepared.json')['sha256'],protocol_sha256=proto['sha256']))
    results=[]
    for adapter in adapters:
        name=adapter['id'];rp=replay_map[name];rr=copy(rp['path'],rp['sha256']); replay=json.loads(snapshots[Path(rp['path']).absolute()][0])
        assert replay['protocol']=='safe-common-git-apply-exact-v1' and replay['generation_sha256']==provenance[name]['raw']['sha256']
        stages={'generation':{'status':'passed','evidence':[provenance[name]['capture_review'],provenance[name]['raw'],provenance[name]['request'],protocols['affine']]}}
        for stage,value in replay['stages'].items():
            stages[stage]=dict(value,evidence=[rr])
        if replay['stages'].get('replay', {}).get('status') != 'passed':
            for stage in ('source','compile','numerical','timing'):
                stages[stage]={'status':'not_run','reason':'Exact replay did not pass'}
        tracks={'strict':stages}
        if name in ('old458','basezeroB'):
            condition='checkpoint458' if name=='old458' else 'zeroB458'
            norm=copy(t/'normalized-actual-v1'/condition/'receipt.json')
            normalized={'replay':{'status':'passed','evidence':[norm]}}
            if name=='old458':
                sr=copy(t/'normalized-actual-v1/checkpoint458/source-review-v2.json')
                cr=copy(t/'candidate-build-diagnostic-v1/checkpoint458-build-v1/build/build-receipt.json')
                nr=copy(t/'candidate-controller-diagnostic-v1/actual-audit-q-v1/review-v1.json')
                normalized.update(source={'status':'passed','evidence':[sr]},compile={'status':'passed','evidence':[cr]},numerical={'status':'failed','outcome_class':'model_output','evidence':[nr],'reason':'215/224 cases passed; 9 failed, 12 rows outside fixed tolerance'},timing={'status':'not_run','reason':'Numerically rejected'})
            else:
                ar=copy(t/'normalized-actual-v1/zeroB458/alignment-review-v1.json')
                normalized['source']={'status':'refused','outcome_class':'model_output','evidence':[ar],'reason':'Misaligned float4 access in 80 fixed cases; no numerical run'}
                for stage in ('compile','numerical','timing'):
                    normalized[stage]={'status':'not_run','reason':'Source safety refused'}
            tracks['normalized']=normalized
        task=tasks[0]
        row=dict(format='kernel-eval-result-v1',task_id='affine',adapter_id=name,task_sha256=task['sha256'],adapter_sha256=adapter['sha256'],prompt_sha256=task['prompt_sha256'],protocol_sha256=task['protocol_sha256'],budget={'attempts':1,'max_output_tokens':4096},attempt=1,tracks=tracks)
        results.append(save(name+'-affine-result.json',row))
    for path,expected in snapshots.items():
        if _read(path,8<<20)!=expected: raise ValueError('Late import source drift')
    save('import-provenance.json',{'source_files_sha256':{str(p):digest(v[0]) for p,v in snapshots.items()},'observations':provenance,'replay_map':replay_map_ref,'shared_identity_checked_across_all_requests':True,'limits':common['limits']})
    suite=dict(format='kernel-eval-suite-v1',id='actual-affine-rms-panel-v1',tasks=tasks,adapters=adapters,budget={'attempts':1,'max_output_tokens':4096},results=results)
    save('suite.json',suite)
    report=build_report(suite,Reader(out));save('comparison.json',report);(out/'comparison.md').write_text(render_markdown(report))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--replays',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(run(a.workspace,a.replays,a.output))
