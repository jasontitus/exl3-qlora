"""Surface concrete source-grounded task conditions; preserve every assistant byte."""
from pathlib import Path
import collections
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / 'loss-fix/apple/src'))
from exl3_mlx.data import load_tokenizer, tokenize_row

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

index_path = ROOT / 'case-index.json'
index = json.loads(index_path.read_text())
original_index_sha = sha(index_path)
original_index_path = ROOT / 'case-index-before-prompt-v2.json'
if original_index_path.exists():
    raise ValueError('Refuse to repeat prompt amendment')
original_index_path.write_bytes(index_path.read_bytes())
manifest = json.loads((ROOT / 'manifest-reviewed.json').read_text())
candidates = {c['id']: c for c in manifest['candidates']}
overrides = {
    'dce06d208cbc': 'The historical reproducer is an evaluated BF16 [1,H,T,264] K/V array sliced on its last axis to [3,259): the stride remains 264 but the base is offset by six bytes. It passes the old stride-only checks and produces incorrect values in kernels that stage through uint4. Preserve the aligned [0,256) route. Unevaluated lazy-array addresses remain outside this captured repair; do not imply that broader case is solved.',
    'fb2abb29e4ef': 'Metal API validation aborts when full-attention or indexed-decode scratch sizes are not multiples of 16 bytes. Support arbitrary context-length residues, both cache formats and invalid selected rows without changing attention indexing, precision or arithmetic.',
    '8fcd61d6bf0d': 'The baseline scratch allocation covers only one stage although the dense projection kernel double-buffers its scratch. Ensure both stages fit while preserving output dimensions and element precision. The captured source supplies no quantitative performance claim.',
    '0658dac84541': 'Review the baseline long-context path: inverse-RoPE readiness is tracked across layers, streaming top-k affects deterministic route selection, and RoPE cross-lane writes feed FP8 stores. Repair the corresponding state lifetime, opt-in behavior and memory-ordering contracts without claiming a measured speedup. These concrete concerns are inferred from the captured code changes; the short commit message reports only a correctness repair.',
    'ab1256ccd614': 'Only two GDN shapes are supported, but runtime parameter structs still carry fixed packed widths and derivable strides, while several guards describe work that the dispatch grid cannot launch. Keep the supported width set, host/kernel ABI and floating-point operation order synchronized; no elapsed-time gain is supplied.',
    '63ceff800643': 'The decode path writes recurrent rows and query intermediates that serving never reads; the historical 27B accounting reports about 4.7 MB and 1.5 MB per lane per step respectively. Keep replay state addressing and live outputs correct. A revised oracle may differ at BF16 rounding boundaries, so do not turn a tolerance comparison into a bitwise claim.',
    '7d60b965630a': 'Each decode GDN threadgroup pays a device fence and atomic completion-counter update, but the sole host reader checks the counters only after successful command completion. Determine which checks are redundant without weakening intra-kernel synchronization, live output/state checks or idle-lane memory checks. No timing result is supplied.',
    '7161da96eac6': 'The baseline top-k count search also computes exponentials and fifteen masses per token on every pass although it uses only a final bracket mass. A following top-p request performs another vocabulary gather over already ranked candidates. Reduce redundant work while respecting ties, wide top-k fallbacks and floating-point nucleus-boundary sensitivity. Historical shared-GPU timings are provisional, not a model-generation result.',
}
tokenizer = load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
amendments = []
for entry in index['cases']:
    if entry.get('curriculum_tier') == 'unmeasured':
        entry['curriculum_tier'] = 'source_grounded_unmeasured'
    if not entry.get('exact_edit_eligible'):
        continue
    candidate = candidates[entry['id']]
    condition = overrides.get(entry['commit'][:12], candidate['annotations']['constraints']['text'])
    condition = re.sub(r'(?<=[a-z])(?=\d)', ' ', condition)
    condition = re.sub(r'(?<=\d)(?=[a-z])', ' ', condition)
    old_raw = Path(entry['paths']['raw'])
    old_prepared = Path(entry['paths']['prepared'])
    raw = json.loads(old_raw.read_text())
    assistant = [m for m in raw['messages'] if m['role'] == 'assistant']
    user = next(m for m in raw['messages'] if m['role'] == 'user')
    first, rest = user['content'].split('\n\n', 1)
    # Remove the old repeated constraints line; keep every source excerpt intact.
    rest = '\n'.join(line for line in rest.split('\n') if not line.startswith('constraints: '))
    user['content'] = first + '\n\nHistorical problem and constraints (source-derived task specification): ' + condition + '\n\n' + rest
    assert [m for m in raw['messages'] if m['role'] == 'assistant'] == assistant
    target_dir = ROOT / 'reviewed-v2' / entry['id']
    target_dir.mkdir(parents=True, exist_ok=False)
    prepared = tokenize_row(raw, tokenizer)
    save(target_dir / 'raw.json', raw)
    save(target_dir / 'prepared.json', prepared)
    message_path = ROOT / 'sourcepacks' / entry['id'] / 'commit-message.txt'
    diff_path = message_path.with_name('commit.diff')
    amendments.append({'id': entry['id'], 'condition': condition,
        'basis': 'source_code_inference_and_reported_correctness' if entry['commit'].startswith('0658') else 'source_grounded_historical_task_specification',
        'source_evidence_sha256': {str(message_path): sha(message_path), str(diff_path): sha(diff_path)},
        'old_input_sha256': {str(old_raw): sha(old_raw), str(old_prepared): sha(old_prepared)},
        'assistant_and_tool_call_unchanged': True,
        'source_and_replay_unchanged': True})
    for kind in ['raw', 'prepared']:
        entry['paths'][kind] = str(target_dir / (kind + '.json'))
        entry['files_sha256'][kind] = sha(target_dir / (kind + '.json'))
    n = len(prepared['input_ids'])
    entry.update(total_serialized_ids=n, supervised_tokens=n-prepared['loss_start'], fits_12k=n<=12288, fits_32k=n<=32768)
    assert entry['fits_32k'], entry['id']
    for kind in ['source', 'replay']:
        assert sha(Path(entry['paths'][kind])) == entry['files_sha256'][kind]

scopes = {}
for cap in [12288, 32768]:
    rows = [r for r in index['cases'] if r.get('exact_edit_eligible') and r['total_serialized_ids'] <= cap]
    scope = {'count': len(rows), 'repositories': dict(collections.Counter(r['repository'] for r in rows)),
        'curriculum_tiers': dict(collections.Counter(r['curriculum_tier'] for r in rows)),
        'serialized_tokens': sum(r['total_serialized_ids'] for r in rows),
        'supervised_tokens': sum(r['supervised_tokens'] for r in rows), 'case_ids': [r['id'] for r in rows]}
    for name, selected in [('exact', rows), ('recommended-core', [r for r in rows if r['recommended_core']]),
                           ('unmeasured-opt-in', [r for r in rows if not r['recommended_core']])]:
        for kind in ['raw', 'prepared']:
            (ROOT / f'{name}-{cap}.{kind}.jsonl').write_text(''.join(json.dumps(json.loads(Path(r['paths'][kind]).read_text()), ensure_ascii=False)+'\n' for r in selected))
        if name != 'exact':
            scope[name] = {'count': len(selected), 'case_ids': [r['id'] for r in selected],
                'serialized_tokens': sum(r['total_serialized_ids'] for r in selected),
                'supervised_tokens': sum(r['supervised_tokens'] for r in selected)}
    scopes[str(cap)] = scope
index['scopes'] = scopes
receipt = ROOT / 'task-spec-amendment-v2.json'
save(receipt, {'format': 'concrete-historical-task-spec-amendment-v2', 'cases': amendments,
    'original_index_sha256': original_index_sha, 'script_sha256': sha(Path(__file__)),
    'scope': 'Source-grounded retrospective user task clarification only. No assistant/tool-call/source/replay changes; no new model or GPU execution.'})
index['task_spec_amendment'] = {'path': str(receipt), 'sha256': sha(receipt)}
save(index_path, index)
assert 'mlx.core' not in sys.modules
print(json.dumps(scopes, indent=2))
