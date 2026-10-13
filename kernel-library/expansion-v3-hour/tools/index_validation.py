"""Index per-case validation evidence without promoting compiler passes to runtime proof."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = ['corpus-32768-v1/provenance.jsonl', 'candidate-inventory.json',
               'compiler-probes/compiler-receipt.json',
               'compiler-probes/ds4-compiler-receipt.json',
               'compiler-probes/ds4-complete-compiler-receipt.json']
    rows = [json.loads(line) for line in (ROOT/sources[0]).read_text().splitlines()]
    rows = [row for row in rows if row['split'] == 'train']
    compilations = {}
    for name in sources[2:]:
        receipt = json.loads((ROOT/name).read_text())
        for index, result in enumerate(receipt['results']):
            compilations.setdefault(result['case_id'], []).append({
                'receipt':name, 'result_index':index, 'side':result['side'],
                'file':result.get('file', 'runtime concatenation'),
                'configuration':result.get('configuration', 'default'),
                'exit_code':result['exit_code']})
    result_rows = []
    for row in rows:
        evidence = compilations.get(row['id'], [])
        result_rows.append({
            'id':row['id'], 'repo':row['repo'], 'commit':row['commit'],
            'tier':row['curriculum_tier'], 'serialized_tokens':row['serialized_tokens'],
            'source_provenance':row['provenance_file'],
            'preparation':'passed scoped source/replay/tokenizer review',
            'compiler_invocations':evidence,
            'compiler_scope':'subset source/configurations only' if evidence else 'not attempted in this hour',
            'new_runtime_numerical_check':False, 'new_runtime_performance_check':False,
            'next_validation':'Reconstruct an isolated pinned build, then run shape/dtype boundary parity and matched repeated timings. Prioritize cases relevant to evaluation workloads; do not execute fetched scripts automatically.'})
    candidates = json.loads((ROOT/'candidate-inventory.json').read_text())['records']
    excluded = [{k:row.get(k) for k in ('id','repo','commit','archival_status','reason')}
                for row in candidates if not row['exact_membership']]
    result = {
        'schema':'metal-validation-index-v1', 'scope':'Core32K training set;12K is a subset. Local checks performed this hour only. Prior evidence remains in per-case provenance.',
        'core_changes':len(result_rows),
        'core_changes_with_any_compiler_attempt':sum(bool(r['compiler_invocations']) for r in result_rows),
        'core_changes_with_any_compiler_pass':sum(any(e['exit_code']==0 for e in r['compiler_invocations']) for r in result_rows),
        'warning':'An invocation can cover only one file/configuration. Before/after compilation is not complete engine build, runtime numerical parity, measured speed, or training effectiveness.',
        'records':result_rows, 'candidates_without_released_exact_membership':excluded,
        'inputs_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}
    (ROOT/'validation-index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','candidates_without_released_exact_membership','inputs_sha256')},indent=2))


if __name__ == '__main__':
    main()
