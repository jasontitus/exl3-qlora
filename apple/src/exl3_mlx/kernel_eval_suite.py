"""CPU-only comparison of pinned generated-code evaluation receipts.

This tool does not certify backend assertions or run generated text. Execution
is delegated separately to kernel_execution's explicitly trusted backends.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from .evaluation_receipts import _read

STAGES = ('generation', 'raw_format', 'replay', 'source', 'compile', 'numerical', 'timing')
STATUSES = ('passed', 'failed', 'pending', 'refused', 'not_run')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def integer(value, minimum=1):
    require(type(value) is int and value >= minimum, 'Invalid integer')
    return value


def budget(value):
    require(isinstance(value, dict) and set(value) == {'max_output_tokens', 'attempts'}, 'Invalid budget')
    for item in value.values():
        integer(item)
    require(value['attempts'] <= 100 and value['max_output_tokens'] <= 1048576, 'Budget exceeds supported bound')
    return value


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


class Reader:
    def __init__(self, base):
        self.base = Path(base)
        self.files = {}
        self.total = 0

    def read(self, ref):
        require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'Invalid file reference')
        sha(ref['sha256'])
        path = Path(ref['path'])
        path = path if path.is_absolute() else self.base / path
        # _read rejects aliases, symlinks, special files and descriptor drift.
        raw, identity = _read(path.absolute(), 8 * 1024 * 1024)
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Evidence hash mismatch')
        if path in self.files:
            require(self.files[path] == (raw, identity), 'Repeated evidence changed')
        else:
            self.total += len(raw)
            require(len(self.files) < 4096 and self.total <= 128 * 1024 * 1024, 'Evidence budget exceeded')
            self.files[path] = (raw, identity)
        return raw

    def json(self, ref):
        return json.loads(self.read(ref), object_pairs_hook=pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

    def recheck(self):
        for path, expected in self.files.items():
            require(_read(path.absolute(), 8 * 1024 * 1024) == expected, 'Evidence changed before report')


def identities(rows, fields):
    require(isinstance(rows, list) and 0 < len(rows) <= 1024, 'Invalid identity list')
    out = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'id', *fields}, 'Invalid identity')
        name = row['id']
        require(isinstance(name, str) and re.fullmatch('[A-Za-z0-9_.-]{1,128}', name), 'Invalid ID')
        require(name not in out, 'Duplicate ID')
        for field in fields:
            sha(row[field])
        out[name] = row
    return out


def validate_result(row, tasks, adapters, expected_budget, reader):
    require(isinstance(row, dict) and row.get('format') == 'kernel-eval-result-v1', 'Invalid result')
    task, adapter = tasks.get(row.get('task_id')), adapters.get(row.get('adapter_id'))
    require(task is not None and adapter is not None, 'Unknown task or adapter')
    for field, expected in (('task_sha256', task['sha256']), ('adapter_sha256', adapter['sha256']),
                            ('prompt_sha256', task['prompt_sha256']), ('protocol_sha256', task['protocol_sha256'])):
        require(row.get(field) == expected, 'Result identity mismatch: ' + field)
    require(budget(row.get('budget')) == expected_budget, 'Result budget mismatch')
    require(integer(row.get('attempt')) <= expected_budget['attempts'], 'Unregistered attempt')
    tracks = row.get('tracks')
    require(isinstance(tracks, dict) and set(tracks) <= {'strict', 'normalized'} and tracks, 'Invalid tracks')
    for track, stages in tracks.items():
        require(isinstance(stages, dict) and set(stages) <= set(STAGES), 'Invalid stages')
        for stage, result in stages.items():
            require(isinstance(result, dict) and result.get('status') in STATUSES, 'Invalid stage status')
            require(result.get('outcome_class', 'unspecified') in ('model_output', 'infrastructure', 'unspecified'), 'Invalid outcome class')
            evidence = result.get('evidence', [])
            require(isinstance(evidence, list) and len(evidence) <= 32, 'Invalid evidence')
            if result['status'] in ('passed', 'failed', 'refused'):
                require(evidence, 'Terminal stage requires evidence')
            for ref in evidence:
                reader.read(ref)
            require(track != 'normalized' or stage not in ('generation', 'raw_format'), 'Normalization cannot claim raw success')
            if result['status'] == 'passed' and stage in ('raw_format', 'replay', 'source', 'compile', 'numerical', 'timing'):
                prior = {'raw_format': 'generation', 'replay': 'raw_format', 'source': 'replay', 'compile': 'source', 'numerical': 'compile', 'timing': 'numerical'}[stage]
                if track == 'normalized' and stage == 'replay':
                    require(tracks.get('strict', {}).get('generation', {}).get('status') == 'passed', 'Normalization needs completed generation')
                    continue
                require(stages.get(prior, {}).get('status') == 'passed', 'Missing passed prerequisite: ' + prior)
            if stage == 'timing' and result['status'] == 'passed':
                m = result.get('measurements')
                require(isinstance(m, dict) and set(m) == {'workload_sha256', 'device', 'cache_state', 'unit', 'samples'}, 'Invalid timing')
                sha(m['workload_sha256'])
                require(all(isinstance(m[k], str) and m[k] for k in ('device', 'cache_state')), 'Missing timing context')
                require(m['unit'] == 'ms' and isinstance(m['samples'], list) and m['samples'], 'Invalid timing samples')
                require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in m['samples']), 'Invalid timing value')
    return row


def build_report(manifest, reader):
    require(manifest.get('format') == 'kernel-eval-suite-v1', 'Invalid suite format')
    tasks = identities(manifest.get('tasks'), ('sha256', 'prompt_sha256', 'protocol_sha256'))
    adapters = identities(manifest.get('adapters'), ('sha256',))
    b = budget(manifest.get('budget'))
    require(len(tasks) * len(adapters) * b['attempts'] <= 4096, 'Suite slot budget exceeded')
    refs = manifest.get('results', [])
    require(isinstance(refs, list) and len(refs) <= 4096, 'Invalid results')
    rows, seen = [], set()
    for ref in refs:
        row = validate_result(reader.json(ref), tasks, adapters, b, reader)
        key = (row['task_id'], row['adapter_id'], row['attempt'])
        require(key not in seen, 'Duplicate attempt')
        seen.add(key)
        rows.append(row)
    indexed = {(r['task_id'], r['adapter_id'], r['attempt']): r for r in rows}
    summaries = []
    for adapter in adapters:
        counts = {track: {stage: Counter() for stage in STAGES} for track in ('strict', 'normalized')}
        for task in tasks:
            for attempt in range(1, b['attempts'] + 1):
                row = indexed.get((task, adapter, attempt), {})
                for track in counts:
                    for stage in STAGES:
                        counts[track][stage][row.get('tracks', {}).get(track, {}).get(stage, {}).get('status', 'pending')] += 1
        summaries.append({'adapter_id': adapter, 'registered_slots': len(tasks) * b['attempts'], 'stages': counts})
    comparisons = []
    for left, right in itertools.combinations(rows, 2):
        if left['task_id'] != right['task_id'] or left['adapter_id'] == right['adapter_id'] or left['attempt'] != right['attempt']:
            continue
        for track in ('strict', 'normalized'):
            a = left['tracks'].get(track, {}).get('timing', {})
            c = right['tracks'].get(track, {}).get('timing', {})
            if a.get('status') != 'passed' or c.get('status') != 'passed':
                continue
            x, y = a['measurements'], c['measurements']
            if any(x[k] != y[k] for k in ('workload_sha256', 'device', 'cache_state', 'unit')):
                continue
            comparisons.append({'task_id': left['task_id'], 'attempt': left['attempt'], 'track': track,
                                'left': left['adapter_id'], 'right': right['adapter_id'],
                                'timing_qualified': False, 'comparison_kind': 'unpaired_descriptive_median_ratio',
                                'left_over_right_median_latency': statistics.median(x['samples']) / statistics.median(y['samples'])})
    # A common slot is resolved for every adapter: a terminal rejection, or
    # passed numerical validation. Unrun infrastructure cannot improve rank.
    def resolved(stages):
        if stages.get('numerical', {}).get('status') == 'passed':
            return True
        generation = stages.get('generation', {})
        if generation.get('status') == 'failed' and generation.get('outcome_class') == 'model_output':
            return True  # e.g. completed capped generation without EOS
        if generation.get('status') != 'passed':
            return False
        return any(stages.get(name, {}).get('status') in ('failed', 'refused') and
                   stages[name].get('outcome_class') == 'model_output'
                   for name in ('raw_format', 'replay', 'source', 'compile', 'numerical'))

    common = []
    for task in tasks:
        for attempt in range(1, b['attempts'] + 1):
            stages = [indexed.get((task, a, attempt), {}).get('tracks', {}).get('strict', {}) for a in adapters]
            if all(resolved(x) for x in stages):
                common.append((task, attempt))
    ranking = []
    for adapter in adapters:
        selected = [indexed[(task, adapter, attempt)]['tracks'].get('strict', {}) for task, attempt in common]
        ranking.append({'adapter_id': adapter, 'common_resolved_slots': len(common),
                        'strict_usable': sum(x.get('numerical', {}).get('status') == 'passed' for x in selected),
                        'measured_failures': sum(any(v.get('status') == 'failed' and v.get('outcome_class') == 'model_output' for v in x.values()) for x in selected),
                        'refused_slots': sum(any(v.get('status') == 'refused' and v.get('outcome_class') == 'model_output' for v in x.values()) for x in selected)})
    ranking.sort(key=lambda x: (-x['strict_usable'], x['measured_failures'], x['adapter_id']))
    best = ranking[0]['strict_usable']
    leaders = [r['adapter_id'] for r in ranking if r['strict_usable'] == best] if best else []
    recommendation = {'status': 'insufficient_evidence' if not best else ('tie' if len(leaders) > 1 else 'leading_on_common_resolved_slots'),
                      'adapters': leaders, 'common_resolved_slots': len(common),
                      'registered_slots_per_adapter': len(tasks) * b['attempts'],
                      'coverage_complete': len(common) == len(tasks) * b['attempts'],
                      'basis': 'Strict usable numerical results on the same resolved task/attempt slots; normalized diagnostics excluded.'}
    reader.recheck()
    return {'format': 'kernel-eval-comparison-v1', 'suite_id': manifest.get('id'), 'summaries': summaries,
            'tasks': list(tasks.values()), 'adapters': list(adapters.values()), 'budget': b,
            'latency_comparisons': comparisons, 'results': rows, 'winner': None,
            'ranking': ranking, 'recommendation': recommendation,
            'inputs_sha256': {str(path): hashlib.sha256(value[0]).hexdigest() for path, value in reader.files.items()},
            'limitations': ['Receipt assertions are not independently certified by this report.',
                            'Pending, refused and not_run are distinct from measured pass or failure.',
                            'Normalized diagnostics do not count as raw model success.',
                            'Recommendation is restricted to common resolved slots, not a global quality claim.']}


def render_markdown(report):
    lines = ['# Generated-code comparison', '', 'Recommendation: ' + report['recommendation']['status'], '',
             '| Adapter | Common slots | Strict usable | Measured failures | Refused |',
             '|---|---:|---:|---:|---:|']
    for row in report['ranking']:
        lines.append('| {adapter_id} | {common_resolved_slots} | {strict_usable} | {measured_failures} | {refused_slots} |'.format(**row))
    lines += ['', report['recommendation']['basis'], '',
              'Coverage complete: ' + str(report['recommendation']['coverage_complete']), '',
              'Normalized diagnostics and pending stages remain separate in the JSON report.']
    lines += ['', '| Task | Adapter | Attempt | Raw format | Exact replay | Strict numerical | Normalized source | Normalized numerical |',
              '|---|---|---:|---|---|---|---|---|']
    indexed = {(r['task_id'], r['adapter_id'], r['attempt']): r for r in report['results']}
    for task in report['tasks']:
        for adapter in report['adapters']:
            for attempt in range(1, report['budget']['attempts'] + 1):
                row = indexed.get((task['id'], adapter['id'], attempt), {})
                def status(track, stage):
                    return row.get('tracks', {}).get(track, {}).get(stage, {}).get('status', 'pending')
                values = [task['id'], adapter['id'], str(attempt), status('strict', 'raw_format'),
                          status('strict', 'replay'), status('strict', 'numerical'),
                          status('normalized', 'source'), status('normalized', 'numerical')]
                lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args(argv)
    path = args.suite.absolute()
    raw, identity = _read(path, 8 * 1024 * 1024)
    reader = Reader(path.parent)
    reader.files[path] = (raw, identity)
    manifest = json.loads(raw, object_pairs_hook=pairs)
    result = build_report(manifest, reader)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if args.markdown:
        with args.markdown.open('x') as stream:
            stream.write(render_markdown(result))
    print(json.dumps({'report': str(args.output), 'adapters': len(result['summaries']), 'winner': None}))


if __name__ == '__main__':
    main()
