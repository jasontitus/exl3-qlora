"""Reconstruct selected historical source pairs, without executing source code."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / 'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate

H = lambda b: hashlib.sha256(b).hexdigest()
CANON = lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
selection = json.loads((ROOT / 'selection.json').read_text())
candidates, ledger = [], []
for pin in selection['pins']:
    sha, parent, key = pin['head'], pin['parent'], pin['repository_key']
    repo = selection['repositories'][key]['path']
    message = subprocess.check_output(['git', '--git-dir=' + repo, 'show', '-s', '--format=%B', sha], text=True).strip()
    title = message.splitlines()[0]
    case = 'tensorfold-' + sha[:12] + '-expansion-v2'
    annotation = lambda text, quote: {'text': text, 'basis': 'source', 'evidence': [{'source': 'commit_message', 'quote': quote}]}
    candidate = {'id': case, 'repository': key, 'commit': sha, 'parent': parent,
                 'source_group': 'tensorfold-connected-training-history', 'protected': False,
                 'outcome': 'reported_improvement', 'split': 'train',
                 'context_policy': 'complete_changed_files', 'task_conditioning': 'historical_source',
                 'task_kind': {'value': 'kernel', 'reviewed_by': 'root source review',
                               'evidence': [{'source': 'commit_message', 'quote': title}]},
                 'annotations': {'operation': annotation(title, title), 'rationale': annotation(message, message)}}
    if sha.startswith('9b32d238'):
        candidate['outcome'] = 'source_only'  # numerical evidence, no timing in captured message
    if True:
        candidate['task_kind']['value'] = 'host_dispatch'
    try:
        record = reconstruct_candidate(candidate, selection['repositories'], {})
        dest = ROOT / 'sourcepacks' / case
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'complete-record.json').write_text(json.dumps(record, indent=2) + '\n')
        (dest / 'commit.diff').write_text(record['diff'])
        units = []
        for name, pair in record['files'].items():
            item = {'path': name, 'kind': 'whole_file'}
            for side in ['before', 'after']:
                raw = (pair[side] or {'content': ''})['content'].encode()
                item[side] = {'start': 0, 'end': len(raw), 'sha256': H(raw)}
            units.append(item)
        candidate['context_policy'] = 'reviewed_units'
        candidate['reviewed_units'] = {'reviewed_by': 'root full-source inventory',
            'dependency_review': 'All changed files retained. Standalone dependency closure and runtime qualification pending.',
            'units': units}
        candidates.append(candidate)
        ledger.append({'id': case, 'status': 'complete_source_reconstructed', 'files': len(record['files']),
                       'patch_bytes': len(record['diff'].encode()), 'outcome': candidate['outcome']})
    except Exception as error:
        ledger.append({'id': case, 'status': 'source_error', 'error': str(error)})
manifest = {'format': 'exl3-mlx-upstream-commits-v1', 'repositories': selection['repositories'],
            'evidence': {}, 'candidates': candidates,
            'protected_keys': ['repository:https://github.com/ggml-org/llama.cpp']}
manifest['manifest_sha256'] = H(CANON(manifest))
(ROOT / 'manifest-fullfiles.json').write_text(json.dumps(manifest, indent=2) + '\n')
(ROOT / 'source-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
print(json.dumps(ledger, indent=2))
