"""Build fresh next-round corpora, preserving the current held-out selection."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
PYTHON = '/Users/jasontitus/experiments/exl3-qlora/apple/.venv/bin/python'
MODEL = '/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw'


def main():
    train = [ROOT / 'preserved/tensorfold/manifest.json', ROOT / 'preserved/mlx-sushi/manifest.json',
             ROOT / 'group-a/manifest-reviewed.json', ROOT / 'group-b/manifest-reviewed-v2.json']
    test = ROOT / 'preserved/llama/manifest.json'
    selected = set()
    for path in [*train, test]:
        data = json.loads(path.read_text())
        selected.update(c['id'] for c in data['candidates'])
    omissions = {}
    for path in [ROOT / 'preserved/documentation-omissions.json',
                 ROOT / 'group-a/documentation-omissions.json', ROOT / 'group-b/documentation-omissions.json']:
        for key, value in json.loads(path.read_text()).items():
            if key in selected:
                if key in omissions:
                    raise ValueError('Duplicate documentation decision: ' + key)
                omissions[key] = value
    policy = ROOT / 'combined-documentation-omissions-v2.json'
    raw = (json.dumps(omissions, indent=2) + '\n').encode()
    with policy.open('xb') as stream:
        stream.write(raw)
    commands = []
    for length in (32768, 12288):
        out = ROOT / f'corpus-{length}-v2'
        command = [PYTHON, str(ROOT / 'build_next_corpus.py')]
        for path in train:
            command += ['--train-manifest', str(path)]
        command += ['--test-manifest', str(test), '--model', MODEL, '--output', str(out),
                    '--max-length', str(length), '--documentation-omissions', str(policy)]
        with (ROOT / f'build-{length}-v2.stdout.log').open('x') as stdout, (ROOT / f'build-{length}-v2.stderr.log').open('x') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr,
                                    env={**os.environ, 'PYTHONPATH': str(WORK / 'loss-fix/apple/src')})
        commands.append({'argv': command, 'exit_code': result.returncode})
        (ROOT / 'assembly-receipt-v2.json').write_text(json.dumps({'commands': commands,
            'documentation_policy_sha256': hashlib.sha256(raw).hexdigest(),
            'scope': 'Prepared historical imitation dataset; no new GPU training or execution qualification'}, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
