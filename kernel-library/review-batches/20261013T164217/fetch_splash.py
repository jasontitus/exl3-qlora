"""Read public pinned Git objects only; never checkout or execute fetched code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
HEAD = '07142b12bc86fff40187b116a06e272de8489eb5'
PARENT = '5ecf1fe8867683fbd39ce63cd659e5d2b1056b3c'
REPO = ROOT / 'splash.git'
LOG = []
ENV = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_NO_REPLACE_OBJECTS': '1',
       'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null'}


def git(*args):
    argv = ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'diff.external=',
            '--git-dir=' + str(REPO), *args]
    p = subprocess.run(argv, env=ENV, capture_output=True, timeout=120)
    LOG.append({'argv': argv, 'returncode': p.returncode,
                'stdout_sha256': hashlib.sha256(p.stdout).hexdigest(),
                'stderr': p.stderr.decode(errors='replace')[-3000:]})
    (ROOT / 'fetch-log.json').write_text(json.dumps(LOG, indent=2) + '\n')
    p.check_returncode()
    return p.stdout


git('init', '--bare')
git('remote', 'add', 'origin', 'https://github.com/incoai/splash.git')
git('fetch', '--no-tags', '--depth=2', '--filter=blob:none', 'origin', HEAD)
assert git('show', '-s', '--format=%P', HEAD).decode().strip() == PARENT
paths = git('diff', '--no-ext-diff', '--no-textconv', '--no-renames',
            '--name-only', '-z', PARENT, HEAD).decode().strip('\0').split('\0')
assert len(paths) <= 24
license_paths = [p for p in git('ls-tree', '--name-only', HEAD).decode().splitlines()
                 if p.upper().startswith(('LICENSE', 'COPYING'))]
files = []
for revision, label in [(PARENT, 'before'), (HEAD, 'after')]:
    for name in dict.fromkeys(paths + license_paths):
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        entry = git('ls-tree', '-z', revision, '--', name).rstrip(b'\0')
        if not entry:
            files.append({'revision': revision, 'path': name, 'absent': True})
            continue
        mode, kind, oid = entry.split(b'\t')[0].split()
        assert kind == b'blob'
        assert int(git('cat-file', '-s', oid.decode())) <= 8 << 20
        raw = git('cat-file', 'blob', oid.decode())
        destination = ROOT / label / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        files.append({'revision': revision, 'path': name, 'mode': mode.decode(),
                      'git_blob': oid.decode(), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw), 'local_path': str(destination.relative_to(ROOT))})
patch = git('diff', '--no-ext-diff', '--no-textconv', '--no-renames', '--binary', PARENT, HEAD)
(ROOT / 'complete.patch').write_bytes(patch)
(ROOT / 'source-receipt.json').write_text(json.dumps({
    'repository': 'https://github.com/incoai/splash', 'head': HEAD, 'parent': PARENT,
    'changed_paths': paths, 'license_paths': license_paths, 'files': files,
    'patch_sha256': hashlib.sha256(patch).hexdigest(),
    'execution': 'No fetched code executed; no compilation or GPU verification',
    'training_eligible': False, 'protection_review': 'pending',
}, indent=2) + '\n')
print('Pinned complete before/after source and license captured:', len(files), 'entries')
