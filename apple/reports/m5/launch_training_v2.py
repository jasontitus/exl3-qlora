"""One-shot detached invocation of the concrete, separately reviewed M5 campaign."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path('/Volumes/Scratch/exl3-training/2026-10-13')
REQUEST = ROOT / 'training-lease-request-v2.json'
REQUEST_SHA = '33549532854732e70b9640ec2a605e7afb7eae0d52f04d148c43c1a8489bc17d'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if socket.gethostname() != 'Jason-M5-MacBook.localdomain':
        raise ValueError('Wrong host')
    if digest(REQUEST) != REQUEST_SHA:
        raise ValueError('Request changed')
    request = json.loads(REQUEST.read_text())
    for name, expected in request['inputs_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Changed input: ' + name)
    python = Path(request['argv'][0])
    if str(python.resolve(strict=True)) not in request['inputs_sha256']:
        raise ValueError('Unbound Python target')
    reports = ROOT / 'reports'
    launch = reports / 'metal-mix198-detached-launch-v1.json'
    stdout = reports / 'metal-mix198-wrapper-v1.stdout.log'
    stderr = reports / 'metal-mix198-wrapper-v1.stderr.log'
    wrapper_receipt = reports / 'metal-mix198-wrapper-v1.json'
    for path in [launch, stdout, stderr, wrapper_receipt, Path(request['completion_receipt'])]:
        if path.exists() or path.is_symlink():
            raise ValueError('Existing output; inspect rather than retry: ' + str(path))
    env = {k: v for k, v in os.environ.items() if not k.startswith('MLX_')}
    env['MLX_ENABLE_TF32'] = '0'
    argv = [str(python), str(ROOT / 'shared_gpu_lease_v2.py'), 'run',
            '--request', str(REQUEST), '--sha256', REQUEST_SHA,
            '--receipt', str(wrapper_receipt)]
    record = {
        'schema': 'm5-concrete-detached-launch-v1', 'status': 'prepared',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'argv': argv, 'cwd': request['cwd'], 'request_sha256': REQUEST_SHA,
        'launcher_sha256': digest(__file__), 'inputs_sha256': request['inputs_sha256'],
        'numerical_environment': {'MLX_ENABLE_TF32': '0'},
        'stdout': str(stdout), 'stderr': str(stderr),
        'wrapper_receipt': str(wrapper_receipt), 'quality_certified': False,
    }
    with launch.open('x') as file:
        json.dump(record, file, indent=2)
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen(argv, cwd=request['cwd'], env=env,
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=True)
    record['wrapper_pid'] = child.pid
    time.sleep(0.5)
    record['observed_exit_code'] = child.poll()
    if record['observed_exit_code'] is None:
        observed = subprocess.run(
            ['/bin/ps', '-p', str(child.pid), '-o', 'pid=,ppid=,pgid=,lstart=,command='],
            capture_output=True, text=True, check=True).stdout.strip()
        if not observed or observed.split()[0] != str(child.pid):
            raise RuntimeError('No live child identity; inspect outputs before recovery')
        record['process_identity'] = observed
        record['status'] = 'wrapper_started'
        # Sleep prevention is separate from GPU ownership and follows this PID only.
        wake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(child.pid)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        record['caffeinate'] = {'pid': wake.pid, 'waits_for_pid': child.pid}
    else:
        record['status'] = 'wrapper_exited_before_observation'
    launch.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
