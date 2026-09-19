#!/usr/bin/env python3
"""Install a verified serving release with reversible systemd overrides.

Run as root on the workstation. No model, workflow, output, repository source,
or existing service unit is replaced. Admission requires an empty ComfyUI queue.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import time
from urllib.request import urlopen
from uuid import uuid4

UNITS = ('comfyui.service', 'llm-inference.service')
SYSTEMD_ROOT = Path('/etc/systemd/system')


def systemctl(*args):
    result = subprocess.run(['systemctl', *args], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError('systemctl ' + ' '.join(args) + ' failed: ' + result.stderr[-1500:])
    return result.stdout.strip()


def get(url):
    with urlopen(url, timeout=5) as response:
        raw = response.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('Readiness response exceeded its limit.')
    return json.loads(raw)


def wait_ready(url, predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = get(url)
            if predicate(value):
                return value
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError('Readiness deadline exceeded for ' + url)


def quoted(argument):
    argument = str(argument)
    if any(c in argument for c in ('\n', '\r', '%')):
        raise ValueError('Unsupported systemd argument.')
    return '"' + argument.replace('\\', '\\\\').replace('"', '\\"') + '"'


def override(argv, environment=()):
    lines = ['[Service]', 'ExecStart=', 'ExecStart=' + ' '.join(map(quoted, argv))]
    lines.extend('Environment=' + quoted(value) for value in environment)
    return ('\n'.join(lines) + '\n').encode()


def verify_release(root):
    manifest = json.loads((root/'release_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = root/name
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise ValueError('Serving release identity mismatch: ' + name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--user', default='augus')
    parser.add_argument('--comfy-root', type=Path, default=Path('/mnt/c/projects/ComfyUI'))
    parser.add_argument('--config', type=Path, default=Path('/mnt/c/projects/tare.tools.local-labs/config/qualified_model_fleet.json'))
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise PermissionError('Systemd migration requires root.')
    release = args.release.resolve(strict=True)
    manifest = verify_release(release)
    source_manifest = release/'comfy_sources.json'
    for name, digest in json.loads(source_manifest.read_text()).items():
        if hashlib.sha256((args.comfy_root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Installed ComfyUI changed before migration: ' + name)
    if hashlib.sha256(args.config.read_bytes()).hexdigest() != manifest['existing_config_sha256']:
        raise ValueError('Existing model registry changed before migration.')
    user = pwd.getpwnam(args.user)
    state = Path(user.pw_dir)/'.local/state/tare-qualified-models'
    if not state.exists():
        state.mkdir(parents=True, mode=0o700)
        os.chown(state, user.pw_uid, user.pw_gid)
    if state.stat().st_uid != user.pw_uid:
        raise ValueError('Serving state must belong to the service user.')
    active = {unit: systemctl('show', unit, '-p', 'ActiveState', '--value') for unit in UNITS}
    embedding_before = systemctl('show', 'llm-embedding.service', '-p', 'MainPID', '--value')
    if active['llm-inference.service'] == 'active':
        raise ValueError('This initial migration expects the text service to be inactive.')
    if active['comfyui.service'] != 'active':
        raise ValueError('Expected the existing ComfyUI service to be active.')
    stats = get('http://127.0.0.1:8080/system_stats')
    if not stats.get('system', {}).get('comfyui_version'):
        raise ValueError('Port 8080 did not prove ComfyUI identity.')

    lock = state/'gpu.lock'
    history_path = state/('comfy-history-'+uuid4().hex+'.json')
    gateway = ['/usr/bin/python3', release/'tools/serving/qualified_model_gateway.py',
               '--config', args.config, '--host', '0.0.0.0', '--port', '8080',
               '--backend-host', '127.0.0.1', '--backend-port', '18080',
               '--state-dir', state, '--gpu-lock', lock, '--route-timeout', '600',
               '--comfy-url', 'http://127.0.0.1:8188',
               '--image-routes', release/'config/comfy_image_routes.json']
    comfy = [Path(user.pw_dir)/'.venvs/comfyui/bin/python',
             release/'tools/serving/coordinated_comfy.py', '--comfy-root', args.comfy_root,
             '--gpu-lock', lock, '--source-manifest', source_manifest,
             '--history-snapshot', history_path, '--',
             '--listen', '0.0.0.0', '--port', '8188', '--preview-method', 'auto']
    replacements = {
        'llm-inference.service': override(gateway),
        'comfyui.service': override(comfy, (
            f'VIRTUAL_ENV={user.pw_dir}/.venvs/comfyui',
            f'PATH={user.pw_dir}/.venvs/comfyui/bin:{user.pw_dir}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib')),
    }
    targets = {unit: SYSTEMD_ROOT/(unit+'.d')/'zzz-shared-gpu.conf' for unit in UNITS}
    for path in targets.values():
        if path.is_symlink():
            raise ValueError('Service override must not be a symlink.')
    previous = {unit: path.read_bytes() if path.exists() else None for unit, path in targets.items()}
    receipt_dir = state/('migration-'+uuid4().hex)
    receipt_dir.mkdir(mode=0o700)
    receipt = {'state': 'PREPARED', 'release': str(release), 'original_active': active,
               'embedding_pid_before': embedding_before,
               'previous_overrides': {k: base64.b64encode(v).decode() if v is not None else None
                                      for k, v in previous.items()}}

    def checkpoint(stage):
        receipt['state'] = stage
        (receipt_dir/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps({'stage': stage, 'receipt': str(receipt_dir/'result.json')}), flush=True)

    checkpoint('PREPARED')
    queue = get('http://127.0.0.1:8080/queue')
    if queue.get('queue_running') != [] or queue.get('queue_pending') != []:
        checkpoint('REFUSED_BUSY_COMFY')
        raise ValueError('ComfyUI has queued/running work; migration refused.')
    history = get('http://127.0.0.1:8080/history')
    if not isinstance(history, dict):
        raise ValueError('Could not preserve completed ComfyUI history.')
    with history_path.open('x', encoding='utf-8') as stream:
        os.chmod(history_path, 0o600)
        os.chown(history_path, user.pw_uid, user.pw_gid)
        json.dump(history, stream, ensure_ascii=False)
    receipt['history_backup'] = str(history_path)
    receipt['history_entries'] = len(history)
    receipt['history_backup_mode'] = oct(history_path.stat().st_mode & 0o777)
    queue = get('http://127.0.0.1:8080/queue')
    if queue.get('queue_running') != [] or queue.get('queue_pending') != []:
        checkpoint('REFUSED_BUSY_COMFY')
        raise ValueError('ComfyUI became busy; migration refused.')

    def expire(signum, frame):
        raise TimeoutError('Service migration exceeded its six-minute deadline.')

    old_alarm = signal.signal(signal.SIGALRM, expire)
    signal.alarm(360)
    try:
        systemctl('stop', 'comfyui.service')
        checkpoint('OLD_COMFY_STOPPED')
        for unit, raw in replacements.items():
            targets[unit].parent.mkdir(exist_ok=True)
            targets[unit].write_bytes(raw)
        systemctl('daemon-reload')
        systemctl('start', 'llm-inference.service')
        wait_ready('http://127.0.0.1:8080/v1/fleet/status',
                   lambda d: d.get('role') == 'qualified-model-gateway'
                   and d.get('gpu_coordination', {}).get('enabled') is True)
        checkpoint('GATEWAY_READY')
        systemctl('start', 'comfyui.service')
        wait_ready('http://127.0.0.1:8188/tare/gpu/status',
                   lambda d: d.get('role') == 'tare-comfy-gpu-coordinator'
                   and d.get('cleanup_blocked') is False and d.get('gpu', {}).get('path') == str(lock))
        after = systemctl('show', 'llm-embedding.service', '-p', 'MainPID', '--value')
        if after != embedding_before:
            raise RuntimeError('Embedding service PID changed during migration.')
        receipt['embedding_pid_after'] = after
        receipt['qualification'] = 'Service readiness only; real text/image switching still requires verification.'
        checkpoint('SERVICES_READY')
    except BaseException as exc:
        signal.alarm(0)
        receipt['error'] = type(exc).__name__ + ': ' + str(exc)
        checkpoint('ROLLING_BACK')
        try:
            for unit in UNITS:
                systemctl('stop', unit)
            for unit, raw in previous.items():
                if raw is None:
                    targets[unit].unlink(missing_ok=True)
                else:
                    targets[unit].write_bytes(raw)
            systemctl('daemon-reload')
            for unit, status in active.items():
                if status == 'active':
                    systemctl('start', unit)
            wait_ready('http://127.0.0.1:8080/system_stats', lambda d: bool(d.get('system')))
            checkpoint('ROLLED_BACK')
        except BaseException as rollback_error:
            receipt['rollback_error'] = type(rollback_error).__name__ + ': ' + str(rollback_error)
            checkpoint('ROLLBACK_INCOMPLETE')
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


if __name__ == '__main__':
    main()
