#!/usr/bin/env python3
"""Run a Linux training command under the gateway/ComfyUI exclusive GPU lease."""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from model_lifecycle.comfy_gpu import yield_text_backend
from model_lifecycle.gpu_lease import SharedGpuLease

LEASE_CONTEXT = 'TARE_GPU_TRAINING_LEASE'


def inherited_lease(lock_path=None, gateway=None):
    """Reuse only a live image lease held by an ancestor supervising this job.

    An environment flag alone is not authority. Verify the kernel lock, nonce,
    owner PID and ancestry before allowing a nested command to skip acquisition.
    """
    raw = os.environ.get(LEASE_CONTEXT)
    if raw is None:
        return None
    try:
        if len(raw) > 4096:
            raise ValueError()
        context = json.loads(raw)
        if (not isinstance(context, dict) or set(context) != {'path', 'gateway', 'pid', 'nonce'}
                or type(context['pid']) is not int or context['pid'] <= 1
                or not isinstance(context['path'], str) or not Path(context['path']).is_absolute()
                or not isinstance(context['gateway'], str) or not isinstance(context['nonce'], str)):
            raise ValueError()
        if (lock_path is not None and Path(lock_path).resolve() != Path(context['path']).resolve()
                or gateway is not None and gateway.rstrip('/') != context['gateway'].rstrip('/')):
            raise ValueError()
        parent = os.getppid()
        for _ in range(256):
            if parent == context['pid']:
                break
            if parent <= 1:
                raise ValueError()
            stat = (Path('/proc')/str(parent)/'stat').read_text()
            parent = int(stat.rsplit(')', 1)[1].split()[1])
        else:
            raise ValueError()
        lease = SharedGpuLease(context['path'])
        status = lease.status()
        owner = status.get('owner') or {}
        if (not status.get('held') or owner.get('pid') != context['pid']
                or owner.get('kind') != 'image' or not lease.validate_image_lease(context['nonce'])):
            raise ValueError()
        return context
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError('Inherited GPU lease is invalid or no longer held; refusing unprotected execution.') from None


def report(event, **fields):
    try:
        print(json.dumps({'event': event, **fields}), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # A closed diagnostic pipe cannot interrupt process cleanup.


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return value


def subreaper(enabled=None):
    """Adopt orphaned grandchildren, including workers that call setsid()."""
    libc = ctypes.CDLL(None, use_errno=True)
    if enabled is None:
        value = ctypes.c_int()
        result = libc.prctl(37, ctypes.byref(value), 0, 0, 0)  # PR_GET_CHILD_SUBREAPER
    else:
        result = libc.prctl(36, int(enabled), 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    if result != 0:
        raise OSError(ctypes.get_errno(), 'Cannot establish training process supervision')
    return value.value if enabled is None else None


def descendants():
    """Read this supervisor's descendants, across every thread's child list."""
    seen, pending = set(), [os.getpid()]
    while pending:
        parent = pending.pop()
        try:
            tasks = list((Path('/proc')/str(parent)/'task').iterdir())
        except FileNotFoundError:
            if parent == os.getpid():
                raise  # Missing procfs is not proof that workers exited.
            continue
        for task in tasks:
            try:
                children = (task/'children').read_text().split()
            except FileNotFoundError:
                continue
            for raw in children:
                pid = int(raw)
                if pid not in seen:
                    seen.add(pid)
                    pending.append(pid)
    return seen


def reap(process):
    """Wait for the root before reaping adopted workers, preserving its exit code."""
    if process.poll() is None:
        return
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                return
        except ChildProcessError:
            return


def signal_tree(process, signum):
    # The process group covers forks racing with enumeration. Adopted/setsid
    # workers are also signalled individually and rescanned during cleanup.
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass
    for pid in descendants():
        try:
            if os.getpgid(pid) != process.pid:
                os.kill(pid, signum)
        except ProcessLookupError:
            pass


def drain(process, signum, grace):
    """Retain ownership until all children are observed exited and reaped.

    The grace period controls escalation, never lease release. Unobservable or
    unkillable children quarantine the supervisor with the lease still held.
    """
    deadline = time.monotonic() + grace
    warned, sent = False, set()
    while True:
        try:
            reap(process)
            live = descendants()
            if process.returncode is not None and not live:
                return
            force = time.monotonic() >= deadline
            current_signal = signal.SIGKILL if force else signum
            if force or live - sent:
                signal_tree(process, current_signal)
                sent |= live
            if force and not warned:
                report('cleanup_wait', message='Grace elapsed; sent SIGKILL. Lease retained until process exit is confirmed.')
                warned = True
        except (OSError, ValueError) as exc:
            if not warned:
                report('cleanup_unconfirmed', error=str(exc), message='Lease retained; process cleanup needs attention.')
                warned = True
        time.sleep(.1)


def supervise(command, pending_signals, grace, *, env=None):
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=True, env=env)
        report('started', pid=process.pid)
        while process.poll() is None:
            if pending_signals:
                drain(process, pending_signals[0], grace)
                break
            time.sleep(.1)
    finally:
        if process is not None:
            drain(process, pending_signals[0] if pending_signals else signal.SIGTERM, grace)
    code = process.returncode
    report('processes_reaped', child_exit_code=code)
    return code if code >= 0 else 128 - code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', help='gateway URL (default: local serving gateway, or verified parent lease)')
    parser.add_argument('--gpu-lock', type=Path, help='lock file (default: serving lock, or verified parent lease)')
    parser.add_argument('--check-inherited', action='store_true',
                        help='check a supervising parent lease without acquiring the GPU; exit 1 when absent')
    parser.add_argument('--gpu-wait', type=positive_seconds, default=3600,
                        help='maximum acquisition wait in seconds (default: 3600; no runtime limit)')
    parser.add_argument('--stop-grace', type=positive_seconds, default=30,
                        help='seconds after a stop request before SIGKILL; never authorizes lease release')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.check_inherited and command:
        parser.error('--check-inherited does not execute a command')
    if not args.check_inherited and not command:
        parser.error('supply a training script or command (optionally after --)')
    if command and command[0].endswith('.sh'):
        command = ['bash', *command]
    if not sys.platform.startswith('linux'):
        parser.error('run on the GPU host in Linux/WSL, as the serving user')

    pending_signals = []
    def interrupted(signum, _frame):
        pending_signals.append(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    was_subreaper = None
    request_id = 'training-' + uuid4().hex
    try:
        inherited = inherited_lease(args.gpu_lock, args.gateway)
        if args.check_inherited:
            return 0 if inherited else 1
        if inherited:
            report('lease_reused', supervisor_pid=inherited['pid'])
            os.execvpe(command[0], command, os.environ)
        args.gateway = args.gateway or 'http://127.0.0.1:8080'
        args.gpu_lock = args.gpu_lock or Path.home()/'.local/state/tare-qualified-models/gpu.lock'
        was_subreaper = subreaper()
        subreaper(True)
        lease = SharedGpuLease(args.gpu_lock, timeout=args.gpu_wait)
        report('waiting', request_id=request_id, timeout_seconds=args.gpu_wait, lock=str(args.gpu_lock))
        with lease.hold('image', request_id, cancelled=lambda: bool(pending_signals)) as receipt:
            report('acquired', request_id=request_id)
            yield_text_backend(args.gateway, receipt['nonce'])
            report('text_backend_released', request_id=request_id)
            if pending_signals:
                return 128 + pending_signals[0]
            context = {'path': str(args.gpu_lock.resolve()), 'gateway': args.gateway,
                       'pid': os.getpid(), 'nonce': receipt['nonce']}
            code = supervise(command, pending_signals, args.stop_grace,
                             env={**os.environ, LEASE_CONTEXT: json.dumps(context)})
        report('lease_released', request_id=request_id, exit_code=code)
        return code
    except InterruptedError:
        report('cancelled_while_waiting', request_id=request_id)
        return 128 + pending_signals[0] if pending_signals else 130
    except TimeoutError as exc:
        report('acquisition_timeout', request_id=request_id, error=str(exc))
        return 124
    except Exception as exc:
        report('failed', request_id=request_id, error=str(exc))
        return 127 if isinstance(exc, OSError) and exc.errno == errno.ENOENT else 125
    finally:
        if was_subreaper is not None:
            subreaper(was_subreaper)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
