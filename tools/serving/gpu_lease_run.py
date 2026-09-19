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


def supervise(command, pending_signals, grace):
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=True)
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
    parser.add_argument('--gateway', default='http://127.0.0.1:8080')
    parser.add_argument('--gpu-lock', type=Path,
                        default=Path.home()/'.local/state/tare-qualified-models/gpu.lock')
    parser.add_argument('--gpu-wait', type=positive_seconds, default=3600,
                        help='maximum acquisition wait in seconds (default: 3600; no runtime limit)')
    parser.add_argument('--stop-grace', type=positive_seconds, default=30,
                        help='seconds after a stop request before SIGKILL; never authorizes lease release')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] != ['--'] or len(args.command) < 2:
        parser.error('supply the training command after --')
    if not sys.platform.startswith('linux'):
        parser.error('run on the GPU host in Linux/WSL, as the serving user')

    pending_signals = []
    def interrupted(signum, _frame):
        pending_signals.append(signum)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    was_subreaper = None
    request_id = 'training-' + uuid4().hex
    try:
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
            code = supervise(args.command[1:], pending_signals, args.stop_grace)
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
