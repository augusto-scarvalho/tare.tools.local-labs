"""Resident-Qwen decisions through gateway-owned leases, with admitted CPU fallback."""
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class Unavailable(Exception):
    pass


def exit_with_failure(reason, stage, exit_code):
    """Retain a small protocol receipt before a hard stop, including in Torch.

    stdout is redirected to stderr during evaluation; descriptor 1 remains the
    protocol channel. Do not raise through native inference or wait for cleanup.
    The process exit releases the CPU flock and the OS reclaims its memory.
    """
    payload = json.dumps({'failure': reason, 'stage': stage}).encode() + b'\n'
    try:
        os.write(1, payload)
    finally:
        os._exit(exit_code)


def http(port, path, data=None, timeout=5):
    request = Request(f'http://127.0.0.1:{port}{path}',
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(1024*1024+1)
    if len(raw) > 1024*1024:
        raise ValueError('ASSESSMENT_OUTPUT_OVERFLOW')
    return json.loads(raw)


def gpu_assess(request, config):
    from model_lifecycle.gpu_lease import SharedGpuLease
    from compute_plane.litjev_choice import assess
    lease = SharedGpuLease(config['gpu_lock'], timeout=.15)
    deadline = time.monotonic()+12
    with contextlib.ExitStack() as preparation:
        try:
            preparation.enter_context(lease.hold('text', 'tare-decision-prep-'+str(os.getpid())))
        except TimeoutError:
            raise Unavailable('GPU_LEASE_BUSY') from None
        try:
            health = http(8080, '/health', timeout=2)
            queue = http(8188, '/queue', timeout=2)
            if (health.get('current_model') != config['model'] or not health.get('backend_healthy')
                    or health.get('resident_readout') != 'pid-bound-v1'
                    or queue.get('queue_running') or queue.get('queue_pending')):
                raise Unavailable('RESIDENT_QWEN_UNAVAILABLE')
            if http(18080, '/props')['model_path'] != config['model_path']:
                raise Unavailable('RESIDENT_IDENTITY_CHANGED')
        except (OSError, ValueError, KeyError):
            raise Unavailable('GPU_STATE_UNAVAILABLE') from None
        pid = health['backend_pid']

        def post(path, data):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise Unavailable('GPU_PREPARATION_TIMEOUT')
            if path != '/completion':
                try:
                    return http(18080, path, data, timeout=min(5, remaining))
                except (OSError, ValueError):
                    raise Unavailable('GPU_PREPARATION_UNAVAILABLE') from None
            # Tokenization is CPU-only. Transfer admission to the gateway for
            # actual inference; it rechecks this exact resident PID under its
            # lease and never loads/switches a model for this request.
            preparation.close()
            try:
                return http(8080, '/completion', {**data, 'model': config['model'],
                    '_tare_resident_backend_pid': pid}, timeout=min(10, remaining))
            except HTTPError as exc:
                try:
                    error = json.loads(exc.read(4096))['error']['type']
                except (ValueError, KeyError, TypeError):
                    raise ValueError('ASSESSMENT_RESOURCE_UNAVAILABLE') from None
                if error in {'resident_backend_unavailable', 'gpu_lease_timeout'}:
                    raise Unavailable('RESIDENT_ADMISSION_CHANGED') from None
                raise ValueError('ASSESSMENT_RESOURCE_UNAVAILABLE') from None
            except (OSError, ValueError):
                # Unknown submission outcome must not launch a second evaluator.
                raise ValueError('ASSESSMENT_RESOURCE_UNAVAILABLE') from None

        return assess(request, post, {'model': config['model'], 'backend_pid': pid,
            'artifact_sha256': config['artifact_sha256'],
            'artifact_identity_source': 'configured-artifact-and-live-path',
            'placement': 'resident-gpu', 'fallback_used': False})


def cpu_assess(request, config):
    import fcntl
    import psutil
    from compute_plane.openjev_choice_worker import OpenJevChoiceModel, preflight_cpu_tokens
    lock = Path(config['cpu_lock'])
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('ASSESSMENT_RESOURCE_BUSY') from None
        if psutil.virtual_memory().available < 36*1024**3:
            raise ValueError('ASSESSMENT_INSUFFICIENT_RAM')
        stage = 'CPU_INPUT_PREFLIGHT'
        def stopped(signum, _frame):
            if signum == signal.SIGALRM:
                exit_with_failure('ASSESSMENT_TIMEOUT', stage, 124)
            exit_with_failure('ASSESSMENT_CANCELLED', stage, 128 + signum)
        handlers = {number: signal.signal(number, stopped)
                    for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.alarm(52)
        try:
            # Six short hypotheses can be cheap; six long ones repeat the full
            # premise and exceed the interactive CPU budget. Never truncate it.
            preflight_cpu_tokens(config['checkpoint'], request, config.get('cpu_max_total_tokens'))
            stage = 'CPU_MODEL_LOAD'
            model = OpenJevChoiceModel(config['checkpoint'])
            stage = 'CPU_SCORING'
            answer = model.assess(request)
            answer['model_identity'].update(fallback_used=True)
            return answer
        finally:
            signal.alarm(0)
            for number, handler in handlers.items():
                signal.signal(number, handler)


def evaluate(request, config):
    from compute_plane.choice_assessment import validate_request
    validate_request(request)
    cpu_enabled = config.get('cpu_fallback_enabled', True)
    if type(cpu_enabled) is not bool:
        raise ValueError('ASSESSMENT_EXECUTION_CONFIG_INVALID')
    try:
        return gpu_assess(request, config)
    except Unavailable as exc:
        if not cpu_enabled:
            # A known resident admission refusal permits the caller's vendor
            # fallback. Never cold-load the CPU model for this boundary.
            raise ValueError('ASSESSMENT_CPU_NOT_ADMITTED') from None
        answer = cpu_assess(request, config)
        answer['model_identity']['fallback_reason'] = str(exc)
        return answer


def main():
    config = json.loads(Path(sys.argv[1]).read_bytes())
    sys.path[:0] = [config['kernel_root'], config['labs_src']]
    try:
        raw = sys.stdin.buffer.read(8193)
        if len(raw) > 8192:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        with contextlib.redirect_stdout(sys.stderr):
            answer = evaluate(json.loads(raw), config)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        reason = str(exc)
        answer = {'failure': reason if reason in {'ASSESSMENT_INPUT_OVERFLOW',
            'ASSESSMENT_CHECKPOINT_CHANGED', 'ASSESSMENT_INSUFFICIENT_RAM',
            'ASSESSMENT_ENVIRONMENT_CHANGED', 'ASSESSMENT_RESOURCE_BUSY',
            'ASSESSMENT_CAPACITY_EXCEEDED', 'ASSESSMENT_EXECUTION_CONFIG_INVALID',
            'ASSESSMENT_CPU_NOT_ADMITTED',
            'ASSESSMENT_RESOURCE_UNAVAILABLE'} else 'ASSESSMENT_RESOURCE_UNAVAILABLE'}
        if reason == 'ASSESSMENT_CAPACITY_EXCEEDED':
            answer['stage'] = 'CPU_INPUT_PREFLIGHT'
    print(json.dumps(answer, allow_nan=False))


if __name__ == '__main__':
    main()
