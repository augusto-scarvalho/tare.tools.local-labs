"""Bounded OpenJEV footprint/quality probe. No qualification or fleet mutation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen


def gpu_memory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free',
        '--format=csv,noheader,nounits'], timeout=5, text=True)
    used, free = [float(v.strip()) for v in raw.splitlines()[0].split(',')]
    return {'used_mib': used, 'free_mib': free}


def json_http(path, data=None):
    request = Request('http://127.0.0.1:8080' + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'hybrid'], default='cpu')
    parser.add_argument('--qwen-coexist', action='store_true')
    args = parser.parse_args()
    if args.device != 'cpu':
        from tare_node.gpu_lease_run import inherited_lease
        if inherited_lease() is None:
            raise ValueError('GPU_PROBE_REQUIRES_SUPERVISING_LEASE')
    if args.qwen_coexist and args.device != 'cpu':
        raise ValueError('EXCLUSIVE_LEASE_DISALLOWS_SIMULTANEOUS_GPU_INFERENCE')
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol['schema'] != 'tare.tools/laya-strategy-screen/1' or len(protocol['cases']) != 6:
        raise ValueError('PROBE_REQUIRES_FROZEN_SIX_CASE_PROTOCOL')
    report = {'schema': 'tare.local-labs/openjev-probe/1', 'status': 'STARTED', 'authority': 'NONE',
        'qualification': 'UNQUALIFIED', 'device': args.device, 'pid': os.getpid(),
        'protocol_sha256': hashlib.sha256(raw).hexdigest(),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': [], 'nli_controls': [], 'memory': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out)
    lock = threading.Lock()
    def save():
        with lock:
            tmp = args.output.with_suffix('.tmp')
            tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            tmp.replace(args.output)
    import psutil
    process = psutil.Process()
    if psutil.virtual_memory().available < (36 if args.device == 'cpu' else 26)*1024**3:
        report.update(status='REFUSED', reason='INSUFFICIENT_RAM_HEADROOM'); save(); return 2
    if args.device != 'cpu' and gpu_memory()['free_mib'] < 16*1024:
        report.update(status='REFUSED', reason='INSUFFICIENT_VRAM_HEADROOM'); save(); return 2
    stop = threading.Event()
    start = time.monotonic()
    def monitor():
        breaches = 0
        while not stop.is_set():
            try:
                sample = {'seconds': time.monotonic()-start, 'rss_bytes': process.memory_info().rss,
                          'available_ram_bytes': psutil.virtual_memory().available, 'gpu': gpu_memory()}
                with lock:
                    report['memory'].append(sample)
                bad = (sample['available_ram_bytes'] < 16*1024**3
                       or (args.device != 'cpu' and sample['gpu']['free_mib'] < 4096))
                breaches = breaches + 1 if bad else 0
                if breaches >= 3:
                    report.update(status='ABORTED', reason='MEMORY_RESERVE_BREACH'); save()
                    os.kill(os.getpid(), signal.SIGTERM)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                report['monitor_error'] = type(exc).__name__
                report.update(status='ABORTED', reason='RESOURCE_OBSERVATION_FAILED'); save()
                os.kill(os.getpid(), signal.SIGTERM)
            stop.wait(1)
    threading.Thread(target=monitor, daemon=True).start()
    def timeout(signum, frame):
        raise TimeoutError('CASE_OR_LOAD_TIMEOUT')
    signal.signal(signal.SIGALRM, timeout)
    sys.path.insert(0, str(args.kernel_root.resolve()))
    try:
        from compute_plane.openjev_choice_worker import OpenJevChoiceModel
        signal.alarm(180)
        model = OpenJevChoiceModel(args.checkpoint, device=args.device, cuda_layers=4 if args.device == 'hybrid' else 0)
        signal.alarm(0)
        report.update(load_seconds=time.monotonic()-start, model_identity=model.identity,
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.model.parameters()),
            parameters_by_device={})
        for p in model.model.parameters():
            key = str(p.device)
            report['parameters_by_device'][key] = report['parameters_by_device'].get(key, 0) + p.numel()*p.element_size()
        save()
        # Three distinct labels prevent a constant-entailment scorer from passing.
        for hypothesis, expected in [('The server is running.', 'entailment'),
                                     ('The server is stopped.', 'contradiction'),
                                     ('The server is hosted in Berlin.', 'neutral')]:
            before = time.monotonic(); signal.alarm(180)
            result = model.predict('The server is running.', hypothesis)
            signal.alarm(0)
            actual = max(result['nli_probabilities'], key=result['nli_probabilities'].get)
            report['nli_controls'].append({'premise': 'The server is running.', 'hypothesis': hypothesis,
                'expected': expected, 'actual': actual, 'seconds': time.monotonic()-before, **result})
            save(); print(json.dumps({'nli': actual, 'expected': expected}), flush=True)
        if args.qwen_coexist:
            # The CPU model remains resident while the ordinary gateway performs
            # a lease-protected Qwen inference. No forced GPU admission.
            before = time.monotonic()
            report['qwen_coexist'] = {'before': json_http('/health')}
            save()
            answer = json_http('/v1/chat/completions', {'model': 'qwen38',
                'messages': [{'role': 'user', 'content': 'Reply with OK.'}], 'max_tokens': 8,
                'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
            report['qwen_coexist'].update(seconds=time.monotonic()-before, model=answer.get('model'),
                usage=answer.get('usage'), after=json_http('/health'), gpu=gpu_memory())
            save()
        for case in protocol['cases']:
            before = time.monotonic(); signal.alarm(180)
            answer = model.assess(case['request'])
            signal.alarm(0)
            chosen = case['choice_identity'].get(answer['choice'])
            report['cases'].append({'id': case['id'], 'answer': answer, 'request': case['request'],
                'nli': model.last_observations, 'seconds': time.monotonic()-before,
                'correct': chosen == case['expected_identity'], 'chosen_identity': chosen})
            save(); print(json.dumps({'case': case['id'], 'correct': report['cases'][-1]['correct'],
                                     'seconds': report['cases'][-1]['seconds']}), flush=True)
        if args.device != 'cpu':
            report['cuda_memory'] = {'allocated_bytes': model.torch.cuda.memory_allocated(),
                'reserved_bytes': model.torch.cuda.memory_reserved(),
                'peak_allocated_bytes': model.torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': model.torch.cuda.max_memory_reserved()}
        report.update(status='MEASURED', correct=sum(r['correct'] for r in report['cases']))
    except Exception as exc:
        report.update(status='FAILED', reason=type(exc).__name__, detail=str(exc))
        print(json.dumps({'error': str(exc)}), flush=True)
    finally:
        signal.alarm(0); stop.set()
        report['total_seconds'] = time.monotonic()-start
        save()
    return 0 if report['status'] == 'MEASURED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
