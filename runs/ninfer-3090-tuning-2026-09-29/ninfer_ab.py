"""NInfer 0.11 A/B on node aaaaa: current qwen38-gsq flags vs the fork's recommended 3090 flags,
plus a LitJev-style first-token screen on the last config. Runs only while the desktop is idle,
holds the shared GPU lease the image pipeline uses, and gives the GPU back at the end."""
import json, math, os, subprocess, sys, time, urllib.request
from pathlib import Path

RELEASE = Path.home()/'.local/share/tare-qualified-models/releases/resident-ninfer011-0aa4cfa'
sys.path.insert(0, str(RELEASE/'src'))
from model_lifecycle.gpu_lease import SharedGpuLease  # noqa: E402
from model_lifecycle.comfy_gpu import yield_text_backend  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE/'results.json'
SERVE = str(Path.home()/'opt/ninfer/v0.11.0-rtx3090-f118551/ninfer-serve')
ARTIFACT = '/mnt/wsl/models/qwen38-gsq/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer'
PORT = 18090
BASE_FLAGS = ['--max-context', '131072', '--kv-capacity', 'auto', '--max-concurrency', '1', '--kv-dtype', 'rk8v4',
              '--gdn-state-fp16', '--default-thinking-budget', '2048', '--first-token-logprobs']
MTP = ['--spec', 'mtp', '--draft-tokens', '3', '--lm-head-draft']
CUBLAS = ['--prefill-cublas', '--prefill-chunk', '2048']
TRADES = ['--embedding-q4', '--lm-head-q6']
CONFIGS = [
    ('A-current', BASE_FLAGS + MTP),
    ('B-cublas', BASE_FLAGS + MTP + CUBLAS),
    ('C-cublas-trades', BASE_FLAGS + MTP + CUBLAS + TRADES),
    ('D-calibrated', BASE_FLAGS + MTP + CUBLAS + TRADES + ['--device-profile', 'calibrate']),
    ('E-dflash2', BASE_FLAGS + ['--spec', 'dflash2', '--draft-tokens', '7', '--lm-head-draft'] + CUBLAS
     + ['--embedding-q4']),
]
report = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'configs': {}}


def save():
    OUT.write_text(json.dumps(report, indent=1))


def post(path, body, timeout=600):
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def idle_seconds():
    try:
        return json.loads(Path('/mnt/c/ProgramData/tare/host-status.json').read_text())['desktop_idle_seconds']
    except (OSError, ValueError, KeyError):
        return 0


def gpu_used_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def prompts():
    source = Path.home()/'src/ninfer-3090-wavecut'
    text = (source/'README.md').read_text(errors='ignore') + (source/'docs/performance.md').read_text(errors='ignore')
    long_doc = text[:60000]  # roughly 15K tokens
    return long_doc


def chat(content, max_tokens, thinking=False, extra=None):
    body = {'model': 'ab', 'max_tokens': max_tokens, 'temperature': 0.0, 'seed': 7,
            'chat_template_kwargs': {'enable_thinking': thinking},
            'messages': [{'role': 'user', 'content': content}], **(extra or {})}
    started = time.time()
    reply = post('/v1/chat/completions', body)
    return reply, time.time() - started


def workload(name):
    rows = {'prefill': [], 'decode': [], 'texts': []}
    doc = prompts()
    for i in range(3):
        reply, wall = chat(f'Run {name}-{i}.\n\n{doc}\n\nSummarize the document above in one sentence.', 48)
        rows['prefill'].append({'wall_s': round(wall, 2), **(reply.get('timings') or {})})
    for i in range(3):
        reply, wall = chat(f'Run {i}. Write a Python module implementing an LRU cache class with get, put and '
                           'a capacity, plus five pytest tests. Code only.', 900)
        rows['decode'].append({'wall_s': round(wall, 2), **(reply.get('timings') or {})})
        rows['texts'].append(reply['choices'][0]['message'].get('content') or '')
    return rows


def start(flags, log):
    process = subprocess.Popen([SERVE, ARTIFACT, '--host', '127.0.0.1', '--port', str(PORT), '--model-id', 'ab', *flags],
                               stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 900
    while time.time() < deadline:
        if process.poll() is not None:
            return process, f'exited {process.returncode}'
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/v1/models', timeout=3)
            return process, None
        except OSError:
            time.sleep(3)
    return process, 'not ready in 900 s'


def stop(process):
    process.terminate()
    try:
        process.wait(60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    time.sleep(5)


def litjev(protocols):
    """First-token restricted softmax over option codes, as LitJEV, through Chat Completions."""
    results = {}
    for name, protocol in protocols.items():
        rows = []
        for case in protocol['cases']:
            request = case['request']
            keys = list(request['options'])
            codes = list('ABCDEF')[:len(keys)]
            question = 'Question: ' + json.dumps({'type': 'choice', 'instructions': request['question'], 'options': [
                {'code': c, 'option': k, 'description': request['options'][k]} for c, k in zip(codes, keys)]},
                ensure_ascii=False) + '\nAnswer:'
            body = {'model': 'ab', 'max_tokens': 1, 'temperature': 1.0, 'top_p': 1.0, 'logprobs': True, 'top_logprobs': 20,
                    'chat_template_kwargs': {'enable_thinking': False},
                    'messages': [{'role': 'system', 'content': 'Evaluate the state using the question and labeled options '
                                  'that follow. Return only the option code. Do not explain or reason aloud.'},
                                 {'role': 'user', 'content': request['state'] + '\n\n' + question}]}
            started = time.time()
            row = {'id': case['id']}
            try:
                reply = post('/v1/chat/completions', body, timeout=120)
                top = reply['choices'][0]['logprobs']['content'][0]['top_logprobs']
                scores = {}
                for entry in top:
                    code = entry['token'].strip()
                    if code in codes:
                        scores[code] = math.log(math.exp(scores[code]) + math.exp(entry['logprob'])) if code in scores \
                            else entry['logprob']
                peak = max(scores.values())
                weights = {k: (math.exp(scores[c] - peak) if c in scores else 0.0) for c, k in zip(codes, keys)}
                total = sum(weights.values())
                probabilities = {k: w / total for k, w in weights.items()}
                row.update(status='OBSERVED', answer={'choice': max(probabilities, key=probabilities.get),
                           'probabilities': probabilities}, missing=[c for c in codes if c not in scores],
                           input_tokens=(reply.get('usage') or {}).get('prompt_tokens'))
            except Exception as exc:  # recorded, not fatal
                row.update(status='FAILED', error=f'{type(exc).__name__}: {exc}'[:200])
            row['seconds'] = round(time.time() - started, 3)
            rows.append(row)
        results[name] = rows
    return results


def main():
    while idle_seconds() < 1800:
        report['waiting'] = f'desktop idle {idle_seconds()} s at {time.strftime("%H:%M:%S")}'
        save()
        time.sleep(60)
    report.pop('waiting', None)
    lease = SharedGpuLease(Path.home()/'.local/state/tare-qualified-models/gpu.lock', timeout=1800)
    with lease.hold('image', 'ninfer-ab') as receipt:
        yield_text_backend('http://127.0.0.1:8080', receipt['nonce'])
        report['gpu_used_after_yield_mib'] = gpu_used_mib()
        save()
        protocols = {p.stem: json.loads(p.read_text()) for p in sorted(HERE.glob('LAYA_*_PROTOCOL_*.json'))}
        for name, flags in CONFIGS:
            entry = report['configs'][name] = {'flags': flags}
            save()
            with open(HERE/f'{name}.log', 'wb') as log:
                started = time.time()
                process, failure = start(flags, log)
                entry['load_s'] = round(time.time() - started, 1)
                if failure:
                    entry['failure'] = failure
                    stop(process)
                    save()
                    continue
                entry['gpu_used_mib'] = gpu_used_mib()
                try:
                    entry.update(workload(name))
                    if name == CONFIGS[-2][0] or name == CONFIGS[-1][0]:
                        entry['litjev'] = litjev(protocols)
                except Exception as exc:
                    entry['failure'] = f'{type(exc).__name__}: {exc}'[:300]
                finally:
                    stop(process)
                save()
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    save()


if __name__ == '__main__':
    main()
