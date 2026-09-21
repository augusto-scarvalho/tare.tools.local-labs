"""Run pinned upstream Von constructors against a frozen input-contract audit."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['marker', 'nli'], required=True)
    for name in ('upstream-root', 'checkpoint', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    cases = protocol[args.backend + '_cases']
    if len(cases) != (40 if args.backend == 'marker' else 25):
        parser.error('Unexpected frozen case count')
    result = {'status': 'STARTED', 'backend': args.backend, 'authority': 'NONE',
              'protocol_sha256': hashlib.sha256(raw).hexdigest(), 'cases': []}
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out)

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                         allow_nan=False) + '\n', encoding='utf-8')

    started = time.perf_counter()
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TF='0', HF_HUB_OFFLINE='1',
                          TRANSFORMERS_OFFLINE='1', VON_DEVICE='cpu')
        sys.path.insert(0, str(args.upstream_root / 'src'))
        import torch
        from von.types import Choice
        torch.set_num_threads(2)
        result['versions'] = {name: importlib.metadata.version(name) for name in
                              ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy')}
        with contextlib.redirect_stdout(sys.stderr):
            if args.backend == 'marker':
                from von.backends.option_marker_backend import OptionMarkerBackend
                backend = OptionMarkerBackend(checkpoint_dir=str(args.checkpoint), device='cpu')
                model = backend._get_model()
                tok = model.tokenizer
            else:
                from von.backends.berta_backend import BertaBackend
                backend = BertaBackend(variant=str(args.checkpoint), device='cpu')
                model, tok = backend._get_model_and_tok()
        result['load_ms'] = round((time.perf_counter() - started) * 1000, 3)
        result['device'] = str(next(model.parameters()).device)
        result['temperature_metadata'] = backend._default_temp
        result['encoder_config'] = (model.encoder.config if args.backend == 'marker'
                                    else model.config).to_dict()
        save()
        for case in cases:
            request = case['request']
            row = {'id': case['id'], 'suite': case['suite'], 'expected': case['expected_identity']}
            q = Choice(instructions=request['question'], criteria=request['options'])
            if args.backend == 'marker':
                packed = model.pack_sequence(request['state'], q.instructions,
                    [value.strip() for value in q.criteria.values()])
                ids = tok(packed, truncation=False)['input_ids']
                row['input_ids'] = ids
                overflow = len(ids) > 2048
            else:
                encoded = tok([request['state']] * len(q.criteria),
                    [q.instructions + ' ' + value for value in q.criteria.values()],
                    padding=False, truncation=False)['input_ids']
                row['input_lengths'] = [len(ids) for ids in encoded]
                overflow = max(row['input_lengths']) > 512
            if overflow:
                row.update(status='REFUSED_TRUNCATION', correct=False)
            else:
                before = time.perf_counter()
                # Call the official public primitive with its default parameters.
                with contextlib.redirect_stdout(sys.stderr):
                    answer = backend.evaluate_choice('decision', request['state'], q)
                observed = case['choice_identity'].get(answer.choice, answer.choice)
                row.update(status='OBSERVED', observed=observed,
                           correct=observed == case['expected_identity'],
                           answer=answer.model_dump(),
                           elapsed_ms=round((time.perf_counter() - before) * 1000, 3))
            result['cases'].append(row)
            save()
            print(json.dumps({k: row.get(k) for k in ('id', 'suite', 'correct', 'elapsed_ms')}), flush=True)
        result['status'] = 'COMPLETED'
    except Exception as exc:
        result.update(status='FAILED', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        result['total_ms'] = round((time.perf_counter() - started) * 1000, 3)
        save()


if __name__ == '__main__':
    main()
