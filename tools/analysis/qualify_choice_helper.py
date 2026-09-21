"""Three-call CPU adapter smoke; intentionally incapable of promoting a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def evaluate(cases, assess):
    rows = []
    for case in cases['cases']:
        request = {'state': case['state'], 'question': cases['question'], 'options': cases['options']}
        started = time.monotonic()
        try:
            answer = assess(request)
            row = {'status': 'OBSERVED', 'answer': answer,
                   'expected_match': answer['choice'] == case['expected']}
        except ValueError as exc:
            row = {'status': 'REFUSED', 'reason': str(exc), 'expected_match': False}
        rows.append({'case': case['id'], 'expected': case['expected'], 'request': request,
                     'elapsed_ms': round((time.monotonic() - started) * 1000, 3), **row})
    return {'schema': 'tare.local-labs/choice-helper-smoke-result/1', 'authority': 'NONE',
            'qualification': 'UNQUALIFIED', 'case_count': len(rows),
            'expected_matches': sum(row['expected_match'] for row in rows), 'cases': rows,
            'decision': 'READY_FOR_BOUNDED_PILOT' if all(row['expected_match'] for row in rows)
                        else 'DO_NOT_ENABLE_AUTOMATIC_ROUTING',
            'limits': 'Three authored smoke cases; no held-out quality, counterfactual executor performance or savings measured.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=Path, default=Path(__file__).resolve().parents[2] / 'config/choice_helper_smoke.json')
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    cases = json.loads(raw)
    if cases.get('schema') != 'tare.local-labs/choice-helper-smoke/1' or len(cases['cases']) != 3:
        parser.error('Exactly three preregistered smoke cases are required.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # An existing result or unfinished intent cannot be overwritten by a retry.
    with args.output.open('x', encoding='utf-8') as output:
        json.dump({'status': 'STARTED', 'cases_sha256': hashlib.sha256(raw).hexdigest()}, output)
    sys.path.insert(0, str(args.kernel_root.resolve()))
    from compute_plane.laya_choice_worker import LayaChoiceModel
    started = time.monotonic()
    model = LayaChoiceModel(args.checkpoint)
    load_ms = round((time.monotonic() - started) * 1000, 3)
    result = evaluate(cases, model.assess)
    result.update(load_ms=load_ms, total_ms=round((time.monotonic() - started) * 1000, 3),
                  cases_sha256=hashlib.sha256(raw).hexdigest(), model_identity=model.identity)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('qualification', 'decision', 'expected_matches', 'case_count', 'load_ms', 'total_ms')}))
    return 0 if result['expected_matches'] == 3 else 1


if __name__ == '__main__':
    raise SystemExit(main())
