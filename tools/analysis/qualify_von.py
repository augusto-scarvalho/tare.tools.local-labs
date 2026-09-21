"""Frozen CPU Von screen. Records observations; never promotes production routing."""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def evaluate(protocol, assess):
    rows = []
    for case in protocol['cases']:
        started = time.monotonic()
        row = {'id': case['id'], 'suite': case['suite'],
               'expected': case['expected_identity'], 'correct': False}
        try:
            answer = assess(case['request'])
            identity = case['choice_identity'].get(answer['choice'], answer['choice'])
            row.update(status='OBSERVED', answer=answer, observed=identity,
                       correct=identity == case['expected_identity'])
        except (ValueError, RuntimeError) as exc:
            row.update(status='REFUSED', reason=str(exc))
        row['elapsed_ms'] = round((time.monotonic() - started) * 1000, 3)
        rows.append(row)
    groups = {}
    for row in rows:
        if row['suite'] != 'demand':
            groups.setdefault(row['id'].rsplit('-', 1)[0], []).append(row)
    stable = sum(len(g) == 2 and all(r['status'] == 'OBSERVED' for r in g)
                 and len({r['observed'] for r in g}) == 1 for g in groups.values())
    times = sorted(r['elapsed_ms'] for r in rows)
    return {'schema': 'tare.local-labs/von-screen-result/1', 'authority': 'NONE',
            'qualification': 'UNQUALIFIED', 'correct': sum(r['correct'] for r in rows),
            'count': len(rows), 'order_stable_groups': stable, 'order_group_count': len(groups),
            'median_ms': statistics.median(times), 'p95_ms': times[int((len(times)-1)*.95 + .999)],
            'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('kernel-root', 'manifest', 'protocol', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    protocol = json.loads(raw)
    if protocol.get('schema') != 'tare.local-labs/von-screen/1' or len(protocol['cases']) != 15:
        parser.error('Exactly 15 frozen cases required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'STARTED', 'protocol_sha256': hashlib.sha256(raw).hexdigest()}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f)
    started = time.monotonic()
    try:
        sys.path.insert(0, str(args.kernel_root.resolve()))
        from compute_plane.von_choice_worker import VonChoiceModel
        with contextlib.redirect_stdout(sys.stderr):
            model = VonChoiceModel(args.manifest)
            load_ms = round((time.monotonic() - started) * 1000, 3)
            result.update(evaluate(protocol, model.assess), status='COMPLETED', load_ms=load_ms,
                          model_identity=model.identity)
    except Exception as exc:
        result.update(status='FAILED', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        result['total_ms'] = round((time.monotonic() - started) * 1000, 3)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'model_identity')}))


if __name__ == '__main__':
    main()
