import json
from pathlib import Path

from tools.analysis.qualify_choice_helper import evaluate


def test_smoke_never_promotes_and_records_all_failures():
    cases = json.loads((Path(__file__).parents[1] / 'config/choice_helper_smoke.json').read_text())
    calls = []
    def assess(request):
        calls.append(request)
        if len(calls) == 2:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        return {'choice': 'bounded'}
    result = evaluate(cases, assess)
    assert len(calls) == 3 and result['expected_matches'] == 1
    assert result['decision'] == 'DO_NOT_ENABLE_AUTOMATIC_ROUTING'
    assert result['qualification'] == 'UNQUALIFIED'
    assert result['cases'][1]['reason'] == 'ASSESSMENT_INPUT_OVERFLOW'


def test_three_matches_are_not_semantic_qualification():
    cases = json.loads((Path(__file__).parents[1] / 'config/choice_helper_smoke.json').read_text())
    choices = iter(case['expected'] for case in cases['cases'])
    result = evaluate(cases, lambda _: {'choice': next(choices)})
    assert result['decision'] == 'READY_FOR_BOUNDED_PILOT'
    assert result['qualification'] == 'UNQUALIFIED'
