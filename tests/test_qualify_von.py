from tools.analysis.qualify_von import evaluate


def test_refusal_does_not_count_as_correct_abstention_or_order_stability():
    cases = [{'id': f'missing-{i}', 'suite': 'boundary', 'request': {},
              'expected_identity': 'insufficient', 'choice_identity': {}} for i in range(2)]
    def refuse(request):
        raise ValueError('ASSESSMENT_CAPACITY_EXCEEDED')
    result = evaluate({'cases': cases}, refuse)
    assert result['correct'] == 0
    assert result['order_stable_groups'] == 0
    assert result['qualification'] == 'UNQUALIFIED'


def test_order_stability_compares_semantics_not_positional_keys():
    cases = [{'id': f'cost-{i}', 'suite': 'strategy', 'request': {},
              'expected_identity': 'cheap', 'choice_identity': {'first': identity}}
             for i, identity in enumerate(['cheap', 'expensive'])]
    result = evaluate({'cases': cases}, lambda request: {'choice': 'first'})
    assert result['correct'] == 1
    assert result['order_stable_groups'] == 0
    assert result['authority'] == 'NONE'
