from tools.analysis.kv_qualification_metrics import compare, jensen_shannon


def distribution(size=1024, winner=7):
    values = [-20.0] * size
    values[winner] = 0.0
    return values


def test_identical_full_distributions_have_zero_shift():
    logits = distribution()
    result = compare([{"id": "a/0", "log_probs": logits}],
                     [{"id": "a/0", "log_probs": list(logits)}])
    assert result["median_jensen_shannon"] == 0.0
    assert result["top1_agreement"] == 1.0


def test_changed_winner_is_detected():
    assert jensen_shannon(distribution(winner=7), distribution(winner=8)) > 0


def test_truncated_distribution_fails_closed():
    try:
        compare([{"id": "a/0", "log_probs": [0.0, -1.0]}],
                [{"id": "a/0", "log_probs": [0.0, -1.0]}])
    except ValueError as error:
        assert "top-k" in str(error)
    else:
        raise AssertionError("expected top-k rejection")
