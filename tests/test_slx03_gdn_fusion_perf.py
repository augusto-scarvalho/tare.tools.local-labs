from tools.research import run_slx03_gdn_fusion_perf as subject


def test_crossover_is_balanced_and_pair_order_alternates():
    assert [block["arm"] for block in subject.BLOCKS] == [
        "off", "on", "on", "off", "off", "on",
        "on", "off", "off", "on", "on", "off",
    ]
    assert len(subject.BLOCKS) == 12
    assert {block["pair"] for block in subject.BLOCKS} == set(range(6))
    assert all(sum(block["arm"] == arm for block in subject.BLOCKS) == 6 for arm in ("off", "on"))


def test_each_pair_has_identical_frozen_prompt_rotation():
    for pair in range(6):
        blocks = [block for block in subject.BLOCKS if block["pair"] == pair]
        assert len(blocks) == 2
        assert blocks[0]["rotation"] == blocks[1]["rotation"]
    assert len(subject.PROMPTS) == 12
    assert len(set(subject.PROMPTS)) == 12


def test_hierarchical_bootstrap_preserves_clear_ratio():
    rows = [{"pair": pair, "ratio": 1.05} for pair in range(6) for _ in range(12)]
    result = subject.hierarchical_bootstrap(rows, "ratio")
    assert result["complete"] is True
    assert result["clusters"] == 6
    assert result["observations"] == 72
    assert result["replicates"] == 20_000
    assert abs(result["point"] - 1.05) < 1e-12
    assert abs(result["ci95_low"] - 1.05) < 1e-12
    assert abs(result["ci95_high"] - 1.05) < 1e-12


def test_hierarchical_bootstrap_fails_closed_on_missing_cluster_observation():
    rows = [{"pair": pair, "ratio": 1.05} for pair in range(6) for _ in range(12)]
    result = subject.hierarchical_bootstrap(rows[:-1], "ratio")
    assert result["complete"] is False
    assert result["ci95_low"] == 0.0
    assert result["replicates"] == 0


def test_gate_comparators_match_preregistration():
    assert subject.gate_pass("eq", 144, 144)
    assert subject.gate_pass("gt", 1.0001, 1.0)
    assert not subject.gate_pass("gt", 1.0, 1.0)
    assert subject.gate_pass("ge", 0.98, 0.98)
