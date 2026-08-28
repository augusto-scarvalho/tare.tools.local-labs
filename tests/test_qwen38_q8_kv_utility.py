from tools.research import run_qwen38_q8_kv_utility as target


def test_panel_is_frozen_broad_and_disjoint():
    rows = target.panel()
    assert len(rows) == 128
    assert rows[0]["task_id"] == "gsm8k/32"
    assert rows[-1]["task_id"] == "gsm8k/159"
    assert target.canonical_json_sha256([row["task_id"] for row in rows]) == target.PANEL_HASH


def test_paired_bootstrap_detects_equal_correctness():
    rows = []
    for arm in ("f16", "q8"):
        rows.extend({"arm": arm, "task_id": str(index), "correct": index % 2 == 0}
                    for index in range(128))
    result = target.paired_bootstrap(rows)
    assert result["point"] == 0
    assert result["lower_95"] == 0
    assert result["upper_95"] == 0


def test_treatment_changes_only_cache_precision():
    assert [row["cache"] for row in target.BLOCKS] == ["f16", "q8_0"]
    assert [row["arm"] for row in target.BLOCKS] == ["f16", "q8"]
