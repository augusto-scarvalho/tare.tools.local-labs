from tools.research import run_qwen38_kv_precision_r2 as r2


def test_q8_abba_is_frozen():
    assert [row["cache"] for row in r2.BLOCKS] == ["f16", "q8_0", "q8_0", "f16"]
    assert [row["pair"] for row in r2.BLOCKS] == [0, 0, 1, 1]


def test_repeat_parity_separates_arm_stability():
    rows = []
    for pair in (0, 1):
        rows.extend({"arm": "f16", "pair": pair, "task_id": str(i), "extracted": str(i)} for i in range(32))
        rows.extend({"arm": "q4", "pair": pair, "task_id": str(i), "extracted": str(i)} for i in range(32))
    assert r2.repeat_parity(rows, "f16") == 1.0
    assert r2.repeat_parity(rows, "q4") == 1.0


def test_q8_uses_legacy_internal_arm_name_only_for_shared_aggregation():
    assert {row["cache"] for row in r2.BLOCKS} == {"f16", "q8_0"}
    assert [row["arm"] for row in r2.BLOCKS] == ["f16", "q4", "q4", "f16"]
