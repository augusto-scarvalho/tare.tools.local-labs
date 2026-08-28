from tools.research.run_trace_distillation_confirmation_r6 import (
    FRESH_LABELS,
    IMPORTED_LABELS,
    actual_qa_ids,
    verify_sources,
)


def test_continuation_sources_bind_nine_complete_workers():
    static, workers = verify_sources()
    assert len(static) == 8
    assert sorted(workers) == sorted(IMPORTED_LABELS)
    assert len(workers) == 9


def test_actual_qa_panel_is_complete_and_partitioned():
    ids = actual_qa_ids()
    assert len(ids) == 48
    assert ids[:10] == [f"f{index:02d}" for index in range(1, 11)]
    assert len(ids[10:]) == 38
    assert len(IMPORTED_LABELS) + len(FRESH_LABELS) == 14
