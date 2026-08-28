from __future__ import annotations

from tools.research import run_trace_vs_behavioral_finalist as experiment


def test_frozen_sources_and_imported_trace_are_complete_and_rescore():
    assert len(experiment.verify_sources()) == 14
    first, second = experiment.r8.panel_ids()
    qa_ids = experiment.r6.actual_qa_ids()
    trace = experiment.load_trace(first, second, qa_ids)
    assert len(trace) == 7
    assert sum(len(row["math_samples"]) + len(row["qa_samples"]) for row in trace) == 3920
    assert experiment.rescore(trace, {"panel_1": first, "panel_2": second}, qa_ids) is True


def test_hierarchical_bootstrap_preserves_clear_family_order():
    ids = [f"task/{index}" for index in range(256)]
    panels = {"panel_1": ids, "panel_2": ids}

    def family(name: str, count: int, correct: bool):
        return [
            {"family": name, "seed": seed, "math_samples": [
                {"panel": panel, "task_id": task_id, "correct": correct}
                for panel in panels for task_id in ids
            ], "qa_samples": []}
            for seed in range(count)
        ]

    interval = experiment.bootstrap(
        family("full_trace", 7, True), family("behavioral_finalist", 2, False),
        panels, replicates=100,
    )
    assert interval["lower_95"] == 1.0
    assert interval["upper_95"] == 1.0
