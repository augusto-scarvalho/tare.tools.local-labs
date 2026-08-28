from __future__ import annotations

from tools.research import run_fleet_humanevalplus_r2 as experiment


def test_r1_qwen_import_is_complete_ordered_and_immutable():
    panel = experiment.r1.load_panel()
    records, samples = experiment.verify_import(panel)
    assert len(records) == len(samples) == 164
    assert [row["task_id"] for row in records] == [row["task_id"] for row in panel]
    assert all(row["model"] == "qwen38" for row in records)


def test_score_command_bootstraps_repository_pythonpath():
    command = experiment.score_command(
        experiment.ROOT / "input.jsonl", experiment.ROOT / "scores.json"
    )
    assert "env" in command
    assignment = next(value for value in command if value.startswith("PYTHONPATH="))
    assert assignment == "PYTHONPATH=/mnt/c/projects/tare.tools.local-labs"
    assert command[-4] == experiment.r1.EVALPLUS_PYTHON
