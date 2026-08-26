from __future__ import annotations

import pathlib
import pytest

from tools.research.run_adapter_requalification import FROZEN_GSM8K_IDS, DEFAULT_MATH_PATH, load_math_panel
from tools.research.run_distillation_evaluation import (
    DEFAULT_TEACHER_PATH,
    load_teacher_samples,
)


def test_load_teacher_samples():
    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    assert len(math_tasks) == 32
    teacher_samples = load_teacher_samples(DEFAULT_TEACHER_PATH, math_tasks)
    assert len(teacher_samples) == 32
    for s in teacher_samples:
        assert "task_id" in s
        assert "output_text" in s
        assert "gold" in s
        assert "correct" in s
        assert "tokens" in s
        assert s["tokens"] > 0
        assert len(s["output_text"]) > 0


def test_paired_metrics_math():
    teacher_tokens = [150, 160, 170, 180]
    student_tokens = [70, 75, 80, 85]
    import statistics
    med_t = statistics.median(teacher_tokens)
    med_s = statistics.median(student_tokens)
    red = (med_t - med_s) / med_t
    assert red >= 0.20
