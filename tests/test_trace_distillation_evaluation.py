from __future__ import annotations

import pathlib
import pytest

from tools.research.run_adapter_requalification import FROZEN_GSM8K_IDS, DEFAULT_MATH_PATH, load_math_panel
from tools.research.run_distillation_evaluation import (
    DEFAULT_TEACHER_PATH,
    load_teacher_samples,
)
from tools.research.run_trace_distillation_evaluation import (
    FINALIST_ADAPTER_PATH,
    FALLBACK_FINALIST_PATH,
)


def test_finalist_adapter_exists():
    exists = FINALIST_ADAPTER_PATH.exists() or FALLBACK_FINALIST_PATH.exists()
    assert exists is True


def test_trace_teacher_samples_loading():
    math_tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    assert len(math_tasks) == 32
    teacher_samples = load_teacher_samples(DEFAULT_TEACHER_PATH, math_tasks)
    assert len(teacher_samples) == 32
    for s in teacher_samples:
        assert "task_id" in s
        assert "output_text" in s
        assert "correct" in s
        assert s["correct"] is True
