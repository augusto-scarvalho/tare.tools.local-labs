from __future__ import annotations

import pathlib
import pytest

from tools.research.run_adapter_training import (
    DEFAULT_MATH_PATH,
    DEFAULT_TEACHER_PATH,
    FROZEN_GSM8K_IDS,
    SEEDS,
    load_training_pairs,
)


def test_seeds_configuration():
    assert len(SEEDS) >= 2
    assert len(set(SEEDS)) == len(SEEDS)


def test_load_training_pairs():
    pairs = load_training_pairs(DEFAULT_TEACHER_PATH, DEFAULT_MATH_PATH, seed=20260824)
    assert len(pairs) == 128
    for p in pairs:
        assert "task_id" in p
        assert "prompt" in p
        assert "completion" in p
        assert p["task_id"] not in FROZEN_GSM8K_IDS
        assert "#### <answer>" in p["prompt"]
        assert len(p["completion"].strip()) > 0


def test_training_pairs_deterministic_per_seed():
    pairs1 = load_training_pairs(DEFAULT_TEACHER_PATH, DEFAULT_MATH_PATH, seed=20260824)
    pairs2 = load_training_pairs(DEFAULT_TEACHER_PATH, DEFAULT_MATH_PATH, seed=20260824)
    assert [p["task_id"] for p in pairs1] == [p["task_id"] for p in pairs2]

    pairs3 = load_training_pairs(DEFAULT_TEACHER_PATH, DEFAULT_MATH_PATH, seed=20260825)
    assert [p["task_id"] for p in pairs1] != [p["task_id"] for p in pairs3]
