from __future__ import annotations

import pathlib
import pytest

from tools.research.run_adapter_requalification import (
    ROOT,
    ADAPTER_SPECS,
    FROZEN_GSM8K_IDS,
    FROZEN_QA_IDS,
    build_artifact_ledger,
    build_dataset_ledger,
    extract_gsm8k_gold,
    extract_gsm8k_pred,
    grade_qa,
    is_gsm8k_correct,
    load_math_panel,
    load_qa_panel,
    DEFAULT_MATH_PATH,
    DEFAULT_QA_PATH,
)


def test_adapter_specs_count_and_identity():
    assert len(ADAPTER_SPECS) == 13
    assert len({spec["id"] for spec in ADAPTER_SPECS}) == 13


def test_build_artifact_ledger():
    ledger = build_artifact_ledger(ROOT)
    assert len(ledger) == 13
    for name, item in ledger.items():
        assert "config" in item
        assert "safetensors" in item
        assert item["config"]["bytes"] > 0
        assert len(item["config"]["sha256"]) == 64
        assert item["safetensors"]["bytes"] > 0
        assert len(item["safetensors"]["sha256"]) == 64


def test_build_dataset_ledger():
    ledger = build_dataset_ledger(ROOT)
    assert "math_panel" in ledger
    assert "qa_panel" in ledger
    assert ledger["math_panel"]["frozen_ids_count"] == 32
    assert ledger["qa_panel"]["frozen_ids_count"] == 16


def test_load_math_panel():
    tasks = load_math_panel(DEFAULT_MATH_PATH, FROZEN_GSM8K_IDS)
    assert len(tasks) == 32
    for t in tasks:
        assert "task_id" in t
        assert "prompt" in t
        assert "gold" in t
        assert t["gold"] != ""


def test_load_qa_panel():
    tasks = load_qa_panel(DEFAULT_QA_PATH, FROZEN_QA_IDS)
    assert len(tasks) == 16
    for q in tasks:
        assert "id" in q
        assert "category" in q
        assert "prompt" in q
        assert "grader" in q


def test_gsm8k_scoring_helpers():
    assert extract_gsm8k_gold("The answer is #### 42") == "42"
    assert extract_gsm8k_gold("#### -10") == "-10"
    assert extract_gsm8k_pred("First step: ... #### 150\nDone") == "150"
    assert extract_gsm8k_pred("The final answer is 30") == "30"
    assert is_gsm8k_correct("42", "42") is True
    assert is_gsm8k_correct("42.0", "42") is True
    assert is_gsm8k_correct("41", "42") is False
    assert is_gsm8k_correct(None, "42") is False


def test_qa_grading_helpers():
    task_exact = {"grader": "exact_any", "expected": ["Canberra"]}
    ok, _ = grade_qa(task_exact, "Canberra")
    assert ok is True
    ok, _ = grade_qa(task_exact, "Sydney")
    assert ok is False

    task_json = {"grader": "json_exact", "expected": {"cidade": "Recife", "temperatura": 28}}
    ok, _ = grade_qa(task_json, '{"cidade": "Recife", "temperatura": 28}')
    assert ok is True
    ok, _ = grade_qa(task_json, "invalid json")
    assert ok is False

    task_lines = {"grader": "lines_exact", "expected": ["ALFA", "BETA", "GAMA"]}
    ok, _ = grade_qa(task_lines, "ALFA\nBETA\nGAMA")
    assert ok is True

    task_contains = {"grader": "contains_all", "required": ["Paris"], "forbidden": ["sim, lyon"], "max_words": 12}
    ok, _ = grade_qa(task_contains, "A capital é Paris.")
    assert ok is True
    ok, _ = grade_qa(task_contains, "Sim, Lyon é a capital.")
    assert ok is False
