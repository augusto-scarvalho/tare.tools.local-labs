from __future__ import annotations

import inspect
import json
import pathlib

from tools.analysis.final_numeric_answer_v2 import extract_concluded_numeric_for_question


FIXTURES = pathlib.Path(__file__).parent / "fixtures/final_numeric_answer_v2_cases.json"


def test_v2_frozen_adversarial_fixtures() -> None:
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert len(cases) >= 15
    for case in cases:
        extraction = extract_concluded_numeric_for_question(case["question"], case["text"])
        assert extraction.value == case["expected"], case["id"]
        assert extraction.method == case["method"], case["id"]


def test_v2_scorer_is_gold_blind_by_signature() -> None:
    assert list(inspect.signature(extract_concluded_numeric_for_question).parameters) == [
        "question",
        "text",
    ]


def test_v2_regressions_from_retained_experiment_outputs() -> None:
    root = pathlib.Path(__file__).parents[1]
    trace_path = root / "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    expected = {
        ("answer_only", "gsm8k/665"): "15",
        ("answer_only", "gsm8k/711"): "4",
        ("answer_only", "gsm8k/838"): "15",
        ("full_trace", "gsm8k/665"): "360",
        ("full_trace", "gsm8k/711"): "1",
        ("full_trace", "gsm8k/838"): "0",
    }
    for arm in trace:
        for row in arm["math_samples"]:
            key = (arm["arm"], row["task_id"])
            if key in expected:
                actual = extract_concluded_numeric_for_question(row["prompt"], row["output_text"])
                assert actual.value == expected[key], key

    workloads = [
        json.loads(line)
        for line in (root / "workloads/gsm8k.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = {row["task_id"]: row["prompt"] for row in workloads}
    q8_rows = [
        json.loads(line)
        for line in (root / "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    water_rows = [row for row in q8_rows if row["task_id"] == "gsm8k/111"]
    assert {row["arm"] for row in water_rows} == {"f16", "q8"}
    for row in water_rows:
        actual = extract_concluded_numeric_for_question(prompts[row["task_id"]], row["content"])
        assert actual.value is None, row["arm"]
