from __future__ import annotations

import json
import pathlib

from tools.analysis.final_numeric_answer import extract_final_numeric, extract_final_numeric_for_question


FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "final_numeric_answer_cases.json"


def test_external_answer_extractor_fixtures() -> None:
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert len(cases) >= 12
    for case in cases:
        extraction = (
            extract_final_numeric_for_question(case["question"], case["text"])
            if case.get("question") else extract_final_numeric(case["text"])
        )
        assert extraction.value == case["expected"], case["id"]
        assert extraction.method == case["method"], case["id"]


def test_explicit_conclusion_beats_later_context_number() -> None:
    extraction = extract_final_numeric(
        "Therefore, the answer is 42 widgets after considering inventory lot 9000."
    )
    assert extraction.value == "42"
    assert extraction.method == "conclusion_cue"


def test_question_unit_disambiguates_context_numbers() -> None:
    extraction = extract_final_numeric_for_question(
        "How many additional hours does the trip require?",
        "Therefore, the trip takes 7 hours to cover the remaining 2800 miles.",
    )
    assert extraction.value == "7"


def test_question_currency_disambiguates_duration() -> None:
    extraction = extract_final_numeric_for_question(
        "How much does the tutor charge?",
        "Therefore, the tutor charges $168 for 2 weeks.",
    )
    assert extraction.value == "168"
