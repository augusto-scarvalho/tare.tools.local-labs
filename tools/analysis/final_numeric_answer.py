#!/usr/bin/env python3
"""Generic, deterministic extraction of an explicitly concluded numeric answer."""
from __future__ import annotations

import re
from dataclasses import dataclass


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
NUMBER_RE = re.compile(NUMBER)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(" + NUMBER + r")\s*\}", re.I)
HASH_RE = re.compile(r"####\s*(" + NUMBER + r")", re.I)
EQUALITY_RE = re.compile(r"=\s*(" + NUMBER + r")(?![\d.])", re.I)
CUE_RE = re.compile(
    r"\b(?:final\s+answer|answer\s+is|answer\s*:|therefore|thus|hence|consequently)\b|\bso\b\s*,?",
    re.I,
)


@dataclass(frozen=True)
class Extraction:
    value: str | None
    method: str


def _stem(token: str) -> str:
    token = token.lower().strip(" .,:;!?()[]{}")
    if token in {"feet"}:
        return "foot"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _question_target(question: str) -> tuple[set[str], str | None]:
    lower = question.lower()
    if re.search(r"how much|cost|spend|charge|pay|price|money|dollar", lower):
        mode = "currency"
    elif re.search(r"how long|how many (?:additional )?(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)", lower):
        mode = "time"
    elif re.search(r"how far|how many (?:additional )?(?:inches?|feet|miles?|meters?|kilometers?)", lower):
        mode = "distance"
    elif "how old" in lower:
        mode = "age"
    else:
        mode = None
    terms: set[str] = set()
    matches = list(re.finditer(r"how (?:many|much)\s+([^?]+)", lower))
    if matches:
        words = re.findall(r"[a-z]+", matches[-1].group(1))
        skip = {"additional", "more", "fewer", "total", "much", "many", "does", "do", "did"}
        for word in words[:5]:
            stem = _stem(word)
            if stem not in skip:
                terms.add(stem)
                break
    return terms, mode


def extract_final_numeric_for_question(question: str, text: str) -> Extraction:
    """Resolve answer candidates using the requested quantity/unit, never the gold."""
    explicit = extract_final_numeric(text)
    if explicit.method in {"boxed", "hash_answer"}:
        return explicit
    terms, mode = _question_target(question)
    if not terms and mode is None:
        return explicit
    candidates = list(NUMBER_RE.finditer(text))
    if not candidates:
        return Extraction(None, "no_number")
    last_cue = max((match.end() for match in CUE_RE.finditer(text)), default=-1)
    terminal_start = max(text.rfind("\n"), text.rfind(". "), text.rfind("? "), text.rfind("! "))
    time_terms = {"second", "minute", "hour", "day", "week", "month", "year"}
    distance_terms = {"inch", "foot", "mile", "meter", "kilometer"}

    def score(match: re.Match[str]) -> tuple[float, int]:
        start, end = match.span()
        before = text[max(0, start - 4):start]
        after = text[end:end + 42]
        context_terms = {_stem(word) for word in re.findall(r"[A-Za-z]+", after)}
        value = _clean(match.group(0))
        points = start / max(1, len(text))
        if terms & context_terms:
            points += 50
        if mode == "currency" and ("$" in before or {"dollar", "cent"} & context_terms):
            points += 50
        if mode == "time" and time_terms & context_terms:
            points += 50
        if mode == "distance" and distance_terms & context_terms:
            points += 50
        if mode == "age" and "year" in context_terms:
            points += 50
        if start >= terminal_start:
            points += 8
        if start >= last_cue:
            points += 4
        if re.search(r"=\s*$", before):
            points += 3
        return points, start

    best = max(candidates, key=score)
    return Extraction(_clean(best.group(0)), "question_unit_ranked")


def _clean(value: str) -> str:
    return value.replace(",", "")


def extract_final_numeric(text: str) -> Extraction:
    """Prefer explicit conclusions and equations before last-number fallback."""
    for pattern, method in ((BOXED_RE, "boxed"), (HASH_RE, "hash_answer")):
        matches = list(pattern.finditer(text))
        if matches:
            return Extraction(_clean(matches[-1].group(1)), method)

    cues = list(CUE_RE.finditer(text))
    for cue in reversed(cues):
        tail = text[cue.end(): cue.end() + 180]
        equalities = list(EQUALITY_RE.finditer(tail))
        if equalities:
            return Extraction(_clean(equalities[-1].group(1)), "conclusion_equation_rhs")
        number = NUMBER_RE.search(tail)
        if number:
            return Extraction(_clean(number.group(0)), "conclusion_cue")

    terminal_segments = [segment for segment in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+", text) if segment.strip()]
    for segment in reversed(terminal_segments):
        numbers = list(NUMBER_RE.finditer(segment))
        if numbers:
            return Extraction(_clean(numbers[-1].group(0)), "terminal_segment")

    equalities = list(EQUALITY_RE.finditer(text))
    if equalities:
        return Extraction(_clean(equalities[-1].group(1)), "equation_rhs")

    numbers = list(NUMBER_RE.finditer(text))
    if numbers:
        return Extraction(_clean(numbers[-1].group(0)), "last_number_fallback")
    return Extraction(None, "no_number")
