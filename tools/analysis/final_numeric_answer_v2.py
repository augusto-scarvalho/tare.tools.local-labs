#!/usr/bin/env python3
"""Conservative, gold-blind extraction of a concluded numeric answer.

Unlike the historical scorer, this module has no generic last-number fallback.
A value must be attached to an explicit conclusion, a target-bearing equation,
or a terminal target-bearing statement. This intentionally prefers a false
negative over crediting an incidental number from an unfinished derivation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
NUMBER_RE = re.compile(NUMBER)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(" + NUMBER + r")\s*\}", re.I)
HASH_RE = re.compile(r"####\s*(" + NUMBER + r")", re.I)
EQUALITY_RE = re.compile(r"=\s*\$?\s*(" + NUMBER + r")(?!\d|\.\d)", re.I)
CUE_RE = re.compile(
    r"\b(?:final\s+answer|answer\s+is|answer\s*:|therefore|thus|hence|"
    r"consequently|so\s*,?)\b",
    re.I,
)
DIRECT_RE = re.compile(
    r"\b(?:answer|total|result|cost|price|charge|amount|difference|remaining|"
    r"left|earned|earns?|paid|spends?|has|have|needs?|requires?|takes?|is|are)\b",
    re.I,
)
CURRENCY_WORDS = {"dollar", "cent", "cost", "price", "charge", "pay", "paid", "money", "budget"}
TIME_WORDS = {"time", "second", "minute", "hour", "day", "week", "month", "year"}
DISTANCE_WORDS = {"inch", "foot", "mile", "meter", "kilometer", "yard"}
MEASURE_WORDS = {
    "liter", "milliliter", "gallon", "pound", "ounce", "gram", "kilogram",
    "calorie", "degree", "percent", "percentage", "dozen",
}
STOP_WORDS = {
    "additional", "more", "fewer", "many", "much", "total", "does", "do", "did",
    "is", "are", "was", "were", "will", "would", "should", "could", "can", "the", "a", "an",
}
AUXILIARIES = {
    "do", "does", "did", "is", "are", "was", "were", "has", "have", "had",
    "will", "would", "should", "can", "could", "need", "needs", "take", "takes",
}


@dataclass(frozen=True)
class Extraction:
    value: str | None
    method: str


def _clean(value: str) -> str:
    return value.replace(",", "")


def _stem(token: str) -> str:
    token = token.lower().strip(" .,:;!?()[]{}")
    if token == "feet":
        return "foot"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _question_target(question: str) -> tuple[set[str], str | None]:
    lower = question.lower()
    last_question = lower.rsplit("?", 1)[0].rsplit("\n", 1)[-1]
    words = [_stem(word) for word in re.findall(r"[a-z]+", last_question)]
    terms: set[str] = set()
    matches = list(re.finditer(r"how (?:many|much)\s+([^?]+)", last_question))
    if matches:
        phrase = [_stem(word) for word in re.findall(r"[a-z]+", matches[-1].group(1))]
        for word in phrase:
            if terms and word in AUXILIARIES:
                break
            if word not in STOP_WORDS:
                terms.add(word)
            if len(terms) >= 4:
                break
    mode: str | None = None
    # Infer the requested quantity from the final question clause, rather than
    # from contextual units elsewhere in the word problem.  For example,
    # "How much water ... at 1.25 feet/second?" asks for water, not time.
    if CURRENCY_WORDS.intersection(terms) or "$" in last_question or re.search(
        r"\b(?:cost|price|charge|pay|paid|money|budget)\b", last_question
    ):
        mode = "currency"
    elif TIME_WORDS.intersection(terms) or (not matches and "time" in words):
        mode = "time"
    elif DISTANCE_WORDS.intersection(terms):
        mode = "distance"
    if mode == "time":
        terms.update(TIME_WORDS.intersection(words))
    elif mode == "distance":
        terms.update(DISTANCE_WORDS.intersection(words))
    elif mode == "currency":
        terms.update(CURRENCY_WORDS.intersection(words))
    else:
        # Measurement nouns in the question are useful only when no stronger
        # time/distance/currency mode has been identified.
        terms.update(MEASURE_WORDS.intersection(words))
    return terms, mode


def _segments(text: str) -> list[tuple[int, int, str]]:
    boundaries = [0]
    for match in re.finditer(r"(?:\r?\n)+|(?<=[.!?])\s+", text):
        boundaries.extend((match.start(), match.end()))
    boundaries.append(len(text))
    rows: list[tuple[int, int, str]] = []
    for start, end in zip(boundaries[::2], boundaries[1::2]):
        segment = text[start:end].strip()
        if segment:
            actual_start = text.find(segment, start, end + 1)
            rows.append((actual_start, actual_start + len(segment), segment))
    if not rows and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        rows.append((start, start + len(stripped), stripped))
    return rows


def extract_concluded_numeric_for_question(question: str, text: str) -> Extraction:
    """Return a concluded numeric value using only question and generated text."""
    for pattern, method in ((BOXED_RE, "boxed"), (HASH_RE, "hash_answer")):
        matches = list(pattern.finditer(text))
        if matches:
            return Extraction(_clean(matches[-1].group(1)), method)

    terms, mode = _question_target(question)
    candidates: list[tuple[int, int, float, int, str, str]] = []
    segments = _segments(text)
    for segment_index, (segment_start, _segment_end, segment) in enumerate(segments):
        numbers = list(NUMBER_RE.finditer(segment))
        if not numbers:
            continue
        words = {_stem(word) for word in re.findall(r"[A-Za-z]+", segment)}
        cue_positions = [match.end() for match in CUE_RE.finditer(segment)]
        last_cue = max(cue_positions, default=-1)
        direct = bool(DIRECT_RE.search(segment))
        equality_values = {match.group(1).replace(",", "") for match in EQUALITY_RE.finditer(segment)}
        terminal = segment_index >= max(0, len(segments) - 2)
        for match in numbers:
            if re.search(r"\b(?:step|stage)\s*$", segment[:match.start()], re.I):
                continue
            value = _clean(match.group(0))
            before = segment[max(0, match.start() - 5):match.start()]
            after = segment[match.end():match.end() + 48]
            after_words = {_stem(word) for word in re.findall(r"[A-Za-z]+", after)}
            target_near = bool(terms.intersection(words) or terms.intersection(after_words))
            currency_near = "$" in before or bool({"dollar", "cent"}.intersection(after_words))
            mode_near = (
                currency_near if mode == "currency" else
                bool(TIME_WORDS.intersection(after_words)) if mode == "time" else
                bool(DISTANCE_WORDS.intersection(after_words)) if mode == "distance" else
                target_near
            )
            after_cue = last_cue >= 0 and match.start() >= last_cue
            equality_rhs = value in equality_values
            eligible = after_cue or (equality_rhs and (target_near or mode_near)) or (terminal and direct and (target_near or mode_near))
            if not eligible:
                continue
            score = 0.0
            method = "terminal_target"
            if after_cue:
                score += 120
                method = "conclusion"
            if equality_rhs:
                score += 70
                method = "target_equation" if not after_cue else method
            if target_near:
                score += 55
            if mode_near:
                score += 45
            if direct:
                score += 15
            if terminal:
                score += 12
            score += (segment_start + match.start()) / max(1, len(text))
            # Evidence class is ordered before recency and local score.  Within
            # an evidence class, the latest segment wins: a final "Thus, 15
            # slices remain" must displace an earlier "Therefore 15+15=30".
            # A discourse cue may close an intermediate subproblem ("Thus,
            # there are 60 teachers").  It is top-tier only when it names the
            # requested quantity; otherwise a later target equation wins.
            tier = 3 if after_cue and (target_near or mode_near) else 2 if equality_rhs else 1
            candidates.append((tier, segment_index, score, segment_start + match.start(), value, method))

    if not candidates:
        return Extraction(None, "no_concluded_answer")
    _, _, _, _, value, method = max(candidates)
    return Extraction(value, method)
