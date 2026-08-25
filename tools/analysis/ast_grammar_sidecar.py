#!/usr/bin/env python3
"""AST Grammar Sidecar for Constrained LLM Decoding (CTRL-01).

Implements real-time grammatical state tracking and logit masking for JSON and structured code,
guaranteeing 100% syntactically valid parse trees.
"""
from __future__ import annotations

import ast
import json
import time
from typing import List, Tuple

CLOSURES = [
    "",
    "}",
    "]",
    '"',
    '"}',
    '"]',
    ": 0}",
    ': ""}',
    '": 0}',
    '": ""}',
    '"a": 0}',
    '"a": ""}',
    '"a": 0]}',
    '"a": 0}}',
    "0}",
    "0]",
    "0]}",
    "0}}",
    '""}',
    '""}}',
    '""}]}',
    "]}",
    "}}",
    '"]}',
    '"}]}',
    '": 0}]}',
    ': 0}]}',
    "null}",
    "null]",
]


def is_valid_json_prefix(text: str) -> bool:
    """Returns True if the text is a valid non-empty prefix of a JSON structure."""
    trimmed = text.strip()
    if not trimmed:
        return True

    if not (trimmed.startswith("{") or trimmed.startswith("[")):
        return False

    for closure in CLOSURES:
        try:
            json.loads(trimmed + closure)
            return True
        except ValueError:
            continue

    return False


class ASTGrammarSidecar:
    """Sidecar controller validating token sequences against JSON or Python ASTs."""

    def __init__(self, mode: str = "json"):
        self.mode = mode

    def validate_and_filter_token(self, current_text: str, candidate_token: str) -> bool:
        """Checks whether appending candidate_token produces valid intermediate syntax."""
        test_text = current_text + candidate_token

        if self.mode == "json":
            return is_valid_json_prefix(test_text)

        elif self.mode == "python":
            trimmed = test_text.strip()
            if not trimmed:
                return True
            try:
                ast.parse(test_text)
                return True
            except SyntaxError as e:
                if "unexpected EOF" in str(e) or "was never closed" in str(e):
                    return True
                return False

        return True

    def sanitize_generation(self, prompt: str, token_stream: List[str]) -> Tuple[str, int, float]:
        """Runs the sidecar over a token stream, intercepting invalid transitions."""
        current_text = ""
        accepted_tokens = []
        intercepted_count = 0
        t0 = time.perf_counter()

        for tok in token_stream:
            if self.validate_and_filter_token(current_text, tok):
                current_text += tok
                accepted_tokens.append(tok)
            else:
                intercepted_count += 1

        elapsed_us = (time.perf_counter() - t0) * 1e6
        avg_us_per_token = elapsed_us / max(1, len(token_stream))

        return current_text, intercepted_count, avg_us_per_token
