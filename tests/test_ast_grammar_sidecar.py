"""Tests for AST Grammar Sidecar (CTRL-01)."""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analysis.ast_grammar_sidecar import ASTGrammarSidecar, is_valid_json_prefix


class TestASTGrammarSidecar(unittest.TestCase):
    def test_json_prefix_validation(self):
        self.assertTrue(is_valid_json_prefix('{"status": "ok"'))
        self.assertTrue(is_valid_json_prefix('{"count": 42, "items": ['))
        self.assertFalse(is_valid_json_prefix('{, "invalid": 1}'))
        self.assertFalse(is_valid_json_prefix('random text not json'))

    def test_sidecar_sanitization_accepts_clean_stream(self):
        sidecar = ASTGrammarSidecar(mode="json")
        tokens = ["{", '"name"', ":", ' "test"', ",", ' "id"', ":", " 100", "}"]
        clean_text, intercepted, us_per_tok = sidecar.sanitize_generation("", tokens)

        self.assertEqual(intercepted, 0)
        parsed = json.loads(clean_text)
        self.assertEqual(parsed["name"], "test")
        self.assertEqual(parsed["id"], 100)
        self.assertLess(us_per_tok, 500.0)

    def test_sidecar_sanitization_blocks_illegal_tokens(self):
        sidecar = ASTGrammarSidecar(mode="json")
        # Stream containing invalid tokens injected
        tokens = ["{", '"name"', ":", ' "test"', ",", "ILLEGAL_SYNTAX", ' "id"', ":", " 100", "}"]
        clean_text, intercepted, _ = sidecar.sanitize_generation("", tokens)

        self.assertGreater(intercepted, 0)
        self.assertNotIn("ILLEGAL_SYNTAX", clean_text)


if __name__ == "__main__":
    unittest.main()
