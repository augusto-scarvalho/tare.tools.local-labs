from __future__ import annotations

from tools.analysis.ast_grammar_sidecar import ASTGrammarSidecar
from tools.research.run_ctrl01_real_token import replay


def test_reproduced_nested_object_false_block() -> None:
    document = '{"meta":{"enabled":true,"note":null}}'
    outcome = replay(ASTGrammarSidecar("json"), list(document))
    assert outcome["intercepted"] > 0
    assert outcome["filtered"] != document


def test_reproduced_nested_array_false_block() -> None:
    document = '[{"id":1,"active":true},{"id":2,"active":false}]'
    outcome = replay(ASTGrammarSidecar("json"), list(document))
    assert outcome["intercepted"] > 0
    assert outcome["filtered"] != document
