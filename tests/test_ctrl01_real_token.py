from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analysis.ast_grammar_sidecar import ASTGrammarSidecar
from tools.research.run_ctrl01_real_token import VALID_CONTROLS, json_valid, percentile, replay, stable_model_identity


def test_valid_control_corpus_is_json() -> None:
    assert all(json_valid(document) for document in VALID_CONTROLS)
    assert all(json.loads(document) is not None for document in VALID_CONTROLS)


def test_replay_accepts_valid_negative_sign() -> None:
    outcome = replay(ASTGrammarSidecar("json"), ['{"delta":', "-", "12", "}"])
    assert outcome["intercepted"] == 0
    assert outcome["filtered"] == '{"delta":-12}'


def test_percentile_is_observed_upper_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_stable_identity_ignores_created_timestamp() -> None:
    first = {"data": [{"id": "m", "created": 1, "meta": {"n_params": 7, "ftype": "q4"}}]}
    second = {"data": [{"id": "m", "created": 2, "meta": {"n_params": 7, "ftype": "q4"}}]}
    assert stable_model_identity(first) == stable_model_identity(second)
