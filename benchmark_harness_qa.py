"""Backward-compatibility module for benchmark_harness_qa.
Canonical implementation lives in src/model_lifecycle/analysis/benchmark_qa.py.
"""
from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model_lifecycle.analysis.benchmark_qa import *  # noqa: F401, F403
from model_lifecycle.analysis.benchmark_qa import (  # noqa: F401
    check_comparable,
    assemble_humaneval_solution,
    validate_samples,
    flag_truncated,
    parse_jsonl_strict,
    bust_stale_results,
    check_identity,
    dataset_hash,
    run_identity,
    COMPARISON_INVALIDATING,
    COMPARISON_ADVISORY,
)
