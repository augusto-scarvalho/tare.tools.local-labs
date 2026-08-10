"""Testable core of the code-benchmark harness — the SINGLE SOURCE OF TRUTH for the glue steps
that have each produced a real scoring incident, plus a samples validator and dataset/benchmark
identity (LAB-QA-001 / LAB-QA-002).

Design rule (Backlog V2 §25): do not duplicate mechanisms. `a2_concision_bench.py` and
`score_subset.py` IMPORT these functions instead of open-coding them, so the self-test in
`tests/benchmark_harness/` exercises the *same* code the real runs use.

stdlib-only on purpose: `score_subset.py` runs inside the separate `/home/augus/evalplus-venv`
(WSL) and `a2_concision_bench.py` runs on Windows python; a sibling stdlib module imports cleanly
in both. NO GPU, no heavy deps.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

# --------------------------------------------------------------------------------------------
# Incident-hardened glue (each carries the incident that motivated it)
# --------------------------------------------------------------------------------------------

def assemble_humaneval_solution(prompt: str, completion: str) -> str:
    """Build a SELF-CONTAINED evalplus `solution` from a HumanEval prompt + a model completion.

    INCIDENT (2026-08-10, commit 81eed6d): the harness stored `solution = completion` alone.
    evalplus does NOT prepend the prompt, so the solution must be self-contained (imports +
    prompt-provided helpers + the target fn). A CONCISE model correctly *continues* the prompt —
    returning only the entry function and reusing e.g. HumanEval/10's prompt-provided
    `is_palindrome` — so its bare completion scored 0 (NameError). This zeroed the ThinkingCap
    models (0/60) and inverted a leaderboard. Prepending the prompt fixes both concise and
    verbose styles and is verified NEUTRAL for already-self-contained models.
    """
    return prompt + "\n" + completion


def pad_subset(mine: dict[str, str], all_ids: list[str]) -> list[dict]:
    """Pad a subset of {task_id: solution} up to the full benchmark id list; missing ids get an
    empty (guaranteed-failing) solution — evalplus insists on the full set."""
    return [{"task_id": t, "solution": mine.get(t, "")} for t in all_ids]


def bust_stale_results(results_path) -> bool:
    """Delete a stale evalplus `<padded>_eval_results.json` before re-scoring. Returns True if a
    file was removed.

    INCIDENT (2026-08-10, commit 81eed6d): evalplus REUSES an existing `_eval_results.json`
    instead of re-evaluating, so re-scoring a corrected samples file under the same padded name
    silently returned the STALE verdicts (the first fixed re-run still read 0/60). Always bust it.
    """
    p = pathlib.Path(results_path)
    if p.exists():
        p.unlink()
        return True
    return False


# --------------------------------------------------------------------------------------------
# Samples validation — a scorer must never silently ingest a malformed/incomplete samples file
# --------------------------------------------------------------------------------------------

def parse_jsonl_strict(text: str) -> tuple[list[dict], list[dict]]:
    """Parse JSONL, returning (records, errors). Each error = {line, reason, raw}. Detects
    malformed lines and non-object records rather than silently dropping them."""
    records, errors = [], []
    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            errors.append({"line": i, "reason": f"malformed-json: {e.msg}", "raw": s[:80]})
            continue
        if not isinstance(obj, dict):
            errors.append({"line": i, "reason": "not-an-object", "raw": s[:80]})
            continue
        records.append(obj)
    return records, errors


def validate_samples(records: list[dict], expected_ids, *, require_solution: bool = True) -> list[dict]:
    """Return a list of detected problems (empty = clean). Covers the LAB-QA-001 integrity cases:
    duplicate / missing / extra / unknown (wrong) task_id, missing task_id field, empty solution.
    `expected_ids` = the canonical id set for the benchmark subset (order-independent)."""
    expected = set(expected_ids)
    problems: list[dict] = []
    seen: dict[str, int] = {}
    present: set[str] = set()

    for idx, r in enumerate(records):
        tid = r.get("task_id")
        if tid is None:
            problems.append({"kind": "missing-task_id-field", "index": idx})
            continue
        seen[tid] = seen.get(tid, 0) + 1
        present.add(tid)
        if tid not in expected:
            problems.append({"kind": "unknown-task_id", "task_id": tid})
        if require_solution and not str(r.get("solution", "")).strip():
            problems.append({"kind": "empty-solution", "task_id": tid})

    for tid, c in seen.items():
        if c > 1:
            problems.append({"kind": "duplicate-task_id", "task_id": tid, "count": c})
    for tid in expected - present:
        problems.append({"kind": "missing-sample", "task_id": tid})
    for tid in present - expected:
        # already reported as unknown-task_id above; keep as extra-sample for the count-view
        problems.append({"kind": "extra-sample", "task_id": tid})
    return problems


def flag_truncated(records: list[dict], *, max_tokens: int) -> list[str]:
    """Return task_ids whose generation looks TRUNCATED: server finish_reason == 'length', or the
    answer token count reached the cap (a non-terminating/over-length generation). Used to keep a
    truncated completion from being scored as a genuine wrong answer."""
    out = []
    for r in records:
        fr = str(r.get("finish_reason") or "")
        at = r.get("answer_tokens")
        if fr == "length" or (isinstance(at, int) and max_tokens and at >= max_tokens):
            if r.get("task_id"):
                out.append(r["task_id"])
    return out


# --------------------------------------------------------------------------------------------
# Dataset / benchmark identity (LAB-QA-002)
# --------------------------------------------------------------------------------------------

def dataset_hash(problems: list[dict]) -> str:
    """Stable content hash of a problem set: sha256 over sorted (task_id, prompt) pairs. Two runs
    that claim the same dataset MUST produce the same hash, so a swapped/edited dataset is caught."""
    h = hashlib.sha256()
    for p in sorted(problems, key=lambda d: str(d.get("task_id"))):
        h.update(str(p.get("task_id")).encode()); h.update(b"\x00")
        h.update(str(p.get("prompt", "")).encode()); h.update(b"\x00")
    return h.hexdigest()


def check_identity(actual: dict, expected: dict) -> list[dict]:
    """Compare a run's recorded identity block against an expected one; return mismatches. Catches
    wrong benchmark_version / dataset_hash / scorer_commit before a score is trusted."""
    out = []
    for k, v in expected.items():
        if actual.get(k) != v:
            out.append({"field": k, "expected": v, "actual": actual.get(k)})
    return out
