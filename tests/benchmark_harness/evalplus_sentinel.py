#!/usr/bin/env python3
"""WA-CLOSE-002 — actual-EvalPlus INTEGRATION sentinel (NOT a unit test).

The 16-case unit suite (`benchmark_harness_selftest.py`) stays fast and emulated. This sentinel
additionally drives the REAL, currently-installed EvalPlus end-to-end on a tiny deterministic
fixture, so a broken/upgraded EvalPlus or a broken glue path is caught for real:

    known-good (prompt + canonical_solution)  --our glue--> real evalplus.evaluate --> PASS
    known-bad  (prompt + wrong body)          --our glue--> real evalplus.evaluate --> FAIL
    stale-cache boundary: good then bad without busting -> stale PASS leaks; with bust -> FAIL

No GPU. ~1-2 min (a few real evalplus runs over 1 task, the rest padded empty). Failure is loud.
Run in the EvalPlus venv:
    /home/augus/evalplus-venv/bin/python tests/benchmark_harness/evalplus_sentinel.py
Writes evalplus_sentinel_report.json (with the actual evalplus version + raw statuses) beside this.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import benchmark_harness_qa as q  # noqa: E402
import evalplus  # noqa: E402
from evalplus.data import get_human_eval_plus  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
TASK = "HumanEval/0"


def _plus_status(task_id: str, full_solution: str, workdir: pathlib.Path, *,
                 padded_name: str, bust: bool) -> tuple[str, str]:
    """Score ONE task's self-contained solution via the REAL evalplus. Returns (plus_status, raw)."""
    allp = list(get_human_eval_plus())
    padded = workdir / padded_name
    with padded.open("w") as f:
        for row in q.pad_subset({task_id: full_solution}, allp):   # our real glue
            f.write(json.dumps(row) + "\n")
    res_path = padded.with_name(padded.stem + "_eval_results.json")
    if bust:
        q.bust_stale_results(res_path)                              # our real glue
    p = subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
                        "--samples", str(padded)], capture_output=True, text=True)
    raw = (p.stdout or "") + (p.stderr or "")
    if not res_path.exists():
        return "NO_RESULTS", raw
    e = json.loads(res_path.read_text())["eval"].get(task_id)
    entry = e[0] if isinstance(e, list) and e else (e or {})
    return (entry.get("plus_status") or "MISSING"), raw


def main() -> int:
    data = get_human_eval_plus()
    t = data[TASK]
    good = t["prompt"] + t["canonical_solution"]               # known-good, self-contained
    bad = t["prompt"] + "    return None  # deliberately wrong\n"   # known-bad
    results, raws = {}, {}

    with tempfile.TemporaryDirectory() as d:
        wd = pathlib.Path(d)
        results["known_good"], raws["known_good"] = _plus_status(TASK, good, wd, padded_name="good.padded.jsonl", bust=True)
        results["known_bad"], raws["known_bad"] = _plus_status(TASK, bad, wd, padded_name="bad.padded.jsonl", bust=True)

        # Real stale-cache boundary on the SAME padded name: score good (pass), then bad.
        results["stale_good_first"], _ = _plus_status(TASK, good, wd, padded_name="stale.padded.jsonl", bust=True)
        # (a) re-score bad WITHOUT busting -> evalplus reuses the good result -> stale 'pass' leaks
        results["stale_bad_nobust"], _ = _plus_status(TASK, bad, wd, padded_name="stale.padded.jsonl", bust=False)
        # (b) re-score bad WITH busting -> current 'fail'
        results["stale_bad_bust"], _ = _plus_status(TASK, bad, wd, padded_name="stale.padded.jsonl", bust=True)

    checks = {
        "known_good_is_pass": results["known_good"] == "pass",
        "known_bad_is_fail": results["known_bad"] == "fail",
        "stale_leaks_without_bust": results["stale_bad_nobust"] == "pass",   # reproduces the bug
        "bust_fixes_stale": results["stale_bad_bust"] == "fail",             # verifies the fix
    }
    ok = all(checks.values())
    report = {"suite": "WA-CLOSE-002 evalplus_sentinel (integration/e2e)",
              "evalplus_version": getattr(evalplus, "__version__", "unknown"),
              "python": sys.executable, "task": TASK,
              "results": results, "checks": checks, "passed": ok,
              "raw_known_good_tail": raws["known_good"][-400:],
              "raw_known_bad_tail": raws["known_bad"][-400:]}
    (HERE / "evalplus_sentinel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in checks.items():
        print(f"  [{'ok  ' if v else 'FAIL'}] {k}  (results: {results})" if not v else f"  [ok  ] {k}")
    print(f"\nWA-CLOSE-002 evalplus sentinel (evalplus {report['evalplus_version']}): "
          + ("QUALIFIED — all checks pass" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
