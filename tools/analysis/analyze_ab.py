"""Settle every A/B this project ran, with the project's own robust machinery — reading
records straight off disk (`runs/ab-*/records.json`).

The pairing math and the editorial PLAN of which comparisons to make now live in
`model_lifecycle.reports.ab`, shared with the Store-backed report generator so the two can
never disagree about a number. This script is the DISK reader: it loads each group's
records.json and renders the same blob as a human report.

The noise floor from ab-null is printed FIRST and every other delta is read against it.
A delta whose median sits at or below the floor is flagged and is not evidence.

    python analyze_ab.py                 # human report
    python analyze_ab.py --json          # machine-readable, for STATUS.md provenance
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.reports.ab import analyse                       # noqa: E402

RUNS = pathlib.Path(__file__).parent / "runs"


def load(name: str) -> list[dict]:
    f = RUNS / name / "records.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else []


def line(r: dict | None) -> str:
    if r is None:
        return "    (no paired data)"
    flag = "  WITHIN NOISE FLOOR" if r["within_floor"] else ""
    return (f"    {r['a']:>8} - {r['b']:<8} {r['metric']:<10} n={r['n']:<2} "
            f"Δmed={r['median_delta']:+8.2f} ({r['median_pct']:+6.2f}%)  "
            f"sign_p={r['sign_p']:.4f}  CI[{r['boot_ci'][0]:+.1f},{r['boot_ci'][1]:+.1f}]"
            f"  δ={r['cliffs']:+.2f}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    blob = analyse(load)
    if args.json:
        print(json.dumps(blob, indent=2))
        return 0

    for c in blob["comparisons"]:
        print(f"\n{c['headline']}\n  [{c['dir']}]")
        rows = c["rows"]
        if any(rows):
            for r in rows:
                print(line(r))
        elif c.get("rejection"):
            rej = c["rejection"]
            print(f"    {rej['rejected']}/{rej['of']} REJECTED — {rej['top_reason']}")
        else:
            print("    (no paired data)")
    print(f"\nnoise floor (null |%| median prefill): {blob['noise_floor_pct']:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
