"""Settle every A/B this project ran, with the project's own robust machinery.

Reads runs/ab-*/records.json, pairs measurements by (round, ncmoe) -- the only keys that
vary a configuration within an arm -- and for each comparison reports the paired sign-test
p, the bootstrap CI of the paired delta, and Cliff's delta on the raw values. Medians, not
means: this host produces cold-start outliers and the project already paid for trusting a
mean once.

The noise floor from ab-null is printed FIRST and every other delta is read against it.
A delta whose CI straddles the floor is not evidence, however tidy its median looks.

    python analyze_ab.py                 # human report
    python analyze_ab.py --json          # machine-readable, for STATUS.md provenance
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.analysis.robust import (                         # noqa: E402
    bootstrap_ci, cliffs_delta, hodges_lehmann, sign_test_p)

RUNS = pathlib.Path(__file__).parent / "runs"


def _median(v):
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def load(name):
    f = RUNS / name / "records.json"
    return json.load(open(f)) if f.exists() else []


def paired(recs, arm_a, arm_b, metric):
    """Pairs (arm_a - arm_b) by (round, ncmoe). Returns (deltas, pcts, raw_a, raw_b)."""
    def index(arm):
        out = {}
        for r in recs:
            if r["arm"] == arm and r.get(metric) is not None:
                out[(r["round"], r.get("ncmoe"))] = r[metric]
        return out
    A, B = index(arm_a), index(arm_b)
    keys = sorted(set(A) & set(B))
    deltas = [A[k] - B[k] for k in keys]
    pcts = [100.0 * (A[k] - B[k]) / B[k] for k in keys if B[k]]
    return deltas, pcts, [A[k] for k in keys], [B[k] for k in keys]


def report(recs, arm_a, arm_b, metric, floor_pct=None):
    deltas, pcts, ra, rb = paired(recs, arm_a, arm_b, metric)
    if not deltas:
        return None
    lo, hi = bootstrap_ci(deltas)
    med = _median(deltas)
    medpct = _median(pcts) if pcts else float("nan")
    p = sign_test_p(deltas)
    cd = cliffs_delta(ra, rb)
    verdict = ""
    if floor_pct is not None and abs(medpct) <= floor_pct:
        verdict = "  WITHIN NOISE FLOOR"
    return {"a": arm_a, "b": arm_b, "metric": metric, "n": len(deltas),
            "median_delta": round(med, 3), "median_pct": round(medpct, 2),
            "sign_p": round(p, 4), "boot_ci": [round(lo, 3), round(hi, 3)],
            "cliffs": round(cd, 3), "hl": round(hodges_lehmann(deltas), 3),
            "verdict": verdict}


def line(r):
    if r is None:
        return "    (no paired data)"
    return (f"    {r['a']:>8} - {r['b']:<8} {r['metric']:<10} n={r['n']:<2} "
            f"Δmed={r['median_delta']:+8.2f} ({r['median_pct']:+6.2f}%)  "
            f"sign_p={r['sign_p']:.4f}  CI[{r['boot_ci'][0]:+.1f},{r['boot_ci'][1]:+.1f}]"
            f"  δ={r['cliffs']:+.2f}{r['verdict']}")


# (directory, [(arm_a, arm_b, metric), ...], headline)
PLAN = [
    ("ab-null-qwen36-35b", [("same", "base", "prompt_tps"), ("same", "base", "gen_tps")],
     "NOISE FLOOR — same binary, same env; true delta is zero by construction"),
    ("ab-pinning-qwen36-35b",
     [("pin", "base", "prompt_tps"), ("pinpf", "pin", "prompt_tps"),
      ("pinpf", "base", "prompt_tps"), ("pin", "base", "gen_tps")], "PINNING — qwen36-35b"),
    ("ab-pinning-qwen3-30b",
     [("pin", "base", "prompt_tps"), ("pinpf", "pin", "prompt_tps"),
      ("pin", "base", "gen_tps")], "PINNING — qwen3-30b (independent geometry)"),
    ("ab-pinning-gpt-oss-20b",
     [("pin", "base", "prompt_tps"), ("pinpf", "pin", "prompt_tps"),
      ("pin", "base", "gen_tps")], "PINNING — gpt-oss-20b (independent geometry)"),
    ("ab-genpin-qwen36-35b", [("pin", "base", "gen_tps"), ("pin", "base", "prompt_tps")],
     "GENPIN — does pinning move GENERATION? (35B, near-resident)"),
    ("ab-decode-qwen36-35b",
     [("turbo", "base", "gen_tps"), ("turbo", "base", "prompt_tps")],
     "TURBO-MMA DECODE — 35B"),
    ("ab-decode-qwen3-30b",
     [("turbo", "base", "gen_tps"), ("turbo", "base", "prompt_tps")],
     "TURBO-MMA DECODE — 30B"),
    ("ab-rebased",
     [("fork", "base", "prompt_tps"), ("rebased", "base", "prompt_tps"),
      ("rebased", "fork", "prompt_tps"), ("rebased", "fork", "gen_tps")],
     "FORK vs REBASED vs BASE — is the fork still worth carrying? (n=18)"),
    ("ab-stack",
     [("prefetch", "base", "prompt_tps"), ("stackpf", "stack", "prompt_tps"),
      ("stack", "base", "prompt_tps"), ("stackpf", "prefetch", "prompt_tps")],
     "STACK 2x2 — is the L18/A-B disagreement about the BUILD or the prefetch?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Floor first: the median |%| of the null's prompt_tps delta.
    null = load("ab-null-qwen36-35b")
    _, npct, _, _ = paired(null, "same", "base", "prompt_tps")
    floor = _median([abs(x) for x in npct]) if npct else 1.0

    blob = {"noise_floor_pct": round(floor, 3), "comparisons": []}
    for d, comps, headline in PLAN:
        recs = load(d)
        rows = [report(recs, a, b, m, floor_pct=floor) for a, b, m in comps]
        blob["comparisons"].append({"dir": d, "headline": headline, "rows": rows})
        if not args.json:
            print(f"\n{headline}\n  [{d}]")
            for r in rows:
                print(line(r))
    if not args.json:
        print(f"\nnoise floor (null |%| median prefill): {floor:.3f}%")
    else:
        print(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
