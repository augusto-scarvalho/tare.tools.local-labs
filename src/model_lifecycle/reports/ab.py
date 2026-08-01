"""The paired-A/B analysis, as reusable functions decoupled from where the records live.

`analyze_ab.py` reads records off disk (`runs/ab-*/records.json`); the report generator
reads the SAME records out of the Store (`ingest:ab-*` plans, the backfill put them there
verbatim). Both need identical pairing math and the identical editorial PLAN of which
comparisons to make. Holding that in two places is how the numbers drift from each other --
the precise failure this project is trying to stop -- so it lives here once, and both callers
pass in a `load(dirname) -> records` function.

What is DATA and what is EDITORIAL is kept separate on purpose:
  * the pairing, the stats, the noise-floor flag are DATA -- fully derived, never authored;
  * the PLAN (which arms to compare, and the one-line headline on each group) is EDITORIAL --
    it encodes what the experiment was FOR, which no amount of data can recover on its own.
"""
from __future__ import annotations

from typing import Callable

from ..analysis.robust import bootstrap_ci, cliffs_delta, hodges_lehmann, sign_test_p


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def paired(recs: list[dict], arm_a: str, arm_b: str, metric: str):
    """Pairs (arm_a - arm_b) by (round, ncmoe) -- the only keys that vary a configuration
    within an arm. Returns (deltas, pcts, raw_a, raw_b), aligned on the shared keys."""
    def index(arm):
        out = {}
        for r in recs:
            if r.get("arm") == arm and r.get(metric) is not None:
                out[(r.get("round"), r.get("ncmoe"))] = r[metric]
        return out
    A, B = index(arm_a), index(arm_b)
    keys = sorted(set(A) & set(B))
    deltas = [A[k] - B[k] for k in keys]
    pcts = [100.0 * (A[k] - B[k]) / B[k] for k in keys if B[k]]
    return deltas, pcts, [A[k] for k in keys], [B[k] for k in keys]


def summarise(recs: list[dict], arm_a: str, arm_b: str, metric: str,
              floor_pct: float | None = None) -> dict | None:
    """One comparison's paired statistics. `floor_pct`, when given, auto-flags a median
    whose magnitude sits at or below the noise floor -- the flag is derived, not judged."""
    deltas, pcts, ra, rb = paired(recs, arm_a, arm_b, metric)
    if not deltas:
        return None
    lo, hi = bootstrap_ci(deltas)
    med = _median(deltas)
    medpct = _median(pcts) if pcts else float("nan")
    within = floor_pct is not None and abs(medpct) <= floor_pct
    return {"a": arm_a, "b": arm_b, "metric": metric, "n": len(deltas),
            "median_delta": round(med, 3), "median_pct": round(medpct, 2),
            "sign_p": round(sign_test_p(deltas), 4),
            "boot_ci": [round(lo, 3), round(hi, 3)],
            "cliffs": round(cliffs_delta(ra, rb), 3),
            "hl": round(hodges_lehmann(deltas), 3),
            "within_floor": within}


def noise_floor(null_recs: list[dict]) -> float:
    """The instrument's resolution: median |%| of the null A/B's prefill delta, where the
    true delta is zero by construction (same binary, same env). Every other delta is read
    against this. Falls back to 1.0 only if the null run is absent."""
    _, npct, _, _ = paired(null_recs, "same", "base", "prompt_tps")
    return _median([abs(x) for x in npct]) if npct else 1.0


# (directory, [(arm_a, arm_b, metric), ...], headline). EDITORIAL: this is the list of
# comparisons the project set out to make, and the headline naming what each was for. The
# directory name is also the Store plan suffix (`ingest:<dir>`), so one PLAN drives both
# the on-disk and the Store-backed reader.
PLAN = [
    ("ab-null-qwen36-35b",
     [("same", "base", "prompt_tps"), ("same", "base", "gen_tps")],
     "NOISE FLOOR — same binary, same env; true delta is zero by construction"),
    ("ab-pinning-qwen36-35b",
     [("pin", "base", "prompt_tps"), ("pinpf", "pin", "prompt_tps"),
      ("pinpf", "base", "prompt_tps"), ("pin", "base", "gen_tps")],
     "PINNING — qwen36-35b (256 experts)"),
    ("ab-pinning-qwen3-30b",
     [("pin", "base", "prompt_tps"), ("pinpf", "pin", "prompt_tps"),
      ("pin", "base", "gen_tps")],
     "PINNING — qwen3-30b (128 experts, independent geometry)"),
    ("ab-pinning-gpt-oss-20b",
     [("pin", "base", "prompt_tps"), ("pinpf", "pin", "prompt_tps"),
      ("pin", "base", "gen_tps")],
     "PINNING — gpt-oss-20b (32 experts, independent geometry)"),
    ("ab-genpin-qwen36-35b",
     [("pin", "base", "gen_tps"), ("pin", "base", "prompt_tps")],
     "GENPIN — does pinning move GENERATION? (35B, near-resident)"),
    ("ab-genpin-nemotron-120b",
     [("pin", "base", "gen_tps"), ("pin", "base", "prompt_tps")],
     "GENPIN — the transfer-bound regime (Nemotron-120B) — §B1"),
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
     "STACK 2×2 — is the L18/A-B disagreement about the BUILD or the prefetch?"),
]


def _rejection_digest(recs: list[dict]) -> dict | None:
    """When a group produced records but no comparable metric, the story is WHY -- almost
    always the guard rejecting on the envelope (this is how §B1 looks in the data). Summarise
    it rather than reporting a bare 'no paired data'."""
    rejected = [r for r in recs if str(r.get("verdict")).upper() == "REJECTED"]
    if not rejected:
        return None
    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[str(r.get("reason") or "?")] = reasons.get(str(r.get("reason") or "?"), 0) + 1
    top = max(reasons.items(), key=lambda kv: kv[1])
    return {"rejected": len(rejected), "of": len(recs), "top_reason": top[0]}


def analyse(load: Callable[[str], list[dict]]) -> dict:
    """Run the whole PLAN against a source. `load(dir)` returns that group's records --
    from disk, from the Store, from anywhere. Returns the machine blob both callers render.

    The floor is computed first (from ab-null) and threaded into every comparison, so the
    `within_floor` flag on each row is against the same instrument resolution.
    """
    floor = noise_floor(load("ab-null-qwen36-35b"))
    blob = {"noise_floor_pct": round(floor, 3), "comparisons": []}
    for d, comps, headline in PLAN:
        recs = load(d)
        rows = [summarise(recs, a, b, m, floor_pct=floor) for a, b, m in comps]
        entry = {"dir": d, "headline": headline, "n_records": len(recs), "rows": rows}
        if recs and not any(rows):           # produced records, none comparable -> why?
            entry["rejection"] = _rejection_digest(recs)
        blob["comparisons"].append(entry)
    return blob
