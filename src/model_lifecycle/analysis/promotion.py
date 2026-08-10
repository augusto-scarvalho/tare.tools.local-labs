"""Promotion semantics — the LEXICOGRAPHIC decision that turns measurements into a choice
(LAB-QA-003).

The lab already uses this order informally; this makes it code. A candidate is compared to the
incumbent in strict stages, and a later stage is consulted ONLY if every earlier one passes:

    eligibility  ->  correctness  ->  quality  ->  performance

A faster candidate CANNOT win if it crashes, breaches the resource envelope, fails correctness,
degrades quality beyond the margin, or does not terminate reliably. This is deliberately NOT a
single weighted score: collapsing safety, correctness, quality and speed into one number is exactly
how "a fast, unstable configuration ends up looking attractive" (see gates.py). Performance is a
tiebreak among the *already-acceptable*, never a way to buy back a failed gate.

Reuses `gates.evaluate` for the eligibility layer (envelope / pass_rate / VRAM margin / TTFT / CV).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sys

from .gates import Gates, evaluate as evaluate_gates

sys.path.insert(0, __import__("pathlib").Path(__file__).resolve().parents[3].as_posix())
from benchmark_harness_qa import check_comparable  # identity comparability guard (WA-CLOSE-003)


@dataclass(frozen=True)
class PromotionMargins:
    """Non-inferiority margins and the performance-win threshold.

    PROVENANCE of the numeric defaults (WA-CLOSE-004): these are **OPERATOR POLICY**, proposed as
    promotion-gate margins in IDEAS_BACKLOG.md §B3 ("±1pp quality, ≥15% wall-clock, ≥20%
    reasoning-tokens, 0.5pp crash ceiling") — NOT empirically derived from a measured noise floor,
    and NOT a scientific invariant. They are configurable per decision (pass a PromotionMargins).
    `perf_win_pct=15.0` in particular is an operator-chosen "is this worth switching for" bar, to be
    revisited once LAB-SERVE-001 establishes the real serving noise floor. Do not read it as ratified.
    """
    quality_margin_pp: float = 1.0        # candidate may be at most this far BELOW baseline quality
    correctness_margin_pp: float = 0.0    # correctness must not regress below baseline - this
    correctness_floor: float | None = None  # optional absolute correctness floor (0..1)
    min_termination_rate: float = 0.98    # must terminate reliably (cf. fable-fusion, LAB-CLOSE-002)
    perf_win_pct: float = 15.0            # OPERATOR POLICY (B3): a WIN needs >= this % wall-clock gain


@dataclass
class PromotionDecision:
    verdict: str                          # PROMOTE | REJECT | HOLD
    stage: str                            # eligibility | correctness | quality | performance
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "stage": self.stage,
                "reasons": self.reasons, "detail": self.detail}


def _pct_improvement(baseline_wall: float, cand_wall: float) -> float:
    """% wall-clock improvement of candidate over baseline (positive = faster)."""
    if not baseline_wall:
        return 0.0
    return (baseline_wall - cand_wall) / baseline_wall * 100.0


def decide(candidate: dict, baseline: dict, *, margins: PromotionMargins | None = None,
           gates: Gates | None = None,
           candidate_identity: dict | None = None, baseline_identity: dict | None = None) -> PromotionDecision:
    """Lexicographic promotion decision of `candidate` against the incumbent `baseline`.

    Expected keys (all optional; absence is treated conservatively):
      eligibility (consumed by gates.evaluate): verdict, pass_rate, min_free_vram_mb, ttft, gen_tps
      correctness: `correctness` in 0..1     quality: `quality` in 0..1 (e.g. HumanEval+ pass@1)
      termination: `termination_rate` in 0..1    performance: `wall_clock_s` (lower is better)

    If both identity blocks are supplied, comparison FAILS CLOSED when they are not comparable
    (different benchmark/dataset/scorer) — returning an INCOMPARABLE verdict instead of silently
    ranking apples vs oranges (WA-CLOSE-003). Commit/timestamp differences are advisory only.
    """
    m = margins or PromotionMargins()

    # ---- Stage 0: comparability (identity) ---------------------------------------------------
    if candidate_identity is not None and baseline_identity is not None:
        comp = check_comparable(candidate_identity, baseline_identity)
        if not comp["comparable"]:
            fields = ", ".join(d["field"] for d in comp["invalidating"])
            return PromotionDecision("INCOMPARABLE", "identity",
                                     [f"incompatible identity on: {fields}"], comp)

    # ---- Stage 1: eligibility (safety / operational) -----------------------------------------
    ge = evaluate_gates(candidate, gates)
    if not ge.eligible:
        return PromotionDecision("REJECT", "eligibility", ge.failures, {"gate": ge.as_dict()})

    term = candidate.get("termination_rate")
    if term is None or term < m.min_termination_rate:
        return PromotionDecision("REJECT", "eligibility",
                                 [f"termination_rate {term} < {m.min_termination_rate}"],
                                 {"termination_rate": term})

    # ---- Stage 2: correctness ----------------------------------------------------------------
    c_cand, c_base = candidate.get("correctness"), baseline.get("correctness")
    if m.correctness_floor is not None and (c_cand is None or c_cand < m.correctness_floor):
        return PromotionDecision("REJECT", "correctness",
                                 [f"correctness {c_cand} < floor {m.correctness_floor}"], {})
    if c_base is not None and c_cand is not None:
        if c_cand < c_base - m.correctness_margin_pp / 100.0:
            return PromotionDecision("REJECT", "correctness",
                                     [f"correctness {c_cand:.3f} < baseline {c_base:.3f} "
                                      f"- {m.correctness_margin_pp}pp"], {})

    # ---- Stage 3: quality (non-inferiority vs baseline) --------------------------------------
    q_cand, q_base = candidate.get("quality"), baseline.get("quality")
    if q_base is not None and q_cand is not None:
        if q_cand < q_base - m.quality_margin_pp / 100.0:
            return PromotionDecision("REJECT", "quality",
                                     [f"quality {q_cand:.3f} < baseline {q_base:.3f} "
                                      f"- {m.quality_margin_pp}pp"], {})

    # ---- Stage 4: performance (tiebreak among the already-acceptable) ------------------------
    w_cand, w_base = candidate.get("wall_clock_s"), baseline.get("wall_clock_s")
    if w_cand is None or w_base is None:
        return PromotionDecision("HOLD", "performance",
                                 ["passed all gates but no comparable wall-clock to judge a win"], {})
    improvement = _pct_improvement(w_base, w_cand)
    detail = {"wall_improvement_pct": round(improvement, 1),
              "threshold_pct": m.perf_win_pct}
    if improvement >= m.perf_win_pct:
        return PromotionDecision("PROMOTE", "performance",
                                 [f"non-inferior on gates/correctness/quality and "
                                  f"{improvement:.1f}% faster (>= {m.perf_win_pct}%)"], detail)
    return PromotionDecision("HOLD", "performance",
                             [f"acceptable but only {improvement:.1f}% faster "
                              f"(< {m.perf_win_pct}% win threshold) — keep incumbent"], detail)


if __name__ == "__main__":
    base = {"verdict": "OK", "pass_rate": 1.0, "min_free_vram_mb": 5000, "correctness": 0.90,
            "quality": 0.90, "termination_rate": 1.0, "wall_clock_s": 100.0}

    # 1) faster but CRASHES the envelope -> REJECT at eligibility (speed cannot buy it back)
    d = decide({**base, "verdict": "REJECTED", "reason": "ram floor", "wall_clock_s": 50.0}, base)
    assert d.verdict == "REJECT" and d.stage == "eligibility", d

    # 2) faster but NON-TERMINATING -> REJECT at eligibility (cf. fable-fusion)
    d = decide({**base, "termination_rate": 0.6, "wall_clock_s": 50.0}, base)
    assert d.verdict == "REJECT" and "termination_rate" in d.reasons[0], d

    # 3) faster but QUALITY degraded beyond 1pp -> REJECT at quality
    d = decide({**base, "quality": 0.80, "wall_clock_s": 50.0}, base)
    assert d.verdict == "REJECT" and d.stage == "quality", d

    # 4) clean AND >=15% faster -> PROMOTE
    d = decide({**base, "wall_clock_s": 80.0}, base)
    assert d.verdict == "PROMOTE" and d.stage == "performance", d

    # 5) clean but only 5% faster -> HOLD (no win, keep incumbent)
    d = decide({**base, "wall_clock_s": 95.0}, base)
    assert d.verdict == "HOLD", d

    # 6) within quality margin (0.5pp below) and fast -> PROMOTE (non-inferior)
    d = decide({**base, "quality": 0.895, "wall_clock_s": 80.0}, base)
    assert d.verdict == "PROMOTE", d

    # 7) WA-CLOSE-004: performance can NEVER compensate a failed earlier stage, even at 100x speed.
    huge_speed = {"wall_clock_s": 0.001}   # ~infinite performance advantage
    for broken, stage in [
        ({"verdict": "REJECTED", "reason": "ram floor"}, "eligibility"),   # ineligible
        ({"termination_rate": 0.5}, "eligibility"),                        # non-terminating
        ({"correctness": 0.10}, "correctness"),                            # wrong
        ({"quality": 0.10}, "quality"),                                    # low quality
    ]:
        d = decide({**base, **broken, **huge_speed}, base)
        assert d.verdict == "REJECT" and d.stage == stage, (broken, d)

    # 8) WA-CLOSE-003: incompatible identity -> INCOMPARABLE, never ranked
    idA = {"benchmark_name": "humaneval", "dataset_hash": "AAA", "benchmark_version": "v1", "scorer_version": "v1"}
    idB = {"benchmark_name": "humaneval", "dataset_hash": "BBB", "benchmark_version": "v1", "scorer_version": "v1"}
    d = decide({**base, "wall_clock_s": 50.0}, base, candidate_identity=idA, baseline_identity=idB)
    assert d.verdict == "INCOMPARABLE" and d.stage == "identity", d
    # ...but an ADVISORY-only difference (different commit) stays comparable
    idC = {**idA, "harness_commit": "deadbeef"}
    d = decide({**base, "wall_clock_s": 80.0}, base, candidate_identity=idC, baseline_identity=idA)
    assert d.verdict == "PROMOTE", d
    print("promotion self-check OK")
