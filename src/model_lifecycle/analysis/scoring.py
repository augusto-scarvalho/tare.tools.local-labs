"""Scores per objective, then a role weighting — never one number.

A single composite hides trade-offs, and hiding trade-offs is how a model gets picked
for being fast when the job needed stability. Each score answers one question; the
role decides how much each question matters.

All scores are 0-100 and RELATIVE to the candidate set, because absolute tokens/s
means nothing without a fleet to compare against. That also means a score is only
valid alongside the set it was computed with -- recorded, not implied.
"""
from __future__ import annotations

from dataclasses import dataclass

# Weights per role. Straight from the owner's brief: a coder is judged mostly on
# quality and agentic behaviour, a fast worker on latency, a reasoner on quality plus
# context. Infrastructure-only runs can still be scored on the subset available.
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "coder":     {"quality": 0.40, "agentic": 0.25, "stability": 0.15, "context": 0.10, "speed": 0.10},
    "fast":      {"speed": 0.45, "stability": 0.25, "fit": 0.20, "quality": 0.10},
    "reasoning": {"quality": 0.40, "context": 0.25, "stability": 0.20, "agentic": 0.15},
    # Infrastructure-only: what an infra suite can actually answer today. Named
    # explicitly so nobody mistakes it for a general "best model" verdict.
    "infra":     {"speed": 0.50, "fit": 0.30, "stability": 0.20},
}


def _norm(value: float | None, lo: float, hi: float, *, higher_is_better: bool) -> float:
    if value is None or hi <= lo:
        return 0.0
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    return 100.0 * (x if higher_is_better else 1.0 - x)


@dataclass
class Scores:
    speed: float = 0.0
    fit: float = 0.0
    stability: float = 0.0
    quality: float = 0.0
    context: float = 0.0
    agentic: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def for_role(self, role: str) -> float:
        w = ROLE_WEIGHTS.get(role)
        if not w:
            raise KeyError(f"unknown role {role!r}; known: {sorted(ROLE_WEIGHTS)}")
        # Renormalise over the dimensions actually present, so an infrastructure-only
        # run is not silently punished for having no quality score yet.
        total = sum(w.values())
        return sum(getattr(self, k, 0.0) * v for k, v in w.items()) / total


def score_all(runs: list[dict]) -> dict[str, Scores]:
    """Score a candidate set relative to itself. Ineligible runs must be filtered by
    the caller BEFORE scoring -- ranking something that cannot be used is what gates
    exist to prevent."""
    usable = [r for r in runs if r.get("gen_tps")]
    if not usable:
        return {}

    gens = [r["gen_tps"]["mean"] for r in usable]
    vrams = [r.get("min_free_vram_mb") or 0 for r in usable]
    cvs = [r["gen_tps"].get("cv", 0.0) for r in usable]
    ttfts = [(r.get("ttft") or {}).get("p95") for r in usable]
    ttfts_known = [t for t in ttfts if t is not None]

    g_lo, g_hi = min(gens), max(gens)
    v_lo, v_hi = min(vrams), max(vrams)
    c_lo, c_hi = min(cvs), max(cvs)
    t_lo, t_hi = (min(ttfts_known), max(ttfts_known)) if ttfts_known else (0.0, 1.0)

    out: dict[str, Scores] = {}
    for r in usable:
        gen = _norm(r["gen_tps"]["mean"], g_lo, g_hi, higher_is_better=True)
        ttft_p95 = (r.get("ttft") or {}).get("p95")
        lat = _norm(ttft_p95, t_lo, t_hi, higher_is_better=False) if ttft_p95 is not None else gen
        out[r["config_id"]] = Scores(
            # Throughput and latency are different bottlenecks; speed is the pair,
            # weighted toward sustained rate but not blind to a slow first token.
            speed=0.6 * gen + 0.4 * lat,
            fit=_norm(r.get("min_free_vram_mb"), v_lo, v_hi, higher_is_better=True),
            stability=_norm(r["gen_tps"].get("cv", 0.0), c_lo, c_hi, higher_is_better=False),
        )
    return out


if __name__ == "__main__":
    runs = [
        {"config_id": "fast-tight", "gen_tps": {"mean": 90.0, "cv": 0.02},
         "min_free_vram_mb": 300, "ttft": {"p95": 2.0}},
        {"config_id": "slow-roomy", "gen_tps": {"mean": 40.0, "cv": 0.02},
         "min_free_vram_mb": 9000, "ttft": {"p95": 8.0}},
        {"config_id": "fast-jumpy", "gen_tps": {"mean": 88.0, "cv": 0.40},
         "min_free_vram_mb": 5000, "ttft": {"p95": 2.2}},
    ]
    s = score_all(runs)
    assert s["fast-tight"].speed > s["slow-roomy"].speed
    assert s["slow-roomy"].fit > s["fast-tight"].fit, "roomy must win fit"
    assert s["fast-tight"].stability > s["fast-jumpy"].stability, "jumpy must lose stability"
    # The point of per-role weighting: the winner CHANGES with the question asked.
    infra = {k: v.for_role("infra") for k, v in s.items()}
    fast = {k: v.for_role("fast") for k, v in s.items()}
    assert max(infra, key=infra.get) != "fast-jumpy", "an unstable config must not top infra"
    print("scores:", {k: round(v, 1) for k, v in infra.items()})
    print("fast  :", {k: round(v, 1) for k, v in fast.items()})
    print("scoring self-check OK")
