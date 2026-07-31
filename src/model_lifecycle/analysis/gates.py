"""Eligibility gates — checked BEFORE any score is computed.

The plan is explicit that gates come first: a candidate that is unsafe or
non-operational must never be ranked, because a ranking implies "choose among these"
and an ineligible entry is not a choice. Scoring an ineligible model is how a fast,
unstable configuration ends up looking attractive.

Gates are declarative and every failure names the measured value. "Rejected" without
the number is a verdict nobody can check.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Gates:
    minimum_pass_rate: float = 0.95
    minimum_free_vram_mb: int = 250
    maximum_ttft_p95_s: float | None = 120.0
    maximum_cv: float | None = 0.20
    require_no_breach: bool = True
    # answer_rate is deliberately NOT gated by default: an infrastructure suite
    # measures whether the machine can serve the model, not whether the model answered
    # well. Gating it here would silently turn a fit benchmark into a quality one.
    minimum_answer_rate: float | None = None


@dataclass
class GateResult:
    eligible: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"eligible": self.eligible, "failures": self.failures,
                "checked": self.checked}


def evaluate(run: dict, gates: Gates | None = None) -> GateResult:
    g = gates or Gates()
    fails: list[str] = []
    checked: list[str] = []

    verdict = run.get("verdict")
    if verdict == "REJECTED":
        # The guard already disqualified it. Recorded as a gate failure so the reason
        # travels with the candidate instead of living only in the run log.
        fails.append(f"envelope breach: {run.get('reason')}")
    checked.append("verdict")

    if verdict in ("ERROR", "SKIPPED"):
        fails.append(f"run did not produce data ({verdict}: {run.get('reason')})")

    pr = run.get("pass_rate")
    checked.append("pass_rate")
    if pr is None or pr < g.minimum_pass_rate:
        fails.append(f"pass_rate {pr} < {g.minimum_pass_rate}")

    vram = run.get("min_free_vram_mb")
    checked.append("min_free_vram_mb")
    if vram is None or vram < g.minimum_free_vram_mb:
        # Two models can both "fit" in 24 GB while one leaves a few dozen MB free and
        # falls over as soon as the KV cache grows. Margin is the property, not fit.
        fails.append(f"min_free_vram_mb {vram} < {g.minimum_free_vram_mb}")

    if g.maximum_ttft_p95_s is not None and (t := run.get("ttft")):
        checked.append("ttft.p95")
        if t.get("p95", 0) > g.maximum_ttft_p95_s:
            fails.append(f"ttft p95 {t['p95']:.1f}s > {g.maximum_ttft_p95_s}s")

    if g.maximum_cv is not None and (gt := run.get("gen_tps")):
        checked.append("gen_tps.cv")
        if gt.get("cv", 0) > g.maximum_cv:
            # Consistency is a gate, not a tiebreaker: a config that is fast on average
            # and occasionally terrible is worse operationally than a slower steady one.
            fails.append(f"gen_tps cv {gt['cv']:.2f} > {g.maximum_cv}")

    if g.minimum_answer_rate is not None:
        checked.append("answer_rate")
        ar = run.get("answer_rate")
        if ar is None or ar < g.minimum_answer_rate:
            fails.append(f"answer_rate {ar} < {g.minimum_answer_rate}")

    return GateResult(eligible=not fails, failures=fails, checked=checked)


if __name__ == "__main__":
    ok_run = {"verdict": "OK", "pass_rate": 1.0, "min_free_vram_mb": 5717,
              "ttft": {"p95": 3.2}, "gen_tps": {"cv": 0.0}, "answer_rate": 0.0}
    r = evaluate(ok_run)
    assert r.eligible, r.failures
    # answer_rate 0 must NOT disqualify an infrastructure run by default...
    assert "answer_rate" not in r.checked
    # ...but must when the caller asks for it.
    r2 = evaluate(ok_run, Gates(minimum_answer_rate=0.9))
    assert not r2.eligible and "answer_rate" in r2.failures[0]

    tight = evaluate({**ok_run, "min_free_vram_mb": 100})
    assert not tight.eligible and "min_free_vram_mb" in tight.failures[0]

    jumpy = evaluate({**ok_run, "gen_tps": {"cv": 0.45}})
    assert not jumpy.eligible and "cv" in jumpy.failures[0]

    rejected = evaluate({"verdict": "REJECTED", "reason": "ram floor", "pass_rate": 0.0,
                         "min_free_vram_mb": 9000})
    assert not rejected.eligible and "envelope breach" in rejected.failures[0]
    print("gates self-check OK")
