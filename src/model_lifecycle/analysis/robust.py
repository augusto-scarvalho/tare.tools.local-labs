"""Robust and design-of-experiments statistics — for the regime this platform is in.

Everything here exists because `statistics.describe()` reports a NORMAL-APPROXIMATION CI
(`1.96*sd/sqrt(n)`), and every comparison in this project runs at n=4 rounds. At that
size the approximation is doing real work and nobody had checked whether it was earning
it. Three consequences, all addressed below:

  * a mean and an sd are both wrecked by one cold-start outlier, and this project has
    produced several (the first configuration of a sweep ran 4% fast, on a cold machine);
  * a percentile bootstrap makes no distributional assumption and is affordable at any n;
  * a distribution-free test at n=4 has a FLOOR on its p-value -- the exact sign test
    cannot go below 1/2^4 = 0.0625 two-sided even if every round agrees. So "the CI
    excludes zero" at n=4 is a parametric claim, and saying so is the honest version.

The Taguchi side (`anova_contributions`) answers the question a marginal-mean table
raises and does not settle: a factor's levels differ by X -- is X large compared with the
noise, or does it just look large in a small table?
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

# NOT `import statistics`. This package contains its own `statistics.py`, and running any
# module in here directly puts this directory first on sys.path, where it shadows the
# stdlib -- `statistics.fmean` then fails with a bare AttributeError that points at the
# wrong thing entirely. Three tiny helpers remove the ambiguity permanently.


def _mean(v: list[float]) -> float:
    return sum(v) / len(v)


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _stdev(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


@dataclass(frozen=True)
class RobustSummary:
    n: int
    median: float
    hodges_lehmann: float   # robust location: median of pairwise means
    mad: float              # robust spread, scaled to be sd-comparable on normal data
    boot_low: float
    boot_high: float

    def as_dict(self) -> dict:
        return {"n": self.n, "median": self.median,
                "hodges_lehmann": self.hodges_lehmann, "mad": self.mad,
                "boot_ci95": [self.boot_low, self.boot_high]}


def mad(vals: list[float]) -> float:
    """Median absolute deviation, scaled by 1.4826 so it matches sd on normal data.

    Preferred over sd here because a single bad round moves sd a lot and the median
    hardly at all -- and bad rounds are a known, recurring feature of this host rather
    than an anomaly."""
    if len(vals) < 2:
        return 0.0
    med = _median(vals)
    return 1.4826 * _median([abs(v - med) for v in vals])


def hodges_lehmann(vals: list[float]) -> float:
    """Median of all pairwise averages. More efficient than the median on near-normal
    data and still resistant to outliers -- the sensible default when n is small and you
    cannot afford to throw a sample away."""
    if not vals:
        raise ValueError("hodges_lehmann needs samples")
    pairs = [(vals[i] + vals[j]) / 2.0
             for i in range(len(vals)) for j in range(i, len(vals))]
    return _median(pairs)


def bootstrap_ci(vals: list[float], *, statistic=_mean, iterations: int = 10000,
                 alpha: float = 0.05, seed: int = 20260725) -> tuple[float, float]:
    """Percentile bootstrap CI. Seeded, because an unseeded CI that moves between runs
    of the same data is not a reportable number.

    Makes no normality assumption. At n=4 it is still limited by the data -- resampling
    cannot invent information the four rounds do not contain -- but it fails honestly:
    the interval comes out wide instead of narrow-and-wrong.
    """
    if len(vals) < 2:
        v = vals[0] if vals else float("nan")
        return (v, v)
    rng = random.Random(seed)
    n = len(vals)
    stats = sorted(statistic([vals[rng.randrange(n)] for _ in range(n)])
                   for _ in range(iterations))
    lo = stats[max(0, int(alpha / 2 * iterations) - 1)]
    hi = stats[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


def summarise(vals: list[float]) -> RobustSummary:
    lo, hi = bootstrap_ci(vals)
    return RobustSummary(n=len(vals), median=_median(vals),
                         hodges_lehmann=hodges_lehmann(vals), mad=mad(vals),
                         boot_low=lo, boot_high=hi)


def sign_test_p(deltas: list[float]) -> float:
    """Exact two-sided sign test on paired differences. Assumes nothing but symmetry of
    the sign under the null.

    Report it next to any paired CI at small n, because it is the floor of what can be
    claimed without distributional assumptions. At n=4 the smallest attainable p is
    0.125; at n=5, 0.0625; **n>=6 is required to reach 0.05**. Every paired comparison in
    this project so far ran at n=4.
    """
    nz = [d for d in deltas if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = sum(1 for d in nz if d > 0)
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Non-parametric effect size in [-1, 1]: the probability that a random value from
    `a` exceeds one from `b`, minus the reverse. Unlike a t-statistic it does not grow
    with n, so it separates "big" from "confidently measured" -- two things this project
    has conflated before."""
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def min_rounds_for(delta_pct: float, cv_pct: float, *, alpha: float = 0.05,
                   power: float = 0.80) -> int:
    """Rounds needed to detect `delta_pct` given round-to-round `cv_pct`, paired.

    Exists so "we found no effect" can be paired with "and here is the smallest effect
    this design could have seen" -- a null result without that number is indistinguishable
    from a blunt instrument, which is a mistake this project has already published once.
    """
    if delta_pct <= 0 or cv_pct <= 0:
        return 0
    z_a, z_b = 1.959964, 0.841621            # two-sided 95%, 80% power
    n = ((z_a + z_b) * cv_pct / delta_pct) ** 2
    return max(2, math.ceil(n))


def anova_contributions(rows: list[dict], factors: dict, response: str) -> list[dict]:
    """Percent contribution to total variance, per factor — the companion a Taguchi
    marginal-mean table needs.

    A spread of X between a factor's levels says nothing on its own: the question is what
    fraction of the total variation it explains, and how that compares with the residual.
    Sum of squares is decomposed the standard orthogonal-array way, which is valid
    BECAUSE the array is balanced -- each level of each factor sees the same mix of every
    other factor. That balance is asserted in the array's own self-check, not assumed.

    Returns factors sorted by contribution, plus a synthetic "residual" row. A factor
    whose contribution is at or below the residual is noise wearing a name.
    """
    vals = [r[response] for r in rows if r.get(response) is not None]
    if len(vals) < 2:
        return []
    grand = _mean(vals)
    ss_total = sum((v - grand) ** 2 for v in vals)
    if ss_total == 0:
        return []

    out, ss_explained = [], 0.0
    for name, (_col, levels) in factors.items():
        ss_f = 0.0
        dof = 0
        for lv in levels:
            group = [r[response] for r in rows
                     if r.get(name) == lv and r.get(response) is not None]
            if not group:
                continue
            ss_f += len(group) * (_mean(group) - grand) ** 2
            dof += 1
        ss_explained += ss_f
        out.append({"factor": name, "ss": ss_f, "dof": max(0, dof - 1),
                    "pct": 100.0 * ss_f / ss_total})
    residual = max(0.0, ss_total - ss_explained)
    out.sort(key=lambda d: -d["pct"])
    out.append({"factor": "(residual)", "ss": residual, "dof": None,
                "pct": 100.0 * residual / ss_total})
    return out


def format_contributions(rows: list[dict]) -> str:
    if not rows:
        return "  (not enough data)"
    res = next((r["pct"] for r in rows if r["factor"] == "(residual)"), 0.0)
    lines = []
    for r in rows:
        mark = ""
        if r["factor"] != "(residual)":
            mark = "  <- at or below noise" if r["pct"] <= res else ""
        lines.append(f"  {r['factor']:<12} {r['pct']:>6.1f}% of variance{mark}")
    return "\n".join(lines)


def _slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys on xs."""
    xm = _mean(xs)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    ym = _mean(ys)
    return sum((xs[i] - xm) * (ys[i] - ym) for i in range(len(xs))) / denom


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (ties broken by position -- fine for distinct ordered doses)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = rank(xs), rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def trend_slope_ci(doses: list[float], deltas_by_dose: list[list[float]], *,
                   iterations: int = 10000, alpha: float = 0.05,
                   seed: int = 20260801) -> dict:
    """Dose-response test: is the paired delta a MONOTONE function of an ordered dose?

    A single positive delta is weak; a delta that RISES with the mechanism's driver is strong.
    §B1's driver is active experts/token; §B2's is context length. Each dose is its own A/B
    (independent), so this fits a slope to (dose -> median paired delta) and bootstraps a CI by
    resampling within each dose's rounds. A CI excluding zero is a real trend; the Spearman rho
    gives the monotone direction without assuming linearity.

    `deltas_by_dose[i]` is the list of paired deltas measured at `doses[i]`.
    """
    if len(doses) != len(deltas_by_dose) or len(doses) < 2:
        raise ValueError("need >=2 doses, aligned with deltas_by_dose")
    meds = [_median(d) for d in deltas_by_dose]
    rng = random.Random(seed)
    slopes = []
    for _ in range(iterations):
        bmeds = []
        for d in deltas_by_dose:
            n = len(d)
            bmeds.append(_median([d[rng.randrange(n)] for _ in range(n)]))
        slopes.append(_slope(doses, bmeds))
    slopes.sort()
    lo = slopes[max(0, int(alpha / 2 * iterations) - 1)]
    hi = slopes[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return {"slope": _slope(doses, meds), "ci95": [lo, hi],
            "spearman": _spearman(doses, meds), "n_doses": len(doses),
            "positive_trend": lo > 0, "negative_trend": hi < 0}


def non_inferiority(deltas: list[float], margin: float, *, better: str = "higher",
                    alpha: float = 0.05, seed: int = 20260801) -> dict:
    """One-sided non-inferiority: is `new` NOT MEANINGFULLY WORSE than `base` within `margin`?

    For the KV-quant trade (q4 vs q8 KV): we do not need q4 to be BETTER on quality, only within
    a pre-declared margin while it is faster/leaner. `deltas` are paired `new - base`.
      better='higher' (quality, higher is better): non-inferior if the one-sided lower bound of
        the delta stays above `-margin`.
      better='lower'  (latency, lower is better):  non-inferior if the one-sided upper bound of
        the delta stays below `+margin`.
    The margin is fixed BEFORE data (pre-registration); tightening is free, loosening is not.
    """
    lo, hi = bootstrap_ci(deltas, alpha=2 * alpha, seed=seed)   # one-sided bound at `alpha`
    med = _median(deltas)
    if better == "higher":
        return {"median_delta": med, "one_sided_bound": lo, "margin": margin,
                "better": better, "non_inferior": lo > -margin}
    return {"median_delta": med, "one_sided_bound": hi, "margin": margin,
            "better": better, "non_inferior": hi < margin}


if __name__ == "__main__":
    clean = [80.0, 81.0, 79.5, 80.5]
    dirty = clean + [40.0]                      # one cold-start style outlier

    # Robust location barely moves; the mean does.
    assert abs(_mean(dirty) - 72.2) < 0.1
    assert abs(hodges_lehmann(dirty) - 80.0) < 1.5, hodges_lehmann(dirty)
    assert mad(dirty) < _stdev(dirty), "MAD must resist what sd does not"

    # Bootstrap is seeded: identical input, identical interval.
    assert bootstrap_ci(clean) == bootstrap_ci(clean)
    lo, hi = bootstrap_ci(clean)
    assert lo < _mean(clean) < hi

    # The floor that matters: n=4 cannot reach 0.05 without assuming a distribution.
    assert abs(sign_test_p([1, 1, 1, 1]) - 0.125) < 1e-9, sign_test_p([1, 1, 1, 1])
    assert abs(sign_test_p([1, 1, 1, 1, 1]) - 0.0625) < 1e-9
    assert sign_test_p([1, 1, 1, 1, 1, 1]) < 0.05, "n=6 is where distribution-free bites"
    assert sign_test_p([1, -1, 1, -1]) == 1.0

    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0

    # Sample-size planning: a smaller effect against the same noise needs more rounds.
    assert min_rounds_for(60.0, 3.0) < min_rounds_for(2.0, 3.0)

    # ANOVA on a synthetic array where factor A drives everything and B drives nothing.
    facs = {"A": (0, [1, 2]), "B": (1, [1, 2])}
    rows = [{"A": a, "B": b, "y": 100.0 + 50.0 * (a - 1)}
            for a in (1, 2) for b in (1, 2)]
    contrib = anova_contributions(rows, facs, "y")
    top = contrib[0]
    assert top["factor"] == "A" and top["pct"] > 99.0, contrib
    assert next(c for c in contrib if c["factor"] == "B")["pct"] < 1.0

    # Dose-response trend: a delta that clearly RISES with dose has a positive slope CI and
    # a monotone Spearman; a flat dose-response does not read as a trend.
    _doses = [4.0, 8.0, 10.0]
    _rising = [[-0.1, 0.0, -0.05], [0.5, 0.6, 0.55], [1.0, 1.1, 0.9]]
    _t = trend_slope_ci(_doses, _rising, iterations=2000)
    assert _t["positive_trend"] and _t["spearman"] > 0.9, _t
    _flat = [[0.1, -0.1, 0.0], [0.0, 0.1, -0.1], [-0.05, 0.05, 0.0]]
    assert not trend_slope_ci(_doses, _flat, iterations=2000)["positive_trend"]

    # Non-inferiority: delta hovering at -1 is non-inferior within margin 3, not within 0.5.
    _near = [-1.0, -0.8, -1.2, -0.9, -1.1, -1.0]
    assert non_inferiority(_near, 3.0, better="higher")["non_inferior"]
    assert not non_inferiority(_near, 0.5, better="higher")["non_inferior"]

    print("robust stats self-check OK")
    print(f"  mean(dirty)={_mean(dirty):.1f}  HL={hodges_lehmann(dirty):.1f}  "
          f"sd={_stdev(dirty):.1f}  MAD={mad(dirty):.1f}")
    print(f"  sign-test floor: n=4 -> {sign_test_p([1]*4):.4f}, "
          f"n=6 -> {sign_test_p([1]*6):.4f}")
    print(format_contributions(contrib))
