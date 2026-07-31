"""Percentiles, not just means.

`50, 51, 52, 49, 10` averages to 42.4 and the disaster disappears. Every metric this
platform reports carries its distribution, because a configuration that is fast on
average and occasionally terrible is not a configuration anyone wants promoted.
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Distribution:
    n: int
    min: float
    max: float
    mean: float
    median: float
    p90: float
    p95: float
    p99: float
    stdev: float
    cv: float            # coefficient of variation: stdev/mean, the stability signal
    ci95_low: float
    ci95_high: float

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear interpolation between order statistics. Small n is the normal case
    here (3-10 repetitions), so nearest-rank would jump coarsely."""
    if not sorted_vals:
        raise ValueError("no samples")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def describe(values: list[float]) -> Distribution:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        raise ValueError("describe() needs at least one sample")
    s = sorted(vals)
    n = len(s)
    mean = st.fmean(s)
    stdev = st.stdev(s) if n > 1 else 0.0
    # Normal-approximation CI. With n<30 it is indicative, not a claim -- reported so
    # a two-run difference is not mistaken for a finding.
    half = 1.96 * stdev / math.sqrt(n) if n > 1 else 0.0
    return Distribution(
        n=n, min=s[0], max=s[-1], mean=mean, median=_percentile(s, 0.50),
        p90=_percentile(s, 0.90), p95=_percentile(s, 0.95), p99=_percentile(s, 0.99),
        stdev=stdev, cv=(stdev / mean) if mean else 0.0,
        ci95_low=mean - half, ci95_high=mean + half,
    )


if __name__ == "__main__":
    d = describe([50, 51, 52, 49, 10])
    assert abs(d.mean - 42.4) < 0.01, d.mean
    assert d.min == 10, "the bad run must stay visible"
    assert d.cv > 0.3, "a run at 1/5 of the others must show as unstable"
    one = describe([7.0])
    assert one.stdev == 0.0 and one.p95 == 7.0
    print(f"mean={d.mean:.1f} median={d.median} p95={d.p95} cv={d.cv:.2f} min={d.min}")
    print("statistics self-check OK")
