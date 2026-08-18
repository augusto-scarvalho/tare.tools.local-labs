"""A2 paired statistics: join a base arm and a ThinkingCap arm by task_id and answer, with
the tests the metrics actually call for -- not a t-test on skewed token counts.

Design decisions, each with a reason the project can defend:

* PAIRED, BY task_id. Each problem is one matched pair (same prompt, same subset). The unit
  of analysis is the per-problem DIFFERENCE, which removes between-problem variance -- the
  whole reason to run both arms on the identical seeded subset.

* REASONING-TOKEN REDUCTION uses the MEDIAN of the per-problem ratio, with a PERCENTILE
  BOOTSTRAP 95% CI, and a WILCOXON signed-rank test on the paired counts. Token counts are
  right-skewed and heavy-tailed; the mean and a t-test are the wrong summary and the wrong
  test. (The vendor reports a mean paired ratio; we report BOTH but lead with the median,
  and give the mean for comparability.)

* ACCURACY is paired binary -> McNemar's exact test on the discordant pairs (base-right/
  cap-wrong vs base-wrong/cap-right). A two-arm accuracy difference of ~1pp on <=200 items
  is within noise by construction; the honest claim is NON-INFERIORITY within a stated
  margin (default 1pp, the project's ROPE), which we report as the CI on the paired delta --
  NOT a p<0.05 "no difference".

* GUARDS AGAINST "SHORT BUT WRONG": we split the reduction by difficulty (tertiles of the
  base arm's reasoning length) and we report, among problems the base got RIGHT, how many the
  cap arm got WRONG and how hard it cut them. A reduction bought by capitulation on the hard
  tail must be visible, not hidden in a mean.

Zero deps beyond numpy: scipy is not installed, so Wilcoxon (normal approx with tie/zero
correction) and McNemar (exact binomial) are implemented here and self-checked against known
values. The percentile bootstrap needs nothing but numpy.

    python a2_stats.py --base qwen36-27b-dense --cap thinkingcap-27b --workload gsm8k --tag a2r0
    python a2_stats.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

import numpy as np

RUNS = pathlib.Path(__file__).parent / "runs" / "a2"

# ----------------------------------------------------------------------------- scoring
# GSM8K: the model was asked for a final "#### <number>" line; fall back to the last number
# in the text if it did not comply (a format miss is a real, but different, failure).
_HASH = re.compile(r"####\s*\$?(-?[0-9][0-9,]*\.?[0-9]*)")
_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def gsm8k_extract(text: str) -> str | None:
    if not text:
        return None
    m = _HASH.search(text)
    if m:
        return m.group(1).replace(",", "").replace("$", "").rstrip(".")
    nums = _NUM.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").replace("$", "").rstrip(".")


def numeric_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


# ----------------------------------------------------------------------------- stats
def bootstrap_ci(values: np.ndarray, stat=np.median, reps: int = 10000,
                 alpha: float = 0.05, seed: int = 20260804) -> tuple[float, float, float]:
    """Percentile bootstrap CI of `stat` over `values`. Returns (point, lo, hi).
    Deterministic (seeded) because every number in this project must be reproducible."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(reps, values.size))
    boot = stat(values[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(values)), float(lo), float(hi))


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank on paired (x, y). Returns (W, p).

    Normal approximation with tie correction and Wilcoxon's zero-difference handling (drop
    zeros). Valid for the sample sizes here (n >= ~20 after dropping ties); for tiny n it is
    conservative. Self-checked against a textbook case below.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    d = x - y
    d = d[d != 0]                        # Wilcoxon: discard zero differences
    n = d.size
    if n == 0:
        return (0.0, 1.0)
    r = _rankdata(np.abs(d))             # average ranks for ties
    w_plus = float(r[d > 0].sum())
    w_minus = float(r[d < 0].sum())
    W = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    # tie correction to the variance
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie = (counts ** 3 - counts).sum()
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    if var <= 0:
        return (W, 1.0)
    z = (W - mean + 0.5) / math.sqrt(var)   # continuity correction
    p = 2.0 * _norm_cdf(z)                   # W <= mean, so z <= 0; two-sided
    return (W, min(1.0, p))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean of their positions."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sa = a[order]
    i = 0
    n = a.size
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial) p-value on discordant counts b, c.
    b = base-correct & cap-wrong; c = base-wrong & cap-correct. Under H0 each discordant
    pair is a fair coin. Exact via the binomial PMF -- correct for the small discordant
    counts these subsets produce (where the chi-square approximation is unsafe)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided: P(X<=k) + P(X>=n-k) = 2*P(X<=k) (symmetric), capped at 1
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * cdf)


# ----------------------------------------------------------------------------- loading
def load_arm(tag: str, model: str, workload: str, lora: str = "") -> dict[str, dict]:
    stem = f"{tag}__{model}__{workload}"
    if lora:
        stem += f"__{lora}"
    path = RUNS / f"{stem}.json"
    if not path.exists():
        raise SystemExit(f"missing arm: {path}")
    recs = json.loads(path.read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in recs if r.get("task_id")}


def correct(rec: dict, workload: str, ext_score: dict | None) -> bool | None:
    """Per-problem correctness. GSM8K scored here (safe numeric match); HumanEval+ needs the
    external evalplus result (executes code) -- None if not supplied."""
    if workload == "gsm8k":
        return numeric_equal(gsm8k_extract(rec.get("completion") or ""), rec.get("gold"))
    if ext_score is not None:
        return ext_score.get(rec["task_id"])
    return None


def _pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def report(base: dict, cap: dict, workload: str,
           ext_base: dict | None, ext_cap: dict | None, margin_pp: float) -> None:
    ids = sorted(set(base) & set(cap))
    print(f"\n=== A2 paired report [{workload}] — {len(ids)} matched problems ===")
    if not ids:
        return

    # ---- reasoning-token reduction (the headline) ----
    br = np.array([base[i]["reasoning_tokens"] for i in ids], float)
    cr = np.array([cap[i]["reasoning_tokens"] for i in ids], float)
    ok = ~(np.isnan(br) | np.isnan(cr)) & (br > 0)   # ratio needs a nonzero base
    ratio = np.where(ok, 1.0 - cr / np.where(br == 0, np.nan, br), np.nan)
    med, lo, hi = bootstrap_ci(ratio[ok], np.median)
    mean, mlo, mhi = bootstrap_ci(ratio[ok], np.mean)
    _, pw = wilcoxon_signed_rank(br[ok], cr[ok])
    print("\n-- reasoning tokens (base -> cap) --")
    print(f"   base median {np.median(br[ok]):.0f}  |  cap median {np.median(cr[ok]):.0f}")
    print(f"   paired reduction: median {_pct(med)} [{_pct(lo)}, {_pct(hi)}]"
          f"   mean {_pct(mean)} [{_pct(mlo)}, {_pct(mhi)}]")
    print(f"   Wilcoxon signed-rank p = {pw:.2e}  (n={ok.sum()})")

    # ---- total generated tokens + wall-clock (what actually costs time) ----
    for label, key in (("total generated tokens", "predicted_n"), ("wall-clock (s)", "wall_s")):
        b = np.array([base[i].get(key) or np.nan for i in ids], float)
        c = np.array([cap[i].get(key) or np.nan for i in ids], float)
        m = ~(np.isnan(b) | np.isnan(c)) & (b > 0)
        red = 1.0 - c[m] / b[m]
        pt, l, h = bootstrap_ci(red, np.median)
        print(f"\n-- {label} -- paired reduction median {_pct(pt)} [{_pct(l)}, {_pct(h)}]"
              f"   (base med {np.median(b[m]):.1f} -> cap med {np.median(c[m]):.1f})")

    # ---- accuracy: paired, McNemar, non-inferiority margin ----
    cb = [correct(base[i], workload, ext_base) for i in ids]
    cc = [correct(cap[i], workload, ext_cap) for i in ids]
    if all(x is not None for x in cb) and all(x is not None for x in cc):
        cb = np.array(cb, bool); cc = np.array(cc, bool)
        acc_b, acc_c = cb.mean(), cc.mean()
        b_only = int((cb & ~cc).sum())      # base right, cap wrong  (regressions)
        c_only = int((~cb & cc).sum())      # base wrong, cap right  (gains)
        p_mn = mcnemar_exact(b_only, c_only)
        # CI on the paired accuracy delta (cap - base) via bootstrap of per-item diff.
        diff = cc.astype(float) - cb.astype(float)
        dpt, dlo, dhi = bootstrap_ci(diff, np.mean)
        print("\n-- accuracy (paired) --")
        print(f"   base {100*acc_b:.1f}%  cap {100*acc_c:.1f}%   delta {_pct(dpt)} "
              f"[{_pct(dlo)}, {_pct(dhi)}]")
        print(f"   discordant: base-only-right {b_only}, cap-only-right {c_only}  "
              f"McNemar exact p = {p_mn:.3f}")
        noninf = dlo >= -margin_pp / 100.0
        print(f"   non-inferiority (margin {margin_pp:.1f}pp): "
              f"{'PASS' if noninf else 'FAIL'} (CI low {_pct(dlo)})")

        # ---- short-but-wrong guard: among base-correct, what did cap lose, and how hard cut? ----
        base_right = cb
        lost = base_right & ~cc
        if lost.any():
            lr = ratio[lost & ok]
            lr = lr[~np.isnan(lr)]
            cut = f"median {_pct(np.median(lr))}" if lr.size else "n/a"
            print(f"   ⚠ short-but-wrong watch: cap lost {int(lost.sum())} of "
                  f"{int(base_right.sum())} base-correct; their reasoning cut {cut}")
    else:
        print("\n-- accuracy -- skipped (HumanEval+ needs external evalplus results; "
              "pass --ext-base/--ext-cap)")

    # ---- difficulty split: tertiles of base reasoning length ----
    if ok.sum() >= 6:
        q = np.nanpercentile(br[ok], [33.3, 66.7])
        blen = br.copy()
        buckets = [("easy", blen <= q[0]), ("med", (blen > q[0]) & (blen <= q[1])),
                   ("hard", blen > q[1])]
        print("\n-- reduction by difficulty (base reasoning-length tertiles) --")
        for name, mask in buckets:
            mm = mask & ok
            if mm.sum():
                r = ratio[mm]; r = r[~np.isnan(r)]
                print(f"   {name:4} (n={mm.sum():2}): reasoning cut median "
                      f"{_pct(np.median(r)) if r.size else 'n/a'}  "
                      f"(base med len {np.median(br[mm]):.0f})")

    # ---- starvation / format ----
    for name, arm in (("base", base), ("cap", cap)):
        ans = sum(1 for i in ids if arm[i].get("answered"))
        print(f"   [{name}] answered {ans}/{len(ids)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base-lora", default="")
    ap.add_argument("--cap-lora", default="")
    ap.add_argument("--ext-base", default="", help="evalplus results JSON {task_id: bool} for HumanEval+ base arm")
    ap.add_argument("--ext-cap", default="")
    ap.add_argument("--margin-pp", type=float, default=1.0, help="non-inferiority ROPE margin")
    args = ap.parse_args()

    base = load_arm(args.tag, args.base, args.workload, args.base_lora)
    cap = load_arm(args.tag, args.cap, args.workload, args.cap_lora)
    ext_base = json.loads(pathlib.Path(args.ext_base).read_text()) if args.ext_base else None
    ext_cap = json.loads(pathlib.Path(args.ext_cap).read_text()) if args.ext_cap else None
    report(base, cap, args.workload, ext_base, ext_cap, args.margin_pp)
    return 0


if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        # bootstrap CI brackets the point estimate and is deterministic
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pt, lo, hi = bootstrap_ci(v, np.median)
        assert lo <= pt <= hi and abs(pt - 0.3) < 1e-9, (pt, lo, hi)
        assert bootstrap_ci(v, np.median)[1] == bootstrap_ci(v, np.median)[1], "seeded"

        # Wilcoxon vs a known small case: differences [+1,+2,+3,+4,-5] -> textbook W- = 5.
        # (ranks of |d|: 1,2,3,4,5; negative rank sum = 5.) p two-sided ~ large (no effect).
        W, p = wilcoxon_signed_rank(np.array([1, 2, 3, 4, 0]), np.array([0, 0, 0, 0, 5]))
        assert abs(W - 5.0) < 1e-9, W
        assert 0.0 <= p <= 1.0
        # A clean monotone reduction (cap always lower) -> tiny p.
        base_r = np.array([100, 200, 300, 400, 500, 600, 700, 800], float)
        cap_r = base_r * 0.5
        _, p2 = wilcoxon_signed_rank(base_r, cap_r)
        assert p2 < 0.05, p2

        # McNemar: all discordant one way -> significant; balanced -> ~1.0
        assert mcnemar_exact(10, 0) < 0.01
        assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-9
        assert mcnemar_exact(0, 0) == 1.0

        # GSM8K extraction: hash marker wins, comma/$ stripped, fallback to last number
        assert gsm8k_extract("blah\n#### 42") == "42"
        assert gsm8k_extract("answer #### 1,234.0") == "1234.0"
        assert gsm8k_extract("the total is 18 apples") == "18"
        assert gsm8k_extract("nothing here!") is None
        assert numeric_equal("42", "42.0") and not numeric_equal("42", "43")

        # rankdata ties share the average rank
        assert list(_rankdata(np.array([10, 10, 20]))) == [1.5, 1.5, 3.0]

        print("a2_stats self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
