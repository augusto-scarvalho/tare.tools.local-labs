#!/usr/bin/env python3
"""LAB-SERVE-001c — open-loop campaign analyzer.

Reads the campaign raw dir (per-cell <tag>.normalized.json + manifest.jsonl + campaign_meta.json),
then produces: per-cell matrix, per-(load point) load-vs-latency curves, ON-OFF paired effects by rep,
per-class (interactive/coding via input_len band) and per output-length-bin stratification, and a
queueing-onset table. n=2 reps ⇒ direction + magnitude only (no p<0.05). Failed/timed-out requests are
counted, never dropped from denominators.

Run:  PYTHONPATH=src python ops/lab_serve_openloop_analyze.py runs/serving/LAB-SERVE-001c/campaign/raw
"""
import csv, json, pathlib, statistics as st, sys
sys.path.insert(0, "src")
from model_lifecycle.analysis.robust import bootstrap_ci, sign_test_p, hodges_lehmann, _median as median, mad

RAW = pathlib.Path(sys.argv[1])
NORM = RAW.parent / "normalized"; NORM.mkdir(parents=True, exist_ok=True)
CODING_MIN_INPUT = 8000            # disjoint band boundary (interactive max ~4091, coding min ~8234)
LEN_BINS = [("short", 0, 256), ("medium", 256, 1024), ("long", 1024, 10**9)]


def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def per_request(cell):
    """Reconstruct per-request rows from the details arrays."""
    d = cell.get("details") or {}
    il = d.get("input_lens") or []; ol = d.get("output_lens") or []
    tt = d.get("ttfts") or []; itl = d.get("itls") or []; err = d.get("errors") or []
    rows = []
    for i in range(len(il)):
        e = err[i] if i < len(err) else None
        ttft = tt[i] if i < len(tt) else None
        itls_i = itl[i] if i < len(itl) else []
        e2e = (ttft + sum(itls_i)) if (ttft is not None) else None
        tpot = (sum(itls_i) / len(itls_i)) if itls_i else None
        rows.append({"input_len": il[i], "output_len": ol[i] if i < len(ol) else None,
                     "cls": "coding" if il[i] >= CODING_MIN_INPUT else "interactive",
                     "error": e, "ok": not e,
                     "ttft_ms": (ttft * 1000) if ttft is not None else None,
                     "e2e_ms": (e2e * 1000) if e2e is not None else None,
                     "tpot_ms": (tpot * 1000) if tpot is not None else None})
    return rows


def summarize_rows(rows):
    ok = [r for r in rows if r["ok"]]
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2e = [r["e2e_ms"] for r in ok if r["e2e_ms"] is not None]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]
    outl = [r["output_len"] for r in ok if r["output_len"] is not None]
    return {"n": len(rows), "n_ok": len(ok), "n_err": len(rows) - len(ok),
            "ttft_median_ms": round(median(ttft), 1) if ttft else None,
            "ttft_p95_ms": round(pct(ttft, 95), 1) if ttft else None,
            "e2e_median_ms": round(median(e2e), 1) if e2e else None,
            "e2e_p95_ms": round(pct(e2e, 95), 1) if e2e else None,
            "tpot_median_ms": round(median(tpot), 2) if tpot else None,
            "out_len_median": round(median(outl), 1) if outl else None,
            "out_len_mean": round(st.mean(outl), 1) if outl else None}


def main():
    manifest = [json.loads(l) for l in (RAW / "manifest.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((RAW / "campaign_meta.json").read_text()) if (RAW / "campaign_meta.json").exists() else {}
    cells = []
    for m in manifest:
        f = RAW / f"{m['tag']}.normalized.json"
        cell = json.loads(f.read_text()) if f.exists() else {}
        rows = per_request(cell) if cell else []
        s = cell.get("upstream_summary", {}) or {}
        v = cell.get("validity", {}) or {}
        wall = cell.get("wall_s") or 0
        agg = summarize_rows(rows)
        rec = {**m, "wall_s": wall,
               "completed": v.get("completed"), "errors": v.get("errors"),
               "timeout": m.get("rc") == 124, "server_failed": m.get("rc") == 97,
               "offered_rate": m.get("rate"),
               "completed_rate": round((v.get("completed") or 0) / wall, 4) if wall else None,
               "request_throughput": round(s.get("request_throughput", 0), 4),
               "output_throughput": round(s.get("output_throughput", 0), 1),
               "token_ratio": v.get("token_accounting_ratio"),
               "vram_peak_mb": (cell.get("gpu") or {}).get("vram_peak_mb"),
               "power_mean_w": (cell.get("gpu") or {}).get("power_mean_w"),
               "util_mean": (cell.get("gpu") or {}).get("util_mean"),
               **agg,
               # per-class
               **{f"{c}_{k}": vv for c in ("interactive", "coding")
                  for k, vv in summarize_rows([r for r in rows if r["cls"] == c]).items()
                  if k in ("n_ok", "ttft_median_ms", "e2e_median_ms", "out_len_mean")},
               # output-length bins (share of ok requests)
               **{f"len_{name}": sum(1 for r in rows if r["ok"] and r["output_len"] is not None
                                     and lo <= r["output_len"] < hi)
                  for name, lo, hi in LEN_BINS},
               "_rows": rows}
        cells.append(rec)

    # matrix.csv (drop _rows)
    cols = [k for k in cells[0].keys() if k != "_rows"] if cells else []
    with (NORM / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for c in cells: w.writerow(c)

    # queueing-onset / load curve: per (point, arm) pooled over reps
    METRICS = ["ttft_median_ms", "ttft_p95_ms", "e2e_median_ms", "e2e_p95_ms", "tpot_median_ms",
               "output_throughput", "completed_rate", "out_len_mean"]
    curve = []
    for pt in ["low", "near", "over"]:
        for arm in ["off", "on"]:
            sub = [c for c in cells if c["point"] == pt and c["arm"] == arm and not c["server_failed"]]
            if not sub: continue
            pooled = [r for c in sub for r in c["_rows"]]
            agg = summarize_rows(pooled)
            curve.append({"point": pt, "arm": arm, "offered_rate": sub[0]["offered_rate"],
                          "n_cells": len(sub), **agg,
                          "completed_rate_mean": round(st.mean([c["completed_rate"] for c in sub if c["completed_rate"]]), 4)
                          if any(c["completed_rate"] for c in sub) else None})
    (NORM / "load_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")
    with (NORM / "load_curve.csv").open("w", newline="", encoding="utf-8") as f:
        if curve:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys()), extrasaction="ignore"); w.writeheader(); w.writerows(curve)

    # paired ON-OFF by rep, per (point, metric)
    effects = []
    for pt in ["low", "near", "over"]:
        for m in METRICS:
            on = {c["rep"]: c.get(m) for c in cells if c["point"] == pt and c["arm"] == "on" and c.get(m) is not None}
            off = {c["rep"]: c.get(m) for c in cells if c["point"] == pt and c["arm"] == "off" and c.get(m) is not None}
            reps = sorted(set(on) & set(off))
            if len(reps) < 1: continue
            deltas = [on[r] - off[r] for r in reps]
            row = {"point": pt, "metric": m, "n_pairs": len(reps),
                   "median_on": round(median([on[r] for r in reps]), 2),
                   "median_off": round(median([off[r] for r in reps]), 2),
                   "paired_delta": round(median(deltas), 2),
                   "deltas": [round(x, 2) for x in deltas],
                   "direction_agree": all(d > 0 for d in deltas) or all(d < 0 for d in deltas)}
            if len(reps) >= 3:
                lo, hi = bootstrap_ci(deltas, statistic=hodges_lehmann, iterations=10000, seed=42)
                row.update({"ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "sign_p": round(sign_test_p(deltas), 4)})
            effects.append(row)
    (NORM / "paired_effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    with (NORM / "paired_effects.csv").open("w", newline="", encoding="utf-8") as f:
        if effects:
            keys = ["point", "metric", "n_pairs", "median_on", "median_off", "paired_delta",
                    "direction_agree", "deltas"]
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(effects)

    summary = {"meta": meta, "n_cells": len(cells),
               "cells_ok": sum(1 for c in cells if not c["server_failed"] and not c["timeout"]),
               "server_failures": [c["tag"] for c in cells if c["server_failed"]],
               "timeouts": [c["tag"] for c in cells if c["timeout"]],
               "token_ratios": sorted({c["token_ratio"] for c in cells if c["token_ratio"] is not None}),
               "load_curve": curve, "paired_effects": effects}
    (NORM / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # console headline
    print(f"cells={len(cells)} ok={summary['cells_ok']} failures={summary['server_failures']} timeouts={summary['timeouts']}")
    print("\n== load curve (TTFT/E2E median ms, throughput) ==")
    for c in curve:
        print(f"  {c['point']:>4} {c['arm']:>3} off_rate={c['offered_rate']:<6} "
              f"compl_rate={c.get('completed_rate_mean')} ttft_med={c['ttft_median_ms']} ttft_p95={c['ttft_p95_ms']} "
              f"e2e_med={c['e2e_median_ms']} e2e_p95={c['e2e_p95_ms']} tpot_med={c['tpot_median_ms']}")
    print("\n== paired ON-OFF (delta = on - off), per load point ==")
    for e in effects:
        if e["metric"] in ("ttft_median_ms", "e2e_median_ms", "tpot_median_ms", "output_throughput"):
            print(f"  {e['point']:>4} {e['metric']:<20} on={e['median_on']:>9} off={e['median_off']:>9} "
                  f"d={e['paired_delta']:>9} dir_agree={e['direction_agree']} deltas={e['deltas']}")


if __name__ == "__main__":
    main()
