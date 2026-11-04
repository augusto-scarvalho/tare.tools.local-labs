#!/usr/bin/env python3
"""LAB-SERVE-001b block-level paired analysis. The statistical unit is the independent server-level
BLOCK (one server start), NOT the individual request. For each (N, metric) it pairs the on-block and
off-block BY REP, forms 5 paired deltas (on-off), and reports median(on)/median(off), the paired
delta with a seeded bootstrap CI, and a sign test — reusing the lab's robust methodology.

Run:  PYTHONPATH=src python ops/lab_serve_analyze.py <raw_dir_with_blocks.json> <label>
Emits matrix.csv (per block), paired_effects.{json,csv}, token_exactness + acceptance summary.
"""
import csv, json, pathlib, sys
sys.path.insert(0, "src")
from model_lifecycle.analysis.robust import (bootstrap_ci, sign_test_p, hodges_lehmann,
                                             _median as median, mad)

RAW = pathlib.Path(sys.argv[1]); LABEL = sys.argv[2] if len(sys.argv) > 2 else RAW.parent.name
NORM = RAW.parent / "normalized"; NORM.mkdir(parents=True, exist_ok=True)
blocks = json.loads((RAW / "blocks.json").read_text())

METRICS = ["output_throughput", "ttft_median_ms", "ttft_p95_ms", "tpot_median_ms", "tpot_p95_ms",
           "itl_median_ms", "e2e_median_ms", "e2e_p95_ms", "vram_peak_mb", "power_mean_w",
           "power_peak_w", "energy_j", "j_per_output_token"]

# flatten per-block cells into rows; enrich with an ENERGY signal (§13) from the cell's raw file.
# NOTE: energy_j ~= power_mean_w * wall_s uses the GPU-sampler window, which spans warmup+run, so
# j_per_output_token is an OBSERVED signal (integration window not exactly the scored interval).
rows = []
for b in blocks:
    for ncell, c in (b.get("cells") or {}).items():
        row = {"rep": b["rep"], "arm": b["arm"], "N": int(ncell[1:]), **c}
        rawf = RAW / f"{c.get('tag')}.normalized.json"
        if rawf.exists():
            nd = json.loads(rawf.read_text()); wall = nd.get("wall_s") or 0
            pw = (nd.get("gpu") or {}).get("power_mean_w") or 0
            tok = (nd.get("validity") or {}).get("total_output_tokens") or 0
            row["energy_j"] = round(pw * wall, 1) if (pw and wall) else None
            row["j_per_output_token"] = round(pw * wall / tok, 3) if (pw and wall and tok) else None
        rows.append(row)
with (NORM / "matrix.csv").open("w", newline="", encoding="utf-8") as f:
    cols = ["rep", "arm", "N", "success_all", "token_ratio", "token_exact", *METRICS]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

# token exactness (§8): forced-length + exact tokenizer must be EXACT, not just within 10%
all_exact = all(r.get("token_exact") for r in rows)
ratios = sorted({r.get("token_ratio") for r in rows})

# acceptance (§12) from on-blocks
acc = []
for b in blocks:
    for a in (b.get("acceptance") or []):
        acc.append(a["accept_rate"])
acc_summary = {"n_probes": len(acc), "median_accept_rate": round(median(acc), 3) if acc else None,
               "min": min(acc) if acc else None, "max": max(acc) if acc else None}

# paired effects by N (pair on/off BY REP)
effects = []
for N in sorted({r["N"] for r in rows}):
    for m in METRICS:
        on = {r["rep"]: r.get(m) for r in rows if r["N"] == N and r["arm"] == "on" and r.get(m) is not None}
        off = {r["rep"]: r.get(m) for r in rows if r["N"] == N and r["arm"] == "off" and r.get(m) is not None}
        reps = sorted(set(on) & set(off))
        if len(reps) < 2:
            continue
        onv = [on[r] for r in reps]; offv = [off[r] for r in reps]
        deltas = [on[r] - off[r] for r in reps]           # paired, block-level
        ci_lo, ci_hi = bootstrap_ci(deltas, statistic=hodges_lehmann, iterations=10000, seed=42)
        effects.append({
            "N": N, "metric": m, "n_blocks_paired": len(reps),
            "median_on": round(median(onv), 3), "mad_on": round(mad(onv), 3),
            "median_off": round(median(offv), 3), "mad_off": round(mad(offv), 3),
            "paired_delta_hl": round(hodges_lehmann(deltas), 3),
            "delta_ci_lo": round(ci_lo, 3), "delta_ci_hi": round(ci_hi, 3),
            "sign_p": round(sign_test_p(deltas), 4),
            "ci_excludes_zero": (ci_lo > 0) or (ci_hi < 0),
        })
(NORM / "paired_effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
with (NORM / "paired_effects.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(effects[0].keys())); w.writeheader(); w.writerows(effects)

summary = {"label": LABEL, "n_blocks": len(blocks),
           "token_exact_all": all_exact, "token_ratios_seen": ratios,
           "acceptance": acc_summary,
           "arm_order": [{"rep": b["rep"], "arm": b["arm"]} for b in blocks],
           "topology_example": next((b["topology"] for b in blocks if b.get("topology")), None)}
(NORM / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"blocks": len(blocks), "token_exact_all": all_exact, "ratios": ratios,
                  "acceptance": acc_summary}, indent=2))
# headline: throughput, tpot, e2e paired deltas per N
for e in effects:
    if e["metric"] in ("output_throughput", "tpot_median_ms", "e2e_median_ms"):
        star = " *CI-excl-0" if e["ci_excludes_zero"] else ""
        print(f"  N={e['N']:>1} {e['metric']:<20} on={e['median_on']:>8} off={e['median_off']:>8} "
              f"d(on-off)={e['paired_delta_hl']:>8} CI[{e['delta_ci_lo']},{e['delta_ci_hi']}] p={e['sign_p']}{star}")
