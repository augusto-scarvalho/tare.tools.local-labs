#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate RNN-06-P0 per-candidate P0_RESULTS_*.json into one P0_CURVES.csv.

Pure post-processing (no GPU, no model). Reads every P0_RESULTS_*.json in the
run directory and emits a tidy long-format CSV with one row per (candidate,dose)
plus the exposed denominators, so the sweep is auditable without JSON tooling.
"""
import csv
import glob
import json
import os
import sys

FIELDS = [
    "candidate_tag", "model_id", "model_runnable", "task_competent",
    "p0_graded_band", "pairs", "seq_len_tokens", "n",
    "constrained_acc", "format_adherence", "unconstrained_exact_acc",
    "chance", "value_vocab_size", "n_constrained_correct", "n_format_ok",
    "eval_seconds",
]


def main(rundir):
    rows = []
    for path in sorted(glob.glob(os.path.join(rundir, "P0_RESULTS_*.json"))):
        rec = json.load(open(path))
        tag = rec.get("candidate_tag")
        mid = rec.get("model_id")
        st = rec.get("status", {})
        runnable = st.get("MODEL_RUNNABLE")
        competent = st.get("TASK_COMPETENT")
        band = st.get("P0_GRADED_BAND")
        curves = rec.get("curves", [])
        if not curves:
            rows.append({"candidate_tag": tag, "model_id": mid,
                         "model_runnable": runnable, "task_competent": competent,
                         "p0_graded_band": band})
            continue
        for c in curves:
            rows.append({
                "candidate_tag": tag, "model_id": mid,
                "model_runnable": runnable, "task_competent": competent,
                "p0_graded_band": band,
                "pairs": c.get("pairs"), "seq_len_tokens": c.get("seq_len_tokens"),
                "n": c.get("n"), "constrained_acc": c.get("constrained_acc"),
                "format_adherence": c.get("format_adherence"),
                "unconstrained_exact_acc": c.get("unconstrained_exact_acc"),
                "chance": c.get("chance"), "value_vocab_size": c.get("value_vocab_size"),
                "n_constrained_correct": c.get("n_constrained_correct"),
                "n_format_ok": c.get("n_format_ok"),
                "eval_seconds": c.get("eval_seconds"),
            })
    out = os.path.join(rundir, "P0_CURVES.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
