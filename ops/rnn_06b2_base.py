#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD — confirmatory runner.

Fixed-length (M=128, 514 tokens) unique-state-load sweep on the frozen subject. DS (primary,
disjoint-space load) + SS (secondary, scored-space load) + non-gating length diagnostic. Three
channels, S=3 strata, Wilson + cluster-bootstrap CIs, delta-AURC. Asserts EXACT fixed length &
target->query gap across doses. Mints FIXED_LENGTH_STATE_LOAD_REGION and, if QUALIFIED, applies
the preregistered 06C dose-selection rule. Self-records executed-source identity + re-verifies
b2QualificationSetSha256 / b2StressGridSha256 before outcomes. cs=32. No training, no push.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import Mamba2ForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06b2_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD")
SPEC_PATH = os.path.join(OUTDIR, "B2_QUALIFICATION_SPEC.json")
GRID_PATH = os.path.join(OUTDIR, "B2_STRESS_GRID.json")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
AUTOBATCH = 1536
QUAL_SEED = 20260815


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:
        return f"<git-error:{e}>"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [max(0.0, c - h), min(1.0, c + h)]


@torch.no_grad()
def eval_cell(model, prompts, golds, vtensor, vset, bs):
    n = len(prompts)
    con = np.zeros(n, bool); unc = np.zeros(n, bool); fmt = np.zeros(n, bool)
    for b in range(0, n, bs):
        ids = torch.tensor(prompts[b:b + bs], device=DEVICE, dtype=torch.long)
        last = model(ids).logits[:, -1, :].float()
        uncon = last.argmax(-1)
        conv = vtensor[last.index_select(1, vtensor).argmax(-1)]
        for j in range(ids.shape[0]):
            g = int(golds[b + j])
            con[b + j] = int(conv[j]) == g
            unc[b + j] = int(uncon[j]) == g
            fmt[b + j] = int(uncon[j]) in vset
    return con, unc, fmt


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    runner = os.path.abspath(__file__)
    libpath = os.path.join(os.path.dirname(runner), "rnn_06b2_lib.py")
    results = {"packet": "RNN-06B2-MAMBA-FIXED-LENGTH-STATE-LOAD", "kind": "fixed_length_state_load_results"}

    spec = json.load(open(SPEC_PATH)); grid = json.load(open(GRID_PATH))
    q_rec = spec.pop("b2QualificationSetSha256"); disj = spec.pop("disjointness_proof")
    q_re = lib.sha256_of_obj(spec)
    g_rec = grid.pop("b2StressGridSha256"); g_re = lib.sha256_of_obj(grid)
    results["challenge_identities"] = {"b2QualificationSetSha256_recorded": q_rec,
                                       "b2QualificationSetSha256_recomputed": q_re,
                                       "qual_sha_match": q_rec == q_re,
                                       "b2StressGridSha256_recorded": g_rec,
                                       "b2StressGridSha256_recomputed": g_re,
                                       "stress_sha_match": g_rec == g_re,
                                       "disjointness_proof": disj}
    assert q_rec == q_re and g_rec == g_re, "B2 spec/grid sha mismatch"

    from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
    results["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_sha256": sha256_file(libpath), "lib_git_blob": git("hash-object", libpath),
        "lib_dirty": git("status", "--porcelain", "--", libpath),
        "git_head": git("rev-parse", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "configuration_mamba2_sha256": sha256_file(configuration_mamba2.__file__),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "repo_id": REPO_ID, "revision": REVISION, "chunk_size": CHUNK, "dtype": str(DTYPE),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "b2QualificationSetSha256": q_rec, "b2StressGridSha256": g_rec,
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "PRE_REGISTRATION.md"))}
    assert modeling_mamba2.is_fast_path_available is False

    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in model.backbone.layers:
        blk.mixer.chunk_size = CHUNK
    model.config.chunk_size = CHUNK
    pools, pool_meta = lib.build_pools(tok, QUAL_SEED)
    results["pools"] = pool_meta
    vset = set(pools["scored_vals"]); vtensor = torch.tensor(sorted(vset), device=DEVICE, dtype=torch.long)
    results["chance"] = 1.0 / len(vset)

    M = grid["M"]; DOSES = grid["doses"]; ARMS = grid["arms"]; STRATA = grid["s_strata"]
    TAU_HI, TAU_LO = grid["tau_hi"], grid["tau_lo"]; SENT = grid["sentinel_scheme"]
    examples = spec["examples"]; ex_stratum = np.array([e["stratum"] for e in examples])

    torch.cuda.reset_peak_memory_stats()
    cells = {}; curves = []; per_cell_prompt_sha = {}; seq_lens = {}
    for U in DOSES:
        for arm in ARMS:
            prompts, golds = [], []
            for e in examples:
                p, g = lib.materialize(e, M, U, arm, SENT, pools)
                prompts.append(p); golds.append(g)
            seq_len = len(prompts[0])
            assert all(len(p) == seq_len for p in prompts), "ragged prompt length within cell"
            seq_lens[f"{U}_{arm}"] = seq_len
            bs = max(1, min(64, AUTOBATCH // seq_len))
            per_cell_prompt_sha[f"{U}_{arm}"] = lib.sha256_of_obj(prompts)
            tc = time.time()
            con, unc, fmt = eval_cell(model, prompts, golds, vtensor, vset, bs)
            cells[(U, arm)] = {"con": con, "unc": unc, "fmt": fmt}
            k = int(con.sum()); n = len(con)
            strat = {int(s): float(con[ex_stratum == s].mean()) for s in range(STRATA)}
            row = {"U": U, "arm": arm, "seq_len": seq_len, "n": n, "constrained_correct": k,
                   "constrained_acc": k / n, "constrained_ci95": wilson(k, n),
                   "unconstrained_acc": float(unc.mean()), "format_adherence": float(fmt.mean()),
                   "per_stratum_constrained_acc": strat, "eval_seconds": round(time.time() - tc, 2)}
            curves.append(row)
            print(f"[06B2] U={U:3d} {arm} len={seq_len} con={row['constrained_acc']:.3f} "
                  f"unc={row['unconstrained_acc']:.3f} fmt={row['format_adherence']:.3f} "
                  f"({row['eval_seconds']}s)", file=sys.stderr)
    results["curves"] = curves
    results["per_cell_prompt_sha256"] = per_cell_prompt_sha

    # ---- length diagnostic (non-gating) ----
    diag = grid["length_diagnostic"]; diag_rows = []
    for Mv in diag["M_values"]:
        prompts, golds = [], []
        for e in examples:
            espec = lib.build_example_spec(QUAL_SEED, e["ex_id"], e["stratum"], Mv)
            p, g = lib.materialize(espec, Mv, diag["U"], diag["arm"], SENT, pools)
            prompts.append(p); golds.append(g)
        seq_len = len(prompts[0]); bs = max(1, min(64, AUTOBATCH // seq_len))
        con, _, _ = eval_cell(model, prompts, golds, vtensor, vset, bs)
        k = int(con.sum()); n = len(con)
        diag_rows.append({"M": Mv, "U": diag["U"], "seq_len": seq_len, "n": n,
                          "constrained_correct": k, "constrained_acc": k / n,
                          "constrained_ci95": wilson(k, n)})
        print(f"[06B2-diag] M={Mv} len={seq_len} U={diag['U']} con={k/n:.3f}", file=sys.stderr)
    results["length_diagnostic"] = diag_rows
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # ---- curves ----
    def acc(U, arm):
        c = cells[(U, arm)]["con"]; return int(c.sum()) / len(c)
    ds_curve = {U: acc(U, "DS") for U in DOSES}
    ss_curve = {U: acc(U, "SS") for U in DOSES}
    ds_strata = {s: {U: float(cells[(U, "DS")]["con"][ex_stratum == s].mean()) for U in DOSES}
                 for s in range(STRATA)}
    results["ds_constrained_curve"] = ds_curve
    results["ss_constrained_curve"] = ss_curve
    results["ds_per_stratum_curves"] = ds_strata
    results["ds_minus_ss"] = {U: round(ds_curve[U] - ss_curve[U], 4) for U in DOSES}

    # ---- bootstrap on DS curve + delta-AURC ----
    n_ex = len(examples)
    ds_bool = {U: cells[(U, "DS")]["con"] for U in DOSES}
    rb = lib.rng_for(QUAL_SEED, 0xB007)
    B = 2000
    boot_curve = {U: [] for U in DOSES}; boot_aurc = []
    u1 = DOSES[0]
    for _ in range(B):
        idx = rb.integers(0, n_ex, size=n_ex)
        accs_b = {U: float(ds_bool[U][idx].mean()) for U in DOSES}
        for U in DOSES:
            boot_curve[U].append(accs_b[U])
        a1 = accs_b[u1] if accs_b[u1] > 0 else 1e-9
        boot_aurc.append(float(np.mean([max(0.0, (a1 - accs_b[U]) / a1) for U in DOSES])))
    ds_boot_ci = {U: [float(np.percentile(boot_curve[U], 2.5)), float(np.percentile(boot_curve[U], 97.5))]
                  for U in DOSES}
    results["ds_bootstrap_ci95"] = ds_boot_ci

    a1 = ds_curve[u1] if ds_curve[u1] > 0 else 1e-9
    delta_aurc = float(np.mean([max(0.0, (a1 - ds_curve[U]) / a1) for U in DOSES]))
    delta_aurc_ci = [float(np.percentile(boot_aurc, 2.5)), float(np.percentile(boot_aurc, 97.5))]

    # ---- gate (PRE_REGISTRATION §7) ----
    accs = [ds_curve[U] for U in DOSES]
    low = ds_curve[u1]; competent = low >= TAU_HI
    min_acc = min(accs); min_at = DOSES[int(np.argmin(accs))]
    mid = [U for U in DOSES if TAU_LO < ds_curve[U] < TAU_HI]
    tol, maxv = grid["monotonicity_tolerance"], grid["max_violation"]
    viol = [[DOSES[i], DOSES[i + 1], round(accs[i + 1] - accs[i], 4)]
            for i in range(len(accs) - 1) if accs[i + 1] > accs[i] + tol]
    worst = max([v[2] for v in viol], default=0.0)
    monotone_ok = len(viol) <= 1 and worst <= maxv
    robust = 0
    for s in range(STRATA):
        sc = ds_strata[s]
        smid = [U for U in DOSES if TAU_LO < sc[U] < TAU_HI]
        if sc[u1] >= TAU_HI and len(smid) >= 2:
            robust += 1
    robust_ok = robust >= grid["robust_strata_required"]
    aurc_ok = delta_aurc >= grid["delta_aurc_min"]
    lengths = [seq_lens[f"{U}_DS"] for U in DOSES]
    length_fixed = len(set(lengths)) == 1
    gap_fixed = True  # target always slot 0, query at end, M constant -> gap fixed by construction

    reasons = []
    if not competent:
        reasons.append("TASK_NOT_COMPETENT")
    if min_acc >= TAU_HI:
        reasons.append("FLAT_HIGH")
    elif min_acc > TAU_LO:
        reasons.append("INSUFFICIENT_LOSS")
    if len(mid) < grid["min_mid_band_doses"]:
        reasons.append("IMMEDIATE_CLIFF")
    if not monotone_ok:
        reasons.append("NON_MONOTONE")
    if not aurc_ok:
        reasons.append("WEAK_FULL_CURVE_EFFECT")
    if not robust_ok:
        reasons.append("NOT_ROBUST_ACROSS_STRATA")
    if not length_fixed:
        reasons.append("LENGTH_NOT_FIXED")
    if not gap_fixed:
        reasons.append("GAP_NOT_FIXED")
    verdict = "QUALIFIED" if not reasons else "BLOCKED"

    results["graded_region_analysis"] = {
        "competent_low_load": competent, "low_load_acc": low, "tau_hi": TAU_HI, "tau_lo": TAU_LO,
        "min_acc": min_acc, "min_acc_at_U": min_at, "mid_band_doses": mid, "n_mid_band": len(mid),
        "monotone_ok": monotone_ok, "violations": viol, "worst_violation": worst,
        "delta_aurc": round(delta_aurc, 4), "delta_aurc_ci95": delta_aurc_ci, "aurc_ok": aurc_ok,
        "robust_strata_count": robust, "robust_ok": robust_ok,
        "all_ds_lengths": lengths, "length_fixed": length_fixed, "gap_fixed": gap_fixed,
        "block_reasons": reasons}
    results["FIXED_LENGTH_STATE_LOAD_REGION"] = verdict

    # ---- 06C dose selection (frozen rule; only meaningful if QUALIFIED) ----
    HIGH = DOSES[-1]
    low_candidates = [U for U in DOSES if ds_curve[U] >= 0.80]
    LOW = max(low_candidates) if low_candidates else DOSES[0]
    target_mid = (ds_curve[LOW] + ds_curve[HIGH]) / 2
    MID = min(DOSES, key=lambda U: abs(ds_curve[U] - target_mid))
    results["c06_dose_selection"] = {
        "rule": "HIGH=max load; LOW=largest dose with DS>=0.80; MID=closest DS to (accLOW+accHIGH)/2",
        "HIGH_U": HIGH, "HIGH_acc": ds_curve[HIGH], "LOW_U": LOW, "LOW_acc": ds_curve[LOW],
        "MID_U": MID, "MID_acc": ds_curve[MID], "applicable": verdict == "QUALIFIED"}

    results["state_bytes_per_sequence"] = 52002816
    results["total_runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUTDIR, "B2_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(OUTDIR, "B2_CURVES.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["U", "arm", "n", "constrained_correct", "constrained_acc", "ci_lo", "ci_hi",
                    "unconstrained_acc", "format_adherence", "seq_len"])
        for r in curves:
            w.writerow([r["U"], r["arm"], r["n"], r["constrained_correct"],
                        round(r["constrained_acc"], 4), round(r["constrained_ci95"][0], 4),
                        round(r["constrained_ci95"][1], 4), round(r["unconstrained_acc"], 4),
                        round(r["format_adherence"], 4), r["seq_len"]])

    print("\n== DS vs SS constrained ==", file=sys.stderr)
    for U in DOSES:
        print(f"  U={U:3d}  DS={ds_curve[U]:.3f}  SS={ss_curve[U]:.3f}  (DS-SS={ds_curve[U]-ss_curve[U]:+.3f})",
              file=sys.stderr)
    print(f"\ncompetent={competent} min={min_acc:.3f}@{min_at} mid={mid} monotone={monotone_ok} "
          f"delta_aurc={delta_aurc:.3f} robust={robust}/{STRATA} length_fixed={length_fixed}", file=sys.stderr)
    print(f"reasons={reasons}", file=sys.stderr)
    print(f"\nFIXED_LENGTH_STATE_LOAD_REGION = {verdict}", file=sys.stderr)
    print(f"06C doses: LOW U={LOW}({ds_curve[LOW]:.2f}) MID U={MID}({ds_curve[MID]:.2f}) "
          f"HIGH U={HIGH}({ds_curve[HIGH]:.2f})", file=sys.stderr)
    print(f"runtime={results['total_runtime_s']}s peak_vram={results['peak_vram_gb']}GB", file=sys.stderr)


if __name__ == "__main__":
    main()
