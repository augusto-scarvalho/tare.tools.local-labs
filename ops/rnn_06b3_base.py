#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION — confirmatory runner.

Order-stable, subpacked, fixed-length (M=192, 770 tok) unique-state-load sweep. DS primary + SS
diagnostic. Enforces & counts: nested binding-identity invariant (all examples, all adjacent
dose pairs), fixed length, fixed gap, positive sentinel reserve (no U=M). Paired example-level
analysis + stratified bootstrap. Mints STATE_LOAD_FORGETTING_PERTURBATION + TRANSITION_SHAPE and
applies the frozen B3->06C dose rule if QUALIFIED. Self-records executed-source identity +
re-verifies b3 shas before outcomes. cs=32. No training, no push.
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
import rnn_06b3_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION")
SPEC_PATH = os.path.join(OUTDIR, "B3_QUALIFICATION_SPEC.json")
GRID_PATH = os.path.join(OUTDIR, "B3_STRESS_GRID.json")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
AUTOBATCH = 1560          # -> batch 2 at 770 tokens
QUAL_SEED = 20260817


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
    libpath = os.path.join(os.path.dirname(runner), "rnn_06b3_lib.py")
    b2libpath = os.path.join(os.path.dirname(runner), "rnn_06b2_lib.py")
    results = {"packet": "RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION",
               "kind": "controlled_state_load_perturbation_results"}

    spec = json.load(open(SPEC_PATH)); grid = json.load(open(GRID_PATH))
    q_rec = spec.pop("b3QualificationSetSha256"); disj = spec.pop("disjointness_proof")
    q_re = lib.sha256_of_obj(spec)
    g_rec = grid.pop("b3StressGridSha256"); g_re = lib.sha256_of_obj(grid)
    results["challenge_identities"] = {"b3QualificationSetSha256_recorded": q_rec,
                                       "b3QualificationSetSha256_recomputed": q_re,
                                       "qual_sha_match": q_rec == q_re,
                                       "b3StressGridSha256_recorded": g_rec,
                                       "b3StressGridSha256_recomputed": g_re,
                                       "stress_sha_match": g_rec == g_re,
                                       "disjointness_proof": disj}
    assert q_rec == q_re and g_rec == g_re, "B3 spec/grid sha mismatch"

    from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
    results["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_sha256": sha256_file(libpath), "lib_git_blob": git("hash-object", libpath),
        "lib_dirty": git("status", "--porcelain", "--", libpath),
        "b2lib_sha256": sha256_file(b2libpath),
        "git_head": git("rev-parse", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "configuration_mamba2_sha256": sha256_file(configuration_mamba2.__file__),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "repo_id": REPO_ID, "revision": REVISION, "chunk_size": CHUNK, "dtype": str(DTYPE),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "b3QualificationSetSha256": q_rec, "b3StressGridSha256": g_rec,
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

    M = grid["M"]; RESERVE = grid["reserve"]; DOSES = grid["doses"]; ARMS = grid["arms"]
    STRATA = grid["s_strata"]; TAU_HI = grid["tau_hi"]; SESOI = grid["sesoi"]
    U_LOW, U_HIGH = grid["u_low"], grid["u_high"]
    examples = spec["examples"]; ex_stratum = np.array([e["stratum"] for e in examples])

    # ---- counters / invariants (§9) ----
    counters = {"examplesEvaluated": 0, "cellsEvaluated": 0, "DSCells": 0, "SSCells": 0,
                "nestedBindingIdentityChecks": 0, "nestedBindingIdentityFailures": 0,
                "fixedLengthChecks": 0, "fixedGapChecks": 0,
                "uniqueBindingsMaterializedByDose": {}, "sentinelSlotsByDose": {}}

    # nested-identity invariant for ALL examples, ALL adjacent dose pairs, both arms (§2)
    nested_fail_detail = None
    for e in examples:
        for arm in ARMS:
            ok, detail = lib.nested_identity_check(e, M, DOSES, arm, pools)
            counters["nestedBindingIdentityChecks"] += 1
            if not ok:
                counters["nestedBindingIdentityFailures"] += 1
                nested_fail_detail = detail
    results["nestedBindingIdentityCheck"] = ("PASS" if counters["nestedBindingIdentityFailures"] == 0
                                             else "FAIL")
    results["nested_fail_detail"] = nested_fail_detail

    # ---- sweep ----
    torch.cuda.reset_peak_memory_stats()
    cells = {}; curves = []; per_cell_prompt_sha = {}; seq_lens = {}
    target_token_ok = True
    for U in DOSES:
        for arm in ARMS:
            prompts, golds = [], []
            for e in examples:
                p, g = lib.materialize_b3(e, M, U, arm, pools, RESERVE)
                prompts.append(p); golds.append(g)
            seq_len = len(prompts[0])
            assert all(len(p) == seq_len for p in prompts), "ragged length"
            counters["fixedLengthChecks"] += 1
            # fixed gap: target key token at position 0, query key token at position 4*M
            tgt_tok = pools["scored_keys"][examples[0]["target_key_slot"]]
            if not (prompts[0][0] == tgt_tok and prompts[0][4 * M] == tgt_tok):
                target_token_ok = False
            counters["fixedGapChecks"] += 1
            seq_lens[f"{U}_{arm}"] = seq_len
            counters["uniqueBindingsMaterializedByDose"][f"{U}_{arm}"] = U - 1
            counters["sentinelSlotsByDose"][f"{U}_{arm}"] = M - U
            bs = max(1, min(64, AUTOBATCH // seq_len))
            per_cell_prompt_sha[f"{U}_{arm}"] = lib.sha256_of_obj(prompts)
            tc = time.time()
            con, unc, fmt = eval_cell(model, prompts, golds, vtensor, vset, bs)
            cells[(U, arm)] = {"con": con, "unc": unc, "fmt": fmt}
            counters["cellsEvaluated"] += 1
            counters["DSCells" if arm == "DS" else "SSCells"] += 1
            counters["examplesEvaluated"] += len(con)
            k = int(con.sum()); n = len(con)
            strat = {int(s): float(con[ex_stratum == s].mean()) for s in range(STRATA)}
            row = {"U": U, "arm": arm, "seq_len": seq_len, "sentinel_slots": M - U,
                   "unique_bindings": U - 1, "n": n, "constrained_correct": k,
                   "constrained_acc": k / n, "constrained_ci95": wilson(k, n),
                   "unconstrained_acc": float(unc.mean()), "format_adherence": float(fmt.mean()),
                   "per_stratum_constrained_acc": strat, "batch": bs,
                   "eval_seconds": round(time.time() - tc, 2)}
            curves.append(row)
            print(f"[06B3] U={U:3d} {arm} sent={M-U:3d} len={seq_len} bs={bs} "
                  f"con={row['constrained_acc']:.3f} unc={row['unconstrained_acc']:.3f} "
                  f"({row['eval_seconds']}s)", file=sys.stderr)
    results["curves"] = curves
    results["per_cell_prompt_sha256"] = per_cell_prompt_sha
    counters["fixedGapCheck_targetTokenConsistent"] = target_token_ok
    results["construction_counters"] = counters
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

    # ---- paired analysis U_low vs U_high (DS) ----
    lo = cells[(U_LOW, "DS")]["con"]; hi = cells[(U_HIGH, "DS")]["con"]
    paired_loss = float(lo.mean() - hi.mean())
    lo_correct_hi_wrong = int(np.sum(lo & ~hi))
    lo_wrong_hi_correct = int(np.sum(~lo & hi))
    n_ex = len(examples)
    # stratified cluster bootstrap
    rb = lib.rng_for(QUAL_SEED, 0xB3B007)
    B = 2000
    boot_loss, boot_strat_ok = [], []
    strat_idx = {s: np.where(ex_stratum == s)[0] for s in range(STRATA)}
    for _ in range(B):
        idx = np.concatenate([si[rb.integers(0, len(si), size=len(si))] for si in strat_idx.values()])
        boot_loss.append(float(lo[idx].mean() - hi[idx].mean()))
    loss_ci = [float(np.percentile(boot_loss, 2.5)), float(np.percentile(boot_loss, 97.5))]
    # per-stratum paired loss
    strat_loss = {int(s): float(cells[(U_LOW, "DS")]["con"][ex_stratum == s].mean()
                                - cells[(U_HIGH, "DS")]["con"][ex_stratum == s].mean())
                  for s in range(STRATA)}
    robust_strata = sum(1 for s in range(STRATA) if strat_loss[s] >= SESOI)
    results["paired_primary"] = {
        "U_low": U_LOW, "U_high": U_HIGH, "acc_low": ds_curve[U_LOW], "acc_high": ds_curve[U_HIGH],
        "paired_loss": round(paired_loss, 4), "paired_loss_ci95_stratified_boot": loss_ci,
        "discordant_low_correct_high_wrong": lo_correct_hi_wrong,
        "discordant_low_wrong_high_correct": lo_wrong_hi_correct,
        "per_stratum_paired_loss": strat_loss, "robust_strata_count": robust_strata}

    # ---- descriptive curve stats (§7) ----
    a1 = ds_curve[U_LOW] if ds_curve[U_LOW] > 0 else 1e-9
    mrrd = float(np.mean([max(0.0, (a1 - ds_curve[U]) / a1) for U in DOSES]))
    deficits = [1.0 - ds_curve[U] / a1 for U in DOSES]
    aurc = float(np.trapz(deficits, DOSES) / (DOSES[-1] - DOSES[0]))
    results["MEAN_RELATIVE_RETENTION_DEFICIT"] = round(mrrd, 4)
    results["DEFICIT_AURC_NORMALIZED"] = round(aurc, 4)

    # ---- transition shape (§6) ----
    accs = [ds_curve[U] for U in DOSES]
    acc_low = ds_curve[U_LOW]; min_acc = min(accs); total_loss = acc_low - min_acc
    interior = sum(1 for U in DOSES if (min_acc + 0.10) < ds_curve[U] < (acc_low - 0.10))
    steps = [accs[i] - accs[i + 1] for i in range(len(accs) - 1)]
    max_step = max(steps) if steps else 0.0
    if total_loss < SESOI:
        shape = "FLAT"
    elif max_step >= grid["shape_cliff_step_fraction"] * total_loss and interior < 2:
        shape = "CLIFF"
    elif interior >= 2 and max_step < grid["shape_cliff_step_fraction"] * total_loss:
        shape = "GRADED"
    else:
        shape = "MIXED"
    results["TRANSITION_SHAPE"] = shape
    results["shape_analysis"] = {"acc_low": acc_low, "min_acc": min_acc, "total_loss": round(total_loss, 4),
                                 "interior_doses": interior, "max_step": round(max_step, 4)}

    # ---- gate (§6) ----
    competent = ds_curve[U_LOW] >= TAU_HI
    material = paired_loss >= SESOI
    length_fixed = len(set(seq_lens[f"{U}_DS"] for U in DOSES)) == 1
    gap_fixed = target_token_ok
    nested_ok = counters["nestedBindingIdentityFailures"] == 0
    reserve_ok = min(M - U for U in DOSES) >= RESERVE
    robust_ok = robust_strata >= grid["robust_strata_required"]
    ci_excludes_trivial = loss_ci[0] > grid["ci_lower_bound_min"]
    reasons = []
    if not competent:
        reasons.append("TASK_NOT_COMPETENT")
    if not material:
        reasons.append("PAIRED_LOSS_BELOW_SESOI")
    if not length_fixed:
        reasons.append("LENGTH_NOT_FIXED")
    if not gap_fixed:
        reasons.append("GAP_NOT_FIXED")
    if not nested_ok:
        reasons.append("NESTED_IDENTITY_FAIL")
    if not reserve_ok:
        reasons.append("PACKING_BOUNDARY")
    if not robust_ok:
        reasons.append("NOT_ROBUST_ACROSS_STRATA")
    if not ci_excludes_trivial:
        reasons.append("CI_INCLUDES_TRIVIAL")
    verdict = "QUALIFIED" if not reasons else "BLOCKED"
    results["gate_checks"] = {"competent": competent, "material_paired_loss": material,
                              "length_fixed": length_fixed, "gap_fixed": gap_fixed,
                              "nested_identity_pass": nested_ok, "reserve_ok": reserve_ok,
                              "robust_across_strata": robust_ok, "ci_excludes_trivial": ci_excludes_trivial,
                              "block_reasons": reasons}
    results["STATE_LOAD_FORGETTING_PERTURBATION"] = verdict

    # ---- 06C dose selection (frozen rule; §12) ----
    paired_losses = {U: ds_curve[U_LOW] - ds_curve[U] for U in DOSES}
    HIGH = max(DOSES, key=lambda U: paired_losses[U])           # max paired loss
    low_c = [U for U in DOSES if ds_curve[U] >= TAU_HI]
    LOW = max(low_c) if low_c else U_LOW
    mid_c = [U for U in DOSES if paired_losses[U] >= SESOI]
    MID = min(mid_c) if mid_c else HIGH
    results["c06_dose_selection"] = {
        "rule": "HIGH=max paired loss; LOW=highest dose acc>=0.75; MID=smallest dose paired_loss>=SESOI",
        "HIGH_U": HIGH, "HIGH_acc": ds_curve[HIGH], "LOW_U": LOW, "LOW_acc": ds_curve[LOW],
        "MID_U": MID, "MID_acc": ds_curve[MID], "applicable": verdict == "QUALIFIED"}
    results["state_bytes_per_sequence"] = 52002816
    results["total_runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUTDIR, "B3_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(os.path.join(OUTDIR, "B3_CURVES.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["U", "arm", "sentinel_slots", "unique_bindings", "n", "constrained_correct",
                    "constrained_acc", "ci_lo", "ci_hi", "unconstrained_acc", "format_adherence", "seq_len"])
        for r in curves:
            w.writerow([r["U"], r["arm"], r["sentinel_slots"], r["unique_bindings"], r["n"],
                        r["constrained_correct"], round(r["constrained_acc"], 4),
                        round(r["constrained_ci95"][0], 4), round(r["constrained_ci95"][1], 4),
                        round(r["unconstrained_acc"], 4), round(r["format_adherence"], 4), r["seq_len"]])

    print("\n== DS vs SS ==", file=sys.stderr)
    for U in DOSES:
        print(f"  U={U:3d} DS={ds_curve[U]:.3f} SS={ss_curve[U]:.3f}", file=sys.stderr)
    print(f"\npaired_loss(U{U_LOW}->U{U_HIGH})={paired_loss:.3f} CI={loss_ci} "
          f"discordant lo!hi={lo_correct_hi_wrong} hi!lo={lo_wrong_hi_correct} "
          f"robust={robust_strata}/{STRATA}", file=sys.stderr)
    print(f"MRRD={mrrd:.3f} DEFICIT_AURC={aurc:.3f} shape={shape} nested={results['nestedBindingIdentityCheck']}",
          file=sys.stderr)
    print(f"reasons={reasons}", file=sys.stderr)
    print(f"\nSTATE_LOAD_FORGETTING_PERTURBATION = {verdict}  TRANSITION_SHAPE = {shape}", file=sys.stderr)
    print(f"06C doses: LOW U={LOW}({ds_curve[LOW]:.2f}) MID U={MID}({ds_curve[MID]:.2f}) "
          f"HIGH U={HIGH}({ds_curve[HIGH]:.2f})", file=sys.stderr)
    print(f"runtime={results['total_runtime_s']}s peak_vram={results['peak_vram_gb']}GB", file=sys.stderr)


if __name__ == "__main__":
    main()
