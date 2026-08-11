#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06B-MAMBA-BASE — confirmatory graded-region qualification runner.

Executes ONLY because CONTINUATION_LIFECYCLE = QUALIFIED (RNN-06A2). Single-token MQAR
associative recall on the exact frozen subject AntonV/mamba2-1.3b-hf @703e19a4,
transformers-native naive bf16 backend, chunk_size=256. Two matched conditions (MP =
memory-pressure vs LC = length/interference control) at identical length/position/gap,
three outcome channels (constrained / unconstrained / format), S=3 seed strata, N=192 per
(dose,condition). Mints exactly one FIXED_BACKBONE_GRADED_REGION per PRE_REGISTRATION §7.

Self-records executed-source identity + re-verifies qualificationSetSha256 / stressGridSha256
before outcomes. No training, no push, no Memory Caching, no historical-state, no RNN-06C.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import Mamba2ForCausalLM, AutoTokenizer

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06B-MAMBA-BASE")
SPEC_PATH = os.path.join(OUTDIR, "QUALIFICATION_SPEC.json")
GRID_PATH = os.path.join(OUTDIR, "STRESS_GRID.json")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE = "cuda"
DTYPE = torch.bfloat16
# Train AMENDMENT 1: pinned SSD chunk-tiling = 32 (memory-feasible for the naive torch_forward
# whose G_intermediate is O(chunk_size^2); cs=256 needs ~45 GiB/seq at MQAR lengths on 24 GiB).
# Same value 06A2 is re-qualified at; not a backend change.
PINNED_CHUNK_SIZE = int(os.environ.get("RNN_CHUNK_SIZE", "32"))
AUTOBATCH_BUDGET = 1536
MASTER_SEED = 20260813


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"<git-error:{e}>"


def rng_for(*ints):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(ints))))


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


# ---------------- token pools: 4 disjoint single-token pools ----------------
def build_pools(tok, pool_size, filler_size, seed):
    vocab = int(getattr(tok, "vocab_size", 0) or len(tok))
    num_re = re.compile(r"^\s?\d{2,4}$")
    word_re = re.compile(r"^\s?[A-Za-z]{3,}$")
    num_ids, word_ids = [], []
    for tid in range(vocab):
        try:
            s = tok.decode([tid])
        except Exception:
            continue
        if num_re.match(s):
            num_ids.append(tid)
        elif word_re.match(s):
            word_ids.append(tid)

    def keep_single(ids):
        seen, out = set(), []
        for tid in ids:
            s = tok.decode([tid])
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1 and enc[0] == tid and s not in seen:
                seen.add(s)
                out.append(tid)
        return out

    num_ids = keep_single(num_ids)
    word_ids = keep_single(word_ids)

    def perm(ids, salt):
        r = rng_for(seed, salt)
        return [int(x) for x in np.array(ids)[r.permutation(len(ids))]]

    words = perm(word_ids, 0xB0B)
    nums = perm(num_ids, 0xA11CE)
    need = pool_size + filler_size
    assert len(words) >= need and len(nums) >= need, \
        f"insufficient single-token pools: words={len(words)} nums={len(nums)} need={need}"
    scored_keys = words[:pool_size]
    filler_keys = words[pool_size:pool_size + filler_size]
    scored_vals = nums[:pool_size]
    filler_vals = nums[pool_size:pool_size + filler_size]

    def one(t):
        enc = tok.encode(t, add_special_tokens=False)
        return enc[-1] if enc else None
    seps = {"eq": one("="), "nl": one("\n")}
    # disjointness assertions
    sets = [set(scored_keys), set(filler_keys), set(scored_vals), set(filler_vals)]
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            assert not (sets[a] & sets[b]), "pool overlap"
    meta = {"n_num_single": len(num_ids), "n_word_single": len(word_ids),
            "scored_keys_sha256": sha256_of_obj(scored_keys),
            "scored_vals_sha256": sha256_of_obj(scored_vals),
            "filler_keys_sha256": sha256_of_obj(filler_keys),
            "filler_vals_sha256": sha256_of_obj(filler_vals),
            "eq_id": seps["eq"], "nl_id": seps["nl"],
            "sample_keys": [tok.decode([i]) for i in scored_keys[:5]],
            "sample_vals": [tok.decode([i]) for i in scored_vals[:5]]}
    return scored_keys, scored_vals, filler_keys, filler_vals, seps, meta


# ---------------- materialize one (example, dose, condition) ----------------
def materialize(ex, P, cond, sk, sv, fk, fv, seps):
    eq, nl = seps["eq"], seps["nl"]
    ks, vs = ex["key_slots"], ex["val_slots"]
    fks, fvs = ex["filler_key_slots"], ex["filler_val_slots"]
    pidx = ex["probe_index"]
    toks = []
    for j in range(P):
        if cond == "MP" or j == pidx:
            toks += [sk[ks[j]], eq, sv[vs[j]], nl]         # scored pair
        else:
            toks += [fk[fks[j]], eq, fv[fvs[j]], nl]       # filler pair (LC only)
    toks += [sk[ks[pidx]], eq]                              # query
    gold = sv[vs[pidx]]
    return toks, gold


@torch.no_grad()
def eval_cell(model, prompts, golds, value_id_tensor, value_id_set, batch_size):
    """Return per-example correctness arrays for the three channels."""
    n = len(prompts)
    con = np.zeros(n, dtype=bool)
    unc = np.zeros(n, dtype=bool)
    fmt = np.zeros(n, dtype=bool)
    for b in range(0, n, batch_size):
        chunk = prompts[b:b + batch_size]
        gchunk = golds[b:b + batch_size]
        ids = torch.tensor(chunk, device=DEVICE, dtype=torch.long)
        out = model(ids)
        last = out.logits[:, -1, :].float()
        uncon = last.argmax(dim=-1)
        sub = last.index_select(1, value_id_tensor)
        conv = value_id_tensor[sub.argmax(dim=-1)]
        for j in range(len(chunk)):
            g = int(gchunk[j])
            con[b + j] = int(conv[j]) == g
            unc[b + j] = int(uncon[j]) == g
            fmt[b + j] = int(uncon[j]) in value_id_set
    return con, unc, fmt


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    runner_path = os.path.abspath(__file__)
    results = {"packet": "RNN-06B-MAMBA-BASE", "kind": "graded_region_results"}

    # ----- load spec + grid, re-verify shas -----
    spec = json.load(open(SPEC_PATH))
    grid = json.load(open(GRID_PATH))
    q_recorded = spec.pop("qualificationSetSha256")
    disjoint_proof = spec.pop("disjointness_proof")
    q_recompute = sha256_of_obj(spec)
    g_recorded = grid.pop("stressGridSha256")
    g_recompute = sha256_of_obj(grid)
    results["challenge_identities"] = {
        "qualificationSetSha256_recorded": q_recorded,
        "qualificationSetSha256_recomputed": q_recompute,
        "qualification_sha_match": q_recorded == q_recompute,
        "stressGridSha256_recorded": g_recorded,
        "stressGridSha256_recomputed": g_recompute,
        "stress_sha_match": g_recorded == g_recompute,
        "disjointness_proof": disjoint_proof}
    assert q_recorded == q_recompute, "qualification spec sha mismatch"
    assert g_recorded == g_recompute, "stress grid sha mismatch"
    assert disjoint_proof["example_level_disjoint"] is True, "not disjoint from P0 calibration"

    DOSES = grid["doses"]; CONDS = grid["conditions"]
    TAU_HI, TAU_LO = grid["tau_hi"], grid["tau_lo"]
    LOW_DOSE = grid["low_dose"]; STRATA = grid["s_strata"]

    # ----- executed-source identity (before outcomes) -----
    from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
    identity = {
        "runner_file": runner_path, "runner_source_sha256": sha256_file(runner_path),
        "runner_git_blob": git("hash-object", runner_path),
        "runner_git_tracked_dirty": git("status", "--porcelain", "--", runner_path),
        "git_head": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "configuration_mamba2_sha256": sha256_file(configuration_mamba2.__file__),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "repo_id": REPO_ID, "revision": REVISION, "dtype": str(DTYPE),
        "pinned_chunk_size": PINNED_CHUNK_SIZE,
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "qualificationSetSha256": q_recorded, "stressGridSha256": g_recorded,
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "PRE_REGISTRATION.md")),
    }
    results["executed_source_identity"] = identity
    assert identity["is_fast_path_available"] is False, "fast path unexpectedly available"
    print("[identity] is_fast_path_available:", identity["is_fast_path_available"], file=sys.stderr)

    # ----- load model + tokenizer (frozen) -----
    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION,
                                              torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in model.backbone.layers:
        blk.mixer.chunk_size = PINNED_CHUNK_SIZE
    model.config.chunk_size = PINNED_CHUNK_SIZE
    results["n_params"] = int(sum(p.numel() for p in model.parameters()))
    results["effective_chunk_size"] = [model.backbone.layers[0].mixer.chunk_size, model.config.chunk_size]
    results["tokenizer"] = {"class": type(tok).__name__,
                            "vocab_size": int(getattr(tok, "vocab_size", 0) or len(tok))}

    sk, sv, fk, fv, seps, pool_meta = build_pools(tok, spec["pool_size"],
                                                  spec["filler_pool_size"], MASTER_SEED)
    results["pools"] = pool_meta
    value_id_set = set(sv)
    value_id_tensor = torch.tensor(sorted(value_id_set), device=DEVICE, dtype=torch.long)
    results["chance"] = 1.0 / len(value_id_set)

    examples = spec["examples"]
    ex_stratum = np.array([e["stratum"] for e in examples])

    # ----- sweep -----
    torch.cuda.reset_peak_memory_stats()
    cells = {}                    # (dose,cond) -> per-example bool arrays
    curves = []
    per_dose_prompt_sha = {}
    for P in DOSES:
        for cond in CONDS:
            prompts, golds = [], []
            for ex in examples:
                p, g = materialize(ex, P, cond, sk, sv, fk, fv, seps)
                prompts.append(p); golds.append(g)
            seq_len = len(prompts[0])
            bs = max(1, min(64, AUTOBATCH_BUDGET // seq_len))
            per_dose_prompt_sha[f"{P}_{cond}"] = sha256_of_obj(prompts)
            tcell = time.time()
            con, unc, fmt = eval_cell(model, prompts, golds, value_id_tensor, value_id_set, bs)
            cells[(P, cond)] = {"con": con, "unc": unc, "fmt": fmt}
            n = len(con)
            k = int(con.sum())
            ci = wilson(k, n)
            strat_acc = {int(s): float(con[ex_stratum == s].mean()) for s in range(STRATA)}
            row = {"dose": P, "condition": cond, "seq_len": seq_len, "batch": bs, "n": n,
                   "constrained_correct": k, "constrained_acc": k / n,
                   "constrained_ci95": ci,
                   "unconstrained_acc": float(unc.mean()), "unconstrained_correct": int(unc.sum()),
                   "format_adherence": float(fmt.mean()), "format_correct": int(fmt.sum()),
                   "per_stratum_constrained_acc": strat_acc,
                   "eval_seconds": round(time.time() - tcell, 2)}
            curves.append(row)
            print(f"[06B] P={P:4d} {cond} len={seq_len:4d} bs={bs:2d} "
                  f"con={row['constrained_acc']:.3f} ci={ci[0]:.2f}-{ci[1]:.2f} "
                  f"unc={row['unconstrained_acc']:.3f} fmt={row['format_adherence']:.3f} "
                  f"({row['eval_seconds']}s)", file=sys.stderr)

    results["curves"] = curves
    results["per_dose_prompt_sha256"] = per_dose_prompt_sha
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # ----- pooled curves -----
    def acc(P, cond):
        c = cells[(P, cond)]["con"]; return int(c.sum()) / len(c)
    mp_curve = {P: acc(P, "MP") for P in DOSES}
    lc_curve = {P: acc(P, "LC") for P in DOSES}
    mp_strata = {s: {P: float(cells[(P, "MP")]["con"][ex_stratum == s].mean()) for P in DOSES}
                 for s in range(STRATA)}
    results["mp_constrained_curve"] = mp_curve
    results["lc_constrained_curve"] = lc_curve
    results["mp_per_stratum_curves"] = mp_strata

    # ----- cluster bootstrap over examples (pooled MP curve) -----
    n_ex = len(examples)
    mp_bool = {P: cells[(P, "MP")]["con"] for P in DOSES}
    B = 2000
    boot = {P: [] for P in DOSES}
    rb = rng_for(MASTER_SEED, 0xB007)
    for _ in range(B):
        idx = rb.integers(0, n_ex, size=n_ex)
        for P in DOSES:
            boot[P].append(float(mp_bool[P][idx].mean()))
    mp_boot_ci = {P: [float(np.percentile(boot[P], 2.5)), float(np.percentile(boot[P], 97.5))]
                  for P in DOSES}
    results["mp_bootstrap_ci95"] = mp_boot_ci

    # ----- graded-region classification (PRE_REGISTRATION §7) -----
    accs = [mp_curve[P] for P in DOSES]
    low = mp_curve[LOW_DOSE]
    competent = low >= TAU_HI
    min_acc = min(accs); min_at = DOSES[int(np.argmin(accs))]
    mid_band = [P for P in DOSES if TAU_LO < mp_curve[P] < TAU_HI]
    tol = grid["monotonicity_tolerance"]; max_viol = grid["max_violation"]
    violations = [[DOSES[i], DOSES[i + 1], round(accs[i + 1] - accs[i], 4)]
                  for i in range(len(accs) - 1) if accs[i + 1] > accs[i] + tol]
    worst_viol = max([v[2] for v in violations], default=0.0)
    monotone_ok = len(violations) <= 1 and worst_viol <= max_viol
    robust = 0
    for s in range(STRATA):
        sc = mp_strata[s]
        s_mid = [P for P in DOSES if TAU_LO < sc[P] < TAU_HI]
        if sc[LOW_DOSE] >= TAU_HI and len(s_mid) >= 2:
            robust += 1
    robust_ok = robust >= grid["robust_strata_required"]
    hi_doses = grid["confound_high_doses"]
    sep = float(np.mean([lc_curve[P] - mp_curve[P] for P in hi_doses]))
    confound_ok = sep >= grid["confound_min_separation"]

    reasons = []
    if not competent:
        reasons.append("TASK_NOT_COMPETENT")
    if min_acc >= TAU_HI:
        reasons.append("FLAT_HIGH")
    elif min_acc > TAU_LO:
        reasons.append("INSUFFICIENT_HIGH_PRESSURE_LOSS")
    if len(mid_band) < grid["min_mid_band_doses"]:
        reasons.append("IMMEDIATE_CLIFF_OR_INSUFFICIENT_INTERIOR")
    if not monotone_ok:
        reasons.append("NON_MONOTONE")
    if not robust_ok:
        reasons.append("NOT_ROBUST_ACROSS_STRATA")
    if not confound_ok:
        reasons.append("CONFOUNDED_WITH_LENGTH")
    verdict = "QUALIFIED" if not reasons else "BLOCKED"

    analysis = {
        "competent_low_dose": competent, "low_dose_acc": low, "tau_hi": TAU_HI, "tau_lo": TAU_LO,
        "min_acc": min_acc, "min_acc_at_dose": min_at, "mid_band_doses": mid_band,
        "n_mid_band": len(mid_band), "monotone_ok": monotone_ok, "violations": violations,
        "worst_violation": worst_viol, "robust_strata_count": robust, "robust_ok": robust_ok,
        "confound_high_doses": hi_doses, "confound_separation_LC_minus_MP": round(sep, 4),
        "confound_ok": confound_ok, "block_reasons": reasons}
    results["graded_region_analysis"] = analysis
    results["FIXED_BACKBONE_GRADED_REGION"] = verdict
    if verdict == "QUALIFIED":
        results["qualified_pressure_region"] = {
            "mid_band_doses": mid_band,
            "competent_dose": LOW_DOSE, "degraded_dose": min_at}

    # ----- state economics carry-forward (PRE_REGISTRATION §11) -----
    state_bytes = 52002816
    region = mid_band if mid_band else DOSES
    results["state_economics"] = {
        "state_bytes_per_sequence": state_bytes,
        "note": "carried from RNN-06A/06A2; ssm-dominated bf16 recurrent cache",
        "derived_06c_snapshot_cost_estimate": {
            "per_snapshot_MiB": round(state_bytes / (1024 * 1024), 2),
            "example_region_doses": region,
            "n_qualification_examples": len(examples),
            "one_snapshot_per_example_region_GiB": round(
                state_bytes * len(examples) * max(1, len(region)) / (1024 ** 3), 3),
            "caveat": "rough upper-bound if 06C snapshotted every example at every region dose; "
                      "06C cadence NOT designed here"}}

    results["total_runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUTDIR, "BASE_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    # CSV
    import csv
    with open(os.path.join(OUTDIR, "BASE_CURVES.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dose", "condition", "n", "constrained_correct", "constrained_acc",
                    "ci_lo", "ci_hi", "unconstrained_acc", "format_adherence", "seq_len"])
        for r in curves:
            w.writerow([r["dose"], r["condition"], r["n"], r["constrained_correct"],
                        round(r["constrained_acc"], 4), round(r["constrained_ci95"][0], 4),
                        round(r["constrained_ci95"][1], 4), round(r["unconstrained_acc"], 4),
                        round(r["format_adherence"], 4), r["seq_len"]])

    print("\n==== MP constrained curve ====", file=sys.stderr)
    for P in DOSES:
        print(f"  P={P:4d}  MP={mp_curve[P]:.3f}  LC={lc_curve[P]:.3f}  "
              f"(LC-MP={lc_curve[P]-mp_curve[P]:+.3f})", file=sys.stderr)
    print(f"\ncompetent={competent} min_acc={min_acc:.3f}@{min_at} mid_band={mid_band} "
          f"monotone={monotone_ok} robust={robust}/{STRATA} confound_sep={sep:+.3f}",
          file=sys.stderr)
    print(f"reasons={reasons}", file=sys.stderr)
    print(f"\nFIXED_BACKBONE_GRADED_REGION = {verdict}", file=sys.stderr)
    print(f"runtime={results['total_runtime_s']}s peak_vram={results['peak_vram_gb']}GB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
