#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06D0 — Recovery Ceiling & Snapshot Schedule qualification runner.

Reads the frozen D0_QUALIFICATION_SPEC.json + SNAPSHOT_SCHEDULE.json. For each held-out example
(randomized target-write slot; all-load DS body) it captures the target-agnostic historical snapshot
pool (K independent prefills at fixed interior boundaries) + FINAL, restores each with full temporal
identity, decodes the target query (constrained argmax), and computes:
  FINAL, FIXED_HISTORICAL_POOL, ORACLE_TARGET_PROXIMAL (diagnostic), ORACLE_BEST_GOLD (upper bound).
Mints RECOVERY_CEILING in {QUALIFIED|TOO_SMALL|NOT_TESTABLE} against thresholds frozen BEFORE
outcomes. Persists per-snapshot constrained logits (D0_READOUTS.npz) for the D1 substrate. No
recovery mechanism, no reader, no gold/target-position use in any FINAL/POOL arm (only ORACLE arms,
flagged diagnostic).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import Mamba2ForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06D")
SPEC_PATH = os.path.join(OUTDIR, "D0_QUALIFICATION_SPEC.json")
SCHED_PATH = os.path.join(OUTDIR, "SNAPSHOT_SCHEDULE.json")
B3_RESULTS = os.path.join(REPO, "runs", "rnn", "RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION",
                          "B3_RESULTS.json")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
POOL_SEED = 20260817
BATCH = 2
AUDIT_SAMPLE = 8
STATE_BYTES = 52002816

# Frozen thresholds (TRAIN_PROTOCOL; not tuned)
CEILING_SESOI = 0.15
FINAL_ACC_MAX = 0.75
TAU_PROX = 0.75
CI_LB_MIN = 0.05
ROBUST_MIN = 2
RECOV_FRAC_MIN = 0.30
RECOV_N_MIN = 20


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
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def tok_sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()


def state_sha(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def build_cache(model, bsz):
    return Mamba2Cache(model.config, bsz, dtype=DTYPE, device=model.device)


@torch.no_grad()
def prefill(model, cache, ids_2d):
    cp = torch.arange(0, ids_2d.shape[1], device=model.device)
    model(input_ids=ids_2d, cache_params=cache, cache_position=cp, use_cache=True)


@torch.no_grad()
def decode_token(model, cache, col_2d, offset):
    cp = torch.tensor([offset], device=model.device)
    return model(input_ids=col_2d, cache_params=cache, cache_position=cp, use_cache=True).logits


def snapshot(cache):
    return cache.conv_states.detach().clone(), cache.ssm_states.detach().clone()


def restore(model, conv, ssm, n):
    c = build_cache(model, n)
    c.conv_states.copy_(conv[:, :n]); c.ssm_states.copy_(ssm[:, :n])
    return c


def main():
    t0 = time.time(); os.makedirs(OUTDIR, exist_ok=True); torch.manual_seed(0)
    runner = os.path.abspath(__file__)
    libpath = os.path.join(os.path.dirname(runner), "rnn_06d_lib.py")
    results = {"packet": "RNN-06D0-RECOVERY-CEILING", "kind": "recovery_ceiling"}

    # ---- context: B3 forgetting regime (not a hard gate for D0; recorded) ----
    b3 = json.load(open(B3_RESULTS))
    results["b3_context"] = {"STATE_LOAD_FORGETTING_PERTURBATION": b3["STATE_LOAD_FORGETTING_PERTURBATION"]}

    spec = json.load(open(SPEC_PATH)); sched = json.load(open(SCHED_PATH))
    q_rec = spec["qualificationSetSha256"]
    spec_core = {k: spec[k] for k in ("kind", "generator_version", "master_seed", "M", "t_band",
                                      "s_strata", "n_per_stratum", "n_total", "examples")}
    q_re = lib.sha256_of_obj(spec_core)
    s_rec = sched["snapshotScheduleSha256"]
    s_re = lib.sha256_of_obj({k: sched[k] for k in ("K", "schedule_slots", "M", "final_slot",
                                                    "generator_version")})
    results["challenge_identities"] = {
        "qualificationSetSha256_recorded": q_rec, "qualificationSetSha256_recomputed": q_re,
        "qual_sha_match": q_rec == q_re, "snapshotScheduleSha256_recorded": s_rec,
        "snapshotScheduleSha256_recomputed": s_re, "sched_sha_match": s_rec == s_re,
        "disjointness_proof": spec.get("disjointness_proof")}
    assert q_rec == q_re, "D0 qual spec sha mismatch"
    assert s_rec == s_re, "D0 schedule sha mismatch"

    M = spec["M"]; schedule = sched["schedule_slots"]; K = sched["K"]; final_slot = sched["final_slot"]
    assert final_slot == M - 1 and schedule == lib.schedule_slots(M, K)

    from transformers.models.mamba2 import modeling_mamba2
    results["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_sha256": sha256_file(libpath), "lib_git_blob": git("hash-object", libpath),
        "lib_dirty": git("status", "--porcelain", "--", libpath),
        "git_head": git("rev-parse", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "repo_id": REPO_ID, "revision": REVISION, "chunk_size": CHUNK, "dtype": str(DTYPE),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "qualificationSetSha256": q_rec, "snapshotScheduleSha256": s_rec, "pool_seed": POOL_SEED,
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "PRE_REGISTRATION.md"))}
    assert modeling_mamba2.is_fast_path_available is False
    backend_id = f"transformers-native-naive-torch_forward-cs{CHUNK}-bf16"

    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in model.backbone.layers:
        blk.mixer.chunk_size = CHUNK
    model.config.chunk_size = CHUNK
    weights_id = hashlib.sha256(
        "|".join(f"{n}:{float(p.float().sum()):.4e}" for n, p in model.named_parameters()).encode()
    ).hexdigest()
    pools, pool_meta = lib.build_pools(tok, POOL_SEED)
    results["pools"] = pool_meta
    vset = sorted(set(pools["scored_vals"]))
    vtensor = torch.tensor(vset, device=DEVICE, dtype=torch.long); V = len(vset)
    results["chance"] = 1.0 / V

    examples = spec["examples"]; N = len(examples)
    strat = np.array([e["stratum"] for e in examples])
    prefixes, queries, golds, tslots = [], [], [], []
    for e in examples:
        toks, gold = lib.materialize_d0(e, M, pools)
        prefixes.append(toks); queries.append(toks[-2:]); golds.append(gold); tslots.append(e["target_slot"])
    golds = np.array(golds); tslots = np.array(tslots)

    counters = {"snapshotsCreated": 0, "snapshotsHashed": 0, "snapshotsRestored": 0,
                "poolReadouts": 0, "finalReadouts": 0, "oracleCalls": 0,
                "snapshotBoundaryChecks": 0, "snapshotBoundaryFailures": 0,
                "stateHashChecks": 0, "queriesEvaluated": 0}
    identity_records = []
    slots_all = list(schedule) + [final_slot]

    @torch.no_grad()
    def readout_at_slot(slot, role):
        """Capture snapshot@slot for all examples, restore, decode query. Returns
        (correct[N], top1[N], sublogits[N,V])."""
        plen = lib.prefix_len_for_slot(slot)
        correct = np.zeros(N, bool); top1 = np.zeros(N, float); sub_all = np.zeros((N, V), np.float32)
        for b in range(0, N, BATCH):
            rows = list(range(b, min(b + BATCH, N)))
            ids = torch.tensor([prefixes[r][:plen] for r in rows], device=DEVICE, dtype=torch.long)
            cache = build_cache(model, len(rows)); prefill(model, cache, ids)
            conv, ssm = snapshot(cache); counters["snapshotsCreated"] += len(rows)
            for li, r in enumerate(rows):
                csha = state_sha(conv[:, li]); ssha = state_sha(ssm[:, li])
                counters["snapshotsHashed"] += 1; counters["snapshotBoundaryChecks"] += 1
                if plen != len(prefixes[r][:plen]):
                    counters["snapshotBoundaryFailures"] += 1
                if sum(1 for x in identity_records if x["snapshotRole"] == role) < 9:
                    identity_records.append({
                        "exampleId": r, "snapshotRole": role, "snapshotOrdinal": slot,
                        "sequenceTokenPosition": plen, "cachePosition": plen,
                        "associationSlotPosition": int(tslots[r]),
                        "recurrenceBoundaryId": f"{role}_slot{slot}_len{plen}",
                        "prefixTokenSha256": tok_sha(prefixes[r][:plen]),
                        "convStateSha256": csha, "ssmStateSha256": ssha,
                        "combinedStateSha256": hashlib.sha256((csha + ssha).encode()).hexdigest(),
                        "modelRevision": REVISION, "modelWeightsIdentity": weights_id,
                        "backendSemanticsId": backend_id, "chunkSize": CHUNK, "dtype": str(DTYPE)})
            rcache = restore(model, conv, ssm, len(rows)); counters["snapshotsRestored"] += len(rows)
            q0 = torch.tensor([[queries[r][0]] for r in rows], device=DEVICE, dtype=torch.long)
            decode_token(model, rcache, q0, plen)
            q1 = torch.tensor([[queries[r][1]] for r in rows], device=DEVICE, dtype=torch.long)
            logits = decode_token(model, rcache, q1, plen + 1)[:, 0, :].float()
            sub = logits.index_select(1, vtensor)
            probs = torch.softmax(sub, dim=-1); pred = vtensor[sub.argmax(-1)]
            for li, r in enumerate(rows):
                correct[r] = int(pred[li]) == int(golds[r]); top1[r] = float(probs[li].max())
                sub_all[r] = sub[li].cpu().numpy(); counters["queriesEvaluated"] += 1
            del cache, rcache, conv, ssm; torch.cuda.empty_cache()
        return correct, top1, sub_all

    # ---- boundary self-check BEFORE outcomes: re-prefill reproduces state hashes ----
    selfcheck = {"checked": 0, "reproduced": 0, "failures": 0}
    for r in range(min(AUDIT_SAMPLE, N)):
        for slot in (schedule[0], final_slot):
            plen = lib.prefix_len_for_slot(slot)
            c1 = build_cache(model, 1); prefill(model, c1, torch.tensor([prefixes[r][:plen]], device=DEVICE))
            h1 = (state_sha(c1.conv_states[:, 0]), state_sha(c1.ssm_states[:, 0]))
            c2 = build_cache(model, 1); prefill(model, c2, torch.tensor([prefixes[r][:plen]], device=DEVICE))
            h2 = (state_sha(c2.conv_states[:, 0]), state_sha(c2.ssm_states[:, 0]))
            counters["stateHashChecks"] += 1; selfcheck["checked"] += 1
            if h1 == h2:
                selfcheck["reproduced"] += 1
            else:
                selfcheck["failures"] += 1
            del c1, c2
    results["boundary_selfcheck"] = selfcheck
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # ---- cost micro-benchmark (warm) on one batch at FINAL ----
    def timed(fn, iters=5):
        torch.cuda.synchronize(); fn()  # warmup (cold, discarded)
        torch.cuda.synchronize(); s = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(); return (time.time() - s) / iters
    rows = list(range(min(BATCH, N))); plen = 4 * M
    ids = torch.tensor([prefixes[r][:plen] for r in rows], device=DEVICE, dtype=torch.long)
    cost = {}
    _cap = {}
    def cap():
        c = build_cache(model, len(rows)); prefill(model, c, ids); _cap["c"] = c
    cost["capture_prefill_final_s_per_batch"] = timed(cap)
    conv, ssm = snapshot(_cap["c"])
    def host():
        _ = conv.cpu(), ssm.cpu()
    cost["gpu_to_host_transfer_s_per_batch"] = timed(host)
    def rest():
        _cap["r"] = restore(model, conv, ssm, len(rows))
    cost["restore_s_per_batch"] = timed(rest)
    rc = _cap["r"]
    q0 = torch.tensor([[queries[r][0]] for r in rows], device=DEVICE, dtype=torch.long)
    q1 = torch.tensor([[queries[r][1]] for r in rows], device=DEVICE, dtype=torch.long)
    def rdo():
        rc2 = restore(model, conv, ssm, len(rows))
        decode_token(model, rc2, q0, plen); decode_token(model, rc2, q1, plen + 1)
    cost["restore_plus_readout_s_per_batch"] = timed(rdo)
    del conv, ssm; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # ---- main readouts ----
    pool_c = np.zeros((N, K), bool); pool_p = np.zeros((N, K), float); pool_logits = np.zeros((N, K, V), np.float32)
    for ki, slot in enumerate(schedule):
        c, p, sub = readout_at_slot(slot, f"POOL{ki}")
        pool_c[:, ki] = c; pool_p[:, ki] = p; pool_logits[:, ki] = sub
    counters["poolReadouts"] = int(counters["queriesEvaluated"])
    q_after_pool = counters["queriesEvaluated"]
    final_c, final_p, final_logits = readout_at_slot(final_slot, "FINAL")
    counters["finalReadouts"] = counters["queriesEvaluated"] - q_after_pool
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # ---- arms ----
    final_acc = float(final_c.mean())
    oracle_best = pool_c.any(axis=1); counters["oracleCalls"] += N
    ob_acc = float(oracle_best.mean())
    prox = [lib.proximal_snapshot_index(schedule, int(t)) for t in tslots]
    prox_idx = np.array([p[0] for p in prox]); prox_post = np.array([p[1] for p in prox])
    proximal_correct = pool_c[np.arange(N), prox_idx]; counters["oracleCalls"] += N
    px_acc = float(proximal_correct.mean())
    pool_slot_acc = {int(schedule[ki]): float(pool_c[:, ki].mean()) for ki in range(K)}

    # coverage
    post_mask = np.stack([np.array(lib.post_target_mask(schedule, int(t))) for t in tslots])  # (N,K)
    coverage = {"pairs_post_target_frac": round(float(post_mask.mean()), 4),
                "examples_with_any_post_target": int((post_mask.any(1)).sum()),
                "examples_with_any_post_target_correct": int(((pool_c & post_mask).any(1)).sum()),
                "mean_post_target_top1": round(float(pool_p[post_mask].mean()), 4) if post_mask.any() else None,
                "mean_pre_target_top1": round(float(pool_p[~post_mask].mean()), 4) if (~post_mask).any() else None}

    # paired OB - FINAL, stratified cluster bootstrap
    rb = np.random.Generator(np.random.PCG64(np.random.SeedSequence([spec["master_seed"], 0xCE111])))
    strat_idx = {s: np.where(strat == s)[0] for s in range(spec["s_strata"])}
    boot = []
    for _ in range(2000):
        idx = np.concatenate([si[rb.integers(0, len(si), size=len(si))] for si in strat_idx.values()])
        boot.append(float(oracle_best[idx].mean() - final_c[idx].mean()))
    ob_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    per_strat = {int(s): float(oracle_best[strat == s].mean() - final_c[strat == s].mean())
                 for s in range(spec["s_strata"])}
    robust = sum(1 for s in per_strat if per_strat[s] >= CEILING_SESOI)

    # recoverable substrate
    fw = ~final_c; n_fw = int(fw.sum())
    n_recov = int((fw & oracle_best).sum()); recov_frac = (n_recov / n_fw) if n_fw else 0.0

    results["arms"] = {
        "FINAL": {"acc": final_acc, "correct": int(final_c.sum()), "n": N, "ci95": wilson(int(final_c.sum()), N)},
        "ORACLE_BEST_GOLD": {"acc": ob_acc, "correct": int(oracle_best.sum()), "ci95": wilson(int(oracle_best.sum()), N),
                             "note": "diagnostic upper bound; uses gold"},
        "ORACLE_TARGET_PROXIMAL": {"acc": px_acc, "correct": int(proximal_correct.sum()),
                                   "ci95": wilson(int(proximal_correct.sum()), N),
                                   "post_target_frac": round(float(prox_post.mean()), 4),
                                   "note": "diagnostic; uses target position"},
        "FIXED_HISTORICAL_POOL": {"per_slot_acc": pool_slot_acc, "mean_slot_acc": round(float(pool_c.mean()), 4),
                                  "note": "substrate for D1"}}
    results["paired_ceiling"] = {
        "oracle_best_minus_final": round(ob_acc - final_acc, 4), "ci95": ob_ci,
        "per_stratum": per_strat, "robust_strata": robust,
        "proximal_minus_final": round(px_acc - final_acc, 4)}
    results["recoverable_substrate"] = {"n_final_wrong": n_fw, "n_recoverable": n_recov,
                                        "recovery_ceiling_frac": round(recov_frac, 4)}
    results["coverage"] = coverage
    results["construction_counters"] = counters
    results["identity_records_sample"] = identity_records[:40]
    results["state_bytes_per_sequence"] = STATE_BYTES
    results["state_bytes_times_K"] = STATE_BYTES * K
    results["cost_profile"] = {k: round(v, 5) for k, v in cost.items()}

    # ---- gate ----
    boundary_ok = counters["snapshotBoundaryFailures"] == 0 and selfcheck["failures"] == 0
    counters_ok = all(counters[k] > 0 for k in ["snapshotsCreated", "snapshotsRestored",
                      "poolReadouts", "finalReadouts", "queriesEvaluated"])
    final_degraded = final_acc <= FINAL_ACC_MAX
    prox_competent = px_acc >= TAU_PROX
    effect = (ob_acc - final_acc) >= CEILING_SESOI
    ci_ok = ob_ci[0] > CI_LB_MIN
    robust_ok = robust >= ROBUST_MIN
    recov_ok = recov_frac >= RECOV_FRAC_MIN and n_recov >= RECOV_N_MIN
    if not boundary_ok or not counters_ok:
        verdict, reason = "NOT_TESTABLE", "INVALID_MACHINERY"
    elif not final_degraded or not prox_competent:
        verdict, reason = "NOT_TESTABLE", ("FINAL_NOT_DEGRADED" if not final_degraded
                                           else "PROXIMAL_NOT_COMPETENT")
    elif effect and ci_ok and robust_ok and recov_ok:
        verdict, reason = "QUALIFIED", "OK"
    else:
        verdict, reason = "TOO_SMALL", "CEILING_BELOW_THRESHOLD_OR_INSUFFICIENT_SUBSTRATE"
    results["gate_checks"] = {"boundary_ok": boundary_ok, "counters_ok": counters_ok,
                              "final_degraded": final_degraded, "proximal_competent": prox_competent,
                              "effect_ge_sesoi": effect, "ci_lb_gt": ci_ok, "robust_ok": robust_ok,
                              "recoverable_ok": recov_ok, "reason": reason}
    results["RECOVERY_CEILING"] = verdict
    results["d1_status"] = "OPEN" if verdict == "QUALIFIED" else "BLOCKED_BY_D0"
    results["total_runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUTDIR, "RECOVERY_CEILING_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    np.savez_compressed(os.path.join(OUTDIR, "D0_READOUTS.npz"),
                        final_logits=final_logits, pool_logits=pool_logits, final_correct=final_c,
                        pool_correct=pool_c, target_slots=tslots, golds=golds, strata=strat,
                        schedule=np.array(schedule), vset=np.array(vset))
    with open(os.path.join(OUTDIR, "SNAPSHOT_IDENTITY_SAMPLE.json"), "w") as f:
        json.dump({"records": identity_records[:40], "boundary_selfcheck": selfcheck,
                   "counters": counters}, f, indent=2, default=str)
    with open(os.path.join(OUTDIR, "MECHANISM_ACTIVATION_D0.json"), "w") as f:
        json.dump(counters, f, indent=2)
    with open(os.path.join(OUTDIR, "COST_PROFILE_D0.json"), "w") as f:
        json.dump({"cost_profile_s_per_batch": results["cost_profile"], "batch": BATCH, "K": K,
                   "state_bytes_per_sequence": STATE_BYTES, "state_bytes_times_K": STATE_BYTES * K,
                   "peak_vram_gb": results["peak_vram_gb"]}, f, indent=2)
    # curves
    with open(os.path.join(OUTDIR, "D0_CURVES.csv"), "w") as f:
        f.write("slot,role,acc\n")
        for ki, s in enumerate(schedule):
            f.write(f"{s},POOL{ki},{pool_slot_acc[int(s)]:.4f}\n")
        f.write(f"{final_slot},FINAL,{final_acc:.4f}\n")

    print(f"\nFINAL={final_acc:.3f} ORACLE_BEST={ob_acc:.3f} PROX={px_acc:.3f} "
          f"OB-FINAL={ob_acc - final_acc:.3f} CI={ob_ci}", file=sys.stderr)
    print(f"recoverable: {n_recov}/{n_fw} ({recov_frac:.2f})  robust={robust}/3 "
          f"coverage_post={coverage['pairs_post_target_frac']}", file=sys.stderr)
    print(f"boundary_ok={boundary_ok} final_degraded={final_degraded} prox_competent={prox_competent}",
          file=sys.stderr)
    print(f"RECOVERY_CEILING = {verdict} ({reason})  d1={results['d1_status']}", file=sys.stderr)
    print(f"runtime={results['total_runtime_s']}s peak_vram={results['peak_vram_gb']}GB", file=sys.stderr)


if __name__ == "__main__":
    main()
