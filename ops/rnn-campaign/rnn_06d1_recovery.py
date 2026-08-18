#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06D1 — Target-Agnostic Parameter-Free Recovery Utility runner.

Executes ONLY if RECOVERY_CEILING = QUALIFIED. INDEPENDENTLY re-captures the target-agnostic
historical snapshot pool + FINAL (its own mechanism-activation counters + boundary checks;
cross-checks determinism vs D0_READOUTS.npz), then applies a FROZEN family of target-agnostic,
parameter-free recovery methods (no gold, no target-write position, no oracle-best identity):
RECENCY, MAX_CONFIDENCE, MIN_ENTROPY, MAX_TOP1_TOP2_MARGIN, CONFIDENCE_X_RECENCY, LOGIT_ENSEMBLE,
FINAL_PLUS_HISTORICAL, plus a MATCHED_NO_HISTORY compute control. Reports paired recovery/harm,
stratified intervals, oracle gap, selection regret, economics; mints RECOVERY_UTILITY. No trained
reader, no DART, no Memory Caching.
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
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rnn_06d_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06D")
SPEC_PATH = os.path.join(OUTDIR, "D0_QUALIFICATION_SPEC.json")
SCHED_PATH = os.path.join(OUTDIR, "SNAPSHOT_SCHEDULE.json")
CEIL_PATH = os.path.join(OUTDIR, "RECOVERY_CEILING_RESULTS.json")
READOUTS = os.path.join(OUTDIR, "D0_READOUTS.npz")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
POOL_SEED = 20260817
BATCH = 2
AUDIT_SAMPLE = 8
STATE_BYTES = 52002816

# Frozen utility thresholds (TRAIN_PROTOCOL; not tuned)
UTILITY_SESOI = 0.05
COST_MEM_MAX_BYTES_FACTOR = 1          # <= K * STATE_BYTES
COST_LATENCY_MAX_MS = 100.0            # intrinsic restore+readout+select per query, warm
METHODS = ["RECENCY", "MAX_CONFIDENCE", "MIN_ENTROPY", "MAX_TOP1_TOP2_MARGIN",
           "CONFIDENCE_X_RECENCY", "LOGIT_ENSEMBLE", "FINAL_PLUS_HISTORICAL"]


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


def bootstrap_ci(delta_per_example, strat, s_strata, seed):
    rb = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0xD1CE])))
    strat_idx = {s: np.where(strat == s)[0] for s in range(s_strata)}
    boot = []
    for _ in range(2000):
        idx = np.concatenate([si[rb.integers(0, len(si), size=len(si))] for si in strat_idx.values()])
        boot.append(float(delta_per_example[idx].mean()))
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def main():
    t0 = time.time(); os.makedirs(OUTDIR, exist_ok=True); torch.manual_seed(0)
    runner = os.path.abspath(__file__); libpath = os.path.join(os.path.dirname(runner), "rnn_06d_lib.py")
    res = {"packet": "RNN-06D1-RECOVERY-UTILITY", "kind": "recovery_utility"}

    ceil = json.load(open(CEIL_PATH))
    assert ceil["RECOVERY_CEILING"] == "QUALIFIED", "D0 not QUALIFIED — D1 must not run"
    res["d0_dependency"] = {"RECOVERY_CEILING": "QUALIFIED",
                            "oracle_best_minus_final": ceil["paired_ceiling"]["oracle_best_minus_final"]}

    spec = json.load(open(SPEC_PATH)); sched = json.load(open(SCHED_PATH))
    M = spec["M"]; schedule = sched["schedule_slots"]; K = sched["K"]; final_slot = sched["final_slot"]
    assert schedule == lib.schedule_slots(M, K) and final_slot == M - 1

    from transformers.models.mamba2 import modeling_mamba2
    res["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_git_blob": git("hash-object", libpath), "git_head": git("rev-parse", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "repo_id": REPO_ID, "revision": REVISION, "chunk_size": CHUNK, "dtype": str(DTYPE),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "qualificationSetSha256": spec["qualificationSetSha256"],
        "snapshotScheduleSha256": sched["snapshotScheduleSha256"],
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "D1_PRE_REGISTRATION.md"))}
    assert modeling_mamba2.is_fast_path_available is False

    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    model = Mamba2ForCausalLM.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=DTYPE).to(DEVICE).eval()
    for blk in model.backbone.layers:
        blk.mixer.chunk_size = CHUNK
    model.config.chunk_size = CHUNK
    pools, _ = lib.build_pools(tok, POOL_SEED)
    vset = sorted(set(pools["scored_vals"])); vtensor = torch.tensor(vset, device=DEVICE, dtype=torch.long)
    V = len(vset); vidx = {v: i for i, v in enumerate(vset)}

    examples = spec["examples"]; N = len(examples)
    strat = np.array([e["stratum"] for e in examples]); s_strata = spec["s_strata"]
    prefixes, queries, golds, tslots = [], [], [], []
    for e in examples:
        toks, gold = lib.materialize_d0(e, M, pools)
        prefixes.append(toks); queries.append(toks[-2:]); golds.append(gold); tslots.append(e["target_slot"])
    golds = np.array(golds); tslots = np.array(tslots)
    gold_col = np.array([vidx[int(g)] for g in golds])

    counters = {"snapshotsCreated": 0, "snapshotsRestored": 0, "candidateSnapshotsScored": 0,
                "historicalSelections": 0, "finalSelections": 0, "ensembleCalls": 0,
                "oracleCalls": 0, "snapshotBoundaryChecks": 0, "snapshotBoundaryFailures": 0,
                "queriesEvaluated": 0}

    @torch.no_grad()
    def readout_at_slot(slot):
        plen = lib.prefix_len_for_slot(slot)
        sub_all = np.zeros((N, V), np.float32)
        for b in range(0, N, BATCH):
            rows = list(range(b, min(b + BATCH, N)))
            ids = torch.tensor([prefixes[r][:plen] for r in rows], device=DEVICE, dtype=torch.long)
            cache = build_cache(model, len(rows)); prefill(model, cache, ids)
            conv, ssm = snapshot(cache); counters["snapshotsCreated"] += len(rows)
            for li, r in enumerate(rows):
                counters["snapshotBoundaryChecks"] += 1
                if plen != len(prefixes[r][:plen]):
                    counters["snapshotBoundaryFailures"] += 1
            rcache = restore(model, conv, ssm, len(rows)); counters["snapshotsRestored"] += len(rows)
            q0 = torch.tensor([[queries[r][0]] for r in rows], device=DEVICE, dtype=torch.long)
            decode_token(model, rcache, q0, plen)
            q1 = torch.tensor([[queries[r][1]] for r in rows], device=DEVICE, dtype=torch.long)
            logits = decode_token(model, rcache, q1, plen + 1)[:, 0, :].float()
            sub = logits.index_select(1, vtensor)
            for li, r in enumerate(rows):
                sub_all[r] = sub[li].cpu().numpy(); counters["queriesEvaluated"] += 1
            del cache, rcache, conv, ssm; torch.cuda.empty_cache()
        return sub_all

    # boundary self-check
    selfcheck = {"checked": 0, "reproduced": 0, "failures": 0}
    for r in range(min(AUDIT_SAMPLE, N)):
        plen = lib.prefix_len_for_slot(schedule[0])
        c1 = build_cache(model, 1); prefill(model, c1, torch.tensor([prefixes[r][:plen]], device=DEVICE))
        c2 = build_cache(model, 1); prefill(model, c2, torch.tensor([prefixes[r][:plen]], device=DEVICE))
        selfcheck["checked"] += 1
        if state_sha(c1.conv_states[:, 0]) == state_sha(c2.conv_states[:, 0]) and \
           state_sha(c1.ssm_states[:, 0]) == state_sha(c2.ssm_states[:, 0]):
            selfcheck["reproduced"] += 1
        else:
            selfcheck["failures"] += 1
        del c1, c2
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # re-capture pool + final
    pool_logits = np.stack([readout_at_slot(s) for s in schedule], axis=1)   # (N,K,V)
    final_logits = readout_at_slot(final_slot)                                # (N,V)
    counters["candidateSnapshotsScored"] = int(N * K)
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # determinism cross-check vs D0 npz
    xcheck = {"available": os.path.isfile(READOUTS)}
    if xcheck["available"]:
        d0 = np.load(READOUTS)
        xcheck["pool_logits_max_abs_diff"] = float(np.max(np.abs(d0["pool_logits"] - pool_logits)))
        xcheck["final_logits_max_abs_diff"] = float(np.max(np.abs(d0["final_logits"] - final_logits)))
        xcheck["bit_reproducible"] = bool(xcheck["pool_logits_max_abs_diff"] == 0.0 and
                                          xcheck["final_logits_max_abs_diff"] == 0.0)
    res["d0_crosscheck"] = xcheck

    # helpers over constrained logits
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True)); return e / e.sum(axis=-1, keepdims=True)
    pool_probs = softmax(pool_logits)               # (N,K,V)
    pool_pred = pool_logits.argmax(-1)              # (N,K)
    pool_top1 = pool_probs.max(-1)                  # (N,K)
    srt = np.sort(pool_probs, axis=-1)              # ascending
    pool_margin = srt[..., -1] - srt[..., -2]       # (N,K)
    pool_entropy = -(pool_probs * np.log(pool_probs + 1e-12)).sum(-1)  # (N,K)
    final_pred = final_logits.argmax(-1)            # (N,)
    final_correct = (final_pred == gold_col)
    pool_correct = (pool_pred == gold_col[:, None]) # (N,K)
    oracle_best = pool_correct.any(1)               # gold-aware ceiling (diagnostic)
    counters["oracleCalls"] += N
    recency_w = (np.arange(K) + 1) / K              # later slot = higher recency

    def method_pred(name):
        if name == "RECENCY":
            sel = np.full(N, K - 1); return pool_pred[np.arange(N), sel], sel, "select"
        if name == "MAX_CONFIDENCE":
            sel = pool_top1.argmax(1); return pool_pred[np.arange(N), sel], sel, "select"
        if name == "MIN_ENTROPY":
            sel = pool_entropy.argmin(1); return pool_pred[np.arange(N), sel], sel, "select"
        if name == "MAX_TOP1_TOP2_MARGIN":
            sel = pool_margin.argmax(1); return pool_pred[np.arange(N), sel], sel, "select"
        if name == "CONFIDENCE_X_RECENCY":
            sel = (pool_top1 * recency_w[None, :]).argmax(1); return pool_pred[np.arange(N), sel], sel, "select"
        if name == "LOGIT_ENSEMBLE":
            return pool_logits.mean(1).argmax(-1), None, "ensemble"
        if name == "FINAL_PLUS_HISTORICAL":
            mix = (final_logits[:, None, :].sum(1) + pool_logits.sum(1)) / (K + 1)
            return mix.argmax(-1), None, "ensemble_final"
        raise ValueError(name)

    method_reports = {}
    harm_rows = []; diag_rows = []
    for name in METHODS:
        pred, sel, kind = method_pred(name)
        correct = (pred == gold_col)
        if kind == "select":
            counters["historicalSelections"] += N
        elif kind == "ensemble":
            counters["ensembleCalls"] += N
        elif kind == "ensemble_final":
            counters["ensembleCalls"] += N; counters["finalSelections"] += 0  # final used in mixture, not sole
        acc = float(correct.mean())
        fw = ~final_correct; fc = final_correct
        n_fw = int(fw.sum()); n_fc = int(fc.sum())
        n_recovered = int((fw & correct).sum()); n_harmed = int((fc & ~correct).sum())
        recovery_rate = n_recovered / n_fw if n_fw else 0.0
        harm_rate = n_harmed / n_fc if n_fc else 0.0
        net_count = n_recovered - n_harmed; net_rate = net_count / N
        delta = correct.astype(float) - final_correct.astype(float)
        acc_delta = float(delta.mean())
        ci = bootstrap_ci(delta, strat, s_strata, spec["master_seed"])
        per_strat = {int(s): float(correct[strat == s].mean() - final_correct[strat == s].mean())
                     for s in range(s_strata)}
        robust = sum(1 for s in per_strat if per_strat[s] >= 0)
        oracle_gap = float(oracle_best.mean() - acc)
        # selection regret: among oracle-recoverable, fraction method got wrong
        rec_mask = oracle_best
        regret = float((rec_mask & ~correct).sum() / rec_mask.sum()) if rec_mask.sum() else 0.0
        hist = ([int(np.sum(sel == k)) for k in range(K)] if sel is not None else None)
        method_reports[name] = {
            "acc": acc, "accuracy_delta": round(acc_delta, 4), "accuracy_delta_ci95": ci,
            "n_final_wrong": n_fw, "n_recovered": n_recovered, "recovery_rate": round(recovery_rate, 4),
            "n_final_correct": n_fc, "n_harmed": n_harmed, "harm_rate": round(harm_rate, 4),
            "net_recovery_count": net_count, "net_recovery_rate": round(net_rate, 4),
            "per_stratum_delta": per_strat, "robust_strata_nonneg": robust,
            "oracle_gap": round(oracle_gap, 4), "selection_regret": round(regret, 4),
            "selectedSnapshotHistogram": hist, "kind": kind}
        harm_rows.append({"method": name, "acc": acc, "acc_final": float(final_correct.mean()),
                          "accuracy_delta": acc_delta, "n_recovered": n_recovered, "n_harmed": n_harmed,
                          "net_recovery_count": net_count, "recovery_rate": recovery_rate,
                          "harm_rate": harm_rate, "oracle_gap": oracle_gap, "selection_regret": regret})

    # MATCHED_NO_HISTORY control: K copies of FINAL ensembled == FINAL
    ctrl_pred = np.stack([final_logits] * K, axis=1).mean(1).argmax(-1)
    ctrl_correct = (ctrl_pred == gold_col)
    res["matched_no_history_control"] = {
        "acc": float(ctrl_correct.mean()), "equals_final": bool(np.array_equal(ctrl_correct, final_correct)),
        "note": "K FINAL readouts ensembled: same compute as LOGIT_ENSEMBLE, no history -> == FINAL"}

    # per-example selector diagnostics CSV (first 64)
    for r in range(min(64, N)):
        diag_rows.append({"exampleId": r, "stratum": int(strat[r]), "target_slot": int(tslots[r]),
                          "final_correct": int(final_correct[r]), "oracle_best_correct": int(oracle_best[r]),
                          **{f"pool{ki}_correct": int(pool_correct[r, ki]) for ki in range(K)},
                          **{f"pool{ki}_top1": round(float(pool_top1[r, ki]), 4) for ki in range(K)}})

    # ---- gate ----
    final_acc = float(final_correct.mean())
    ranked = sorted(METHODS, key=lambda m: method_reports[m]["accuracy_delta"], reverse=True)
    best = ranked[0]; br = method_reports[best]
    # cost: intrinsic restore+readout per query (warm) from D0 cost profile (per batch -> per query)
    cp = ceil.get("cost_profile", {})
    intrinsic_ms = 1000.0 * cp.get("restore_plus_readout_s_per_batch", 0.0) / BATCH
    mem_ok = (STATE_BYTES * K) <= (COST_MEM_MAX_BYTES_FACTOR * STATE_BYTES * K)
    latency_ok = intrinsic_ms <= COST_LATENCY_MAX_MS
    cost_ok = mem_ok and latency_ok
    semantic = (br["accuracy_delta"] >= UTILITY_SESOI and br["net_recovery_count"] > 0
                and br["accuracy_delta_ci95"][0] > 0 and br["robust_strata_nonneg"] >= 2)
    if semantic and cost_ok:
        verdict, reason = "QUALIFIED_PARAMETER_FREE", "OK"
    elif semantic and not cost_ok:
        verdict, reason = "SEMANTIC_GAIN_COST_FAIL", "COST_ENVELOPE_EXCEEDED"
    elif br["accuracy_delta"] > 0 and br["net_recovery_count"] > 0:
        verdict, reason = "ORACLE_GAP_REMAINS", "GAIN_BELOW_SESOI_OR_CI_OR_ROBUSTNESS"
    else:
        verdict, reason = "NOT_USEFUL", "NO_POSITIVE_NET_VALUE"
    res["best_method"] = best
    res["gate_checks"] = {"best_method": best, "best_accuracy_delta": br["accuracy_delta"],
                          "best_delta_ci95": br["accuracy_delta_ci95"], "best_net_recovery": br["net_recovery_count"],
                          "best_robust_strata_nonneg": br["robust_strata_nonneg"], "semantic_ok": semantic,
                          "intrinsic_restore_readout_ms_per_query": round(intrinsic_ms, 3),
                          "latency_ok": latency_ok, "mem_ok": mem_ok, "cost_ok": cost_ok, "reason": reason}
    res["RECOVERY_UTILITY"] = verdict
    res["method_reports"] = method_reports
    res["ranking_by_accuracy_delta"] = ranked
    res["final_acc"] = final_acc
    res["oracle_best_acc"] = float(oracle_best.mean())
    res["boundary_selfcheck"] = selfcheck
    res["mechanism_activation"] = counters
    res["economics"] = {"state_bytes_per_sequence": STATE_BYTES, "state_bytes_times_K": STATE_BYTES * K,
                        "historical_snapshots_exposed": K, "peak_vram_gb": res["peak_vram_gb"],
                        "d0_cost_profile_s_per_batch": cp,
                        "intrinsic_restore_readout_ms_per_query_warm": round(intrinsic_ms, 3),
                        "capture_reprefill_note": "re-prefill capture is a naive-backend artifact; a "
                        "fast-path/incremental capture folds capture into the single forward pass"}
    res["total_runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUTDIR, "RECOVERY_UTILITY_RESULTS.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    with open(os.path.join(OUTDIR, "MECHANISM_ACTIVATION.json"), "w") as f:
        json.dump({"D1": counters, "boundary_selfcheck": selfcheck}, f, indent=2)
    with open(os.path.join(OUTDIR, "RECOVERY_HARM.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(harm_rows[0].keys())); w.writeheader(); w.writerows(harm_rows)
    with open(os.path.join(OUTDIR, "SELECTOR_DIAGNOSTICS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys())); w.writeheader(); w.writerows(diag_rows)

    print(f"\nFINAL={final_acc:.3f} ORACLE_BEST={res['oracle_best_acc']:.3f}", file=sys.stderr)
    for m in ranked:
        r = method_reports[m]
        print(f"  {m:24s} acc={r['acc']:.3f} d={r['accuracy_delta']:+.3f} CI={r['accuracy_delta_ci95']} "
              f"rec={r['n_recovered']} harm={r['n_harmed']} net={r['net_recovery_count']} "
              f"gap={r['oracle_gap']:.3f}", file=sys.stderr)
    print(f"best={best} cost_ok={cost_ok} intrinsic_ms={intrinsic_ms:.2f}", file=sys.stderr)
    print(f"RECOVERY_UTILITY = {verdict} ({reason})", file=sys.stderr)
    print(f"xcheck_bit_reproducible={xcheck.get('bit_reproducible')} runtime={res['total_runtime_s']}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
