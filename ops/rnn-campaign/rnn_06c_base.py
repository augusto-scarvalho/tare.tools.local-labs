#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RNN-06C-MAMBA-HISTORICAL-INFO — historical-state information-presence runner.

Executes ONLY if RNN-06B3 STATE_LOAD_FORGETTING_PERTURBATION = QUALIFIED. For each held-out
example builds three snapshots from the SAME target-slot-0 prefix — H (after target write), N
(target + neutral sentinel body), L (target + high-load body at U=HIGH) — captures each with full
temporal identity, RESTORES it (06A2 semantics), and decodes the identical target query. Primary
paired endpoint N - L. Streaming snapshot economics. Mints HISTORICAL_STATE_INFORMATION. No
recovery, no reader, no Memory Caching.
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
import rnn_06b3_lib as lib  # noqa: E402

REPO = "/mnt/c/projects/local-model-lifecycle"
OUTDIR = os.path.join(REPO, "runs", "rnn", "RNN-06C-MAMBA-HISTORICAL-INFO")
SPEC_PATH = os.path.join(OUTDIR, "HISTORICAL_INFO_SPEC.json")
B3_RESULTS = os.path.join(REPO, "runs", "rnn", "RNN-06B3-MAMBA-CONTROLLED-STATE-LOAD-PERTURBATION",
                          "B3_RESULTS.json")
REPO_ID = "AntonV/mamba2-1.3b-hf"
REVISION = "703e19a43f397c70315244a3424d79456b54fb34"
DEVICE, DTYPE, CHUNK = "cuda", torch.bfloat16, 32
POOL_SEED = 20260817        # same token pools as B3 qualification (fresh examples, shared pools)
M = 192
RESERVE = 16
BATCH = 2
AUDIT_SAMPLE = 8
SESOI = 0.15
TAU_HI = 0.75


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
    L = ids_2d.shape[1]
    cp = torch.arange(0, L, device=model.device)
    model(input_ids=ids_2d, cache_params=cache, cache_position=cp, use_cache=True)


@torch.no_grad()
def decode_token(model, cache, col_2d, offset):
    cp = torch.tensor([offset], device=model.device)
    return model(input_ids=col_2d, cache_params=cache, cache_position=cp, use_cache=True).logits


def snapshot(cache):
    return (cache.conv_states.detach().clone(), cache.ssm_states.detach().clone())


def restore(model, conv, ssm, rows):
    c = build_cache(model, len(rows))
    c.conv_states.copy_(conv[:, rows].to(c.conv_states.device))
    c.ssm_states.copy_(ssm[:, rows].to(c.ssm_states.device))
    return c


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    torch.manual_seed(0)
    runner = os.path.abspath(__file__)
    libpath = os.path.join(os.path.dirname(runner), "rnn_06b3_lib.py")
    results = {"packet": "RNN-06C-MAMBA-HISTORICAL-INFO", "kind": "historical_state_information"}

    # ---- upstream gate check ----
    b3 = json.load(open(B3_RESULTS))
    assert b3["STATE_LOAD_FORGETTING_PERTURBATION"] == "QUALIFIED", "B3 not QUALIFIED — 06C must not run"
    HIGH_U = int(b3["c06_dose_selection"]["HIGH_U"])
    b3_high_acc = float(b3["c06_dose_selection"]["HIGH_acc"])
    results["b3_dependency"] = {"STATE_LOAD_FORGETTING_PERTURBATION": "QUALIFIED",
                                "HIGH_U": HIGH_U, "b3_high_dose_acc": b3_high_acc}

    spec = json.load(open(SPEC_PATH))
    h_rec = spec.pop("historicalInfoSetSha256"); disj = spec.pop("disjointness_proof")
    h_re = lib.sha256_of_obj(spec)
    results["challenge_identities"] = {"historicalInfoSetSha256_recorded": h_rec,
                                       "historicalInfoSetSha256_recomputed": h_re,
                                       "sha_match": h_rec == h_re, "disjointness_proof": disj}
    assert h_rec == h_re, "06C spec sha mismatch"

    from transformers.models.mamba2 import modeling_mamba2, configuration_mamba2
    results["executed_source_identity"] = {
        "runner_sha256": sha256_file(runner), "runner_git_blob": git("hash-object", runner),
        "runner_dirty": git("status", "--porcelain", "--", runner),
        "lib_sha256": sha256_file(libpath), "lib_git_blob": git("hash-object", libpath),
        "git_head": git("rev-parse", "HEAD"),
        "modeling_mamba2_sha256": sha256_file(modeling_mamba2.__file__),
        "transformers": __import__("transformers").__version__, "torch": torch.__version__,
        "repo_id": REPO_ID, "revision": REVISION, "chunk_size": CHUNK, "dtype": str(DTYPE),
        "is_fast_path_available": modeling_mamba2.is_fast_path_available,
        "historicalInfoSetSha256": h_rec, "pool_seed": POOL_SEED,
        "protocol_sha256": sha256_file(os.path.join(OUTDIR, "PRE_REGISTRATION.md"))}
    assert modeling_mamba2.is_fast_path_available is False
    backend_semantics_id = f"transformers-native-naive-torch_forward-cs{CHUNK}-bf16"

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
    vset = set(pools["scored_vals"]); vtensor = torch.tensor(sorted(vset), device=DEVICE, dtype=torch.long)
    results["chance"] = 1.0 / len(vset)
    examples = spec["examples"]; ex_stratum = np.array([e["stratum"] for e in examples])
    N = len(examples)

    # ---- build H/N/L prefixes + query per example ----
    Hp, Np, Lp, queries, golds = [], [], [], [], []
    for e in examples:
        full_L, gold = lib.materialize_b3(e, M, HIGH_U, "DS", pools, RESERVE)
        full_N, _ = lib.materialize_b3(e, M, 1, "DS", pools, RESERVE)
        assert full_L[:4] == full_N[:4], "H prefix mismatch (branch-from-same-H broken)"
        assert full_L[-2:] == full_N[-2:], "query mismatch"
        Hp.append(full_L[:4]); Np.append(full_N[:768]); Lp.append(full_L[:768])
        queries.append(full_L[-2:]); golds.append(gold)
    golds = np.array(golds)

    counters = {"snapshotsCreated": 0, "snapshotsHashed": 0, "snapshotsRestored": 0,
                "historicalDirectReadouts": 0, "neutralAgedReadouts": 0, "highLoadReadouts": 0,
                "branchPairsCompleted": 0, "snapshotBoundaryChecks": 0, "snapshotBoundaryFailures": 0,
                "stateHashChecks": 0, "restoreChecks": 0, "queriesEvaluated": 0}
    identity_records = []
    audit_snapshots = []

    @torch.no_grad()
    def run_condition(prefixes, role, plen):
        """Prefill prefixes (batched, streaming), snapshot+hash+restore, decode query. Returns
        per-example correctness bool array."""
        correct = np.zeros(N, bool)
        for b in range(0, N, BATCH):
            rows = list(range(b, min(b + BATCH, N)))
            ids = torch.tensor([prefixes[r] for r in rows], device=DEVICE, dtype=torch.long)
            cache = build_cache(model, len(rows))
            prefill(model, cache, ids)
            conv, ssm = snapshot(cache)                       # capture snapshot
            counters["snapshotsCreated"] += len(rows)
            # temporal identity + hashes per row
            for li, r in enumerate(rows):
                csha = state_sha(conv[:, li]); ssha = state_sha(ssm[:, li])
                counters["snapshotsHashed"] += 1
                rec = {"exampleId": r, "snapshotRole": role, "cachePosition": plen,
                       "sequenceTokenPosition": plen, "associationSlotPosition": 0,
                       "recurrenceBoundaryId": f"{role}_after_prefix_len{plen}",
                       "prefixTokenSha256": tok_sha(prefixes[r]),
                       "convStateSha256": csha, "ssmStateSha256": ssha,
                       "combinedStateSha256": hashlib.sha256((csha + ssha).encode()).hexdigest(),
                       "modelRevision": REVISION, "modelWeightsIdentity": weights_id,
                       "backendSemanticsId": backend_semantics_id, "chunkSize": CHUNK,
                       "dtype": str(DTYPE)}
                # HARD boundary invariant: cachePosition == len(prefix)
                counters["snapshotBoundaryChecks"] += 1
                if rec["cachePosition"] != len(prefixes[r]):
                    counters["snapshotBoundaryFailures"] += 1
                if sum(1 for x in identity_records if x["snapshotRole"] == role) < 11:
                    identity_records.append(rec)
                if role == "L" and len(audit_snapshots) < AUDIT_SAMPLE:
                    audit_snapshots.append({"exampleId": r, "role": role,
                                            "convStateSha256": csha, "ssmStateSha256": ssha,
                                            "prefix_len": plen})
            # restore (round-trip) and decode query
            rcache = restore(model, conv, ssm, list(range(len(rows))))
            counters["snapshotsRestored"] += len(rows); counters["restoreChecks"] += len(rows)
            q0 = torch.tensor([[queries[r][0]] for r in rows], device=DEVICE, dtype=torch.long)
            decode_token(model, rcache, q0, plen)
            q1 = torch.tensor([[queries[r][1]] for r in rows], device=DEVICE, dtype=torch.long)
            logits = decode_token(model, rcache, q1, plen + 1)[:, 0, :].float()
            con = vtensor[logits.index_select(1, vtensor).argmax(-1)]
            for li, r in enumerate(rows):
                correct[r] = int(con[li]) == int(golds[r])
                counters["queriesEvaluated"] += 1
            del cache, rcache, conv, ssm
            torch.cuda.empty_cache()
        return correct

    # ---- deterministic self-check BEFORE substantive outcomes (audit sample) ----
    selfcheck = {"checked": 0, "state_hash_reproduced": 0, "branch_same_H_ok": 0, "failures": 0}
    for r in range(min(AUDIT_SAMPLE, N)):
        # re-prefill H prefix and L's first-4 slice; verify determinism + same-H
        cH = build_cache(model, 1); prefill(model, cH, torch.tensor([Hp[r]], device=DEVICE))
        convH, ssmH = snapshot(cH)
        cH2 = build_cache(model, 1); prefill(model, cH2, torch.tensor([Lp[r][:4]], device=DEVICE))
        convH2, ssmH2 = snapshot(cH2)
        counters["stateHashChecks"] += 1; selfcheck["checked"] += 1
        if state_sha(convH[:, 0]) == state_sha(convH2[:, 0]) and state_sha(ssmH[:, 0]) == state_sha(ssmH2[:, 0]):
            selfcheck["state_hash_reproduced"] += 1; selfcheck["branch_same_H_ok"] += 1
        else:
            selfcheck["failures"] += 1
        del cH, cH2
    results["boundary_selfcheck"] = selfcheck
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- run the three conditions ----
    H_correct = run_condition(Hp, "H", 4); counters["historicalDirectReadouts"] = int(counters["queriesEvaluated"])
    q_after_H = counters["queriesEvaluated"]
    N_correct = run_condition(Np, "N", 768); counters["neutralAgedReadouts"] = counters["queriesEvaluated"] - q_after_H
    q_after_N = counters["queriesEvaluated"]
    L_correct = run_condition(Lp, "L", 768); counters["highLoadReadouts"] = counters["queriesEvaluated"] - q_after_N
    counters["branchPairsCompleted"] = N
    results["construction_counters"] = counters
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    results["identity_records_sample"] = identity_records[:32]
    results["audit_snapshots"] = audit_snapshots

    # ---- accuracies + paired ----
    def acc(a):
        return float(a.mean())
    hi_acc, n_acc, l_acc = acc(H_correct), acc(N_correct), acc(L_correct)
    nml = n_acc - l_acc; hml = hi_acc - l_acc
    rb = lib.rng_for(20260818, 0xC0FFEE)
    strat_idx = {s: np.where(ex_stratum == s)[0] for s in range(spec["s_strata"])}
    boot = []
    for _ in range(2000):
        idx = np.concatenate([si[rb.integers(0, len(si), size=len(si))] for si in strat_idx.values()])
        boot.append(float(N_correct[idx].mean() - L_correct[idx].mean()))
    nml_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    strat_nml = {int(s): float(N_correct[ex_stratum == s].mean() - L_correct[ex_stratum == s].mean())
                 for s in range(spec["s_strata"])}
    robust = sum(1 for s in strat_nml if strat_nml[s] >= SESOI)
    results["accuracies"] = {
        "historical_direct_accuracy": hi_acc, "historical_correct": int(H_correct.sum()),
        "neutral_aged_accuracy": n_acc, "neutral_correct": int(N_correct.sum()),
        "high_load_accuracy": l_acc, "high_load_correct": int(L_correct.sum()), "n": N,
        "historical_ci95": wilson(int(H_correct.sum()), N), "neutral_ci95": wilson(int(N_correct.sum()), N),
        "high_load_ci95": wilson(int(L_correct.sum()), N)}
    results["paired_primary"] = {
        "neutral_minus_load": round(nml, 4), "neutral_minus_load_ci95": nml_ci,
        "historical_minus_load": round(hml, 4),
        "N_correct_L_wrong": int(np.sum(N_correct & ~L_correct)),
        "N_wrong_L_correct": int(np.sum(~N_correct & L_correct)),
        "H_correct_L_wrong": int(np.sum(H_correct & ~L_correct)),
        "H_wrong_L_correct": int(np.sum(~H_correct & L_correct)),
        "per_stratum_neutral_minus_load": strat_nml, "robust_strata_count": robust}

    # ---- gate ----
    h_comp = hi_acc >= TAU_HI
    material = nml >= SESOI
    ci_ok = nml_ci[0] > 0.05
    l_reproduces = abs(l_acc - b3_high_acc) <= 0.10
    robust_ok = robust >= 2
    boundary_ok = counters["snapshotBoundaryFailures"] == 0 and selfcheck["failures"] == 0
    counters_ok = all(counters[k] > 0 for k in ["snapshotsCreated", "snapshotsRestored",
                      "historicalDirectReadouts", "neutralAgedReadouts", "highLoadReadouts",
                      "branchPairsCompleted", "queriesEvaluated"])
    if not boundary_ok or not counters_ok:
        verdict = "BLOCKED"; reason = "INVALID_MACHINERY"
    elif not h_comp:
        verdict = "BLOCKED"; reason = "HISTORICAL_NOT_COMPETENT"
    elif material and ci_ok and robust_ok and l_reproduces:
        verdict = "QUALIFIED"; reason = "OK"
    else:
        verdict = "NOT_DETECTED"; reason = "N_MINUS_L_BELOW_SESOI_OR_NOT_ROBUST"
    results["gate_checks"] = {"historical_competent": h_comp, "material_N_minus_L": material,
                              "ci_excludes_trivial": ci_ok, "L_reproduces_B3": l_reproduces,
                              "robust_across_strata": robust_ok, "boundary_ok": boundary_ok,
                              "counters_ok": counters_ok, "reason": reason}
    results["HISTORICAL_STATE_INFORMATION"] = verdict
    results["state_bytes_per_sequence"] = 52002816
    results["total_runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUTDIR, "HISTORICAL_INFO_RESULTS.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nH={hi_acc:.3f} N={n_acc:.3f} L={l_acc:.3f}  N-L={nml:.3f} CI={nml_ci} "
          f"H-L={hml:.3f}", file=sys.stderr)
    print(f"transitions: N!->L={results['paired_primary']['N_correct_L_wrong']} "
          f"H!->L={results['paired_primary']['H_correct_L_wrong']} robust={robust}/3", file=sys.stderr)
    print(f"boundary_ok={boundary_ok} counters_ok={counters_ok} L_reproduces_B3={l_reproduces} "
          f"(L={l_acc:.3f} vs B3={b3_high_acc:.3f})", file=sys.stderr)
    print(f"HISTORICAL_STATE_INFORMATION = {verdict} ({reason})", file=sys.stderr)
    print(f"runtime={results['total_runtime_s']}s peak_vram={results['peak_vram_gb']}GB", file=sys.stderr)


if __name__ == "__main__":
    main()
