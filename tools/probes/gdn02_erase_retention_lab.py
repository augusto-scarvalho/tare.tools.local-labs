#!/usr/bin/env python3
"""GDN-02: Gated DeltaNet-2 Query-Conditioned Erase & Retention Lab.

Evaluates selective memory erasure and retention across 50 associative facts in 64x64 recurrent states,
comparing Static Decay vs Classic DeltaNet vs Query-Gated DeltaNet-2.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_gdn_experiment(num_facts: int = 50, d_k: int = 64, d_v: int = 64, torch=None) -> dict:
    torch.manual_seed(20260824)

    # 1. Generate 50 orthogonal/semi-orthogonal keys and random value vectors
    raw_keys = torch.randn(num_facts, d_k, device="cuda")
    keys = torch.nn.functional.normalize(raw_keys, p=2, dim=-1)
    values = torch.randn(num_facts, d_v, device="cuda")

    # Replacement fact for key 5
    target_idx = 5
    v_old = values[target_idx].clone()
    v_new = torch.randn(d_v, device="cuda")

    mechanisms = ["STATIC_DECAY_SSM", "CLASSIC_DELTANET", "QUERY_GATED_DELTANET2"]
    results = {}

    for mech in mechanisms:
        state = torch.zeros(d_v, d_k, device="cuda")

        # Step 1: Ingest first 25 facts
        for i in range(25):
            k_i = keys[i:i+1].t()  # (d_k, 1)
            v_i = values[i:i+1].t()  # (d_v, 1)

            if mech == "STATIC_DECAY_SSM":
                state = 0.98 * state + torch.matmul(v_i, k_i.t())
            elif mech == "CLASSIC_DELTANET":
                pred_v = torch.matmul(state, k_i)
                state = state + 0.8 * torch.matmul(v_i - pred_v, k_i.t())
            elif mech == "QUERY_GATED_DELTANET2":
                pred_v = torch.matmul(state, k_i)
                # Selective orthogonal projection gate
                state = state - 0.9 * torch.matmul(torch.matmul(state, k_i), k_i.t()) + 0.9 * torch.matmul(v_i, k_i.t())

        # Step 2: Inject Update/Erase on target fact 5 at step 25
        k_target = keys[target_idx:target_idx+1].t()
        v_target = v_new.unsqueeze(1)

        if mech == "STATIC_DECAY_SSM":
            state = 0.98 * state + torch.matmul(v_target, k_target.t())
        elif mech == "CLASSIC_DELTANET":
            pred_v = torch.matmul(state, k_target)
            state = state + 0.8 * torch.matmul(v_target - pred_v, k_target.t())
        elif mech == "QUERY_GATED_DELTANET2":
            # Direct subspace wipe on k_target + write v_new
            state = state - torch.matmul(torch.matmul(state, k_target), k_target.t()) + torch.matmul(v_target, k_target.t())

        # Step 3: Ingest remaining 25 facts (steps 26 to 50)
        for i in range(25, num_facts):
            k_i = keys[i:i+1].t()
            v_i = values[i:i+1].t()

            if mech == "STATIC_DECAY_SSM":
                state = 0.98 * state + torch.matmul(v_i, k_i.t())
            elif mech == "CLASSIC_DELTANET":
                pred_v = torch.matmul(state, k_i)
                state = state + 0.8 * torch.matmul(v_i - pred_v, k_i.t())
            elif mech == "QUERY_GATED_DELTANET2":
                pred_v = torch.matmul(state, k_i)
                state = state - 0.9 * torch.matmul(torch.matmul(state, k_i), k_i.t()) + 0.9 * torch.matmul(v_i, k_i.t())

        # Step 4: Evaluate Recall
        # A) Recall on target fact (Similarity to v_new vs v_old)
        rec_target = torch.matmul(state, k_target).squeeze()
        sim_new = torch.nn.functional.cosine_similarity(rec_target, v_new, dim=0).item()
        sim_old = torch.nn.functional.cosine_similarity(rec_target, v_old, dim=0).item()
        leakage_pct = max(0.0, sim_old) * 100.0

        # B) Collateral Retention across all other 49 facts
        collateral_sims = []
        for i in range(num_facts):
            if i == target_idx:
                continue
            k_query = keys[i:i+1].t()
            rec_v = torch.matmul(state, k_query).squeeze()
            true_v = values[i]
            sim = torch.nn.functional.cosine_similarity(rec_v, true_v, dim=0).item()
            collateral_sims.append(sim)

        avg_collateral_retention = sum(collateral_sims) / len(collateral_sims)
        collateral_pass_cnt = sum(1 for s in collateral_sims if s >= 0.70)
        collateral_retention_pct = (collateral_pass_cnt / len(collateral_sims)) * 100.0

        results[mech] = {
            "updated_fact_similarity_to_new": round(sim_new, 4),
            "updated_fact_fidelity_pct": round(max(0.0, sim_new) * 100.0, 2),
            "old_fact_leakage_pct": round(leakage_pct, 2),
            "collateral_average_cosine": round(avg_collateral_retention, 4),
            "collateral_memory_retention_pct": round(collateral_retention_pct, 2),
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="GDN-02 Gated DeltaNet-2 Lab")
    parser.add_argument("--output", default="runs/research/GDN-02-ERASE-RETENTION-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    import torch

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== GDN-02 Gated DeltaNet-2 Erase & Retention Lab ===", flush=True)
    res = run_gdn_experiment(num_facts=50, d_k=64, d_v=64, torch=torch)

    print("\nRecurrent Mechanism Performance (50 Facts):")
    for name, d in res.items():
        print(f"  [{name:22}]: Update Fidelity = {d['updated_fact_fidelity_pct']:5.1f}% | Old Leakage = {d['old_fact_leakage_pct']:4.1f}% | Collateral Retention = {d['collateral_memory_retention_pct']:5.1f}% (Avg Cosine = {d['collateral_average_cosine']})")

    gdn_s = res["QUERY_GATED_DELTANET2"]
    gates = {
        "old_fact_leakage_le_5pct": gdn_s["old_fact_leakage_pct"] <= 5.0,
        "collateral_retention_ge_90pct": gdn_s["collateral_memory_retention_pct"] >= 90.0,
        "update_fidelity_ge_95pct": gdn_s["updated_fact_fidelity_pct"] >= 95.0,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "results": res,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  GDN-02 GATED DELTANET-2 VERDICT: {verdict}", flush=True)
    print(f"  Old Fact Leakage:     {gdn_s['old_fact_leakage_pct']}% (Gate <=5%: {gates['old_fact_leakage_le_5pct']})")
    print(f"  Collateral Retention: {gdn_s['collateral_memory_retention_pct']}% (Gate >=90%: {gates['collateral_retention_ge_90pct']})")
    print(f"  Update Fidelity:      {gdn_s['updated_fact_fidelity_pct']}% (Gate >=95%: {gates['update_fidelity_ge_95pct']})")
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
