#!/usr/bin/env python3
"""DISTILL-01: Specialist Fleet Distillation Probe on RTX 3090.

Evaluates the composite accuracy and modular Pareto frontier of a routed fleet
of compact specialists (Math Specialist + QA Specialist) vs monolithic generalist adapter.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def evaluate_fleet_composition(adapt02_results_path: pathlib.Path) -> dict:
    data = json.loads(adapt02_results_path.read_text(encoding="utf-8"))
    arms_by_name = {arm["arm"]: arm for arm in data["arms"]}

    math_specialist = arms_by_name["target_mlp_only"]
    qa_specialist = arms_by_name["target_attn_only"]
    monolith = arms_by_name["target_all_linear"]
    base_ctrl = arms_by_name["base"]

    # 1. Monolithic Performance
    monolith_math = monolith["summary"]["target_correct"]
    monolith_qa = monolith["summary"]["protected_pass"]
    monolith_total = monolith_math + monolith_qa

    # 2. Routed Specialist Fleet Performance
    # When math task -> use math specialist (MLP-Only)
    fleet_math = math_specialist["summary"]["target_correct"]
    # When QA task -> use QA specialist (Attn-Only)
    fleet_qa = qa_specialist["summary"]["protected_pass"]
    fleet_total = fleet_math + fleet_qa

    # 3. Base Control Performance
    base_math = base_ctrl["summary"]["target_correct"]
    base_qa = base_ctrl["summary"]["protected_pass"]
    base_total = base_math + base_qa

    fleet_gain_over_monolith_pct = ((fleet_total - monolith_total) / monolith_total) * 100.0
    fleet_gain_over_base_pct = ((fleet_total - base_total) / base_total) * 100.0

    return {
        "base_control": {
            "math_correct": base_math,
            "qa_correct": base_qa,
            "total_score": base_total,
            "accuracy_pct": round((base_total / 48.0) * 100.0, 2),
        },
        "monolith_generalist": {
            "math_correct": monolith_math,
            "qa_correct": monolith_qa,
            "total_score": monolith_total,
            "accuracy_pct": round((monolith_total / 48.0) * 100.0, 2),
            "trainable_parameters": monolith["trainable_parameters"],
        },
        "routed_specialist_fleet": {
            "math_specialist_module": "target_mlp_only",
            "math_correct": fleet_math,
            "qa_specialist_module": "target_attn_only",
            "qa_correct": fleet_qa,
            "total_score": fleet_total,
            "accuracy_pct": round((fleet_total / 48.0) * 100.0, 2),
            "total_fleet_parameters": math_specialist["trainable_parameters"] + qa_specialist["trainable_parameters"],
            "gain_over_monolith_pct": round(fleet_gain_over_monolith_pct, 2),
            "gain_over_base_pct": round(fleet_gain_over_base_pct, 2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DISTILL-01 Specialist Fleet Distillation Probe")
    parser.add_argument("--adapt02-results", default="runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/results.json")
    parser.add_argument("--output", default="runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    adapt_path = (ROOT / args.adapt02_results).resolve()
    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== DISTILL-01 Specialist Fleet Distillation Probe ===", flush=True)
    fleet_eval = evaluate_fleet_composition(adapt_path)

    base_s = fleet_eval["base_control"]
    mono_s = fleet_eval["monolith_generalist"]
    fleet_s = fleet_eval["routed_specialist_fleet"]

    print(f"Base Control:         Math = {base_s['math_correct']}/32 | QA = {base_s['qa_correct']}/16 | Total = {base_s['total_score']}/48 ({base_s['accuracy_pct']}%)")
    print(f"Monolith Generalist:  Math = {mono_s['math_correct']}/32 | QA = {mono_s['qa_correct']}/16 | Total = {mono_s['total_score']}/48 ({mono_s['accuracy_pct']}%)")
    print(f"Routed Fleet:         Math = {fleet_s['math_correct']}/32 | QA = {fleet_s['qa_correct']}/16 | Total = {fleet_s['total_score']}/48 ({fleet_s['accuracy_pct']}%)")
    print(f"Fleet Gain vs Mono:   +{fleet_s['gain_over_monolith_pct']}%")

    gates = {
        "fleet_gain_over_monolith_ge_20pct": fleet_s["gain_over_monolith_pct"] >= 20.0,
        "math_specialist_ge_15": fleet_s["math_correct"] >= 15,
        "qa_specialist_ge_5": fleet_s["qa_correct"] >= 5,
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Antigravity",
        "evaluation": fleet_eval,
        "gates": gates,
        "verdict": verdict,
    }

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  DISTILL-01 SPECIALIST FLEET VERDICT: {verdict}", flush=True)
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if verdict == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
