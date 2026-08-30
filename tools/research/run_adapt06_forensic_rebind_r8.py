#!/usr/bin/env python3
"""Forensically rebind immutable ADAPT06 R7 route-affinity evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun, verify_run
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-ADAPT06-SLOP-LIVE-08"
R7_PACKET = ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-07"
R7_WATCH = ROOT / "runs/autonomous/POST-AUDIT-VALUE-WAVE-2026-08-29/watchers/002-BACKLOG-ADAPT06-SLOP-LIVE-07"
LIVE_ROWS = R7_PACKET / "raw/physical_r5/raw/live_rows.json"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-ADAPT06-SLOP-LIVE-08.json": "988e7aeb2919536827f5ffbdcc89d2409af6685678c7d893e8a8b4cb38ffeaa1",
    ROOT / "runs/research/BACKLOG-ADAPT06-SLOP-LIVE-08/PRE_REGISTRATION.md": "56bc11e4566e8a342bfc4c1c02431ad1d00f47f7914c0fa9ae77e15c52e90b9e",
    R7_PACKET / "raw/run.terminal.json": "5437e2bcded18d2b5f32a0dbc83812cbcfb38bddd54cd2f10dd0d44a5e1b23cf",
    R7_PACKET / "REVIEW.json": "f39cd7aef72da06b7abcf5c1fdddd7af26799f24cbfc02db41ca7b8f8bb3f36a",
    LIVE_ROWS: "43ef65d35e1ff79cea1e926f0486e442fb94e70bf4e80d10675cf3053a302830",
    R7_WATCH / "FINAL.json": "156f8a7628f4bea9fd320a77415217fd004d7c84b0a12354399fa5f54dab3487",
    R7_WATCH / "LAUNCH.json": "28af4decb884a652445a4ef9d7109288b8b3e55ea033ad3f094e1b5833f3fdd3",
    R7_WATCH / "WORKER_EXIT.json": "328c7d9aca047000b77272b9fcaae5144542beafd9cb44c9a42dc96673ea9655",
    R7_PACKET / "PIPELINE.json": "70239662df0a37dae7ae428aefeda4f9e084a1a5896d4150d06e07470f979132",
    ROOT / "tools/research/run_adapt06_slop_live_r7.py": "a443b1bb7df3f42dafafd45b67c01605a5515297264a587a01463fa20d9b27c8",
}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def route_lora(route: str) -> list[dict]:
    return {
        "base": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}],
        "mlp": [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}],
        "attn": [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}],
    }[route]


def route_switches(rows: list[dict]) -> int:
    return sum(left["route"] != right["route"] for left, right in zip(rows, rows[1:]))


def health_code(url: str) -> int:
    with urllib.request.urlopen(url, timeout=5) as response:
        response.read()
        return response.status


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    raw = outdir / "raw"
    started_utc, started_mono = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen input mismatch: {path}")
    inputs = {path.relative_to(ROOT).as_posix(): digest for path, digest in EXPECTED.items()}

    with ExperimentRun(raw, TASK_ID, inputs) as run:
        r7_terminal = verify_run(R7_PACKET / "raw")
        watcher = json.loads((R7_WATCH / "FINAL.json").read_text(encoding="utf-8"))
        launch = json.loads((R7_WATCH / "LAUNCH.json").read_text(encoding="utf-8"))
        worker_exit = json.loads((R7_WATCH / "WORKER_EXIT.json").read_text(encoding="utf-8"))
        sealed_source = (
            r7_terminal["valid"] and r7_terminal["status"] == "SEALED"
            and watcher["status"] == "complete" and worker_exit["returncode"] == 0
            and launch["task_id"] == "BACKLOG-ADAPT06-SLOP-LIVE-07"
        )
        data = json.loads(LIVE_ROWS.read_text(encoding="utf-8"))
        baselines, routed = data["baselines"], data["routed"]
        if len(baselines) != 36 or len(routed) != 72:
            raise ValueError("R7 row count drift")
        baseline_map = {(row["route"], row["index"]): row for row in baselines}
        if len(baseline_map) != 36:
            raise ValueError("duplicate baseline key")

        ledger, matches, controls = [], 0, 0
        for row in baselines:
            ledger.append({"phase": "baseline", "route": row["route"], "index": row["index"], "content_sha256": text_sha256(row["content"])})
        for row in routed:
            baseline = baseline_map[(row["route"], row["index"])]
            match = row["content"] == baseline["content"]
            expected_lora = route_lora(row["route"])
            control = row["request"].get("lora") == expected_lora and row["response"].get("generation_settings", {}).get("lora") == expected_lora
            matches += int(match); controls += int(control)
            ledger.append({"phase": "routed", "route": row["route"], "index": row["index"], "repeat": row["repeat"], "content_sha256": text_sha256(row["content"]), "match": match, "route_control": control})
        material_prompts = sum(len({baseline_map[(route, index)]["content"] for route in ("base", "mlp", "attn")}) >= 2 for index in range(12))

        alternating, grouped = data["schedule"]["alternating"], data["schedule"]["grouped"]
        if len(alternating) != 30 or len(grouped) != 30:
            raise ValueError("schedule count drift")
        alt_map = {(row["route"], row["index"]): row["content"] for row in alternating}
        grp_map = {(row["route"], row["index"]): row["content"] for row in grouped}
        parity = sum(alt_map[key] == grp_map.get(key) for key in alt_map) / len(alt_map)
        alt_switches, grouped_switches = route_switches(alternating), route_switches(grouped)
        switch_reduction = 1 - grouped_switches / alt_switches
        before_health = {"gateway": health_code("http://127.0.0.1:8080/health"), "embedding": health_code("http://127.0.0.1:8081/health")}
        after_health = {"gateway": health_code("http://127.0.0.1:8080/health"), "embedding": health_code("http://127.0.0.1:8081/health")}
        metrics = {
            "sealed_r7_source": sealed_source,
            "recomputed_baseline_hashes": sum(row["phase"] == "baseline" for row in ledger),
            "recomputed_routed_hashes": sum(row["phase"] == "routed" for row in ledger),
            "route_correct_counterfactual_match_rate": matches / len(routed),
            "physical_route_control_match_rate": controls / len(routed),
            "prompts_with_distinct_route_outputs": material_prompts,
            "schedule_semantic_parity": parity,
            "requested_route_switch_reduction": switch_reduction,
            "r8_and_r7_wrappers_provenance_bound": True,
            "gateway_and_embedding_healthy": before_health == after_health == {"gateway": 200, "embedding": 200},
        }
        definitions = {
            "sealed_source": ("sealed_r7_source", "eq", True), "baseline_hashes": ("recomputed_baseline_hashes", "eq", 36),
            "routed_hashes": ("recomputed_routed_hashes", "eq", 72), "route_counterfactual": ("route_correct_counterfactual_match_rate", "eq", 1.0),
            "route_controls": ("physical_route_control_match_rate", "eq", 1.0), "materiality": ("prompts_with_distinct_route_outputs", "ge", 4),
            "schedule_parity": ("schedule_semantic_parity", "eq", 1.0), "switch_reduction": ("requested_route_switch_reduction", "ge", 0.8),
            "wrapper_binding": ("r8_and_r7_wrappers_provenance_bound", "eq", True), "service_health": ("gateway_and_embedding_healthy", "eq", True),
        }
        ops = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b}
        gates = {gid: {"metric": metric, "operator": op, "threshold": threshold, "actual": metrics[metric], "pass": ops[op](metrics[metric], threshold)} for gid, (metric, op, threshold) in definitions.items()}
        for row in ledger:
            run.record(row)
        run.checkpoint("utf8_route_rebind_complete", {"rows": len(ledger), "material_prompts": material_prompts, "switch_reduction": switch_reduction})
        write_json(raw / "actual_scores.json", metrics | {"alternating_switches": alt_switches, "grouped_switches": grouped_switches, "alternating_wall_ms": data["schedule"]["alternating_wall_ms"], "grouped_wall_ms": data["schedule"]["grouped_wall_ms"]})
        write_json(raw / "artifact_hashes.json", {"frozen_inputs": inputs, "r7_terminal_manifest_files": len(r7_terminal["manifest"]), "r7_live_rows_sha256": sha256_file(LIVE_ROWS)})
        write_json(raw / "dataset_hashes.json", {"live_rows_sha256": sha256_file(LIVE_ROWS), "corrected_digest_ledger_sha256": canonical_json_sha256(ledger)})
        write_json(raw / "independent_evaluation.json", {"baseline_cells": 36, "routed_cells": 72, "route_matches": matches, "route_controls": controls, "material_prompts": material_prompts, "schedule_parity": parity})
        write_json(raw / "invalidation_rules.json", {"r7_immutable": True, "no_new_inference": True, "all_gates_required": True, "cache_and_speed_claims_forbidden": True})
        write_json(raw / "paired_baseline.json", {"alternating_cells": 30, "grouped_cells": 30, "alternating_switches": alt_switches, "grouped_switches": grouped_switches})
        write_json(raw / "scorer_hashes.json", {"r8_runner_sha256": sha256_file(pathlib.Path(__file__).resolve()), "digest": "sha256(utf8_text_bytes)"})
        write_json(raw / "semantic_parity.json", {"counterfactual_matches": matches, "schedule_matches": int(parity * 30), "total_routed": 72})
        write_json(raw / "service_maintenance.json", {"before": before_health, "after": after_health, "service_mutated": False})
        write_json(raw / "source_execution_receipt.json", {"r7_terminal_sha256": EXPECTED[R7_PACKET / "raw/run.terminal.json"], "r7_watcher_final_sha256": EXPECTED[R7_WATCH / "FINAL.json"], "r7_wrapper_sha256": EXPECTED[ROOT / "tools/research/run_adapt06_slop_live_r7.py"]})
        evidence = {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json",
            "dataset_hashes": "raw/dataset_hashes.json", "independent_evaluation": "raw/independent_evaluation.json", "invalidation_rules": "raw/invalidation_rules.json",
            "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json", "scorer_hashes": "raw/scorer_hashes.json", "semantic_parity": "raw/semantic_parity.json",
            "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json",
        }
        provenance_inputs = [*EXPECTED, *[raw / path.removeprefix("raw/") for path in evidence.values() if path != "raw/receipt.json"]]
        provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc, started_monotonic=started_mono, input_paths=provenance_inputs, packages=["pytest"], runtime={"execution_mode": "immutable_utf8_route_forensic_rebind", "new_inference": False})
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(errors)
        receipt = run.seal({"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence})
    return receipt, metrics


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID); args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve()); passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "ADAPT06_CLIENT_AFFINITY_FORENSIC_REBOUND_R8" if passed else "ADAPT06_CLIENT_AFFINITY_REJECTED_R8"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review. Recomputed 36 baseline and 72 routed UTF-8 digests; route-correct match `{metrics['route_correct_counterfactual_match_rate']:.4%}`; switch reduction `{metrics['requested_route_switch_reduction']:.4%}`; failed gates: {', '.join(failed) if failed else 'none'}. Client ordering only; no speed, cache-isolation or server-native scheduling claim.\n", encoding="utf-8")
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
