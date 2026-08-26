#!/usr/bin/env python3
"""Canonical host runner for BACKLOG-GDN02-LEARNED-STATE-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-GDN02-LEARNED-STATE-01"
MODEL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
CORPUS = ROOT / "workloads/gsm8k.jsonl"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-GDN02-LEARNED-STATE-01.json": "469495bdcaa74f2bb5ba0a953c796f62d5fbfb7545935c153c197a802ef9ea87",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-01/PRE_REGISTRATION.md": "ec4d9514ed8497552e91b714c270020e61314155889776d1a12cbd98f7ae2373",
    ROOT / "runs/research/GDN-02-ERASE-RETENTION-2026-08-25/PRE_REGISTRATION.md": "9b72e8acb64f365a3d2bbed82eca1381e3640635a26073d5cd6c274762c1cb4b",
    ROOT / "runs/research/GDN-02-ERASE-RETENTION-2026-08-25/RESULT.md": "677560ec0bd1e191cc7c187a594fa320f236b0ae501646563183695bf197c766",
    ROOT / "runs/research/GDN-02-ERASE-RETENTION-2026-08-25/raw/receipt.json": "0494fc573181c9dfa6d16371f38faf72461da8942f8c8659f8a40ad8d7266e33",
    ROOT / "tools/probes/gdn02_erase_retention_lab.py": "e34cdcfdcb6f9df13bc87aeb67c549922c7015eb24c3268ca2d13f6fdc786ae7",
    ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/raw/model_hash.json": "45f10080c70897cb106b21013bc4953f6a5696a27296098972d60ca132fad1ec",
    CORPUS: "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}
MODEL_SHA256 = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"


def write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wsl_path(path: pathlib.Path) -> str:
    resolved = path.resolve(); return f"/mnt/{resolved.drive[0].lower()}/{resolved.as_posix()[3:]}"


def service_state() -> dict[str, Any]:
    completed = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "systemctl", "show", "llm-inference.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveState", "--no-pager"], capture_output=True, text=True, check=False, timeout=30)
    systemd = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    health = {}
    for port, name in ((8080, "inference"), (8081, "embedding")):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
                health[name] = response.status
        except Exception:
            health[name] = None
    return {"systemd": systemd, "health": health}


def run(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"; started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); mono = time.monotonic()
    if any(raw.iterdir()): raise RuntimeError("raw directory is not empty")
    ledger = {}
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected: raise ValueError(f"source mismatch {path}: {actual}")
        ledger[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}
    before = service_state()
    worker = ROOT / "tools/research/gdn02_learned_state_worker.py"
    worker_json = raw / "worker.json"
    command = ["wsl", "-d", "Ubuntu-24.04", "--", "/home/augus/.venvs/adapt00-20260824/bin/python", wsl_path(worker), "--model", MODEL, "--corpus", wsl_path(CORPUS), "--output", wsl_path(worker_json), "--batch-size", "5"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=1800)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0: raise RuntimeError(f"worker failed {completed.returncode}: {completed.stderr[-5000:]}")
    result = json.loads(worker_json.read_text(encoding="utf-8"))
    if result["model_file_sha256"] != MODEL_SHA256 or result["corpus_sha256"] != EXPECTED[CORPUS]: raise ValueError("worker source identity mismatch")
    after = service_state()
    if before["systemd"].get("MainPID") != after["systemd"].get("MainPID") or after["systemd"].get("NRestarts") != "0": raise RuntimeError("serving process changed")
    if after["health"] != {"inference": 200, "embedding": 200}: raise RuntimeError(f"service unhealthy: {after}")
    cells = result["cells"]; metrics = result["metrics"]
    recomputed = {
        "learned_gdn_layer_cells": len(cells),
        "median_old_fact_leakage_pct": statistics.median(row["old_fact_leakage_pct"] for row in cells),
        "median_collateral_retention_pct": statistics.median(row["collateral_retention_pct"] for row in cells),
        "median_updated_fact_fidelity_pct": statistics.median(row["updated_fact_fidelity_pct"] for row in cells),
        "distinct_recurrent_state_conditions": min(row["distinct_recurrent_state_conditions"] for row in cells),
    }
    if canonical_json_sha256(recomputed) != canonical_json_sha256(metrics): raise ValueError("independent aggregate mismatch")
    token_matched = result["token_lengths"]["baseline"] == result["token_lengths"]["treatment"] and result["token_lengths"]["baseline"][5] == result["token_lengths"]["oracle"][0]
    if not token_matched: raise ValueError("token matching failed")

    write(raw / "actual_scores.json", metrics); write(raw / "artifact_hashes.json", {**ledger, "model_file": {"path": MODEL, "sha256": MODEL_SHA256}, "worker": {"sha256": sha256_file(worker)}})
    write(raw / "dataset_hashes.json", {"corpus_sha256": EXPECTED[CORPUS], "records_semantic_sha256": canonical_json_sha256(result["records"])})
    write(raw / "source_execution_receipt.json", {"historical_receipt_sha256": EXPECTED[ROOT / "runs/research/GDN-02-ERASE-RETENTION-2026-08-25/raw/receipt.json"], "historical_verdict": "REJECTED"})
    write(raw / "falsifiable_hypothesis.json", {"leakage_max_pct": 5.0, "retention_min_pct": 90.0, "fidelity_min_pct": 95.0})
    write(raw / "invariant_controls.json", {"layers": [0,1,2], "records": 50, "target_index": 5, "old_value": "41", "new_value": "42", "token_counts_matched": token_matched})
    write(raw / "invalidation_rules.json", {"oracle_materiality_min": 1e-4, "all_gates_required": True, "learned_parameters_unchanged": True})
    write(raw / "target_materiality.json", {"baseline_oracle_distances": [row["baseline_oracle_distance"] for row in cells], "all_material": all(row["baseline_oracle_distance"] >= 1e-4 for row in cells)})
    write(raw / "paired_baseline.json", {"baseline": "old plus reaffirm-old", "treatment": "old plus correction-new", "oracle": "new plus reaffirm-new", "same_templates_and_lengths": token_matched})
    write(raw / "real_implementation.json", {"module_classes": [row["module_class"] for row in cells], "official_recurrent_kernel": "torch_chunk_gated_delta_rule", "learned_checkpoint": MODEL_SHA256})
    write(raw / "hardware_metrics.json", result["hardware"])
    write(raw / "independent_evaluation.json", {"aggregate_exact_match": True, "token_counts_matched": token_matched, "all_state_conditions_distinct": all(row["distinct_recurrent_state_conditions"] == 3 for row in cells)})
    write(raw / "semantic_parity.json", {"collateral_cosine_retention_by_layer": [row["collateral_retention_pct"] for row in cells]})
    write(raw / "failure_reproduction.json", {"historical": {"leakage_pct": 2.84, "retention_pct": 65.31, "fidelity_pct": 73.32, "source": "random_associative_matrices"}, "learned_successor": metrics})
    write(raw / "service_maintenance.json", {"before": before, "after": after, "service_untouched": True})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in cells: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    defs = {"learned_states":("learned_gdn_layer_cells","ge",3),"target_leakage":("median_old_fact_leakage_pct","le",5.0),"collateral_retention":("median_collateral_retention_pct","ge",90.0),"update_fidelity":("median_updated_fact_fidelity_pct","ge",95.0),"state_materiality":("distinct_recurrent_state_conditions","ge",3)}
    ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b,"le":lambda a,b:a<=b}; gates={g:{"metric":m,"operator":o,"threshold":t,"actual":metrics[m],"pass":ops[o](metrics[m],t)} for g,(m,o,t) in defs.items()}
    names=("actual_scores.json","artifact_hashes.json","dataset_hashes.json","failure_reproduction.json","falsifiable_hypothesis.json","hardware_metrics.json","independent_evaluation.json","invalidation_rules.json","invariant_controls.json","paired_baseline.json","real_implementation.json","samples.jsonl","semantic_parity.json","service_maintenance.json","source_execution_receipt.json","target_materiality.json","worker.json")
    files=[raw/name for name in names]
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,worker,*files],packages=["pytest"],runtime={"execution_mode":"learned_qwen35_gated_delta_state","worker_command":command})
    complete,errors=provenance_complete(provenance)
    if not complete: raise ValueError(errors)
    evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","semantic_parity":"raw/semantic_parity.json","source_execution_receipt":"raw/source_execution_receipt.json","target_materiality":"raw/target_materiality.json"}
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write(raw/"receipt.json",receipt);return receipt


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);args=parser.parse_args();outdir=args.outdir.resolve();receipt=run(outdir);metrics=json.loads((outdir/"raw/actual_scores.json").read_text(encoding="utf-8"));all_pass=all(v["pass"] for v in receipt["gates"].values());claim="GDN02_FALSE_NEGATIVE_CONFIRMED_R1" if all_pass else "GDN02_NEGATIVE_RETAINED_R1";failed=[k for k,v in receipt["gates"].items() if not v["pass"]];(outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nThree learned Qwen3.5 GatedDeltaNet cells yielded median leakage `{metrics['median_old_fact_leakage_pct']:.4f}%`, collateral retention `{metrics['median_collateral_retention_pct']:.4f}%`, and update fidelity `{metrics['median_updated_fact_fidelity_pct']:.4f}%`. Failed gates: `{', '.join(failed) if failed else 'none'}`. Scope is representation-level learned recurrent state only.\n",encoding="utf-8");print(json.dumps(receipt["gates"],indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
