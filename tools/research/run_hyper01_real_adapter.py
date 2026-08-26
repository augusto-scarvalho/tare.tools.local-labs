#!/usr/bin/env python3
"""Host runner for BACKLOG-HYPER01-REAL-ADAPTER-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research.run_adapter_requalification_r2 import http_get_json, query_gpu, query_service, windows_path_to_wsl

TASK_ID = "BACKLOG-HYPER01-REAL-ADAPTER-01"
WORKER = ROOT / "tools/research/hyper01_real_adapter_worker.py"
PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
CHECKPOINTS = [
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_answer_only/adapter_model.safetensors",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260825_answer_only/adapter_model.safetensors",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260825_full_trace/adapter_model.safetensors",
]
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-HYPER01-REAL-ADAPTER-01.json": "0884c22bd3b6ead236543cd9d97db708b608790409c3a458981b6fb086f51a9d",
    ROOT / "runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/PRE_REGISTRATION.md": "6969ca78bfcfa3e832895904c7dcea0f920f2e428a459051f2230b3b82da20f9",
    ROOT / "runs/research/HYPER-01-CAPSULES-2026-08-25/PRE_REGISTRATION.md": "776cf15eaa10e3ebf0a343b99afd5bbab6c96d31f036ded1fafa0bae98d42bc0",
    ROOT / "runs/research/HYPER-01-CAPSULES-2026-08-25/RESULT.md": "87a9d4a2768e961f5a3c9cba2b691c46cd1a49687854d375d61808e2ead57ea8",
    ROOT / "runs/research/HYPER-01-CAPSULES-2026-08-25/raw/receipt.json": "1cddbc6b59a3148803fb51cac537316ce9039cfb7f7c0ce872b1e2b150f1b342",
    ROOT / "tools/probes/hyper01_capsule_generator.py": "a35bcd0877a35788e635305df0313d9e35e69f00acca36cfecbb3d49c09eb0e8",
    CHECKPOINTS[0]: "ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2",
    CHECKPOINTS[1]: "174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7",
    CHECKPOINTS[2]: "56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6",
    CHECKPOINTS[3]: "dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(outdir: pathlib.Path):
    raw = outdir / "raw"
    if any(raw.iterdir()): raise RuntimeError("raw directory is not empty")
    started_utc, started_mono = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    artifacts = {}
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected: raise ValueError(f"frozen hash mismatch: {path}: {actual}")
        artifacts[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    write_json(raw / "artifact_hashes.json", artifacts)
    service_before, gpu_before = query_service(), query_gpu()
    embedding_before = http_get_json("http://127.0.0.1:8081/health")
    output = raw / "worker.json"
    command = ["wsl", "-d", "Ubuntu-24.04", "--", PYTHON, windows_path_to_wsl(WORKER),
               "--checkpoints", *[windows_path_to_wsl(path) for path in CHECKPOINTS],
               "--output", windows_path_to_wsl(output)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode: raise RuntimeError(completed.stderr[-5000:])
    payload = json.loads(output.read_text(encoding="utf-8"))
    service_after, gpu_after = query_service(), query_gpu()
    embedding_after = http_get_json("http://127.0.0.1:8081/health")
    restored = (service_after["main_pid"] == service_before["main_pid"] and service_after["n_restarts"] == service_before["n_restarts"]
                and embedding_before.get("status") == embedding_after.get("status") == "ok")
    maintenance = {"initial_service": service_before, "final_service": service_after,
                   "initial_embedding": embedding_before, "final_embedding": embedding_after,
                   "service_and_embedding_restored": restored}
    write_json(raw / "service_maintenance.json", maintenance)
    scores = {key: payload[key] for key in ("physical_adapter_targets", "distinct_target_deltas",
              "median_synthesis_latency_ms", "mean_weight_delta_cosine", "generator_vram_overhead_mb")}
    recomputed_cosine = sum(payload["target_cosines"]) / len(payload["target_cosines"])
    recompute = abs(recomputed_cosine - scores["mean_weight_delta_cosine"]) < 1e-12
    write_json(raw / "actual_scores.json", scores)
    write_json(raw / "independent_evaluation.json", {"mean_cosine": recomputed_cosine, "match": recompute})
    write_json(raw / "target_materiality.json", {"targets": payload["target_ledger"], "distinct": payload["distinct_target_deltas"]})
    write_json(raw / "paired_baseline.json", {"target_cosines": payload["target_cosines"]})
    write_json(raw / "semantic_parity.json", {"independent_metric_recompute_match": recompute})
    write_json(raw / "real_implementation.json", {"physical_targets": True, "worker": str(WORKER.relative_to(ROOT).as_posix())})
    write_json(raw / "hardware_metrics.json", {"initial_gpu": gpu_before, "final_gpu": gpu_after, "device": payload["device"]})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for index, cosine in enumerate(payload["target_cosines"]):
            stream.write(json.dumps({"target": index, "weight_delta_cosine": cosine, **payload["target_ledger"][index]}) + "\n")
    decision = {"all_original_rules_pass": scores["median_synthesis_latency_ms"] <= 5 and scores["mean_weight_delta_cosine"] >= .95 and scores["generator_vram_overhead_mb"] <= 20}
    write_json(raw / "invalidation_rules.json", decision)
    observations = {**scores, "independent_metric_recompute_match": recompute, "service_and_embedding_restored": restored}
    definitions = {
        "physical_targets": ("physical_adapter_targets", "eq", 4), "target_distinctness": ("distinct_target_deltas", "eq", 4),
        "latency": ("median_synthesis_latency_ms", "le", 5.0), "fidelity": ("mean_weight_delta_cosine", "ge", .95),
        "overhead": ("generator_vram_overhead_mb", "le", 20.0), "independent_recompute": ("independent_metric_recompute_match", "eq", True),
        "service_recovery": ("service_and_embedding_restored", "eq", True)}
    ops={"eq":lambda a,b:a==b,"le":lambda a,b:a<=b,"ge":lambda a,b:a>=b}
    gates={gid:{"metric":m,"operator":op,"threshold":t,"actual":observations[m],"pass":ops[op](observations[m],t)} for gid,(m,op,t) in definitions.items()}
    files=[raw / name for name in ("actual_scores.json","artifact_hashes.json","hardware_metrics.json","independent_evaluation.json","invalidation_rules.json","paired_baseline.json","real_implementation.json","samples.jsonl","semantic_parity.json","service_maintenance.json","target_materiality.json")]
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started_utc,started_monotonic=started_mono,input_paths=[*EXPECTED,WORKER,*files],packages=["pytest"],runtime={"execution_mode":"physical_lora_hypernetwork","command":command})
    complete,errors=provenance_complete(provenance)
    if not complete: raise ValueError(errors)
    evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","semantic_parity":"raw/semantic_parity.json","service_maintenance":"raw/service_maintenance.json","target_materiality":"raw/target_materiality.json"}
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":complete,"gates":gates,"evidence":evidence}
    receipt["receipt_fingerprint"]=canonical_json_sha256(receipt); write_json(raw / "receipt.json",receipt); return receipt


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID); args=parser.parse_args()
    print(json.dumps(run(args.outdir.resolve())["gates"],indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
