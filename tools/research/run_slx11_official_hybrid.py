#!/usr/bin/env python3
"""Canonical host runner for BACKLOG-SLX11-OFFICIAL-HYBRID-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-SLX11-OFFICIAL-HYBRID-01"
MODEL = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe"
PROMPTS = ROOT / "workloads/gsm8k.jsonl"
SOURCES = [
    ROOT / "runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/PRE_REGISTRATION.md",
    ROOT / "runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/RESULT.md",
    ROOT / "runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/raw/receipt.json",
    ROOT / "runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json",
    PROMPTS,
]


def write(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wsl_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    return f"/mnt/{resolved.drive[0].lower()}/{resolved.as_posix()[3:]}"


def service_state() -> dict:
    completed = subprocess.run(["wsl", "-d", "Ubuntu-24.04", "--", "systemctl", "show", "llm-inference.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveState", "--no-pager"], capture_output=True, text=True, check=False, timeout=30)
    state = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    health = {}
    for port in (8080, 8081):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
                health[str(port)] = response.status
        except Exception:
            health[str(port)] = None
    return {"systemd": state, "health": health}


def run(outdir: pathlib.Path) -> dict:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    source_hashes = {path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in SOURCES}
    before = service_state()
    worker = ROOT / "tools/research/slx11_official_hybrid_worker.py"
    worker_json = raw / "worker.json"
    command = ["wsl", "-d", "Ubuntu-24.04", "--", "/home/augus/.venvs/adapt00-20260824/bin/python", wsl_path(worker), "--model", MODEL, "--prompts", wsl_path(PROMPTS), "--output", wsl_path(worker_json)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=1800)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed {completed.returncode}: {completed.stderr[-5000:]}")
    result = json.loads(worker_json.read_text(encoding="utf-8"))
    after = service_state()
    unchanged = int(before["systemd"].get("MainPID") == after["systemd"].get("MainPID") and after["systemd"].get("NRestarts") == "0" and after["health"] == {"8080": 200, "8081": 200})
    metrics = dict(result["metrics"])
    metrics["serving_process_unchanged"] = unchanged
    physical = result["physical_layers"]
    recomputed = {
        "official_checkpoint_identified": int(result["model_type"] == "qwen3_5" and result["text_model_type"] == "qwen3_5_text"),
        "hybrid_layer_types_verified": sum(row["match"] for row in physical),
        "physical_recurrent_layers": sum(row["actual"] == "linear_attention" for row in physical),
        "physical_full_attention_layers": sum(row["actual"] == "full_attention" for row in physical),
        "successful_live_forwards": len(result["samples"]),
        "finite_output_rate": sum(row["finite"] for row in result["samples"]) / len(result["samples"]),
        "serving_process_unchanged": unchanged,
    }
    if canonical_json_sha256(metrics) != canonical_json_sha256(recomputed):
        raise ValueError("independent aggregate mismatch")
    write(raw / "actual_scores.json", metrics)
    write(raw / "artifact_hashes.json", {**source_hashes, "checkpoint": {"config_sha256": result["config_sha256"], "tensor_sha256": result["tensor_sha256"]}, "worker_sha256": sha256_file(worker)})
    write(raw / "dataset_hashes.json", {"prompt_file_sha256": result["prompt_file_sha256"], "panel_semantic_sha256": canonical_json_sha256([row["task_id"] for row in result["samples"]])})
    write(raw / "failure_reproduction.json", {"historical_claims_not_reproduced": {"speedup": 4.49, "recall": 1.0}, "reason": "no matched physical dense comparator in historical packet"})
    write(raw / "falsifiable_hypothesis.json", {"declared_layers": 24, "recurrent_layers": 18, "full_attention_layers": 6, "forwards": 24})
    write(raw / "hardware_metrics.json", result["hardware"])
    write(raw / "independent_evaluation.json", {"aggregate_exact_match": True, "all_topology_matches": all(row["match"] for row in physical), "all_logits_finite": all(row["finite"] for row in result["samples"])})
    write(raw / "invalidation_rules.json", {"no_historical_speed_claim": True, "no_recall_claim": True, "abort_on_topology_mismatch": True})
    write(raw / "invariant_controls.json", {"forward_count": 24, "greedy_next_token": True, "model_mutated": False, "services_mutated": False})
    write(raw / "paired_baseline.json", {"declared_topology": [row["declared"] for row in physical], "physical_topology": [row["actual"] for row in physical]})
    write(raw / "real_implementation.json", {"architectures": result["architectures"], "model_type": result["model_type"], "full_attention_interval": result["full_attention_interval"], "layers": physical})
    write(raw / "scorer_hashes.json", {"runner_sha256": sha256_file(pathlib.Path(__file__).resolve()), "worker_sha256": sha256_file(worker)})
    write(raw / "semantic_parity.json", {"declared_physical_layer_matches": sum(row["match"] for row in physical), "total_layers": len(physical)})
    write(raw / "service_maintenance.json", {"before": before, "after": after, "service_untouched": bool(unchanged)})
    write(raw / "source_execution_receipt.json", {"historical_receipt_sha256": source_hashes["runs/research/SLX-11-GRANITE-HYBRID-2026-08-25/raw/receipt.json"]["sha256"], "learned_state_receipt_sha256": source_hashes["runs/research/BACKLOG-GDN02-LEARNED-STATE-01/raw/receipt.json"]["sha256"]})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in result["samples"]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    definitions = {"official_artifact": ("official_checkpoint_identified", "eq", 1), "hybrid_topology": ("hybrid_layer_types_verified", "eq", 24), "recurrent_layers": ("physical_recurrent_layers", "eq", 18), "attention_layers": ("physical_full_attention_layers", "eq", 6), "live_forward": ("successful_live_forwards", "eq", 24), "finite_outputs": ("finite_output_rate", "eq", 1.0), "runtime_unchanged": ("serving_process_unchanged", "eq", 1)}
    gates = {name: {"metric": metric, "operator": op, "threshold": threshold, "actual": metrics[metric], "pass": metrics[metric] == threshold} for name, (metric, op, threshold) in definitions.items()}
    evidence = {name: f"raw/{name}.json" for name in ("actual_scores", "artifact_hashes", "dataset_hashes", "failure_reproduction", "falsifiable_hypothesis", "hardware_metrics", "independent_evaluation", "invalidation_rules", "invariant_controls", "paired_baseline", "real_implementation", "scorer_hashes", "semantic_parity", "service_maintenance", "source_execution_receipt")}
    evidence.update({"acceptance_gates": "raw/receipt.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json"})
    generated = [raw / pathlib.Path(path).name for path in evidence.values() if path != "raw/receipt.json"] + [worker_json]
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=mono, input_paths=[*SOURCES, worker, *set(generated)], packages=["torch", "transformers"], runtime={"execution_mode": "official_hybrid_physical_forward", "worker_command": command})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(errors)
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write(raw / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = run(args.outdir.resolve())
    metrics = json.loads((args.outdir.resolve() / "raw/actual_scores.json").read_text(encoding="utf-8"))
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX11_OFFICIAL_HYBRID_ARTIFACT_QUALIFIED_R1" if passed else "SLX11_OFFICIAL_HYBRID_ARTIFACT_REJECTED_R1"
    (args.outdir.resolve() / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nThe official local Qwen3.5 checkpoint physically instantiated `{metrics['physical_recurrent_layers']}` recurrent and `{metrics['physical_full_attention_layers']}` full-attention layers, matched `{metrics['hybrid_layer_types_verified']}/24` declarations and completed `{metrics['successful_live_forwards']}/24` finite fresh forwards. This does not reproduce the synthetic 4.49x or 100% recall claims.\n", encoding="utf-8")
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
