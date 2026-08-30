#!/usr/bin/env python3
"""Close HYPER-01 at the largest factor rank below the 20 MiB gate."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import build_provenance, provenance_complete, sha256_file
from tools.research.run_adapter_requalification_r2 import http_get_json, query_gpu, query_service, windows_path_to_wsl

TASK_ID = "BACKLOG-HYPER01-FACTORIZED-03"
PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
WORKER = ROOT / "tools/research/hyper01_factorized_worker_r2.py"
SCORER = ROOT / "tools/research/hyper01_factorized_scorer_r2.py"
SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]
CHECKPOINTS = [
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_answer_only/adapter_model.safetensors",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260824_full_trace/adapter_model.safetensors",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260825_answer_only/adapter_model.safetensors",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/checkpoints/seed_20260825_full_trace/adapter_model.safetensors",
]
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-HYPER01-FACTORIZED-03.json": "be40adf342d0f311b0115a0c93be89f4ef1ee20a7384c8c80bbec7f763af91fa",
    ROOT / "runs/research/BACKLOG-HYPER01-FACTORIZED-03/PRE_REGISTRATION.md": "0bdf81c2d41fb72f4cdf00f5b3a4dbc74452cbf15ce7732717af107d54c2ab0e",
    ROOT / "runs/research/BACKLOG-HYPER01-FACTORIZED-02/raw/receipt.json": "0662336e28e49661f56920470bf0c5a5fb06572362d5890f23659f6359d54028",
    ROOT / "runs/research/BACKLOG-HYPER01-FACTORIZED-02/REVIEW.json": "3b4e7ff3b6bb4fc3eb89cbbd2f6da9f3365e589c9ffe3776616214843a2272ef",
    ROOT / "runs/research/BACKLOG-HYPER01-FACTORIZED-02/raw/generator_states.safetensors": "efdf974bdc05aa6ffcdfbd074c5300f1c7e338314195110ca342558f48b71268",
    ROOT / "runs/research/BACKLOG-HYPER01-FACTORIZED-02/raw/generated_tensors.safetensors": "4121cb87fbed81aeb3702ac9e547802c970a603d94688b337fc034843a15566c",
    CHECKPOINTS[0]: "ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2",
    CHECKPOINTS[1]: "174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7",
    CHECKPOINTS[2]: "56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6",
    CHECKPOINTS[3]: "dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a",
}


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    raw = outdir / "raw"
    started_utc, started_mono = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.monotonic()
    for path, expected in EXPECTED.items():
        if sha256_file(path) != expected:
            raise ValueError(f"frozen hash mismatch: {path}")
    inputs = {path.relative_to(ROOT).as_posix(): digest for path, digest in EXPECTED.items()}

    with ExperimentRun(raw, TASK_ID, inputs) as run:
        service_before, gpu_before = query_service(), query_gpu()
        gateway_before = http_get_json("http://127.0.0.1:8080/health")
        embedding_before = http_get_json("http://127.0.0.1:8081/health")
        if service_before["active_state"] != "active" or gpu_before["memory_free_mib"] < 2048:
            raise RuntimeError("unsafe service/GPU baseline")
        run.checkpoint("preflight", {"free_gpu_mib": gpu_before["memory_free_mib"], "rank": 135, "steps": 3000})

        worker_json, states, generated = raw / "worker.json", raw / "generator_states.safetensors", raw / "generated_tensors.safetensors"
        worker_command = [
            "wsl", "-d", "Ubuntu-24.04", "--", PYTHON, windows_path_to_wsl(WORKER),
            "--checkpoints", *[windows_path_to_wsl(path) for path in CHECKPOINTS],
            "--output", windows_path_to_wsl(worker_json), "--state-output", windows_path_to_wsl(states),
            "--generated-output", windows_path_to_wsl(generated), "--steps", "3000", "--rank", "135",
            "--seeds", *[str(seed) for seed in SEEDS],
        ]
        completed = subprocess.run(worker_command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=7200)
        (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(completed.stderr[-5000:])
        payload = json.loads(worker_json.read_text(encoding="utf-8"))

        score_path = raw / "independent_evaluation.json"
        scorer_command = [
            "wsl", "-d", "Ubuntu-24.04", "--", PYTHON, windows_path_to_wsl(SCORER),
            "--checkpoints", *[windows_path_to_wsl(path) for path in CHECKPOINTS],
            "--generated", windows_path_to_wsl(generated), "--states", windows_path_to_wsl(states),
            "--seeds", *[str(seed) for seed in SEEDS], "--output", windows_path_to_wsl(score_path),
        ]
        scored = subprocess.run(scorer_command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        (raw / "scorer.stdout.log").write_text(scored.stdout, encoding="utf-8")
        (raw / "scorer.stderr.log").write_text(scored.stderr, encoding="utf-8")
        if scored.returncode:
            raise RuntimeError(scored.stderr[-5000:])
        independent = json.loads(score_path.read_text(encoding="utf-8"))

        service_after, gpu_after = query_service(), query_gpu()
        gateway_after = http_get_json("http://127.0.0.1:8080/health")
        embedding_after = http_get_json("http://127.0.0.1:8081/health")
        unchanged = (
            service_after["main_pid"] == service_before["main_pid"]
            and service_after["n_restarts"] == service_before["n_restarts"]
            and service_after["active_state"] == service_before["active_state"] == "active"
            and gateway_before.get("status") == gateway_after.get("status")
            and embedding_before.get("status") == embedding_after.get("status") == "ok"
        )
        metrics = {
            "physical_adapter_targets": payload["physical_adapter_targets"],
            "distinct_target_deltas": payload["distinct_target_deltas"],
            "completed_seeds": payload["completed_seeds"],
            "median_synthesis_latency_ms": payload["median_synthesis_latency_ms"],
            "worst_seed_mean_cosine": independent["worst_seed_mean_cosine"],
            "mean_weight_delta_cosine": independent["mean_weight_delta_cosine"],
            "generator_fp32_storage_mb": payload["generator_fp32_storage_mb"],
            "retained_seed_states": independent["retained_seed_states"],
            "independently_recomputed_cosines": sum(len(row["target_cosines"]) for row in independent["rows"]),
            "service_and_embedding_unchanged": unchanged,
        }
        definitions = {
            "physical_targets": ("physical_adapter_targets", "eq", 4), "target_distinctness": ("distinct_target_deltas", "eq", 4),
            "completed_seeds": ("completed_seeds", "eq", 5), "latency": ("median_synthesis_latency_ms", "le", 5.0),
            "worst_seed_fidelity": ("worst_seed_mean_cosine", "ge", 0.95), "overhead": ("generator_fp32_storage_mb", "le", 20.0),
            "retained_states": ("retained_seed_states", "eq", 5), "independent_rows": ("independently_recomputed_cosines", "eq", 20),
            "service_recovery": ("service_and_embedding_unchanged", "eq", True),
        }
        ops = {"eq": lambda a, b: a == b, "le": lambda a, b: a <= b, "ge": lambda a, b: a >= b}
        gates = {gid: {"metric": metric, "operator": op, "threshold": threshold, "actual": metrics[metric], "pass": ops[op](metrics[metric], threshold)} for gid, (metric, op, threshold) in definitions.items()}

        write_json(raw / "actual_scores.json", metrics | {"rank_136_fp32_mib": 20.097198486328125, "peak_allocated_mib": payload["peak_allocated_mib"], "peak_reserved_mib": payload["peak_reserved_mib"]})
        write_json(raw / "artifact_hashes.json", {"frozen_inputs": inputs, "states_sha256": sha256_file(states), "generated_sha256": sha256_file(generated), "worker_sha256": sha256_file(WORKER), "scorer_sha256": sha256_file(SCORER)})
        write_json(raw / "hardware_metrics.json", {"before": gpu_before, "after": gpu_after, "peak_allocated_mib": payload["peak_allocated_mib"], "peak_reserved_mib": payload["peak_reserved_mib"]})
        write_json(raw / "invalidation_rules.json", {"terminal_family_member": True, "all_gates_required": True, "no_further_rank_or_step_successor": True})
        write_json(raw / "paired_baseline.json", {"r2_rank": 32, "r2_worst_seed_cosine": 0.47980393559551493, "r2_storage_mib": 5.2686767578125, "r3_rank": 135})
        write_json(raw / "real_implementation.json", {"architecture": [64, 256, 512, 135, 36864], "steps": 3000, "seeds": SEEDS, "worker_command": worker_command, "scorer_command": scorer_command})
        write_json(raw / "repeatability.json", {"rows": independent["rows"], "worst_seed_mean_cosine": independent["worst_seed_mean_cosine"]})
        write_json(raw / "retained_tensors.json", {"states": {"bytes": states.stat().st_size, "sha256": sha256_file(states)}, "generated": {"bytes": generated.stat().st_size, "sha256": sha256_file(generated)}})
        write_json(raw / "semantic_parity.json", {"acceptance_source": "independent retained-byte scorer", "rows": 20})
        write_json(raw / "service_maintenance.json", {"before": service_before, "after": service_after, "gateway_before": gateway_before, "gateway_after": gateway_after, "embedding_before": embedding_before, "embedding_after": embedding_after, "unchanged": unchanged})
        write_json(raw / "target_materiality.json", {"targets": payload["target_ledger"], "distinct": payload["distinct_target_deltas"]})
        for row in independent["rows"]:
            for target, cosine in enumerate(row["target_cosines"]):
                run.record({"seed": row["seed"], "target": target, "weight_delta_cosine": cosine})
        run.checkpoint("five_seed_frontier_complete", {"rank": 135, "rows": 20, "worst_seed": independent["worst_seed_mean_cosine"]})

        evidence = {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json",
            "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json", "invalidation_rules": "raw/invalidation_rules.json",
            "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
            "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json", "repeatability": "raw/repeatability.json",
            "retained_tensors": "raw/retained_tensors.json", "semantic_parity": "raw/semantic_parity.json", "service_maintenance": "raw/service_maintenance.json",
            "target_materiality": "raw/target_materiality.json",
        }
        provenance_inputs = [*EXPECTED, WORKER, SCORER, worker_json, states, generated, score_path, *[raw / path.removeprefix("raw/") for path in evidence.values() if path != "raw/receipt.json"]]
        provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc, started_monotonic=started_mono, input_paths=provenance_inputs, packages=["torch", "safetensors"], runtime={"execution_mode": "max_rank_factorized_physical_targets", "worker_command": worker_command, "scorer_command": scorer_command})
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(errors)
        receipt = run.seal({"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence})
    return receipt, metrics


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID); args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve()); passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "HYPER01_MAX_RANK_PHYSICAL_FIT_R3" if passed else "HYPER01_RESOURCE_FIDELITY_FRONTIER_CLOSED_R3"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review. Rank 135, five seeds and 15,000 total steps completed. Worst-seed cosine `{metrics['worst_seed_mean_cosine']:.9f}`, storage `{metrics['generator_fp32_storage_mb']:.6f} MiB`; failed gates: {', '.join(failed) if failed else 'none'}. This terminal family member does not establish unseen-task or whole-adapter generalization.\n", encoding="utf-8")
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
