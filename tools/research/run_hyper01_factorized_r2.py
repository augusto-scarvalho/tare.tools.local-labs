#!/usr/bin/env python3
"""Host runner for BACKLOG-HYPER01-FACTORIZED-02."""
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

TASK_ID = "BACKLOG-HYPER01-FACTORIZED-02"
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
    ROOT / "config/research_backlog_admissions/BACKLOG-HYPER01-FACTORIZED-02.json": "4773ccda02ad27395bb30caf75333d4ae6afe99081ab6ac263dd10a1109a4752",
    ROOT / "runs/research/BACKLOG-HYPER01-FACTORIZED-02/PRE_REGISTRATION.md": "fa76b374e8b904686a224b1b7c2f663872d679295e7431b1062f545fcc6e4342",
    ROOT / "runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/PRE_REGISTRATION.md": "6969ca78bfcfa3e832895904c7dcea0f920f2e428a459051f2230b3b82da20f9",
    ROOT / "runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/RESULT.md": "f88b3e2934caa5726eb80529bdfb225735e0a54241a5dd1320c638ebed53249d",
    ROOT / "runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/raw/receipt.json": "60e9d83a46a06c65b28067c141b06d8bfdb9ba5009fbfbcdb1dce37afe2f2fa3",
    ROOT / "runs/research/BACKLOG-HYPER01-REAL-ADAPTER-01/REVIEW.json": "09073f38bb1d21e4ad2dc59f060ffb4a32c68288cda0d54848d3ac632a80ace5",
    CHECKPOINTS[0]: "ef5bec8822e856883eaec930d2b851892bb6b681bde1fda5f76005667adbf1a2",
    CHECKPOINTS[1]: "174832aa1bd25cbc5ed7f0ff717ad253ec94e2c23edc82e6f828ceadeed566b7",
    CHECKPOINTS[2]: "56ff9be8c5ac0876389cf12fe23a2ac301eac7c99cef977fa455b76f5817a2e6",
    CHECKPOINTS[3]: "dc696b7553cf8e4d920f8554ec4e3dee484a04da374ef0d54bcb48160044050a",
}


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(outdir: pathlib.Path) -> tuple[dict, dict]:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    artifacts = {}
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen hash mismatch: {path}: {actual}")
        artifacts[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    write_json(raw / "artifact_hashes.json", artifacts)

    service_before, gpu_before = query_service(), query_gpu()
    gateway_before = http_get_json("http://127.0.0.1:8080/health")
    embedding_before = http_get_json("http://127.0.0.1:8081/health")
    if service_before["active_state"] != "active" or gpu_before["memory_free_mib"] < 2048:
        raise RuntimeError(f"unsafe baseline service={service_before} gpu={gpu_before}")

    worker_json = raw / "worker.json"
    states = raw / "generator_states.safetensors"
    generated = raw / "generated_tensors.safetensors"
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--", PYTHON, windows_path_to_wsl(WORKER),
        "--checkpoints", *[windows_path_to_wsl(path) for path in CHECKPOINTS],
        "--output", windows_path_to_wsl(worker_json),
        "--state-output", windows_path_to_wsl(states),
        "--generated-output", windows_path_to_wsl(generated),
        "--steps", "1200", "--rank", "32", "--seeds", *[str(seed) for seed in SEEDS],
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(completed.stderr[-5000:])
    payload = json.loads(worker_json.read_text(encoding="utf-8"))

    independent_json = raw / "independent_evaluation.json"
    scorer_command = [
        "wsl", "-d", "Ubuntu-24.04", "--", PYTHON, windows_path_to_wsl(SCORER),
        "--checkpoints", *[windows_path_to_wsl(path) for path in CHECKPOINTS],
        "--generated", windows_path_to_wsl(generated), "--states", windows_path_to_wsl(states),
        "--seeds", *[str(seed) for seed in SEEDS],
        "--output", windows_path_to_wsl(independent_json),
    ]
    scored = subprocess.run(scorer_command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    (raw / "scorer.stdout.log").write_text(scored.stdout, encoding="utf-8")
    (raw / "scorer.stderr.log").write_text(scored.stderr, encoding="utf-8")
    if scored.returncode:
        raise RuntimeError(scored.stderr[-5000:])
    independent = json.loads(independent_json.read_text(encoding="utf-8"))
    recompute_delta = max(
        abs(left - right)
        for worker_row, scorer_row in zip(payload["seeds"], independent["rows"], strict=True)
        for left, right in zip(worker_row["target_cosines"], scorer_row["target_cosines"], strict=True)
    )
    independent_match = recompute_delta <= 1e-7

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
    direct_target_storage_mib = 4 * (8 * 1024 + 3584 * 8) * 4 / (1024 ** 2)
    retained_seed_states = independent["retained_seed_states"]
    observations = {
        "physical_adapter_targets": payload["physical_adapter_targets"],
        "distinct_target_deltas": payload["distinct_target_deltas"],
        "completed_seeds": payload["completed_seeds"],
        "median_synthesis_latency_ms": payload["median_synthesis_latency_ms"],
        "worst_seed_mean_cosine": independent["worst_seed_mean_cosine"],
        "generator_fp32_storage_mb": payload["generator_fp32_storage_mb"],
        "retained_seed_states": retained_seed_states,
        "independent_metric_recompute_match": independent_match,
        "service_and_embedding_unchanged": unchanged,
    }
    definitions = {
        "physical_targets": ("physical_adapter_targets", "eq", 4),
        "target_distinctness": ("distinct_target_deltas", "eq", 4),
        "completed_seeds": ("completed_seeds", "eq", 5),
        "latency": ("median_synthesis_latency_ms", "le", 5.0),
        "worst_seed_fidelity": ("worst_seed_mean_cosine", "ge", 0.95),
        "overhead": ("generator_fp32_storage_mb", "le", 20.0),
        "retained_states": ("retained_seed_states", "eq", 5),
        "independent_recompute": ("independent_metric_recompute_match", "eq", True),
        "service_recovery": ("service_and_embedding_unchanged", "eq", True),
    }
    ops = {"eq": lambda a, b: a == b, "le": lambda a, b: a <= b, "ge": lambda a, b: a >= b}
    gates = {gid: {"metric": metric, "operator": op, "threshold": threshold, "actual": observations[metric], "pass": ops[op](observations[metric], threshold)} for gid, (metric, op, threshold) in definitions.items()}

    write_json(raw / "actual_scores.json", observations | {
        "mean_weight_delta_cosine": independent["mean_weight_delta_cosine"],
        "peak_worker_allocated_mib": payload["peak_allocated_mib"],
        "peak_worker_reserved_mib": payload["peak_reserved_mib"],
        "direct_target_storage_mib": direct_target_storage_mib,
        "generator_to_direct_storage_ratio": payload["generator_fp32_storage_mb"] / direct_target_storage_mib,
    })
    write_json(raw / "hardware_metrics.json", {"before": gpu_before, "after": gpu_after, "worker_peak_allocated_mib": payload["peak_allocated_mib"], "worker_peak_reserved_mib": payload["peak_reserved_mib"]})
    write_json(raw / "service_maintenance.json", {"before": service_before, "after": service_after, "gateway_before": gateway_before, "gateway_after": gateway_after, "embedding_before": embedding_before, "embedding_after": embedding_after, "unchanged": unchanged})
    write_json(raw / "repeatability.json", {"seeds": SEEDS, "seed_mean_cosines": independent["rows"], "worst_seed_mean_cosine": independent["worst_seed_mean_cosine"]})
    write_json(raw / "retained_tensors.json", {"states": {"path": states.name, "bytes": states.stat().st_size, "sha256": sha256_file(states)}, "generated": {"path": generated.name, "bytes": generated.stat().st_size, "sha256": sha256_file(generated)}, "retained_seed_states": retained_seed_states})
    write_json(raw / "paired_baseline.json", {"r1_generator_fp32_mib": 72.7060546875, "direct_four_target_fp32_mib": direct_target_storage_mib, "r2_generator_fp32_mib": payload["generator_fp32_storage_mb"]})
    write_json(raw / "real_implementation.json", {"architecture": [64, 256, 512, 32, 36864], "rank": 32, "steps_per_seed": 1200, "worker_command": command, "scorer_command": scorer_command})
    write_json(raw / "semantic_parity.json", {"max_worker_scorer_cosine_delta": recompute_delta, "match": independent_match})
    write_json(raw / "target_materiality.json", {"targets": payload["target_ledger"], "distinct": payload["distinct_target_deltas"]})
    write_json(raw / "invalidation_rules.json", {"all_frozen_gates_pass": all(row["pass"] for row in gates.values()), "no_post_observation_changes": True})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in independent["rows"]:
            for target, cosine in enumerate(row["target_cosines"]):
                stream.write(json.dumps({"seed": row["seed"], "target": target, "weight_delta_cosine": cosine}) + "\n")

    evidence_files = [path for path in raw.iterdir() if path.name != "receipt.json"]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc, started_monotonic=started_mono,
        input_paths=[*EXPECTED, WORKER, SCORER, *evidence_files], packages=["pytest", "safetensors"],
        runtime={"execution_mode": "physical_factorized_lora_hypernetwork", "worker_command": command, "scorer_command": scorer_command},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(errors)
    evidence = {name: f"raw/{name}.json" for name in (
        "actual_scores", "hardware_metrics", "independent_evaluation", "invalidation_rules", "paired_baseline",
        "real_implementation", "repeatability", "retained_tensors", "semantic_parity", "service_maintenance", "target_materiality",
    )}
    evidence |= {"acceptance_gates": "raw/receipt.json", "artifact_hashes": "raw/artifact_hashes.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json"}
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": complete, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    return receipt, observations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, observations = run(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "HYPER01_COMPACT_PHYSICAL_FIT_R2" if passed else "HYPER01_COMPACT_NEGATIVE_R2"
    failed = [name for name, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Five physical-target seeds completed. "
        f"Worst-seed mean cosine `{observations['worst_seed_mean_cosine']:.9f}`; generator FP32 storage "
        f"`{observations['generator_fp32_storage_mb']:.6f} MiB`; failed gates: {', '.join(failed) if failed else 'none'}. "
        "This is a matched-target memorization/resource result only, not unseen-task or whole-adapter generalization.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
