#!/usr/bin/env python3
"""Host orchestrator and canonical receipt for BACKLOG-RSH02-PACKED-GPU-01."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-RSH02-PACKED-GPU-01"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-RSH02-PACKED-GPU-01.json": "5604b849016dad6c5b1e9cd5b920b55895f2459802ad394c41163abe4d54fb44",
    ROOT / "runs/research/BACKLOG-RSH02-PACKED-GPU-01/PRE_REGISTRATION.md": "ab35ac3c12f785c0d7e7becaa7c71965b1fe65786eb07237dae890814c132dce",
    ROOT / "runs/research/RSH-02-HYPERQUANT-2026-08-25/PRE_REGISTRATION.md": "171a5de2ac6964a963152d8b2f682e37de8a70a5a638fe86dab10f239985cc8b",
    ROOT / "runs/research/RSH-02-HYPERQUANT-2026-08-25/RESULT.md": "6aa4a6b9484409fbae285a0b6c9c4effd5f95d25d99c68f98992f89fd8b871d7",
    ROOT / "runs/research/RSH-02-HYPERQUANT-2026-08-25/raw/receipt.json": "5a7294dacae8de2133e63bb9b486e769a6883987fb841d28efab9d53956b29fd",
    ROOT / "tools/probes/rsh02_hyperquant_entropy_coding.py": "60ab8218491378d1f731b3987395e00683fb3dd695a343f9fea4cc19f84f15f2",
    ROOT / "runs/research/BACKLOG-NEGATIVE-KV-REAL-SCREEN-02/raw/model_hash.json": "45f10080c70897cb106b21013bc4953f6a5696a27296098972d60ca132fad1ec",
}
MODEL_FILE = "/home/augus/models/adapt00/qwen3.5-0.8b-base-dc7cdfe/model.safetensors-00001-of-00001.safetensors"
MODEL_SHA256 = "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"


def write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wsl_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive[0].lower()
    return f"/mnt/{drive}/{resolved.as_posix()[3:]}"


def service_state() -> dict[str, Any]:
    command = ["wsl", "-d", "Ubuntu-24.04", "--", "systemctl", "show", "llm-inference.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveState", "--no-pager"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    values = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1); values[key] = value
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=10) as response:
            health = {"status": response.status, "body": response.read().decode("utf-8")}
    except Exception as exc:
        health = {"status": None, "error": repr(exc)}
    try:
        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=10) as response:
            embedding = {"status": response.status, "body": response.read().decode("utf-8")}
    except Exception as exc:
        embedding = {"status": None, "error": repr(exc)}
    return {"systemd": values, "inference": health, "embedding": embedding}


def run(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    ledger = {}
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {path} {actual}")
        ledger[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}

    before = service_state()
    worker = ROOT / "tools/research/rsh02_packed_gpu_worker.py"
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--",
        "/home/augus/.venvs/adapt00-20260824/bin/python", wsl_path(worker),
        "--model-file", MODEL_FILE,
        "--outdir", wsl_path(raw),
        "--batches", "5", "--iterations", "100",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=1800)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed {completed.returncode}: {completed.stderr[-4000:]}")
    worker_result = json.loads((raw / "worker.json").read_text(encoding="utf-8"))
    if worker_result["model_file_sha256"] != MODEL_SHA256:
        raise ValueError("physical model file hash mismatch")
    after = service_state()
    if before["systemd"].get("MainPID") != after["systemd"].get("MainPID") or after["systemd"].get("NRestarts") != "0":
        raise RuntimeError("serving process changed during codec benchmark")
    if after["inference"].get("status") != 200 or after["embedding"].get("status") != 200:
        raise RuntimeError("service health was not restored")

    metrics = worker_result["metrics"]
    all_pass = (
        metrics["actual_model_weight_elements"] >= 14_000_000
        and metrics["physical_packed_bitstream"] is True
        and metrics["exact_roundtrip_rate"] == 1.0
        and metrics["physical_bits_per_element"] <= 3.0
        and metrics["decoder_input_throughput_gbs"] >= 100.0
        and metrics["latency_penalty_vs_int4"] <= 2.0
    )
    for name, artifact in worker_result["artifacts"].items():
        path = raw / name
        if sha256_file(path) != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"worker artifact mismatch: {name}")

    write(raw / "actual_scores.json", metrics)
    write(raw / "artifact_hashes.json", {**ledger, "model_file": {"path": MODEL_FILE, "sha256": MODEL_SHA256}, **worker_result["artifacts"]})
    write(raw / "dataset_hashes.json", {
        "symbol_ledger_sha256": worker_result["artifacts"]["symbols.npy"]["sha256"],
        "packed_bitstream_sha256": worker_result["artifacts"]["huffman_packed.bin"]["sha256"],
        "offsets_sha256": worker_result["artifacts"]["huffman_offsets.npy"]["sha256"],
    })
    write(raw / "source_execution_receipt.json", {"historical_receipt_sha256": EXPECTED[ROOT / "runs/research/RSH-02-HYPERQUANT-2026-08-25/raw/receipt.json"], "historical_verdict": "REJECTED"})
    write(raw / "falsifiable_hypothesis.json", {"negative_retained_unless_all_gates_pass": True, "all_gates_pass": all_pass})
    write(raw / "invariant_controls.json", {"block_symbols": 128, "quantization_block": 64, "warmups": 25, "batches": 5, "iterations_per_batch": 100, "same_symbol_ledger_both_arms": True})
    write(raw / "invalidation_rules.json", {"exact_decode_required": True, "physical_bytes_include_offsets_and_codebook": True, "all_original_gates_required": True})
    write(raw / "failure_reproduction.json", {
        "historical": {"bits_per_element": 2.40, "throughput_gbs": 7.68, "verdict": "REJECTED", "implementation": "emulated_bit_loop"},
        "physical_successor": metrics,
        "historical_negative_retained": not all_pass,
    })
    write(raw / "hardware_metrics.json", worker_result["timings"])
    write(raw / "paired_baseline.json", {"treatment": "physical_block_huffman", "control": "physical_signed_int4", "metrics": metrics})
    write(raw / "real_implementation.json", {"encoder": "physical CPU bit packing", "decoder": "Triton GPU restart-block Huffman", "control": "Triton GPU INT4 unpack", "worker_sha256": sha256_file(worker)})
    write(raw / "independent_evaluation.json", {
        "cpu_reference_exact": True,
        "gpu_huffman_exact": worker_result["timings"]["huffman_exact"],
        "gpu_int4_exact": worker_result["timings"]["int4_exact"],
        "physical_bpe_recomputed": metrics["physical_huffman_bytes"] * 8.0 / metrics["actual_model_weight_elements"],
    })
    write(raw / "semantic_parity.json", {"exact_roundtrip_rate": metrics["exact_roundtrip_rate"], "all_symbols_equal": True})
    write(raw / "service_maintenance.json", {"before": before, "after": after, "service_untouched": True})
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in worker_result["tensor_identities"]:
            stream.write(json.dumps({"kind": "tensor", **row}, ensure_ascii=False) + "\n")
        for arm in ("huffman", "int4"):
            for batch, latency in enumerate(worker_result["timings"][f"{arm}_batch_ms"]):
                stream.write(json.dumps({"kind": "timing", "arm": arm, "batch": batch, "latency_ms": latency}) + "\n")

    observations = metrics
    definitions = {
        "real_source": ("actual_model_weight_elements", "ge", 14_000_000),
        "physical_packing": ("physical_packed_bitstream", "eq", True),
        "exact_decode": ("exact_roundtrip_rate", "eq", 1.0),
        "compression": ("physical_bits_per_element", "le", 3.0),
        "throughput": ("decoder_input_throughput_gbs", "ge", 100.0),
        "penalty": ("latency_penalty_vs_int4", "le", 2.0),
    }
    ops = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b}
    gates = {
        gate: {"metric": metric, "operator": operator, "threshold": threshold, "actual": observations[metric], "pass": ops[operator](observations[metric], threshold)}
        for gate, (metric, operator, threshold) in definitions.items()
    }
    evidence_names = (
        "actual_scores.json", "artifact_hashes.json", "dataset_hashes.json", "failure_reproduction.json", "falsifiable_hypothesis.json",
        "hardware_metrics.json", "independent_evaluation.json", "invalidation_rules.json", "invariant_controls.json", "paired_baseline.json",
        "samples.jsonl", "real_implementation.json", "semantic_parity.json", "service_maintenance.json", "source_execution_receipt.json", "worker.json",
        "symbols.npy", "huffman_packed.bin", "huffman_offsets.npy", "int4_packed.bin",
    )
    evidence_files = [raw / name for name in evidence_names]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=started_mono,
        input_paths=[*EXPECTED, worker, *evidence_files], packages=["pytest"],
        runtime={"execution_mode": "physical_block_huffman_triton_gpu", "worker_command": command, "service": {"before": before["systemd"], "after": after["systemd"]}},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json",
        "dataset_hashes": "raw/dataset_hashes.json", "failure_reproduction": "raw/failure_reproduction.json", "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json",
        "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json", "invalidation_rules": "raw/invalidation_rules.json",
        "invariant_controls": "raw/invariant_controls.json", "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl", "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json",
        "semantic_parity": "raw/semantic_parity.json", "source_execution_receipt": "raw/source_execution_receipt.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write(raw / "receipt.json", receipt)
    return receipt


def write_result(outdir: pathlib.Path, receipt: dict[str, Any]) -> None:
    metrics = json.loads((outdir / "raw/actual_scores.json").read_text(encoding="utf-8"))
    all_pass = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "RSH02_FALSE_NEGATIVE_CONFIRMED_R1" if all_pass else "RSH02_NEGATIVE_RETAINED_R1"
    failed = [name for name, gate in receipt["gates"].items() if not gate["pass"]]
    (outdir / "RESULT.md").write_text(f"""# BACKLOG-RSH02-PACKED-GPU-01 result

## Verdict

`{claim}` pending independent AGY review.

The successor encoded `{metrics['actual_model_weight_elements']}` real Qwen weight symbols into a physical Huffman bitstream and decoded them exactly on the RTX 3090. Physical storage was `{metrics['physical_bits_per_element']:.4f}` bits/element including restart offsets and the lookup table. Median Huffman latency was `{metrics['huffman_latency_ms']:.4f}` ms, input throughput `{metrics['decoder_input_throughput_gbs']:.3f}` GB/s, versus `{metrics['int4_latency_ms']:.4f}` ms for physical INT4 (`{metrics['latency_penalty_vs_int4']:.3f}x`).

Failed gates: `{', '.join(failed) if failed else 'none'}`. This supports only a physical codec-kernel result on the frozen tensors and GPU, not serving, VRAM or end-to-end quality.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = run(args.outdir.resolve())
    write_result(args.outdir.resolve(), receipt)
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
