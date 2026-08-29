#!/usr/bin/env python3
"""Rescore immutable SLX-08 R2 vectors with a canonical fixed-order metric."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research.run_adapter_requalification_r2 import (
    http_get_json,
    query_gpu,
    query_service,
    windows_path_to_wsl,
)
from tools.research.run_slx08_real_fidelity import normalize_exec, write_json

TASK_ID = "BACKLOG-SLX08-REAL-FIDELITY-03"
WSL_PYTHON = "/home/augus/.venvs/adapt00-20260824/bin/python"
SCORER = ROOT / "tools/research/slx08_canonical_scorer.py"
RUNNER = pathlib.Path(__file__).resolve()
SOURCE_PACKET = ROOT / "runs/research/BACKLOG-SLX08-REAL-FIDELITY-02"
SOURCE_RECEIPT = SOURCE_PACKET / "raw/receipt.json"
SOURCE_REVIEW = SOURCE_PACKET / "REVIEW.json"
SOURCE_BUNDLE = SOURCE_PACKET / "raw/context_vectors.safetensors"
SOURCE_SAMPLES = SOURCE_PACKET / "raw/samples.jsonl"
EXPECTED_INPUTS = {
    ROOT / "config/research_backlog_admissions/BACKLOG-SLX08-REAL-FIDELITY-03.json": "0b703704ce40191d1c6222778e4a2c52f2945dc160ad51c995f7ffcb763644a6",
    ROOT / "runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/PRE_REGISTRATION.md": "5d638b3a5a3941ce61c2d5ded6f6f57208c09b7a42b6b8d05799ca9bc247436b",
    SOURCE_RECEIPT: "9715c2fc0c82d7be475268eb3470127ad058e063a3041db34dc195935c1b9143",
    SOURCE_REVIEW: "e8947002265c7a2d8f33765a25e8a3fe49351a3d70341b0837682c87e8e0edf6",
    SOURCE_BUNDLE: "859ea9e3088de4e1f354a51a3c5502fd845ac1289dd0ad7b83d8c4f35b76cc58",
    SOURCE_SAMPLES: "e5e8bb64db6e3a1a680bc266601122caaaf0cc8f4d1b7903110aa8aa2f94017e",
    ROOT / "tools/research/slx08_real_fidelity_worker_r2.py": "5c4f966e95baf572e4e1942098b80a6be0a3d9c53de636ca43df37da41fd91ed",
    ROOT / "tools/research/slx08_context_scorer.py": "37d5d808e317494293bcbb3933ec25107a72526dcb238d8f280ac9bf2ca8e995",
}


def verify_inputs() -> dict:
    ledger = {}
    for path, expected in EXPECTED_INPUTS.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {path}: {actual} != {expected}")
        ledger[str(path.relative_to(ROOT).as_posix())] = {"bytes": path.stat().st_size, "sha256": actual}
    return ledger


def run_scorer(output: pathlib.Path, stdout: pathlib.Path, stderr: pathlib.Path) -> list[str]:
    command = [
        "wsl", "-d", "Ubuntu-24.04", "--", WSL_PYTHON,
        windows_path_to_wsl(SCORER),
        "--bundle", windows_path_to_wsl(SOURCE_BUNDLE),
        "--samples", windows_path_to_wsl(SOURCE_SAMPLES),
        "--output", windows_path_to_wsl(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False)
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"canonical scorer failed ({completed.returncode}): {completed.stderr[-5000:]}")
    return command


def run_experiment(outdir: pathlib.Path) -> dict:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError(f"raw directory is not empty: {raw}")
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    source_ledger = verify_inputs()
    before_service, before_gpu = query_service(), query_gpu()
    before_health = {
        "inference": http_get_json("http://127.0.0.1:8080/health"),
        "embedding": http_get_json("http://127.0.0.1:8081/health"),
    }
    first_path, second_path = raw / "canonical_evaluation.json", raw / "repeat_evaluation.json"
    first_command = run_scorer(first_path, raw / "scorer_first.stdout.log", raw / "scorer_first.stderr.log")
    second_command = run_scorer(second_path, raw / "scorer_second.stdout.log", raw / "scorer_second.stderr.log")
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    repeat_match = canonical_json_sha256(first) == canonical_json_sha256(second)
    after_service, after_gpu = query_service(), query_gpu()
    after_health = {
        "inference": http_get_json("http://127.0.0.1:8080/health"),
        "embedding": http_get_json("http://127.0.0.1:8081/health"),
    }
    unchanged = (
        before_service["active_state"] == after_service["active_state"] == "active"
        and before_service["main_pid"] == after_service["main_pid"]
        and before_service["n_restarts"] == after_service["n_restarts"]
        and normalize_exec(before_service["exec_start"]) == normalize_exec(after_service["exec_start"])
        and before_health["inference"].get("status") == after_health["inference"].get("status") == "ok"
        and before_health["embedding"].get("status") == after_health["embedding"].get("status") == "ok"
    )
    service = {
        "before_service": before_service, "after_service": after_service,
        "before_health": before_health, "after_health": after_health,
        "serving_process_unchanged": unchanged,
        "inference_requests_issued": 0, "service_actions_issued": 0,
    }
    write_json(raw / "service_maintenance.json", service)
    source_receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    write_json(raw / "source_receipt_binding.json", {
        "path": str(SOURCE_RECEIPT.relative_to(ROOT).as_posix()),
        "sha256": sha256_file(SOURCE_RECEIPT),
        "receipt_fingerprint": source_receipt["receipt_fingerprint"],
        "source_receipt_bound": True,
    })
    write_json(raw / "bundle_identity.json", {
        "path": str(SOURCE_BUNDLE.relative_to(ROOT).as_posix()),
        "sha256": sha256_file(SOURCE_BUNDLE), "bytes": SOURCE_BUNDLE.stat().st_size,
        "bundle_sha256_match": sha256_file(SOURCE_BUNDLE) == EXPECTED_INPUTS[SOURCE_BUNDLE],
    })
    write_json(raw / "dataset_hashes.json", source_ledger)
    summary = first["summary"]
    write_json(raw / "actual_scores.json", {**summary, "canonical_repeat_match": repeat_match})
    write_json(raw / "independent_evaluation.json", {
        "first_sha256": canonical_json_sha256(first),
        "second_sha256": canonical_json_sha256(second),
        "canonical_repeat_match": repeat_match,
    })
    write_json(raw / "paired_baseline.json", {
        "rows": [{"cell": row["cell"], "corrected": row["selected_block_context_cosine"], "legacy": row["legacy_first_half_context_cosine"]} for row in first["rows"]]
    })
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in first["rows"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(raw / "real_implementation.json", {
        "scorer": str(SCORER.relative_to(ROOT).as_posix()),
        "physical_inference": False, "retained_evidence_only": True,
    })
    write_json(raw / "scorer_hashes.json", {
        "scorer": {"path": str(SCORER.relative_to(ROOT).as_posix()), "sha256": sha256_file(SCORER)},
        "runner": {"path": str(RUNNER.relative_to(ROOT).as_posix()), "sha256": sha256_file(RUNNER)},
    })
    write_json(raw / "semantic_parity.json", {
        "canonical_repeat_match": repeat_match,
        "tensor_hash_match_rate": summary["tensor_hash_match_rate"],
    })
    write_json(raw / "treatment_controls.json", {
        "method": first["method"], "invocations": 2,
        "new_inference": False, "threshold_changed": False,
    })
    write_json(raw / "hardware_metrics.json", {
        "before_gpu": before_gpu, "after_gpu": after_gpu,
        "elapsed_seconds": time.monotonic() - started_mono,
        "gpu_inference_used": False,
    })
    observations = {
        "source_receipt_bound": True,
        "bundle_sha256_match": sha256_file(SOURCE_BUNDLE) == EXPECTED_INPUTS[SOURCE_BUNDLE],
        **summary,
        "canonical_repeat_match": repeat_match,
        "serving_process_unchanged": unchanged,
    }
    definitions = {
        "source_receipt_binding": ("source_receipt_bound", "eq", True),
        "bundle_identity": ("bundle_sha256_match", "eq", True),
        "bundle_key_contract": ("retained_context_tensors", "eq", 36),
        "cell_coverage": ("retained_context_cells", "eq", 12),
        "tensor_identity": ("tensor_hash_match_rate", "eq", 1.0),
        "finite_contexts": ("nonfinite_values", "eq", 0),
        "canonical_repeat": ("canonical_repeat_match", "eq", True),
        "fidelity": ("canonical_median_selected_block_context_cosine", "ge", 0.95),
        "service_untouched": ("serving_process_unchanged", "eq", True),
    }
    operators = {"eq": lambda actual, threshold: actual == threshold, "ge": lambda actual, threshold: actual >= threshold}
    gates = {gate_id: {"metric": metric, "operator": operator, "threshold": threshold, "actual": observations[metric], "pass": operators[operator](observations[metric], threshold)} for gate_id, (metric, operator, threshold) in definitions.items()}
    raw_inputs = [raw / name for name in (
        "actual_scores.json", "bundle_identity.json", "canonical_evaluation.json", "dataset_hashes.json",
        "hardware_metrics.json", "independent_evaluation.json", "paired_baseline.json", "real_implementation.json",
        "repeat_evaluation.json", "samples.jsonl", "scorer_first.stderr.log", "scorer_first.stdout.log",
        "scorer_hashes.json", "scorer_second.stderr.log", "scorer_second.stdout.log", "semantic_parity.json",
        "service_maintenance.json", "source_receipt_binding.json", "treatment_controls.json",
    )]
    provenance = build_provenance(
        script_path=RUNNER, started_at_utc=started_utc, started_monotonic=started_mono,
        input_paths=[*EXPECTED_INPUTS, SCORER, RUNNER, *raw_inputs], packages=["pytest"],
        runtime={"execution_mode": "retained_slx08_canonical_rescore_r3", "first_command": first_command, "second_command": second_command},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "bundle_identity": "raw/bundle_identity.json", "canonical_evaluation": "raw/canonical_evaluation.json",
        "dataset_hashes": "raw/dataset_hashes.json", "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json",
        "repeat_evaluation": "raw/repeat_evaluation.json", "scorer_hashes": "raw/scorer_hashes.json",
        "semantic_parity": "raw/semantic_parity.json", "service_maintenance": "raw/service_maintenance.json",
        "source_receipt_binding": "raw/source_receipt_binding.json", "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": complete, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    claim = "SLX08_FIDELITY_FALSE_NEGATIVE_CANONICAL_RESCORE_R3" if all(gate["pass"] for gate in gates.values()) else "SLX08_FIDELITY_CANONICAL_RESCORE_NEGATIVE_R3"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Canonical retained-context median: `{summary['canonical_median_selected_block_context_cosine']:.12f}`; "
        f"repeat match: `{repeat_match}`. No physical inference or service action was performed.\n",
        encoding="utf-8",
    )
    write_json(raw / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt = run_experiment(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
