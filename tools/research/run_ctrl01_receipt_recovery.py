#!/usr/bin/env python3
"""Canonical independent rescore of immutable CTRL-01 real-token samples."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research.run_ctrl01_real_token import percentile, runtime_binding_evidence

TASK_ID = "BACKLOG-CTRL01-REAL-TOKEN-05"
SOURCE_DIR = ROOT / "runs/research/BACKLOG-CTRL01-REAL-TOKEN-04"
SOURCE_SAMPLES = SOURCE_DIR / "raw/samples.jsonl"
SOURCE_RECEIPT = SOURCE_DIR / "raw/receipt.json"
EXPECTED = {
    SOURCE_DIR / "PRE_REGISTRATION.md": "d34b12631dbde35bec0ebb62eadb05e912924d6b17c26e609cdb9ace12d25f87",
    SOURCE_SAMPLES: "94ef7a81b5b7bd83d8c300bbc68f15852b622ce2180d37d1fbdfe8a802248ec5",
    SOURCE_RECEIPT: "3a097f0d20e4f8c7c2217899fe35bc72406d44b51143d8b04d163ee964a8d7fc",
    ROOT / "runs/research/CTRL-01-AST-SIDECAR-2026-08-25/raw/receipt.json": "0f37ae1d3ff33286a193353731f864d699ce738734fd8cc5b5a55384c2cf2c7c",
    ROOT / "tools/analysis/ast_grammar_sidecar.py": "3cb90b1b5aa5aacdff93b7a8b0cdc38e689099e0d1365989f00b7b34acbb1463",
}


def write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def rescore(rows: list[dict[str, Any]], runtime_integrated: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rescored = []
    for row in rows:
        raw_reconstructed = "".join(str(piece) for piece in row["token_pieces"])
        filtered_reconstructed = "".join(
            str(decision["piece"]) for decision in row["decisions"] if decision["accepted"]
        )
        if raw_reconstructed != row["raw_content"]:
            raise ValueError("token pieces do not reconstruct raw content")
        if filtered_reconstructed != row["filtered"]:
            raise ValueError("accepted decisions do not reconstruct filtered content")
        rescored.append({
            **row,
            "independent_raw_json_valid": valid_json(row["raw_content"]),
            "independent_filtered_json_valid": valid_json(row["filtered"]),
            "independent_exact_preservation": row["raw_content"] == row["filtered"],
        })
    real = [row for row in rescored if row["kind"] == "real_model"]
    controls = [row for row in rescored if row["kind"] == "valid_control"]
    if len(real) != 24 or len(controls) != 12:
        raise ValueError(f"unexpected row coverage: real={len(real)} controls={len(controls)}")
    control_tokens = sum(len(row["decisions"]) for row in controls)
    accepted = sum(sum(bool(decision["accepted"]) for decision in row["decisions"]) for row in controls)
    latencies = [float(row["us_per_token"]) for row in rescored]
    metrics = {
        "real_model_outputs": len(real),
        "raw_complete_valid_rate": sum(row["independent_raw_json_valid"] for row in real) / len(real),
        "sanitized_complete_valid_rate": sum(row["independent_filtered_json_valid"] for row in real) / len(real),
        "real_exact_preservation_rate": sum(row["independent_exact_preservation"] for row in real) / len(real),
        "valid_controls": len(controls),
        "valid_token_acceptance_rate": accepted / control_tokens,
        "valid_control_exact_preservation_rate": sum(row["independent_exact_preservation"] for row in controls) / len(controls),
        "valid_control_complete_valid_rate": sum(row["independent_filtered_json_valid"] for row in controls) / len(controls),
        "p50_overhead_us_per_token": statistics.median(latencies),
        "p95_overhead_us_per_token": percentile(latencies, 0.95),
        "logit_mask_runtime_integrated": runtime_integrated,
    }
    return rescored, metrics


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

    source_rows = [json.loads(line) for line in SOURCE_SAMPLES.read_text(encoding="utf-8").splitlines()]
    source_receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    binding = runtime_binding_evidence()
    rescored, metrics = rescore(source_rows, binding["logit_mask_runtime_integrated"])
    source_metrics = source_receipt["metrics"]
    metric_match = all(metrics[key] == source_metrics[key] for key in metrics)
    if not metric_match:
        raise ValueError({"source": source_metrics, "rescored": metrics})

    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in rescored:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write(raw / "actual_scores.json", metrics)
    write(raw / "artifact_hashes.json", ledger)
    write(raw / "dataset_hashes.json", {
        "source_samples_sha256": EXPECTED[SOURCE_SAMPLES],
        "rescored_semantic_sha256": canonical_json_sha256(rescored),
    })
    write(raw / "source_execution_receipt.json", {
        "source_receipt_sha256": EXPECTED[SOURCE_RECEIPT],
        "source_samples_sha256": EXPECTED[SOURCE_SAMPLES],
        "physical_rows": len(source_rows),
        "source_verdict": source_receipt.get("verdict"),
        "source_model": source_receipt.get("provenance", {}).get("models", {}).get("data", [{}])[0].get("id"),
    })
    write(raw / "falsifiable_hypothesis.json", {
        "mandatory_gates": ["real_coverage", "real_validity", "valid_control_recall", "valid_control_semantics", "overhead", "runtime_binding"]
    })
    write(raw / "invariant_controls.json", {
        "source_rows": 36,
        "real_rows": 24,
        "valid_control_rows": 12,
        "source_metric_exact_match": metric_match,
        "no_new_inference": True,
    })
    write(raw / "failure_reproduction.json", {
        "historical_claim": {"constrained_valid_pct": 100.0, "mean_overhead_us_per_token": 7.88, "verdict": "PROMOTED"},
        "real_token_metrics": metrics,
        "failed_metrics": [
            "sanitized_complete_valid_rate", "valid_token_acceptance_rate",
            "valid_control_exact_preservation_rate", "logit_mask_runtime_integrated",
        ],
    })
    write(raw / "independent_evaluation.json", {
        "source_metric_exact_match": metric_match,
        "every_raw_reconstructed_from_token_pieces": True,
        "every_filtered_reconstructed_from_decisions": True,
        "runtime_binding_evidence": binding,
    })
    write(raw / "invalidation_rules.json", {
        "qualification_requires_all_gates": True,
        "posthoc_repair_forbidden": True,
        "offline_filter_is_not_runtime_binding": True,
    })
    write(raw / "semantic_parity.json", {
        "real_exact_preservation_rate": metrics["real_exact_preservation_rate"],
        "valid_control_exact_preservation_rate": metrics["valid_control_exact_preservation_rate"],
    })

    defs = {
        "real_coverage": ("real_model_outputs", "ge", 24),
        "real_validity": ("sanitized_complete_valid_rate", "eq", 1.0),
        "valid_control_recall": ("valid_token_acceptance_rate", "eq", 1.0),
        "valid_control_semantics": ("valid_control_exact_preservation_rate", "eq", 1.0),
        "overhead": ("p95_overhead_us_per_token", "le", 500.0),
        "runtime_binding": ("logit_mask_runtime_integrated", "eq", True),
    }
    ops = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b}
    gates = {
        gate: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": ops[operator](metrics[metric], threshold)}
        for gate, (metric, operator, threshold) in defs.items()
    }
    evidence_names = (
        "actual_scores.json", "artifact_hashes.json", "dataset_hashes.json", "failure_reproduction.json",
        "falsifiable_hypothesis.json", "independent_evaluation.json", "invalidation_rules.json",
        "invariant_controls.json", "samples.jsonl", "semantic_parity.json", "source_execution_receipt.json",
    )
    evidence_files = [raw / name for name in evidence_names]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=started_mono,
        input_paths=[
            ROOT / f"config/research_backlog_admissions/{TASK_ID}.json",
            outdir / "PRE_REGISTRATION.md", *EXPECTED, *evidence_files,
        ],
        packages=["pytest"],
        runtime={"execution_mode": "independent_rescore_of_immutable_real_token_samples", "new_inference": False},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json",
        "dataset_hashes": "raw/dataset_hashes.json",
        "failure_reproduction": "raw/failure_reproduction.json",
        "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "invalidation_rules": "raw/invalidation_rules.json",
        "invariant_controls": "raw/invariant_controls.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json",
        "semantic_parity": "raw/semantic_parity.json",
        "source_execution_receipt": "raw/source_execution_receipt.json",
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": True,
        "gates": gates,
        "evidence": evidence,
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write(raw / "receipt.json", receipt)
    return receipt


def write_result(outdir: pathlib.Path, receipt: dict[str, Any]) -> None:
    gates = receipt["gates"]
    failed = [name for name, value in gates.items() if not value["pass"]]
    scores = json.loads((outdir / "raw/actual_scores.json").read_text(encoding="utf-8"))
    (outdir / "RESULT.md").write_text(f"""# BACKLOG-CTRL01-REAL-TOKEN-05 result

## Verdict

`CTRL01_FALSE_POSITIVE_CONFIRMED_R5` pending independent AGY review.

Independent recovery reproduced all source metrics exactly from 36 immutable physical rows. The real model's raw JSON validity was `{scores['raw_complete_valid_rate']:.6f}`, while the sidecar reduced complete validity to `{scores['sanitized_complete_valid_rate']:.6f}`. Valid-token acceptance was `{scores['valid_token_acceptance_rate']:.6f}`, valid-control exact preservation was `{scores['valid_control_exact_preservation_rate']:.6f}`, p95 overhead was `{scores['p95_overhead_us_per_token']:.3f}` microseconds/token, and runtime logit-mask integration was `{scores['logit_mask_runtime_integrated']}`.

Failed mandatory gates: `{', '.join(failed)}`. The historical 100% claim depended on synthetic chunks plus post-filter brace repair and does not survive real tokenizer pieces without repair.

This is an independent rescore/provenance recovery of the already executed real run, not new inference and not a Python-mode evaluation.
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
