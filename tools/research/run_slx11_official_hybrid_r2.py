#!/usr/bin/env python3
"""Qualify the official hybrid checkpoint with retained, rescored logits."""
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
from tools.research import run_slx11_official_hybrid as base


TASK_ID = "BACKLOG-SLX11-OFFICIAL-HYBRID-02"
MODEL = base.MODEL
PROMPTS = base.PROMPTS
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-SLX11-OFFICIAL-HYBRID-02.json": "7e199fdc5df92f333fa4e50281d789e770412924a59709409dd00c599ad24225",
    "runs/research/BACKLOG-SLX11-OFFICIAL-HYBRID-01/raw/receipt.json": "d539dda7fb419ae390be32eb82e1d2c34ac44a44845c5992d505da35cdc28caf",
    "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
    "tools/research/run_slx11_official_hybrid.py": "43e5259b7fbbf6fdb733d40d696fe65ba44fb756e1ad5e4b2816fab48b25e049",
    "tools/research/slx11_official_hybrid_worker.py": "94e6b0fbae2aed4c97bbea0072a182b9dbf4df691caf80c08776c2de6ceb36ae",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def write(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def run_command(command: list[str], timeout: float = 1800.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)


def aggregate(worker: dict, evaluation: dict, unchanged: int) -> dict:
    physical = worker["physical_layers"]
    return {
        "official_checkpoint_identified": int(worker["model_type"] == "qwen3_5" and worker["text_model_type"] == "qwen3_5_text"),
        "hybrid_layer_types_verified": sum(row["match"] for row in physical),
        "physical_recurrent_layers": sum(row["actual"] == "linear_attention" for row in physical),
        "physical_full_attention_layers": sum(row["actual"] == "full_attention" for row in physical),
        "successful_live_forwards": len(worker["samples"]),
        "retained_logits_tensors": evaluation["retained_logits_tensors"],
        "recomputed_finite_output_rate": evaluation["recomputed_finite_output_rate"],
        "recomputed_projection_match_rate": evaluation["recomputed_projection_match_rate"],
        "serving_process_unchanged": unchanged,
    }


def run(outdir: pathlib.Path) -> dict:
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    pipeline = json.loads((outdir / "PIPELINE.json").read_text(encoding="utf-8"))
    prereg = ROOT / pipeline["preregistration"]["path"]
    if sha256_file(prereg) != pipeline["preregistration"]["sha256"]:
        raise ValueError("pipeline preregistration binding mismatch")
    sources = []
    source_ledger = {}
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source mismatch: {relative}: {actual} != {expected}")
        sources.append(path)
        source_ledger[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    sources.append(prereg)
    source_ledger[pipeline["preregistration"]["path"]] = {"sha256": sha256_file(prereg), "bytes": prereg.stat().st_size}

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    before = base.service_state()
    worker = ROOT / "tools/research/slx11_official_hybrid_worker_r2.py"
    scorer = ROOT / "tools/research/slx11_logits_scorer.py"
    worker_json = raw / "worker.json"
    logits_bundle = raw / "logits.safetensors"
    evaluation_path = raw / "logits_evaluation.json"
    worker_command = [
        "wsl", "-d", "Ubuntu-24.04", "--", "/home/augus/.venvs/adapt00-20260824/bin/python",
        base.wsl_path(worker), "--model", MODEL, "--prompts", base.wsl_path(PROMPTS),
        "--output", base.wsl_path(worker_json), "--logits-output", base.wsl_path(logits_bundle),
    ]
    completed = run_command(worker_command)
    (raw / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8", newline="\n")
    (raw / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed {completed.returncode}: {completed.stderr[-5000:]}")
    scorer_command = [
        "wsl", "-d", "Ubuntu-24.04", "--", "/home/augus/.venvs/adapt00-20260824/bin/python",
        base.wsl_path(scorer), "--metadata", base.wsl_path(worker_json),
        "--bundle", base.wsl_path(logits_bundle), "--output", base.wsl_path(evaluation_path),
    ]
    scored = run_command(scorer_command)
    (raw / "scorer.stdout.log").write_text(scored.stdout, encoding="utf-8", newline="\n")
    (raw / "scorer.stderr.log").write_text(scored.stderr, encoding="utf-8", newline="\n")
    if scored.returncode != 0:
        raise RuntimeError(f"scorer failed {scored.returncode}: {scored.stderr[-5000:]}")

    worker_result = json.loads(worker_json.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    after = base.service_state()
    unchanged = int(
        before["systemd"].get("MainPID") == after["systemd"].get("MainPID")
        and before["systemd"].get("NRestarts") == after["systemd"].get("NRestarts") == "0"
        and after["health"] == {"8080": 200, "8081": 200}
    )
    metrics = aggregate(worker_result, evaluation, unchanged)
    write(raw / "actual_scores.json", metrics)
    write(raw / "artifact_hashes.json", {"sources": source_ledger, "checkpoint": worker_result["checkpoint"], "logits_bundle": {"sha256": sha256_file(logits_bundle), "bytes": logits_bundle.stat().st_size}})
    write(raw / "dataset_hashes.json", {"prompt_file_sha256": worker_result["prompt_file_sha256"], "panel_semantic_sha256": canonical_json_sha256([row["task_id"] for row in worker_result["samples"]])})
    write(raw / "effective_route.json", {"model": worker_result["model"], "worker_command": worker_command, "scorer_command": scorer_command})
    write(raw / "hardware_metrics.json", worker_result["hardware"])
    write(raw / "independent_evaluation.json", {"metrics": metrics, "projection_rows": len(evaluation["projections"]), "derived_from_retained_bundle": True})
    write(raw / "paired_baseline.json", {"declared_topology": [row["declared"] for row in worker_result["physical_layers"]], "physical_topology": [row["actual"] for row in worker_result["physical_layers"]]})
    write(raw / "scorer_hashes.json", {"runner_sha256": sha256_file(pathlib.Path(__file__).resolve()), "worker_sha256": sha256_file(worker), "scorer_sha256": sha256_file(scorer)})
    write(raw / "service_maintenance.json", {"before": before, "after": after, "service_untouched": bool(unchanged)})
    write(raw / "treatment_controls.json", {"checkpoint": MODEL, "forwards": 24, "dtype": "bfloat16", "use_cache": False, "next_token_only": True})
    with (raw / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in worker_result["samples"]:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    definitions = {
        "official_artifact": ("official_checkpoint_identified", 1),
        "hybrid_topology": ("hybrid_layer_types_verified", 24),
        "recurrent_layers": ("physical_recurrent_layers", 18),
        "attention_layers": ("physical_full_attention_layers", 6),
        "live_forward": ("successful_live_forwards", 24),
        "logits_bundle_coverage": ("retained_logits_tensors", 24),
        "finite_outputs": ("recomputed_finite_output_rate", 1.0),
        "logits_projection_match": ("recomputed_projection_match_rate", 1.0),
        "runtime_unchanged": ("serving_process_unchanged", 1),
    }
    gates = {
        name: {"metric": metric, "operator": "eq", "threshold": threshold, "actual": metrics[metric], "pass": metrics[metric] == threshold}
        for name, (metric, threshold) in definitions.items()
    }
    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json",
        "dataset_hashes": "raw/dataset_hashes.json",
        "effective_route": "raw/effective_route.json",
        "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "logits_bundle": "raw/logits.safetensors",
        "logits_evaluation": "raw/logits_evaluation.json",
        "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json",
        "scorer_hashes": "raw/scorer_hashes.json",
        "service_maintenance": "raw/service_maintenance.json",
        "treatment_controls": "raw/treatment_controls.json",
    }
    generated = sorted(path for path in raw.iterdir() if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=mono,
        input_paths=[*sources, worker, scorer, *generated],
        packages=["torch", "transformers", "safetensors"],
        runtime={"execution_mode": "official_hybrid_retained_logits", "worker_command": worker_command, "scorer_command": scorer_command},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(errors)
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write(raw / "receipt.json", receipt)
    passed = all(gate["pass"] for gate in gates.values())
    claim = "SLX11_OFFICIAL_HYBRID_ARTIFACT_QUALIFIED_WITH_LOGITS_R2" if passed else "SLX11_OFFICIAL_HYBRID_ARTIFACT_REJECTED_WITH_LOGITS_R2"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Matched `{metrics['hybrid_layer_types_verified']}/24` layers and independently rescored "
        f"`{metrics['retained_logits_tensors']}` retained logits tensors with finite rate "
        f"`{metrics['recomputed_finite_output_rate']:.6f}`. Historical speed, recall and quality claims remain excluded.\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        metrics = aggregate(
            {"model_type": "qwen3_5", "text_model_type": "qwen3_5_text", "physical_layers": [{"match": True, "actual": "linear_attention"}] * 18 + [{"match": True, "actual": "full_attention"}] * 6, "samples": [{}] * 24},
            {"retained_logits_tensors": 24, "recomputed_finite_output_rate": 1.0, "recomputed_projection_match_rate": 1.0},
            1,
        )
        assert metrics["hybrid_layer_types_verified"] == 24
        return 0
    receipt = run(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
