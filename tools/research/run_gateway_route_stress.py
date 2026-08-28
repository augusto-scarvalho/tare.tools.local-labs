#!/usr/bin/env python3
"""Repeated cold-switch stress test for every qualified gateway alias."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import run_fleet_regression_screen as fleet


TASK_ID = "BACKLOG-GATEWAY-ROUTE-STRESS-01"
MODELS = ("qwen38", "hauhaucs", "fable-tc", "qwen36-moe", "gemma-vision", "muse-vision")
PRE_REG_SHA256 = "15928db4b6c40047430c3445dcf59ffa7b3643ada4120b3f69c0e5940a06d506"
SOURCES = {
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "tools/serving/qualified_model_gateway.py": "b4afb885e1b23ed0bc227620f4140038eeb6bcfc2b066d645666ce07e10a3ccb",
    "docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md": "895fec3ac345bdf26350b4a97f513bf4f4b3bad9898d09701db07a985f8b7d55",
}
PROBES = (
    ("exact", "Reply with exactly ROUTE_OK and nothing else."),
    ("arithmetic", "Compute 37 * 19. Reply with only the integer."),
    ("json", 'Return only compact JSON with keys "alpha" and "beta" mapped to 2 and 5.'),
    ("recall", "Remember the code word LANTERN. The code word is LANTERN. What is the code word? Reply only with it."),
)


def write_json(path: pathlib.Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def verify_inputs() -> tuple[dict[str, Any], list[pathlib.Path]]:
    ledger: dict[str, Any] = {"host": {}, "artifacts": {}}
    paths = []
    for relative, expected in SOURCES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source identity mismatch: {relative}: {actual}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        paths.append(path)
    prereg = ROOT / "runs/research" / TASK_ID / "PRE_REGISTRATION.md"
    actual = sha256_file(prereg)
    if actual != PRE_REG_SHA256:
        raise ValueError(f"preregistration identity mismatch: {actual}")
    ledger["host"][str(prereg.relative_to(ROOT)).replace("\\", "/")] = {"bytes": prereg.stat().st_size, "sha256": actual}
    paths.append(prereg)
    registry = json.loads((ROOT / "config/qualified_model_fleet.json").read_text(encoding="utf-8"))
    for model in MODELS:
        artifact = registry["models"][model]["artifact"]
        size = fleet.wsl("stat", "-c", "%s", artifact["path"])
        digest = fleet.wsl("sha256sum", artifact["path"], timeout=1800.0)
        actual_hash = digest["stdout"].split()[0] if digest["returncode"] == 0 else ""
        if size["returncode"] != 0 or int(size["stdout"]) != artifact["bytes"] or actual_hash != artifact["sha256"]:
            raise ValueError(f"artifact identity mismatch for {model}")
        ledger["artifacts"][model] = {"path": artifact["path"], "bytes": artifact["bytes"], "sha256": actual_hash}
    return ledger, paths


def payload(model: str, prompt: str) -> dict[str, Any]:
    return {
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0.0, "seed": 20260826,
        "stream": False, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    samples_path = raw / "samples.jsonl"
    switches_path = raw / "switches.jsonl"
    state_path = raw / "runner_state.json"
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    frozen, frozen_paths = verify_inputs()
    write_json(raw / "frozen_inputs.json", frozen)
    initial_service = fleet.service_state()
    initial_gateway = fleet.gateway_status()
    initial_model = str(initial_gateway["current_model"])
    if initial_service["active_state"] != "active" or fleet.embedding_health() != 200:
        raise RuntimeError("gateway or embedding endpoint unhealthy")
    rows = read_jsonl(samples_path)
    switches = read_jsonl(switches_path)
    completed = {(int(row["cycle"]), row["model"], row["probe"]) for row in rows}
    state = {"task_id": TASK_ID, "started_at_utc": started_utc, "status": "running", "initial_service": initial_service, "initial_gateway": initial_gateway, "initial_model": initial_model, "recorded_requests": len(rows)}
    write_json(state_path, state)
    error: str | None = None
    try:
        for cycle in range(5):
            order = MODELS[cycle:] + MODELS[:cycle]
            for model in order:
                status, gpu = fleet.switch_model(model)
                switch = {"cycle": cycle, "model": model, "order": list(order), "status": status, "gpu": gpu, "embedding_health": fleet.embedding_health()}
                append_jsonl(switches_path, switch)
                switches.append(switch)
                if switch["embedding_health"] != 200:
                    raise RuntimeError(f"embedding failure after switching to {model}")
                consecutive = 0
                for probe_id, prompt in PROBES:
                    key = (cycle, model, probe_id)
                    if key in completed:
                        continue
                    begin = time.perf_counter()
                    code, response = fleet.http_json(f"{fleet.BASE_URL}/v1/chat/completions", payload(model, prompt))
                    wall_ms = round((time.perf_counter() - begin) * 1000, 3)
                    projection = fleet.semantic_projection(response)
                    row = {"cycle": cycle, "model": model, "probe": probe_id, "http_status": code, "error": response.get("_error"), "wall_ms": wall_ms, "response": response, "semantic_projection": projection, "semantic_sha256": canonical_json_sha256(projection)}
                    append_jsonl(samples_path, row)
                    rows.append(row)
                    completed.add(key)
                    consecutive = consecutive + 1 if code != 200 else 0
                    state.update({"recorded_requests": len(rows), "last": list(key)})
                    write_json(state_path, state)
                    print(f"{len(rows):03d}/120 cycle={cycle} model={model} probe={probe_id} http={code}", flush=True)
                    if consecutive >= 3:
                        raise RuntimeError(f"three consecutive failures on {model}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        state.update({"status": "aborted", "error": error})
        write_json(state_path, state)
        raise
    finally:
        try:
            restored_status, restored_gpu = fleet.switch_model(initial_model)
            restoration = {"status": restored_status, "gpu": restored_gpu, "embedding_health": fleet.embedding_health()}
        except Exception as exc:
            restoration = {"error": f"{type(exc).__name__}: {exc}"}
            if error is None:
                state.update({"status": "aborted", "error": restoration["error"]})
        state["restoration"] = restoration
        write_json(state_path, state)

    rows = read_jsonl(samples_path)
    switches = read_jsonl(switches_path)
    baseline = {(row["model"], row["probe"]): row["semantic_sha256"] for row in rows if row["cycle"] == 0}
    comparisons = [row["semantic_sha256"] == baseline[(row["model"], row["probe"])] for row in rows if row["cycle"] > 0]
    final_service = fleet.service_state()
    restoration = state["restoration"]
    metrics = {
        "verified_switches": sum(s["status"].get("current_model") == s["model"] and s["status"].get("backend_healthy") is True for s in switches),
        "recorded_requests": len(rows), "successful_response_rate": sum(row["http_status"] == 200 for row in rows) / len(rows),
        "route_identity_rate": sum(s["status"].get("current_model") == s["model"] for s in switches) / len(switches),
        "exact_cycle_repeat_rate": sum(comparisons) / len(comparisons),
        "service_restarts": final_service["n_restarts"] - initial_service["n_restarts"],
        "embedding_boundary_successes": sum(s["embedding_health"] == 200 for s in switches),
        "initial_model_restored": restoration.get("status", {}).get("current_model") == initial_model and restoration.get("embedding_health") == 200,
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "effective_route.json", {"switches": switches})
    write_json(raw / "hardware_metrics.json", {"latency_p50_ms": statistics.median(row["wall_ms"] for row in rows), "switch_gpu_snapshots": [s["gpu"] for s in switches]})
    write_json(raw / "independent_evaluation.json", {"all_rows_rescored": True, "metrics": metrics, "independent_review_pending": True})
    write_json(raw / "paired_baseline.json", {"cycle_zero": 0, "comparison_cycles": [1, 2, 3, 4], "comparisons": len(comparisons)})
    write_json(raw / "recovery_state.json", restoration)
    write_json(raw / "service_identity.json", {"initial": initial_service, "final": final_service, "gateway_initial": initial_gateway})
    write_json(raw / "service_maintenance.json", {"gateway_service_stopped": False, "embedding_service_stopped": False, "initial_model_restored": metrics["initial_model_restored"]})
    write_json(raw / "treatment_controls.json", {"models": MODELS, "cycles": 5, "probes": [p[0] for p in PROBES], "temperature": 0.0, "seed": 20260826})
    definitions = {
        "switch_coverage": ("verified_switches", "eq", 30), "request_coverage": ("recorded_requests", "eq", 120),
        "request_integrity": ("successful_response_rate", "eq", 1.0), "route_identity": ("route_identity_rate", "eq", 1.0),
        "cycle_repeatability": ("exact_cycle_repeat_rate", "ge", 0.95), "service_integrity": ("service_restarts", "eq", 0),
        "embedding_integrity": ("embedding_boundary_successes", "eq", 30), "service_recovery": ("initial_model_restored", "eq", True),
    }
    gates = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}
    evidence_files = sorted(path for path in raw.iterdir() if path.is_file())
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc, started_monotonic=started_mono, input_paths=[*frozen_paths, *evidence_files], packages=[], runtime={"execution_mode": "gateway_cold_switch_stress", "switches": len(switches), "requests": len(rows)})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    evidence = {"acceptance_gates": "raw/receipt.json", "effective_route": "raw/effective_route.json", "service_identity": "raw/service_identity.json", "paired_baseline": "raw/paired_baseline.json", "recovery_state": "raw/recovery_state.json", "hardware_metrics": "raw/hardware_metrics.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json", "independent_evaluation": "raw/independent_evaluation.json", "treatment_controls": "raw/treatment_controls.json", "service_maintenance": "raw/service_maintenance.json"}
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    passed = all(gate["pass"] for gate in gates.values())
    claim = "QUALIFIED_GATEWAY_ROUTE_STRESS_PASSED_R1" if passed else "QUALIFIED_GATEWAY_ROUTE_STRESS_REJECTED_R1"
    failed = [gate for gate, result in gates.items() if not result["pass"]]
    (outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nCompleted `{metrics['verified_switches']}/30` verified switches and `{metrics['recorded_requests']}/120` requests with repeat rate `{metrics['exact_cycle_repeat_rate']:.6f}`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n", encoding="utf-8", newline="\n")
    state.update({"status": "completed", "claim": claim, "failed_gates": failed})
    write_json(state_path, state)
    return receipt


def selfcheck() -> None:
    assert len(MODELS) == 6 and len(PROBES) == 4
    assert sum(len(MODELS[index:] + MODELS[:index]) for index in range(5)) == 30
    print("gateway route stress self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck(); return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = subprocess.run([sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex executor"], cwd=ROOT, capture_output=True, text=True, check=False)
    print(json.dumps({"returncode": advance.returncode, "stdout": advance.stdout, "stderr": advance.stderr}, indent=2), flush=True)
    return 0 if advance.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
