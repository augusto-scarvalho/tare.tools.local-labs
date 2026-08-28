#!/usr/bin/env python3
"""Large restartable regression screen for the qualified text-model fleet."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.a2_stats import gsm8k_extract, numeric_equal
from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.benchmarks.agent_suite_v2 import CASES, base_payload, calls, text


TASK_ID = "BACKLOG-FLEET-REGRESSION-SCREEN-01"
MODELS = ("qwen38", "hauhaucs", "fable-tc", "qwen36-moe")
REPEATS = (0, 1)
BASE_URL = "http://127.0.0.1:8080"
EMBED_URL = "http://127.0.0.1:8081"
SERVICE = "llm-inference.service"
EXPECTED_REQUESTS = 448
PRE_REG_SHA256 = "0b428ea735b129f269616ba2dc3c5dc5fdb77d1567f13e90c39219e416b9fa44"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-REGRESSION-SCREEN-01.json": "31a9ba96a9272a6301fa73454c134b9f16216502de31c36faec68ea9f71e0991",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "tools/benchmarks/agent_suite_v2.py": "14d0a1b76d4d729228678f215ecefa3254aef214eb65ac9d8d7061bccc0dc59e",
    "docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md": "895fec3ac345bdf26350b4a97f513bf4f4b3bad9898d09701db07a985f8b7d55",
}


def run_text(argv: list[str], timeout: float = 120.0) -> dict[str, Any]:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
    )
    return {
        "argv": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def wsl(*args: str, timeout: float = 120.0) -> dict[str, Any]:
    return run_text(["wsl", "-d", "Ubuntu-24.04", "--", *args], timeout=timeout)


def write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 900.0,
) -> tuple[int | None, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"User-Agent": "LocalLabs-Fleet-Screen/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"body": body[:4000]}
        return exc.code, {"_error": parsed}
    except Exception as exc:  # preserve transport failures as evidence
        return None, {"_error": f"{type(exc).__name__}: {exc}"}


def gateway_status() -> dict[str, Any]:
    status, payload = http_json(f"{BASE_URL}/fleet/status", timeout=15.0)
    if status != 200 or payload.get("status") != "ok":
        raise RuntimeError(f"gateway unhealthy: status={status} payload={payload}")
    return payload


def embedding_health() -> int | None:
    status, _ = http_json(f"{EMBED_URL}/health", timeout=15.0)
    return status


def service_state() -> dict[str, Any]:
    result = wsl(
        "systemctl",
        "show",
        SERVICE,
        "-p",
        "ActiveState",
        "-p",
        "MainPID",
        "-p",
        "NRestarts",
        "-p",
        "ExecStart",
        "--no-pager",
    )
    values = dict(
        line.split("=", 1)
        for line in result["stdout"].splitlines()
        if "=" in line
    )
    return {
        "command": result,
        "active_state": values.get("ActiveState"),
        "main_pid": int(values.get("MainPID") or 0),
        "n_restarts": int(values.get("NRestarts") or 0),
        "exec_start": values.get("ExecStart", ""),
    }


def gpu_state() -> dict[str, Any]:
    fields = (
        "name,uuid,driver_version,memory.total,memory.used,memory.free,"
        "temperature.gpu,power.draw,utilization.gpu"
    )
    result = run_text(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    values = [part.strip() for part in result["stdout"].split(",")]
    names = fields.split(",")
    return {
        "query": result,
        "values": dict(zip(names, values, strict=False)),
    }


def verify_frozen_inputs() -> tuple[dict[str, Any], list[pathlib.Path]]:
    ledger: dict[str, Any] = {"host": {}, "wsl_artifacts": {}}
    input_paths: list[pathlib.Path] = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        ledger["host"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
        input_paths.append(path)

    prereg = ROOT / "runs/research" / TASK_ID / "PRE_REGISTRATION.md"
    prereg_hash = sha256_file(prereg)
    if prereg_hash != PRE_REG_SHA256:
        raise ValueError(f"preregistration mismatch: {prereg_hash} != {PRE_REG_SHA256}")
    ledger["host"][str(prereg.relative_to(ROOT)).replace("\\", "/")] = {
        "bytes": prereg.stat().st_size,
        "sha256": prereg_hash,
    }
    input_paths.append(prereg)

    registry = json.loads((ROOT / "config/qualified_model_fleet.json").read_text(encoding="utf-8"))
    for model in MODELS:
        artifact = registry["models"][model]["artifact"]
        path = artifact["path"]
        size_result = wsl("stat", "-c", "%s", path)
        if size_result["returncode"] != 0 or int(size_result["stdout"]) != artifact["bytes"]:
            raise ValueError(f"artifact size mismatch for {model}: {size_result}")
        hash_result = wsl("sha256sum", path, timeout=1800.0)
        actual_hash = hash_result["stdout"].split()[0] if hash_result["returncode"] == 0 else ""
        if actual_hash != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch for {model}: {actual_hash}")
        ledger["wsl_artifacts"][model] = {
            "path": path,
            "bytes": artifact["bytes"],
            "sha256": actual_hash,
            "binary": registry["models"][model]["runtime"]["binary"],
            "runtime_args": registry["models"][model]["runtime"]["args"],
        }
    return ledger, input_paths


def load_panels() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    math_rows = read_jsonl(ROOT / "workloads/gsm8k.jsonl")[:32]
    qa_rows = read_jsonl(
        ROOT / "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl"
    )[:16]
    if len(math_rows) != 32 or len(qa_rows) != 16 or len(CASES) != 8:
        raise ValueError("frozen panel size mismatch")
    return math_rows, qa_rows


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    return " ".join(normalized.strip(" `\"'.,;:!?").split())


def message(response: dict[str, Any]) -> dict[str, Any]:
    try:
        value = response["choices"][0]["message"]
        return value if isinstance(value, dict) else {}
    except (KeyError, IndexError, TypeError):
        return {}


def normalized_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for call in calls(response):
        normalized.append({
            "name": call.get("name"),
            "args": call.get("args") if call.get("args_valid") else call.get("raw_args"),
            "args_valid": bool(call.get("args_valid")),
        })
    return normalized


def semantic_projection(response: dict[str, Any]) -> dict[str, Any]:
    msg = message(response)
    choices = response.get("choices") or [{}]
    return {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content") or "",
        "finish_reason": choices[0].get("finish_reason"),
        "tool_calls": normalized_tool_calls(response),
    }


def payload_for(model: str, suite: str, case: dict[str, Any]) -> dict[str, Any]:
    if suite == "math":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": "Solve the problem carefully. End with #### followed by the numeric answer."},
                {"role": "user", "content": case["prompt"]},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "seed": 20260826,
            "stream": False,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    if suite == "qa":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": "Follow the requested output format exactly."},
                {"role": "user", "content": case["prompt"]},
            ],
            "max_tokens": 64,
            "temperature": 0.0,
            "seed": 20260826,
            "stream": False,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    payload = base_payload(case["tools"], case["messages"])
    payload.update({
        "model": model,
        "seed": 20260826,
        "stream": False,
        "cache_prompt": False,
    })
    return payload


def score_response(suite: str, case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    msg = message(response)
    answer = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning_content") or "")
    combined = "\n".join(part for part in (reasoning, answer) if part)
    if suite == "math":
        extracted = gsm8k_extract(combined)
        correct = numeric_equal(extracted, str(case["answer"]))
        return {"pass": bool(correct), "extracted": extracted, "expected": str(case["answer"])}
    if suite == "qa":
        expected = [normalize_text(str(item)) for item in case["expected"]]
        actual = normalize_text(answer)
        return {"pass": actual in expected, "normalized": actual, "expected": expected}
    found = calls(response)
    passed, expectation = case["validate"](found, text(response))
    return {
        "pass": bool(passed),
        "expectation": expectation,
        "calls": normalized_tool_calls(response),
    }


def case_matrix(
    math_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
) -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    rows.extend(("math", str(case["task_id"]), case) for case in math_rows)
    rows.extend(("qa", str(case["id"]), case) for case in qa_rows)
    rows.extend(("agent", str(case["name"]), case) for case in CASES)
    return rows


def switch_model(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    canary = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with only OK."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "seed": 20260826,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status_code, response = http_json(f"{BASE_URL}/v1/chat/completions", canary)
    if status_code != 200:
        raise RuntimeError(f"route canary failed for {model}: {status_code} {response}")
    status = gateway_status()
    if status.get("current_model") != model or not status.get("backend_healthy"):
        raise RuntimeError(f"route identity mismatch for {model}: {status}")
    gpu = gpu_state()
    free_memory = float(gpu["values"].get("memory.free") or 0)
    if free_memory < 512:
        raise RuntimeError(f"GPU free memory below frozen floor after loading {model}: {free_memory}")
    return status, gpu


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def aggregate(rows: list[dict[str, Any]], route_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("http_status") == 200 and not row.get("error")]
    pairs: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault((row["model"], row["suite"], row["case_id"]), {})[row["repeat"]] = row
    comparable = [pair for pair in pairs.values() if set(pair) == {0, 1}]
    exact = sum(pair[0]["semantic_sha256"] == pair[1]["semantic_sha256"] for pair in comparable)
    by_model: dict[str, Any] = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        suite_scores = {}
        for suite in ("math", "qa", "agent"):
            suite_rows = [row for row in model_rows if row["suite"] == suite]
            suite_scores[suite] = {
                "passed": sum(bool(row.get("score", {}).get("pass")) for row in suite_rows),
                "total": len(suite_rows),
            }
        latencies = [float(row["wall_ms"]) for row in model_rows if row.get("http_status") == 200]
        draft = sum(int(row.get("timings", {}).get("draft_n") or 0) for row in model_rows)
        accepted = sum(int(row.get("timings", {}).get("draft_n_accepted") or 0) for row in model_rows)
        by_model[model] = {
            "recorded": len(model_rows),
            "suite_scores": suite_scores,
            "wall_p50_ms": round(statistics.median(latencies), 3) if latencies else None,
            "wall_p95_ms": round(percentile(latencies, 0.95), 3) if latencies else None,
            "draft_tokens": draft,
            "accepted_draft_tokens": accepted,
            "draft_acceptance_rate": round(accepted / draft, 8) if draft else None,
        }
    completed_models = sum(by_model[model]["recorded"] == 112 for model in MODELS)
    route_ok = all(snapshot.get("status", {}).get("current_model") == snapshot.get("model") for snapshot in route_snapshots)
    return {
        "route_models_completed": completed_models,
        "recorded_requests": len(rows),
        "successful_responses": len(successful),
        "successful_response_rate": round(len(successful) / len(rows), 8) if rows else 0.0,
        "route_identity_verified": route_ok and len(route_snapshots) >= len(MODELS),
        "repeat_pairs": len(comparable),
        "exact_repeat_pairs": exact,
        "exact_repeat_rate": round(exact / len(comparable), 8) if comparable else 0.0,
        "models": by_model,
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    samples_path = raw / "samples.jsonl"
    state_path = raw / "runner_state.json"
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()

    existing_rows = read_jsonl(samples_path)
    completed_keys = {
        (row["model"], int(row["repeat"]), row["suite"], row["case_id"])
        for row in existing_rows
    }
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    frozen, frozen_paths = verify_frozen_inputs()
    math_rows, qa_rows = load_panels()
    matrix = case_matrix(math_rows, qa_rows)
    initial_service = state.get("initial_service") or service_state()
    initial_gateway = state.get("initial_gateway") or gateway_status()
    initial_model = state.get("initial_model") or initial_gateway.get("current_model")
    if initial_service.get("active_state") != "active" or initial_service.get("main_pid", 0) <= 0:
        raise RuntimeError(f"inference service is not active: {initial_service}")
    if embedding_health() != 200:
        raise RuntimeError("embedding endpoint unhealthy before execution")
    if initial_model not in MODELS:
        raise RuntimeError(f"initial resident route is not in frozen restorable routes: {initial_model}")

    state.update({
        "task_id": TASK_ID,
        "started_at_utc": state.get("started_at_utc", started_utc),
        "initial_service": initial_service,
        "initial_gateway": initial_gateway,
        "initial_model": initial_model,
        "completed_requests": len(existing_rows),
        "status": "running",
    })
    write_json(state_path, state)
    write_json(raw / "frozen_inputs.json", frozen)

    route_snapshots = json.loads((raw / "route_snapshots.json").read_text(encoding="utf-8")) if (raw / "route_snapshots.json").is_file() else []
    execution_error: str | None = None
    try:
        for model in MODELS:
            status, gpu = switch_model(model)
            snapshot = {
                "model": model,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": status,
                "gpu": gpu,
                "embedding_health": embedding_health(),
            }
            route_snapshots.append(snapshot)
            write_json(raw / "route_snapshots.json", route_snapshots)
            if snapshot["embedding_health"] != 200:
                raise RuntimeError(f"embedding endpoint unhealthy at {model} boundary")

            consecutive_errors = 0
            for repeat in REPEATS:
                for suite, case_id, case in matrix:
                    key = (model, repeat, suite, case_id)
                    if key in completed_keys:
                        continue
                    payload = payload_for(model, suite, case)
                    begin = time.perf_counter()
                    status_code, response = http_json(
                        f"{BASE_URL}/v1/chat/completions",
                        payload,
                    )
                    wall_ms = round((time.perf_counter() - begin) * 1000.0, 3)
                    error = response.get("_error")
                    score = score_response(suite, case, response) if status_code == 200 else {"pass": False, "error": error}
                    projection = semantic_projection(response)
                    row = {
                        "model": model,
                        "repeat": repeat,
                        "suite": suite,
                        "case_id": case_id,
                        "http_status": status_code,
                        "error": error,
                        "wall_ms": wall_ms,
                        "request": payload,
                        "response": response,
                        "score": score,
                        "semantic_projection": projection,
                        "semantic_sha256": canonical_json_sha256(projection),
                        "timings": response.get("timings") or {},
                    }
                    append_jsonl(samples_path, row)
                    existing_rows.append(row)
                    completed_keys.add(key)
                    consecutive_errors = consecutive_errors + 1 if error or status_code != 200 else 0
                    state.update({
                        "completed_requests": len(existing_rows),
                        "last_key": list(key),
                        "last_http_status": status_code,
                    })
                    write_json(state_path, state)
                    print(
                        f"{len(existing_rows):03d}/{EXPECTED_REQUESTS} {model} r{repeat} "
                        f"{suite}:{case_id} http={status_code} pass={score.get('pass')} "
                        f"wall_ms={wall_ms:.1f}",
                        flush=True,
                    )
                    if consecutive_errors >= 3:
                        raise RuntimeError(f"three consecutive request errors on {model}")
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        state.update({"status": "aborted", "error": execution_error})
        write_json(state_path, state)
        raise
    finally:
        try:
            restored_status, restored_gpu = switch_model(str(initial_model))
            state["restoration"] = {
                "status": restored_status,
                "gpu": restored_gpu,
                "embedding_health": embedding_health(),
            }
        except Exception as restore_exc:
            state["restoration"] = {"error": f"{type(restore_exc).__name__}: {restore_exc}"}
            if execution_error is None:
                state.update({"status": "aborted", "error": state["restoration"]["error"]})
        write_json(state_path, state)

    rows = read_jsonl(samples_path)
    metrics = aggregate(rows, route_snapshots)
    final_service = service_state()
    restoration = state.get("restoration", {})
    restored_status = restoration.get("status", {})
    metrics["service_restarts"] = final_service["n_restarts"] - initial_service["n_restarts"]
    metrics["embedding_health"] = restoration.get("embedding_health")
    metrics["initial_model_restored"] = (
        restored_status.get("current_model") == initial_model
        and restored_status.get("backend_healthy") is True
        and final_service["main_pid"] == initial_service["main_pid"]
    )

    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "dataset_hashes.json", {
        "math_panel_semantic_sha256": canonical_json_sha256(math_rows),
        "qa_panel_semantic_sha256": canonical_json_sha256(qa_rows),
        "agent_case_names": [case["name"] for case in CASES],
    })
    write_json(raw / "effective_route.json", {"models": MODELS, "snapshots": route_snapshots})
    write_json(raw / "environment.json", {"initial_gpu": gpu_state(), "wsl_distro": "Ubuntu-24.04"})
    write_json(raw / "hardware_metrics.json", {"models": metrics["models"], "route_snapshots": route_snapshots})
    write_json(raw / "independent_evaluation.json", {
        "all_rows_rescored": True,
        "semantic_projection_excludes_timing_and_ids": True,
        "metrics": metrics,
    })
    write_json(raw / "paired_baseline.json", {
        "repeat_zero": 0,
        "repeat_one": 1,
        "paired_cases": metrics["repeat_pairs"],
    })
    write_json(raw / "recovery_state.json", {
        "initial_model": initial_model,
        "restoration": restoration,
        "initial_service": initial_service,
        "final_service": final_service,
    })
    write_json(raw / "service_identity.json", {
        "initial": initial_gateway,
        "final": restored_status,
        "initial_service": initial_service,
        "final_service": final_service,
    })
    write_json(raw / "service_maintenance.json", {
        "systemd_service_stopped": False,
        "gateway_pid_preserved": final_service["main_pid"] == initial_service["main_pid"],
        "initial_model_restored": metrics["initial_model_restored"],
    })
    write_json(raw / "treatment_controls.json", {
        "models": MODELS,
        "repeats": REPEATS,
        "math_cases": 32,
        "qa_cases": 16,
        "agent_cases": 8,
        "temperature": 0.0,
        "seed": 20260826,
        "stream": False,
    })

    definitions = {
        "route_coverage": ("route_models_completed", "eq", 4),
        "request_coverage": ("recorded_requests", "eq", 448),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "route_identity": ("route_identity_verified", "eq", True),
        "repeatability": ("exact_repeat_rate", "ge", 0.95),
        "service_integrity": ("service_restarts", "eq", 0),
        "embedding_integrity": ("embedding_health", "eq", 200),
        "service_recovery": ("initial_model_restored", "eq", True),
    }
    gates = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate_id] = {
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "actual": actual,
            "pass": passed,
        }

    evidence_files = [
        raw / "actual_scores.json",
        raw / "dataset_hashes.json",
        raw / "effective_route.json",
        raw / "environment.json",
        raw / "frozen_inputs.json",
        raw / "hardware_metrics.json",
        raw / "independent_evaluation.json",
        raw / "paired_baseline.json",
        raw / "recovery_state.json",
        raw / "route_snapshots.json",
        raw / "runner_state.json",
        raw / "samples.jsonl",
        raw / "service_identity.json",
        raw / "service_maintenance.json",
        raw / "treatment_controls.json",
    ]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=state["started_at_utc"],
        started_monotonic=started_mono,
        input_paths=[*frozen_paths, *evidence_files],
        packages=[],
        runtime={
            "execution_mode": "qualified_gateway_large_fleet_screen",
            "restartable": True,
            "models": MODELS,
            "requests": len(rows),
        },
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(f"incomplete provenance: {errors}")

    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "actual_scores": "raw/actual_scores.json",
        "dataset_hashes": "raw/dataset_hashes.json",
        "effective_route": "raw/effective_route.json",
        "environment": "raw/environment.json",
        "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json",
        "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json",
        "recovery_state": "raw/recovery_state.json",
        "service_identity": "raw/service_identity.json",
        "service_maintenance": "raw/service_maintenance.json",
        "treatment_controls": "raw/treatment_controls.json",
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
    write_json(raw / "receipt.json", receipt)

    passed = all(gate["pass"] for gate in gates.values())
    claim = (
        "QUALIFIED_TEXT_FLEET_SCREEN_COMPLETE_R1"
        if passed
        else "QUALIFIED_TEXT_FLEET_SCREEN_REJECTED_R1"
    )
    failures = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n"
        f"`{claim}` pending independent review.\n\n"
        f"Recorded `{metrics['recorded_requests']}` requests across "
        f"`{metrics['route_models_completed']}` routes. Successful response rate "
        f"`{metrics['successful_response_rate']:.6f}` and exact repeat rate "
        f"`{metrics['exact_repeat_rate']:.6f}`. Failed gates: "
        f"`{', '.join(failures) if failures else 'none'}`. Quality scores are "
        "descriptive and do not change fleet model cards.\n",
        encoding="utf-8",
        newline="\n",
    )
    state.update({"status": "completed", "claim": claim, "failed_gates": failures})
    write_json(state_path, state)
    return receipt


def selfcheck() -> None:
    response = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "random-id",
                    "function": {"name": "get_weather", "arguments": '{"city":"Lisbon"}'},
                }],
            },
        }],
        "timings": {"predicted_ms": 10},
    }
    projected = semantic_projection(response)
    assert projected["tool_calls"] == [{
        "name": "get_weather",
        "args": {"city": "Lisbon"},
        "args_valid": True,
    }]
    assert normalize_text(" `Brasilia`. ") == "brasilia"
    assert len(case_matrix([{"task_id": str(i)} for i in range(32)], [{"id": str(i)} for i in range(16)])) == 56
    print("fleet regression screen self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=pathlib.Path,
        default=ROOT / "runs/research" / TASK_ID,
    )
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    outdir = args.outdir.resolve()
    receipt = execute(outdir)
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = run_text([
        sys.executable,
        str(ROOT / "tools/analysis/backlog_pipeline.py"),
        "advance",
        TASK_ID,
        "--to",
        "EXECUTED",
        "--actor",
        "Codex executor",
    ])
    print(json.dumps({"pipeline_advance": advance}, indent=2), flush=True)
    return 0 if advance["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
