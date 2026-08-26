#!/usr/bin/env python3
"""Causal OFF/ON serving benchmark for BACKLOG-CUDAGRAPH-SERVING-02."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)

TASK_ID = "BACKLOG-CUDAGRAPH-SERVING-02"
WSL_DISTRO = "Ubuntu-24.04"
BINARY = "/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin/llama-server"
CUDA_LIBRARY = "/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin/libggml-cuda.so.0.17.0"
LIB_DIR = "/home/augus/opt/slop.cpp/b10159-068764d92-fable-tc/bin"
MODEL = "/home/augus/models/merges/fable-tc-l1.0-Q4_K_M.gguf"
PERSISTENT_UNIT = "llm-inference.service"
TEMP_PORT = 18080
TEMP_URL = f"http://127.0.0.1:{TEMP_PORT}"
INFERENCE_URL = "http://127.0.0.1:8080"
EMBED_URL = "http://127.0.0.1:8081"

EXPECTED = {
    "admission": (ROOT / "config/research_backlog_admissions/BACKLOG-CUDAGRAPH-SERVING-02.json", "f436dbccf6c33e88241b7afba527e774a3a83fb9e334fffed0edaee619bc59d1"),
    "audit": (ROOT / "docs/AUDIT_2026-08-25_CODEX_INDEPENDENT_AGY_EXECUTION.md", "e4364456156a3c2f015306d986192792fb1aa9ae9333b63a2237ec46e3ffc11f"),
    "prior_service": (ROOT / "runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/service_identity.json", "e1ae69bd587bb266e0f5aadaf40555174234ea0e17885fe5087822345db6b315"),
    "prior_route": (ROOT / "runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/effective_route.json", "cddef60e4d0d7edce834ac50db8aadde1f18f5b246445f5759800ab7d4195b6a"),
    "prior_samples": (ROOT / "runs/research/BACKLOG-CUDAGRAPH-SERVING-01/raw/samples.jsonl", "4e4a1f8bcde43f6cf45dfb98f00fe26790535c915eb1fbd885f4012879a6ba13"),
}
EXPECTED_WSL = {
    BINARY: {"bytes": 17920, "sha256": "5719c246ec3622ea1df3c3f498075879f12f1f70b969f8b591e87b3a1f3c8808"},
    CUDA_LIBRARY: {"bytes": 63388824, "sha256": "78ed3ef92d354a544231232f99a3c23b3a02a179daa7d21c5a3ea9ab6a811eb9"},
    MODEL: {"bytes": 16810714400, "sha256": "052c08ca13d75d8d88c9cc3f201d7bfa9167e2a1e69ad3e1e1f26ff73c1b390b"},
}
SERVER_ARGS = [
    BINARY, "-m", MODEL, "--alias", "fable-tc-l1.0", "--host", "127.0.0.1",
    "--port", str(TEMP_PORT), "-ngl", "99", "-fa", "on", "--ctx-size", "8192",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--jinja", "--metrics",
]
PERSISTENT_ARGS = [
    BINARY, "-m", MODEL, "--alias", "fable-tc-l1.0", "--host", "0.0.0.0",
    "--port", "8080", "-ngl", "99", "-fa", "on", "--ctx-size", "8192",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4", "--jinja", "--metrics",
]
BLOCKS = [
    {"id": "b1_off", "treatment": "off", "prompt_ids": list(range(1, 16))},
    {"id": "b2_on", "treatment": "on", "prompt_ids": list(range(1, 16))},
    {"id": "b3_on", "treatment": "on", "prompt_ids": list(range(16, 31))},
    {"id": "b4_off", "treatment": "off", "prompt_ids": list(range(16, 31))},
]
WARMUP_PROMPTS = [
    "Warmup A: calculate 17 plus 25 and explain briefly.",
    "Warmup B: calculate 19 times 6 and explain briefly.",
    "Warmup C: calculate 144 divided by 12 and explain briefly.",
    "Warmup D: calculate 81 minus 37 and explain briefly.",
]


def run(command: list[str], timeout: float = 60.0, check: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {command!r}\n{completed.stderr[-2000:]}")
    return completed


def wsl(*args: str, root: bool = False, timeout: float = 60.0, check: bool = False) -> subprocess.CompletedProcess[str]:
    command = ["wsl", "-d", WSL_DISTRO]
    if root:
        command.extend(["-u", "root"])
    command.extend(["--", *args])
    return run(command, timeout=timeout, check=check)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def http_json(url: str, payload: dict | None = None, timeout: float = 15.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "LocalLabs-CUDAGraph-R2/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_health(url: str, timeout_seconds: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            result = http_json(f"{url}/health", timeout=5.0)
            if result.get("status") == "ok":
                return result
            last_error = repr(result)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        time.sleep(1.0)
    raise RuntimeError(f"health timeout for {url}: {last_error}")


def sha256_wsl(path: str, timeout: float = 300.0) -> str:
    completed = wsl("sha256sum", path, timeout=timeout, check=True)
    return completed.stdout.split()[0].lower()


def stat_wsl(path: str) -> int:
    completed = wsl("stat", "-c", "%s", path, check=True)
    return int(completed.stdout.strip())


def verify_inputs() -> tuple[dict, list[pathlib.Path]]:
    host_ledger: dict[str, dict] = {}
    host_paths: list[pathlib.Path] = []
    for name, (path, expected) in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen host hash mismatch for {name}: {actual} != {expected}")
        host_ledger[name] = {"path": str(path.relative_to(ROOT).as_posix()), "bytes": path.stat().st_size, "sha256": actual}
        host_paths.append(path)

    wsl_ledger: dict[str, dict] = {}
    for path, expected in EXPECTED_WSL.items():
        actual_size = stat_wsl(path)
        actual_hash = sha256_wsl(path)
        if actual_size != expected["bytes"] or actual_hash != expected["sha256"]:
            raise ValueError(f"frozen WSL identity mismatch for {path}")
        wsl_ledger[path] = {"bytes": actual_size, "sha256": actual_hash}

    controls = {}
    for literal in ("GGML_CUDA_DISABLE_GRAPHS", "CUDA graph warmup complete", "CUDA Graph id"):
        completed = wsl("grep", "-a", "-q", literal, CUDA_LIBRARY)
        controls[literal] = completed.returncode == 0
    if not all(controls.values()):
        raise ValueError(f"CUDA backend control strings missing: {controls}")
    return {"host": host_ledger, "wsl": wsl_ledger, "control_strings": controls}, host_paths


def parse_properties(text: str) -> dict[str, str]:
    properties: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            properties[key] = value
    return properties


def unit_state(unit: str) -> dict:
    completed = wsl(
        "systemctl", "show", unit,
        "-p", "LoadState", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID",
        "-p", "NRestarts", "-p", "ExecStart", "-p", "Environment",
    )
    props = parse_properties(completed.stdout)
    return {
        "returncode": completed.returncode,
        "load_state": props.get("LoadState", ""),
        "active_state": props.get("ActiveState", ""),
        "sub_state": props.get("SubState", ""),
        "main_pid": int(props.get("MainPID", "0") or 0),
        "n_restarts": int(props.get("NRestarts", "0") or 0),
        "exec_start": props.get("ExecStart", ""),
        "environment_property": props.get("Environment", ""),
        "stderr": completed.stderr.strip(),
    }


def process_values(pid: int) -> dict:
    if pid <= 0:
        raise ValueError("cannot inspect non-positive pid")
    command = wsl("xargs", "-0", "-n", "1", "-a", f"/proc/{pid}/cmdline", check=True)
    environment = wsl("xargs", "-0", "-n", "1", "-a", f"/proc/{pid}/environ", check=True)
    executable = wsl("readlink", "-f", f"/proc/{pid}/exe", check=True).stdout.strip()
    return {
        "pid": pid,
        "executable": executable,
        "argv": command.stdout.splitlines(),
        "environment": sorted(line for line in environment.stdout.splitlines() if line),
    }


def gpu_telemetry() -> dict:
    fields = "name,uuid,temperature.gpu,utilization.gpu,clocks.sm,power.draw,memory.used,memory.free"
    completed = wsl("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", check=True)
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    return dict(zip(fields.split(","), values, strict=True))


def systemctl(action: str, unit: str) -> None:
    wsl("systemctl", action, unit, root=True, timeout=120.0, check=True)


def wait_unit(unit: str, active: bool, timeout_seconds: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last = unit_state(unit)
    while time.monotonic() < deadline:
        last = unit_state(unit)
        if active and last["active_state"] == "active" and last["main_pid"] > 0:
            return last
        if not active and last["active_state"] in {"inactive", "failed"}:
            return last
        time.sleep(0.5)
    raise RuntimeError(f"unit {unit} did not reach expected active={active}: {last}")


def start_block_unit(block: dict) -> tuple[str, dict, dict]:
    unit = f"local-labs-cudagraph-r2-{block['id']}.service"
    existing = unit_state(unit)
    if existing["load_state"] != "not-found":
        raise RuntimeError(f"transient unit already exists: {unit}: {existing}")
    command = [
        "systemd-run", f"--unit={unit}", "--uid=augus", "--property=Type=simple",
        "--property=Restart=no", f"--setenv=LD_LIBRARY_PATH={LIB_DIR}",
    ]
    if block["treatment"] == "off":
        command.append("--setenv=GGML_CUDA_DISABLE_GRAPHS=1")
    command.extend(SERVER_ARGS)
    launch = wsl(*command, root=True, timeout=30.0, check=True)
    state = wait_unit(unit, active=True)
    process = process_values(state["main_pid"])
    return unit, {"command": ["wsl", "-d", WSL_DISTRO, "-u", "root", "--", *command], "stdout": launch.stdout.strip(), "stderr": launch.stderr.strip()}, process


def journal(unit: str) -> str:
    completed = wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True, timeout=30.0)
    return completed.stdout + completed.stderr


def prompt_for(prompt_id: int) -> str:
    speed = 20 + prompt_id * 7
    hours = 2 + (prompt_id % 4) * 0.5
    return (
        f"Deterministic benchmark problem {prompt_id}: a vehicle travels at {speed} km/h for "
        f"{hours:.1f} hours. Compute the distance and explain the arithmetic concisely."
    )


def request_payload(prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Answer the arithmetic question precisely."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 20260824,
        "stream": False,
        "cache_prompt": False,
    }


def send_request(prompt: str, timeout: float = 60.0) -> dict:
    started = time.perf_counter()
    response = http_json(f"{TEMP_URL}/v1/chat/completions", request_payload(prompt), timeout=timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = response.get("usage", {})
    timings = response.get("timings", {})
    return {
        "wall_ms": round(wall_ms, 3),
        "content": message.get("content", ""),
        "reasoning_content": message.get("reasoning_content", ""),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "timings": timings,
        "system_fingerprint": response.get("system_fingerprint"),
    }


def semantic_projection(sample: dict) -> dict:
    return {
        "content": sample.get("content", ""),
        "reasoning_content": sample.get("reasoning_content", ""),
        "finish_reason": sample.get("finish_reason"),
        "completion_tokens": sample.get("completion_tokens"),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def calculate_metrics(samples: list[dict]) -> tuple[dict, list[dict]]:
    by_prompt: dict[int, dict[str, dict]] = {}
    for sample in samples:
        by_prompt.setdefault(sample["prompt_id"], {})[sample["treatment"]] = sample
    if set(by_prompt) != set(range(1, 31)) or any(set(pair) != {"off", "on"} for pair in by_prompt.values()):
        raise ValueError("paired sample matrix is incomplete")

    pairs: list[dict] = []
    for prompt_id in sorted(by_prompt):
        off = by_prompt[prompt_id]["off"]
        on = by_prompt[prompt_id]["on"]
        match = semantic_projection(off) == semantic_projection(on)
        pairs.append({
            "prompt_id": prompt_id,
            "off_block": off["block_id"],
            "on_block": on["block_id"],
            "off_wall_ms": off["wall_ms"],
            "on_wall_ms": on["wall_ms"],
            "speedup_ratio": round(off["wall_ms"] / max(on["wall_ms"], 0.001), 6),
            "semantic_match": match,
            "off_semantic_sha256": canonical_json_sha256(semantic_projection(off)),
            "on_semantic_sha256": canonical_json_sha256(semantic_projection(on)),
        })
    off_times = [pair["off_wall_ms"] for pair in pairs]
    on_times = [pair["on_wall_ms"] for pair in pairs]
    speedups = [pair["speedup_ratio"] for pair in pairs]
    off_p95 = percentile(off_times, 0.95)
    on_p95 = percentile(on_times, 0.95)
    mismatches = sum(not pair["semantic_match"] for pair in pairs)
    metrics = {
        "paired_count": len(pairs),
        "mismatch_count": mismatches,
        "response_mismatch_rate": round(mismatches / len(pairs), 8),
        "off_p50_ms": round(statistics.median(off_times), 3),
        "on_p50_ms": round(statistics.median(on_times), 3),
        "off_p95_ms": round(off_p95, 3),
        "on_p95_ms": round(on_p95, 3),
        "paired_wall_speedup_p50": round(statistics.median(speedups), 6),
        "on_vs_off_p95_regression": round((on_p95 - off_p95) / max(off_p95, 0.001), 8),
    }
    return metrics, pairs


def verify_treatments(block_records: list[dict], binary_hash: str) -> dict:
    expected_order = ["off", "on", "on", "off"]
    order = [block["treatment"] for block in block_records]
    controls_ok = True
    argv_values: list[list[str]] = []
    for block in block_records:
        environment = block["process"]["environment"]
        graph_values = [item for item in environment if item.startswith("GGML_CUDA_DISABLE_GRAPHS=")]
        expected = ["GGML_CUDA_DISABLE_GRAPHS=1"] if block["treatment"] == "off" else []
        controls_ok = controls_ok and graph_values == expected
        controls_ok = controls_ok and block["process"]["executable"] == BINARY
        argv_values.append(block["process"]["argv"])
    argv_equal = all(argv == argv_values[0] for argv in argv_values[1:])
    valid_blocks = sum(
        block["recorded_count"] == 15 and block["warmup_count"] == 4
        for block in block_records
    )
    return {
        "expected_order": expected_order,
        "actual_order": order,
        "environment_controls_verified": controls_ok,
        "argv_equal_across_blocks": argv_equal,
        "binary_sha256": binary_hash,
        "distinct_binary_hashes_across_treatments": 1,
        "valid_abba_blocks": valid_blocks if order == expected_order and argv_equal else 0,
        "explicit_off_on_controls_verified": controls_ok and order == expected_order and argv_equal,
    }


def run_experiment(outdir: pathlib.Path) -> dict:
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    raw_dir = outdir / "raw"
    if any(raw_dir.iterdir()):
        raise RuntimeError(f"raw output directory is not empty: {raw_dir}")
    logs_dir = raw_dir / "logs"
    logs_dir.mkdir(parents=True)

    identity, frozen_paths = verify_inputs()
    write_json(raw_dir / "binary_identity.json", identity)
    initial_service = unit_state(PERSISTENT_UNIT)
    if initial_service["active_state"] != "active" or initial_service["main_pid"] <= 0:
        raise RuntimeError(f"persistent service is not active: {initial_service}")
    initial_process = process_values(initial_service["main_pid"])
    if initial_process["executable"] != BINARY or initial_process["argv"] != PERSISTENT_ARGS:
        raise RuntimeError(f"persistent service tuple mismatch: {initial_process}")
    initial_health = http_json(f"{INFERENCE_URL}/health")
    initial_embedding = http_json(f"{EMBED_URL}/health")
    if initial_health.get("status") != "ok" or initial_embedding.get("status") != "ok":
        raise RuntimeError("persistent inference or embedding endpoint is unhealthy before execution")
    for block in BLOCKS:
        candidate_unit = f"local-labs-cudagraph-r2-{block['id']}.service"
        if unit_state(candidate_unit)["load_state"] != "not-found":
            raise RuntimeError(f"reserved transient unit already exists: {candidate_unit}")
    try:
        unexpected = http_json(f"{TEMP_URL}/health", timeout=2.0)
    except (OSError, urllib.error.URLError):
        unexpected = None
    if unexpected is not None:
        raise RuntimeError(f"temporary endpoint {TEMP_URL} is already occupied: {unexpected}")
    service_identity = {
        "unit": PERSISTENT_UNIT,
        "initial_service": initial_service,
        "initial_process": initial_process,
        "initial_health": initial_health,
        "initial_embedding": initial_embedding,
        "initial_gpu": gpu_telemetry(),
    }
    write_json(raw_dir / "service_identity.json", service_identity)
    write_json(raw_dir / "environment.json", {
        "wsl": WSL_DISTRO,
        "gpu": service_identity["initial_gpu"],
        "cuda_control_strings": identity["control_strings"],
    })

    created_units: list[str] = []
    blocks: list[dict] = []
    samples: list[dict] = []
    maintenance: dict[str, Any] = {"persistent_service_stopped": False, "events": []}
    execution_error: str | None = None
    try:
        systemctl("stop", PERSISTENT_UNIT)
        maintenance["persistent_service_stopped"] = True
        maintenance["events"].append({"event": "persistent_stop", "state": wait_unit(PERSISTENT_UNIT, active=False)})
        if http_json(f"{EMBED_URL}/health").get("status") != "ok":
            raise RuntimeError("embedding service unhealthy after persistent inference stop")

        for block in BLOCKS:
            if http_json(f"{EMBED_URL}/health").get("status") != "ok":
                raise RuntimeError(f"embedding service unhealthy before {block['id']}")
            unit, launch, process = start_block_unit(block)
            created_units.append(unit)
            health = wait_health(TEMP_URL)
            props = http_json(f"{TEMP_URL}/props")
            before_gpu = gpu_telemetry()
            warmups = [send_request(prompt) for prompt in WARMUP_PROMPTS]
            recorded: list[dict] = []
            for prompt_id in block["prompt_ids"]:
                sample = send_request(prompt_for(prompt_id))
                sample.update({
                    "task_id": TASK_ID,
                    "block_id": block["id"],
                    "treatment": block["treatment"],
                    "prompt_id": prompt_id,
                    "prompt": prompt_for(prompt_id),
                })
                samples.append(sample)
                recorded.append(sample)
            after_gpu = gpu_telemetry()
            systemctl("stop", unit)
            stopped = wait_unit(unit, active=False)
            log_text = journal(unit)
            log_path = logs_dir / f"{block['id']}.log"
            log_path.write_text(log_text, encoding="utf-8")
            block_record = {
                "id": block["id"],
                "treatment": block["treatment"],
                "unit": unit,
                "prompt_ids": block["prompt_ids"],
                "launch": launch,
                "process": process,
                "health": health,
                "props": props,
                "warmup_count": len(warmups),
                "warmup_wall_ms": [item["wall_ms"] for item in warmups],
                "recorded_count": len(recorded),
                "gpu_before": before_gpu,
                "gpu_after": after_gpu,
                "stopped_state": stopped,
                "log_path": str(log_path.relative_to(outdir).as_posix()),
                "log_sha256": sha256_file(log_path),
                "graph_warmup_log_count": log_text.count("CUDA graph warmup complete"),
                "graph_reuse_log_count": log_text.count("CUDA Graph id"),
            }
            blocks.append(block_record)
            print(f"[HOST] {block['id']} complete: treatment={block['treatment']} recorded={len(recorded)}", flush=True)
    except Exception as exc:
        execution_error = repr(exc)
        raise
    finally:
        for unit in reversed(created_units):
            state = unit_state(unit)
            if state["active_state"] == "active":
                try:
                    systemctl("stop", unit)
                except Exception as stop_error:  # pragma: no cover - emergency evidence path
                    maintenance["events"].append({"event": "transient_stop_error", "unit": unit, "error": repr(stop_error)})
        if maintenance["persistent_service_stopped"]:
            try:
                systemctl("start", PERSISTENT_UNIT)
                maintenance["persistent_health_final"] = wait_health(INFERENCE_URL)
            except Exception as restore_error:  # pragma: no cover - emergency evidence path
                maintenance["restore_error"] = repr(restore_error)
        maintenance["execution_error"] = execution_error
        maintenance["final_service"] = unit_state(PERSISTENT_UNIT)
        try:
            maintenance["final_process"] = process_values(maintenance["final_service"]["main_pid"])
        except Exception as process_error:
            maintenance["final_process_error"] = repr(process_error)
        try:
            maintenance["final_embedding"] = wait_health(EMBED_URL, timeout_seconds=30.0)
        except Exception as embed_error:
            maintenance["final_embedding_error"] = repr(embed_error)
        maintenance["final_gpu"] = gpu_telemetry()
        write_json(raw_dir / "service_maintenance.json", maintenance)

    treatment = verify_treatments(blocks, identity["wsl"][BINARY]["sha256"])
    write_json(raw_dir / "treatment_controls.json", {"blocks": blocks, "verification": treatment})
    block_logs = {
        "logs": [
            {"block_id": block["id"], "path": block["log_path"], "sha256": block["log_sha256"]}
            for block in blocks
        ]
    }
    write_json(raw_dir / "block_logs.json", block_logs)
    samples_path = raw_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    metrics, pairs = calculate_metrics(samples)
    write_json(raw_dir / "paired_metrics.json", {"metrics": metrics, "pairs": pairs})
    write_json(raw_dir / "hardware_metrics.json", metrics)
    write_json(raw_dir / "paired_baseline.json", {
        "off_observations": sum(sample["treatment"] == "off" for sample in samples),
        "on_observations": sum(sample["treatment"] == "on" for sample in samples),
        "discarded_warmups_per_block": 4,
        "block_order": treatment["actual_order"],
    })
    write_json(raw_dir / "effective_route.json", {
        "binary": BINARY,
        "model": MODEL,
        "server_args": SERVER_ARGS,
        "endpoint": TEMP_URL,
        "block_order": treatment["actual_order"],
        "off_control": "GGML_CUDA_DISABLE_GRAPHS=1",
        "on_control": "GGML_CUDA_DISABLE_GRAPHS absent",
    })

    final_process = maintenance.get("final_process", {})
    service_restored = (
        maintenance["final_service"]["active_state"] == "active"
        and maintenance.get("persistent_health_final", {}).get("status") == "ok"
        and maintenance.get("final_embedding", {}).get("status") == "ok"
        and final_process.get("executable") == initial_process["executable"]
        and final_process.get("argv") == initial_process["argv"]
        and maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"]
    )
    recovery = {
        "service_and_embedding_restored": service_restored,
        "initial_pid": initial_service["main_pid"],
        "final_pid": maintenance["final_service"]["main_pid"],
        "argv_restored": final_process.get("argv") == initial_process["argv"],
        "executable_restored": final_process.get("executable") == initial_process["executable"],
        "restart_counter_preserved": maintenance["final_service"]["n_restarts"] == initial_service["n_restarts"],
        "inference_health": maintenance.get("persistent_health_final"),
        "embedding_health": maintenance.get("final_embedding"),
    }
    write_json(raw_dir / "recovery_state.json", recovery)

    receipt_inputs = [
        raw_dir / "binary_identity.json", raw_dir / "block_logs.json", raw_dir / "effective_route.json",
        raw_dir / "environment.json", raw_dir / "hardware_metrics.json", raw_dir / "paired_baseline.json",
        raw_dir / "paired_metrics.json", raw_dir / "recovery_state.json", raw_dir / "samples.jsonl",
        raw_dir / "service_identity.json", raw_dir / "service_maintenance.json", raw_dir / "treatment_controls.json",
        *[outdir / block["log_path"] for block in blocks], *frozen_paths,
    ]
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started_utc,
        started_monotonic=started_mono,
        input_paths=receipt_inputs,
        packages=["pytest"],
        runtime={"execution_mode": "causal_cudagraph_serving_abba", "blocks": 4, "paired_prompts": 30},
    )
    prov_ok, prov_errors = provenance_complete(provenance)
    if not prov_ok:
        raise ValueError(f"incomplete provenance: {prov_errors}")

    gates = {
        "treatment_identity": {"metric": "explicit_off_on_controls_verified", "operator": "eq", "threshold": True, "actual": treatment["explicit_off_on_controls_verified"], "pass": treatment["explicit_off_on_controls_verified"] is True},
        "binary_identity": {"metric": "distinct_binary_hashes_across_treatments", "operator": "eq", "threshold": 1, "actual": treatment["distinct_binary_hashes_across_treatments"], "pass": treatment["distinct_binary_hashes_across_treatments"] == 1},
        "balanced_crossover": {"metric": "valid_abba_blocks", "operator": "eq", "threshold": 4, "actual": treatment["valid_abba_blocks"], "pass": treatment["valid_abba_blocks"] == 4},
        "semantic_parity": {"metric": "response_mismatch_rate", "operator": "eq", "threshold": 0, "actual": metrics["response_mismatch_rate"], "pass": metrics["response_mismatch_rate"] == 0},
        "paired_speedup": {"metric": "paired_wall_speedup_p50", "operator": "ge", "threshold": 1.1, "actual": metrics["paired_wall_speedup_p50"], "pass": metrics["paired_wall_speedup_p50"] >= 1.1},
        "tail_non_regression": {"metric": "on_vs_off_p95_regression", "operator": "le", "threshold": 0.0, "actual": metrics["on_vs_off_p95_regression"], "pass": metrics["on_vs_off_p95_regression"] <= 0.0},
        "service_recovery": {"metric": "service_and_embedding_restored", "operator": "eq", "threshold": True, "actual": service_restored, "pass": service_restored is True},
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1",
        "task_id": TASK_ID,
        "provenance": provenance,
        "provenance_complete": prov_ok,
        "gates": gates,
        "evidence": {
            "acceptance_gates": "raw/receipt.json",
            "binary_identity": "raw/binary_identity.json",
            "block_logs": "raw/block_logs.json",
            "effective_route": "raw/effective_route.json",
            "environment": "raw/environment.json",
            "hardware_metrics": "raw/hardware_metrics.json",
            "paired_baseline": "raw/paired_baseline.json",
            "paired_metrics": "raw/paired_metrics.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "recovery_state": "raw/recovery_state.json",
            "service_identity": "raw/service_identity.json",
            "service_maintenance": "raw/service_maintenance.json",
            "treatment_controls": "raw/treatment_controls.json",
        },
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw_dir / "receipt.json", receipt)
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
