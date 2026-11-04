#!/usr/bin/env python3
"""SLX-01B: Stateful Serving Torture Matrix on RTX 3090.

Applies aggressive concurrency, streaming client disconnects, and mixed-depth
workloads to verify that multi-slot servers recover cleanly without zombie locks,
KV corruption, or memory leaks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import random
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (  # noqa: E402
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
)


def http_get_json(url: str, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ServingTorture/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def send_normal_request(base_url: str, prompt_id: int) -> dict:
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": f"Explain in 2 sentences why step {prompt_id} requires deterministic verification."}
        ],
        "max_tokens": 48,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "ServingTorture/1.0"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed = (time.perf_counter() - t0) * 1000.0
            return {
                "id": prompt_id,
                "status": "COMPLETED",
                "latency_ms": round(elapsed, 2),
                "tokens": body.get("usage", {}).get("completion_tokens", 0),
            }
    except Exception as e:
        return {"id": prompt_id, "status": f"ERROR: {e}", "latency_ms": 0, "tokens": 0}


def send_aborted_streaming_request(base_url: str, prompt_id: int) -> dict:
    """Connects via raw socket to stream SSE, reads first chunk, then closes socket abruptly."""
    payload = {
        "messages": [
            {"role": "user", "content": f"Generate a long essay of 500 words about system resilience iteration {prompt_id}."}
        ],
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": True,
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    
    parsed_host = "127.0.0.1"
    parsed_port = 8080

    t0 = time.perf_counter()
    try:
        s = socket.create_connection((parsed_host, parsed_port), timeout=5.0)
        req = (
            f"POST /v1/chat/completions HTTP/1.1\r\n"
            f"Host: {parsed_host}:{parsed_port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("ascii") + body_bytes
        s.sendall(req)

        # Read first buffer and immediately close socket to simulate hard client abort
        chunk = s.recv(512)
        s.close()
        elapsed = (time.perf_counter() - t0) * 1000.0
        return {
            "id": prompt_id,
            "status": "ABORTED_CLEANLY",
            "latency_ms": round(elapsed, 2),
            "bytes_received": len(chunk),
        }
    except Exception as e:
        return {"id": prompt_id, "status": f"ABORT_ERROR: {e}", "latency_ms": 0}


def check_slots_idle(base_url: str) -> tuple[bool, list[dict]]:
    slots = http_get_json(f"{base_url}/slots")
    if not isinstance(slots, list) or not slots:
        return False, []
    all_idle = all(s.get("is_processing") is False for s in slots)
    return all_idle, slots


def query_gpu_used_mib() -> float | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=15,
        )
        if completed.returncode != 0:
            return None
        return float(completed.stdout.strip().splitlines()[0])
    except Exception:
        return None


def query_systemd_state(unit: str, wsl_distro: str) -> dict:
    try:
        completed = subprocess.run(
            [
                "wsl", "-d", wsl_distro, "--", "systemctl", "show", unit,
                "--property=ActiveState,SubState,MainPID,NRestarts", "--no-pager",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=30,
        )
        fields = {}
        for line in completed.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value
        return {
            "returncode": completed.returncode,
            "fields": fields,
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": None, "fields": {}, "stderr": repr(exc)}


def run_canary(base_url: str, canary_id: int) -> bool:
    payload = {
        "messages": [{"role": "user", "content": "respond with exact text: adapt00-baseline-restored-ok"}],
        "max_tokens": 128,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "ServingTorture/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            msg = body.get("choices", [{}])[0].get("message", {})
            return msg.get("content", "").strip().lower() == "adapt00-baseline-restored-ok"
    except Exception:
        return False


def main() -> int:
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description="SLX-01B Stateful Serving Torture Matrix")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--expected-slots", type=int, default=4)
    parser.add_argument("--max-vram-drift-mib", type=float, default=20.0)
    parser.add_argument("--settle-seconds", type=float, default=10.0)
    parser.add_argument("--systemd-unit", default="llm-inference.service")
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    parser.add_argument("--output", default="runs/research/SLX-01C-SERVING-TORTURE-2026-08-25/raw/receipt.json")
    args = parser.parse_args()

    out_path = (ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== SLX-01B Stateful Serving Torture Matrix ===", flush=True)
    print(f"Target Endpoint: {args.endpoint}", flush=True)

    service_before = query_systemd_state(args.systemd_unit, args.wsl_distro)
    vram_before_mib = query_gpu_used_mib()

    # Initial Slots Check
    init_idle, init_slots = check_slots_idle(args.endpoint)
    print(f"Initial Slots State: {len(init_slots)} slots detected (All Idle: {init_idle})", flush=True)

    # Phase 1: 20 Concurrent Standard Requests (Workers = 4)
    print("\n--- Phase 1: 20 Concurrent Completions (Concurrency=4) ---", flush=True)
    p1_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(send_normal_request, args.endpoint, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            p1_results.append(res)
            print(f"  Req {res['id']}: {res['status']} ({res.get('latency_ms', 0)} ms)", flush=True)

    # Phase 2: 20 Aggressive Aborted Streaming Requests (Workers = 6)
    print("\n--- Phase 2: 20 Hard Aborted Streams (Concurrency=6) ---", flush=True)
    p2_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(send_aborted_streaming_request, args.endpoint, i) for i in range(20)]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            p2_results.append(res)
            print(f"  Abort Req {res['id']}: {res['status']}", flush=True)

    # Phase 3: Mixed High-Pressure Burst (10 Normal + 10 Abort simultaneous)
    print("\n--- Phase 3: Mixed High-Pressure Storm (10 Normal + 10 Abort) ---", flush=True)
    p3_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(send_normal_request, args.endpoint, 100 + i) if i % 2 == 0
            else executor.submit(send_aborted_streaming_request, args.endpoint, 100 + i)
            for i in range(20)
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            p3_results.append(res)

    print(f"Cooling down {args.settle_seconds:.1f} seconds...", flush=True)
    time.sleep(args.settle_seconds)

    # Post-Torture Slots Audit
    post_idle, post_slots = check_slots_idle(args.endpoint)
    print(f"\nPost-Torture Slots State: {len(post_slots)} slots detected (All Idle: {post_idle})", flush=True)

    # Post-Torture 5x Canary Verification
    canary_passes = 0
    for c_id in range(5):
        passed = run_canary(args.endpoint, c_id)
        if passed:
            canary_passes += 1
        print(f"  Canary {c_id}: {'PASS' if passed else 'FAIL'}", flush=True)

    service_after = query_systemd_state(args.systemd_unit, args.wsl_distro)
    vram_after_mib = query_gpu_used_mib()
    vram_drift_mib = (
        vram_after_mib - vram_before_mib
        if vram_before_mib is not None and vram_after_mib is not None
        else None
    )

    phase2_clean = sum(
        1 for result in p2_results
        if result.get("status") == "ABORTED_CLEANLY" and result.get("bytes_received", 0) > 0
    )
    phase3_normal = [result for result in p3_results if result.get("id", 0) % 2 == 0]
    phase3_abort = [result for result in p3_results if result.get("id", 0) % 2 == 1]
    phase3_normal_clean = sum(result.get("status") == "COMPLETED" for result in phase3_normal)
    phase3_abort_clean = sum(
        result.get("status") == "ABORTED_CLEANLY" and result.get("bytes_received", 0) > 0
        for result in phase3_abort
    )
    before_fields = service_before.get("fields", {})
    after_fields = service_after.get("fields", {})
    service_stable = (
        service_before.get("returncode") == 0
        and service_after.get("returncode") == 0
        and before_fields.get("ActiveState") == "active"
        and after_fields.get("ActiveState") == "active"
        and before_fields.get("SubState") == "running"
        and after_fields.get("SubState") == "running"
        and before_fields.get("MainPID") == after_fields.get("MainPID")
        and before_fields.get("NRestarts") == after_fields.get("NRestarts")
    )

    gates = {
        "phase1_success_ge_90pct": sum(1 for r in p1_results if r["status"] == "COMPLETED") >= 18,
        "phase2_aborts_20_of_20_clean": phase2_clean == 20,
        "phase3_normal_10_of_10_completed": phase3_normal_clean == 10,
        "phase3_aborts_10_of_10_clean": phase3_abort_clean == 10,
        "expected_slot_count_before": len(init_slots) == args.expected_slots,
        "expected_slot_count_after": len(post_slots) == args.expected_slots,
        "slots_recovered_100pct": post_idle,
        "canary_recovered_5_of_5": canary_passes == 5,
        "service_pid_and_restart_count_stable": service_stable,
        "vram_drift_within_limit": (
            vram_drift_mib is not None and vram_drift_mib <= args.max_vram_drift_mib
        ),
    }

    verdict = "PROMOTED" if all(gates.values()) else "REJECTED"

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "agent": "Codex",
        "target_endpoint": args.endpoint,
        "initial_slots_count": len(init_slots),
        "initial_slots": init_slots,
        "service_before": service_before,
        "vram_before_mib": vram_before_mib,
        "phase1_summary": {
            "total": len(p1_results),
            "completed": sum(1 for r in p1_results if r["status"] == "COMPLETED"),
            "avg_latency_ms": round(statistics.mean(r["latency_ms"] for r in p1_results if r["status"] == "COMPLETED"), 2),
        },
        "phase1_results": sorted(p1_results, key=lambda item: item["id"]),
        "phase2_summary": {"total": len(p2_results), "clean_aborts": phase2_clean},
        "phase2_results": sorted(p2_results, key=lambda item: item["id"]),
        "phase3_summary": {
            "total": len(p3_results),
            "normal_completed": phase3_normal_clean,
            "aborts_clean": phase3_abort_clean,
        },
        "phase3_results": sorted(p3_results, key=lambda item: item["id"]),
        "post_torture_slots": post_slots,
        "canary_verification": {
            "attempts": 5,
            "passed": canary_passes,
        },
        "service_after": service_after,
        "vram_after_mib": vram_after_mib,
        "vram_drift_mib": vram_drift_mib,
        "gates": gates,
        "verdict": verdict,
    }

    provenance = build_provenance(
        script_path=pathlib.Path(__file__),
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
        runtime={
            "endpoint": args.endpoint,
            "expected_slots": args.expected_slots,
            "max_vram_drift_mib": args.max_vram_drift_mib,
            "settle_seconds": args.settle_seconds,
            "systemd_unit": args.systemd_unit,
            "wsl_distro": args.wsl_distro,
        },
    )
    provenance_ok, provenance_errors = provenance_complete(provenance)
    final_payload["provenance"] = provenance
    final_payload["provenance_complete"] = provenance_ok
    final_payload["provenance_errors"] = provenance_errors
    if not provenance_ok:
        final_payload["verdict"] = "UNVERIFIED"
    final_payload["receipt_fingerprint"] = canonical_json_sha256(final_payload)

    out_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n==================================================", flush=True)
    print(f"  SLX-01B TORTURE MATRIX VERDICT: {verdict}", flush=True)
    print(f"  Phase 1 Completed: {final_payload['phase1_summary']['completed']}/20")
    print(f"  Slots Fully Recovered: {post_idle}")
    print(f"  Canary Pass: {canary_passes}/5")
    print(f"  Receipt written to: {out_path}", flush=True)
    print(f"==================================================", flush=True)
    return 0 if final_payload["verdict"] == "PROMOTED" else 1


if __name__ == "__main__":
    sys.exit(main())
