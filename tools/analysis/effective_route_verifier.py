#!/usr/bin/env python3
"""BEE-L1: Effective Route Verifier for local LLM runtimes.

Audits runtime execution across the 4 canonical lifecycle levels:
  1. REQUESTED: Command-line arguments and configuration flags.
  2. RESOLVED: Parsed descriptors and capability negotiation.
  3. REALIZED: Physical device buffer allocations and memory maps.
  4. EXERCISED: Empirical token generation and execution telemetry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
)


@dataclass
class RouteLevel:
    status: str
    details: dict[str, Any]
    hash: str


@dataclass
class RouteReceipt:
    timestamp: str
    agent: str
    target_endpoint: str
    verdict: str
    divergences: list[str]
    levels: dict[str, Any]
    receipt_fingerprint: str


def compute_hash(data: Any) -> str:
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:16]


def http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EffectiveRouteVerifier/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return None


def http_post_json(url: str, payload: dict, timeout: float = 10.0) -> dict | None:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "EffectiveRouteVerifier/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    return None


def _run_capture(argv: list[str], timeout: float = 60.0) -> dict:
    try:
        completed = subprocess.run(
            argv,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": repr(exc)}


def collect_systemd_runtime(unit: str, wsl_distro: str | None = None, hash_model: bool = False) -> dict:
    prefix = ["wsl", "-d", wsl_distro, "--"] if wsl_distro else []
    show = _run_capture(prefix + [
        "systemctl", "show", unit,
        "--property=FragmentPath,ExecStart,ActiveState,SubState,MainPID,NRestarts,Environment",
        "--no-pager",
    ])
    unit_text = _run_capture(prefix + ["systemctl", "cat", unit, "--no-pager"])
    fields = {}
    for line in show["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value

    exec_start = fields.get("ExecStart", "")
    model_match = re.search(r"(?:argv\[\]=[^\n]*?\s|\s)-m\s+([^\s;]+)", exec_start)
    model_path = model_match.group(1) if model_match else None
    main_pid = fields.get("MainPID", "0")
    cmdline = {"returncode": None, "stdout": "", "stderr": "invalid MainPID"}
    if main_pid.isdigit() and int(main_pid) > 0:
        cmdline = _run_capture(prefix + ["cat", f"/proc/{main_pid}/cmdline"])
        cmdline["stdout"] = cmdline["stdout"].replace("\x00", " ").strip()

    model_hash = None
    model_hash_error = None
    if hash_model and model_path:
        hashed = _run_capture(prefix + ["sha256sum", model_path], timeout=900.0)
        if hashed["returncode"] == 0 and hashed["stdout"]:
            model_hash = hashed["stdout"].split()[0]
        else:
            model_hash_error = hashed["stderr"] or "sha256sum failed"

    return {
        "unit": unit,
        "wsl_distro": wsl_distro,
        "show": fields,
        "effective_exec_start": exec_start,
        "effective_model_path": model_path,
        "main_pid": int(main_pid) if main_pid.isdigit() else None,
        "process_cmdline": cmdline["stdout"],
        "model_sha256": model_hash,
        "model_hash_error": model_hash_error,
        "unit_text_sha256": hashlib.sha256(unit_text["stdout"].encode("utf-8")).hexdigest(),
        "collection_errors": [
            result["stderr"]
            for result in (show, unit_text, cmdline)
            if result["returncode"] != 0 and result["stderr"]
        ],
    }


def audit_endpoint(
    base_url: str,
    expected_model_substring: str | None = None,
    runtime_evidence: dict | None = None,
    expected_model_sha256: str | None = None,
) -> RouteReceipt:
    divergences = []
    
    # 1. Probe Props & Health (RESOLVED & REALIZED)
    props = http_get_json(f"{base_url}/props")
    health = http_get_json(f"{base_url}/health")
    slots = http_get_json(f"{base_url}/slots")

    if not health or health.get("status") != "ok":
        divergences.append("HEALTH_UNAVAILABLE: endpoint did not return status ok")
        return RouteReceipt(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            agent="Codex",
            target_endpoint=base_url,
            verdict="FAIL_UNAVAILABLE",
            divergences=divergences,
            levels={},
            receipt_fingerprint="",
        )

    # 1. LEVEL: REQUESTED — effective service argv, not caller intent alone.
    req_details = {
        "endpoint": base_url,
        "expected_model": expected_model_substring,
        "expected_model_sha256": expected_model_sha256,
        "runtime": runtime_evidence,
    }
    if not runtime_evidence:
        divergences.append("RUNTIME_EVIDENCE_MISSING: effective launch argv was not collected")
    else:
        show = runtime_evidence.get("show", {})
        if show.get("ActiveState") != "active" or show.get("SubState") != "running":
            divergences.append(
                f"UNIT_NOT_RUNNING: ActiveState={show.get('ActiveState')} SubState={show.get('SubState')}"
            )
        if not runtime_evidence.get("effective_exec_start"):
            divergences.append("EFFECTIVE_ARGV_MISSING")
        if not runtime_evidence.get("process_cmdline"):
            divergences.append("PROCESS_CMDLINE_MISSING")
        if runtime_evidence.get("collection_errors"):
            divergences.append("RUNTIME_COLLECTION_ERROR")
        if expected_model_sha256:
            observed_hash = runtime_evidence.get("model_sha256")
            if observed_hash != expected_model_sha256:
                divergences.append(
                    f"MODEL_HASH_MISMATCH: expected='{expected_model_sha256}' got='{observed_hash}'"
                )
    level_requested = RouteLevel(
        status="CAPTURED" if runtime_evidence and not divergences else "DIVERGENT",
        details=req_details,
        hash=compute_hash(req_details),
    )

    # 2. LEVEL: RESOLVED
    resolved_details = {
        "model_path": props.get("model_path") if props else None,
        "system_fingerprint": health.get("system_fingerprint") if health else None,
    }
    if expected_model_substring and props:
        actual_path = props.get("model_path", "")
        if expected_model_substring.lower() not in actual_path.lower():
            divergences.append(
                f"MODEL_MISMATCH: expected substring '{expected_model_substring}', got '{actual_path}'"
            )
    if runtime_evidence and props:
        effective_model = runtime_evidence.get("effective_model_path")
        resolved_model = props.get("model_path")
        if effective_model != resolved_model:
            divergences.append(
                f"EFFECTIVE_MODEL_MISMATCH: systemd='{effective_model}' props='{resolved_model}'"
            )
        build_info = props.get("build_info", "")
        executable = runtime_evidence.get("process_cmdline", "").split(" ", 1)[0]
        build_token = build_info.split("-", 1)[0] if build_info else ""
        if build_token and build_token not in executable:
            divergences.append(
                f"BUILD_MISMATCH: props build_info='{build_info}' executable='{executable}'"
            )

    level_resolved = RouteLevel(
        status="RESOLVED" if not divergences else "DIVERGENT",
        details=resolved_details,
        hash=compute_hash(resolved_details),
    )

    # 3. LEVEL: REALIZED
    realized_details = {
        "slots_count": len(slots) if isinstance(slots, list) else 0,
        "slots": [
            {
                "id": slot.get("id"),
                "n_ctx": slot.get("n_ctx"),
                "speculative": slot.get("speculative"),
                "is_processing": slot.get("is_processing"),
            }
            for slot in slots
        ] if isinstance(slots, list) else [],
        "props_total_slots": props.get("total_slots") if props else None,
        "default_generation_settings": props.get("default_generation_settings") if props else None,
    }
    if not isinstance(slots, list) or not slots:
        divergences.append("SLOTS_UNAVAILABLE")
    elif props and props.get("total_slots") != len(slots):
        divergences.append(
            f"SLOT_COUNT_MISMATCH: props={props.get('total_slots')} endpoint={len(slots)}"
        )
    elif any(not isinstance(slot.get("n_ctx"), int) or slot.get("n_ctx", 0) <= 0 for slot in slots):
        divergences.append("INVALID_SLOT_ALLOCATION")
    level_realized = RouteLevel(
        status="OBSERVED" if not any(d.startswith(("SLOTS_", "SLOT_COUNT", "INVALID_SLOT")) for d in divergences) else "DIVERGENT",
        details=realized_details,
        hash=compute_hash(realized_details),
    )

    # 4. LEVEL: EXERCISED (Synthetic probe test)
    probe_payload = {
        "messages": [{"role": "user", "content": "Return exactly this text and nothing else: route-receipt-ok"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    start_time = time.perf_counter()
    chat_res = http_post_json(f"{base_url}/v1/chat/completions", probe_payload)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    response_content = ""
    if chat_res:
        response_content = chat_res.get("choices", [{}])[0].get("message", {}).get("content", "")
    exercised_details = {
        "probe_latency_ms": round(elapsed_ms, 2),
        "response_received": bool(chat_res),
        "response_content": response_content,
        "usage": chat_res.get("usage") if chat_res else None,
        "timings": chat_res.get("timings") if chat_res else None,
    }

    if not chat_res:
        divergences.append("EXERCISE_FAILED: chat completions endpoint did not return valid response")
        level_exercised = RouteLevel(status="FAILED", details=exercised_details, hash=compute_hash(exercised_details))
    elif response_content.strip() != "route-receipt-ok":
        divergences.append(f"EXERCISE_CONTENT_MISMATCH: got {response_content!r}")
        level_exercised = RouteLevel(status="DIVERGENT", details=exercised_details, hash=compute_hash(exercised_details))
    else:
        level_exercised = RouteLevel(status="EXERCISED", details=exercised_details, hash=compute_hash(exercised_details))

    levels = {
        "requested": asdict(level_requested),
        "resolved": asdict(level_resolved),
        "realized": asdict(level_realized),
        "exercised": asdict(level_exercised),
    }

    verdict = "VERIFIED" if not divergences else "DIVERGENT"
    receipt_data = {
        "endpoint": base_url,
        "verdict": verdict,
        "levels": levels,
    }
    fingerprint = compute_hash(receipt_data)

    return RouteReceipt(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        agent="Codex",
        target_endpoint=base_url,
        verdict=verdict,
        divergences=divergences,
        levels=levels,
        receipt_fingerprint=fingerprint,
    )


def main() -> int:
    started_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser(description="BEE-L1 Effective Route Verifier")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--expected-model", default=None)
    parser.add_argument("--systemd-unit", default="llm-inference.service")
    parser.add_argument("--wsl-distro", default="Ubuntu-24.04")
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--expected-model-sha256", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    runtime_evidence = collect_systemd_runtime(
        args.systemd_unit,
        wsl_distro=args.wsl_distro or None,
        hash_model=args.hash_model,
    )
    receipt = audit_endpoint(
        args.endpoint,
        args.expected_model,
        runtime_evidence,
        expected_model_sha256=args.expected_model_sha256,
    )
    receipt_dict = asdict(receipt)
    provenance = build_provenance(
        script_path=pathlib.Path(__file__),
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
        packages=[],
        runtime={
            "endpoint": args.endpoint,
            "systemd_unit": args.systemd_unit,
            "wsl_distro": args.wsl_distro,
        },
    )
    provenance_ok, provenance_errors = provenance_complete(provenance)
    receipt_dict["provenance"] = provenance
    receipt_dict["provenance_complete"] = provenance_ok
    receipt_dict["provenance_errors"] = provenance_errors
    if not provenance_ok:
        receipt_dict["divergences"].append("PROVENANCE_INCOMPLETE")
        receipt_dict["verdict"] = "UNVERIFIED"
    receipt_dict["receipt_fingerprint"] = canonical_json_sha256({
        key: value for key, value in receipt_dict.items() if key != "receipt_fingerprint"
    })

    print(json.dumps(receipt_dict, indent=2, ensure_ascii=False))

    if args.output:
        out_path = pathlib.Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(receipt_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n[RECEIPT SAVED]: {out_path}", flush=True)

    return 0 if receipt.verdict == "VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
