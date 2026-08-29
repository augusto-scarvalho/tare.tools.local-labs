#!/usr/bin/env python3
"""Seeded stability with atomic HTTP binding and frozen executable identity."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256, sha256_file
from tools.research import run_fleet_seeded_stability_r3 as r3


r2 = r3.r2
TASK_ID = "BACKLOG-FLEET-SEEDED-STABILITY-04"
MODELS = tuple(r3.MODELS)
BINARY_LEDGER_PATH = ROOT / "config/fleet_runtime_binary_identities_2026-08-29.json"
SOURCES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-SEEDED-STABILITY-04.json": "4cdb2bc941d5287968b235cdde8f041d71794b37fe0bd4831361479a67bb30c0",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "config/fleet_runtime_binary_identities_2026-08-29.json": "b1142241dc28556d407821b5d663da2e71e88e1d365d0e93a8a82f3918aaa7dd",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "tools/research/run_fleet_regression_screen.py": "7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3",
    "tools/research/run_fleet_seeded_stability.py": "71189428e1d1c7aff8a3ddb55d56c2f2034cd89c9df516f1a6490e7a2888676c",
    "tools/research/run_fleet_seeded_stability_r2.py": "1c6e0f1123acfef0a5756017b8f18e59cb7c626d72fdfa02b71093220e37f3c3",
    "tools/research/run_fleet_seeded_stability_r3.py": "c9e04f5557f07ba4cde146e50ad403ed081f577f74ce85e86cdcb63fbe4a79b1",
    "runs/research/BACKLOG-FLEET-SEEDED-STABILITY-03/raw/receipt.json": "fcb390b241a9d8b369e58c08d938e03782e71ed6c99efeaecffe1ec14b404e7c",
    "runs/research/BACKLOG-FLEET-SEEDED-STABILITY-03/REVIEW.json": "897d6e94113f66f916180c29172ac23e17aba6d0346753cd53e3b788f348e3cb",
}

_provenance_build = r3._original_build
_original_http = r3._original_http
_original_append = r3._original_append
_r3_evaluate_process_identity = r3.evaluate_process_identity
_pending_calls: list[dict[str, Any]] = []
_active_outdir: pathlib.Path | None = None


def _binary_ledger() -> dict[str, Any]:
    return json.loads(BINARY_LEDGER_PATH.read_text(encoding="utf-8"))["binaries"]


def evaluate_process_identity(
    model: str,
    status: dict[str, Any],
    process: dict[str, Any],
    binary: dict[str, Any],
    artifact: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    result = _r3_evaluate_process_identity(model, status, process, binary, artifact, card)
    expected = _binary_ledger().get(str(binary.get("resolved_path")))
    exact_checks = {
        "binary_identity_frozen": isinstance(expected, dict),
        "binary_hash_match": isinstance(expected, dict) and binary.get("sha256") == expected.get("sha256"),
        "binary_bytes_match": isinstance(expected, dict) and binary.get("bytes") == expected.get("bytes"),
    }
    result["expected"]["binary_identity"] = expected
    result["checks"].update(exact_checks)
    result["pass"] = all(result["checks"].values())
    return result


def is_experimental_request(payload: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(payload, dict)
        and payload.get("model") in MODELS
        and payload.get("temperature") == 0.2
        and payload.get("top_p") == 0.95
        and payload.get("seed") == 20260826
        and payload.get("stream") is False
        and payload.get("cache_prompt") is False
    )


def _capturing_http(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 900.0,
) -> tuple[int | None, dict[str, Any]]:
    status, response = _original_http(url, payload, timeout)
    if url.endswith("/v1/chat/completions") and is_experimental_request(payload):
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise RuntimeError("experimental response has no bindable response id")
        _pending_calls.append({
            "request": copy.deepcopy(payload),
            "response_id": response_id,
        })
    return status, response


def _capturing_append(path: pathlib.Path, value: dict[str, Any]) -> None:
    if path.name == "samples.jsonl":
        if not _pending_calls:
            raise RuntimeError("sample append has no captured experimental call")
        call = _pending_calls.pop(0)
        response_id = (value.get("response") or {}).get("id")
        if response_id != call["response_id"]:
            raise RuntimeError(
                f"captured response id mismatch: {call['response_id']!r} != {response_id!r}"
            )
        value = dict(value)
        value["request"] = call["request"]
        value["request_capture"] = "atomic_http_argument_response_id"
        value["request_sha256"] = canonical_json_sha256(call["request"])
        value["captured_response_id"] = call["response_id"]
    _original_append(path, value)


def validate_captured_requests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    panel = {(suite, case_id): case for suite, case_id, case in r2.base.cases()}
    for row in rows:
        expected = r2.base.fleet.payload_for(
            row["model"], row["suite"], panel[(row["suite"], row["case_id"])]
        )
        expected.update({"temperature": 0.2, "top_p": 0.95, "seed": 20260826})
        if row.get("request") != expected:
            raise RuntimeError("captured request differs from frozen experimental contract")
        if row.get("request_sha256") != canonical_json_sha256(row["request"]):
            raise RuntimeError("captured request hash mismatch")
        if row.get("captured_response_id") != (row.get("response") or {}).get("id"):
            raise RuntimeError("captured response id differs from stored response")
    return rows


def _finalizing_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    assert _active_outdir is not None
    raw = _active_outdir / "raw"
    if _pending_calls:
        raise RuntimeError(f"pending captured calls before receipt: {len(_pending_calls)}")
    rows = validate_captured_requests(r2.base.rj(raw / "samples.jsonl"))
    identities = [r3._route_identities[model] for model in MODELS]
    metrics_path = raw / "actual_scores.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update({
        "captured_request_hash_match_rate": sum(
            row["request_sha256"] == canonical_json_sha256(row["request"]) for row in rows
        ) / len(rows),
        "captured_response_id_match_rate": sum(
            row["captured_response_id"] == row["response"].get("id") for row in rows
        ) / len(rows),
        "pending_captured_calls": len(_pending_calls),
        "physical_binary_hash_match_rate": sum(
            row["checks"]["binary_hash_match"] and row["checks"]["binary_bytes_match"]
            for row in identities
        ) / len(identities),
    })
    r2._original_write(metrics_path, metrics)
    state_path = raw / "runner_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stable = metrics["exact_seeded_repeat_rate"] >= 0.9
    state.update({
        "status": "completed",
        "claim": (
            "QUALIFIED_TEXT_FLEET_SEEDED_ATOMIC_PHYSICAL_STABLE_R4"
            if stable else "QUALIFIED_TEXT_FLEET_SEEDED_ATOMIC_PHYSICAL_UNSTABLE_R4"
        ),
        "atomic_request_binding_before_receipt": True,
        "physical_identity_bound_before_receipt": True,
        "final_state_bound_before_receipt": True,
    })
    r2._original_write(state_path, state)
    return _provenance_build(*args, **kwargs)


def configure(outdir: pathlib.Path) -> None:
    global _active_outdir
    _active_outdir = outdir
    _pending_calls.clear()
    r3.configure(outdir)
    pipeline = json.loads((outdir / "PIPELINE.json").read_text(encoding="utf-8"))
    prereg = ROOT / pipeline["preregistration"]["path"]
    if sha256_file(prereg) != pipeline["preregistration"]["sha256"]:
        raise ValueError("pipeline preregistration binding mismatch")
    r3.evaluate_process_identity = evaluate_process_identity
    r3._original_build = _finalizing_build
    r2.enrich_requests = validate_captured_requests
    r2.base.TASK_ID = TASK_ID
    r2.base.PRE = pipeline["preregistration"]["sha256"]
    r2.base.SOURCES = SOURCES
    r2.base.aj = _capturing_append
    r2.base.fleet.http_json = _capturing_http
    r2.base.__file__ = __file__


def _eq_gate(metric: str, actual: Any, threshold: Any) -> dict[str, Any]:
    return {"metric": metric, "operator": "eq", "threshold": threshold, "actual": actual, "pass": actual == threshold}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    configure(outdir)
    receipt = r2.base.execute(outdir)
    raw = outdir / "raw"
    metrics = json.loads((raw / "actual_scores.json").read_text(encoding="utf-8"))
    receipt["gates"].update({
        "request_retention": _eq_gate("retained_request_payloads", metrics["retained_request_payloads"], 288),
        "request_hash_integrity": _eq_gate("captured_request_hash_match_rate", metrics["captured_request_hash_match_rate"], 1.0),
        "response_binding": _eq_gate("captured_response_id_match_rate", metrics["captured_response_id_match_rate"], 1.0),
        "capture_queue_drained": _eq_gate("pending_captured_calls", metrics["pending_captured_calls"], 0),
        "terminal_binding": _eq_gate("final_runner_state_bound", metrics["final_runner_state_bound"], True),
        "physical_route_coverage": _eq_gate("physical_routes_bound", metrics["physical_routes_bound"], 4),
        "physical_model_identity": _eq_gate("physical_model_hash_match_rate", metrics["physical_model_hash_match_rate"], 1.0),
        "physical_binary_identity": _eq_gate("physical_binary_hash_match_rate", metrics["physical_binary_hash_match_rate"], 1.0),
        "process_command_identity": _eq_gate("process_command_identity_rate", metrics["process_command_identity_rate"], 1.0),
    })
    receipt["evidence"].update({
        "request_payloads": "raw/samples.jsonl",
        "final_state_binding": "raw/runner_state.json",
        "physical_artifact_identity": "raw/physical_artifact_identity.json",
    })
    receipt.pop("receipt_fingerprint", None)
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r2._original_write(raw / "receipt.json", receipt)
    state = json.loads((raw / "runner_state.json").read_text(encoding="utf-8"))
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{state['claim']}` pending independent review.\n\n"
        f"Recorded `{metrics['recorded_requests']}` atomically bound requests with exact seeded "
        f"repeat rate `{metrics['exact_seeded_repeat_rate']:.6f}` and `{metrics['physical_routes_bound']}` "
        "routes bound to frozen executable and GGUF identities.\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def selfcheck() -> None:
    canary = {"model": "qwen38", "temperature": 0.0, "seed": 20260826, "stream": False, "cache_prompt": False}
    treatment = {"model": "qwen38", "temperature": 0.2, "top_p": 0.95, "seed": 20260826, "stream": False, "cache_prompt": False}
    assert is_experimental_request(canary) is False
    assert is_experimental_request(treatment) is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    outdir = args.outdir.resolve()
    receipt = execute(outdir)
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = r2.base.subprocess.run(
        [sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex continuous executor"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(advance.stdout, flush=True)
    return 0 if advance.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
