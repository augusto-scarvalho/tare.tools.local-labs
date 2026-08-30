#!/usr/bin/env python3
"""Repeat SLX08 with streamed TTFT, semantic floors and stable restoration identity."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import build_provenance, provenance_complete, sha256_file
from tools.research import run_slx08_physical_prefill_r4 as base

TASK_ID = "BACKLOG-SLX08-PHYSICAL-PREFILL-05"
PRE_REG_SHA256 = "c5eb12fc80c6b7c8bdc3ed62e28c7c3fcaede86435114f9695c72bd9fb02c1a3"
SOURCE_HASHES = {
    "config/research_backlog_admissions/BACKLOG-SLX08-PHYSICAL-PREFILL-05.json": "80b1b43bc7fc05ea8f48ead7a4e2c61e258def4e7ec7fbde93861cc5a84c2936",
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04/raw/run.terminal.json": "089cf385fea79885906485a84697be27e1bbf32b66652ac0147a0bf2c9fa9271",
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04/raw/samples.jsonl": "940a99adbd8b6a35fc009b991d819edc5bf9520c207d8361752108f9c820a861",
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04/raw/service_identity.json": "fe3344e921fb94717bea6178db570cdaf64476d7a5196a4e63427f43bd615fab",
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04/raw/actual_scores.json": "6edfae68ec0ae70e357034d108ec040e2d55ac0695709c764a9bf4cb45fa33f0",
    "tools/research/run_slx08_physical_prefill_r4.py": "42141bffd6c51635f1b0ec6e1ff3b531f4c7ee2b8eca933ee3371b673b86bf6d",
}


def stable_service_identity(state: dict) -> dict:
    systemd = state["systemd"]
    raw_exec = systemd.get("ExecStart", "")
    match = re.search(r"argv\[\]=(.*?) ; ignore_errors=", raw_exec)
    return {
        "active_state": systemd.get("ActiveState"),
        "exec_argv": match.group(1) if match else raw_exec,
        "gateway_role": (state.get("health") or {}).get("role"),
        "current_model": (state.get("health") or {}).get("current_model"),
        "health_status": state.get("health_status"),
    }


def stream_completion(payload: dict, timeout: float = 180.0) -> tuple[int, dict, str, float]:
    request = urllib.request.Request(
        f"{base.TEMP_BASE}/completion",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_content_ms = None
    chunks = []
    final = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = response.status
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            chunk = json.loads(data)
            content = chunk.get("content", "")
            if content:
                chunks.append(content)
                if first_content_ms is None:
                    first_content_ms = (time.perf_counter() - started) * 1000.0
            if chunk.get("stop") is True:
                final = chunk
    if first_content_ms is None or final is None:
        raise RuntimeError(f"stream lacked first content or final telemetry: first={first_content_ms}, final={final}")
    return status, final, "".join(chunks), first_content_ms


def evaluate_gates(metrics: dict) -> dict:
    gates = base.evaluate_gates(metrics)
    for gate, metric in (("dense_semantic_floor", "dense_accuracy"), ("treatment_semantic_floor", "selected_block_accuracy")):
        gates[gate] = {"metric": metric, "operator": "ge", "threshold": 0.9, "actual": metrics[metric], "pass": metrics[metric] >= 0.9}
    return gates


def execute(outdir: pathlib.Path) -> tuple[dict, dict]:
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for relative, expected in SOURCE_HASHES.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        inputs[relative] = actual
    prereg = outdir / "PRE_REGISTRATION.md"
    if sha256_file(prereg) != PRE_REG_SHA256:
        raise ValueError("preregistration mismatch")
    inputs[prereg.relative_to(ROOT).as_posix()] = PRE_REG_SHA256

    binary_sha = base.wsl_sha256(base.EXPERIMENT_BINARY)
    model_sha = base.wsl_sha256(base.MODEL)
    if binary_sha != "4395a601202ec76bcaef1d10db97849a92b311d8c31e4afce4d8b961609807a1":
        raise ValueError(f"experimental binary mismatch: {binary_sha}")
    if model_sha != base.MODEL_SHA256:
        raise ValueError(f"model mismatch: {model_sha}")
    slop_sources = [
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-context.cpp"),
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.cpp"),
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.h"),
    ]
    source_ledger = {str(path): sha256_file(path) for path in slop_sources}
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    original = base.service_state()
    original_stable = stable_service_identity(original)
    if original_stable["active_state"] != "active" or original_stable["health_status"] != 200:
        raise RuntimeError(f"original inference service is not healthy: {original_stable}")
    if base.health(8081)[0] != 200:
        raise RuntimeError("embedding service is not healthy before maintenance")

    pairs = []
    temporary = None
    log_handle = None
    restored_state = None
    restored_stable = None
    embedding_status = None
    with ExperimentRun(raw, TASK_ID, inputs, requires_restoration=True) as run:
        try:
            stopped = base.wsl_command(["systemctl", "stop", "llm-inference.service"], root=True)
            if stopped["returncode"]:
                raise RuntimeError(f"cannot stop inference service: {stopped}")
            base.wait_health(8080, None, 60.0)
            if base.health(8081)[0] != 200:
                raise RuntimeError("embedding service failed after inference stop")

            command = [
                "wsl", "-d", base.WSL_DISTRO, "--", "env", "SLOP_EXPERIMENTAL_SLX08=1", f"LD_LIBRARY_PATH={base.EXPERIMENT_LIB_DIR}",
                base.EXPERIMENT_BINARY, "-m", base.MODEL, "--alias", "slx08-qwen38", "--host", "127.0.0.1", "--port", str(base.TEMP_PORT),
                "--ctx-size", "8192", "--flash-attn", "on", "--gpu-layers", "all", "--parallel", "1", "--batch-size", "2048",
                "--ubatch-size", "512", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--no-mmproj", "--metrics",
            ]
            log_handle = (raw / "temporary_server.log").open("w", encoding="utf-8")
            temporary = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            base.wait_health(base.TEMP_PORT, 200, 300.0)
            run.checkpoint("experimental_server_ready", {"binary_sha256": binary_sha, "model_sha256": model_sha, "port": base.TEMP_PORT})

            intro = base.tokenize("You are evaluating a long-context retrieval fixture. Ignore all background filler. At the final Question, output only the single-digit numeric answer.", add_special=True)
            filler = base.tokenize(" Background record: irrelevant archival material; ignore it.", add_special=False)
            fixtures = []
            for case_id in range(base.PAIRS):
                left = case_id % 5
                right = (case_id * 3 + 1) % 5
                expected = left + right
                question = base.tokenize(f"\nCase {case_id}. Question: What is {left} plus {right}? Answer:", add_special=False)
                fixtures.append(base.build_fixture(case_id, intro, filler, question, expected))
            base.write_json(raw / "fixtures.json", [{key: value for key, value in fixture.items() if key != "tokens"} for fixture in fixtures])

            invalid_status, invalid_response = base.http_json(f"{base.TEMP_BASE}/completion", {"prompt": fixtures[0]["tokens"][:-1], "slx08_selected_block_prefill": True, "n_predict": 1})
            if invalid_status == 200:
                raise RuntimeError("invalid selected-block prompt did not fail closed")
            base.write_json(raw / "failure_reproduction.json", {"r4_false_restore": "volatile ExecStart PID/time compared", "r4_empty_semantics": "n_predict=1 emitted newline only", "invalid_prompt_status": invalid_status, "response": invalid_response})

            for enabled in (False, True):
                stream_completion({
                    "prompt": fixtures[0]["tokens"], "slx08_selected_block_prefill": enabled, "n_predict": 8, "stream": True,
                    "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False,
                })

            for fixture in fixtures:
                arms = [False, True] if fixture["case_id"] % 2 == 0 else [True, False]
                pair = {"case_id": fixture["case_id"], "expected": fixture["expected"], "prompt_sha256": fixture["prompt_sha256"]}
                for enabled in arms:
                    request = {
                        "prompt": fixture["tokens"], "slx08_selected_block_prefill": enabled, "n_predict": 8, "stream": True,
                        "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False,
                    }
                    status, final, content, ttft_ms = stream_completion(request)
                    telemetry = final.get("slx08_prefill")
                    if status != 200 or not isinstance(telemetry, dict):
                        raise RuntimeError(f"invalid final response: {status}: {final}")
                    expected_route = "selected_block" if enabled else "dense"
                    row = {
                        "task_id": TASK_ID, "case_id": fixture["case_id"], "arm": "on" if enabled else "off",
                        "prompt_sha256": fixture["prompt_sha256"], "request": request, "http_status": status,
                        "final_response": final, "content": content, "correct": base.answer_correct(content, fixture["expected"]),
                        "ttft_ms": ttft_ms, "telemetry": telemetry, "route_observed": telemetry.get("route") == expected_route,
                    }
                    expected_tokens = (4096, 2048) if enabled else (4096, 4096)
                    if (telemetry.get("original_prompt_tokens"), telemetry.get("retained_prompt_tokens")) != expected_tokens:
                        raise RuntimeError(f"route materiality mismatch: {telemetry}")
                    pair[row["arm"]] = row
                    run.record(row)
                if pair["off"]["prompt_sha256"] != pair["on"]["prompt_sha256"]:
                    raise RuntimeError("paired prompt bytes differ")
                pairs.append(pair)
                if len(pairs) % 8 == 0:
                    run.checkpoint("paired_progress", {"completed_pairs": len(pairs), "expected_pairs": base.PAIRS})
        finally:
            if temporary is not None:
                temporary.terminate()
                try:
                    temporary.wait(timeout=20.0)
                except subprocess.TimeoutExpired:
                    temporary.kill()
                    temporary.wait(timeout=10.0)
            if log_handle is not None:
                log_handle.close()
            if base.health(base.TEMP_PORT)[0] is not None:
                base.wsl_command(["pkill", "-TERM", "-f", f"{base.EXPERIMENT_BINARY}.*--port {base.TEMP_PORT}"], timeout=20.0)
                base.wait_health(base.TEMP_PORT, None, 30.0)
            started = base.wsl_command(["systemctl", "start", "llm-inference.service"], root=True, timeout=60.0)
            if started["returncode"]:
                raise RuntimeError(f"cannot restore inference service: {started}")
            restored_health = base.wait_health(8080, 200, 300.0)
            restored_state = base.service_state()
            restored_stable = stable_service_identity(restored_state)
            embedding_status = base.health(8081)[0]
            restored = original_stable == restored_stable and restored_health.get("current_model") == original_stable["current_model"] and embedding_status == 200
            run.restored({"original_stable": original_stable, "restored_stable": restored_stable, "volatile_original": original, "volatile_restored": restored_state, "embedding_health": embedding_status}, ok=restored)

        metrics = base.score(pairs, restored, embedding_status)
        gates = evaluate_gates(metrics)
        base.write_json(raw / "actual_scores.json", metrics)
        base.write_json(raw / "artifact_hashes.json", {"frozen_inputs": inputs, "slop_sources": source_ledger, "binary_sha256": binary_sha, "model_sha256": model_sha})
        base.write_json(raw / "dataset_hashes.json", {"fixtures_sha256": base.canonical_sha256([{key: value for key, value in fixture.items() if key != "tokens"} for fixture in fixtures]), "prompt_hashes": [fixture["prompt_sha256"] for fixture in fixtures]})
        base.write_json(raw / "effective_route.json", {"control": "dense prefill", "treatment": "server token-block compaction before dense prefill", "environment_gate": "SLOP_EXPERIMENTAL_SLX08=1", "block_size_tokens": base.BLOCK_SIZE})
        base.write_json(raw / "falsifiable_hypothesis.json", {"pairs": base.PAIRS, "prompt_tokens": base.PROMPT_TOKENS, "retained_fraction": 0.5, "n_predict": 8, "all_gates_required": True})
        base.write_json(raw / "hardware_metrics.json", {key: value for key, value in metrics.items() if "ttft" in key})
        base.write_json(raw / "independent_evaluation.json", {"scorer": "single-digit answer plus paired normal CI", "ttft_contract": "host monotonic to first non-empty streamed content", "metrics": metrics, "recomputable_from": "raw/samples.jsonl"})
        base.write_json(raw / "invalidation_rules.json", {"dense_labeled_on_invalid": True, "empty_semantic_parity_invalid": True, "unequal_prompt_hash_invalid": True, "missing_route_telemetry_invalid": True, "stable_restoration_failure_invalid": True})
        base.write_json(raw / "invariant_controls.json", {"decode": {"n_predict": 8, "stream": True, "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False}, "pair_order": "alternating", "prompt_tokens": base.PROMPT_TOKENS})
        base.write_json(raw / "paired_baseline.json", [{"case_id": pair["case_id"], "prompt_sha256": pair["prompt_sha256"], "off_ttft_ms": pair["off"]["ttft_ms"], "on_ttft_ms": pair["on"]["ttft_ms"], "off_correct": pair["off"]["correct"], "on_correct": pair["on"]["correct"]} for pair in pairs])
        base.write_json(raw / "physical_route_telemetry.json", [{"case_id": pair["case_id"], "off": pair["off"]["telemetry"], "on": pair["on"]["telemetry"]} for pair in pairs])
        base.write_json(raw / "real_implementation.json", {"source_root": r"C:\projects\slop.cpp", "server_token_block_compaction": True, "cuda_kernel_added": False, "production_default": False})
        base.write_json(raw / "recovery_state.json", {"original_stable": original_stable, "restored_stable": restored_stable, "volatile_original": original, "volatile_restored": restored_state})
        base.write_json(raw / "semantic_parity.json", {"dense_accuracy": metrics["dense_accuracy"], "selected_block_accuracy": metrics["selected_block_accuracy"], "paired_delta_ci95_low": metrics["paired_accuracy_delta_ci95_low"]})
        base.write_json(raw / "service_identity.json", {"original_stable": original_stable, "temporary": {"binary": base.EXPERIMENT_BINARY, "binary_sha256": binary_sha, "model": base.MODEL, "model_sha256": model_sha}, "restored_stable": restored_stable})
        base.write_json(raw / "service_maintenance.json", {"original_service_stopped": True, "temporary_port": base.TEMP_PORT, "original_service_restored": bool(metrics["original_service_restored"]), "embedding_health": embedding_status})
        base.write_json(raw / "source_execution_receipt.json", {"r4_terminal_sha256": SOURCE_HASHES["runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04/raw/run.terminal.json"], "r4_samples_sha256": SOURCE_HASHES["runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-04/raw/samples.jsonl"]})
        base.write_json(raw / "treatment_materiality.json", {"requests": base.PAIRS, "original_tokens": 4096, "retained_tokens": 2048, "original_blocks": 16, "retained_blocks": 8, "response_telemetry_bound": True})

        evidence_paths = sorted(path for path in raw.iterdir() if path.is_file() and path.name not in {"receipt.json", "run.events.jsonl", "run.terminal.json"})
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc, started_monotonic=started_mono,
            input_paths=[*[ROOT / relative for relative in SOURCE_HASHES], prereg, pathlib.Path(__file__).resolve(), *slop_sources, *evidence_paths],
            packages=["pytest"], runtime={"execution_mode": "physical_selected_block_prefill_streamed_r5", "binary": base.EXPERIMENT_BINARY, "binary_sha256": binary_sha, "model_sha256": model_sha, "gpu": "RTX 3090"},
        )
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(f"incomplete provenance: {errors}")
        evidence = {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
            "effective_route": "raw/effective_route.json", "failure_reproduction": "raw/failure_reproduction.json", "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json",
            "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json", "invalidation_rules": "raw/invalidation_rules.json",
            "invariant_controls": "raw/invariant_controls.json", "paired_baseline": "raw/paired_baseline.json", "physical_route_telemetry": "raw/physical_route_telemetry.json",
            "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json",
            "recovery_state": "raw/recovery_state.json", "semantic_parity": "raw/semantic_parity.json", "service_identity": "raw/service_identity.json",
            "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json", "treatment_materiality": "raw/treatment_materiality.json",
        }
        receipt = run.seal({"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence})
    return receipt, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_QUALIFIED_R5" if passed else "SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_REJECTED_R5"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. "
        f"Executed 64 dense and 64 selected-block streamed requests; dense accuracy {metrics['dense_accuracy']:.4f}; treatment accuracy {metrics['selected_block_accuracy']:.4f}; "
        f"p50 TTFT speedup {metrics['paired_p50_ttft_speedup']:.4f}x; p95 TTFT speedup {metrics['paired_p95_ttft_speedup']:.4f}x; failed gates: {', '.join(failed) if failed else 'none'}. "
        "The treatment is bounded server-side token-block compaction, not generic sparse attention or a production claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
