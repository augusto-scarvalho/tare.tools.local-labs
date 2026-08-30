#!/usr/bin/env python3
"""Run the physical OFF/ON selected-block prefill qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import build_provenance, provenance_complete, sha256_file

TASK_ID = "BACKLOG-SLX08-PHYSICAL-PREFILL-04"
PRE_REG_SHA256 = "a8fdb75d58c8d02f0a9865bfb2c7e9538747b725c2192321960c6e626bff94db"
WSL_DISTRO = "Ubuntu-24.04"
TEMP_PORT = 18082
TEMP_BASE = f"http://127.0.0.1:{TEMP_PORT}"
EXPERIMENT_BINARY = "/home/augus/build/slop-slx08/bin/llama-server"
EXPERIMENT_LIB_DIR = "/home/augus/build/slop-slx08/bin"
MODEL = "/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MODEL_SHA256 = "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"
BLOCK_SIZE = 256
PROMPT_TOKENS = 4096
PAIRS = 64
SOURCE_HASHES = {
    "runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/raw/receipt.json": "d2f76165a548e08c1713d44f976f3a1d5aa30158e6745b90bce0d72aa688bddb",
    "runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/REVIEW.json": "e07df1828f2a8f1f86f32138dcae53735273587683c29701a9ccddf46503a778",
    "runs/research/BACKLOG-SLX08-REAL-FIDELITY-02/raw/context_vectors.safetensors": "859ea9e3088de4e1f354a51a3c5502fd845ac1289dd0ad7b83d8c4f35b76cc58",
    "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
    "tools/probes/slx08_speculative_prefill_oracle.py": "5b85dd266c3fc72ae47a7cabe6e5ae3246e4aab544e87e6ee7cd47eab81bdc37",
}


def write_json(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def canonical_sha256(value) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def run_command(command: list[str], timeout: float = 120.0) -> dict:
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def wsl_command(arguments: list[str], *, root: bool = False, timeout: float = 120.0) -> dict:
    command = ["wsl", "-d", WSL_DISTRO]
    if root:
        command.extend(["-u", "root"])
    command.extend(["--", *arguments])
    return run_command(command, timeout)


def http_json(url: str, payload: dict | None = None, timeout: float = 120.0) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return error.code, parsed


def health(port: int) -> tuple[int | None, dict | None]:
    try:
        return http_json(f"http://127.0.0.1:{port}/health", timeout=3.0)
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None, None


def wait_health(port: int, expected: int | None, timeout: float) -> dict | None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, body = health(port)
        last = {"status": status, "body": body}
        if status == expected:
            return body
        time.sleep(0.5)
    raise TimeoutError(f"port {port} health did not become {expected}: {last}")


def service_state() -> dict:
    result = wsl_command([
        "systemctl", "show", "llm-inference.service", "-p", "ActiveState", "-p", "MainPID", "-p", "NRestarts", "-p", "ExecStart", "--no-pager",
    ])
    if result["returncode"]:
        raise RuntimeError(f"cannot inspect inference service: {result}")
    values = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    status, body = health(8080)
    return {"systemd": values, "health_status": status, "health": body}


def wsl_sha256(path: str) -> str:
    result = wsl_command(["sha256sum", path], timeout=600.0)
    if result["returncode"]:
        raise RuntimeError(f"cannot hash {path}: {result}")
    return result["stdout"].split()[0]


def tokenize(content: str, *, add_special: bool) -> list[int]:
    status, response = http_json(f"{TEMP_BASE}/tokenize", {"content": content, "add_special": add_special, "parse_special": True})
    if status != 200 or not isinstance(response.get("tokens"), list):
        raise RuntimeError(f"tokenization failed: {status}: {response}")
    return response["tokens"]


def pad_tokens(prefix: list[int], filler: list[int], size: int, suffix: list[int] | None = None) -> list[int]:
    suffix = suffix or []
    remaining = size - len(prefix) - len(suffix)
    if remaining < 0 or not filler:
        raise ValueError("fixture components do not fit in the requested block")
    middle = [filler[index % len(filler)] for index in range(remaining)]
    return [*prefix, *middle, *suffix]


def build_fixture(case_id: int, intro: list[int], filler: list[int], question_tokens: list[int], expected: int) -> dict:
    first = pad_tokens(intro, filler, BLOCK_SIZE)
    last = pad_tokens([], filler, BLOCK_SIZE, question_tokens)
    tokens = [*first]
    for _ in range(PROMPT_TOKENS // BLOCK_SIZE - 2):
        tokens.extend(pad_tokens([], filler, BLOCK_SIZE))
    tokens.extend(last)
    if len(tokens) != PROMPT_TOKENS:
        raise AssertionError(f"fixture {case_id} has {len(tokens)} tokens")
    return {"case_id": case_id, "tokens": tokens, "expected": expected, "prompt_sha256": canonical_sha256(tokens)}


def answer_correct(content: str, expected: int) -> bool:
    match = re.search(r"\b([0-9])\b", content)
    return match is not None and int(match.group(1)) == expected


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_accuracy_ci_low(rows: list[dict]) -> float:
    differences = [int(pair["on"]["correct"]) - int(pair["off"]["correct"]) for pair in rows]
    mean = statistics.fmean(differences)
    if len(set(differences)) == 1:
        return mean
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return mean - 1.96 * standard_error


def score(pairs: list[dict], restored: bool, embedding_status: int | None) -> dict:
    off = [pair["off"] for pair in pairs]
    on = [pair["on"] for pair in pairs]
    fractions = [row["telemetry"]["retained_attention_fraction"] for row in on]
    ratios = [pair["off"]["ttft_ms"] / pair["on"]["ttft_ms"] for pair in pairs]
    return {
        "physical_selected_block_prefill_requests": len(on),
        "physical_dense_prefill_requests": len(off),
        "selected_block_route_observation_rate": statistics.fmean(row["route_observed"] for row in on),
        "median_retained_attention_fraction": statistics.median(fractions),
        "dense_accuracy": statistics.fmean(row["correct"] for row in off),
        "selected_block_accuracy": statistics.fmean(row["correct"] for row in on),
        "paired_accuracy_delta": statistics.fmean(int(pair["on"]["correct"]) - int(pair["off"]["correct"]) for pair in pairs),
        "paired_accuracy_delta_ci95_low": paired_accuracy_ci_low(pairs),
        "paired_p50_ttft_speedup": statistics.median(ratios),
        "paired_p95_ttft_speedup": percentile([row["ttft_ms"] for row in off], 0.95) / percentile([row["ttft_ms"] for row in on], 0.95),
        "dense_p50_ttft_ms": statistics.median(row["ttft_ms"] for row in off),
        "selected_block_p50_ttft_ms": statistics.median(row["ttft_ms"] for row in on),
        "dense_p95_ttft_ms": percentile([row["ttft_ms"] for row in off], 0.95),
        "selected_block_p95_ttft_ms": percentile([row["ttft_ms"] for row in on], 0.95),
        "original_service_restored": int(restored),
        "embedding_health": embedding_status,
    }


def evaluate_gates(metrics: dict) -> dict:
    definitions = {
        "physical_treatment": ("physical_selected_block_prefill_requests", "ge", 64),
        "dense_control": ("physical_dense_prefill_requests", "ge", 64),
        "route_observation": ("selected_block_route_observation_rate", "eq", 1.0),
        "retained_fraction": ("median_retained_attention_fraction", "eq", 0.5),
        "semantic_noninferiority": ("paired_accuracy_delta_ci95_low", "ge", -0.03),
        "ttft_gain": ("paired_p50_ttft_speedup", "ge", 1.10),
        "tail_safety": ("paired_p95_ttft_speedup", "ge", 1.0),
        "service_restore": ("original_service_restored", "eq", 1),
        "embedding_integrity": ("embedding_health", "eq", 200),
    }
    operators = {"eq": lambda actual, threshold: actual == threshold, "ge": lambda actual, threshold: actual >= threshold}
    return {
        gate: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": operators[operator](metrics[metric], threshold)}
        for gate, (metric, operator, threshold) in definitions.items()
    }


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

    binary_sha = wsl_sha256(EXPERIMENT_BINARY)
    model_sha = wsl_sha256(MODEL)
    if model_sha != MODEL_SHA256:
        raise ValueError(f"model mismatch: {model_sha} != {MODEL_SHA256}")
    slop_sources = [
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-context.cpp"),
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.cpp"),
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.h"),
    ]
    source_ledger = {str(path): sha256_file(path) for path in slop_sources}
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    original = service_state()
    if original["systemd"].get("ActiveState") != "active" or original["health_status"] != 200:
        raise RuntimeError(f"original inference service is not healthy: {original}")
    if health(8081)[0] != 200:
        raise RuntimeError("embedding service is not healthy before maintenance")

    pairs = []
    temporary = None
    log_handle = None
    restored_state = None
    with ExperimentRun(raw, TASK_ID, inputs, requires_restoration=True) as run:
        try:
            stopped = wsl_command(["systemctl", "stop", "llm-inference.service"], root=True)
            if stopped["returncode"]:
                raise RuntimeError(f"cannot stop inference service: {stopped}")
            wait_health(8080, None, 60.0)
            if health(8081)[0] != 200:
                raise RuntimeError("embedding service failed after inference stop")

            command = [
                "wsl", "-d", WSL_DISTRO, "--", "env", "SLOP_EXPERIMENTAL_SLX08=1", f"LD_LIBRARY_PATH={EXPERIMENT_LIB_DIR}",
                EXPERIMENT_BINARY, "-m", MODEL, "--alias", "slx08-qwen38", "--host", "127.0.0.1", "--port", str(TEMP_PORT),
                "--ctx-size", "8192", "--flash-attn", "on", "--gpu-layers", "all", "--parallel", "1", "--batch-size", "2048",
                "--ubatch-size", "512", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--no-mmproj", "--metrics",
            ]
            log_handle = (raw / "temporary_server.log").open("w", encoding="utf-8")
            temporary = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            wait_health(TEMP_PORT, 200, 300.0)
            run.checkpoint("experimental_server_ready", {"binary_sha256": binary_sha, "model_sha256": model_sha, "port": TEMP_PORT})

            intro = tokenize("You are evaluating a long-context retrieval fixture. Ignore all background filler. At the final Question, output only the single-digit numeric answer.", add_special=True)
            filler = tokenize(" Background record: irrelevant archival material; ignore it.", add_special=False)
            fixtures = []
            for case_id in range(PAIRS):
                left = case_id % 5
                right = (case_id * 3 + 1) % 5
                expected = left + right
                question = tokenize(f"\nCase {case_id}. Question: What is {left} plus {right}? Answer:", add_special=False)
                fixtures.append(build_fixture(case_id, intro, filler, question, expected))
            write_json(raw / "fixtures.json", [{key: value for key, value in fixture.items() if key != "tokens"} for fixture in fixtures])

            invalid_status, invalid_response = http_json(f"{TEMP_BASE}/completion", {
                "prompt": fixtures[0]["tokens"][:-1], "slx08_selected_block_prefill": True, "n_predict": 1,
            })
            if invalid_status == 200:
                raise RuntimeError("invalid selected-block prompt did not fail closed")
            write_json(raw / "failure_reproduction.json", {"invalid_prompt_status": invalid_status, "response": invalid_response})

            for enabled in (False, True, False, True):
                http_json(f"{TEMP_BASE}/completion", {
                    "prompt": fixtures[enabled]["tokens"], "slx08_selected_block_prefill": enabled, "n_predict": 1,
                    "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False,
                }, timeout=180.0)

            for fixture in fixtures:
                arms = [False, True] if fixture["case_id"] % 2 == 0 else [True, False]
                pair = {"case_id": fixture["case_id"], "expected": fixture["expected"], "prompt_sha256": fixture["prompt_sha256"]}
                for enabled in arms:
                    request = {
                        "prompt": fixture["tokens"], "slx08_selected_block_prefill": enabled, "n_predict": 1,
                        "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False,
                    }
                    start = time.perf_counter()
                    status, response = http_json(f"{TEMP_BASE}/completion", request, timeout=180.0)
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                    if status != 200:
                        raise RuntimeError(f"request failed: {status}: {response}")
                    telemetry = response.get("slx08_prefill")
                    if not isinstance(telemetry, dict):
                        raise RuntimeError(f"missing route telemetry: {response}")
                    expected_route = "selected_block" if enabled else "dense"
                    row = {
                        "task_id": TASK_ID,
                        "case_id": fixture["case_id"],
                        "arm": "on" if enabled else "off",
                        "prompt_sha256": fixture["prompt_sha256"],
                        "request": request,
                        "http_status": status,
                        "response": response,
                        "content": response.get("content", ""),
                        "correct": answer_correct(response.get("content", ""), fixture["expected"]),
                        "ttft_ms": ttft_ms,
                        "telemetry": telemetry,
                        "route_observed": telemetry.get("route") == expected_route,
                    }
                    if enabled and (telemetry.get("original_prompt_tokens"), telemetry.get("retained_prompt_tokens")) != (4096, 2048):
                        raise RuntimeError(f"selected-block token materiality mismatch: {telemetry}")
                    if not enabled and (telemetry.get("original_prompt_tokens"), telemetry.get("retained_prompt_tokens")) != (4096, 4096):
                        raise RuntimeError(f"dense token materiality mismatch: {telemetry}")
                    pair[row["arm"]] = row
                    run.record(row)
                if pair["off"]["prompt_sha256"] != pair["on"]["prompt_sha256"]:
                    raise RuntimeError("paired prompt bytes differ")
                pairs.append(pair)
                if len(pairs) % 8 == 0:
                    run.checkpoint("paired_progress", {"completed_pairs": len(pairs), "expected_pairs": PAIRS})
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
            if health(TEMP_PORT)[0] is not None:
                wsl_command(["pkill", "-TERM", "-f", f"{EXPERIMENT_BINARY}.*--port {TEMP_PORT}"], timeout=20.0)
                wait_health(TEMP_PORT, None, 30.0)
            started = wsl_command(["systemctl", "start", "llm-inference.service"], root=True, timeout=60.0)
            if started["returncode"]:
                raise RuntimeError(f"cannot restore inference service: {started}")
            restored_health = wait_health(8080, 200, 300.0)
            restored_state = service_state()
            embedding_status = health(8081)[0]
            restored = (
                restored_state["systemd"].get("ActiveState") == "active"
                and restored_state["systemd"].get("ExecStart") == original["systemd"].get("ExecStart")
                and restored_health.get("current_model") == original["health"].get("current_model")
                and embedding_status == 200
            )
            run.restored({"original": original, "restored": restored_state, "embedding_health": embedding_status}, ok=restored)

        metrics = score(pairs, restored, embedding_status)
        gates = evaluate_gates(metrics)
        write_json(raw / "actual_scores.json", metrics)
        write_json(raw / "artifact_hashes.json", {"frozen_inputs": inputs, "slop_sources": source_ledger, "binary_sha256": binary_sha, "model_sha256": model_sha})
        write_json(raw / "dataset_hashes.json", {"fixtures_sha256": canonical_sha256([{key: value for key, value in fixture.items() if key != "tokens"} for fixture in fixtures]), "prompt_hashes": [fixture["prompt_sha256"] for fixture in fixtures]})
        write_json(raw / "effective_route.json", {"control": "dense prefill", "treatment": "server token-block compaction before dense prefill", "environment_gate": "SLOP_EXPERIMENTAL_SLX08=1", "block_size_tokens": BLOCK_SIZE})
        write_json(raw / "falsifiable_hypothesis.json", {"pairs": PAIRS, "prompt_tokens": PROMPT_TOKENS, "retained_fraction": 0.5, "all_gates_required": True})
        write_json(raw / "hardware_metrics.json", {key: value for key, value in metrics.items() if "ttft" in key})
        write_json(raw / "independent_evaluation.json", {"scorer": "answer_correct plus paired normal CI", "metrics": metrics, "recomputable_from": "raw/samples.jsonl"})
        write_json(raw / "invalidation_rules.json", {"dense_labeled_on_invalid": True, "unequal_prompt_hash_invalid": True, "missing_route_telemetry_invalid": True, "restoration_failure_invalid": True})
        write_json(raw / "invariant_controls.json", {"decode": {"n_predict": 1, "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False}, "pair_order": "alternating", "prompt_tokens": PROMPT_TOKENS})
        write_json(raw / "paired_baseline.json", [{"case_id": pair["case_id"], "prompt_sha256": pair["prompt_sha256"], "off_ttft_ms": pair["off"]["ttft_ms"], "on_ttft_ms": pair["on"]["ttft_ms"], "off_correct": pair["off"]["correct"], "on_correct": pair["on"]["correct"]} for pair in pairs])
        write_json(raw / "physical_route_telemetry.json", [{"case_id": pair["case_id"], "off": pair["off"]["telemetry"], "on": pair["on"]["telemetry"]} for pair in pairs])
        write_json(raw / "real_implementation.json", {"source_root": r"C:\projects\slop.cpp", "server_token_block_compaction": True, "cuda_kernel_added": False, "production_default": False})
        write_json(raw / "recovery_state.json", {"original": original, "restored": restored_state})
        write_json(raw / "semantic_parity.json", {"dense_accuracy": metrics["dense_accuracy"], "selected_block_accuracy": metrics["selected_block_accuracy"], "paired_delta_ci95_low": metrics["paired_accuracy_delta_ci95_low"]})
        write_json(raw / "service_identity.json", {"original": original, "temporary": {"binary": EXPERIMENT_BINARY, "binary_sha256": binary_sha, "model": MODEL, "model_sha256": model_sha}, "restored": restored_state})
        write_json(raw / "service_maintenance.json", {"original_service_stopped": True, "temporary_port": TEMP_PORT, "original_service_restored": bool(metrics["original_service_restored"]), "embedding_health": embedding_status})
        write_json(raw / "source_execution_receipt.json", {"predecessor_receipt_sha256": SOURCE_HASHES["runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/raw/receipt.json"], "predecessor_review_sha256": SOURCE_HASHES["runs/research/BACKLOG-SLX08-REAL-FIDELITY-03/REVIEW.json"]})
        write_json(raw / "treatment_materiality.json", {"requests": PAIRS, "original_tokens": 4096, "retained_tokens": 2048, "original_blocks": 16, "retained_blocks": 8, "response_telemetry_bound": True})

        evidence_paths = sorted(path for path in raw.iterdir() if path.is_file() and path.name not in {"receipt.json", "run.events.jsonl", "run.terminal.json"})
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(),
            started_at_utc=started_utc,
            started_monotonic=started_mono,
            input_paths=[*[ROOT / relative for relative in SOURCE_HASHES], prereg, pathlib.Path(__file__).resolve(), *slop_sources, *evidence_paths],
            packages=["pytest"],
            runtime={"execution_mode": "physical_selected_block_prefill", "binary": EXPERIMENT_BINARY, "binary_sha256": binary_sha, "model_sha256": model_sha, "gpu": "RTX 3090"},
        )
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(f"incomplete provenance: {errors}")
        evidence = {
            "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json",
            "dataset_hashes": "raw/dataset_hashes.json", "effective_route": "raw/effective_route.json", "failure_reproduction": "raw/failure_reproduction.json",
            "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json", "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json",
            "invalidation_rules": "raw/invalidation_rules.json", "invariant_controls": "raw/invariant_controls.json", "paired_baseline": "raw/paired_baseline.json",
            "physical_route_telemetry": "raw/physical_route_telemetry.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
            "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
            "semantic_parity": "raw/semantic_parity.json", "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json",
            "source_execution_receipt": "raw/source_execution_receipt.json", "treatment_materiality": "raw/treatment_materiality.json",
        }
        receipt = run.seal({"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence})
    return receipt, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_QUALIFIED_R4" if passed else "SLX08_PHYSICAL_SELECTED_BLOCK_PREFILL_REJECTED_R4"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. "
        f"Executed {metrics['physical_dense_prefill_requests']} dense and {metrics['physical_selected_block_prefill_requests']} selected-block requests; "
        f"p50 speedup {metrics['paired_p50_ttft_speedup']:.4f}x; p95 speedup {metrics['paired_p95_ttft_speedup']:.4f}x; "
        f"paired accuracy CI95 low {metrics['paired_accuracy_delta_ci95_low']:.4f}; failed gates: {', '.join(failed) if failed else 'none'}. "
        "The treatment is experimental server-side token-block compaction, not a generic sparse-attention kernel or production claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
