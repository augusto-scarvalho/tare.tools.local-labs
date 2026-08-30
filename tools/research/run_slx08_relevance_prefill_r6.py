#!/usr/bin/env python3
"""Run the three-arm SLX08 relevance-selected prefill experiment."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import build_provenance, provenance_complete, sha256_file
from tools.research import run_slx08_physical_prefill_r4 as base
from tools.research import run_slx08_physical_prefill_r5 as r5

TASK_ID = "BACKLOG-SLX08-RELEVANCE-PREFILL-06"
PRE_REG_SHA256 = "12e81ac83987c353331796e0fbccb95fa6790387e1ca93df3e15a1e08278625c"
EXPECTED_BINARY_SHA256 = "4395a601202ec76bcaef1d10db97849a92b311d8c31e4afce4d8b961609807a1"
SERVER_LIBRARY = "/home/augus/build/slop-slx08/bin/libllama-server-impl.so"
EXPECTED_SERVER_LIBRARY_SHA256 = "3e01af35056e6168c8a3713aa5abaefb449fbe2600789a63d5b15e0f1721a5ef"
SOURCE_HASHES = {
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/raw/receipt.json": "ed54519837bf23f174fcf4fdeef451a5d8e776fc8d3857b91b5e61b5190eb4eb",
    "runs/research/BACKLOG-SLX08-PHYSICAL-PREFILL-05/REVIEW.json": "1415bc422459a095e93b0474fca0d262474a8c87480ac618f3402bf477f96de6",
}
ARMS = ("dense", "naive", "relevance")
NAIVE_BLOCKS = [0, 2, 4, 6, 8, 10, 12, 15]


def answer_correct(content: str, expected: int) -> bool:
    return any(int(value) == expected for value in re.findall(r"(?<!\d)(\d{4})(?!\d)", content))


def select_relevant_blocks(block_texts: list[str], query_key: str, retain: int = 8) -> list[int]:
    """Keep endpoints and rank middle blocks by exact query-key occurrence."""
    if len(block_texts) != 16 or retain != 8:
        raise ValueError("R6 selector requires 16 blocks and an 8-block budget")
    ranked = sorted(range(1, 15), key=lambda index: (query_key not in block_texts[index], index))
    return sorted([0, 15, *ranked[: retain - 2]])


def build_fixture(case_id: int) -> dict:
    query_key = f"R6CASE{case_id:03d}"
    expected = 7000 + case_id
    evidence_block = 1 + case_id % 14
    block_texts = [
        "Retrieval task. At the final query, output only the four-digit vault code for the exact case key. Ignore every other case key and all filler."
    ]
    for block in range(1, 15):
        if block == evidence_block:
            block_texts.append(f"Authoritative record: exact case key {query_key} has vault code {expected:04d}.")
        else:
            decoy_key = f"DECOY{case_id:03d}B{block:02d}"
            decoy_code = 8000 + ((case_id * 17 + block) % 999)
            block_texts.append(f"Unrelated record: case key {decoy_key} has vault code {decoy_code:04d}.")
    block_texts.append(f"Final query: What is the vault code for exact case key {query_key}? Answer only four digits:")

    filler = base.tokenize(" Archived filler with no answer for the requested key.", add_special=False)
    token_blocks = []
    for index, text in enumerate(block_texts):
        text_tokens = base.tokenize(text, add_special=index == 0)
        if index == 15:
            token_blocks.append(base.pad_tokens([], filler, base.BLOCK_SIZE, text_tokens))
        else:
            token_blocks.append(base.pad_tokens(text_tokens, filler, base.BLOCK_SIZE))
    tokens = [token for block in token_blocks for token in block]
    selected = select_relevant_blocks(block_texts, query_key)
    if evidence_block not in selected or len(tokens) != base.PROMPT_TOKENS:
        raise AssertionError("fixture construction violated R6 invariants")
    return {
        "case_id": case_id,
        "query_key": query_key,
        "expected": expected,
        "evidence_block": evidence_block,
        "relevance_blocks": selected,
        "naive_retains_evidence": evidence_block in NAIVE_BLOCKS,
        "prompt_sha256": base.canonical_sha256(tokens),
        "tokens": tokens,
    }


def paired_ci_low(pairs: list[dict], left: str, right: str) -> float:
    differences = [int(pair[left]["correct"]) - int(pair[right]["correct"]) for pair in pairs]
    mean = statistics.fmean(differences)
    if len(set(differences)) == 1:
        return mean
    return mean - 1.96 * statistics.stdev(differences) / math.sqrt(len(differences))


def score(pairs: list[dict], restored: bool, embedding_status: int | None) -> dict:
    rows = {arm: [pair[arm] for pair in pairs] for arm in ARMS}
    relevance_ratios = [pair["dense"]["ttft_ms"] / pair["relevance"]["ttft_ms"] for pair in pairs]
    return {
        "dense_requests": len(rows["dense"]),
        "naive_requests": len(rows["naive"]),
        "relevance_requests": len(rows["relevance"]),
        "relevance_route_observation_rate": statistics.fmean(row["route_observed"] for row in rows["relevance"]),
        "relevance_evidence_retention_rate": statistics.fmean(row["evidence_retained"] for row in rows["relevance"]),
        "naive_evidence_retention_rate": statistics.fmean(row["evidence_retained"] for row in rows["naive"]),
        "relevance_median_retained_fraction": statistics.median(row["telemetry"]["retained_attention_fraction"] for row in rows["relevance"]),
        "dense_accuracy": statistics.fmean(row["correct"] for row in rows["dense"]),
        "naive_accuracy": statistics.fmean(row["correct"] for row in rows["naive"]),
        "relevance_accuracy": statistics.fmean(row["correct"] for row in rows["relevance"]),
        "relevance_vs_dense_accuracy_delta": statistics.fmean(int(pair["relevance"]["correct"]) - int(pair["dense"]["correct"]) for pair in pairs),
        "relevance_vs_dense_accuracy_delta_ci95_low": paired_ci_low(pairs, "relevance", "dense"),
        "relevance_vs_naive_accuracy_delta": statistics.fmean(int(pair["relevance"]["correct"]) - int(pair["naive"]["correct"]) for pair in pairs),
        "relevance_vs_dense_p50_ttft_speedup": statistics.median(relevance_ratios),
        "relevance_vs_dense_p95_ttft_speedup": base.percentile([row["ttft_ms"] for row in rows["dense"]], 0.95) / base.percentile([row["ttft_ms"] for row in rows["relevance"]], 0.95),
        **{f"{arm}_p50_ttft_ms": statistics.median(row["ttft_ms"] for row in arm_rows) for arm, arm_rows in rows.items()},
        **{f"{arm}_p95_ttft_ms": base.percentile([row["ttft_ms"] for row in arm_rows], 0.95) for arm, arm_rows in rows.items()},
        "original_service_restored": int(restored),
        "embedding_health": embedding_status,
    }


def evaluate_gates(metrics: dict) -> dict:
    definitions = {
        "dense_control": ("dense_requests", "ge", 64),
        "naive_control": ("naive_requests", "ge", 64),
        "relevance_treatment": ("relevance_requests", "ge", 64),
        "route_observation": ("relevance_route_observation_rate", "eq", 1.0),
        "evidence_retention": ("relevance_evidence_retention_rate", "eq", 1.0),
        "retained_fraction": ("relevance_median_retained_fraction", "eq", 0.5),
        "dense_semantic_floor": ("dense_accuracy", "ge", 0.9),
        "relevance_semantic_floor": ("relevance_accuracy", "ge", 0.9),
        "semantic_noninferiority": ("relevance_vs_dense_accuracy_delta_ci95_low", "ge", -0.03),
        "selector_value": ("relevance_vs_naive_accuracy_delta", "ge", 0.2),
        "ttft_gain": ("relevance_vs_dense_p50_ttft_speedup", "ge", 1.1),
        "tail_safety": ("relevance_vs_dense_p95_ttft_speedup", "ge", 1.0),
        "service_restore": ("original_service_restored", "eq", 1),
        "embedding_integrity": ("embedding_health", "eq", 200),
    }
    operators = {"eq": lambda actual, threshold: actual == threshold, "ge": lambda actual, threshold: actual >= threshold}
    return {
        gate: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": operators[operator](metrics[metric], threshold)}
        for gate, (metric, operator, threshold) in definitions.items()
    }


def request_for(fixture: dict, arm: str) -> dict:
    request = {
        "prompt": fixture["tokens"],
        "slx08_selected_block_prefill": arm != "dense",
        "n_predict": 16,
        "stream": True,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,
    }
    if arm == "relevance":
        request["slx08_selected_block_indices"] = fixture["relevance_blocks"]
    return request


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
    library_sha = base.wsl_sha256(SERVER_LIBRARY)
    model_sha = base.wsl_sha256(base.MODEL)
    if binary_sha != EXPECTED_BINARY_SHA256 or library_sha != EXPECTED_SERVER_LIBRARY_SHA256 or model_sha != base.MODEL_SHA256:
        raise ValueError(f"runtime identity mismatch: binary={binary_sha}, library={library_sha}, model={model_sha}")
    slop_sources = [
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-context.cpp"),
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.cpp"),
        pathlib.Path(r"C:\projects\slop.cpp\tools\server\server-task.h"),
    ]
    source_ledger = {str(path): sha256_file(path) for path in slop_sources}
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    original = base.service_state()
    original_stable = r5.stable_service_identity(original)
    if original_stable["active_state"] != "active" or original_stable["health_status"] != 200 or base.health(8081)[0] != 200:
        raise RuntimeError(f"serving baseline is not healthy: {original_stable}")

    fixtures = [build_fixture(case_id) for case_id in range(base.PAIRS)]
    fixture_public = [{key: value for key, value in fixture.items() if key != "tokens"} for fixture in fixtures]
    base.write_json(raw / "fixtures.json", fixture_public)
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
                base.EXPERIMENT_BINARY, "-m", base.MODEL, "--alias", "slx08-r6-qwen38", "--host", "127.0.0.1", "--port", str(base.TEMP_PORT),
                "--ctx-size", "8192", "--flash-attn", "on", "--gpu-layers", "all", "--parallel", "1", "--batch-size", "2048",
                "--ubatch-size", "512", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--no-mmproj", "--metrics",
            ]
            log_handle = (raw / "temporary_server.log").open("w", encoding="utf-8")
            temporary = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            base.wait_health(base.TEMP_PORT, 200, 300.0)
            run.checkpoint("experimental_server_ready", {"binary_sha256": binary_sha, "server_library_sha256": library_sha, "model_sha256": model_sha})

            invalid_requests = {
                "indices_without_route": {"prompt": fixtures[0]["tokens"], "slx08_selected_block_prefill": False, "slx08_selected_block_indices": fixtures[0]["relevance_blocks"], "n_predict": 1},
                "wrong_count": {"prompt": fixtures[0]["tokens"], "slx08_selected_block_prefill": True, "slx08_selected_block_indices": [0, 15], "n_predict": 1},
                "duplicate": {"prompt": fixtures[0]["tokens"], "slx08_selected_block_prefill": True, "slx08_selected_block_indices": [0, 1, 2, 3, 4, 5, 5, 15], "n_predict": 1},
                "missing_endpoint": {"prompt": fixtures[0]["tokens"], "slx08_selected_block_prefill": True, "slx08_selected_block_indices": [1, 2, 3, 4, 5, 6, 7, 15], "n_predict": 1},
            }
            invalid_results = {}
            for name, payload in invalid_requests.items():
                status, response = base.http_json(f"{base.TEMP_BASE}/completion", payload)
                invalid_results[name] = {"status": status, "response": response}
                if status == 200:
                    raise RuntimeError(f"invalid explicit selection did not fail closed: {name}")
            base.write_json(raw / "failure_reproduction.json", invalid_results)

            for arm in ARMS:
                r5.stream_completion(request_for(fixtures[0], arm))

            for fixture in fixtures:
                offset = fixture["case_id"] % len(ARMS)
                order = [*ARMS[offset:], *ARMS[:offset]]
                pair = {"case_id": fixture["case_id"], "expected": fixture["expected"], "evidence_block": fixture["evidence_block"], "prompt_sha256": fixture["prompt_sha256"]}
                for arm in order:
                    request = request_for(fixture, arm)
                    status, final, content, ttft_ms = r5.stream_completion(request)
                    telemetry = final.get("slx08_prefill")
                    if status != 200 or not isinstance(telemetry, dict):
                        raise RuntimeError(f"invalid final response for {arm}: {status}: {final}")
                    expected_mode = {"dense": "dense", "naive": "default_alternating", "relevance": "explicit"}[arm]
                    expected_blocks = list(range(16)) if arm == "dense" else (NAIVE_BLOCKS if arm == "naive" else fixture["relevance_blocks"])
                    route_observed = telemetry.get("selection_mode") == expected_mode and telemetry.get("selected_block_indices") == expected_blocks
                    expected_tokens = 4096 if arm == "dense" else 2048
                    if telemetry.get("original_prompt_tokens") != 4096 or telemetry.get("retained_prompt_tokens") != expected_tokens or not route_observed:
                        raise RuntimeError(f"route materiality mismatch for {arm}: {telemetry}")
                    row = {
                        "task_id": TASK_ID,
                        "case_id": fixture["case_id"],
                        "arm": arm,
                        "expected": fixture["expected"],
                        "evidence_block": fixture["evidence_block"],
                        "prompt_sha256": fixture["prompt_sha256"],
                        "request": request,
                        "http_status": status,
                        "final_response": final,
                        "content": content,
                        "correct": answer_correct(content, fixture["expected"]),
                        "ttft_ms": ttft_ms,
                        "telemetry": telemetry,
                        "route_observed": route_observed,
                        "evidence_retained": fixture["evidence_block"] in telemetry["selected_block_indices"],
                    }
                    pair[arm] = row
                    run.record(row)
                if len({pair[arm]["prompt_sha256"] for arm in ARMS}) != 1:
                    raise RuntimeError("three-arm prompt bytes differ")
                pairs.append(pair)
                if len(pairs) % 8 == 0:
                    run.checkpoint("three_arm_progress", {"completed_fixtures": len(pairs), "expected_fixtures": base.PAIRS})
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
            restored_stable = r5.stable_service_identity(restored_state)
            embedding_status = base.health(8081)[0]
            restored = original_stable == restored_stable and restored_health.get("current_model") == original_stable["current_model"] and embedding_status == 200
            run.restored({"original_stable": original_stable, "restored_stable": restored_stable, "embedding_health": embedding_status}, ok=restored)

        metrics = score(pairs, restored, embedding_status)
        gates = evaluate_gates(metrics)
        base.write_json(raw / "actual_scores.json", metrics)
        base.write_json(raw / "artifact_hashes.json", {"frozen_inputs": inputs, "slop_sources": source_ledger, "binary_sha256": binary_sha, "server_library_sha256": library_sha, "model_sha256": model_sha})
        base.write_json(raw / "dataset_hashes.json", {"fixtures_sha256": base.canonical_sha256(fixture_public), "prompt_hashes": [fixture["prompt_sha256"] for fixture in fixtures]})
        base.write_json(raw / "effective_route.json", {"dense": "ordinary 16-block prefill", "naive": "default alternating 8-block token compaction", "relevance": "explicit query-key-selected 8-block token compaction", "environment_gate": "SLOP_EXPERIMENTAL_SLX08=1"})
        base.write_json(raw / "falsifiable_hypothesis.json", {"fixtures": 64, "arms": list(ARMS), "prompt_tokens": 4096, "retained_fraction": 0.5, "all_gates_required": True})
        base.write_json(raw / "hardware_metrics.json", {key: value for key, value in metrics.items() if "ttft" in key})
        base.write_json(raw / "independent_evaluation.json", {"scorer": "exact standalone four-digit code", "selector": "exact case-key overlap with deterministic tie break", "ttft_contract": "host monotonic to first non-empty streamed content", "recomputable_from": "raw/samples.jsonl"})
        base.write_json(raw / "invalidation_rules.json", {"incorrect_route_telemetry_invalid": True, "evidence_not_retained_invalid": True, "unequal_prompt_hash_invalid": True, "invalid_indices_accepted_invalid": True, "restoration_failure_invalid": True})
        base.write_json(raw / "invariant_controls.json", {"decode": {"n_predict": 16, "stream": True, "temperature": 0.0, "top_k": 1, "seed": 0, "cache_prompt": False}, "arm_order": "three-period Latin rotation", "prompt_tokens": 4096})
        base.write_json(raw / "paired_baseline.json", [{"case_id": pair["case_id"], "prompt_sha256": pair["prompt_sha256"], **{f"{arm}_correct": pair[arm]["correct"] for arm in ARMS}, **{f"{arm}_ttft_ms": pair[arm]["ttft_ms"] for arm in ARMS}} for pair in pairs])
        base.write_json(raw / "physical_route_telemetry.json", [{"case_id": pair["case_id"], **{arm: pair[arm]["telemetry"] for arm in ARMS}} for pair in pairs])
        base.write_json(raw / "real_implementation.json", {"source_root": r"C:\projects\slop.cpp", "explicit_block_index_api": True, "server_side_semantic_selector": False, "cuda_kernel_added": False, "production_default": False})
        base.write_json(raw / "recovery_state.json", {"original_stable": original_stable, "restored_stable": restored_stable})
        base.write_json(raw / "semantic_parity.json", {key: value for key, value in metrics.items() if "accuracy" in key or "evidence_retention" in key})
        base.write_json(raw / "service_identity.json", {"original_stable": original_stable, "temporary": {"binary_sha256": binary_sha, "server_library_sha256": library_sha, "model_sha256": model_sha}, "restored_stable": restored_stable})
        base.write_json(raw / "service_maintenance.json", {"original_service_stopped": True, "temporary_port": base.TEMP_PORT, "original_service_restored": bool(metrics["original_service_restored"]), "embedding_health": embedding_status})
        base.write_json(raw / "source_execution_receipt.json", SOURCE_HASHES)
        base.write_json(raw / "treatment_materiality.json", {"requests_per_arm": 64, "arms": list(ARMS), "original_blocks": 16, "selected_blocks": 8, "response_indices_bound": True})

        evidence_paths = sorted(path for path in raw.iterdir() if path.is_file() and path.name not in {"receipt.json", "run.events.jsonl", "run.terminal.json"})
        provenance = build_provenance(
            script_path=pathlib.Path(__file__).resolve(),
            started_at_utc=started_utc,
            started_monotonic=started_mono,
            input_paths=[*[ROOT / relative for relative in SOURCE_HASHES], prereg, pathlib.Path(__file__).resolve(), *slop_sources, *evidence_paths],
            packages=["pytest"],
            runtime={"execution_mode": "physical_relevance_selected_prefill_r6", "binary_sha256": binary_sha, "server_library_sha256": library_sha, "model_sha256": model_sha, "gpu": "RTX 3090"},
        )
        complete, errors = provenance_complete(provenance)
        if not complete:
            raise ValueError(f"incomplete provenance: {errors}")
        evidence = {key: f"raw/{filename}" for key, filename in {
            "acceptance_gates": "receipt.json", "actual_scores": "actual_scores.json", "artifact_hashes": "artifact_hashes.json", "dataset_hashes": "dataset_hashes.json",
            "effective_route": "effective_route.json", "failure_reproduction": "failure_reproduction.json", "falsifiable_hypothesis": "falsifiable_hypothesis.json",
            "hardware_metrics": "hardware_metrics.json", "independent_evaluation": "independent_evaluation.json", "invalidation_rules": "invalidation_rules.json",
            "invariant_controls": "invariant_controls.json", "paired_baseline": "paired_baseline.json", "physical_route_telemetry": "physical_route_telemetry.json",
            "provenance": "receipt.json", "raw_samples": "samples.jsonl", "real_implementation": "real_implementation.json", "receipt_fingerprint": "receipt.json",
            "recovery_state": "recovery_state.json", "semantic_parity": "semantic_parity.json", "service_identity": "service_identity.json",
            "service_maintenance": "service_maintenance.json", "source_execution_receipt": "source_execution_receipt.json", "treatment_materiality": "treatment_materiality.json",
        }.items()}
        receipt = run.seal({"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence})
    return receipt, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = execute(args.outdir.resolve())
    passed = all(gate["pass"] for gate in receipt["gates"].values())
    claim = "SLX08_RELEVANCE_SELECTED_PREFILL_QUALIFIED_R6" if passed else "SLX08_RELEVANCE_SELECTED_PREFILL_REJECTED_R6"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review. Dense/naive/relevance accuracy: "
        f"{metrics['dense_accuracy']:.4f}/{metrics['naive_accuracy']:.4f}/{metrics['relevance_accuracy']:.4f}; relevance p50 TTFT speedup "
        f"{metrics['relevance_vs_dense_p50_ttft_speedup']:.4f}x; failed gates: {', '.join(failed) if failed else 'none'}. "
        "Bounded to client-selected server token compaction on the frozen R6 panel.\n",
        encoding="utf-8",
    )
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
