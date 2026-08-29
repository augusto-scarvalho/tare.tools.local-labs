#!/usr/bin/env python3
"""Paired Release-build performance crossover for the SLX-03 GDN fusion route."""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (  # noqa: E402
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import run_slx03_gdn_fusion_runtime as runtime  # noqa: E402

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-PERF-01"
BINARY = "/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03/bin/llama-server"
LIB_DIR = "/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-03/bin"
CUDA_LIB = f"{LIB_DIR}/libggml-cuda.so"
MODEL = "/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
SEED = 2026082820
BOOTSTRAP_SEED = 2026082821
BOOTSTRAP_REPLICATES = 20_000
N_PREDICT = 64

HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-PERF-01.json": "650e6c3df295feac7fbc584bc23d5f4cd970bdf45a6923d95e9798a4fb8f1c13",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-PERF-01/PRE_REGISTRATION.md": "fc1899a312beb67d3672431a593ecfe551bf8397d54248d0cab1ddb06d042ede",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/raw/receipt.json": "a46690e67b723368328a2f996d8b0d4e05e36d4c03e89590724561046e814029",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/REVIEW.json": "59bfad4ba63444b508b45908d772547846603accfeb9e0c7f3539280b79667ba",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/raw/receipt.json": "bb1391f3fb13792b0c71a658da4eb55eeeb64ac277cde00f93ee37bd78a9a256",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/REVIEW.json": "4ef6e60706cf87ef74052eebffdeafb7d3901a644cc0af6e6a3ef3a32925ceb0",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}
WSL_INPUTS = {
    BINARY: {"bytes": 17920, "sha256": "0267affe48ff9d49a13dbe0891b33598ead1179edd5db85ecb3b2c86c7e1fd0b"},
    CUDA_LIB: {"bytes": 63364248, "sha256": "378d85d3a09ae61982b016b186166dbe88a8dedf4ff9337dddafbe75ce70c7ce"},
    MODEL: {"bytes": 17923394624, "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"},
}
PROMPTS = (
    "Reply with only the result: 17 + 28 =",
    "Reply with only the result: 9 times 13 =",
    "Reply with only the result: 144 divided by 12 =",
    "Reply with only the result: 81 minus 37 =",
    "Responda apenas com o resultado: 23 + 19 =",
    "Responda apenas com o resultado: 7 vezes 16 =",
    "Reply with only the result: the next integer after 399 is",
    "Reply with only the result: half of 86 is",
    "Responda apenas com o resultado: 225 dividido por 15 =",
    "Reply with only the result: 19 squared =",
    "Responda apenas com o resultado: 1000 menos 457 =",
    "Reply with only the result: three quarters of 200 =",
)
PAIR_ORDERS = (("off", "on"), ("on", "off"), ("off", "on"), ("on", "off"), ("off", "on"), ("on", "off"))
BLOCKS = tuple(
    {
        "id": f"b{pair * 2 + offset + 1:02d}_{arm}",
        "arm": arm,
        "disable": "1" if arm == "off" else "0",
        "pair": pair,
        "position": offset,
        "rotation": (pair * 2) % len(PROMPTS),
    }
    for pair, order in enumerate(PAIR_ORDERS)
    for offset, arm in enumerate(order)
)
SERVER_ARGS = [
    BINARY,
    "-m", MODEL,
    "--alias", "qwen38-slx03-perf",
    "--host", "127.0.0.1",
    "--port", str(runtime.infra.PORT),
    "--ctx-size", "32768",
    "--flash-attn", "on",
    "--gpu-layers", "all",
    "--metrics",
    "--jinja",
    "--no-mmproj",
    "--cache-type-k", "q4_0",
    "--cache-type-v", "q4_0",
    "--parallel", "1",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--ctx-checkpoints", "32",
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "3",
]


def write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_inputs() -> tuple[dict[str, Any], list[pathlib.Path]]:
    ledger: dict[str, Any] = {"host": {}, "wsl": {}}
    frozen: list[pathlib.Path] = []
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"host identity mismatch: {relative}: {actual}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        frozen.append(path)
    for path, expected in WSL_INPUTS.items():
        size = int(runtime.infra.checked(runtime.infra.wsl("stat", "-L", "-c", "%s", path, timeout=120), f"stat {path}")["stdout"])
        digest = runtime.infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"WSL identity mismatch: {path}: {size} {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return ledger, frozen


def start_block(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    unit = f"local-labs-slx03-perf-{block['id']}.service"
    if runtime.infra.unit_state(unit)["load_state"] != "not-found":
        raise RuntimeError(f"reserved unit exists: {unit}")
    argv = [
        "systemd-run",
        f"--unit={unit}",
        "--collect",
        "--uid=augus",
        "--property=Type=simple",
        "--property=Restart=no",
        f"--setenv=LD_LIBRARY_PATH={LIB_DIR}",
        f"--setenv=GGML_CUDA_DISABLE_FUSION={block['disable']}",
        *SERVER_ARGS,
    ]
    launch = runtime.infra.checked(runtime.infra.wsl(*argv, root=True, timeout=60), f"launch {unit}")
    state = runtime.infra.wait_unit(unit, active=True)
    process = runtime.infra.process_values(state["main_pid"])
    environment = runtime.process_environment(state["main_pid"])
    runtime.infra.wait_health(runtime.infra.BASE_URL, timeout_seconds=600)
    if process["executable"] != BINARY:
        raise RuntimeError(f"wrong executable: {process['executable']}")
    if environment.get("GGML_CUDA_DISABLE_FUSION") != block["disable"] or environment.get("LD_LIBRARY_PATH") != LIB_DIR:
        raise RuntimeError(f"treatment identity mismatch: {environment.get('GGML_CUDA_DISABLE_FUSION')} {environment.get('LD_LIBRARY_PATH')}")
    return unit, {
        "launch": launch,
        "state": state,
        "process": process,
        "treatment_environment": {
            "GGML_CUDA_DISABLE_FUSION": environment.get("GGML_CUDA_DISABLE_FUSION"),
            "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH"),
        },
    }


def completion(prompt: str) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = runtime.infra.http_json(
        f"{runtime.infra.BASE_URL}/completion",
        {
            "prompt": prompt,
            "n_predict": N_PREDICT,
            "ignore_eos": True,
            "temperature": 0.0,
            "top_k": 1,
            "seed": SEED,
            "cache_prompt": False,
            "id_slot": 0,
            "stream": False,
        },
        timeout=900.0,
    )
    return {
        "http_status": status,
        "error": body.get("_error"),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "content": str(body.get("content") or ""),
        "tokens": body.get("tokens") or [],
        "tokens_predicted": body.get("tokens_predicted"),
        "timings": body.get("timings") or {},
        "response": body,
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def hierarchical_bootstrap(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = float(row[field])
        if value > 0 and math.isfinite(value):
            grouped.setdefault(int(row["pair"]), []).append(value)
    complete = len(grouped) == 6 and all(len(values) == len(PROMPTS) for values in grouped.values())
    if not complete:
        return {"field": field, "complete": False, "point": geometric_mean([value for values in grouped.values() for value in values]), "ci95_low": 0.0, "ci95_high": 0.0, "replicates": 0}
    pair_ids = sorted(grouped)
    generator = random.Random(BOOTSTRAP_SEED)
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        values: list[float] = []
        for _cluster in pair_ids:
            selected_pair = generator.choice(pair_ids)
            cluster = grouped[selected_pair]
            values.extend(generator.choice(cluster) for _ in range(len(PROMPTS)))
        replicates.append(geometric_mean(values))
    replicates.sort()
    observed = [value for pair in pair_ids for value in grouped[pair]]
    return {
        "field": field,
        "complete": True,
        "point": geometric_mean(observed),
        "ci95_low": percentile(replicates, 0.025),
        "ci95_high": percentile(replicates, 0.975),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "clusters": len(pair_ids),
        "observations": len(observed),
    }


def gate_pass(operator: str, actual: Any, threshold: Any) -> bool:
    if operator == "eq":
        return actual == threshold
    if operator == "gt":
        return actual > threshold
    if operator == "ge":
        return actual >= threshold
    raise ValueError(f"unsupported operator: {operator}")


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    samples_path = raw / "samples.jsonl"
    blocks_path = raw / "blocks.jsonl"
    if samples_path.exists() or blocks_path.exists() or (raw / "receipt.json").exists():
        raise RuntimeError("immutable execution outputs already exist")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    identities, frozen_paths = verify_inputs()
    write_json(raw / "binary_identity.json", {"verified": True, "identities": identities})

    initial_service = runtime.infra.unit_state(runtime.infra.PERSISTENT_UNIT)
    initial_gateway = runtime.infra.gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = runtime.infra.health(runtime.infra.EMBED_URL)
    if initial_service["active_state"] != "active" or embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError("persistent gateway or embedding unhealthy before experiment")
    write_json(raw / "runner_state.json", {"task_id": TASK_ID, "status": "running", "initial_service": initial_service, "initial_gateway": initial_gateway, "initial_model": initial_model})
    restoration: dict[str, Any] = {}
    execution_error: str | None = None
    try:
        runtime.infra.systemctl("stop", runtime.infra.PERSISTENT_UNIT)
        runtime.infra.wait_unit(runtime.infra.PERSISTENT_UNIT, active=False)
        occupied, body = runtime.infra.health(runtime.infra.BASE_URL)
        if occupied is not None:
            raise RuntimeError(f"temporary endpoint remains occupied: {occupied} {body}")
        for block in BLOCKS:
            unit = ""
            try:
                unit, launch = start_block(block)
                gpu_before = runtime.infra.gpu_state()
                warmups = [completion(f"Discarded fixed warmup {index}: continue deterministically.") for index in range(2)]
                if any(row["http_status"] != 200 or row["tokens_predicted"] != N_PREDICT for row in warmups):
                    raise RuntimeError(f"warmup failure in {block['id']}")
                order = list(range(len(PROMPTS)))
                rotation = int(block["rotation"])
                order = order[rotation:] + order[:rotation]
                consecutive_errors = 0
                for request_position, prompt_id in enumerate(order):
                    response = completion(PROMPTS[prompt_id])
                    consecutive_errors = consecutive_errors + 1 if response["http_status"] != 200 else 0
                    append_jsonl(samples_path, {
                        "block_id": block["id"],
                        "arm": block["arm"],
                        "pair": block["pair"],
                        "position": block["position"],
                        "request_position": request_position,
                        "prompt_id": prompt_id,
                        "prompt": PROMPTS[prompt_id],
                        **response,
                    })
                    print(f"{block['id']} {request_position + 1:02d}/{len(PROMPTS):02d} http={response['http_status']} tokens={response['tokens_predicted']}", flush=True)
                    if consecutive_errors >= 3:
                        raise RuntimeError(f"three consecutive failures in {block['id']}")
                append_jsonl(blocks_path, {
                    **block,
                    "block_id": block["id"],
                    "complete": True,
                    "launch": launch,
                    "warmups": [{"http_status": row["http_status"], "tokens_predicted": row["tokens_predicted"], "timings": row["timings"]} for row in warmups],
                    "gpu_before": gpu_before,
                    "gpu_after": runtime.infra.gpu_state(),
                    "recorded": len(PROMPTS),
                    "prompt_order": order,
                })
            finally:
                if unit:
                    runtime.infra.wsl("systemctl", "stop", unit, root=True, timeout=180)
                    try:
                        runtime.infra.wait_unit(unit, active=False, timeout_seconds=180)
                    except RuntimeError:
                        pass
            boundary, _ = runtime.infra.health(runtime.infra.EMBED_URL)
            if boundary != 200:
                raise RuntimeError(f"embedding unhealthy after {block['id']}")
    except Exception as error:
        execution_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if runtime.infra.unit_state(runtime.infra.PERSISTENT_UNIT)["active_state"] != "active":
                runtime.infra.systemctl("start", runtime.infra.PERSISTENT_UNIT)
            runtime.infra.wait_health(runtime.infra.GATEWAY_URL, timeout_seconds=600)
            restored_gateway = runtime.infra.restore_model(initial_model)
            final_service = runtime.infra.unit_state(runtime.infra.PERSISTENT_UNIT)
            final_embed, final_embed_body = runtime.infra.health(runtime.infra.EMBED_URL)
            restoration = {
                "gateway": restored_gateway,
                "service": final_service,
                "embedding": {"http_status": final_embed, "body": final_embed_body},
                "initial_model_restored": restored_gateway.get("current_model") == initial_model,
            }
        except Exception as error:
            restoration = {"error": f"{type(error).__name__}: {error}", "initial_model_restored": False}
        write_json(raw / "recovery_state.json", restoration)
        write_json(raw / "runner_state.json", {"task_id": TASK_ID, "status": "aborted" if execution_error else "blocks_complete", "error": execution_error, "initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})

    samples = read_jsonl(samples_path)
    blocks = read_jsonl(blocks_path)
    paired: list[dict[str, Any]] = []
    for pair in range(len(PAIR_ORDERS)):
        off = {int(row["prompt_id"]): row for row in samples if row["pair"] == pair and row["arm"] == "off"}
        on = {int(row["prompt_id"]): row for row in samples if row["pair"] == pair and row["arm"] == "on"}
        for prompt_id in sorted(set(off) & set(on)):
            off_row = off[prompt_id]
            on_row = on[prompt_id]
            off_tps = float(off_row.get("timings", {}).get("predicted_per_second") or 0.0)
            on_tps = float(on_row.get("timings", {}).get("predicted_per_second") or 0.0)
            off_wall = float(off_row.get("wall_ms") or 0.0)
            on_wall = float(on_row.get("wall_ms") or 0.0)
            paired.append({
                "pair": pair,
                "prompt_id": prompt_id,
                "prompt": PROMPTS[prompt_id],
                "off_block": off_row["block_id"],
                "on_block": on_row["block_id"],
                "off_predicted_per_second": off_tps,
                "on_predicted_per_second": on_tps,
                "decode_tps_ratio": on_tps / off_tps if off_tps > 0 else 0.0,
                "off_wall_ms": off_wall,
                "on_wall_ms": on_wall,
                "wall_throughput_ratio": off_wall / on_wall if on_wall > 0 else 0.0,
                "content_exact_match": off_row["content"] == on_row["content"],
                "tokens_exact_match": off_row["tokens"] == on_row["tokens"],
            })
    decode_statistics = hierarchical_bootstrap(paired, "decode_tps_ratio")
    wall_statistics = hierarchical_bootstrap(paired, "wall_throughput_ratio")
    statistics = {
        "method": "hierarchical_cluster_bootstrap_geometric_mean",
        "cluster_unit": "fresh_process_pair",
        "within_cluster_unit": "prompt",
        "decode": decode_statistics,
        "wall": wall_statistics,
    }
    treatment_ok = all(row.get("launch", {}).get("treatment_environment", {}).get("GGML_CUDA_DISABLE_FUSION") == row["disable"] for row in blocks)
    expected_order = [arm for order in PAIR_ORDERS for arm in order]
    actual_order = [row["arm"] for row in blocks]
    success_rate = sum(row["http_status"] == 200 for row in samples) / len(samples) if samples else 0.0
    fixed_rate = sum(row.get("tokens_predicted") == N_PREDICT for row in samples) / len(samples) if samples else 0.0
    parity_rate = sum(row["content_exact_match"] and row["tokens_exact_match"] for row in paired) / len(paired) if paired else 0.0
    service_restored = restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200 and restoration.get("service", {}).get("active_state") == "active"
    valid_blocks = len(blocks) if actual_order == expected_order and all(row["recorded"] == len(PROMPTS) for row in blocks) else 0
    metrics = {
        "binary_and_model_identity_verified": True,
        "explicit_fusion_controls_verified": treatment_ok,
        "valid_crossover_blocks": valid_blocks,
        "recorded_requests": len(samples),
        "successful_response_rate": success_rate,
        "fixed_decode_token_rate": fixed_rate,
        "exact_output_parity_rate": parity_rate,
        "cluster_bootstrap_ratio": decode_statistics["point"],
        "cluster_bootstrap_ratio_ci95_low": decode_statistics["ci95_low"],
        "cluster_bootstrap_ratio_ci95_high": decode_statistics["ci95_high"],
        "cluster_bootstrap_wall_ratio": wall_statistics["point"],
        "cluster_bootstrap_wall_ratio_ci95_low": wall_statistics["ci95_low"],
        "cluster_bootstrap_wall_ratio_ci95_high": wall_statistics["ci95_high"],
        "service_gateway_embedding_restored": service_restored,
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "paired_metrics.json", paired)
    write_json(raw / "statistical_analysis.json", statistics)
    write_json(raw / "dependency_hashes.json", {"host": HOST_INPUTS, "wsl": WSL_INPUTS})
    write_json(raw / "effective_route.json", {
        "release_treatments_observed": [{"block_id": row["block_id"], "arm": row["arm"], "pid": row["launch"]["state"]["main_pid"], "environment": row["launch"]["treatment_environment"]} for row in blocks],
        "runtime_route_proof": {"packet": "BACKLOG-SLX03-GDN-FUSION-RUNTIME-03", "receipt_sha256": HOST_INPUTS["runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/raw/receipt.json"], "review_sha256": HOST_INPUTS["runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-03/REVIEW.json"]},
        "claim_boundary": "causal Release-build ON/OFF performance contrast; route marker itself was established on the matching instrumented lineage",
    })
    write_json(raw / "environment.json", {"gpu_after": runtime.infra.gpu_state(), "wsl_distro": runtime.infra.WSL_DISTRO})
    write_json(raw / "hardware_metrics.json", {"blocks": [{"block_id": row["block_id"], "arm": row["arm"], "gpu_before": row["gpu_before"], "gpu_after": row["gpu_after"]} for row in blocks], "primary_endpoint": "server predicted_per_second", "secondary_endpoint": "client wall throughput"})
    write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "frozen single-slot Qwen3.8 Release decode throughput only"})
    write_json(raw / "paired_baseline.json", {"baseline": "GGML_CUDA_DISABLE_FUSION=1", "treatment": "GGML_CUDA_DISABLE_FUSION=0", "process_pairs": len(PAIR_ORDERS), "paired_requests": len(paired)})
    write_json(raw / "service_identity.json", {"initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})
    write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True, "embedding_service_stopped": False, "restoration": restoration})
    write_json(raw / "treatment_controls.json", {"order": expected_order, "blocks": list(BLOCKS), "server_args": SERVER_ARGS, "environment_variable": "GGML_CUDA_DISABLE_FUSION", "seed": SEED, "n_predict": N_PREDICT, "ignore_eos": True, "prompts": list(PROMPTS), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES})
    exact_paths = [samples_path, blocks_path, raw / "paired_metrics.json", raw / "statistical_analysis.json"]
    write_json(raw / "end_to_end_artifact.json", {"exact_files": {str(path.relative_to(raw)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in exact_paths}, "hash_semantics": "raw file bytes"})

    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", "eq", True),
        "treatment_identity": ("explicit_fusion_controls_verified", "eq", True),
        "balanced_crossover": ("valid_crossover_blocks", "eq", 12),
        "sample_size": ("recorded_requests", "eq", 144),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "fixed_decode_work": ("fixed_decode_token_rate", "eq", 1.0),
        "semantic_parity": ("exact_output_parity_rate", "eq", 1.0),
        "decode_speedup": ("cluster_bootstrap_ratio_ci95_low", "gt", 1.0),
        "wall_non_regression": ("cluster_bootstrap_wall_ratio_ci95_low", "ge", 0.98),
        "service_recovery": ("service_gateway_embedding_restored", "eq", True),
    }
    gates = {
        name: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": gate_pass(operator, metrics[metric], threshold)}
        for name, (metric, operator, threshold) in definitions.items()
    }
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(),
        started_at_utc=started,
        started_monotonic=monotonic,
        input_paths=[*frozen_paths, *evidence_files],
        packages=[],
        runtime={"execution_mode": "release_runtime_paired_crossover", "blocks": len(blocks), "requests": len(samples), "model": MODEL, "n_predict": N_PREDICT, "bootstrap_replicates": BOOTSTRAP_REPLICATES},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json",
        "binary_identity": "raw/binary_identity.json",
        "dependency_hashes": "raw/dependency_hashes.json",
        "effective_route": "raw/effective_route.json",
        "end_to_end_artifact": "raw/end_to_end_artifact.json",
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
        "statistical_analysis": "raw/statistical_analysis.json",
        "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "SLX03_GDN_FUSION_DECODE_SPEEDUP_CONFIRMED_R1" if not failed else "SLX03_GDN_FUSION_DECODE_SPEEDUP_NOT_DEMONSTRATED_R1"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Decode ratio `{decode_statistics['point']:.6f}` with hierarchical-bootstrap 95% CI `[{decode_statistics['ci95_low']:.6f}, {decode_statistics['ci95_high']:.6f}]`; "
        f"wall-throughput ratio `{wall_statistics['point']:.6f}` with 95% CI `[{wall_statistics['ci95_low']:.6f}, {wall_statistics['ci95_high']:.6f}]`; "
        f"requests `{len(samples)}/144`; exact parity `{parity_rate:.4f}` over `{len(paired)}` pairs; service restored `{service_restored}`. "
        f"Failed gates: `{', '.join(failed) if failed else 'none'}`. Claim is limited to the frozen single-slot Qwen3.8 Release request shape.\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
