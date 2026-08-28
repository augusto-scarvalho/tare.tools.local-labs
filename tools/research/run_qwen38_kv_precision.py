#!/usr/bin/env python3
"""Physical F16/Q4_0 KV-cache crossover for Qwen3.8 serving."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
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
from tools.research import run_mtp_persistence_first_instance as infra


TASK_ID = "BACKLOG-QWEN38-KV-PRECISION-01"
PRE_REG_SHA256 = "8b0ddc15d0bca6cc6ce0021a6d929f739fef2fc33de432524d84e5b1948f51b2"
SOURCE_HASHES = {
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "docs/HANDOFF_2026-08-26_CONSOLIDATED_RESEARCH_BACKLOG.md": "895fec3ac345bdf26350b4a97f513bf4f4b3bad9898d09701db07a985f8b7d55",
}
BLOCKS = (
    {"id": "b1_f16", "arm": "f16", "cache": "f16", "pair": 0},
    {"id": "b2_q4", "arm": "q4", "cache": "q4_0", "pair": 0},
    {"id": "b3_q4", "arm": "q4", "cache": "q4_0", "pair": 1},
    {"id": "b4_f16", "arm": "f16", "cache": "f16", "pair": 1},
)
SERVER_COMMON = [
    infra.BINARY, "-m", infra.MODEL, "--alias", "qwen38-kv-precision", "--host", "127.0.0.1",
    "--port", str(infra.PORT), "--ctx-size", "32768", "--flash-attn", "on",
    "--gpu-layers", "all", "--metrics", "--jinja", "--no-mmproj", "--parallel", "1",
    "--batch-size", "2048", "--ubatch-size", "512", "--ctx-checkpoints", "32",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
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
    paths: list[pathlib.Path] = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {relative}: {actual}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        paths.append(path)
    prereg = ROOT / "runs/research" / TASK_ID / "PRE_REGISTRATION.md"
    actual_prereg = sha256_file(prereg)
    if actual_prereg != PRE_REG_SHA256:
        raise ValueError(f"preregistration mismatch: {actual_prereg}")
    ledger["host"][str(prereg.relative_to(ROOT)).replace("\\", "/")] = {"bytes": prereg.stat().st_size, "sha256": actual_prereg}
    paths.append(prereg)
    for path, expected in infra.EXPECTED_WSL.items():
        size = infra.stat_wsl(path)
        digest = infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"frozen WSL identity mismatch: {path}: {size} {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return ledger, paths


def panel() -> list[dict[str, Any]]:
    rows = read_jsonl(ROOT / "workloads/gsm8k.jsonl")[:32]
    if len(rows) != 32:
        raise ValueError("frozen GSM8K panel does not contain 32 rows")
    return rows


def start_block(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    unit = f"local-labs-kv-precision-{block['id']}.service"
    if infra.unit_state(unit)["load_state"] != "not-found":
        raise RuntimeError(f"reserved unit exists: {unit}")
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--uid=augus", "--property=Type=simple",
        "--property=Restart=no", f"--setenv=LD_LIBRARY_PATH={infra.LIB_DIR}", *SERVER_COMMON,
        "--cache-type-k", block["cache"], "--cache-type-v", block["cache"],
    ]
    launch = infra.checked(infra.wsl(*argv, root=True, timeout=60.0), f"launch {unit}")
    state = infra.wait_unit(unit, active=True)
    process = infra.process_values(state["main_pid"])
    infra.wait_health(infra.BASE_URL)
    cache_tokens = [process["argv"][index + 1] for index, token in enumerate(process["argv"]) if token in {"--cache-type-k", "--cache-type-v"}]
    if cache_tokens != [block["cache"], block["cache"]]:
        raise RuntimeError(f"cache treatment mismatch in argv: {cache_tokens}")
    return unit, {"launch": launch, "state": state, "process": process, "cache_tokens": cache_tokens}


def completion(prompt: str, n_predict: int = 256) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = infra.http_json(f"{infra.BASE_URL}/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "seed": 20260826, "cache_prompt": False, "id_slot": 0, "stream": False,
    })
    timings = body.get("timings") or {}
    predicted_n = int(timings.get("predicted_n") or body.get("tokens_predicted") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0)
    return {
        "http_status": status, "error": body.get("_error"),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "content": str(body.get("content") or ""), "timings": timings,
        "predicted_n": predicted_n, "predicted_ms": predicted_ms,
        "throughput_tps": round(predicted_n * 1000 / predicted_ms, 6) if predicted_ms > 0 else None,
        "response": body,
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    logs = raw / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    samples_path = raw / "samples.jsonl"
    blocks_path = raw / "blocks.jsonl"
    state_path = raw / "runner_state.json"
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    frozen, frozen_paths = verify_inputs()
    write_json(raw / "binary_identity.json", frozen)
    tasks = panel()
    existing_samples = read_jsonl(samples_path)
    existing_blocks = read_jsonl(blocks_path)
    completed_blocks = {row["block_id"] for row in existing_blocks if row.get("complete")}

    initial_service = infra.unit_state(infra.PERSISTENT_UNIT)
    initial_gateway = infra.gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = infra.health(infra.EMBED_URL)
    if initial_service["active_state"] != "active" or embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError("persistent gateway or embedding endpoint unhealthy before experiment")
    state = {
        "task_id": TASK_ID, "started_at_utc": started_utc, "status": "running",
        "initial_service": initial_service, "initial_gateway": initial_gateway,
        "initial_model": initial_model, "completed_blocks": sorted(completed_blocks),
    }
    write_json(state_path, state)
    restoration: dict[str, Any] = {}
    execution_error: str | None = None
    try:
        infra.systemctl("stop", infra.PERSISTENT_UNIT)
        infra.wait_unit(infra.PERSISTENT_UNIT, active=False)
        occupied, body = infra.health(infra.BASE_URL)
        if occupied is not None:
            raise RuntimeError(f"temporary endpoint remains occupied: {occupied} {body}")
        for block in BLOCKS:
            if block["id"] in completed_blocks:
                continue
            unit = ""
            record: dict[str, Any] = {}
            try:
                unit, launch = start_block(block)
                warmups = [completion(f"Discarded warmup {index}: compute {17 + index} plus {31 + index}.", 64) for index in range(4)]
                if any(row["http_status"] != 200 for row in warmups):
                    raise RuntimeError(f"warmup failure in {block['id']}")
                gpu = infra.gpu_state()
                consecutive_errors = 0
                for index, task in enumerate(tasks):
                    prompt = "Solve carefully and end with #### followed by the numeric answer.\n" + str(task["prompt"])
                    response = completion(prompt)
                    extracted = gsm8k_extract(response["content"])
                    correct = numeric_equal(extracted, str(task["answer"]))
                    sample = {
                        "block_id": block["id"], "arm": block["arm"], "cache": block["cache"],
                        "pair": block["pair"], "index": index, "task_id": task["task_id"],
                        "prompt": prompt, "gold": str(task["answer"]), "extracted": extracted,
                        "correct": bool(correct), **response,
                    }
                    append_jsonl(samples_path, sample)
                    existing_samples.append(sample)
                    consecutive_errors = consecutive_errors + 1 if response["http_status"] != 200 else 0
                    print(f"{block['id']} {index + 1:02d}/32 http={response['http_status']} correct={correct}", flush=True)
                    if consecutive_errors >= 3:
                        raise RuntimeError(f"three consecutive failures in {block['id']}")
                record = {**block, "block_id": block["id"], "complete": True, "launch": launch, "warmups": warmups, "gpu": gpu, "recorded": 32}
                append_jsonl(blocks_path, record)
                existing_blocks.append(record)
                completed_blocks.add(block["id"])
                state["completed_blocks"] = sorted(completed_blocks)
                write_json(state_path, state)
            finally:
                if unit:
                    journal = infra.wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True)["stdout"]
                    (logs / f"{unit}.log").write_text(journal, encoding="utf-8")
                    infra.wsl("systemctl", "stop", unit, root=True, timeout=180.0)
                    try:
                        infra.wait_unit(unit, active=False, timeout_seconds=180.0)
                    except RuntimeError:
                        pass
            boundary, _ = infra.health(infra.EMBED_URL)
            if boundary != 200:
                raise RuntimeError(f"embedding unhealthy after {block['id']}")
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        state.update({"status": "aborted", "error": execution_error})
        write_json(state_path, state)
        raise
    finally:
        try:
            if infra.unit_state(infra.PERSISTENT_UNIT)["active_state"] != "active":
                infra.systemctl("start", infra.PERSISTENT_UNIT)
            infra.wait_health(infra.GATEWAY_URL)
            restored_gateway = infra.restore_model(initial_model)
            final_service = infra.unit_state(infra.PERSISTENT_UNIT)
            final_embed, final_embed_body = infra.health(infra.EMBED_URL)
            restoration = {
                "gateway": restored_gateway, "service": final_service,
                "embedding": {"http_status": final_embed, "body": final_embed_body},
                "initial_model_restored": restored_gateway.get("current_model") == initial_model,
            }
        except Exception as exc:
            restoration = {"error": f"{type(exc).__name__}: {exc}", "initial_model_restored": False}
            if execution_error is None:
                state.update({"status": "aborted", "error": restoration["error"]})
        state["restoration"] = restoration
        write_json(raw / "recovery_state.json", restoration)
        write_json(state_path, state)

    samples = read_jsonl(samples_path)
    block_records = read_jsonl(blocks_path)
    f16 = [row for row in samples if row["arm"] == "f16"]
    q4 = [row for row in samples if row["arm"] == "q4"]
    pairs = []
    for pair_index in (0, 1):
        left = {row["task_id"]: row for row in f16 if row["pair"] == pair_index}
        right = {row["task_id"]: row for row in q4 if row["pair"] == pair_index}
        for task_id in sorted(set(left) & set(right)):
            pairs.append({
                "pair": pair_index, "task_id": task_id,
                "f16_extracted": left[task_id]["extracted"], "q4_extracted": right[task_id]["extracted"],
                "match": left[task_id]["extracted"] == right[task_id]["extracted"],
            })
    f16_accuracy = sum(row["correct"] for row in f16) / len(f16)
    q4_accuracy = sum(row["correct"] for row in q4) / len(q4)
    f16_tps = statistics.median(row["throughput_tps"] for row in f16 if row["throughput_tps"] is not None)
    q4_tps = statistics.median(row["throughput_tps"] for row in q4 if row["throughput_tps"] is not None)
    f16_vram = statistics.median(float(row["gpu"]["memory.used"]) for row in block_records if row["arm"] == "f16")
    q4_vram = statistics.median(float(row["gpu"]["memory.used"]) for row in block_records if row["arm"] == "q4")
    expected_order = ["f16", "q4", "q4", "f16"]
    actual_order = [row["arm"] for row in block_records]
    metrics = {
        "explicit_cache_controls_verified": all(row.get("launch", {}).get("cache_tokens") == [row["cache"], row["cache"]] for row in block_records),
        "valid_abba_blocks": len(block_records) if actual_order == expected_order and all(row["recorded"] == 32 for row in block_records) else 0,
        "recorded_requests": len(samples), "successful_response_rate": sum(row["http_status"] == 200 for row in samples) / len(samples),
        "extracted_answer_parity_rate": sum(row["match"] for row in pairs) / len(pairs),
        "f16_accuracy": f16_accuracy, "q4_accuracy": q4_accuracy,
        "q4_accuracy_regression": f16_accuracy - q4_accuracy,
        "f16_median_tps": f16_tps, "q4_median_tps": q4_tps,
        "q4_vs_f16_throughput_ratio": q4_tps / f16_tps,
        "f16_vram_mib": f16_vram, "q4_vram_mib": q4_vram, "vram_saving_mib": f16_vram - q4_vram,
        "service_and_embedding_restored": restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200,
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "paired_metrics.json", pairs)
    write_json(raw / "dataset_hashes.json", {"panel_semantic_sha256": canonical_json_sha256(tasks), "task_ids": [row["task_id"] for row in tasks]})
    write_json(raw / "effective_route.json", {"blocks": block_records, "order": actual_order})
    write_json(raw / "environment.json", {"gpu": infra.gpu_state(), "wsl_distro": infra.WSL_DISTRO})
    write_json(raw / "hardware_metrics.json", {key: metrics[key] for key in ("f16_median_tps", "q4_median_tps", "q4_vs_f16_throughput_ratio", "f16_vram_mib", "q4_vram_mib", "vram_saving_mib")})
    write_json(raw / "independent_evaluation.json", {"executor_rescore": metrics, "independent_review_pending": True})
    write_json(raw / "paired_baseline.json", {"baseline": "f16", "treatment": "q4_0", "pairs": len(pairs)})
    write_json(raw / "service_identity.json", {"initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})
    write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True, "embedding_service_stopped": False, "restoration": restoration})
    write_json(raw / "treatment_controls.json", {"order": expected_order, "server_common": SERVER_COMMON, "cache_types": ["f16", "q4_0"], "warmups_per_block": 4})

    definitions = {
        "treatment_identity": ("explicit_cache_controls_verified", "eq", True),
        "balanced_crossover": ("valid_abba_blocks", "eq", 4),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "semantic_parity": ("extracted_answer_parity_rate", "ge", 0.9),
        "quality_non_regression": ("q4_accuracy_regression", "le", 0.03125),
        "physical_memory_saving": ("vram_saving_mib", "ge", 1000),
        "throughput_non_regression": ("q4_vs_f16_throughput_ratio", "ge", 0.95),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    gates = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold if operator == "ge" else actual <= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=[*frozen_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "physical_kv_cache_abba", "blocks": 4, "requests": len(samples)},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "binary_identity": "raw/binary_identity.json", "block_logs": "raw/logs",
        "dataset_hashes": "raw/dataset_hashes.json", "effective_route": "raw/effective_route.json",
        "environment": "raw/environment.json", "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
        "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json",
        "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    passed = all(gate["pass"] for gate in gates.values())
    claim = "QWEN38_Q4_KV_PHYSICALLY_QUALIFIED_R1" if passed else "QWEN38_Q4_KV_PHYSICALLY_REJECTED_R1"
    failed = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Q4_0 saved `{metrics['vram_saving_mib']:.1f}` MiB, reached `{metrics['q4_vs_f16_throughput_ratio']:.4f}x` F16 throughput, "
        f"and had answer parity `{metrics['extracted_answer_parity_rate']:.4f}`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    state.update({"status": "completed", "claim": claim, "failed_gates": failed})
    write_json(state_path, state)
    return receipt


def selfcheck() -> None:
    assert [row["arm"] for row in BLOCKS] == ["f16", "q4", "q4", "f16"]
    assert [row["pair"] for row in BLOCKS] == [0, 0, 1, 1]
    print("Qwen3.8 KV precision self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = infra.run_text([sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex executor"])
    print(json.dumps({"pipeline_advance": advance}, indent=2), flush=True)
    return 0 if advance["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
