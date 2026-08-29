#!/usr/bin/env python3
"""Two-slot concurrent Qwen3.8 F16/Q8 long-context crossover."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import sys
import threading
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file  # noqa: E402
from tools.research import run_qwen38_q8_kv_longcontext as r1  # noqa: E402

TASK_ID = "BACKLOG-QWEN38-Q8-KV-CONCURRENCY-01"
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-CONCURRENCY-01.json": "8f9e94dbb36284f745571b132a7ed500ddfaaca3ff8bb14327f4de52b1f84a6c",
    "runs/research/BACKLOG-QWEN38-Q8-KV-CONCURRENCY-01/PRE_REGISTRATION.md": "9eaec53e511131548aa41dc56626dee47fdb8d9e23d06be3d549116e360afdb5",
    "runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02/raw/receipt.json": "b5f2c691d270754857290fd43aa74c248698b3ad767026bd6dcc116aad0c8575",
    "runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-02/REVIEW.json": "3f759898e5e58e1f5a7d305724be869974a4d75ca71e92274084c703df08a15a",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "tools/research/run_qwen38_q8_kv_longcontext.py": "4197afacf8c2bd9c4f3a96802612afcd41a4ca47ed4f1190284d170cf13419c8",
    "tools/research/run_qwen38_q8_kv_longcontext_r2.py": "4c5facc502b25aa1de4465328b18473be6ae74147f932b6e09d1cbfb6487e89a",
}
SERVER_COMMON = [
    r1.infra.BINARY, "-m", r1.infra.MODEL, "--alias", "qwen38-q8-concurrency",
    "--host", "127.0.0.1", "--port", str(r1.infra.PORT), "--ctx-size", "65536",
    "--flash-attn", "on", "--gpu-layers", "all", "--metrics", "--jinja", "--no-mmproj",
    "--parallel", "2", "--batch-size", "2048", "--ubatch-size", "512", "--ctx-checkpoints", "32",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
]


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
    for path, expected in r1.WSL_INPUTS.items():
        size = int(r1.infra.checked(r1.infra.wsl("stat", "-L", "-c", "%s", path, timeout=120), f"stat {path}")["stdout"])
        digest = r1.infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"WSL identity mismatch: {path}: {size} {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return ledger, frozen


def start_block(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    unit = f"local-labs-q8-concurrency-{block['id']}.service"
    if r1.infra.unit_state(unit)["load_state"] != "not-found":
        raise RuntimeError(f"reserved unit exists: {unit}")
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--uid=augus", "--property=Type=simple", "--property=Restart=no",
        f"--setenv=LD_LIBRARY_PATH={r1.infra.LIB_DIR}", *SERVER_COMMON,
        "--cache-type-k", block["cache"], "--cache-type-v", block["cache"],
    ]
    launch = r1.infra.checked(r1.infra.wsl(*argv, root=True, timeout=60), f"launch {unit}")
    state = r1.infra.wait_unit(unit, active=True)
    process = r1.infra.process_values(state["main_pid"])
    r1.infra.wait_health(r1.infra.BASE_URL, timeout_seconds=600)
    cache_tokens = [process["argv"][index + 1] for index, token in enumerate(process["argv"]) if token in {"--cache-type-k", "--cache-type-v"}]
    parallel = [process["argv"][index + 1] for index, token in enumerate(process["argv"]) if token == "--parallel"]
    contexts = [process["argv"][index + 1] for index, token in enumerate(process["argv"]) if token == "--ctx-size"]
    if process["executable"] != r1.infra.BINARY or cache_tokens != [block["cache"], block["cache"]] or parallel != ["2"] or contexts != ["65536"]:
        raise RuntimeError(f"runtime controls mismatch: {process['executable']} {cache_tokens} {parallel} {contexts}")
    return unit, {"launch": launch, "state": state, "process": process, "cache_tokens": cache_tokens, "parallel": parallel, "ctx_size": contexts}


def chat_request(prompt: str, slot: int, barrier: threading.Barrier, max_tokens: int = 32) -> dict[str, Any]:
    barrier.wait(timeout=30)
    started = time.perf_counter()
    status, body = r1.infra.http_json(
        f"{r1.infra.BASE_URL}/v1/chat/completions",
        {
            "model": "qwen38-q8-concurrency", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_k": 1, "seed": r1.SEED,
            "cache_prompt": False, "id_slot": slot, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=1800.0,
    )
    ended = time.perf_counter()
    try:
        choice = body["choices"][0]
        content = str(choice["message"].get("content") or "")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        content, finish_reason = "", None
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    predicted_n = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    return {
        "requested_slot": slot, "client_started": started, "client_ended": ended,
        "http_status": status, "error": body.get("_error"), "wall_ms": (ended - started) * 1000.0,
        "content": content, "finish_reason": finish_reason, "timings": timings, "usage": usage,
        "prompt_n": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
        "predicted_n": predicted_n, "predicted_ms": predicted_ms,
        "throughput_tps": predicted_n * 1000.0 / predicted_ms if predicted_ms > 0 else None,
        "response": body,
    }


def concurrent_batch(cases: list[dict[str, Any]], batch_id: str, max_tokens: int = 32) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(cases) != 2:
        raise ValueError("concurrent batch must contain two cases")
    barrier = threading.Barrier(3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(chat_request, case["prompt"], slot, barrier, max_tokens) for slot, case in enumerate(cases)]
        barrier.wait(timeout=30)
        results = [future.result(timeout=1800) for future in futures]
    union_start = min(row["client_started"] for row in results)
    union_end = max(row["client_ended"] for row in results)
    overlap = min(row["client_ended"] for row in results) - max(row["client_started"] for row in results)
    record = {
        "batch_id": batch_id, "overlap": overlap > 0, "overlap_ms": overlap * 1000.0,
        "union_wall_ms": (union_end - union_start) * 1000.0,
        "aggregate_output_rate": sum(row["predicted_n"] for row in results) / (union_end - union_start),
        "slots_requested": [row["requested_slot"] for row in results],
    }
    return results, record


def gate_pass(operator: str, actual: Any, threshold: Any) -> bool:
    return actual == threshold if operator == "eq" else actual >= threshold if operator == "ge" else False


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw, finalized, logs = outdir / "raw", outdir / "raw/finalized", outdir / "raw/logs"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized.mkdir(parents=True)
    logs.mkdir(parents=True)
    samples_path, blocks_path, batches_path = raw / "samples.jsonl", raw / "blocks.jsonl", raw / "batches.jsonl"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    identities, frozen_paths = verify_inputs()
    r1.write_json(raw / "binary_identity.json", {"verified": True, "identities": identities})
    manifest = [case | {"prompt": None} for block in (r1.BLOCKS[0], r1.BLOCKS[2]) for case in r1.cases_for(block)]
    r1.write_json(raw / "case_manifest.json", {"generator": "associative_decoy_archive_q8_v1", "near_label_decoys": 31, "cases": manifest})

    initial_service = r1.infra.unit_state(r1.infra.PERSISTENT_UNIT)
    initial_gateway = r1.infra.gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = r1.infra.health(r1.infra.EMBED_URL)
    if initial_service["active_state"] != "active" or embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError("persistent gateway or embedding unhealthy before experiment")
    restoration: dict[str, Any] = {}
    execution_error: str | None = None
    try:
        r1.infra.systemctl("stop", r1.infra.PERSISTENT_UNIT)
        r1.infra.wait_unit(r1.infra.PERSISTENT_UNIT, active=False)
        occupied, body = r1.infra.health(r1.infra.BASE_URL)
        if occupied is not None:
            raise RuntimeError(f"temporary endpoint occupied: {occupied} {body}")
        for block in r1.BLOCKS:
            unit = ""
            try:
                unit, launch = start_block(block)
                warm_cases = [{"prompt": f"Reply only with READY-{slot}."} for slot in range(2)]
                warmups, warm_batch = concurrent_batch(warm_cases, f"{block['id']}-warmup", 16)
                if any(row["http_status"] != 200 for row in warmups) or not warm_batch["overlap"]:
                    raise RuntimeError(f"warmup concurrency failure in {block['id']}")
                gpu_before = r1.infra.gpu_state()
                cases = r1.cases_for(block)
                correct = 0
                failed_batches = 0
                for batch_index in range(6):
                    batch_cases = cases[batch_index * 2:batch_index * 2 + 2]
                    responses, batch = concurrent_batch(batch_cases, f"{block['id']}-batch-{batch_index:02d}")
                    batch.update({"block_id": block["id"], "arm": block["arm"], "cache": block["cache"], "pair": block["pair"], "batch_index": batch_index})
                    r1.append_jsonl(batches_path, batch)
                    if not batch["overlap"]:
                        raise RuntimeError(f"missing physical overlap in {batch['batch_id']}")
                    batch_failed = False
                    for slot, (case, response) in enumerate(zip(batch_cases, responses, strict=True)):
                        exact = r1.normalize(response["content"]) == case["code"]
                        low, high = r1.TARGETS[case["target_tokens"]]["band"]
                        row = {
                            "block_id": block["id"], "arm": block["arm"], "cache": block["cache"], "pair": block["pair"],
                            "batch_id": batch["batch_id"], "batch_index": batch_index, "slot": slot, **case,
                            "normalized_output": r1.normalize(response["content"]), "exact_recall": exact,
                            "within_target_token_band": low <= response["prompt_n"] <= high, **response,
                        }
                        r1.append_jsonl(samples_path, row)
                        correct += int(exact)
                        batch_failed = batch_failed or response["http_status"] != 200 or bool(response["error"])
                    failed_batches = failed_batches + 1 if batch_failed else 0
                    print(f"{block['id']} batch {batch_index + 1:02d}/06 overlap_ms={batch['overlap_ms']:.1f} exact={sum(r1.normalize(response['content']) == case['code'] for case, response in zip(batch_cases, responses, strict=True))}/2", flush=True)
                    if failed_batches >= 3:
                        raise RuntimeError(f"three consecutive failed batches in {block['id']}")
                record = {**block, "block_id": block["id"], "complete": True, "launch": launch, "warmup_batch": warm_batch, "gpu_before": gpu_before, "gpu_after": r1.infra.gpu_state(), "recorded": 12, "correct": correct}
                r1.append_jsonl(blocks_path, record)
                r1.write_json(finalized / f"{block['id']}.json", {"block_id": block["id"], "recorded": 12, "correct": correct})
            finally:
                if unit:
                    journal = r1.infra.wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True, timeout=180)["stdout"]
                    (logs / f"{unit}.log").write_text(journal, encoding="utf-8", newline="\n")
                    r1.infra.wsl("systemctl", "stop", unit, root=True, timeout=180)
                    try:
                        r1.infra.wait_unit(unit, active=False, timeout_seconds=180)
                    except RuntimeError:
                        pass
            if r1.infra.health(r1.infra.EMBED_URL)[0] != 200:
                raise RuntimeError(f"embedding unhealthy after {block['id']}")
    except Exception as error:
        execution_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if r1.infra.unit_state(r1.infra.PERSISTENT_UNIT)["active_state"] != "active":
                r1.infra.systemctl("start", r1.infra.PERSISTENT_UNIT)
            r1.infra.wait_health(r1.infra.GATEWAY_URL, timeout_seconds=600)
            restored = r1.infra.restore_model(initial_model)
            final_service = r1.infra.unit_state(r1.infra.PERSISTENT_UNIT)
            final_embed, final_embed_body = r1.infra.health(r1.infra.EMBED_URL)
            restoration = {"gateway": restored, "service": final_service, "embedding": {"http_status": final_embed, "body": final_embed_body}, "initial_model_restored": restored.get("current_model") == initial_model}
        except Exception as error:
            restoration = {"error": f"{type(error).__name__}: {error}", "initial_model_restored": False}
        r1.write_json(raw / "recovery_state.json", restoration)
        r1.write_json(raw / "runner_state.json", {"task_id": TASK_ID, "status": "aborted" if execution_error else "blocks_complete", "error": execution_error, "initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})

    samples, blocks, batches = r1.read_jsonl(samples_path), r1.read_jsonl(blocks_path), r1.read_jsonl(batches_path)
    paired = []
    for pair in range(2):
        f16 = {row["case_id"]: row for row in samples if row["pair"] == pair and row["arm"] == "f16"}
        q8 = {row["case_id"]: row for row in samples if row["pair"] == pair and row["arm"] == "q8"}
        for case_id in sorted(set(f16) & set(q8)):
            paired.append({"pair": pair, "case_id": case_id, "f16_correct": f16[case_id]["exact_recall"], "q8_correct": q8[case_id]["exact_recall"], "f16_output": f16[case_id]["normalized_output"], "q8_output": q8[case_id]["normalized_output"], "exact_output_match": f16[case_id]["normalized_output"] == q8[case_id]["normalized_output"]})
    comparison = r1.paired_bootstrap(paired)
    f16_rows, q8_rows = [row for row in samples if row["arm"] == "f16"], [row for row in samples if row["arm"] == "q8"]
    f16_batches, q8_batches = [row for row in batches if row["arm"] == "f16"], [row for row in batches if row["arm"] == "q8"]
    f16_rate = statistics.median(row["aggregate_output_rate"] for row in f16_batches)
    q8_rate = statistics.median(row["aggregate_output_rate"] for row in q8_batches)
    f16_vram = statistics.median(float(row["gpu_before"]["memory.used"]) for row in blocks if row["arm"] == "f16")
    q8_vram = statistics.median(float(row["gpu_before"]["memory.used"]) for row in blocks if row["arm"] == "q8")
    restored_ok = restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200 and restoration.get("service", {}).get("active_state") == "active"
    metrics = {
        "binary_and_model_identity_verified": True,
        "explicit_cache_controls_verified": all(row["launch"]["cache_tokens"] == [row["cache"], row["cache"]] for row in blocks),
        "explicit_two_slot_controls_verified": all(row["launch"]["parallel"] == ["2"] and row["launch"]["ctx_size"] == ["65536"] for row in blocks),
        "valid_crossover_blocks": len(blocks) if [row["arm"] for row in blocks] == ["f16", "q8", "q8", "f16"] and all(row["recorded"] == 12 for row in blocks) else 0,
        "recorded_requests": len(samples), "recorded_concurrent_batches": len(batches),
        "overlapping_batch_rate": sum(row["overlap"] for row in batches) / len(batches),
        "requests_within_target_token_bands": sum(row["within_target_token_band"] for row in samples),
        "successful_response_rate": sum(row["http_status"] == 200 and not row["error"] for row in samples) / len(samples),
        "nonempty_content_rate": sum(bool(r1.normalize(row["content"])) for row in samples) / len(samples),
        "nondegenerate_timing_rate": sum(int(row["predicted_n"]) >= 2 and float(row["predicted_ms"]) >= 1 and 1 <= float(row["throughput_tps"]) <= 1000 for row in samples) / len(samples),
        "f16_exact_recall": sum(row["exact_recall"] for row in f16_rows) / len(f16_rows), "q8_exact_recall": sum(row["exact_recall"] for row in q8_rows) / len(q8_rows),
        "paired_q8_minus_f16": comparison, "paired_bootstrap_ci95_low_q8_minus_f16": comparison["lower_95"],
        "f16_median_batch_output_rate": f16_rate, "q8_median_batch_output_rate": q8_rate, "q8_vs_f16_median_batch_rate_ratio": q8_rate / f16_rate,
        "median_f16_vram_mib": f16_vram, "median_q8_vram_mib": q8_vram, "median_vram_saving_mib": f16_vram - q8_vram,
        "service_gateway_embedding_restored": restored_ok,
    }
    r1.write_json(raw / "actual_scores.json", metrics)
    r1.write_json(raw / "paired_metrics.json", paired)
    r1.write_json(raw / "dependency_hashes.json", {"host": HOST_INPUTS, "wsl": r1.WSL_INPUTS})
    r1.write_json(raw / "effective_route.json", {"blocks": blocks, "batches": batches, "cache_argument_scope": "both K and V", "concurrency": 2})
    r1.write_json(raw / "environment.json", {"gpu_after": r1.infra.gpu_state(), "wsl_distro": r1.infra.WSL_DISTRO})
    r1.write_json(raw / "hardware_metrics.json", {key: metrics[key] for key in ("f16_median_batch_output_rate", "q8_median_batch_output_rate", "q8_vs_f16_median_batch_rate_ratio", "median_f16_vram_mib", "median_q8_vram_mib", "median_vram_saving_mib")})
    r1.write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "two-slot overlapping Qwen3.8 chat retrieval at 8k and 16k per slot only"})
    r1.write_json(raw / "paired_baseline.json", {"baseline": "f16", "treatment": "q8_0", "paired_cases": len(paired), "comparison": comparison})
    r1.write_json(raw / "service_identity.json", {"initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})
    r1.write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True, "embedding_service_stopped": False, "restoration": restoration})
    r1.write_json(raw / "treatment_controls.json", {"order": ["f16", "q8", "q8", "f16"], "parallel": 2, "ctx_size": 65536, "explicit_slots": [0, 1], "server_common": SERVER_COMMON, "request_contract": "/v1/chat/completions", "thinking": False, "bootstrap_seed": r1.BOOTSTRAP_SEED, "bootstrap_replicates": r1.BOOTSTRAP_REPLICATES})
    exact_paths = [samples_path, blocks_path, batches_path, raw / "paired_metrics.json", raw / "case_manifest.json"]
    r1.write_json(raw / "end_to_end_artifact.json", {"exact_files": {str(path.relative_to(raw)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in exact_paths}, "hash_semantics": "raw file bytes"})
    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", "eq", True), "cache_treatment_identity": ("explicit_cache_controls_verified", "eq", True),
        "concurrency_identity": ("explicit_two_slot_controls_verified", "eq", True), "balanced_crossover": ("valid_crossover_blocks", "eq", 4),
        "sample_size": ("recorded_requests", "eq", 48), "concurrent_batches": ("recorded_concurrent_batches", "eq", 24), "physical_overlap": ("overlapping_batch_rate", "eq", 1.0),
        "physical_context": ("requests_within_target_token_bands", "eq", 48), "request_integrity": ("successful_response_rate", "eq", 1.0),
        "chat_content": ("nonempty_content_rate", "eq", 1.0), "timing_integrity": ("nondegenerate_timing_rate", "eq", 1.0),
        "f16_retrieval": ("f16_exact_recall", "ge", 0.95), "q8_retrieval": ("q8_exact_recall", "ge", 0.95),
        "paired_noninferiority": ("paired_bootstrap_ci95_low_q8_minus_f16", "ge", -0.05), "batch_throughput_nonregression": ("q8_vs_f16_median_batch_rate_ratio", "ge", 0.9),
        "memory_saving": ("median_vram_saving_mib", "ge", 1000.0), "service_recovery": ("service_gateway_embedding_restored", "eq", True),
    }
    gates = {name: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": gate_pass(operator, metrics[metric], threshold)} for name, (metric, operator, threshold) in definitions.items()}
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*frozen_paths, *evidence_files], packages=[], runtime={"execution_mode": "qwen38_q8_two_slot_longcontext", "blocks": len(blocks), "requests": len(samples), "batches": len(batches), "parallel": 2, "ctx_size": 65536, "model": r1.infra.MODEL})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {"acceptance_gates":"raw/receipt.json","binary_identity":"raw/binary_identity.json","dependency_hashes":"raw/dependency_hashes.json","effective_route":"raw/effective_route.json","end_to_end_artifact":"raw/end_to_end_artifact.json","environment":"raw/environment.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","service_identity":"raw/service_identity.json","service_maintenance":"raw/service_maintenance.json","treatment_controls":"raw/treatment_controls.json"}
    receipt = {"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":True,"gates":gates,"evidence":evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r1.write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "QWEN38_Q8_KV_CONCURRENT_LONGCONTEXT_NONINFERIOR_R1" if not failed else "QWEN38_Q8_KV_CONCURRENT_LONGCONTEXT_NOT_NONINFERIOR_R1"
    (outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nF16/Q8 recall `{metrics['f16_exact_recall']:.4f}`/`{metrics['q8_exact_recall']:.4f}`; overlap `{metrics['overlapping_batch_rate']:.4f}` over `{len(batches)}` batches; Q8/F16 median batch rate `{metrics['q8_vs_f16_median_batch_rate_ratio']:.4f}`; VRAM saving `{metrics['median_vram_saving_mib']:.1f}` MiB; paired CI lower `{comparison['lower_95']:.4f}`; service restored `{restored_ok}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No single-slot scaling claim.\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
