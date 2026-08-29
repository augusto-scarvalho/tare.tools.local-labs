#!/usr/bin/env python3
"""Fresh paired Qwen3.8 F16/Q8 KV long-context retrieval experiment."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import statistics
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
from tools.research import run_mtp_persistence_first_instance as infra  # noqa: E402

TASK_ID = "BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01"
CUDA_LIB = f"{infra.LIB_DIR}/libggml-cuda.so"
SEED = 2026082822
BOOTSTRAP_SEED = 2026082823
BOOTSTRAP_REPLICATES = 20_000
TARGETS = {8000: {"filler": 320, "band": (7800, 8200)}, 16000: {"filler": 684, "band": (15800, 16200)}}
POSITIONS = ("start", "middle", "end")
BLOCKS = (
    {"id": "b1_f16", "arm": "f16", "cache": "f16", "pair": 0, "replicates": (0, 1)},
    {"id": "b2_q8", "arm": "q8", "cache": "q8_0", "pair": 0, "replicates": (0, 1)},
    {"id": "b3_q8", "arm": "q8", "cache": "q8_0", "pair": 1, "replicates": (2, 3)},
    {"id": "b4_f16", "arm": "f16", "cache": "f16", "pair": 1, "replicates": (2, 3)},
)
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01.json": "3062b916148f9a75c069b78892ba79f8f476c79e06c8b1c6c8e5f3f3deb81d5a",
    "runs/research/BACKLOG-QWEN38-Q8-KV-LONGCONTEXT-01/PRE_REGISTRATION.md": "10ad52207f939821d497e8db1c31ed68a163d810f79292a67e0fb9331f0a4c6a",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/raw/receipt.json": "f94153b21ab3196000b321d06fb79b0b59c3862146de519f05d69cb47d2fa9fe",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/REVIEW.json": "d3d230ca8fe27b198ce2170d54f1c95feae6cb53b22d15e14ccce98e727d3e54",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}
WSL_INPUTS = {
    infra.BINARY: {"bytes": 17920, "sha256": "efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2"},
    CUDA_LIB: {"bytes": 75928784, "sha256": "ca18591576b67632bfb09eaee56b958ff951a2a0c558a36ea2232f15032a8c6b"},
    infra.MODEL: {"bytes": 17923394624, "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"},
}
SERVER_COMMON = [
    infra.BINARY, "-m", infra.MODEL, "--alias", "qwen38-q8-longcontext",
    "--host", "127.0.0.1", "--port", str(infra.PORT), "--ctx-size", "32768",
    "--flash-attn", "on", "--gpu-layers", "all", "--metrics", "--jinja", "--no-mmproj",
    "--parallel", "1", "--batch-size", "2048", "--ubatch-size", "512", "--ctx-checkpoints", "32",
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
    frozen: list[pathlib.Path] = []
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"host identity mismatch: {relative}: {actual}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        frozen.append(path)
    for path, expected in WSL_INPUTS.items():
        size = int(infra.checked(infra.wsl("stat", "-L", "-c", "%s", path, timeout=120), f"stat {path}")["stdout"])
        digest = infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"WSL identity mismatch: {path}: {size} {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return ledger, frozen


def filler_line(index: int) -> str:
    return f"Archive record {index:06d}: amber catalog ordinary transit ledger contains no requested access credential.\n"


def code_for(target: int, position: str, replicate: int) -> str:
    return f"NX-Q8-{target:05d}-{position[0].upper()}{replicate}"


def make_prompt(target: int, position: str, replicate: int) -> tuple[str, str]:
    code = code_for(target, position, replicate)
    records = [filler_line(index) for index in range(TARGETS[target]["filler"])]
    prefix = "-".join(code.split("-")[:3])
    for index in range(1, 32):
        insertion = min(len(records), round(index * len(records) / 32))
        records.insert(insertion, f"SECURE ACCESS RECORD [ORION-DELTA-{index:02d}]: The access code is {prefix}-D{index:02d}.\n")
    needle = f"SECURE ACCESS RECORD [ORION-DELTA]: The access code is {code}.\n"
    insertion = 0 if position == "start" else len(records) // 2 if position == "middle" else len(records)
    records.insert(insertion, needle)
    prompt = (
        "Read the archive. Return the access code for the exact record label [ORION-DELTA]. "
        "Do not use numbered labels such as [ORION-DELTA-01]. Return only the code, with no explanation.\n\n"
        + "".join(records)
        + "\nQuestion: What is the access code for exact label [ORION-DELTA]? Return only the code."
    )
    return prompt, code


def cases_for(block: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGETS:
        for position in POSITIONS:
            for replicate in block["replicates"]:
                prompt, code = make_prompt(target, position, replicate)
                rows.append({
                    "case_id": f"ctx{target}-{position}-r{replicate}",
                    "target_tokens": target,
                    "position": position,
                    "replicate": replicate,
                    "code": code,
                    "prompt": prompt,
                    "prompt_chars": len(prompt),
                    "prompt_sha256": canonical_json_sha256(prompt),
                })
    return rows


def normalize(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    return value.strip("`\"' \t\r\n.,;:")


def start_block(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    unit = f"local-labs-q8-long-{block['id']}.service"
    if infra.unit_state(unit)["load_state"] != "not-found":
        raise RuntimeError(f"reserved unit exists: {unit}")
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--uid=augus",
        "--property=Type=simple", "--property=Restart=no", f"--setenv=LD_LIBRARY_PATH={infra.LIB_DIR}",
        *SERVER_COMMON, "--cache-type-k", block["cache"], "--cache-type-v", block["cache"],
    ]
    launch = infra.checked(infra.wsl(*argv, root=True, timeout=60), f"launch {unit}")
    state = infra.wait_unit(unit, active=True)
    process = infra.process_values(state["main_pid"])
    infra.wait_health(infra.BASE_URL, timeout_seconds=600)
    cache_tokens = [process["argv"][index + 1] for index, token in enumerate(process["argv"]) if token in {"--cache-type-k", "--cache-type-v"}]
    if process["executable"] != infra.BINARY or cache_tokens != [block["cache"], block["cache"]]:
        raise RuntimeError(f"cache treatment mismatch: {process['executable']} {cache_tokens}")
    return unit, {"launch": launch, "state": state, "process": process, "cache_tokens": cache_tokens}


def completion(prompt: str, n_predict: int = 32) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = infra.http_json(f"{infra.BASE_URL}/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "seed": SEED, "cache_prompt": False, "id_slot": 0, "stream": False,
    }, timeout=900.0)
    timings = body.get("timings") or {}
    predicted_n = int(timings.get("predicted_n") or body.get("tokens_predicted") or 0)
    predicted_ms = float(timings.get("predicted_ms") or 0.0)
    return {
        "http_status": status, "error": body.get("_error"),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "content": str(body.get("content") or ""), "timings": timings,
        "prompt_n": int(timings.get("prompt_n") or 0), "predicted_n": predicted_n,
        "throughput_tps": predicted_n * 1000.0 / predicted_ms if predicted_ms > 0 else None,
        "response": body,
    }


def paired_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    differences = [int(row["q8_correct"]) - int(row["f16_correct"]) for row in rows]
    generator = random.Random(BOOTSTRAP_SEED)
    estimates = [sum(differences[generator.randrange(len(differences))] for _ in differences) / len(differences) for _ in range(BOOTSTRAP_REPLICATES)] if differences else []
    estimates.sort()
    return {
        "point": statistics.mean(differences) if differences else 0.0,
        "lower_95": estimates[int(0.025 * BOOTSTRAP_REPLICATES)] if estimates else -1.0,
        "upper_95": estimates[int(0.975 * BOOTSTRAP_REPLICATES)] if estimates else 1.0,
        "replicates": BOOTSTRAP_REPLICATES if estimates else 0,
        "seed": BOOTSTRAP_SEED,
        "paired_cases": len(differences),
        "q8_only_correct": sum(value == 1 for value in differences),
        "f16_only_correct": sum(value == -1 for value in differences),
    }


def gate_pass(operator: str, actual: Any, threshold: Any) -> bool:
    return actual == threshold if operator == "eq" else actual >= threshold if operator == "ge" else False


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    finalized = raw / "finalized"
    logs = raw / "logs"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    finalized.mkdir(parents=True)
    logs.mkdir(parents=True)
    samples_path = raw / "samples.jsonl"
    blocks_path = raw / "blocks.jsonl"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    identities, frozen_paths = verify_inputs()
    write_json(raw / "binary_identity.json", {"verified": True, "identities": identities})
    manifest = [case | {"prompt": None} for pair in range(2) for case in cases_for(BLOCKS[pair * 2])]
    write_json(raw / "case_manifest.json", {"generator": "associative_decoy_archive_q8_v1", "near_label_decoys": 31, "cases": manifest})

    initial_service = infra.unit_state(infra.PERSISTENT_UNIT)
    initial_gateway = infra.gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = infra.health(infra.EMBED_URL)
    if initial_service["active_state"] != "active" or embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError("persistent gateway or embedding unhealthy before experiment")
    write_json(raw / "runner_state.json", {"task_id": TASK_ID, "status": "running", "initial_service": initial_service, "initial_gateway": initial_gateway, "initial_model": initial_model})
    restoration: dict[str, Any] = {}
    execution_error: str | None = None
    try:
        infra.systemctl("stop", infra.PERSISTENT_UNIT)
        infra.wait_unit(infra.PERSISTENT_UNIT, active=False)
        occupied, body = infra.health(infra.BASE_URL)
        if occupied is not None:
            raise RuntimeError(f"temporary endpoint remains occupied: {occupied} {body}")
        for block in BLOCKS:
            unit = ""
            try:
                unit, launch = start_block(block)
                warmups = [completion(f"Discarded warmup {index}: reply with the word ready.", 16) for index in range(2)]
                if any(row["http_status"] != 200 for row in warmups):
                    raise RuntimeError(f"warmup failure in {block['id']}")
                gpu_before = infra.gpu_state()
                consecutive = 0
                block_correct = 0
                for index, case in enumerate(cases_for(block)):
                    response = completion(case["prompt"])
                    correct = normalize(response["content"]) == case["code"]
                    low, high = TARGETS[case["target_tokens"]]["band"]
                    within_band = low <= response["prompt_n"] <= high
                    row = {
                        "block_id": block["id"], "arm": block["arm"], "cache": block["cache"], "pair": block["pair"],
                        "index": index, **case, "normalized_output": normalize(response["content"]),
                        "exact_recall": correct, "within_target_token_band": within_band, **response,
                    }
                    append_jsonl(samples_path, row)
                    block_correct += int(correct)
                    consecutive = consecutive + 1 if response["http_status"] != 200 or response["error"] else 0
                    print(f"{block['id']} {index + 1:02d}/12 http={response['http_status']} prompt_n={response['prompt_n']} exact={correct}", flush=True)
                    if consecutive >= 3:
                        raise RuntimeError(f"three consecutive failures in {block['id']}")
                record = {**block, "block_id": block["id"], "complete": True, "launch": launch, "gpu_before": gpu_before, "gpu_after": infra.gpu_state(), "recorded": 12, "correct": block_correct}
                append_jsonl(blocks_path, record)
                write_json(finalized / f"{block['id']}.json", {"block_id": block["id"], "recorded": 12, "correct": block_correct})
            finally:
                if unit:
                    journal = infra.wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True, timeout=180)["stdout"]
                    (logs / f"{unit}.log").write_text(journal, encoding="utf-8", newline="\n")
                    infra.wsl("systemctl", "stop", unit, root=True, timeout=180)
                    try:
                        infra.wait_unit(unit, active=False, timeout_seconds=180)
                    except RuntimeError:
                        pass
            boundary, _ = infra.health(infra.EMBED_URL)
            if boundary != 200:
                raise RuntimeError(f"embedding unhealthy after {block['id']}")
    except Exception as error:
        execution_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if infra.unit_state(infra.PERSISTENT_UNIT)["active_state"] != "active":
                infra.systemctl("start", infra.PERSISTENT_UNIT)
            infra.wait_health(infra.GATEWAY_URL, timeout_seconds=600)
            restored_gateway = infra.restore_model(initial_model)
            final_service = infra.unit_state(infra.PERSISTENT_UNIT)
            final_embed, final_embed_body = infra.health(infra.EMBED_URL)
            restoration = {"gateway": restored_gateway, "service": final_service, "embedding": {"http_status": final_embed, "body": final_embed_body}, "initial_model_restored": restored_gateway.get("current_model") == initial_model}
        except Exception as error:
            restoration = {"error": f"{type(error).__name__}: {error}", "initial_model_restored": False}
        write_json(raw / "recovery_state.json", restoration)
        write_json(raw / "runner_state.json", {"task_id": TASK_ID, "status": "aborted" if execution_error else "blocks_complete", "error": execution_error, "initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})

    samples = read_jsonl(samples_path)
    blocks = read_jsonl(blocks_path)
    paired: list[dict[str, Any]] = []
    for pair in range(2):
        f16 = {row["case_id"]: row for row in samples if row["pair"] == pair and row["arm"] == "f16"}
        q8 = {row["case_id"]: row for row in samples if row["pair"] == pair and row["arm"] == "q8"}
        for case_id in sorted(set(f16) & set(q8)):
            paired.append({
                "pair": pair, "case_id": case_id, "target_tokens": f16[case_id]["target_tokens"],
                "f16_correct": f16[case_id]["exact_recall"], "q8_correct": q8[case_id]["exact_recall"],
                "f16_output": f16[case_id]["normalized_output"], "q8_output": q8[case_id]["normalized_output"],
                "exact_output_match": f16[case_id]["normalized_output"] == q8[case_id]["normalized_output"],
            })
    comparison = paired_bootstrap(paired)
    f16_rows = [row for row in samples if row["arm"] == "f16"]
    q8_rows = [row for row in samples if row["arm"] == "q8"]
    f16_tps = statistics.median(float(row["throughput_tps"]) for row in f16_rows if row["throughput_tps"] is not None)
    q8_tps = statistics.median(float(row["throughput_tps"]) for row in q8_rows if row["throughput_tps"] is not None)
    f16_vram = statistics.median(float(row["gpu_before"]["memory.used"]) for row in blocks if row["arm"] == "f16")
    q8_vram = statistics.median(float(row["gpu_before"]["memory.used"]) for row in blocks if row["arm"] == "q8")
    expected_order = ["f16", "q8", "q8", "f16"]
    service_restored = restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200 and restoration.get("service", {}).get("active_state") == "active"
    metrics = {
        "binary_and_model_identity_verified": True,
        "explicit_cache_controls_verified": all(row["launch"]["cache_tokens"] == [row["cache"], row["cache"]] for row in blocks),
        "valid_crossover_blocks": len(blocks) if [row["arm"] for row in blocks] == expected_order and all(row["recorded"] == 12 for row in blocks) else 0,
        "recorded_requests": len(samples),
        "requests_within_target_token_bands": sum(row["within_target_token_band"] for row in samples),
        "successful_response_rate": sum(row["http_status"] == 200 and not row["error"] for row in samples) / len(samples),
        "f16_exact_recall": sum(row["exact_recall"] for row in f16_rows) / len(f16_rows),
        "q8_exact_recall": sum(row["exact_recall"] for row in q8_rows) / len(q8_rows),
        "paired_q8_minus_f16": comparison,
        "paired_bootstrap_ci95_low_q8_minus_f16": comparison["lower_95"],
        "f16_median_tps": f16_tps, "q8_median_tps": q8_tps,
        "q8_vs_f16_median_tps_ratio": q8_tps / f16_tps,
        "median_f16_vram_mib": f16_vram, "median_q8_vram_mib": q8_vram,
        "median_vram_saving_mib": f16_vram - q8_vram,
        "service_gateway_embedding_restored": service_restored,
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "paired_metrics.json", paired)
    write_json(raw / "dependency_hashes.json", {"host": HOST_INPUTS, "wsl": WSL_INPUTS})
    write_json(raw / "effective_route.json", {"blocks": blocks, "cache_argument_scope": "both K and V"})
    write_json(raw / "environment.json", {"gpu_after": infra.gpu_state(), "wsl_distro": infra.WSL_DISTRO})
    write_json(raw / "hardware_metrics.json", {key: metrics[key] for key in ("f16_median_tps", "q8_median_tps", "q8_vs_f16_median_tps_ratio", "median_f16_vram_mib", "median_q8_vram_mib", "median_vram_saving_mib")})
    write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "Qwen3.8 single-slot associative retrieval at 8k and 16k only"})
    write_json(raw / "paired_baseline.json", {"baseline": "f16", "treatment": "q8_0", "paired_cases": len(paired), "comparison": comparison})
    write_json(raw / "service_identity.json", {"initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})
    write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True, "embedding_service_stopped": False, "restoration": restoration})
    write_json(raw / "treatment_controls.json", {"order": expected_order, "blocks": list(BLOCKS), "targets": TARGETS, "positions": POSITIONS, "seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "server_common": SERVER_COMMON})
    exact_paths = [samples_path, blocks_path, raw / "paired_metrics.json", raw / "case_manifest.json"]
    write_json(raw / "end_to_end_artifact.json", {"exact_files": {str(path.relative_to(raw)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in exact_paths}, "hash_semantics": "raw file bytes"})

    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", "eq", True),
        "cache_treatment_identity": ("explicit_cache_controls_verified", "eq", True),
        "balanced_crossover": ("valid_crossover_blocks", "eq", 4),
        "sample_size": ("recorded_requests", "eq", 48),
        "physical_context": ("requests_within_target_token_bands", "eq", 48),
        "request_integrity": ("successful_response_rate", "eq", 1.0),
        "f16_retrieval": ("f16_exact_recall", "ge", 0.95),
        "q8_retrieval": ("q8_exact_recall", "ge", 0.95),
        "paired_noninferiority": ("paired_bootstrap_ci95_low_q8_minus_f16", "ge", -0.05),
        "throughput_nonregression": ("q8_vs_f16_median_tps_ratio", "ge", 0.9),
        "memory_saving": ("median_vram_saving_mib", "ge", 500.0),
        "service_recovery": ("service_gateway_embedding_restored", "eq", True),
    }
    gates = {name: {"metric": metric, "operator": operator, "threshold": threshold, "actual": metrics[metric], "pass": gate_pass(operator, metrics[metric], threshold)} for name, (metric, operator, threshold) in definitions.items()}
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*frozen_paths, *evidence_files], packages=[], runtime={"execution_mode": "qwen38_q8_longcontext_crossover", "blocks": len(blocks), "requests": len(samples), "targets": list(TARGETS), "model": infra.MODEL})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "binary_identity": "raw/binary_identity.json", "dependency_hashes": "raw/dependency_hashes.json",
        "effective_route": "raw/effective_route.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "environment": "raw/environment.json",
        "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json",
        "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json", "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "QWEN38_Q8_KV_LONGCONTEXT_NONINFERIOR_R1" if not failed else "QWEN38_Q8_KV_LONGCONTEXT_NOT_NONINFERIOR_R1"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"F16/Q8 recall `{metrics['f16_exact_recall']:.4f}`/`{metrics['q8_exact_recall']:.4f}`; paired Q8-minus-F16 `{comparison['point']:.4f}` with 95% CI `[{comparison['lower_95']:.4f}, {comparison['upper_95']:.4f}]`; "
        f"Q8 throughput ratio `{metrics['q8_vs_f16_median_tps_ratio']:.4f}`; median VRAM saving `{metrics['median_vram_saving_mib']:.1f}` MiB; physical target-band requests `{metrics['requests_within_target_token_bands']}/48`; service restored `{service_restored}`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert len(BLOCKS) == 4
        assert all(len(cases_for(block)) == 12 for block in BLOCKS)
        assert len({case["case_id"] for block in (BLOCKS[0], BLOCKS[2]) for case in cases_for(block)}) == 24
        return 0
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
