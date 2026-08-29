#!/usr/bin/env python3
"""Observe the instrumented SLX-03 GDN fusion route with an explicit OFF control."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research import run_mtp_persistence_first_instance as infra

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-RUNTIME-01"
BINARY = "/home/augus/src/slop.cpp-main/build-slx03-gdn-instrumented-01/bin/llama-server"
LIB_DIR = "/home/augus/src/slop.cpp-main/build-slx03-gdn-instrumented-01/bin"
CUDA_LIB = f"{LIB_DIR}/libggml-cuda.so"
MODEL = "/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MARKER = "fused gated_delta_net snapshot copies"
SEED = 2026082816
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-RUNTIME-01.json": "d681a7c6b54c9eedfd0d3f226a53fc6640b4532bf8cfce6003d5eb56e0dc3926",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-RUNTIME-01/PRE_REGISTRATION.md": "eaebd52fba1dff1c091415271664045b62a3b91b288f8acd02f4702a65b47246",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/raw/receipt.json": "1d8fbaeac548e83e5df3360338e3e6dedb4143fc55ee72f9902b645ee784b80b",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/REVIEW.json": "008b1321b2b01f9ddd05f5807e4e7edfec308ca68c0dcae32b0178d6f64680bf",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}
WSL_INPUTS = {
    BINARY: {"bytes": 17920, "sha256": "c00261d903f722214511f0f6b999de77ff98dedbf9b8da292501b7743bbaecac"},
    CUDA_LIB: {"bytes": 63364248, "sha256": "e166987226156e67b4a3180f23dba86b65448391ee666b1a800177752b7a9614"},
    MODEL: {"bytes": 17923394624, "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"},
}
BLOCKS = (
    {"id": "b1_off", "arm": "off", "disable": "1", "pair": 0},
    {"id": "b2_on", "arm": "on", "disable": "0", "pair": 0},
    {"id": "b3_on", "arm": "on", "disable": "0", "pair": 1},
    {"id": "b4_off", "arm": "off", "disable": "1", "pair": 1},
)
PROMPTS = (
    "Reply with only the result: 17 + 28 =",
    "Reply with only the result: 9 times 13 =",
    "Reply with only the result: 144 divided by 12 =",
    "Reply with only the result: 81 minus 37 =",
    "Responda apenas com o resultado: 23 + 19 =",
    "Responda apenas com o resultado: 7 vezes 16 =",
    "Reply with only the result: the next integer after 399 is",
    "Reply with only the result: half of 86 is",
)
SERVER_ARGS = [
    BINARY, "-m", MODEL, "--alias", "qwen38-slx03-runtime", "--host", "127.0.0.1", "--port", str(infra.PORT),
    "--ctx-size", "32768", "--flash-attn", "on", "--gpu-layers", "all", "--metrics", "--jinja", "--no-mmproj",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1", "--batch-size", "2048", "--ubatch-size", "512",
    "--ctx-checkpoints", "32", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
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
    frozen_paths: list[pathlib.Path] = []
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"host identity mismatch: {relative}: {actual}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        frozen_paths.append(path)
    for path, expected in WSL_INPUTS.items():
        size_result = infra.checked(infra.wsl("stat", "-L", "-c", "%s", path, timeout=120), f"stat {path}")
        size = int(size_result["stdout"])
        digest = infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"WSL identity mismatch: {path}: {size} {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return ledger, frozen_paths


def process_environment(pid: int) -> dict[str, str]:
    result = infra.checked(infra.wsl("xargs", "-0", "-n", "1", "-a", f"/proc/{pid}/environ"), "read process environment")
    values = {}
    for line in result["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def start_block(block: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    unit = f"local-labs-slx03-runtime-{block['id']}.service"
    if infra.unit_state(unit)["load_state"] != "not-found":
        raise RuntimeError(f"reserved unit exists: {unit}")
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--uid=augus", "--property=Type=simple", "--property=Restart=no",
        f"--setenv=LD_LIBRARY_PATH={LIB_DIR}", f"--setenv=GGML_CUDA_DISABLE_FUSION={block['disable']}", *SERVER_ARGS,
    ]
    launch = infra.checked(infra.wsl(*argv, root=True, timeout=60), f"launch {unit}")
    state = infra.wait_unit(unit, active=True)
    process = infra.process_values(state["main_pid"])
    environment = process_environment(state["main_pid"])
    infra.wait_health(infra.BASE_URL, timeout_seconds=600)
    if process["executable"] != BINARY or environment.get("GGML_CUDA_DISABLE_FUSION") != block["disable"] or environment.get("LD_LIBRARY_PATH") != LIB_DIR:
        raise RuntimeError(f"treatment identity mismatch: {process} {environment.get('GGML_CUDA_DISABLE_FUSION')} {environment.get('LD_LIBRARY_PATH')}")
    return unit, {"launch": launch, "state": state, "process": process, "treatment_environment": {"GGML_CUDA_DISABLE_FUSION": environment.get("GGML_CUDA_DISABLE_FUSION"), "LD_LIBRARY_PATH": environment.get("LD_LIBRARY_PATH")}}


def completion(prompt: str, n_predict: int = 64) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = infra.http_json(f"{infra.BASE_URL}/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": SEED, "cache_prompt": False, "id_slot": 0, "stream": False})
    return {"http_status": status, "error": body.get("_error"), "wall_ms": round((time.perf_counter() - started) * 1000, 3), "content": str(body.get("content") or ""), "timings": body.get("timings") or {}, "response": body}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    logs = raw / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    samples_path = raw / "samples.jsonl"
    blocks_path = raw / "blocks.jsonl"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    identities, frozen_paths = verify_inputs()
    write_json(raw / "binary_identity.json", {"verified": True, "identities": identities})

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
                warmups = [completion(f"Discarded warmup {index}: reply only with {31 + index}.", 32) for index in range(2)]
                if any(row["http_status"] != 200 for row in warmups):
                    raise RuntimeError(f"warmup failure in {block['id']}")
                consecutive_errors = 0
                for index, prompt in enumerate(PROMPTS):
                    response = completion(prompt)
                    consecutive_errors = consecutive_errors + 1 if response["http_status"] != 200 else 0
                    sample = {"block_id": block["id"], "arm": block["arm"], "pair": block["pair"], "index": index, "prompt": prompt, **response}
                    append_jsonl(samples_path, sample)
                    print(f"{block['id']} {index + 1:02d}/08 http={response['http_status']}", flush=True)
                    if consecutive_errors >= 3:
                        raise RuntimeError(f"three consecutive failures in {block['id']}")
                journal = infra.wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "20000", root=True, timeout=180)["stdout"]
                (logs / f"{unit}.log").write_text(journal, encoding="utf-8", newline="\n")
                marker_count = journal.count(MARKER)
                record = {**block, "block_id": block["id"], "complete": True, "launch": launch, "warmups": warmups, "gpu": infra.gpu_state(), "recorded": len(PROMPTS), "marker": MARKER, "marker_count": marker_count}
                append_jsonl(blocks_path, record)
            finally:
                if unit:
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
    pairs = []
    for pair in (0, 1):
        off = {(row["index"], row["prompt"]): row for row in samples if row["pair"] == pair and row["arm"] == "off"}
        on = {(row["index"], row["prompt"]): row for row in samples if row["pair"] == pair and row["arm"] == "on"}
        for key in sorted(set(off) & set(on)):
            pairs.append({"pair": pair, "index": key[0], "prompt": key[1], "off_content": off[key]["content"], "on_content": on[key]["content"], "exact_match": off[key]["content"] == on[key]["content"]})
    treatment_ok = all(row.get("launch", {}).get("treatment_environment", {}).get("GGML_CUDA_DISABLE_FUSION") == row["disable"] for row in blocks)
    order = [row["arm"] for row in blocks]
    metrics = {
        "binary_and_model_identity_verified": True,
        "explicit_fusion_controls_verified": treatment_ok,
        "valid_abba_blocks": len(blocks) if order == ["off", "on", "on", "off"] and all(row["recorded"] == len(PROMPTS) for row in blocks) else 0,
        "recorded_requests": len(samples),
        "successful_response_rate": sum(row["http_status"] == 200 for row in samples) / len(samples),
        "on_blocks_with_marker": sum(row["arm"] == "on" and row["marker_count"] > 0 for row in blocks),
        "off_blocks_without_marker": sum(row["arm"] == "off" and row["marker_count"] == 0 for row in blocks),
        "exact_output_parity_rate": sum(row["exact_match"] for row in pairs) / len(pairs),
        "service_gateway_embedding_restored": restoration.get("initial_model_restored") is True and restoration.get("embedding", {}).get("http_status") == 200 and restoration.get("service", {}).get("active_state") == "active",
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "paired_metrics.json", pairs)
    write_json(raw / "dependency_hashes.json", {"host": HOST_INPUTS, "wsl": WSL_INPUTS})
    write_json(raw / "effective_route.json", {"blocks": blocks, "order": order, "marker": MARKER})
    write_json(raw / "environment.json", {"gpu_after": infra.gpu_state(), "wsl_distro": infra.WSL_DISTRO})
    write_json(raw / "hardware_metrics.json", {"per_block_gpu": [{"block_id": row["block_id"], "gpu": row["gpu"], "marker_count": row["marker_count"]} for row in blocks], "performance_claimed": False})
    write_json(raw / "independent_evaluation.json", {"executor_metrics": metrics, "independent_review_pending": True, "claim_boundary": "runtime route and exact parity only"})
    write_json(raw / "paired_baseline.json", {"baseline": "GGML_CUDA_DISABLE_FUSION=1", "treatment": "GGML_CUDA_DISABLE_FUSION=0", "pairs": len(pairs)})
    write_json(raw / "service_identity.json", {"initial_service": initial_service, "initial_gateway": initial_gateway, "restoration": restoration})
    write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True, "embedding_service_stopped": False, "restoration": restoration})
    write_json(raw / "treatment_controls.json", {"order": ["off", "on", "on", "off"], "server_args": SERVER_ARGS, "environment_variable": "GGML_CUDA_DISABLE_FUSION", "seed": SEED, "prompts": list(PROMPTS)})
    write_json(raw / "end_to_end_artifact.json", {"samples_semantic_sha256": canonical_json_sha256(samples), "paired_semantic_sha256": canonical_json_sha256(pairs), "block_semantic_sha256": canonical_json_sha256(blocks)})

    definitions = {
        "binary_model_identity": ("binary_and_model_identity_verified", "eq", True), "treatment_identity": ("explicit_fusion_controls_verified", "eq", True),
        "balanced_crossover": ("valid_abba_blocks", "eq", 4), "request_integrity": ("successful_response_rate", "eq", 1.0),
        "runtime_route": ("on_blocks_with_marker", "eq", 2), "negative_control": ("off_blocks_without_marker", "eq", 2),
        "semantic_parity": ("exact_output_parity_rate", "eq", 1.0), "service_recovery": ("service_gateway_embedding_restored", "eq", True),
    }
    gates = {}
    for gate, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        gates[gate] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": actual == threshold}
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*frozen_paths, *evidence_files], packages=[], runtime={"execution_mode": "instrumented_runtime_abba", "blocks": 4, "requests": len(samples), "model": MODEL})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "binary_identity": "raw/binary_identity.json", "block_logs": "raw/logs", "dependency_hashes": "raw/dependency_hashes.json",
        "effective_route": "raw/effective_route.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "environment": "raw/environment.json", "hardware_metrics": "raw/hardware_metrics.json",
        "independent_evaluation": "raw/independent_evaluation.json", "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json", "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json", "treatment_controls": "raw/treatment_controls.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "SLX03_GDN_FUSION_RUNTIME_ROUTE_CONFIRMED_R1" if not failed else "SLX03_GDN_FUSION_RUNTIME_ROUTE_NOT_CONFIRMED_R1"
    (outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nON blocks with marker `{metrics['on_blocks_with_marker']}/2`; OFF blocks without marker `{metrics['off_blocks_without_marker']}/2`; exact paired output parity `{metrics['exact_output_parity_rate']:.4f}` over `{len(pairs)}` pairs; successful responses `{metrics['successful_response_rate']:.4f}`; service restored `{metrics['service_gateway_embedding_restored']}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No performance, write-reduction or deployment claim is made.\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
