#!/usr/bin/env python3
"""Physical functional-reuse test for MTP slot save and restore."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import run_mtp_persistence_first_instance as infra


TASK_ID = "BACKLOG-MTP-PERSISTENCE-FUNCTIONAL-03"
CYCLES = 16
TRANSIENT_INDEX = 99
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-MTP-PERSISTENCE-FUNCTIONAL-03.json": "9a706365ebfbd9a46de47e99df643a824c12a0d4000f9283e2d25b53c77a5fa9",
    "runs/research/BACKLOG-MTP-PERSISTENCE-FUNCTIONAL-03/PRE_REGISTRATION.md": "f6effb18628b83973eb8b884e52b0929a00efa7485765b586e3501f8564977f4",
    "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/receipt.json": "6a52f8690b9f20c6583bdfa80e947ca4f7469e3297b24e046195a799c4116079",
    "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/samples.jsonl": "ae235a397007db32e6c5adc4228f26023fd0a7a5efa4a93cab8d7ba9e6b1b34d",
    "runs/research/BACKLOG-MTP-PERSISTENCE-RESCORE-02/raw/receipt.json": "b7ab4e3567f48be73c8be933316af1126fcde644f3e0adca8450a0672d611021",
    "tools/research/run_mtp_persistence_first_instance.py": "2a70897b1a9d73fb5fbf77159fa8c6706a41786c95bd428061e8963b8abde7b1",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def lifecycle_ok(*actions: dict[str, Any]) -> bool:
    return all(action["http_status"] == 200 and "_error" not in action["body"] for action in actions)


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    logs = raw / "logs"
    finalized = raw / "finalized"
    if any(raw.iterdir()):
        raise RuntimeError("raw directory is not empty")
    logs.mkdir(parents=True)
    finalized.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mono = time.monotonic()
    input_paths: list[pathlib.Path] = []
    host_ledger: dict[str, Any] = {}
    for relative, expected in HOST_INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen source mismatch: {relative}: {actual} != {expected}")
        input_paths.append(path)
        host_ledger[relative] = {"bytes": path.stat().st_size, "sha256": actual}
    wsl_ledger: dict[str, Any] = {}
    for path, expected in infra.EXPECTED_WSL.items():
        size = infra.stat_wsl(path)
        digest = infra.sha256_wsl(path)
        if size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"frozen WSL identity mismatch: {path}: {size} {digest}")
        wsl_ledger[path] = {"bytes": size, "sha256": digest}

    initial_service = infra.unit_state(infra.PERSISTENT_UNIT)
    initial_gateway = infra.gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = infra.health(infra.EMBED_URL)
    if initial_service["active_state"] != "active" or initial_model != "qwen38":
        raise RuntimeError(f"unexpected service baseline: {initial_service} {initial_gateway}")
    if embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError(f"embedding baseline unhealthy: {embed_status} {embed_body}")

    unit = ""
    save_dir = ""
    launch: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    restoration: dict[str, Any] = {}
    execution_error: Exception | None = None
    try:
        infra.systemctl("stop", infra.PERSISTENT_UNIT)
        infra.wait_unit(infra.PERSISTENT_UNIT, active=False)
        occupied_status, occupied_body = infra.health(infra.BASE_URL)
        if occupied_status is not None:
            raise RuntimeError(f"temporary endpoint occupied: {occupied_status} {occupied_body}")
        unit, save_dir, launch = infra.start_observation({"index": TRANSIENT_INDEX, "arm": "warm", "block": 0})
        for index in range(CYCLES):
            code = f"CEDAR{index:02d}"
            shared = (f"MTP-FUNCTIONAL-{index:02d}: the stable archive prefix remains unchanged. " * 420)
            prompt = shared + f"\nThe exact archive code is {code}. Reply with ONLY {code}."
            erased_before = infra.action("erase")
            cold = infra.completion(prompt, n_predict=24)
            filename = f"functional-{index:02d}.bin"
            saved = infra.action("save", filename)
            slot_path = f"{save_dir}/{filename}"
            slot_identity = {"path": slot_path, "bytes": infra.stat_wsl(slot_path),
                             "sha256": infra.sha256_wsl(slot_path, timeout=600.0)}
            erased = infra.action("erase")
            restored = infra.action("restore", filename)
            warm = infra.completion(prompt, n_predict=24)
            erased_after = infra.action("erase")
            lifecycle = (
                lifecycle_ok(erased_before, saved, erased, restored, erased_after)
                and int(saved["body"].get("n_saved") or 0) > 0
                and int(restored["body"].get("n_restored") or 0) > 0
                and cold["http_status"] == 200 and warm["http_status"] == 200
            )
            cold_prompt_n = int(cold["timings"].get("prompt_n") or 0)
            warm_prompt_n = int(warm["timings"].get("prompt_n") or 0)
            cache_n = int(warm["timings"].get("cache_n") or 0)
            cache_fraction = cache_n / cold_prompt_n if cold_prompt_n else 0.0
            prefill_reduction = 1.0 - (warm_prompt_n / cold_prompt_n) if cold_prompt_n else 0.0
            row = {
                "index": index, "code": code, "prompt_sha256": canonical_json_sha256(prompt),
                "launch_unit": unit, "erased_before": erased_before, "cold": cold,
                "saved": saved, "slot_identity": slot_identity, "erased": erased,
                "restored": restored, "warm": warm, "erased_after": erased_after,
                "lifecycle_ok": lifecycle, "cache_n": cache_n,
                "cold_prompt_n": cold_prompt_n, "warm_prompt_n": warm_prompt_n,
                "cache_fraction_of_prompt": cache_fraction,
                "prefill_token_reduction": prefill_reduction,
                "exact_continuation": cold["content"] == warm["content"],
            }
            rows.append(row)
            append_jsonl(raw / "samples.jsonl", row)
            write_json(finalized / f"cycle-{index:02d}.json", {
                "index": index, "cache_n": cache_n,
                "prefill_token_reduction": prefill_reduction,
                "exact_continuation": row["exact_continuation"],
            })
            print(f"{index + 1:02d}/{CYCLES} cache_n={cache_n} prefill_reduction={prefill_reduction:.4f} exact={row['exact_continuation']}", flush=True)
            if (index + 1) % 4 == 0 and infra.health(infra.EMBED_URL)[0] != 200:
                raise RuntimeError("embedding unhealthy at cycle boundary")
    except Exception as caught:
        execution_error = caught
    finally:
        if unit:
            journal = infra.wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True)["stdout"]
            (logs / f"{unit}.log").write_text(journal, encoding="utf-8")
            infra.wsl("systemctl", "stop", unit, root=True, timeout=180.0)
            try:
                infra.wait_unit(unit, active=False, timeout_seconds=180.0)
            except RuntimeError:
                pass
        if save_dir:
            expected_dir = f"{infra.SAVE_ROOT}/obs-{TRANSIENT_INDEX:02d}"
            if save_dir != expected_dir:
                raise RuntimeError(f"refusing unexpected cleanup target: {save_dir}")
            infra.wsl("rm", "-rf", "--", save_dir, root=True, timeout=180.0)
        try:
            if infra.unit_state(infra.PERSISTENT_UNIT)["active_state"] != "active":
                infra.systemctl("start", infra.PERSISTENT_UNIT)
            infra.wait_health(infra.GATEWAY_URL)
            restored_gateway = infra.restore_model(initial_model)
            final_embed_status, final_embed_body = infra.health(infra.EMBED_URL)
            restoration = {
                "gateway": restored_gateway,
                "service": infra.unit_state(infra.PERSISTENT_UNIT),
                "embedding": {"http_status": final_embed_status, "body": final_embed_body},
                "initial_model_restored": restored_gateway.get("current_model") == initial_model,
            }
        except Exception as restore_error:
            restoration = {"error": f"{type(restore_error).__name__}: {restore_error}",
                           "initial_model_restored": False}
            if execution_error is None:
                execution_error = restore_error
        write_json(raw / "recovery_state.json", restoration)
    if execution_error:
        raise execution_error

    service_restored = (
        restoration.get("initial_model_restored") is True
        and restoration.get("service", {}).get("active_state") == "active"
        and restoration.get("embedding", {}).get("http_status") == 200
    )
    metrics = {
        "sources_and_runtime_verified": True,
        "completed_cycles": len(rows),
        "successful_lifecycle_rate": sum(row["lifecycle_ok"] for row in rows) / len(rows),
        "cycles_with_cache_n_positive": sum(row["cache_n"] > 0 for row in rows),
        "median_cache_fraction_of_prompt": statistics.median(row["cache_fraction_of_prompt"] for row in rows),
        "median_prefill_token_reduction": statistics.median(row["prefill_token_reduction"] for row in rows),
        "exact_continuation_rate": sum(row["exact_continuation"] for row in rows) / len(rows),
        "median_cold_prompt_n": statistics.median(row["cold_prompt_n"] for row in rows),
        "median_warm_prompt_n": statistics.median(row["warm_prompt_n"] for row in rows),
        "service_and_embedding_restored": service_restored,
    }
    definitions = {
        "source_integrity": ("sources_and_runtime_verified", "eq", True),
        "cycle_coverage": ("completed_cycles", "eq", 16),
        "lifecycle_integrity": ("successful_lifecycle_rate", "eq", 1.0),
        "functional_cache_reuse": ("cycles_with_cache_n_positive", "eq", 16),
        "material_cache_reuse": ("median_cache_fraction_of_prompt", "ge", 0.80),
        "prefill_reduction": ("median_prefill_token_reduction", "ge", 0.80),
        "continuation_parity": ("exact_continuation_rate", "eq", 1.0),
        "service_recovery": ("service_and_embedding_restored", "eq", True),
    }
    gates: dict[str, Any] = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold,
                          "actual": actual, "pass": passed}

    r1_receipt = json.loads((ROOT / "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/receipt.json").read_text(encoding="utf-8"))
    r2_receipt = json.loads((ROOT / "runs/research/BACKLOG-MTP-PERSISTENCE-RESCORE-02/raw/receipt.json").read_text(encoding="utf-8"))
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "artifact_hashes.json", {"host": host_ledger, "wsl": wsl_ledger,
               "slot_files": [row["slot_identity"] for row in rows]})
    write_json(raw / "dataset_hashes.json", {"prompt_hashes": [row["prompt_sha256"] for row in rows],
               "prompt_set_sha256": canonical_json_sha256([row["prompt_sha256"] for row in rows])})
    write_json(raw / "effective_route.json", {"launch": launch, "unit": unit, "mtp_args": infra.MTP_ARGS})
    write_json(raw / "functional_reuse.json", {"cache_n": [row["cache_n"] for row in rows],
               "cache_fraction": [row["cache_fraction_of_prompt"] for row in rows],
               "prefill_reduction": [row["prefill_token_reduction"] for row in rows]})
    write_json(raw / "hardware_metrics.json", {"gpu": infra.gpu_state(),
               "median_cold_prompt_n": metrics["median_cold_prompt_n"],
               "median_warm_prompt_n": metrics["median_warm_prompt_n"]})
    write_json(raw / "independent_evaluation.json", {"functional_reuse_requires_all_three": [
               "cache_n_positive", "prefill_reduction", "continuation_parity"], "metrics": metrics})
    write_json(raw / "paired_baseline.json", {"baseline": "cold_after_erase", "treatment": "same_prompt_after_restore",
               "paired_cycles": CYCLES})
    write_json(raw / "service_identity.json", {"initial_service": initial_service,
               "initial_gateway": initial_gateway, "restoration": restoration})
    write_json(raw / "service_maintenance.json", {"persistent_service_stopped_via_systemd": True,
               "embedding_service_stopped": False, "transient_unit": unit, "restoration": restoration})
    write_json(raw / "source_execution_receipt.json", {
        "r1": {"sha256": sha256_file(ROOT / "runs/research/BACKLOG-MTP-PERSISTENCE-01/raw/receipt.json"),
               "fingerprint": r1_receipt["receipt_fingerprint"]},
        "r2": {"sha256": sha256_file(ROOT / "runs/research/BACKLOG-MTP-PERSISTENCE-RESCORE-02/raw/receipt.json"),
               "fingerprint": r2_receipt["receipt_fingerprint"]},
    })
    evidence = {
        "acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json",
        "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json",
        "effective_route": "raw/effective_route.json", "functional_reuse": "raw/functional_reuse.json",
        "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json",
        "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json",
        "raw_samples": "raw/samples.jsonl", "receipt_fingerprint": "raw/receipt.json",
        "recovery_state": "raw/recovery_state.json", "service_identity": "raw/service_identity.json",
        "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json",
    }
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started,
        started_monotonic=mono, input_paths=[*input_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "physical_mtp_functional_reuse", "cycles": CYCLES,
                 "transient_unit": unit},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
               "provenance": provenance, "provenance_complete": True,
               "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "MTP_PERSISTENCE_FUNCTIONAL_REUSE_CONFIRMED_R3" if not failed else "MTP_PERSISTENCE_FUNCTIONAL_REUSE_REJECTED_R3"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Positive cache reuse `{metrics['cycles_with_cache_n_positive']}/{CYCLES}`; median cache fraction "
        f"`{metrics['median_cache_fraction_of_prompt']:.4f}`, median prefill reduction "
        f"`{metrics['median_prefill_token_reduction']:.4f}`, exact continuation "
        f"`{metrics['exact_continuation_rate']:.4f}`. Failed gates: "
        f"`{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert CYCLES == 16 and TRANSIENT_INDEX == 99
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
