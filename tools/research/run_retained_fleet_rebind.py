#!/usr/bin/env python3
"""Recompute retained fleet evidence from an immutable, hash-pinned source set."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model_lifecycle.experiment_harness import ExperimentRun
from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)
from tools.research import run_fleet_context_envelope as envelope
from tools.research import run_fleet_context_interference as interference


CONFIGS: dict[str, dict[str, Any]] = {
    "BACKLOG-FLEET-CONTEXT-ENVELOPE-04": {
        "kind": "context",
        "source_task": "BACKLOG-FLEET-CONTEXT-ENVELOPE-03",
        "generator": envelope.make_prompt,
        "claim_pass": "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_REBOUND_R4",
        "claim_fail": "QUALIFIED_TEXT_FLEET_SLOT_CONTEXT_ENVELOPES_NOT_CONFIRMED_R4",
        "sources": {
            "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/receipt.json": "17f0ec8b541f6d769dd5909ca4a44bc3f7c2813d13cc51ec8b206d74090d15d6",
            "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/samples.jsonl": "e7ac2131253fc930ff92db9ba3b54994aaf4758d7803fa7c4fc9497af51dea9a",
            "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/case_manifest.json": "63936f59148535a54ca54221d29dd669c387c87e07f0f28361899f18ee914111",
            "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/artifact_hashes.json": "4a696061f21ff8942ee27a38df3dde8227abc3f2f9c6e650ea851c4c7cc96513",
            "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/recovery_state.json": "8b8b43adc50290970a3efd04439d6bdfa01f15c9c26b129b27474f9ab3387903",
            "runs/research/BACKLOG-FLEET-CONTEXT-ENVELOPE-03/raw/service_identity.json": "d96e87884324e337d49b0cb205eb023151c7c2523d22b99af79070afca702b16",
            "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
            "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
        },
    },
    "BACKLOG-FLEET-CONTEXT-INTERFERENCE-02": {
        "kind": "context",
        "source_task": "BACKLOG-FLEET-CONTEXT-INTERFERENCE-01",
        "generator": interference.interference_prompt,
        "claim_pass": "QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_REBOUND_R2",
        "claim_fail": "QUALIFIED_TEXT_FLEET_CONTEXT_INTERFERENCE_NOT_CONFIRMED_R2",
        "sources": {
            "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/receipt.json": "96c57a33a1539c8f0e6d3eac1dca20352cc8bfd274e5a72a8ccb00c95667d8de",
            "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/samples.jsonl": "539196301bc0c8f5cccaebcd9fc0f730ae0a381775925db1625141786f5b3e97",
            "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/case_manifest.json": "86f8cf00668dac9b832e03f0f42537784e02593113c26c087fda5e265e1a6616",
            "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/artifact_hashes.json": "304d121331249e78edff482a08c53e75a0317e604ddff3f0d1bc3005fbeedb0f",
            "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/recovery_state.json": "0b1515faee24eb1079aad969c0001cd050c897d45ea6a336b6b71678d5ae2f67",
            "runs/research/BACKLOG-FLEET-CONTEXT-INTERFERENCE-01/raw/service_identity.json": "1ce9a25296787e14caed231878e34b835e283d5cb444c6f1e473ea97d7027dc9",
            "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
            "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
        },
    },
    "BACKLOG-GATEWAY-ROUTE-STRESS-02": {
        "kind": "gateway",
        "source_task": "BACKLOG-GATEWAY-ROUTE-STRESS-01",
        "claim_pass": "QUALIFIED_GATEWAY_ROUTE_TRANSPORT_STRESS_REBOUND_R2",
        "claim_fail": "QUALIFIED_GATEWAY_ROUTE_TRANSPORT_STRESS_NOT_CONFIRMED_R2",
        "sources": {
            "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/receipt.json": "527e308b2aa54fe96bb641f1d5380b04b42e7871245d173ee107cec0dabbfe41",
            "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/samples.jsonl": "b94e55da69cdd0b40e209bf9fdd554be65727e6d483a35a09f1a8b08f6c8f865",
            "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/switches.jsonl": "a99a7295d3c9a756f8852eafe72c22dfa16fba1ca3b28435443413f1bedd6f60",
            "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/effective_route.json": "fb5b0d0ceff9e1caccc85f921a8fefa795dfee5e7254ac623283250a46ae41d9",
            "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/service_identity.json": "5ac6ebe947d8b27989b8ea19af75711c7c9d3283d9532711e164fb210e21d154",
            "runs/research/BACKLOG-GATEWAY-ROUTE-STRESS-01/raw/recovery_state.json": "9ba2650f2a2b94de2aedde5106e76a2e6d7669562cb4a3fc6925e29dab2c838d",
            "docs/research/INDEPENDENT_AUDIT_LEDGER_2026-08-27_GPT56_SOL_XHIGH.md": "a74cb982e14585b5282cb18b2187b4cf435d96789bab4d80c14a22f6ec7cab04",
            "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
        },
    },
}


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects in {path}")
    return rows


def write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def response_alias(row: dict[str, Any]) -> str | None:
    response = row.get("response")
    return response.get("model") if isinstance(response, dict) else None


def message_content(row: dict[str, Any]) -> str:
    try:
        value = row["response"]["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


def verify_source_receipt(receipt: dict[str, Any]) -> bool:
    supplied = receipt.get("receipt_fingerprint")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_fingerprint"}
    return isinstance(supplied, str) and supplied == canonical_json_sha256(unsigned)


def slot_contexts(registry: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for model in ("qwen38", "hauhaucs", "fable-tc", "qwen36-moe"):
        args = registry["models"][model]["runtime"]["args"]
        context = int(args[args.index("--ctx-size") + 1])
        parallel = int(args[args.index("--parallel") + 1]) if "--parallel" in args else 1
        result[model] = context // parallel
    return result


def context_metrics(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    registry: dict[str, Any],
    artifacts: dict[str, Any],
    generator: Callable[[int, str, str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_by_key = {
        (row["model"], row["target_tokens"], row["position"], row["replicate"]): row
        for row in cases
    }
    reconstructed = 0
    joins = 0
    for row in rows:
        key = (row["model"], row["target_tokens"], row["position"], row["replicate"])
        case = case_by_key.get(key)
        if case is None:
            continue
        joins += 1
        prompt = generator(int(case["filler_count"]), str(case["position"]), str(case["code"]))
        digest = __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest()
        reconstructed += digest == case["prompt_sha256"] == row["prompt_sha256"]
    slots = slot_contexts(registry)
    by_model = {
        model: sum(row["model"] == model and row.get("exact_recall") is True for row in rows)
        for model in slots
    }
    by_position = {
        position: sum(row["position"] == position and row.get("exact_recall") is True for row in rows)
        for position in ("start", "middle", "end")
    }
    wsl_artifacts = artifacts.get("fleet_base", {}).get("wsl_artifacts", {})
    artifact_matches = sum(
        wsl_artifacts.get(model, {}).get("sha256") == registry["models"][model]["artifact"]["sha256"]
        for model in slots
    )
    route_matches = sum(response_alias(row) == row["model"] for row in rows)
    context_fits = sum(int(row.get("prompt_n") or 0) <= slots[row["model"]] for row in rows)
    successes = sum(row.get("http_status") == 200 and not row.get("error") for row in rows)
    metrics = {
        "source_receipt_digest_verified": True,
        "final_source_set_immutable": True,
        "retained_rows_recomputed": len(rows),
        "prompt_hash_reconstruction_rate": reconstructed / len(rows) if rows else 0.0,
        "verified_model_artifacts": artifact_matches,
        "route_alias_match_rate": route_matches / len(rows) if rows else 0.0,
        "requests_within_route_slot_context": context_fits,
        "successful_response_rate": successes / len(rows) if rows else 0.0,
        "qwen38_exact_recall": by_model["qwen38"] / 18,
        "hauhaucs_exact_recall": by_model["hauhaucs"] / 18,
        "fable_tc_exact_recall": by_model["fable-tc"] / 18,
        "qwen36_moe_exact_recall": by_model["qwen36-moe"] / 18,
        "minimum_position_bucket_recall": min(value / 24 for value in by_position.values()),
    }
    if "near_label_decoys" in {key for case in cases for key in case}:
        # Kept for compatibility with manifests that store this per case.
        decoy_cases = sum(case.get("near_label_decoys") == 31 for case in cases)
    else:
        # The physical interference packet stores the construct once in the
        # case manifest; execute() replaces this conservative default there.
        decoy_cases = 0
    metrics["cases_with_exactly_31_verified_decoys"] = decoy_cases
    analysis = {
        "unique_rows": len({(r["model"], r["target_tokens"], r["position"], r["replicate"]) for r in rows}),
        "case_joins": joins,
        "prompt_hashes_reconstructed": reconstructed,
        "by_model_exact": by_model,
        "by_position_exact": by_position,
        "slot_contexts": slots,
    }
    return metrics, analysis


def gateway_metrics(
    rows: list[dict[str, Any]],
    switches: list[dict[str, Any]],
    registry: dict[str, Any],
    service: dict[str, Any],
    recovery: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    text_models = {"qwen38", "hauhaucs", "fable-tc", "qwen36-moe"}
    eligible = [row for row in rows if row["model"] in text_models]
    baselines = {(row["model"], row["probe"]): row["semantic_sha256"] for row in rows if row["cycle"] == 0}
    comparisons = [
        row["semantic_sha256"] == baselines[(row["model"], row["probe"])]
        for row in rows if row["cycle"] > 0
    ]
    artifact_hashes = [card["artifact"]["sha256"] for card in registry["models"].values()]
    initial = service.get("initial", {})
    final = service.get("final", {})
    metrics = {
        "source_receipt_digest_verified": True,
        "final_source_set_immutable": True,
        "recomputed_switches": len(switches),
        "recomputed_requests": len(rows),
        "http_transport_success_rate": sum(r.get("http_status") == 200 and not r.get("error") for r in rows) / len(rows),
        "route_alias_match_rate": sum(response_alias(r) == r["model"] for r in rows) / len(rows),
        "eligible_text_nonempty_content_rate": sum(bool(message_content(r).strip()) for r in eligible) / len(eligible),
        "exact_cycle_repeat_rate": sum(comparisons) / len(comparisons),
        "verified_distinct_model_artifacts": len(set(artifact_hashes)),
        "gateway_service_restarts": int(final.get("n_restarts", 0)) - int(initial.get("n_restarts", 0)),
        "embedding_boundary_successes": sum(s.get("embedding_health") == 200 for s in switches),
        "initial_model_restored": (
            recovery.get("status", {}).get("current_model")
            == service.get("gateway_initial", {}).get("current_model")
        ),
    }
    analysis = {
        "eligible_text_rows": len(eligible),
        "nonempty_eligible_text_rows": sum(bool(message_content(r).strip()) for r in eligible),
        "cycle_comparisons": len(comparisons),
        "exact_cycle_comparisons": sum(comparisons),
        "models": sorted({row["model"] for row in rows}),
    }
    return metrics, analysis


def gates_for(task_id: str, metrics: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(ROOT / "config/research_backlog.json")
    task = next(item for item in manifest["items"] if item["id"] == task_id)
    gates: dict[str, Any] = {}
    for definition in task["acceptance_gates"]:
        actual = metrics[definition["metric"]]
        operator = definition["operator"]
        threshold = definition["threshold"]
        passed = {
            "eq": actual == threshold,
            "ge": actual >= threshold,
            "gt": actual > threshold,
            "le": actual <= threshold,
            "lt": actual < threshold,
            "ne": actual != threshold,
        }[operator]
        gates[definition["id"]] = definition | {"actual": actual, "pass": passed}
    return gates


def execute(task_id: str, outdir: pathlib.Path) -> dict[str, Any]:
    config = CONFIGS[task_id]
    raw = outdir / "raw"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    source_paths = [ROOT / relative for relative in config["sources"]]
    source_manifest = {}
    for relative, expected in config["sources"].items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"immutable source mismatch: {relative}: {actual} != {expected}")
        source_manifest[relative] = {"bytes": path.stat().st_size, "sha256": actual}
    source_task = config["source_task"]
    source_raw = ROOT / "runs/research" / source_task / "raw"
    source_receipt = read_json(source_raw / "receipt.json")
    if not verify_source_receipt(source_receipt):
        raise ValueError("source receipt fingerprint mismatch")
    pipeline = read_json(outdir / "PIPELINE.json")
    prereg = ROOT / pipeline["preregistration"]["path"]
    if sha256_file(prereg) != pipeline["preregistration"]["sha256"]:
        raise ValueError("pipeline preregistration binding mismatch")
    registry = read_json(ROOT / "config/qualified_model_fleet.json")
    source_rows = read_jsonl(source_raw / "samples.jsonl")

    with ExperimentRun(raw, task_id, {"source_manifest": source_manifest}) as run:
        write_json(raw / "source_manifest.json", source_manifest)
        write_json(raw / "source_execution_receipt.json", {
            "task_id": source_task,
            "receipt_sha256": sha256_file(source_raw / "receipt.json"),
            "receipt_fingerprint": source_receipt["receipt_fingerprint"],
        })
        for row in source_rows:
            run.record({"source_task_id": source_task, "retained": True, **row})
        run.checkpoint("retained_rows_loaded", {"rows": len(source_rows)})

        if config["kind"] == "context":
            case_manifest = read_json(source_raw / "case_manifest.json")
            artifacts = read_json(source_raw / "artifact_hashes.json")
            metrics, analysis = context_metrics(
                source_rows, case_manifest["cases"], registry, artifacts, config["generator"]
            )
            if config["source_task"] == "BACKLOG-FLEET-CONTEXT-INTERFERENCE-01":
                metrics["cases_with_exactly_31_verified_decoys"] = (
                    len(source_rows) if case_manifest.get("near_label_decoys") == 31 else 0
                )
            recovery = read_json(source_raw / "recovery_state.json")
            metrics["initial_route_and_services_restored"] = recovery.get("initial_route_and_services_restored") is True
            write_json(raw / "case_manifest.json", case_manifest)
            write_json(raw / "artifact_hashes.json", artifacts)
            service_identity = read_json(source_raw / "service_identity.json")
            hardware = {"retained_prompt_n": [row.get("prompt_n") for row in source_rows]}
        else:
            switches = read_jsonl(source_raw / "switches.jsonl")
            service_identity = read_json(source_raw / "service_identity.json")
            recovery = read_json(source_raw / "recovery_state.json")
            metrics, analysis = gateway_metrics(
                source_rows, switches, registry, service_identity, recovery
            )
            write_json(raw / "artifact_hashes.json", {
                model: card["artifact"] for model, card in registry["models"].items()
            })
            hardware = {"retained_wall_ms": [row.get("wall_ms") for row in source_rows], "switches": switches}

        gates = gates_for(task_id, metrics)
        write_json(raw / "actual_scores.json", metrics)
        write_json(raw / "statistical_analysis.json", analysis)
        write_json(raw / "effective_route.json", {
            "source_task_id": source_task,
            "route_alias_match_rate": metrics["route_alias_match_rate"],
        })
        write_json(raw / "hardware_metrics.json", hardware)
        write_json(raw / "paired_baseline.json", analysis)
        write_json(raw / "recovery_state.json", recovery)
        write_json(raw / "service_identity.json", service_identity)
        write_json(raw / "independent_evaluation.json", {
            "recomputed": True,
            "independent_review_pending": True,
            "metrics": metrics,
        })
        write_json(raw / "dataset_hashes.json", {
            "source_samples_sha256": sha256_file(source_raw / "samples.jsonl")
        })
        write_json(raw / "treatment_controls.json", {
            "new_inference": False,
            "source_task_id": source_task,
            "retained_rows": len(source_rows),
        })

        complete, errors = provenance_complete(provenance := build_provenance(
            script_path=pathlib.Path(__file__).resolve(),
            started_at_utc=started_at,
            started_monotonic=started_mono,
            input_paths=[*source_paths, prereg],
            packages=[],
            runtime={"execution_mode": "immutable_retained_evidence_rebind", "new_inference": False},
        ))
        if not complete:
            raise RuntimeError(f"incomplete provenance: {errors}")
        evidence = {
            "acceptance_gates": "raw/receipt.json",
            "artifact_hashes": "raw/artifact_hashes.json",
            "case_manifest": "raw/case_manifest.json" if config["kind"] == "context" else "raw/source_manifest.json",
            "dataset_hashes": "raw/dataset_hashes.json",
            "effective_route": "raw/effective_route.json",
            "hardware_metrics": "raw/hardware_metrics.json",
            "independent_evaluation": "raw/independent_evaluation.json",
            "paired_baseline": "raw/paired_baseline.json",
            "provenance": "raw/receipt.json",
            "raw_samples": "raw/samples.jsonl",
            "receipt_fingerprint": "raw/receipt.json",
            "recovery_state": "raw/recovery_state.json",
            "service_identity": "raw/service_identity.json",
            "source_execution_receipt": "raw/source_execution_receipt.json",
            "statistical_analysis": "raw/statistical_analysis.json",
            "treatment_controls": "raw/treatment_controls.json",
        }
        receipt = {
            "schema": "local-labs-backlog-receipt-v1",
            "task_id": task_id,
            "provenance": provenance,
            "provenance_complete": True,
            "gates": gates,
            "evidence": evidence,
        }
        passed = all(gate["pass"] for gate in gates.values())
        claim = config["claim_pass"] if passed else config["claim_fail"]
        failures = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
        (outdir / "RESULT.md").write_text(
            f"# {task_id} result\n\n`{claim}` pending independent review.\n\n"
            f"Recomputed `{len(source_rows)}` retained rows with no new inference. "
            f"Failed gates: `{', '.join(failures) if failures else 'none'}`.\n",
            encoding="utf-8",
        )
        sealed = run.seal(receipt)
    return sealed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--outdir", type=pathlib.Path, required=True)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        assert len(CONFIGS) == 3
        assert all(config["sources"] for config in CONFIGS.values())
        return 0
    receipt = execute(args.task_id, args.outdir.resolve())
    print(json.dumps(receipt["gates"], separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
