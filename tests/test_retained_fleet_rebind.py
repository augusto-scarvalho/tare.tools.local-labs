from __future__ import annotations

import copy

from tools.analysis.experiment_provenance import canonical_json_sha256
from tools.research import run_retained_fleet_rebind as target


def _registry():
    models = {}
    for index, model in enumerate(("qwen38", "hauhaucs", "fable-tc", "qwen36-moe"), 1):
        models[model] = {
            "artifact": {"sha256": str(index) * 64},
            "runtime": {"args": ["--ctx-size", "100", "--parallel", "2"]},
        }
    return {"models": models}


def _context_fixture():
    rows = []
    cases = []
    for model in ("qwen38", "hauhaucs", "fable-tc", "qwen36-moe"):
        for position in ("start", "middle", "end"):
            for replicate in range(6):
                code = f"{model}-{position}-{replicate}"
                prompt = f"prompt:{replicate}:{position}:{code}"
                case = {
                    "model": model,
                    "target_tokens": 10,
                    "position": position,
                    "replicate": replicate,
                    "code": code,
                    "filler_count": replicate,
                    "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                }
                cases.append(case)
                rows.append(case | {
                    "http_status": 200,
                    "error": None,
                    "exact_recall": True,
                    "prompt_n": 40,
                    "response": {"model": model},
                })
    artifacts = {"fleet_base": {"wsl_artifacts": {
        model: {"sha256": card["artifact"]["sha256"]}
        for model, card in _registry()["models"].items()
    }}}
    return rows, cases, artifacts


def test_context_fixture_recomputes_all_72_rows_and_buckets():
    rows, cases, artifacts = _context_fixture()

    metrics, analysis = target.context_metrics(
        rows,
        cases,
        _registry(),
        artifacts,
        lambda count, position, code: f"prompt:{count}:{position}:{code}",
    )

    assert metrics["retained_rows_recomputed"] == 72
    assert metrics["prompt_hash_reconstruction_rate"] == 1.0
    assert metrics["minimum_position_bucket_recall"] == 1.0
    assert metrics["verified_model_artifacts"] == 4
    assert metrics["requests_within_route_slot_context"] == 72
    assert analysis["case_joins"] == 72


def _response(model: str, content: str):
    return {"model": model, "choices": [{"message": {"content": content}}]}


def test_gateway_fixture_separates_transport_from_semantic_content():
    registry = _registry()
    registry["models"].update({
        "gemma-vision": {"artifact": {"sha256": "5" * 64}},
        "muse-vision": {"artifact": {"sha256": "6" * 64}},
    })
    rows = []
    for cycle in range(5):
        for model in registry["models"]:
            for probe in range(4):
                content = "" if model.endswith("vision") else "OK"
                rows.append({
                    "cycle": cycle,
                    "model": model,
                    "probe": str(probe),
                    "http_status": 200,
                    "error": None,
                    "response": _response(model, content),
                    "semantic_sha256": canonical_json_sha256({"model": model, "probe": probe}),
                })
    switches = [{"embedding_health": 200} for _ in range(30)]
    service = {
        "initial": {"n_restarts": 0},
        "final": {"n_restarts": 0},
        "gateway_initial": {"current_model": "qwen38"},
    }

    metrics, analysis = target.gateway_metrics(
        rows, switches, registry, service, {"status": {"current_model": "qwen38"}}
    )

    assert metrics["http_transport_success_rate"] == 1.0
    assert metrics["route_alias_match_rate"] == 1.0
    assert metrics["eligible_text_nonempty_content_rate"] == 1.0
    assert metrics["verified_distinct_model_artifacts"] == 6
    assert metrics["initial_model_restored"] is True
    assert analysis["eligible_text_rows"] == 80


def test_source_receipt_fingerprint_rejects_tampering():
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": "X"}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    assert target.verify_source_receipt(receipt)
    tampered = copy.deepcopy(receipt)
    tampered["task_id"] = "Y"
    assert not target.verify_source_receipt(tampered)
