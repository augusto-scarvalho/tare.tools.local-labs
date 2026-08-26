#!/usr/bin/env python3
"""Independent CTRL-01 replay on real llama-server tokenizer pieces."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.ast_grammar_sidecar import ASTGrammarSidecar

SOURCE_PATHS = [
    "runs/research/CTRL-01-AST-SIDECAR-2026-08-25/PRE_REGISTRATION.md",
    "runs/research/CTRL-01-AST-SIDECAR-2026-08-25/RESULT.md",
    "runs/research/CTRL-01-AST-SIDECAR-2026-08-25/raw/receipt.json",
    "tools/analysis/ast_grammar_sidecar.py",
    "tools/probes/ctrl01_ast_sidecar_probe.py",
]

EXPECTED_SOURCE_HASHES = {
    SOURCE_PATHS[0]: "c07cb593242fdd22dda6dbe9058967da0df87204d75c3c8d3aa4ff14a5510946",
    SOURCE_PATHS[1]: "cd4319c345bdaa8224da7d9782fde7ee8e2f861fe1b3c6caefb3041adf4b31eb",
    SOURCE_PATHS[2]: "0f37ae1d3ff33286a193353731f864d699ce738734fd8cc5b5a55384c2cf2c7c",
    SOURCE_PATHS[3]: "3cb90b1b5aa5aacdff93b7a8b0cdc38e689099e0d1365989f00b7b34acbb1463",
    SOURCE_PATHS[4]: "9e7ed6d27936952f20bbb27f0fbcf6530f2ebbec24dbf46faf46de8c22feb669",
}

SCHEMAS = [
    ("flat", 'Return only one JSON object with keys "name" (string) and "count" (integer). Use name alpha and count 7.'),
    ("array", 'Return only one JSON object with key "items", an array containing exactly 1, 2, and 3.'),
    ("nested", 'Return only one JSON object: status is ok and meta contains enabled true and note null.'),
    ("negative", 'Return only one JSON object with key "delta" and numeric value negative 12.'),
    ("decimal", 'Return only one JSON object with key "ratio" and numeric value 0.125.'),
    ("escape", 'Return only one JSON object with key "text" whose value contains a newline escape between a and b.'),
    ("unicode", 'Return only one JSON object with key "city" and string value São Paulo.'),
    ("records", 'Return only one JSON array with two objects, ids 1 and 2, each with active boolean true then false.'),
]
SEEDS = [20260824, 20260825, 20260826]
VALID_CONTROLS = [
    '{"name":"alpha","count":7}',
    '{"items":[1,2,3]}',
    '{"meta":{"enabled":true,"note":null}}',
    '{"delta":-12}',
    '{"ratio":0.125}',
    '{"tiny":1e-6}',
    '{"huge":-2.5E+12}',
    '{"text":"a\\nb"}',
    '{"quote":"a\\\"b","slash":"c\\\\d"}',
    '{"city":"São Paulo"}',
    '[{"id":1,"active":true},{"id":2,"active":false}]',
    '  { "space" : [ null, false, true ] }  ',
]


def sha256_path(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def token_pieces(base_url: str, text: str) -> list[str]:
    payload = post_json(
        f"{base_url}/tokenize",
        {"content": text, "add_special": False, "with_pieces": True},
        timeout=30.0,
    )
    tokens = payload.get("tokens", [])
    if not tokens or not all(isinstance(row, dict) and "piece" in row for row in tokens):
        raise RuntimeError("tokenizer did not return exact pieces")
    return [str(row["piece"]) for row in tokens]


def json_valid(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def stable_model_identity(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Project /v1/models onto fields that identify loaded weights/runtime shape."""
    identities = []
    for row in payload.get("data", []):
        meta = row.get("meta", {})
        identities.append({
            "id": row.get("id"),
            "aliases": row.get("aliases"),
            "owned_by": row.get("owned_by"),
            "vocab_type": meta.get("vocab_type"),
            "n_vocab": meta.get("n_vocab"),
            "n_ctx": meta.get("n_ctx"),
            "n_ctx_train": meta.get("n_ctx_train"),
            "n_embd": meta.get("n_embd"),
            "n_params": meta.get("n_params"),
            "size": meta.get("size"),
            "ftype": meta.get("ftype"),
        })
    return identities


def replay(sidecar: ASTGrammarSidecar, pieces: list[str]) -> dict[str, Any]:
    current = ""
    decisions: list[dict[str, Any]] = []
    t0 = time.perf_counter_ns()
    for index, piece in enumerate(pieces):
        accepted = sidecar.validate_and_filter_token(current, piece)
        decisions.append({"index": index, "piece": piece, "accepted": accepted})
        if accepted:
            current += piece
    elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
    return {
        "filtered": current,
        "intercepted": sum(not row["accepted"] for row in decisions),
        "elapsed_us": elapsed_us,
        "us_per_token": elapsed_us / max(1, len(pieces)),
        "decisions": decisions,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def runtime_binding_evidence() -> dict[str, Any]:
    matches: list[str] = []
    excluded = {
        (ROOT / "tools/analysis/ast_grammar_sidecar.py").resolve(),
        (ROOT / "tools/probes/ctrl01_ast_sidecar_probe.py").resolve(),
        pathlib.Path(__file__).resolve(),
    }
    for base in (ROOT / "src", ROOT / "ops"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in {".py", ".cpp", ".cc", ".c", ".h", ".hpp"}:
                continue
            if path.resolve() in excluded:
                continue
            try:
                body = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "ASTGrammarSidecar" in body or "ast_grammar_sidecar" in body:
                matches.append(path.relative_to(ROOT).as_posix())
    try:
        service = subprocess.run(
            ["wsl", "-d", "Ubuntu-24.04", "--", "systemctl", "show", "llm-inference.service", "-p", "ExecStart", "-p", "MainPID", "--no-pager"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        service = "unavailable"
    return {
        "production_code_matches": sorted(matches),
        "active_service": service,
        "logit_mask_runtime_integrated": bool(matches),
        "criterion": "requires a production code reference that binds the sidecar to candidate logits before sampling",
    }


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(outdir: pathlib.Path, base_url: str, max_tokens: int = 128) -> dict[str, Any]:
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    actual_hashes = {name: sha256_path(ROOT / name) for name in SOURCE_PATHS}
    if actual_hashes != EXPECTED_SOURCE_HASHES:
        raise RuntimeError(f"frozen source hash mismatch: {actual_hashes}")

    health_before = get_json(f"{base_url}/health")
    models_before = get_json(f"{base_url}/v1/models")
    sidecar = ASTGrammarSidecar(mode="json")
    samples: list[dict[str, Any]] = []
    latencies: list[float] = []

    for schema_id, instruction in SCHEMAS:
        for seed in SEEDS:
            request_payload = {
                "model": "local",
                "messages": [
                    {"role": "system", "content": "Follow the requested output syntax exactly. Do not use Markdown."},
                    {"role": "user", "content": instruction},
                ],
                "temperature": 0.0,
                "seed": seed,
                "max_tokens": max_tokens,
                "stream": False,
            }
            response = post_json(f"{base_url}/v1/chat/completions", request_payload)
            content = str(response["choices"][0]["message"].get("content") or "")
            pieces = token_pieces(base_url, content)
            replayed = replay(sidecar, pieces)
            latencies.append(replayed["us_per_token"])
            samples.append({
                "kind": "real_model",
                "schema_id": schema_id,
                "seed": seed,
                "request": request_payload,
                "response": response,
                "raw_content": content,
                "raw_json_valid": json_valid(content),
                "token_pieces": pieces,
                **replayed,
                "filtered_json_valid": json_valid(replayed["filtered"]),
                "exact_preservation": replayed["filtered"] == content,
            })

    for index, document in enumerate(VALID_CONTROLS):
        pieces = token_pieces(base_url, document)
        replayed = replay(sidecar, pieces)
        latencies.append(replayed["us_per_token"])
        samples.append({
            "kind": "valid_control",
            "control_id": index,
            "raw_content": document,
            "raw_json_valid": json_valid(document),
            "token_pieces": pieces,
            **replayed,
            "filtered_json_valid": json_valid(replayed["filtered"]),
            "exact_preservation": replayed["filtered"] == document,
        })

    health_after = get_json(f"{base_url}/health")
    models_after = get_json(f"{base_url}/v1/models")
    if stable_model_identity(models_before) != stable_model_identity(models_after):
        raise RuntimeError("server model identity changed during execution")

    real = [row for row in samples if row["kind"] == "real_model"]
    controls = [row for row in samples if row["kind"] == "valid_control"]
    control_tokens = sum(len(row["token_pieces"]) for row in controls)
    control_accepted = sum(sum(decision["accepted"] for decision in row["decisions"]) for row in controls)
    binding = runtime_binding_evidence()
    metrics = {
        "real_model_outputs": len(real),
        "raw_complete_valid_rate": sum(row["raw_json_valid"] for row in real) / len(real),
        "sanitized_complete_valid_rate": sum(row["filtered_json_valid"] for row in real) / len(real),
        "real_exact_preservation_rate": sum(row["exact_preservation"] for row in real) / len(real),
        "valid_controls": len(controls),
        "valid_token_acceptance_rate": control_accepted / control_tokens,
        "valid_control_exact_preservation_rate": sum(row["exact_preservation"] for row in controls) / len(controls),
        "valid_control_complete_valid_rate": sum(row["filtered_json_valid"] for row in controls) / len(controls),
        "p50_overhead_us_per_token": statistics.median(latencies),
        "p95_overhead_us_per_token": percentile(latencies, 0.95),
        "logit_mask_runtime_integrated": binding["logit_mask_runtime_integrated"],
    }
    gates = {
        "real_coverage": metrics["real_model_outputs"] >= 24,
        "real_validity": metrics["sanitized_complete_valid_rate"] == 1.0,
        "valid_control_recall": metrics["valid_token_acceptance_rate"] == 1.0,
        "valid_control_semantics": metrics["valid_control_exact_preservation_rate"] == 1.0,
        "overhead": metrics["p95_overhead_us_per_token"] <= 500.0,
        "runtime_binding": metrics["logit_mask_runtime_integrated"] is True,
    }
    verdict = "QUALIFIED" if all(gates.values()) else "FALSE_POSITIVE_CONFIRMED"
    claim_code = "CTRL01_RUNTIME_QUALIFIED_R1" if verdict == "QUALIFIED" else "CTRL01_FALSE_POSITIVE_CONFIRMED_R1"

    samples_path = raw_dir / "samples.jsonl"
    samples_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    provenance = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": "python tools/research/run_ctrl01_real_token.py --outdir runs/research/BACKLOG-CTRL01-REAL-TOKEN-01",
        "python": sys.version,
        "platform": platform.platform(),
        "base_url": base_url,
        "health_before": health_before,
        "health_after": health_after,
        "models": models_before,
        "source_hashes": actual_hashes,
        "preregistration_sha256": sha256_path(outdir / "PRE_REGISTRATION.md"),
        "runner_sha256": sha256_path(pathlib.Path(__file__)),
        "samples_sha256": sha256_path(samples_path),
        "runtime_binding_evidence": binding,
    }
    receipt = {
        "schema": "local-labs-ctrl01-real-token-v1",
        "executor": "Codex executor",
        "metrics": metrics,
        "gates": gates,
        "verdict": verdict,
        "claim_code_pending_independent_review": claim_code,
        "provenance": provenance,
    }
    canonical = json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode("utf-8")
    receipt["receipt_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_json(raw_dir / "receipt.json", receipt)
    write_json(raw_dir / "artifact_hashes.json", {
        **actual_hashes,
        "runs/research/BACKLOG-CTRL01-REAL-TOKEN-01/PRE_REGISTRATION.md": provenance["preregistration_sha256"],
        "tools/research/run_ctrl01_real_token.py": provenance["runner_sha256"],
        "runs/research/BACKLOG-CTRL01-REAL-TOKEN-01/raw/samples.jsonl": provenance["samples_sha256"],
    })
    return receipt


def write_result(outdir: pathlib.Path, receipt: dict[str, Any]) -> None:
    m = receipt["metrics"]
    failed = [name for name, passed in receipt["gates"].items() if not passed]
    result = f"""# BACKLOG-CTRL01-REAL-TOKEN-01 result

## Verdict

`{receipt['verdict']}` pending independent AGY review. Historical promotion is not supported by this successor.

## Actual results

- Real llama-server outputs: `{m['real_model_outputs']}`.
- Raw complete JSON validity: `{m['raw_complete_valid_rate']:.6f}`.
- Sidecar output complete JSON validity without repair: `{m['sanitized_complete_valid_rate']:.6f}`.
- Valid-token acceptance rate: `{m['valid_token_acceptance_rate']:.6f}`.
- Valid-control exact preservation: `{m['valid_control_exact_preservation_rate']:.6f}`.
- p50/p95 overhead: `{m['p50_overhead_us_per_token']:.3f}` / `{m['p95_overhead_us_per_token']:.3f}` microseconds/token.
- Pre-sampling runtime/logit binding found: `{m['logit_mask_runtime_integrated']}`.
- Failed mandatory gates: `{', '.join(failed) if failed else 'none'}`.

## Interpretation and claim limit

The historical probe repaired filtered strings by appending `}}` before parsing and operated on hand-built corrupted chunks. This successor applies no repair, uses exact tokenizer pieces for both real outputs and valid controls, and finds no production binding between the Python post-filter and the active sampler. It therefore cannot substantiate a guarantee of constrained decoding, even if some offline latency or validity submetrics pass.

Allowed pending claim: `{receipt['claim_code_pending_independent_review']}`. This does not evaluate Python mode, semantic correctness, or a future grammar-integrated runtime.

## Evidence

- `raw/receipt.json`
- `raw/samples.jsonl`
- `raw/artifact_hashes.json`
"""
    (outdir / "RESULT.md").write_text(result, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    outdir = (ROOT / args.outdir).resolve()
    receipt = run(outdir, args.base_url.rstrip("/"))
    write_result(outdir, receipt)
    print(json.dumps({"metrics": receipt["metrics"], "gates": receipt["gates"], "verdict": receipt["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
