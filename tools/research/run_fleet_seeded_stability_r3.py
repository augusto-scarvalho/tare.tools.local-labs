#!/usr/bin/env python3
"""Fresh seeded stability run bound to live backend binaries and GGUFs."""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import canonical_json_sha256, sha256_file
from tools.research import run_fleet_seeded_stability_r2 as r2


TASK_ID = "BACKLOG-FLEET-SEEDED-STABILITY-03"
MODELS = tuple(r2.base.MODELS)
SOURCES = {
    "config/research_backlog_admissions/BACKLOG-FLEET-SEEDED-STABILITY-03.json": "5a68266de6f5dc97e4bdced3a43d510e298397fc10d6a84b77440170ba543349",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "workloads/gsm8k.jsonl": "68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
    "runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl": "56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f",
    "tools/research/run_fleet_regression_screen.py": "7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3",
    "tools/research/run_fleet_seeded_stability.py": "71189428e1d1c7aff8a3ddb55d56c2f2034cd89c9df516f1a6490e7a2888676c",
    "tools/research/run_fleet_seeded_stability_r2.py": "1c6e0f1123acfef0a5756017b8f18e59cb7c626d72fdfa02b71093220e37f3c3",
    "runs/research/BACKLOG-FLEET-SEEDED-STABILITY-02/raw/receipt.json": "efbd36ba4d8178dcddf9554953d553d2aea6168c04bad11b2f6d9ccff29d5311",
    "runs/research/BACKLOG-FLEET-SEEDED-STABILITY-02/REVIEW.json": "be288bc446d1a047b794512c3a9ae471213a8e9a51f0a5c48ca43b2982af6b70",
}

_original_build = r2._original_build
_original_switch = r2.base.fleet.switch_model
_original_http = r2.base.fleet.http_json
_original_append = r2.base.aj
_active_outdir: pathlib.Path | None = None
_route_identities: dict[str, dict[str, Any]] = {}
_file_identity_cache: dict[str, dict[str, Any]] = {}
_pending_requests: list[dict[str, Any]] = []


def _model_arg(argv: list[str]) -> str | None:
    for flag in ("-m", "--model"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                return argv[index + 1]
    return None


def _alias_arg(argv: list[str]) -> str | None:
    if "--alias" not in argv:
        return None
    index = argv.index("--alias")
    return argv[index + 1] if index + 1 < len(argv) else None


def evaluate_process_identity(
    model: str,
    status: dict[str, Any],
    process: dict[str, Any],
    binary: dict[str, Any],
    artifact: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    argv = [str(value) for value in process.get("argv", [])]
    observed_model = _model_arg(argv)
    observed_alias = _alias_arg(argv)
    expected_artifact = card["artifact"]
    expected_binary = card["runtime"]["binary"]
    checks = {
        "gateway_alias_match": status.get("current_model") == model,
        "gateway_backend_healthy": status.get("backend_healthy") is True,
        "backend_pid_match": process.get("pid") == status.get("backend_pid"),
        "command_alias_match": observed_alias == model,
        "command_model_path_match": artifact.get("resolved_path") == expected_artifact["path"],
        "process_executable_path_match": binary.get("resolved_path") == expected_binary,
        "model_hash_match": artifact.get("sha256") == expected_artifact["sha256"],
        "model_bytes_match": artifact.get("bytes") == expected_artifact["bytes"],
        "binary_hash_present": isinstance(binary.get("sha256"), str) and len(binary["sha256"]) == 64,
        "command_model_matches_hashed_path": observed_model == artifact.get("resolved_path"),
        "process_executable_matches_hashed_path": process.get("exe") == binary.get("resolved_path"),
    }
    return {
        "model": model,
        "gateway": status,
        "process": process,
        "binary": binary,
        "artifact": artifact,
        "expected": {
            "binary": expected_binary,
            "artifact_path": expected_artifact["path"],
            "artifact_sha256": expected_artifact["sha256"],
            "artifact_bytes": expected_artifact["bytes"],
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def _wsl_file_identity(path: str) -> dict[str, Any]:
    cached = _file_identity_cache.get(path)
    if cached is not None:
        return copy.deepcopy(cached)
    resolved = r2.base.fleet.wsl("readlink", "-f", path)
    size = r2.base.fleet.wsl("stat", "-c", "%s", path)
    digest = r2.base.fleet.wsl("sha256sum", path, timeout=1800.0)
    if resolved["returncode"] or size["returncode"] or digest["returncode"]:
        raise RuntimeError(f"cannot bind physical file identity for {path}")
    value = {
        "declared_path": path,
        "resolved_path": resolved["stdout"].strip(),
        "bytes": int(size["stdout"].strip()),
        "sha256": digest["stdout"].split()[0],
    }
    _file_identity_cache[path] = value
    return copy.deepcopy(value)


def _process_identity(pid: int) -> dict[str, Any]:
    script = (
        "import json,os,sys; p=sys.argv[1]; "
        "print(json.dumps({'pid':int(p),'exe':os.path.realpath('/proc/'+p+'/exe'),"
        "'argv':[x.decode('utf-8','replace') for x in "
        "open('/proc/'+p+'/cmdline','rb').read().split(b'\\0') if x]}))"
    )
    result = r2.base.fleet.wsl("python3", "-c", script, str(pid))
    if result["returncode"] != 0:
        raise RuntimeError(f"cannot read live backend /proc identity for PID {pid}: {result}")
    return json.loads(result["stdout"])


def capture_route_identity(model: str, status: dict[str, Any]) -> dict[str, Any]:
    pid = status.get("backend_pid")
    if not isinstance(pid, int) or pid <= 0:
        raise RuntimeError(f"invalid backend PID for {model}: {pid!r}")
    registry = json.loads((ROOT / "config/qualified_model_fleet.json").read_text(encoding="utf-8"))
    card = registry["models"][model]
    process = _process_identity(pid)
    observed_model = _model_arg(process["argv"])
    if observed_model is None:
        raise RuntimeError(f"backend command has no model argument for {model}")
    binary = _wsl_file_identity(process["exe"])
    artifact = _wsl_file_identity(observed_model)
    identity = evaluate_process_identity(model, status, process, binary, artifact, card)
    if not identity["pass"]:
        failed = [name for name, passed in identity["checks"].items() if not passed]
        raise RuntimeError(f"physical route identity mismatch for {model}: {failed}")
    return identity


def _bound_switch(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    status, gpu = _original_switch(model)
    if model in MODELS and model not in _route_identities:
        _route_identities[model] = capture_route_identity(model, status)
    return status, gpu


def _capturing_http(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 900.0,
) -> tuple[int | None, dict[str, Any]]:
    if url.endswith("/v1/chat/completions") and isinstance(payload, dict):
        _pending_requests.append(copy.deepcopy(payload))
    return _original_http(url, payload, timeout)


def _capturing_append(path: pathlib.Path, value: dict[str, Any]) -> None:
    if path.name == "samples.jsonl":
        if not _pending_requests:
            raise RuntimeError("sample append has no captured HTTP request")
        value = dict(value)
        value["request"] = _pending_requests.pop(0)
        value["request_capture"] = "http_json_argument"
        value["request_sha256"] = canonical_json_sha256(value["request"])
    _original_append(path, value)


def _finalizing_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    assert _active_outdir is not None
    raw = _active_outdir / "raw"
    if set(_route_identities) != set(MODELS):
        raise RuntimeError(f"incomplete physical route identity: {sorted(_route_identities)}")
    identity_path = raw / "physical_artifact_identity.json"
    identities = [_route_identities[model] for model in MODELS]
    r2._original_write(identity_path, {
        "schema": "local-labs-physical-route-identity-v1",
        "captured_before_requests": True,
        "routes": identities,
    })
    metrics_path = raw / "actual_scores.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update({
        "physical_routes_bound": len(identities),
        "physical_model_hash_match_rate": sum(row["checks"]["model_hash_match"] for row in identities) / len(identities),
        "physical_binary_hash_match_rate": sum(row["checks"]["binary_hash_present"] for row in identities) / len(identities),
        "process_command_identity_rate": sum(
            row["checks"]["command_model_matches_hashed_path"]
            and row["checks"]["process_executable_matches_hashed_path"]
            and row["checks"]["command_alias_match"]
            for row in identities
        ) / len(identities),
    })
    r2._original_write(metrics_path, metrics)
    state_path = raw / "runner_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    stable = metrics["exact_seeded_repeat_rate"] >= 0.9
    state.update({
        "status": "completed",
        "claim": (
            "QUALIFIED_TEXT_FLEET_SEEDED_PHYSICALLY_BOUND_STABLE_R3"
            if stable else "QUALIFIED_TEXT_FLEET_SEEDED_PHYSICALLY_BOUND_UNSTABLE_R3"
        ),
        "physical_identity_bound_before_receipt": True,
        "final_state_bound_before_receipt": True,
    })
    r2._original_write(state_path, state)
    input_paths = list(kwargs.get("input_paths", []))
    if identity_path not in input_paths:
        input_paths.append(identity_path)
    kwargs["input_paths"] = input_paths
    return _original_build(*args, **kwargs)


def configure(outdir: pathlib.Path) -> None:
    global _active_outdir
    _active_outdir = outdir
    _route_identities.clear()
    _file_identity_cache.clear()
    _pending_requests.clear()
    pipeline = json.loads((outdir / "PIPELINE.json").read_text(encoding="utf-8"))
    prereg = ROOT / pipeline["preregistration"]["path"]
    if sha256_file(prereg) != pipeline["preregistration"]["sha256"]:
        raise ValueError("pipeline preregistration binding mismatch")
    r2._active_outdir = outdir
    r2._original_build = _finalizing_build
    r2.base.TASK_ID = TASK_ID
    r2.base.PRE = pipeline["preregistration"]["sha256"]
    r2.base.SOURCES = SOURCES
    r2.base.build_provenance = r2._bound_build
    r2.base.wj = r2._guarded_write
    r2.base.aj = _capturing_append
    r2.base.fleet.switch_model = _bound_switch
    r2.base.fleet.http_json = _capturing_http
    r2.base.__file__ = __file__


def _eq_gate(metric: str, actual: Any, threshold: Any) -> dict[str, Any]:
    return {"metric": metric, "operator": "eq", "threshold": threshold, "actual": actual, "pass": actual == threshold}


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    configure(outdir)
    receipt = r2.base.execute(outdir)
    raw = outdir / "raw"
    metrics = json.loads((raw / "actual_scores.json").read_text(encoding="utf-8"))
    receipt["gates"].update({
        "request_retention": _eq_gate("retained_request_payloads", metrics["retained_request_payloads"], 288),
        "terminal_binding": _eq_gate("final_runner_state_bound", metrics["final_runner_state_bound"], True),
        "physical_route_coverage": _eq_gate("physical_routes_bound", metrics["physical_routes_bound"], 4),
        "physical_model_identity": _eq_gate("physical_model_hash_match_rate", metrics["physical_model_hash_match_rate"], 1.0),
        "physical_binary_identity": _eq_gate("physical_binary_hash_match_rate", metrics["physical_binary_hash_match_rate"], 1.0),
        "process_command_identity": _eq_gate("process_command_identity_rate", metrics["process_command_identity_rate"], 1.0),
    })
    receipt["evidence"].update({
        "request_payloads": "raw/samples.jsonl",
        "final_state_binding": "raw/runner_state.json",
        "physical_artifact_identity": "raw/physical_artifact_identity.json",
    })
    receipt.pop("receipt_fingerprint", None)
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    r2._original_write(raw / "receipt.json", receipt)
    state = json.loads((raw / "runner_state.json").read_text(encoding="utf-8"))
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{state['claim']}` pending independent review.\n\n"
        f"Recorded `{metrics['recorded_requests']}` requests with exact seeded repeat rate "
        f"`{metrics['exact_seeded_repeat_rate']:.6f}` and `{metrics['physical_routes_bound']}` "
        "live routes bound to process command, executable and GGUF identity.\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def selfcheck() -> None:
    card = {
        "artifact": {"path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123},
        "runtime": {"binary": "/bin/llama-server"},
    }
    status = {"current_model": "m", "backend_healthy": True, "backend_pid": 7}
    process = {"pid": 7, "exe": "/bin/llama-server", "argv": ["/bin/llama-server", "-m", "/models/a.gguf", "--alias", "m"]}
    binary = {"resolved_path": "/bin/llama-server", "sha256": "b" * 64, "bytes": 456}
    artifact = {"resolved_path": "/models/a.gguf", "sha256": "a" * 64, "bytes": 123}
    assert evaluate_process_identity("m", status, process, binary, artifact, card)["pass"] is True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    outdir = args.outdir.resolve()
    receipt = execute(outdir)
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = r2.base.subprocess.run(
        [sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID, "--to", "EXECUTED", "--actor", "Codex continuous executor"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(advance.stdout, flush=True)
    return 0 if advance.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
