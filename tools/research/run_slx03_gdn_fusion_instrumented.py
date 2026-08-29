#!/usr/bin/env python3
"""Build the independently qualified SLX-03 source with a runtime marker."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete
from tools.research import run_slx03_gdn_fusion_build as base

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01"
BUILD = "/home/augus/src/slop.cpp-main/build-slx03-gdn-instrumented-01"
CUDA_PATH = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
INPUTS = {
    "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01.json": "35bd09fa15eb9156617bb9152afed14b9bbc48fa4e1d42fb1fe600bc0e2d81db",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-INSTRUMENTED-01/PRE_REGISTRATION.md": "063bb8fbe11c34887678ac8505c09b05dc8694ced24716a860feb3261239fb57",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/raw/receipt.json": "a46690e67b723368328a2f996d8b0d4e05e36d4c03e89590724561046e814029",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/REVIEW.json": "59bfad4ba63444b508b45908d772547846603accfeb9e0c7f3539280b79667ba",
}


def configured_source_command(*args: str, timeout: int = 1800):
    command = list(args)
    if len(command) >= 3 and command[0] == "cmake" and command[1:3] == ["-S", "."]:
        command.append("-DCMAKE_CUDA_FLAGS=-DGGML_CUDA_DEBUG")
    return base.cmd("--cd", base.SOURCE, "env", f"PATH={CUDA_PATH}", "CUDACXX=/usr/local/cuda/bin/nvcc", *command, timeout=timeout)


def execute(outdir: pathlib.Path) -> None:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    base.TASK_ID = TASK_ID
    base.BUILD = BUILD
    base.INPUTS = INPUTS
    base.at_source = configured_source_command
    base.run(outdir)

    raw = outdir / "raw"
    metrics_path = raw / "actual_scores.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cache = base.cmd("grep", "-F", "CMAKE_CUDA_FLAGS:STRING=-DGGML_CUDA_DEBUG", f"{BUILD}/CMakeCache.txt", timeout=60)
    size = base.cmd("stat", "-L", "-c", "%s", f"{BUILD}/bin/libggml-cuda.so", timeout=120)
    metrics["debug_define_recorded"] = cache["returncode"] == 0 and "CMAKE_CUDA_FLAGS:STRING=-DGGML_CUDA_DEBUG" in cache["stdout"]
    metrics["cuda_library_referent_bytes"] = int(size["stdout"].strip()) if size["returncode"] == 0 else None
    base.wj(metrics_path, metrics)

    end_artifact_path = raw / "end_to_end_artifact.json"
    artifacts = json.loads(end_artifact_path.read_text(encoding="utf-8"))
    artifacts["libggml_cuda"]["bytes"] = metrics["cuda_library_referent_bytes"]
    artifacts["libggml_cuda"]["size_semantics"] = "stat -L referent"
    base.wj(end_artifact_path, artifacts)
    correctness_path = raw / "correctness_receipts.json"
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    correctness["cmake_cuda_flags"] = cache
    correctness["debug_define_recorded"] = metrics["debug_define_recorded"]
    base.wj(correctness_path, correctness)

    definitions = {
        "source_revision": ("exact_source_commit", "eq", True), "tracked_clean": ("tracked_source_clean", "eq", True),
        "configure": ("cmake_configure_exit", "eq", 0), "build": ("llama_server_build_exit", "eq", 0),
        "debug_define": ("debug_define_recorded", "eq", True), "fusion_marker": ("gdn_fusion_marker_present", "eq", True),
        "self_linkage": ("project_libraries_resolve_to_new_build", "eq", True), "callability": ("server_version_exit", "eq", 0),
        "dereferenced_size": ("cuda_library_referent_bytes", "ge", 60000000), "service_invariance": ("gateway_and_embedding_unchanged", "eq", True),
    }
    gates = {}
    for gate, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}

    host_inputs = [ROOT / relative for relative in INPUTS]
    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file() and path.name != "receipt.json")
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*host_inputs, *evidence_files], packages=[], runtime={"execution_mode": "instrumented_release_cuda_build", "wsl_source": base.SOURCE, "wsl_build": BUILD, "cmake_cuda_flags": "-DGGML_CUDA_DEBUG", "model_loaded": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    evidence = {
        "acceptance_gates": "raw/receipt.json", "build_receipts": "raw/build_receipts.json", "correctness_receipts": "raw/correctness_receipts.json",
        "dependency_hashes": "raw/dependency_hashes.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "independent_evaluation": "raw/independent_evaluation.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/build.stdout.log", "receipt_fingerprint": "raw/receipt.json", "source_revision": "raw/source_revision.json",
    }
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    base.wj(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "SLX03_GDN_INSTRUMENTED_BUILD_CALLABLE_R1" if not failed else "SLX03_GDN_INSTRUMENTED_BUILD_NOT_CONFIRMED_R1"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nConfigure/build/version `{metrics['cmake_configure_exit']}/{metrics['llama_server_build_exit']}/{metrics['server_version_exit']}`; debug define `{metrics['debug_define_recorded']}`; marker `{metrics['gdn_fusion_marker_present']}`; own linkage `{metrics['project_libraries_resolve_to_new_build']}`; CUDA library `{metrics['cuda_library_referent_bytes']}` bytes; services unchanged `{metrics['gateway_and_embedding_unchanged']}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No model was loaded and no runtime or performance claim is made.\n",
        encoding="utf-8", newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    execute(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
