#!/usr/bin/env python3
"""Forensic qualification of the retained SLX-03 CUDA build."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-SLX03-GDN-FUSION-BUILD-04"
DISTRO = "Ubuntu-24.04"
SOURCE = "/home/augus/src/slop.cpp-main"
BUILD = f"{SOURCE}/build-slx03-gdn-audit-03"
LIB = f"{BUILD}/bin/libggml-cuda.so"
BINARY = f"{BUILD}/bin/llama-server"
COMMIT = "87a416bd75d5a64e66e55846b779c0a54eca21bd"
NORMAL = "_Z28ggml_cuda_op_gated_delta_netR25ggml_backend_cuda_contextP11ggml_tensor"
FUSED = "_Z40ggml_cuda_op_gated_delta_net_fused_cacheR25ggml_backend_cuda_contextP11ggml_tensor37ggml_cuda_gated_delta_net_fused_cache"
DISPATCH = "_ZL18ggml_cuda_try_fuseP25ggml_backend_cuda_contextP11ggml_cgraphi"
INPUTS = {
    "config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-BUILD-04.json": "5d04d3899e3799a6737c6cc2964eb99967b23d54dd168956d1834a19c42abc2f",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-04/PRE_REGISTRATION.md": "b3258c898d14eabf541b10450d28634e6e90b853cd0adcf43a5ddbc670df6beb",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/receipt.json": "8e8ac1dc2451643861754a3aaa603d0fb11e6489230d4e5199b5b63de4af4219",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/build_receipts.json": "f4bf585a9525c6890e9a7ac1748aa7b92efcc46ee316dec687d9474374aa12eb",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/end_to_end_artifact.json": "80bbd6e0ea577966cac3ea5cba0328674dbec9f87769c46ff738a467840dd521",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/source_revision.json": "474d05f17dd4dc5ee65979fc9fa76df0148c0c0920bd3a22dc0fc94aa6d651c5",
    "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/REVIEW.json": "0dc76a9b411ef50ef0dd9cc2f38cb03af7b339e2052b402d30a445418e610b9a",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def cmd(*args: str, timeout: int = 300) -> dict:
    process = subprocess.run(["wsl.exe", "-d", DISTRO, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return {"argv": ["wsl.exe", "-d", DISTRO, *args], "returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}


def health(url: str):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status
    except Exception as error:
        return f"{type(error).__name__}:{error}"


def service() -> dict:
    result = cmd("systemctl", "show", "llm-inference.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveState", "-p", "SubState", "--no-pager", timeout=30)
    values = {"returncode": result["returncode"]}
    for line in result["stdout"].splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def snapshot_service() -> dict:
    return {"service": service(), "gateway": health("http://127.0.0.1:8080/health"), "embedding": health("http://127.0.0.1:8081/health")}


def run(outdir: pathlib.Path) -> None:
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    host_inputs = []
    for relative, expected in INPUTS.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen input mismatch: {relative}: {actual}")
        host_inputs.append(path)

    before = snapshot_service()
    head = cmd("git", "-C", SOURCE, "rev-parse", "HEAD", timeout=30)
    clean = cmd("git", "-C", SOURCE, "status", "--porcelain", "--untracked-files=no", timeout=30)
    source_ok = head["returncode"] == 0 and head["stdout"].strip() == COMMIT and clean["returncode"] == 0 and not clean["stdout"].strip()

    prior = json.loads((ROOT / "runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-03/raw/end_to_end_artifact.json").read_text(encoding="utf-8"))
    hashes = {}
    for name, path in (("llama_server", BINARY), ("libggml_cuda", LIB)):
        digest = cmd("sha256sum", path, timeout=120)
        size = cmd("stat", "-L", "-c", "%s", path, timeout=120)
        hashes[name] = {
            "path": path,
            "sha256": digest["stdout"].split()[0] if digest["returncode"] == 0 else None,
            "bytes": int(size["stdout"].strip()) if size["returncode"] == 0 else None,
        }
    hashes_match = all(hashes[name]["sha256"] == prior[name]["sha256"] for name in hashes)

    objects = cmd("cat", f"{BUILD}/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/objects1.rsp", timeout=120)
    linked_objects = objects["returncode"] == 0 and all(token in objects["stdout"] for token in ("gated_delta_net.cu.o", "ggml-cuda.cu.o"))
    symbols = cmd("nm", "-a", "--defined-only", LIB, timeout=180)
    symbol_lines = symbols["stdout"].splitlines()
    required_symbols = symbols["returncode"] == 0 and all(any(re.search(r"\bT\s+" + re.escape(name) + r"$", line) for line in symbol_lines) for name in (NORMAL, FUSED))

    symbol_sizes = cmd("nm", "-S", "-a", LIB, timeout=180)
    dispatcher = None
    for line in symbol_sizes["stdout"].splitlines():
        match = re.match(r"^([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+[tT]\s+" + re.escape(DISPATCH) + r"$", line.strip())
        if match:
            dispatcher = (int(match.group(1), 16), int(match.group(2), 16))
            break
    disassembly = {"argv": [], "returncode": -1, "stdout": "", "stderr": "dispatcher not uniquely resolved"}
    if dispatcher:
        start, size = dispatcher
        disassembly = cmd("objdump", "-d", f"--start-address=0x{start:x}", f"--stop-address=0x{start + size:x}", LIB, timeout=180)
    call_edge = disassembly["returncode"] == 0 and FUSED in disassembly["stdout"] and "call" in disassembly["stdout"]

    cubins = cmd("/usr/local/cuda/bin/cuobjdump", "--list-elf", LIB, timeout=300)
    cubin_lines = [line.strip() for line in cubins["stdout"].splitlines() if "ELF file" in line]
    sm86_only = cubins["returncode"] == 0 and bool(cubin_lines) and all(".sm_86.cubin" in line for line in cubin_lines)

    linkage = cmd("env", f"LD_LIBRARY_PATH={BUILD}/bin", "ldd", BINARY, timeout=120)
    project_lines = [line.strip() for line in linkage["stdout"].splitlines() if any(name in line for name in ("libllama", "libggml", "libmtmd", "libllama-common"))]
    own_linkage = linkage["returncode"] == 0 and bool(project_lines) and all((f"=> {BUILD}/bin/" in line) or line.startswith(f"{BUILD}/bin/") for line in project_lines)
    version = cmd("env", f"LD_LIBRARY_PATH={BUILD}/bin", BINARY, "--version", timeout=120)
    after = snapshot_service()
    service_same = before == after

    metrics = {
        "exact_source_commit": source_ok,
        "retained_artifact_hashes_match": hashes_match,
        "required_cuda_objects_linked": linked_objects,
        "required_gdn_symbols_defined": required_symbols,
        "dispatcher_calls_fused_cache": call_edge,
        "only_sm86_cubins_observed": sm86_only,
        "cuda_library_referent_bytes": hashes["libggml_cuda"]["bytes"],
        "project_libraries_resolve_to_retained_build": own_linkage,
        "server_version_exit": version["returncode"],
        "gateway_and_embedding_unchanged": service_same,
    }
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "source_revision.json", {"repository": SOURCE, "commit": head["stdout"].strip(), "expected": COMMIT, "tracked_status": clean["stdout"]})
    write_json(raw / "dependency_hashes.json", {"host_inputs": INPUTS})
    write_json(raw / "end_to_end_artifact.json", {"artifacts": hashes, "prior": prior})
    write_json(raw / "build_receipts.json", {"retained_build": BUILD, "objects_response": f"{BUILD}/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/objects1.rsp", "cubin_count": len(cubin_lines)})
    write_json(raw / "correctness_receipts.json", {"linked_objects": linked_objects, "project_lines": project_lines, "dispatcher": dispatcher, "symbol_names": [NORMAL, FUSED], "call_edge": call_edge, "cubin_lines": cubin_lines})
    write_json(raw / "independent_evaluation.json", {"build_only": True, "runtime_branch_not_observed": True, "metrics": metrics})
    write_json(raw / "service_before.json", before)
    write_json(raw / "service_after.json", after)
    (raw / "nm_defined.txt").write_text(symbols["stdout"], encoding="utf-8", newline="\n")
    (raw / "dispatcher_disassembly.txt").write_text(disassembly["stdout"], encoding="utf-8", newline="\n")
    (raw / "cuobjdump_list_elf.txt").write_text(cubins["stdout"], encoding="utf-8", newline="\n")

    definitions = {
        "source_revision": ("exact_source_commit", "eq", True), "artifact_identity": ("retained_artifact_hashes_match", "eq", True),
        "linked_objects": ("required_cuda_objects_linked", "eq", True), "elf_symbols": ("required_gdn_symbols_defined", "eq", True),
        "dispatcher_edge": ("dispatcher_calls_fused_cache", "eq", True), "sm86_code": ("only_sm86_cubins_observed", "eq", True),
        "dereferenced_size": ("cuda_library_referent_bytes", "ge", 60000000), "self_linkage": ("project_libraries_resolve_to_retained_build", "eq", True),
        "callability": ("server_version_exit", "eq", 0), "service_invariance": ("gateway_and_embedding_unchanged", "eq", True),
    }
    gates = {}
    for gate, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}

    evidence = {
        "acceptance_gates": "raw/receipt.json", "build_receipts": "raw/build_receipts.json", "correctness_receipts": "raw/correctness_receipts.json",
        "dependency_hashes": "raw/dependency_hashes.json", "end_to_end_artifact": "raw/end_to_end_artifact.json", "independent_evaluation": "raw/independent_evaluation.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/nm_defined.txt", "receipt_fingerprint": "raw/receipt.json", "source_revision": "raw/source_revision.json",
    }
    generated = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started, started_monotonic=monotonic, input_paths=[*host_inputs, *generated], packages=[], runtime={"execution_mode": "retained_build_forensics", "wsl_source": SOURCE, "wsl_build": BUILD, "model_loaded": False})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(errors)
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)
    failed = [name for name, gate in gates.items() if not gate["pass"]]
    claim = "SLX03_GDN_FUSION_BUILD_CALLABLE_R4" if not failed else "SLX03_GDN_FUSION_BUILD_NOT_CONFIRMED_R4"
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nRetained hashes match `{hashes_match}`; linked objects `{linked_objects}`; ELF symbols `{required_symbols}`; dispatcher call edge `{call_edge}`; SM86 cubins `{len(cubin_lines)}` only `{sm86_only}`; dereferenced CUDA library `{hashes['libggml_cuda']['bytes']}` bytes; self-linkage `{own_linkage}`; version exit `{version['returncode']}`; services unchanged `{service_same}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No runtime branch or performance claim is made.\n",
        encoding="utf-8", newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    run(args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
