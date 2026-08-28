#!/usr/bin/env python3
"""Fresh-process MTP persistence experiment for BACKLOG-MTP-PERSISTENCE-01."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.experiment_provenance import (
    build_provenance,
    canonical_json_sha256,
    provenance_complete,
    sha256_file,
)


TASK_ID = "BACKLOG-MTP-PERSISTENCE-01"
WSL_DISTRO = "Ubuntu-24.04"
PERSISTENT_UNIT = "llm-inference.service"
BASE_URL = "http://127.0.0.1:18080"
GATEWAY_URL = "http://127.0.0.1:8080"
EMBED_URL = "http://127.0.0.1:8081"
PORT = 18080
BINARY = "/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server"
LIB_DIR = "/home/augus/opt/slop.cpp/b10165-71676e46c/bin"
MODEL = "/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
SAVE_ROOT = "/home/augus/.local/state/tare-mtp-persistence/BACKLOG-MTP-PERSISTENCE-01"
PRE_REG_SHA256 = "72d726941ab2f0b38880955844aa2827e4f2669c7742eff6d0b9abe99e8005f6"
EXPECTED_WSL = {
    BINARY: {
        "bytes": 17920,
        "sha256": "efb2f06c19d26605a1934c0a9ed5b65dd69034e8765f2d29d0426b7a011cfbe2",
    },
    MODEL: {
        "bytes": 17923394624,
        "sha256": "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372",
    },
}
SOURCE_HASHES = {
    "runs/cache/LAB-CACHE-001-MTP-2026-08-22/RESULT.md": "5dd8cd202fafcd021a16eebdb4cbf9885d6ef080862179c5507619409a018c47",
    "docs/research/BLOCKER_REVALIDATION_2026-08-24.md": "d5888a2ee3c11711113fbc0a1596a00caa21cc31bfc553d9dd87cb0144aba658",
    "config/qualified_model_fleet.json": "042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82",
    "tools/probes/slot_save_restore_probe.py": "b24b5f3e2f4ef6bd8687763791a3c78d33ae91e9a129997aa4a16295b3cb81cd",
    "tools/probes/cache_correctness_v2.py": "4d23c2effc666912af5166085c1d9dd756a5be885cb5571a1af80345fc44a7fb",
}
SERVER_ARGS = [
    BINARY, "-m", MODEL, "--alias", "qwen38-mtp-persistence", "--host", "127.0.0.1",
    "--port", str(PORT), "--ctx-size", "32768", "--flash-attn", "on",
    "--gpu-layers", "all", "--metrics", "--jinja", "--no-mmproj",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0", "--parallel", "1",
    "--batch-size", "2048", "--ubatch-size", "512", "--ctx-checkpoints", "32",
]
MTP_ARGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
CONTROL_COUNT = 4
TREATMENT_ORDER = tuple(arm for _ in range(10) for arm in ("cold", "warm", "warm", "cold"))
EXPECTED_COUNT = CONTROL_COUNT + len(TREATMENT_ORDER)


def run_text(argv: list[str], timeout: float = 120.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
        return {
            "argv": argv, "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": repr(exc)}


def wsl(*args: str, root: bool = False, timeout: float = 120.0) -> dict[str, Any]:
    argv = ["wsl", "-d", WSL_DISTRO]
    if root:
        argv.extend(["-u", "root"])
    argv.extend(["--", *args])
    return run_text(argv, timeout=timeout)


def checked(result: dict[str, Any], description: str) -> dict[str, Any]:
    if result["returncode"] != 0:
        raise RuntimeError(f"{description} failed: {result}")
    return result


def write_json(path: pathlib.Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: pathlib.Path, value: object) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 900.0) -> tuple[int | None, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"User-Agent": "LocalLabs-MTP-Persistence/1.0"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"body": body[:4000]}
        return exc.code, {"_error": parsed}
    except Exception as exc:
        return None, {"_error": f"{type(exc).__name__}: {exc}"}


def health(url: str) -> tuple[int | None, dict[str, Any]]:
    return http_json(f"{url}/health", timeout=10.0)


def wait_health(url: str, timeout_seconds: float = 300.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: object = None
    while time.monotonic() < deadline:
        status, body = health(url)
        last = {"status": status, "body": body}
        if status == 200 and body.get("status") == "ok":
            return body
        time.sleep(1.0)
    raise RuntimeError(f"health timeout for {url}: {last}")


def parse_properties(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def unit_state(unit: str) -> dict[str, Any]:
    result = wsl(
        "systemctl", "show", unit, "-p", "LoadState", "-p", "ActiveState", "-p", "SubState",
        "-p", "MainPID", "-p", "NRestarts", "-p", "ExecStart", "--no-pager",
    )
    props = parse_properties(result["stdout"])
    return {
        "returncode": result["returncode"], "load_state": props.get("LoadState", ""),
        "active_state": props.get("ActiveState", ""), "sub_state": props.get("SubState", ""),
        "main_pid": int(props.get("MainPID") or 0), "n_restarts": int(props.get("NRestarts") or 0),
        "exec_start": props.get("ExecStart", ""), "stderr": result["stderr"],
    }


def process_values(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise ValueError("non-positive pid")
    argv = checked(wsl("xargs", "-0", "-n", "1", "-a", f"/proc/{pid}/cmdline"), "read argv")
    executable = checked(wsl("readlink", "-f", f"/proc/{pid}/exe"), "read executable")
    return {"pid": pid, "executable": executable["stdout"], "argv": argv["stdout"].splitlines()}


def wait_unit(unit: str, active: bool, timeout_seconds: float = 300.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = unit_state(unit)
    while time.monotonic() < deadline:
        last = unit_state(unit)
        if active and last["active_state"] == "active" and last["main_pid"] > 0:
            return last
        if not active and last["active_state"] in {"inactive", "failed", ""}:
            return last
        time.sleep(0.5)
    raise RuntimeError(f"unit {unit} did not reach active={active}: {last}")


def systemctl(action: str, unit: str) -> None:
    checked(wsl("systemctl", action, unit, root=True, timeout=180.0), f"systemctl {action} {unit}")


def gpu_state() -> dict[str, Any]:
    fields = "name,uuid,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw"
    result = checked(wsl("nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"), "GPU query")
    return dict(zip(fields.split(","), (part.strip() for part in result["stdout"].split(",")), strict=False))


def sha256_wsl(path: str, timeout: float = 1800.0) -> str:
    result = checked(wsl("sha256sum", path, timeout=timeout), f"hash {path}")
    return result["stdout"].split()[0].lower()


def stat_wsl(path: str) -> int:
    return int(checked(wsl("stat", "-c", "%s", path), f"stat {path}")["stdout"])


def verify_inputs() -> tuple[dict[str, Any], list[pathlib.Path]]:
    ledger: dict[str, Any] = {"host": {}, "wsl": {}}
    frozen_paths: list[pathlib.Path] = []
    for relative, expected in SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"host identity mismatch for {relative}: {actual}")
        ledger["host"][relative] = {"bytes": path.stat().st_size, "sha256": actual}
        frozen_paths.append(path)
    prereg = ROOT / "runs/research" / TASK_ID / "PRE_REGISTRATION.md"
    actual_prereg = sha256_file(prereg)
    if actual_prereg != PRE_REG_SHA256:
        raise ValueError(f"preregistration identity mismatch: {actual_prereg}")
    ledger["host"][str(prereg.relative_to(ROOT)).replace("\\", "/")] = {
        "bytes": prereg.stat().st_size, "sha256": actual_prereg,
    }
    frozen_paths.append(prereg)
    for path, expected in EXPECTED_WSL.items():
        size = stat_wsl(path)
        digest = sha256_wsl(path)
        if size != expected["bytes"]:
            raise ValueError(f"WSL size mismatch for {path}: {size}")
        if expected["sha256"] is not None and digest != expected["sha256"]:
            raise ValueError(f"WSL hash mismatch for {path}: {digest}")
        ledger["wsl"][path] = {"bytes": size, "sha256": digest}
    return ledger, frozen_paths


def gateway_status() -> dict[str, Any]:
    status, body = http_json(f"{GATEWAY_URL}/fleet/status", timeout=15.0)
    if status != 200 or body.get("status") != "ok" or not body.get("backend_healthy"):
        raise RuntimeError(f"gateway unhealthy: {status} {body}")
    return body


def restore_model(alias: str) -> dict[str, Any]:
    payload = {
        "model": alias, "messages": [{"role": "user", "content": "Reply with only OK."}],
        "max_tokens": 8, "temperature": 0.0, "seed": 20260826, "stream": False,
        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False},
    }
    status, response = http_json(f"{GATEWAY_URL}/v1/chat/completions", payload, timeout=900.0)
    if status != 200:
        raise RuntimeError(f"restore canary failed: {status} {response}")
    restored = gateway_status()
    if restored.get("current_model") != alias:
        raise RuntimeError(f"resident model not restored: {restored}")
    return restored


def observation_plan() -> list[dict[str, Any]]:
    plan = [{"index": index, "arm": "nospec", "block": None} for index in range(CONTROL_COUNT)]
    for offset, arm in enumerate(TREATMENT_ORDER):
        plan.append({"index": CONTROL_COUNT + offset, "arm": arm, "block": offset // 4})
    return plan


def start_observation(item: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    index = item["index"]
    unit = f"local-labs-mtp-persist-{index:02d}.service"
    save_dir = f"{SAVE_ROOT}/obs-{index:02d}"
    if unit_state(unit)["load_state"] != "not-found":
        raise RuntimeError(f"reserved transient unit exists: {unit}")
    checked(wsl("install", "-d", "-o", "augus", "-g", "augus", "-m", "0700", save_dir, root=True), "create save dir")
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--uid=augus", "--property=Type=simple",
        "--property=Restart=no", f"--setenv=LD_LIBRARY_PATH={LIB_DIR}", *SERVER_ARGS,
        "--slot-save-path", save_dir,
    ]
    if item["arm"] != "nospec":
        argv.extend(MTP_ARGS)
    launch = checked(wsl(*argv, root=True, timeout=60.0), f"launch {unit}")
    state = wait_unit(unit, active=True)
    process = process_values(state["main_pid"])
    if process["executable"] != BINARY or process["argv"][0] != BINARY:
        raise RuntimeError(f"unexpected transient identity: {process}")
    wait_health(BASE_URL)
    return unit, save_dir, {"launch": launch, "state": state, "process": process}


def action(name: str, filename: str | None = None) -> dict[str, Any]:
    payload = {} if filename is None else {"filename": filename}
    status, body = http_json(f"{BASE_URL}/slots/0?action={name}", payload)
    return {"http_status": status, "body": body}


def completion(prompt: str, n_predict: int = 64) -> dict[str, Any]:
    started = time.perf_counter()
    status, body = http_json(f"{BASE_URL}/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "seed": 20260826, "cache_prompt": True, "id_slot": 0, "stream": False,
    })
    return {
        "http_status": status, "wall_ms": round((time.perf_counter() - started) * 1000, 3),
        "content": str(body.get("content") or ""), "timings": body.get("timings") or {},
        "body": body,
    }


def normalize(value: str) -> str:
    return " ".join(value.strip().lower().split()).strip(" .,!?:;`\"'")


def run_observation(item: dict[str, Any], raw: pathlib.Path) -> dict[str, Any]:
    unit = ""
    save_dir = ""
    launch: dict[str, Any] = {}
    journal_text = ""
    cleanup: dict[str, Any] = {}
    try:
        unit, save_dir, launch = start_observation(item)
        priming = None
        if item["arm"] == "warm":
            priming = completion(
                "Priming lifecycle: continue this ordered phrase exactly: alpha beta gamma delta epsilon zeta eta theta. " * 8,
                n_predict=128,
            )
            accepted = int(priming["timings"].get("draft_n_accepted") or 0)
            if priming["http_status"] != 200 or accepted <= 0:
                raise RuntimeError(f"priming materiality failed: accepted={accepted}: {priming}")
            priming["discarded"] = True
            priming["erase"] = action("erase")

        shared = ("SLOT-FIRST-INSTANCE: routine state remains stable. " * 500) + "The persistent code word is MAGNOLIA."
        prompt = shared + "\nWhat is the persistent code word? Reply with ONLY the exact code word, with no explanation or punctuation."
        erased_before = action("erase")
        cold = completion(prompt)
        filename = f"mtp-persistence-{item['index']:02d}.bin"
        saved = action("save", filename)
        slot_path = f"{save_dir}/{filename}"
        slot_identity = {
            "path": slot_path,
            "bytes": stat_wsl(slot_path),
            "sha256": sha256_wsl(slot_path, timeout=600.0),
        }
        erased = action("erase")
        restored = action("restore", filename)
        warm = completion(prompt)
        erased_after = action("erase")

        saved_body = saved["body"]
        restored_body = restored["body"]
        lifecycle_ok = (
            all(step["http_status"] == 200 and "_error" not in step["body"] for step in (saved, erased, restored))
            and int(saved_body.get("n_saved") or 0) > 0
            and int(restored_body.get("n_restored") or 0) > 0
        )
        exact = cold["content"] == warm["content"]
        oracle = "magnolia" in normalize(cold["content"]) and "magnolia" in normalize(warm["content"])
        cache_n = int(warm["timings"].get("cache_n") or 0)
        passed = (
            lifecycle_ok and exact and oracle and cache_n > 0
            and cold["http_status"] == 200 and warm["http_status"] == 200
        )
        return {
            **item, "unit": unit, "save_dir": save_dir, "launch": launch,
            "gpu": gpu_state(), "priming": priming, "erased_before": erased_before,
            "cold": cold, "saved": saved, "slot_identity": slot_identity, "erased": erased,
            "restored": restored, "warm": warm, "erased_after": erased_after,
            "lifecycle_ok": lifecycle_ok, "exact_cold_restored": exact,
            "oracle_pass": oracle, "restored_cache_n": cache_n, "pass": passed,
        }
    finally:
        if unit:
            journal_text = wsl("journalctl", "-u", unit, "--no-pager", "-o", "short-iso", "-n", "5000", root=True)["stdout"]
            stop = wsl("systemctl", "stop", unit, root=True, timeout=180.0)
            try:
                wait_unit(unit, active=False, timeout_seconds=180.0)
            except RuntimeError:
                pass
            cleanup["stop"] = stop
        if save_dir:
            cleanup["remove"] = wsl("rm", "-rf", "--", save_dir, root=True, timeout=180.0)
        if unit:
            (raw / "logs" / f"{unit}.log").write_text(journal_text, encoding="utf-8")
        if cleanup:
            write_json(raw / "cleanup" / f"obs-{item['index']:02d}.json", cleanup)


def aggregate(rows: list[dict[str, Any]], service_recovered: bool, embedding_status: int | None) -> dict[str, Any]:
    controls = [row for row in rows if row["arm"] == "nospec"]
    cold = [row for row in rows if row["arm"] == "cold"]
    warm = [row for row in rows if row["arm"] == "warm"]
    all_counts = len(rows) == EXPECTED_COUNT and len(controls) == 4 and len(cold) == 20 and len(warm) == 20
    controls_pass = len(controls) == 4 and all(row["pass"] for row in controls)
    priming_material = len(warm) == 20 and all(int((row.get("priming") or {}).get("timings", {}).get("draft_n_accepted") or 0) > 0 for row in warm)
    original_failure = controls_pass and len(cold) == 20 and any(not row["pass"] for row in cold)
    fixed_successes = sum(bool(row["pass"]) for row in warm)
    warm_mismatches = sum(not bool(row["exact_cold_restored"]) for row in warm)
    return {
        "observations": len(rows), "control_observations": len(controls),
        "unprimed_observations": len(cold), "primed_observations": len(warm),
        "control_successes": sum(bool(row["pass"]) for row in controls),
        "unprimed_failures": sum(not bool(row["pass"]) for row in cold),
        "primed_failures": sum(not bool(row["pass"]) for row in warm),
        "original_failure_reproduced": original_failure,
        "successful_fixed_path_repeats": fixed_successes,
        "post_fix_mismatch_rate": round(warm_mismatches / len(warm), 8) if warm else 1.0,
        "invariant_controls_pass": all_counts and controls_pass and priming_material and service_recovered and embedding_status == 200,
        "service_recovered": service_recovered, "embedding_health": embedding_status,
        "median_slot_bytes": statistics.median(row["slot_identity"]["bytes"] for row in rows) if rows else None,
    }


def execute(outdir: pathlib.Path) -> dict[str, Any]:
    raw = outdir / "raw"
    (raw / "logs").mkdir(parents=True, exist_ok=True)
    (raw / "cleanup").mkdir(parents=True, exist_ok=True)
    samples_path = raw / "samples.jsonl"
    state_path = raw / "runner_state.json"
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_mono = time.monotonic()
    existing = read_jsonl(samples_path)
    completed = {int(row["index"]) for row in existing}

    frozen, frozen_paths = verify_inputs()
    write_json(raw / "frozen_inputs.json", frozen)
    initial_service = unit_state(PERSISTENT_UNIT)
    initial_gateway = gateway_status()
    initial_model = str(initial_gateway.get("current_model"))
    embed_status, embed_body = health(EMBED_URL)
    if initial_service["active_state"] != "active" or initial_service["main_pid"] <= 0:
        raise RuntimeError(f"persistent gateway service is not active: {initial_service}")
    if embed_status != 200 or embed_body.get("status") != "ok":
        raise RuntimeError(f"embedding endpoint unhealthy: {embed_status} {embed_body}")
    state = {
        "task_id": TASK_ID, "started_at_utc": started_utc, "status": "running",
        "initial_service": initial_service, "initial_gateway": initial_gateway,
        "initial_model": initial_model, "completed_observations": len(existing),
    }
    write_json(state_path, state)
    write_json(raw / "service_identity.json", {
        "initial_service": initial_service, "initial_gateway": initial_gateway,
        "initial_embedding": {"http_status": embed_status, "body": embed_body},
        "initial_gpu": gpu_state(),
    })

    restoration: dict[str, Any] = {}
    execution_error: str | None = None
    try:
        systemctl("stop", PERSISTENT_UNIT)
        wait_unit(PERSISTENT_UNIT, active=False)
        occupied_status, occupied_body = health(BASE_URL)
        if occupied_status is not None:
            raise RuntimeError(f"temporary endpoint remained occupied: {occupied_status} {occupied_body}")
        for item in observation_plan():
            if item["index"] in completed:
                continue
            row = run_observation(item, raw)
            append_jsonl(samples_path, row)
            existing.append(row)
            completed.add(item["index"])
            state.update({"completed_observations": len(existing), "last_index": item["index"], "last_arm": item["arm"], "last_pass": row["pass"]})
            write_json(state_path, state)
            print(
                f"{len(existing):02d}/{EXPECTED_COUNT} arm={item['arm']} pass={row['pass']} "
                f"exact={row['exact_cold_restored']} oracle={row['oracle_pass']} cache_n={row['restored_cache_n']}",
                flush=True,
            )
            if len(existing) % 4 == 0:
                boundary_status, _ = health(EMBED_URL)
                if boundary_status != 200:
                    raise RuntimeError(f"embedding unhealthy at observation boundary: {boundary_status}")
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
        state.update({"status": "aborted", "error": execution_error})
        write_json(state_path, state)
        raise
    finally:
        try:
            if unit_state(PERSISTENT_UNIT)["active_state"] != "active":
                systemctl("start", PERSISTENT_UNIT)
            restored_health = wait_health(GATEWAY_URL)
            restored_gateway = restore_model(initial_model)
            final_service = unit_state(PERSISTENT_UNIT)
            final_embed_status, final_embed_body = health(EMBED_URL)
            restoration = {
                "health": restored_health, "gateway": restored_gateway, "service": final_service,
                "embedding": {"http_status": final_embed_status, "body": final_embed_body},
                "initial_model_restored": restored_gateway.get("current_model") == initial_model,
            }
        except Exception as exc:
            restoration = {"error": f"{type(exc).__name__}: {exc}", "initial_model_restored": False}
            if execution_error is None:
                state.update({"status": "aborted", "error": restoration["error"]})
        write_json(raw / "recovery_state.json", restoration)
        state["restoration"] = restoration
        write_json(state_path, state)

    rows = read_jsonl(samples_path)
    service_recovered = bool(restoration.get("initial_model_restored")) and restoration.get("service", {}).get("active_state") == "active"
    final_embedding = restoration.get("embedding", {}).get("http_status")
    metrics = aggregate(rows, service_recovered, final_embedding)
    write_json(raw / "actual_scores.json", metrics)
    write_json(raw / "failure_reproduction.json", {
        "historical": "first MTP slot save/restore observation failed while fresh reruns passed",
        "unprimed_failures": metrics["unprimed_failures"],
        "original_failure_reproduced": metrics["original_failure_reproduced"],
    })
    write_json(raw / "falsifiable_hypothesis.json", {
        "cause": "persistence before one accepted-draft lifecycle in a fresh MTP process",
        "predicted": "at least one unprimed failure and zero primed failures",
        "falsified_if": ["zero unprimed failures", "any primed failure", "any no-spec control failure"],
    })
    write_json(raw / "invariant_controls.json", {
        "plan": observation_plan(), "counts_valid": metrics["observations"] == EXPECTED_COUNT,
        "decode": {"temperature": 0.0, "top_k": 1, "seed": 20260826, "cache_prompt": True, "slot": 0},
        "service_recovered": service_recovered, "embedding_health": final_embedding,
    })
    write_json(raw / "invalidation_rules.json", {
        "identity_mismatch_aborts": True, "priming_without_accepted_drafts_aborts": True,
        "service_restore_failure_invalidates": True, "oracle_mismatch_is_evidence_not_abort": True,
    })
    write_json(raw / "independent_evaluation.json", {"executor_rescore": metrics, "independent_review_pending": True})
    write_json(raw / "semantic_parity.json", {
        "primed_exact_mismatches": sum(not row["exact_cold_restored"] for row in rows if row["arm"] == "warm"),
        "post_fix_mismatch_rate": metrics["post_fix_mismatch_rate"],
    })
    write_json(raw / "hardware_metrics.json", {
        "slot_bytes": [row["slot_identity"]["bytes"] for row in rows],
        "wall_ms": {arm: [row["cold"]["wall_ms"] + row["warm"]["wall_ms"] for row in rows if row["arm"] == arm] for arm in ("nospec", "cold", "warm")},
    })
    write_json(raw / "service_maintenance.json", {
        "persistent_service_stopped_via_systemd": True, "embedding_service_stopped": False,
        "restoration": restoration,
    })

    definitions = {
        "original_failure": ("original_failure_reproduced", "eq", True),
        "controls": ("invariant_controls_pass", "eq", True),
        "fixed_repeats": ("successful_fixed_path_repeats", "ge", 20),
        "semantic_parity": ("post_fix_mismatch_rate", "eq", 0),
    }
    gates = {}
    for gate_id, (metric, operator, threshold) in definitions.items():
        actual = metrics[metric]
        passed = actual == threshold if operator == "eq" else actual >= threshold
        gates[gate_id] = {"metric": metric, "operator": operator, "threshold": threshold, "actual": actual, "pass": passed}

    evidence_files = sorted(path for path in raw.rglob("*") if path.is_file())
    provenance = build_provenance(
        script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_utc,
        started_monotonic=started_mono, input_paths=[*frozen_paths, *evidence_files], packages=[],
        runtime={"execution_mode": "fresh_systemd_units", "observations": len(rows), "restartable": True},
    )
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise RuntimeError(f"incomplete provenance: {errors}")
    evidence = {
        "acceptance_gates": "raw/receipt.json", "failure_reproduction": "raw/failure_reproduction.json",
        "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json", "independent_evaluation": "raw/independent_evaluation.json",
        "invalidation_rules": "raw/invalidation_rules.json", "invariant_controls": "raw/invariant_controls.json",
        "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl",
        "receipt_fingerprint": "raw/receipt.json", "semantic_parity": "raw/semantic_parity.json",
    }
    receipt = {
        "schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID,
        "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence,
    }
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write_json(raw / "receipt.json", receipt)

    passed = all(gate["pass"] for gate in gates.values())
    claim = "MTP_PERSISTENCE_ROOT_CAUSED" if passed else "MTP_PERSISTENCE_HYPOTHESIS_REJECTED"
    failed = [gate_id for gate_id, gate in gates.items() if not gate["pass"]]
    (outdir / "RESULT.md").write_text(
        f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Observed `{metrics['unprimed_failures']}` failures in 20 unprimed fresh MTP processes and "
        f"`{metrics['primed_failures']}` failures in 20 primed processes. The four no-spec controls had "
        f"`{metrics['control_successes']}/4` successes. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",
        encoding="utf-8", newline="\n",
    )
    state.update({"status": "completed", "claim": claim, "failed_gates": failed})
    write_json(state_path, state)
    return receipt


def selfcheck() -> None:
    plan = observation_plan()
    assert len(plan) == 44
    assert [item["arm"] for item in plan].count("nospec") == 4
    assert [item["arm"] for item in plan].count("cold") == 20
    assert [item["arm"] for item in plan].count("warm") == 20
    assert normalize(" `MAGNOLIA`. ") == "magnolia"
    print("MTP persistence first-instance self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    receipt = execute(args.outdir.resolve())
    print(json.dumps(receipt["gates"], indent=2), flush=True)
    advance = run_text([
        sys.executable, str(ROOT / "tools/analysis/backlog_pipeline.py"), "advance", TASK_ID,
        "--to", "EXECUTED", "--actor", "Codex executor",
    ])
    print(json.dumps({"pipeline_advance": advance}, indent=2), flush=True)
    return 0 if advance["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
