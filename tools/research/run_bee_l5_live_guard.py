#!/usr/bin/env python3
"""Real-trace and live-stream audit of the historical BEE-L5 guard."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.analysis.reasoning_loop_guard import ReasoningLoopGuard

TASK_ID = "BACKLOG-BEE-L5-LIVE-GUARD-01"
BASE = "http://127.0.0.1:8080"
EXPECTED = {
    ROOT / "config/research_backlog_admissions/BACKLOG-BEE-L5-LIVE-GUARD-01.json": "27fd58b9f50ef2c3f687d628a7e1d1e113e407b2445ea6a8bb3edc70598139c5",
    ROOT / "runs/research/BACKLOG-BEE-L5-LIVE-GUARD-01/PRE_REGISTRATION.md": "122f9a436259697ff6aa26a5972cd7d15fe733ece6c89783a19bf4b19fb22754",
    ROOT / "runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/PRE_REGISTRATION.md": "22709ab07db66a4ba3506ab82dc5699f2bc4828b0e42c10882cf3d12f2a3690d",
    ROOT / "runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/RESULT.md": "e3a542796c62d543b82d873e1b9ed07ad423f09cf4ef06b51b030a92febda885",
    ROOT / "runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json": "3e279af76d4f357cc1a5fba1dff0892790d95f4813c63cc59373eee4973cd933",
    ROOT / "tools/analysis/reasoning_loop_guard.py": "a4bf8e7f29ff4cff5dbfefbb8c3accdfad24a9e43803ed8525aa0d44366e6358",
    ROOT / "tests/test_reasoning_loop_guard.py": "36f187c30dcd9557183eed2d316a90fc53573cbe601bf97bb1e3164f57833c44",
    ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json": "e545dfa0a35b97b00b72a1f1b32e35052a083b74d7bf8b26145ec6e87dcd102a",
}


def write(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post_json(path: str, payload: dict, timeout: int = 180):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def run_text(argv):
    done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=60)
    return {"argv": argv, "returncode": done.returncode, "stdout": done.stdout.strip(), "stderr": done.stderr.strip()}


def service_identity():
    show = run_text(["wsl", "-d", "Ubuntu-24.04", "--", "systemctl", "show", "llm-inference.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveState", "-p", "ExecStart", "--no-pager"])
    values = dict(line.split("=", 1) for line in show["stdout"].splitlines() if "=" in line)
    binary = values.get("ExecStart", "").split("path=", 1)[1].split(" ;", 1)[0] if "path=" in values.get("ExecStart", "") else ""
    digest = run_text(["wsl", "-d", "Ubuntu-24.04", "--", "sha256sum", binary]) if binary else {}
    health = {}
    for port, name in ((8080, "inference"), (8081, "embedding")):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
                health[name] = response.status
        except Exception:
            health[name] = None
    return {"systemd": values, "binary": binary, "binary_sha256": digest.get("stdout", "").split(" ", 1)[0], "health": health}


def slots():
    value = get_json(BASE + "/slots")
    return value if isinstance(value, list) else value.get("value", value)


def wait_idle(slot: int | None = None, timeout: float = 30.0):
    deadline = time.time() + timeout
    last = []
    while time.time() < deadline:
        last = slots()
        chosen = last if slot is None else [row for row in last if int(row.get("id", -1)) == slot]
        if chosen and all(not row.get("is_processing") for row in chosen):
            return last
        time.sleep(0.1)
    raise RuntimeError(f"slots failed to become idle: {last}")


def tokenize(text: str):
    response = post_json("/tokenize", {"content": text, "with_pieces": True})
    pieces = response.get("pieces") or []
    return [piece if isinstance(piece, str) else str(piece) for piece in pieces]


def guard_pieces(pieces):
    guard = ReasoningLoopGuard(window_size=32, max_reversals=3, max_ngram_reps=3)
    latencies = []
    for index, piece in enumerate(pieces, 1):
        started = time.perf_counter_ns()
        triggered, reason = guard.feed_token(piece)
        latencies.append((time.perf_counter_ns() - started) / 1000.0)
        if triggered:
            return True, reason, index, latencies
    return False, None, None, latencies


def stream_guard(payload):
    request = urllib.request.Request(BASE + "/completion", data=json.dumps({**payload, "stream": True}).encode(), headers={"Content-Type": "application/json"})
    chunks, latencies, reason, trigger_token = [], [], None, None
    response = urllib.request.urlopen(request, timeout=180)
    guard = ReasoningLoopGuard(window_size=32, max_reversals=3, max_ngram_reps=3)
    try:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            body = raw_line[6:].strip()
            if not body or body == b"[DONE]":
                continue
            event = json.loads(body.decode())
            piece = str(event.get("content") or "")
            if not piece:
                continue
            chunks.append({"piece": piece, "event": event})
            started = time.perf_counter_ns()
            fired, reason = guard.feed_token(piece)
            latencies.append((time.perf_counter_ns() - started) / 1000.0)
            if fired:
                trigger_token = len(chunks)
                response.close()
                break
    finally:
        response.close()
    return {"chunks": chunks, "triggered": trigger_token is not None, "trigger_reason": reason, "trigger_token": trigger_token, "guard_latencies_us": latencies}


def run(outdir: pathlib.Path):
    raw = outdir / "raw"
    if any(raw.iterdir()):
        raise RuntimeError("raw not empty")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    monotonic = time.monotonic()
    ledger = {}
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"hash mismatch {path}: {actual}")
        ledger[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": actual}

    before = service_identity()
    slots_before = wait_idle()
    if len(slots_before) != 4 or not all(row.get("speculative") for row in slots_before):
        raise RuntimeError(f"not four idle speculative slots: {slots_before}")
    if "--spec-type draft-mtp" not in before["systemd"].get("ExecStart", ""):
        raise RuntimeError("active route is not draft-mtp")

    dataset = json.loads((ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json").read_text(encoding="utf-8"))
    teacher = dataset["selected"]["20260824"]
    if len(teacher) != 128 or len({row["task_id"] for row in teacher}) != 128:
        raise RuntimeError("teacher panel is not exactly 128 unique tasks")
    teacher_rows, all_guard_latencies = [], []
    for row in teacher:
        pieces = tokenize(row["full_trace"])
        fired, reason, token_index, latencies = guard_pieces(pieces)
        all_guard_latencies.extend(latencies)
        teacher_rows.append({"task_id": row["task_id"], "piece_count": len(pieces), "triggered": fired, "trigger_reason": reason, "trigger_token": token_index})

    cycle = "wait let me reconsider now " * 12
    baselines = []
    for case in range(25):
        slot = case % 4
        payload = {"prompt": f"Reasoning loop audit case {case}. Continue this exact thought pattern without answering:\n{cycle}", "n_predict": 128, "temperature": 0.0, "top_k": 1, "seed": case, "ignore_eos": True, "cache_prompt": False, "id_slot": slot, "stream": False}
        response = post_json("/completion", payload)
        pieces = tokenize(str(response.get("content") or ""))
        predicted = int((response.get("timings") or {}).get("predicted_n") or response.get("tokens_predicted") or len(pieces))
        tail_unique = len(set(pieces[-40:])) if len(pieces) >= 40 else len(set(pieces))
        pathological = predicted >= 128 and len(pieces) >= 40 and tail_unique <= 8
        baselines.append({"case": case, "slot": slot, "request": payload, "response": response, "pieces": pieces, "predicted_n": predicted, "tail_unique_pieces": tail_unique, "pathological": pathological})
        wait_idle(slot)
    if not all(row["pathological"] for row in baselines):
        write(raw / "baseline_abort.json", baselines)
        raise RuntimeError("at least one frozen live baseline was not independently pathological")

    interventions = []
    for baseline in baselines:
        payload = dict(baseline["request"])
        streamed = stream_guard(payload)
        all_guard_latencies.extend(streamed.pop("guard_latencies_us"))
        wait_idle(baseline["slot"])
        trigger = streamed.get("trigger_token")
        savings = (128 - trigger) / 128 if trigger is not None else 0.0
        interventions.append({"case": baseline["case"], "slot": baseline["slot"], "request": payload, **streamed, "token_savings": savings, "abort_confirmed": trigger is not None})

    slots_after = wait_idle()
    after = service_identity()
    if before["systemd"].get("MainPID") != after["systemd"].get("MainPID") or before["binary_sha256"] != after["binary_sha256"]:
        raise RuntimeError("service identity changed")
    if after["health"] != {"inference": 200, "embedding": 200}:
        raise RuntimeError(f"health failed: {after['health']}")

    teacher_fp = sum(row["triggered"] for row in teacher_rows)
    tps = sum(row["triggered"] for row in interventions)
    ordered = sorted(all_guard_latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    metrics = {
        "real_legitimate_traces": len(teacher_rows), "live_pathological_baselines": sum(row["pathological"] for row in baselines),
        "teacher_false_positives": teacher_fp, "false_alarm_fpr": teacher_fp / len(teacher_rows),
        "live_true_positives": tps, "sensitivity_tpr": tps / len(interventions),
        "stream_aborts_confirmed": sum(row["abort_confirmed"] for row in interventions),
        "median_trigger_token": statistics.median(row["trigger_token"] for row in interventions if row["trigger_token"] is not None),
        "median_token_savings": statistics.median(row["token_savings"] for row in interventions),
        "guard_p95_us_per_token": p95, "guard_token_calls": len(all_guard_latencies),
        "service_restarts": int(after["systemd"].get("NRestarts") or -1), "idle_slots_after": sum(not row.get("is_processing") for row in slots_after),
    }
    with (raw / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in teacher_rows:
            stream.write(json.dumps({"kind": "real_teacher", **row}, ensure_ascii=False) + "\n")
        for row in baselines:
            stream.write(json.dumps({"kind": "live_baseline", **row}, ensure_ascii=False) + "\n")
        for row in interventions:
            stream.write(json.dumps({"kind": "live_intervention", **row}, ensure_ascii=False) + "\n")
    write(raw / "actual_scores.json", metrics)
    write(raw / "artifact_hashes.json", ledger)
    write(raw / "dataset_hashes.json", {"teacher_source_sha256": EXPECTED[ROOT / "runs/research/BACKLOG-ADAPT-TRACE-DISTILL-03/raw/teacher_samples.json"], "teacher_panel_semantic_sha256": canonical_json_sha256(teacher), "baseline_prompts_semantic_sha256": canonical_json_sha256([row["request"] for row in baselines])})
    write(raw / "effective_route.json", {"endpoint": BASE, "spec_type": "draft-mtp", "slots_before": slots_before, "exec_start": before["systemd"].get("ExecStart")})
    write(raw / "failure_reproduction.json", {"historical_inputs": "50 script-generated strings split on spaces", "successor_inputs": "128 real teacher traces tokenized by server plus 25 live completions", "historical_guard_unchanged": True})
    write(raw / "falsifiable_hypothesis.json", {"specificity_panel": 128, "pathological_panel": 25, "all_gates_required": True})
    write(raw / "hardware_metrics.json", {"guard_latencies_us": all_guard_latencies, "server_timings": [row["response"].get("timings") for row in baselines]})
    write(raw / "independent_evaluation.json", {"teacher_rows": teacher_rows, "intervention_summary": [{k: row[k] for k in ("case", "slot", "triggered", "trigger_reason", "trigger_token", "token_savings", "abort_confirmed")} for row in interventions]})
    write(raw / "invalidation_rules.json", {"baseline_pathology_required_before_intervention": True, "hash_mismatch_aborts": True, "service_identity_change_aborts": True})
    write(raw / "invariant_controls.json", {"guard": {"window_size": 32, "max_reversals": 3, "max_ngram_reps": 3}, "decode": {"n_predict": 128, "temperature": 0.0, "top_k": 1, "ignore_eos": True}})
    write(raw / "paired_baseline.json", {"baselines": [{"case": row["case"], "slot": row["slot"], "predicted_n": row["predicted_n"], "tail_unique_pieces": row["tail_unique_pieces"], "pathological": row["pathological"]} for row in baselines], "interventions": [{"case": row["case"], "trigger_token": row["trigger_token"], "token_savings": row["token_savings"]} for row in interventions]})
    write(raw / "real_implementation.json", {"teacher_tokenization": "active /tokenize exact pieces", "intervention": "client closes active SSE response at first historical guard trigger", "server_side_integration": False})
    write(raw / "recovery_state.json", {"slots_after": slots_after, "idle_slots": metrics["idle_slots_after"], "main_pid_unchanged": True})
    write(raw / "semantic_parity.json", {"final_answer_preservation_not_measured": True, "historical_guard_source_hash": EXPECTED[ROOT / "tools/analysis/reasoning_loop_guard.py"]})
    write(raw / "service_identity.json", {"before": before, "after": after})
    write(raw / "service_maintenance.json", {"service_untouched": True, "before": before, "after": after})
    write(raw / "source_execution_receipt.json", {"historical_receipt_sha256": EXPECTED[ROOT / "runs/research/BEE-L5-REASONING-LOOP-GUARD-2026-08-25/raw/receipt.json"]})

    defs = {"legitimate_coverage": ("real_legitimate_traces", "eq", 128), "pathology_coverage": ("live_pathological_baselines", "eq", 25), "sensitivity": ("sensitivity_tpr", "ge", .95), "specificity": ("false_alarm_fpr", "le", .02), "physical_intervention": ("stream_aborts_confirmed", "eq", 25), "token_savings": ("median_token_savings", "ge", .80), "guard_overhead": ("guard_p95_us_per_token", "le", 2.0), "service_integrity": ("service_restarts", "eq", 0), "idle_recovery": ("idle_slots_after", "eq", 4)}
    ops = {"eq": lambda a, b: a == b, "ge": lambda a, b: a >= b, "le": lambda a, b: a <= b}
    gates = {gate: {"metric": metric, "operator": op, "threshold": threshold, "actual": metrics[metric], "pass": ops[op](metrics[metric], threshold)} for gate, (metric, op, threshold) in defs.items()}
    evidence = {"acceptance_gates": "raw/receipt.json", "actual_scores": "raw/actual_scores.json", "artifact_hashes": "raw/artifact_hashes.json", "dataset_hashes": "raw/dataset_hashes.json", "effective_route": "raw/effective_route.json", "failure_reproduction": "raw/failure_reproduction.json", "falsifiable_hypothesis": "raw/falsifiable_hypothesis.json", "hardware_metrics": "raw/hardware_metrics.json", "independent_evaluation": "raw/independent_evaluation.json", "invalidation_rules": "raw/invalidation_rules.json", "invariant_controls": "raw/invariant_controls.json", "paired_baseline": "raw/paired_baseline.json", "provenance": "raw/receipt.json", "raw_samples": "raw/samples.jsonl", "real_implementation": "raw/real_implementation.json", "receipt_fingerprint": "raw/receipt.json", "recovery_state": "raw/recovery_state.json", "semantic_parity": "raw/semantic_parity.json", "service_identity": "raw/service_identity.json", "service_maintenance": "raw/service_maintenance.json", "source_execution_receipt": "raw/source_execution_receipt.json"}
    evidence_files = sorted({raw / value.removeprefix("raw/") for value in evidence.values() if value != "raw/receipt.json"})
    provenance = build_provenance(script_path=pathlib.Path(__file__).resolve(), started_at_utc=started_at, started_monotonic=monotonic, input_paths=[*EXPECTED, *evidence_files], packages=["pytest"], runtime={"execution_mode": "live_stream_abort", "endpoint": BASE, "service_identity": before})
    complete, errors = provenance_complete(provenance)
    if not complete:
        raise ValueError(errors)
    receipt = {"schema": "local-labs-backlog-receipt-v1", "task_id": TASK_ID, "provenance": provenance, "provenance_complete": True, "gates": gates, "evidence": evidence}
    receipt["receipt_fingerprint"] = canonical_json_sha256(receipt)
    write(raw / "receipt.json", receipt)
    journal = run_text(["wsl", "-d", "Ubuntu-24.04", "--", "journalctl", "-u", "llm-inference.service", "-n", "200", "--no-pager"])
    write(raw / "service_logs.json", journal)
    return receipt, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=pathlib.Path, default=ROOT / "runs/research" / TASK_ID)
    args = parser.parse_args()
    receipt, metrics = run(args.outdir.resolve())
    passed = all(row["pass"] for row in receipt["gates"].values())
    claim = "BEE_L5_LIVE_GUARD_QUALIFIED_R1" if passed else "BEE_L5_FALSE_POSITIVE_CONFIRMED_R1"
    failed = [gate for gate, row in receipt["gates"].items() if not row["pass"]]
    (args.outdir / "RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nReal teacher traces: {metrics['real_legitimate_traces']}; false alarms: {metrics['teacher_false_positives']} ({metrics['false_alarm_fpr']:.4%}). Live pathological baselines: {metrics['live_pathological_baselines']}; guard triggers/aborts: {metrics['stream_aborts_confirmed']}/25; median trigger token: {metrics['median_trigger_token']}; median savings: {metrics['median_token_savings']:.4%}; guard p95: {metrics['guard_p95_us_per_token']:.3f} us/token. Failed gates: {', '.join(failed) if failed else 'none'}. This is a client-side streaming intervention, not server integration.\n", encoding="utf-8")
    print(json.dumps({"claim": claim, "metrics": metrics, "gates": receipt["gates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
