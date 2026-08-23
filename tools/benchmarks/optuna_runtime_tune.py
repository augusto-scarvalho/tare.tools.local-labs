#!/usr/bin/env python3
"""LAB-OPT-001: bounded Optuna search over safe Qwen3.8 runtime knobs.

The harness deliberately does not own the canonical systemd service or the SERVE/LAB
transition.  The operator stops/restores those outside this process so a harness crash
cannot silently mutate the deployment.  It owns only port 8092 and always releases it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

try:
    import optuna
except ImportError as exc:  # pragma: no cover - exercised by operator preflight
    raise SystemExit(
        "Optuna is required; run `python -m pip install -r requirements-experiments.txt`"
    ) from exc

from energy_phase_bench import run_rep


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "runs" / "optimization" / "LAB-OPT-001-2026-08-22"
DISTRO = "Ubuntu-24.04"
PORT = 8092
MODEL = "/home/augus/models/qwen38-27b/unsloth/Qwen3.8-27B-UD-Q4_K_XL.gguf"
SERVER_BIN = "/home/augus/src/slop.cpp/build/bin/llama-server"
INCUMBENT = (3, 2048)
GRID = tuple((draft_n, ubatch) for draft_n in (2, 3, 4) for ubatch in (1024, 2048))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
PROBES = (
    "Complete deterministically and briefly: The sum of 17 and 25 is",
    "Continue this delimiter-separated sequence with one item: alpha|beta|gamma|",
    "Complete this Python function correctly:\ndef add(a, b):\n    \"\"\"Return their sum.\"\"\"\n",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_wsl(argv: list[str], *, root: bool = False, timeout: float = 30) -> subprocess.CompletedProcess:
    cmd = ["wsl.exe", "-d", DISTRO]
    if root:
        cmd += ["-u", "root"]
    cmd += ["--", *argv]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, creationflags=NO_WINDOW)


def health(port: int, timeout: float = 3) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_health(port: int, timeout_s: float = 360) -> float | None:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        if health(port):
            return time.monotonic() - started
        time.sleep(1)
    return None


def free_vram_mib() -> float:
    proc = subprocess.run(
        ["nvidia-smi.exe", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        creationflags=NO_WINDOW,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise RuntimeError(f"nvidia-smi failed: {proc.stderr[:300]}")
    return float(proc.stdout.splitlines()[0].strip())


def available_host_gib() -> float:
    proc = run_wsl(["cat", "/proc/meminfo"])
    for line in proc.stdout.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def stop_candidate() -> None:
    run_wsl(["fuser", "-k", f"{PORT}/tcp"], root=True, timeout=30)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not health(PORT, timeout=1):
            return
        time.sleep(0.5)
    raise RuntimeError(f"candidate port {PORT} remained healthy after cleanup")


def server_argv(draft_n: int, ubatch: int, ctx_size: int = 32768) -> list[str]:
    return [
        SERVER_BIN, "-m", MODEL, "--alias", f"qwen38-opt-n{draft_n}-ub{ubatch}",
        "--host", "0.0.0.0", "--port", str(PORT), "--ctx-size", str(ctx_size),
        "--flash-attn", "on", "--gpu-layers", "all", "--metrics", "--jinja",
        "--no-mmproj", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n),
        "-np", "1", "--ctx-checkpoints", "32", "--batch-size", "2048",
        "--ubatch-size", str(ubatch),
    ]


def completion(prompt: str, n_predict: int = 64) -> dict:
    payload = {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "seed": 0, "cache_prompt": False, "stream": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    return {
        "content": result.get("content", ""),
        "content_sha256": hashlib.sha256(result.get("content", "").encode("utf-8")).hexdigest(),
        "tokens_predicted": int(result.get("tokens_predicted") or 0),
        "timings": result.get("timings") or {},
        "stop": result.get("stop"),
        "stop_type": result.get("stop_type"),
    }


def start_candidate(draft_n: int, ubatch: int, log_path: pathlib.Path,
                    ctx_size: int = 32768) -> tuple[subprocess.Popen, float]:
    if health(PORT):
        raise RuntimeError(f"port {PORT} is already occupied")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["wsl.exe", "-d", DISTRO, "--", *server_argv(draft_n, ubatch, ctx_size)],
        stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
        creationflags=NO_WINDOW,
    )
    proc._lab_opt_log_handle = log_handle  # type: ignore[attr-defined]
    load_s = wait_health(PORT)
    if load_s is None:
        stop_candidate()
        log_handle.close()
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"server never became healthy; log tail: {tail}")
    return proc, load_s


def finish_candidate(proc: subprocess.Popen) -> None:
    cleanup_error = None
    try:
        stop_candidate()
    except Exception as exc:  # preserve the original trial data, but cleanup remains fatal
        cleanup_error = exc
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)
    log_handle = getattr(proc, "_lab_opt_log_handle", None)
    if log_handle:
        log_handle.close()
    if cleanup_error:
        raise cleanup_error


def measure_cell(draft_n: int, ubatch: int, *, round_name: str, reps: int,
                 reference_hashes: list[str] | None, output: pathlib.Path,
                 ctx_size: int = 32768) -> dict:
    tag = f"n{draft_n}-ub{ubatch}"
    record: dict = {
        "config": tag, "draft_n": draft_n, "ubatch": ubatch, "round": round_name,
        "ctx_size": ctx_size, "started_at": utc_now(), "feasible": False, "errors": [],
    }
    proc = None
    wall_started = time.monotonic()
    try:
        proc, load_s = start_candidate(
            draft_n, ubatch, output / "logs" / f"{round_name}-{tag}.log", ctx_size
        )
        record["load_s"] = load_s
        record["free_vram_after_load_mib"] = free_vram_mib()
        if record["free_vram_after_load_mib"] < 4096:
            raise RuntimeError(
                f"VRAM reserve gate: {record['free_vram_after_load_mib']:.0f} MiB < 4096 MiB"
            )

        probes = [completion(prompt) for prompt in PROBES]
        record["probes"] = probes
        hashes = [item["content_sha256"] for item in probes]
        record["probe_hashes"] = hashes
        if any(item["tokens_predicted"] <= 0 for item in probes):
            raise RuntimeError("an equivalence probe returned no predicted tokens")
        if reference_hashes is not None and hashes != reference_hashes:
            mismatches = [index for index, pair in enumerate(zip(hashes, reference_hashes)) if pair[0] != pair[1]]
            raise RuntimeError(f"byte-equivalence gate failed on probe(s) {mismatches}")
        record["accepted_probes"] = len(probes)

        rows = []
        if round_name == "round1":
            row = run_rep(f"http://127.0.0.1:{PORT}", "short", 240, 96, 0.08)
            row["rep"] = 0
            rows.append(row)
        else:
            cells = (("short", 240), ("long", 1200))
            for rep in range(reps):
                ordered = cells if rep % 2 == 0 else tuple(reversed(cells))
                for cell, repeats in ordered:
                    row = run_rep(f"http://127.0.0.1:{PORT}", cell, repeats, 128, 0.08)
                    row["rep"] = rep
                    rows.append(row)
        if any(not row["boundaries_monotonic"] or row["telemetry_errors"]
               or row["predicted_tokens"] <= 0 for row in rows):
            raise RuntimeError("performance request or telemetry gate failed")
        record["runs"] = rows
        record["feasible"] = True
    except Exception as exc:  # partial receipts are first-class evidence
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        record["wall_s"] = time.monotonic() - wall_started
        if proc is not None:
            try:
                finish_candidate(proc)
            except Exception as exc:
                record["errors"].append(f"cleanup {type(exc).__name__}: {exc}")
                record["feasible"] = False

    record["finished_at"] = utc_now()
    if record["feasible"]:
        rows = record["runs"]
        record["metrics"] = summarize(rows, record)
    path = output / "trials" / f"{round_name}-{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def summarize(rows: list[dict], record: dict) -> dict:
    short = [row for row in rows if row["cell"] == "short"]
    long = [row for row in rows if row["cell"] == "long"]
    all_rows = short + long
    prompt_group = long or short
    return {
        "ttft_s_median": statistics.median(row["prefill_ttft_s"] for row in prompt_group),
        "prompt_tps_median": statistics.median(
            row["prompt_tokens"] / row["prefill_ttft_s"] for row in prompt_group
        ),
        "short_prompt_tps_median": statistics.median(
            row["prompt_tokens"] / row["prefill_ttft_s"] for row in short
        ),
        "long_prompt_tps_median": (statistics.median(
            row["prompt_tokens"] / row["prefill_ttft_s"] for row in long
        ) if long else None),
        "decode_s_per_token_median": statistics.median(
            row["decode_s"] / row["decode_tokens_after_first"] for row in all_rows
        ),
        "decode_tps_median": statistics.median(
            row["decode_tokens_after_first"] / row["decode_s"] for row in all_rows
        ),
        "vram_peak_mb": max(row["vram_peak_mb"] for row in all_rows),
        "cost_s_per_accepted_probe": record["wall_s"] / record["accepted_probes"],
        "power_peak_w": max(row["power_peak_active_w"] for row in all_rows),
        "temp_peak_c": max(row["temp_peak_c"] for row in all_rows),
    }


def objectives(record: dict) -> tuple[float, float, float, float]:
    if not record["feasible"]:
        return (1e9, 1e9, 1e9, 1e9)
    metrics = record["metrics"]
    return (metrics["ttft_s_median"], metrics["decode_s_per_token_median"],
            metrics["vram_peak_mb"], metrics["cost_s_per_accepted_probe"])


def nondominated(records: list[dict]) -> list[dict]:
    feasible = [record for record in records if record["feasible"]]
    result = []
    for candidate in feasible:
        values = objectives(candidate)
        dominated = any(
            all(a <= b for a, b in zip(objectives(other), values))
            and any(a < b for a, b in zip(objectives(other), values))
            for other in feasible if other is not candidate
        )
        if not dominated:
            result.append(candidate)
    return result


def rank_score(record: dict, feasible: list[dict]) -> int:
    return sum(
        sorted(feasible, key=lambda item: objectives(item)[axis]).index(record)
        for axis in range(4)
    )


def select_survivors(records: list[dict], limit: int = 3) -> list[dict]:
    feasible = [record for record in records if record["feasible"]]
    ordered = sorted(nondominated(feasible), key=lambda item: (rank_score(item, feasible), item["config"]))
    incumbent = next((item for item in feasible
                      if (item["draft_n"], item["ubatch"]) == INCUMBENT), None)
    if incumbent and incumbent not in ordered[:limit]:
        ordered = [incumbent, *[item for item in ordered if item is not incumbent]]
    for item in sorted(feasible, key=lambda value: (rank_score(value, feasible), value["config"])):
        if item not in ordered:
            ordered.append(item)
    return ordered[:limit]


def trial_dump(study: optuna.study.Study) -> list[dict]:
    return [{"number": trial.number, "state": trial.state.name, "params": trial.params,
             "values": trial.values, "user_attrs": trial.user_attrs}
            for trial in study.trials]


def promotion(round2: list[dict]) -> dict:
    incumbent = next((item for item in round2
                      if (item["draft_n"], item["ubatch"]) == INCUMBENT), None)
    if incumbent is None or not incumbent["feasible"]:
        return {"recommended": None, "reason": "incumbent missing or infeasible; no default change"}
    base = incumbent["metrics"]
    candidates = []
    for item in round2:
        if item is incumbent or not item["feasible"]:
            continue
        metrics = item["metrics"]
        prompt_gain = metrics["long_prompt_tps_median"] / base["long_prompt_tps_median"] - 1
        decode_gain = metrics["decode_tps_median"] / base["decode_tps_median"] - 1
        item["comparison_to_incumbent"] = {
            "long_prompt_tps_delta_pct": prompt_gain * 100,
            "decode_tps_delta_pct": decode_gain * 100,
        }
        if max(prompt_gain, decode_gain) >= 0.05 and min(prompt_gain, decode_gain) >= -0.03:
            candidates.append((max(prompt_gain, decode_gain), item))
    if not candidates:
        return {"recommended": None,
                "reason": "no candidate met >=5% gain on one axis and <=3% regression on the other"}
    winner = max(candidates, key=lambda pair: pair[0])[1]
    return {"recommended": winner["config"], "reason": "frozen bounded-screen rule met",
            "comparison": winner["comparison_to_incumbent"]}


def compare_pair(control: dict, challenger: dict) -> dict:
    if not control["feasible"] or not challenger["feasible"]:
        return {"confirmed": False, "reason": "control or challenger failed a hard gate"}
    base = control["metrics"]
    candidate = challenger["metrics"]
    prompt_gain = candidate["long_prompt_tps_median"] / base["long_prompt_tps_median"] - 1
    decode_gain = candidate["decode_tps_median"] / base["decode_tps_median"] - 1
    comparison = {"long_prompt_tps_delta_pct": prompt_gain * 100,
                  "decode_tps_delta_pct": decode_gain * 100}
    confirmed = max(prompt_gain, decode_gain) >= 0.05 and min(prompt_gain, decode_gain) >= -0.03
    return {"confirmed": confirmed,
            "reason": ("frozen >=5% gain / <=3% regression rule met" if confirmed else
                       "frozen gain/regression rule not met"),
            "comparison": comparison}


def confirm_live_default(output: pathlib.Path, reps: int) -> int:
    output.mkdir(parents=True, exist_ok=True)
    preflight()
    control = measure_cell(3, 512, round_name="confirm", reps=reps,
                           reference_hashes=None, output=output, ctx_size=131072)
    if not control["feasible"]:
        report = {"campaign": "LAB-OPT-001b", "timestamp": utc_now(),
                  "optuna_version": optuna.__version__, "qualified": False,
                  "method": {"control": "draft_n=3, explicit ubatch=512 (binary live default)",
                             "challenger": "draft_n=4, explicit ubatch=1024",
                             "ctx_size": 131072, "reps_per_short_long_cell": reps,
                             "hard_vram_free_floor_mib": 4096},
                  "control": control, "challenger": None,
                  "decision": {"confirmed": False,
                               "reason": "control failed a hard gate; challenger not launched"}}
        (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report["decision"], indent=2), flush=True)
        print(f"evidence={output / 'results.json'}", flush=True)
        return 1
    challenger = measure_cell(4, 1024, round_name="confirm", reps=reps,
                              reference_hashes=control["probe_hashes"], output=output,
                              ctx_size=131072)
    decision = compare_pair(control, challenger)
    report = {
        "campaign": "LAB-OPT-001b", "timestamp": utc_now(), "optuna_version": optuna.__version__,
        "qualified": control["feasible"] and challenger["feasible"],
        "method": {"control": "draft_n=3, explicit ubatch=512 (binary live default)",
                   "challenger": "draft_n=4, explicit ubatch=1024",
                   "ctx_size": 131072, "reps_per_short_long_cell": reps,
                   "hard_vram_free_floor_mib": 4096},
        "control": control, "challenger": challenger, "decision": decision,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"evidence={output / 'results.json'}", flush=True)
    return 0 if report["qualified"] else 1


def preflight() -> None:
    if not health(8081):
        raise RuntimeError("embedding endpoint 8081 is not healthy")
    if health(8080):
        raise RuntimeError("canonical endpoint 8080 is still live; stop it before LAB-OPT-001")
    if health(PORT):
        raise RuntimeError(f"candidate port {PORT} is occupied")
    host_gib = available_host_gib()
    if host_gib < 16:
        raise RuntimeError(f"host RAM preflight failed: {host_gib:.1f} GiB available")


def selfcheck() -> None:
    records = []
    for index, values in enumerate(((1, 4, 10, 8), (2, 2, 10, 7), (3, 5, 12, 9))):
        records.append({"config": str(index), "draft_n": 3 if index == 0 else index,
                        "ubatch": 2048 if index == 0 else 1024, "feasible": True,
                        "metrics": dict(zip(("ttft_s_median", "decode_s_per_token_median",
                                             "vram_peak_mb", "cost_s_per_accepted_probe"), values))})
    assert {item["config"] for item in nondominated(records)} == {"0", "1"}
    assert len(select_survivors(records, limit=2)) == 2
    print("LAB-OPT-001 self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--round2-reps", type=int, default=3)
    parser.add_argument("--confirm-live-default", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if args.confirm_live_default:
        default_confirm = ROOT / "runs" / "optimization" / "LAB-OPT-001b-2026-08-22"
        output = default_confirm if args.output == DEFAULT_OUTPUT else args.output
        return confirm_live_default(output, args.round2_reps)
    args.output.mkdir(parents=True, exist_ok=True)
    preflight()

    cache: dict[tuple[int, int], dict] = {}
    reference_hashes: list[str] | None = None
    baseline = measure_cell(*INCUMBENT, round_name="round1", reps=1,
                            reference_hashes=None, output=args.output)
    cache[INCUMBENT] = baseline
    if not baseline["feasible"]:
        raise RuntimeError(f"incumbent failed the round-1 gate: {baseline['errors']}")
    reference_hashes = baseline["probe_hashes"]

    sampler = optuna.samplers.GridSampler({"draft_n": [2, 3, 4], "ubatch": [1024, 2048]}, seed=0)
    stage1 = optuna.create_study(
        study_name="LAB-OPT-001-round1", directions=["minimize"] * 4, sampler=sampler
    )

    def stage1_objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        draft_n = trial.suggest_categorical("draft_n", [2, 3, 4])
        ubatch = trial.suggest_categorical("ubatch", [1024, 2048])
        key = (draft_n, ubatch)
        record = cache.get(key)
        if record is None:
            record = measure_cell(draft_n, ubatch, round_name="round1", reps=1,
                                  reference_hashes=reference_hashes, output=args.output)
            cache[key] = record
        trial.set_user_attr("config", record["config"])
        trial.set_user_attr("feasible", record["feasible"])
        trial.set_user_attr("errors", record["errors"])
        return objectives(record)

    stage1.optimize(stage1_objective, n_trials=len(GRID), show_progress_bar=False)
    round1 = [cache[key] for key in GRID]
    survivors = select_survivors(round1)
    if not any((item["draft_n"], item["ubatch"]) == INCUMBENT for item in survivors):
        raise RuntimeError("incumbent did not survive despite forced-control rule")

    survivor_names = [item["config"] for item in survivors]
    survivor_map = {item["config"]: item for item in survivors}
    stage2 = optuna.create_study(
        study_name="LAB-OPT-001-round2", directions=["minimize"] * 4,
        sampler=optuna.samplers.GridSampler({"config": survivor_names}, seed=0),
    )
    round2: dict[str, dict] = {}

    def stage2_objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
        name = trial.suggest_categorical("config", survivor_names)
        source = survivor_map[name]
        record = measure_cell(source["draft_n"], source["ubatch"], round_name="round2",
                              reps=args.round2_reps, reference_hashes=reference_hashes,
                              output=args.output)
        round2[name] = record
        trial.set_user_attr("draft_n", source["draft_n"])
        trial.set_user_attr("ubatch", source["ubatch"])
        trial.set_user_attr("feasible", record["feasible"])
        trial.set_user_attr("errors", record["errors"])
        return objectives(record)

    stage2.optimize(stage2_objective, n_trials=len(survivors), show_progress_bar=False)
    round2_records = [round2[name] for name in survivor_names]
    decision = promotion(round2_records)
    report = {
        "campaign": "LAB-OPT-001", "timestamp": utc_now(), "optuna_version": optuna.__version__,
        "qualified": all(item["feasible"] for item in round2_records),
        "method": {"space": {"draft_n": [2, 3, 4], "ubatch": [1024, 2048]},
                   "scheduler": "single-GPU deterministic successive halving",
                   "round1_cells": 6, "round2_survivors": len(survivors),
                   "round2_reps_per_short_long_cell": args.round2_reps,
                   "hard_vram_free_floor_mib": 4096,
                   "objectives": ["ttft_s", "decode_s_per_token", "vram_peak_mb",
                                  "cost_s_per_accepted_probe"]},
        "reference_probe_hashes": reference_hashes,
        "round1": round1, "survivors": survivor_names, "round2": round2_records,
        "pareto_round2": [item["config"] for item in nondominated(round2_records)],
        "decision": decision, "optuna_trials": {"round1": trial_dump(stage1),
                                                  "round2": trial_dump(stage2)},
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"survivors": survivor_names, "pareto": report["pareto_round2"],
                      "decision": decision}, indent=2), flush=True)
    print(f"evidence={args.output / 'results.json'}", flush=True)
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
