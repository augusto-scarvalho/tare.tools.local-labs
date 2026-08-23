#!/usr/bin/env python3
"""Qwen 3.8 evidence requalification: strict MQAR, token-calibrated NIAH, and GSM8K.

The 2026-08-20 preliminary scripts were useful scouts but not promotion-qualified instruments.
This runner preserves those artifacts and creates a new, source-bound campaign with:

* exact answer contracts (no substring scoring);
* deterministic, balanced fixtures and honest denominators;
* score-bearing dataset hashes and complete raw replies;
* llama-server/model/template/process identity captured before and after each run;
* exact chat-template token counts for long-context probes;
* incremental JSONL receipts so an interrupted run can resume without losing completed work.

No result is written into the legacy ``runs/qwen38-*`` paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.analysis.benchmark_qa import (  # noqa: E402
    benchmark_content_hash,
    lenient_last_number,
    numeric_equal,
    strict_exact_reply,
    strict_gsm8k_answer,
    wilson_interval,
)

SCHEMA_VERSION = "qwen38-requal-v1"
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_CAMPAIGN = ROOT / "runs" / "requalification" / "QWEN38-2026-08-20"
VALUE_RE = re.compile(r"^V\d{5}X$")
NIAH_CODE_RE = re.compile(r"^ZK-[A-F0-9]{12}-Q$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    digest = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "UNKNOWN"


def http_json(base_url: str, path: str, payload=None, *, timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {path}: {body[:500]}") from exc


def selected_server_identity(base_url: str, model_sha256: str | None) -> dict:
    props = http_json(base_url, "/props", timeout=15)
    models = http_json(base_url, "/v1/models", timeout=15)
    default = props.get("default_generation_settings", {})
    model_data = (models.get("data") or [{}])[0]
    template = props.get("chat_template") or ""
    process_argv = "UNKNOWN"
    try:
        port = urllib.parse.urlparse(base_url).port or 80
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-lc",
             f"ps -ww -eo args | grep '[l]lama-server' | grep -- '--port {port}' | head -1"],
            capture_output=True, text=True, timeout=15,
        )
        process_argv = proc.stdout.strip() or "UNKNOWN"
    except Exception:
        pass
    return {
        "captured_at": utc_now(),
        "base_url": base_url,
        "health": http_json(base_url, "/health", timeout=10),
        "model_alias": props.get("model_alias"),
        "model_path": props.get("model_path"),
        "model_sha256": model_sha256 or "NOT_COMPUTED",
        "build_info": props.get("build_info"),
        "n_ctx": default.get("n_ctx"),
        "total_slots": props.get("total_slots"),
        "default_params": default.get("params"),
        "model_meta": model_data.get("meta"),
        "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
        "chat_template_chars": len(template),
        "process_argv": process_argv,
    }


def identity_stable_view(identity: dict) -> dict:
    return {k: v for k, v in identity.items() if k != "captured_at"}


def write_json_atomic(path: pathlib.Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or not obj.get("row_id"):
            raise RuntimeError(f"invalid row at {path}:{line_no}")
        rows.append(obj)
    if len({r["row_id"] for r in rows}) != len(rows):
        raise RuntimeError(f"duplicate row_id in {path}")
    return rows


class CampaignRun:
    def __init__(self, outdir: pathlib.Path, kind: str, config: dict, identity: dict):
        self.outdir = outdir
        self.kind = kind
        self.config = config
        self.rows_path = outdir / f"{kind.upper()}_ROWS.jsonl"
        self.manifest_path = outdir / f"{kind.upper()}_MANIFEST.json"
        self.rows = load_jsonl(self.rows_path)
        self.done = {r["row_id"] for r in self.rows}
        source_path = pathlib.Path(__file__).resolve()
        expected = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "config": config,
            "source_sha256": sha256_file(source_path),
            "repo_head": git_head(),
            "server_identity_start": identity,
        }
        if self.manifest_path.exists():
            prior = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for field in ("schema_version", "kind", "config", "source_sha256"):
                if prior.get(field) != expected[field]:
                    raise RuntimeError(f"resume refused: manifest {field} differs")
            if identity_stable_view(prior["server_identity_start"]) != identity_stable_view(identity):
                raise RuntimeError("resume refused: live server identity differs from manifest")
            self.manifest = prior
        else:
            if self.rows:
                raise RuntimeError("rows exist without a manifest")
            self.manifest = dict(expected, started_at=utc_now(), status="running")
            write_json_atomic(self.manifest_path, self.manifest)

    def append(self, row: dict) -> None:
        row_id = row["row_id"]
        if row_id in self.done:
            return
        with self.rows_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        self.done.add(row_id)

    def finish(self, identity_after: dict, summary: dict) -> None:
        stable = identity_stable_view(self.manifest["server_identity_start"]) == identity_stable_view(identity_after)
        self.manifest.update(
            status="complete" if stable else "invalid_server_drift",
            completed_at=utc_now(),
            server_identity_after=identity_after,
            server_identity_stable=stable,
            row_count=len(self.rows),
            summary=summary,
        )
        write_json_atomic(self.manifest_path, self.manifest)
        if not stable:
            raise RuntimeError("server identity drifted during run; evidence marked invalid")


def apply_template_token_count(base_url: str, prompt: str,
                               chat_template_kwargs: dict | None = None) -> int:
    template_kwargs = chat_template_kwargs or {"enable_thinking": False}
    rendered = http_json(
        base_url,
        "/apply-template",
        {"messages": [{"role": "user", "content": prompt}],
         "chat_template_kwargs": template_kwargs,
         "add_generation_prompt": True},
        timeout=30,
    )
    tokenized = http_json(
        base_url, "/tokenize",
        {"content": rendered["prompt"], "add_special": False}, timeout=30,
    )
    return len(tokenized.get("tokens") or [])


def chat(base_url: str, prompt: str, *, max_tokens: int, timeout: float,
         chat_template_kwargs: dict | None = None) -> dict:
    template_kwargs = chat_template_kwargs or {"enable_thinking": False}
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": template_kwargs,
    }
    started = time.monotonic()
    response = http_json(base_url, "/v1/chat/completions", body, timeout=timeout)
    wall_s = time.monotonic() - started
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "response": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": response.get("usage") or {},
        "timings": response.get("timings") or {},
        "wall_s": wall_s,
        "response_id": response.get("id"),
    }


def proportion_summary(rows: list[dict], score_field: str) -> dict:
    n = len(rows)
    successes = sum(bool(row.get(score_field)) for row in rows)
    if not n:
        return {"n": 0, "successes": 0, "rate": None, "wilson95": None}
    lo, hi = wilson_interval(successes, n)
    return {"n": n, "successes": successes, "rate": successes / n,
            "wilson95": [lo, hi]}


def mqar_fixture(master_seed: int, pmax: int, depth: float, replicate: int) -> tuple[list[str], list[str]]:
    pool_size = max(4096, pmax * 2)
    rng = random.Random(stable_seed(master_seed, "mqar", depth, replicate))
    key_slots = rng.sample(range(pool_size), pmax)
    value_slots = rng.sample(range(pool_size), pmax)
    return ([f"K{i:05d}Q" for i in key_slots], [f"V{i:05d}X" for i in value_slots])


def mqar_prompt(keys: list[str], values: list[str], dose: int, depth: float) -> tuple[str, str, int]:
    probe_index = min(dose - 1, max(0, int(round(depth * (dose - 1)))))
    lines = ["Memorize this key-value mapping:"]
    lines.extend(f"{key} = {value}" for key, value in zip(keys[:dose], values[:dose]))
    lines.append("")
    lines.append(f"What value is assigned to {keys[probe_index]}?")
    lines.append("Reply with ONLY the exact value token, with no prose or punctuation.")
    return "\n".join(lines), values[probe_index], probe_index


def run_mqar(args) -> None:
    identity = selected_server_identity(args.base_url, args.model_sha256)
    config = {"doses": args.doses, "depths": args.depths, "replicates": args.replicates,
              "master_seed": args.seed, "max_tokens": 12, "scorer": "strict_exact_reply",
              "ordering": "block-alternating-forward-reverse"}
    run = CampaignRun(args.outdir, "mqar", config, identity)
    n_ctx = int(identity.get("n_ctx") or 0)
    pmax = max(args.doses)
    blocks = [(depth, rep) for depth in args.depths for rep in range(args.replicates)]
    for block_index, (depth, rep) in enumerate(blocks):
        keys, values = mqar_fixture(args.seed, pmax, depth, rep)
        doses = args.doses if block_index % 2 == 0 else list(reversed(args.doses))
        for dose in doses:
            row_id = f"mqar-depth{depth:.2f}-rep{rep:02d}-p{dose:05d}"
            if row_id in run.done:
                continue
            prompt, expected, probe_index = mqar_prompt(keys, values, dose, depth)
            preflight_tokens = apply_template_token_count(args.base_url, prompt)
            if n_ctx and preflight_tokens + 64 >= n_ctx:
                raise RuntimeError(f"{row_id}: {preflight_tokens} prompt tokens exceeds safe n_ctx={n_ctx}")
            result = chat(args.base_url, prompt, max_tokens=12, timeout=args.timeout)
            response = result["response"].strip()
            row = {
                "schema_version": SCHEMA_VERSION, "row_id": row_id, "kind": "mqar",
                "dose_pairs": dose, "depth": depth, "replicate": rep,
                "probe_index": probe_index, "expected": expected, "response": result["response"],
                "exact": strict_exact_reply(result["response"], expected),
                "format_ok": bool(VALUE_RE.fullmatch(response)),
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "preflight_prompt_tokens": preflight_tokens,
                "finish_reason": result["finish_reason"], "usage": result["usage"],
                "timings": result["timings"], "wall_s": result["wall_s"],
                "recorded_at": utc_now(),
            }
            run.append(row)
            print(f"[MQAR] p={dose:4d} depth={depth:.2f} rep={rep} exact={row['exact']} "
                  f"tokens={preflight_tokens}", flush=True)
    by_dose = {}
    for dose in args.doses:
        dose_rows = [row for row in run.rows if row["dose_pairs"] == dose]
        by_dose[str(dose)] = {
            "exact": proportion_summary(dose_rows, "exact"),
            "format": proportion_summary(dose_rows, "format_ok"),
            "actual_prompt_tokens": sorted({row["preflight_prompt_tokens"] for row in dose_rows}),
        }
    summary = {"primary_metric": "strict exact recall", "by_dose": by_dose,
               "all": proportion_summary(run.rows, "exact")}
    write_json_atomic(args.outdir / "MQAR_RESULTS.json", summary)
    run.finish(selected_server_identity(args.base_url, args.model_sha256), summary)


NIAH_WORDS = (
    "amber birch cobalt delta elm frost granite harbor ivory juniper kinetic linen maple "
    "nickel ochre pine quartz river slate timber umber violet willow xenon yarrow zephyr"
).split()


def niah_units(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    units = []
    for index in range(count):
        words = rng.sample(NIAH_WORDS, 7)
        ref = rng.randrange(100000, 999999)
        units.append(f"Record {index:05d}: {' '.join(words)}; routine status nominal; reference {ref}.")
    return units


def niah_prompt(units: list[str], depth: float, code: str | None) -> str:
    material = list(units)
    if code is not None:
        position = min(len(material), max(0, int(round(depth * len(material)))))
        material.insert(position, f"AUTHORITATIVE FACT: the Grover facility access code is {code}.")
    body = "\n".join(material)
    return (
        body + "\n\nQuestion: What is the Grover facility access code? "
        "If and only if an AUTHORITATIVE FACT above states it, reply with ONLY that exact code. "
        "Otherwise reply with ONLY NOT_PRESENT."
    )


def calibrate_niah_units(base_url: str, target: int, seed: int, depth: float,
                         code: str | None) -> tuple[list[str], int]:
    # Build one deterministic superset, then binary-search the largest prefix at or below target.
    max_units = max(512, target // 6)
    units = niah_units(seed, max_units)
    low, high, best_count, best_tokens = 0, len(units), 0, 0
    while low <= high:
        mid = (low + high) // 2
        tokens = apply_template_token_count(base_url, niah_prompt(units[:mid], depth, code))
        if tokens <= target:
            best_count, best_tokens = mid, tokens
            low = mid + 1
        else:
            high = mid - 1
    return units[:best_count], best_tokens


def run_niah(args) -> None:
    identity = selected_server_identity(args.base_url, args.model_sha256)
    config = {"target_prompt_tokens": args.targets, "depths": args.depths,
              "positive_replicates": args.replicates, "negative_replicates": args.negative_replicates,
              "master_seed": args.seed, "max_tokens": 24, "scorer": "strict_exact_reply",
              "token_calibration": "apply-template then tokenize, largest prefix <= target"}
    run = CampaignRun(args.outdir, "niah", config, identity)
    n_ctx = int(identity.get("n_ctx") or 0)
    cells = []
    for target_index, target in enumerate(args.targets):
        for depth in args.depths:
            for rep in range(args.replicates):
                cells.append((target, depth, rep, True))
        for rep in range(args.negative_replicates):
            cells.append((target, 0.5, rep, False))
        if target_index % 2:
            cells[-(len(args.depths) * args.replicates + args.negative_replicates):] = reversed(
                cells[-(len(args.depths) * args.replicates + args.negative_replicates):])
    for target, depth, rep, positive in cells:
        polarity = "pos" if positive else "neg"
        row_id = f"niah-t{target:05d}-d{depth:.2f}-r{rep:02d}-{polarity}"
        if row_id in run.done:
            continue
        seed = stable_seed(args.seed, row_id)
        code = f"ZK-{sha256_bytes(str(seed).encode())[:12].upper()}-Q" if positive else None
        units, calibrated_tokens = calibrate_niah_units(args.base_url, target, seed, depth, code)
        prompt = niah_prompt(units, depth, code)
        preflight_tokens = apply_template_token_count(args.base_url, prompt)
        if preflight_tokens != calibrated_tokens:
            raise RuntimeError(f"{row_id}: token calibration was not deterministic")
        if target - preflight_tokens > args.tolerance:
            raise RuntimeError(f"{row_id}: token target miss {target-preflight_tokens} > {args.tolerance}")
        if n_ctx and preflight_tokens + 96 >= n_ctx:
            raise RuntimeError(f"{row_id}: {preflight_tokens} prompt tokens exceeds safe n_ctx={n_ctx}")
        expected = code if positive else "NOT_PRESENT"
        result = chat(args.base_url, prompt, max_tokens=24, timeout=args.timeout)
        response = result["response"].strip()
        row = {
            "schema_version": SCHEMA_VERSION, "row_id": row_id, "kind": "niah",
            "target_prompt_tokens": target, "actual_prompt_tokens_preflight": preflight_tokens,
            "depth": depth if positive else None, "replicate": rep, "positive": positive,
            "expected": expected, "response": result["response"],
            "exact": strict_exact_reply(result["response"], expected),
            "format_ok": bool(NIAH_CODE_RE.fullmatch(response)) if positive else response == "NOT_PRESENT",
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")), "unit_count": len(units),
            "finish_reason": result["finish_reason"], "usage": result["usage"],
            "timings": result["timings"], "wall_s": result["wall_s"], "recorded_at": utc_now(),
        }
        run.append(row)
        print(f"[NIAH] target={target:5d} actual={preflight_tokens:5d} {polarity} "
              f"depth={depth:.2f} rep={rep} exact={row['exact']}", flush=True)
    by_target = {}
    for target in args.targets:
        positive_rows = [r for r in run.rows if r["target_prompt_tokens"] == target and r["positive"]]
        negative_rows = [r for r in run.rows if r["target_prompt_tokens"] == target and not r["positive"]]
        by_depth = {}
        for depth in args.depths:
            depth_rows = [r for r in positive_rows if r["depth"] == depth]
            by_depth[f"{depth:.2f}"] = proportion_summary(depth_rows, "exact")
        by_target[str(target)] = {
            "positive_exact": proportion_summary(positive_rows, "exact"),
            "negative_control_exact": proportion_summary(negative_rows, "exact"),
            "by_depth": by_depth,
            "actual_prompt_tokens": sorted({r["actual_prompt_tokens_preflight"] for r in positive_rows + negative_rows}),
        }
    summary = {"primary_metric": "strict exact retrieval", "by_target": by_target,
               "positive_all": proportion_summary([r for r in run.rows if r["positive"]], "exact"),
               "negative_all": proportion_summary([r for r in run.rows if not r["positive"]], "exact")}
    write_json_atomic(args.outdir / "NIAH_RESULTS.json", summary)
    run.finish(selected_server_identity(args.base_url, args.model_sha256), summary)


def load_gsm8k(path: pathlib.Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(key in row for key in ("task_id", "prompt", "answer")):
            raise RuntimeError(f"invalid GSM8K row {line_no}")
        rows.append({key: row[key] for key in ("task_id", "prompt", "answer")})
    if len({r["task_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate GSM8K task_id")
    return rows


def gsm8k_prompt(problem: str) -> str:
    return (
        "Solve the math problem. Show concise work. The final non-empty line MUST be exactly "
        "`#### <number>` with no text after it.\n\n" + problem
    )


def run_gsm8k(args) -> None:
    dataset = args.dataset.resolve()
    all_problems = load_gsm8k(dataset)
    if args.task_id:
        requested = set(args.task_id)
        chosen = [problem for problem in all_problems if problem["task_id"] in requested]
        missing = requested - {problem["task_id"] for problem in chosen}
        if missing:
            raise ValueError(f"unknown --task-id values: {sorted(missing)}")
    else:
        order = list(range(len(all_problems)))
        random.Random(args.seed).shuffle(order)
        chosen = [all_problems[index] for index in order[:args.subset]]
    identity = selected_server_identity(args.base_url, args.model_sha256)
    config = {
        "dataset_path": str(dataset.relative_to(ROOT)),
        "dataset_file_sha256": sha256_file(dataset),
        "dataset_content_sha256": benchmark_content_hash(all_problems),
        "dataset_n": len(all_problems), "subset": args.subset, "subset_seed": args.seed,
        "subset_task_ids": [p["task_id"] for p in chosen], "max_tokens": args.max_tokens,
        "reasoning_strength": args.reasoning_strength,
        "primary_scorer": "strict final non-empty line: #### <number>",
        "lenient_scorer": "diagnostic only: last numeric token",
    }
    run = CampaignRun(args.outdir, "gsm8k", config, identity)
    for ordinal, problem in enumerate(chosen):
        row_id = f"gsm8k-{problem['task_id'].replace('/', '-')}"
        if row_id in run.done:
            continue
        prompt = gsm8k_prompt(problem["prompt"])
        if args.reasoning_strength == "off":
            template_kwargs = {"enable_thinking": False}
        elif args.reasoning_strength:
            template_kwargs = {"reasoning_strength": args.reasoning_strength,
                               "reasoning_effort": args.reasoning_strength}
        else:
            template_kwargs = None
        result = chat(args.base_url, prompt, max_tokens=args.max_tokens, timeout=args.timeout,
                      chat_template_kwargs=template_kwargs)
        strict_pred = strict_gsm8k_answer(result["response"])
        lenient_pred = lenient_last_number(result["response"])
        row = {
            "schema_version": SCHEMA_VERSION, "row_id": row_id, "kind": "gsm8k",
            "ordinal": ordinal, "task_id": problem["task_id"], "gold": str(problem["answer"]),
            "response": result["response"], "reasoning_content": result["reasoning_content"],
            "strict_pred": strict_pred, "lenient_pred": lenient_pred,
            "strict_correct": numeric_equal(strict_pred, str(problem["answer"])),
            "lenient_correct": numeric_equal(lenient_pred, str(problem["answer"])),
            "format_ok": strict_pred is not None,
            "truncated": result["finish_reason"] == "length",
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "finish_reason": result["finish_reason"], "usage": result["usage"],
            "timings": result["timings"], "wall_s": result["wall_s"], "recorded_at": utc_now(),
        }
        run.append(row)
        print(f"[GSM8K] {ordinal+1:03d}/{len(chosen)} {problem['task_id']} "
              f"strict={row['strict_correct']} format={row['format_ok']}", flush=True)
    summary = {
        "primary_metric": "strict numeric accuracy",
        "strict_accuracy": proportion_summary(run.rows, "strict_correct"),
        "lenient_accuracy_diagnostic": proportion_summary(run.rows, "lenient_correct"),
        "format_adherence": proportion_summary(run.rows, "format_ok"),
        "truncation": proportion_summary(run.rows, "truncated"),
    }
    write_json_atomic(args.outdir / "GSM8K_RESULTS.json", summary)
    run.finish(selected_server_identity(args.base_url, args.model_sha256), summary)


def selfcheck() -> None:
    keys, values = mqar_fixture(123, 32, 0.5, 0)
    prompt, expected, index = mqar_prompt(keys, values, 16, 0.5)
    assert expected == values[index] and f"{keys[index]}?" in prompt
    assert strict_exact_reply(expected, expected)
    assert not strict_exact_reply(expected + "0", expected)
    assert VALUE_RE.fullmatch(expected)
    positive = niah_prompt(["Record 1: nominal."], 0.5, "ZK-ABCDEF123456-Q")
    negative = niah_prompt(["Record 1: nominal."], 0.5, None)
    assert "access code is ZK-ABCDEF123456-Q" in positive
    assert "access code is ZK-ABCDEF123456-Q" not in negative
    problems = [{"task_id": "gsm8k/0", "prompt": "one plus one", "answer": "2"}]
    assert benchmark_content_hash(problems) != benchmark_content_hash([
        {"task_id": "gsm8k/0", "prompt": "one plus one", "answer": "3"}])
    print("qwen38_requal selfcheck: PASS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    sub = parser.add_subparsers(dest="command")

    def common(p, name):
        p.add_argument("--base-url", default=DEFAULT_BASE_URL)
        p.add_argument("--model-sha256", default="")
        p.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_CAMPAIGN / name)
        p.add_argument("--timeout", type=float, default=600.0)
        p.add_argument("--seed", type=int, default=20260820)

    mqar = sub.add_parser("mqar")
    common(mqar, "mqar")
    mqar.add_argument("--doses", type=int, nargs="+", default=[4, 32, 128, 512, 1024, 1792])
    mqar.add_argument("--depths", type=float, nargs="+", default=[0.10, 0.25, 0.50, 0.75, 0.90])
    mqar.add_argument("--replicates", type=int, default=8)

    niah = sub.add_parser("niah")
    common(niah, "niah")
    niah.add_argument("--targets", type=int, nargs="+", default=[4096, 8192, 16384, 24576, 30000])
    niah.add_argument("--depths", type=float, nargs="+", default=[0.10, 0.50, 0.75, 0.90])
    niah.add_argument("--replicates", type=int, default=3)
    niah.add_argument("--negative-replicates", type=int, default=2)
    niah.add_argument("--tolerance", type=int, default=64)

    gsm = sub.add_parser("gsm8k")
    common(gsm, "gsm8k")
    gsm.add_argument("--dataset", type=pathlib.Path, default=ROOT / "workloads" / "gsm8k.jsonl")
    gsm.add_argument("--subset", type=int, default=100)
    gsm.add_argument("--task-id", action="append", default=[],
                     help="run explicit task id(s), bypassing the seeded subset")
    gsm.add_argument("--max-tokens", type=int, default=512)
    gsm.add_argument("--reasoning-strength", choices=("off", "low", "medium", "high", "xhigh"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    if not args.command:
        raise SystemExit("choose mqar, niah, or gsm8k (or --selfcheck)")
    args.outdir = args.outdir.resolve()
    if args.command == "mqar":
        run_mqar(args)
    elif args.command == "niah":
        run_niah(args)
    elif args.command == "gsm8k":
        run_gsm8k(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
