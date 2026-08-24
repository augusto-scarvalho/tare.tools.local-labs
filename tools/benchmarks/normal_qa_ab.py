#!/usr/bin/env python3
"""Deterministic ordinary-question A/B runner for local Qwen3.8 variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
import sys
import time
import unicodedata
import urllib.request
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model_lifecycle.collectors.request import chat_stream, count_tokens  # noqa: E402
from model_lifecycle.models import MODELS  # noqa: E402
from model_lifecycle.servers.llama_cpp import LlamaCppAdapter, ServerProfile  # noqa: E402

DEFAULT_TASKS = ROOT / "runs" / "requalification" / "QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23" / "tasks.jsonl"
DEFAULT_OUT = DEFAULT_TASKS.parent
DEFAULT_BIN = "/home/augus/opt/slop.cpp/b10165-71676e46c/bin/llama-server"
DEFAULT_LIB = "/home/augus/opt/slop.cpp/b10165-71676e46c/bin"


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        return "\n".join(lines[1:-1]).strip()
    return text


def grade(task: dict, answer: str) -> tuple[bool, str]:
    kind = task["grader"]
    clean = strip_fence(answer)
    if kind == "exact_any":
        got = normalized(clean)
        choices = [normalized(str(item)) for item in task["expected"]]
        ok = got in choices
        return ok, f"normalized={got!r} expected={choices!r}"
    if kind == "json_exact":
        try:
            got = json.loads(clean)
        except (TypeError, ValueError) as exc:
            return False, f"invalid_json={exc}"
        ok = got == task["expected"]
        return ok, f"json={got!r}"
    if kind == "lines_exact":
        got = [line.strip() for line in clean.splitlines() if line.strip()]
        ok = got == task["expected"]
        return ok, f"lines={got!r}"
    if kind == "contains_all":
        haystack = normalized(clean)
        missing = [item for item in task.get("required", [])
                   if normalized(str(item)) not in haystack]
        forbidden = [item for item in task.get("forbidden", [])
                     if normalized(str(item)) in haystack]
        words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
        too_long = bool(task.get("max_words") and len(words) > task["max_words"])
        ok = not missing and not forbidden and not too_long
        return ok, f"missing={missing!r} forbidden={forbidden!r} words={len(words)}"
    if kind == "pt_question":
        lowered = clean.casefold()
        forbidden = [item for item in task.get("forbidden", []) if item.casefold() in lowered]
        words = re.findall(r"\b\w+\b", clean, flags=re.UNICODE)
        pt_signals = ("qual", "quais", "que", "como", "onde", "quanto", "quantos",
                      "você", "seu", "sua", "deseja", "objetivo", "orçamento", "é")
        has_pt_signal = any(re.search(rf"\b{re.escape(signal)}\b", lowered)
                            for signal in pt_signals)
        one_question = clean.count("?") == 1 and clean.rstrip().endswith("?")
        too_long = bool(task.get("max_words") and len(words) > task["max_words"])
        ok = one_question and has_pt_signal and not forbidden and not too_long
        return ok, (f"one_question={one_question} pt_signal={has_pt_signal} "
                    f"forbidden={forbidden!r} words={len(words)}")
    raise ValueError(f"unknown grader: {kind}")


def load_tasks(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError(f"expected a nonempty unique task set, got {len(rows)} rows / {len(set(ids))} ids")
    return rows


def summarize(records: list[dict]) -> dict:
    categories = sorted({row["category"] for row in records})
    by_category = {}
    for category in categories:
        rows = [row for row in records if row["category"] == category]
        by_category[category] = {"pass": sum(row["pass"] for row in rows), "n": len(rows)}
    walls = [row["wall_s"] for row in records]
    tokens = [row["answer_tokens"] for row in records if row["answer_tokens"] is not None]
    return {
        "pass": sum(row["pass"] for row in records),
        "n": len(records),
        "by_category": by_category,
        "answered": sum(row["answered"] for row in records),
        "median_wall_s": statistics.median(walls) if walls else None,
        "median_answer_tokens": statistics.median(tokens) if tokens else None,
        "failure_ids": [row["id"] for row in records if not row["pass"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--tasks", type=pathlib.Path, default=DEFAULT_TASKS)
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", default="")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--system-prompt-file", type=pathlib.Path)
    args = parser.parse_args()

    if args.system_prompt and args.system_prompt_file:
        parser.error("use only one of --system-prompt or --system-prompt-file")
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()

    tasks = load_tasks(args.tasks)
    args.outdir.mkdir(parents=True, exist_ok=True)
    suffix = f"__{args.tag}" if args.tag else ""
    out_path = args.outdir / f"responses__{args.model}{suffix}.json"
    summary_path = args.outdir / f"summary__{args.model}{suffix}.json"
    records = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    task_by_id = {task["id"]: task for task in tasks}
    if len({row.get("id") for row in records}) != len(records):
        raise SystemExit(f"duplicate IDs in existing response file: {out_path}")
    for row in records:
        task = task_by_id.get(row.get("id"))
        if task is None or row.get("prompt") != task["prompt"]:
            raise SystemExit(f"existing response/task identity mismatch at {row.get('id')!r}")
    done = {row["id"] for row in records}
    todo = [task for task in tasks if task["id"] not in done]
    print(f"model={args.model} tasks={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)

    if todo:
        env = {"GGML_CUDA_REGISTER_HOST": "1", "LD_LIBRARY_PATH": DEFAULT_LIB}
        adapter = LlamaCppAdapter(server_bin=DEFAULT_BIN, env=env)
        profile = ServerProfile(
            model_path=MODELS[args.model].path,
            port=8080,
            n_gpu_layers=99,
            ctx_size=8192,
            batch=2048,
            ubatch=512,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            flash_attn="on",
            extra_args=("--jinja", "--reasoning-format", "deepseek", "--no-mmproj", "--parallel", "1"),
        )
        if not adapter.is_port_free(profile.port):
            raise SystemExit(f"PORT {profile.port} IS OCCUPIED; refusing contaminated run")
        handle = adapter.start(profile)
        try:
            if not adapter.wait_until_healthy(handle, timeout_s=600):
                print("SERVER FAILED")
                for line in handle.stderr_tail[-30:]:
                    print(line)
                return 2
            with urllib.request.urlopen(f"{handle.base_url}/props", timeout=5) as response:
                props = json.load(response)
            observed_path = props.get("model_path")
            if observed_path != profile.model_path:
                raise RuntimeError(
                    f"MODEL IDENTITY MISMATCH expected={profile.model_path!r} observed={observed_path!r}")
            for index, task in enumerate(todo, 1):
                started = time.monotonic()
                response = chat_stream(
                    handle.base_url,
                    task["prompt"],
                    max_tokens=task.get("max_tokens", 256),
                    temperature=0.0,
                    cache_prompt=False,
                    chat_template_kwargs={"enable_thinking": False},
                    system_prompt=system_prompt or None,
                )
                wall = time.monotonic() - started
                answer = response.text or ""
                passed, detail = grade(task, answer)
                records.append({
                    "id": task["id"], "category": task["category"],
                    "prompt": task["prompt"], "answer": answer,
                    "pass": passed, "grade_detail": detail,
                    "answered": response.answered, "error": response.error,
                    "reasoning_tokens": count_tokens(handle.base_url, response.reasoning_text),
                    "answer_tokens": count_tokens(handle.base_url, answer),
                    "predicted_n": response.predicted_n,
                    "predicted_ms": response.predicted_ms,
                    "wall_s": round(wall, 3),
                })
                out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  {len(done)+index:02}/{len(tasks)} {task['id']} {task['category']:12} "
                      f"{'PASS' if passed else 'FAIL'} {wall:.2f}s", flush=True)
        finally:
            adapter.stop(handle)
            adapter.force_stop(handle)
            time.sleep(15)

    result = summarize(records)
    result.update({
        "model": args.model,
        "model_path": MODELS[args.model].path,
        "tasks_path": str(args.tasks),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "engine": "b10165-71676e46c",
        "sampling": {"temperature": 0.0, "thinking": "instruct", "mtp": "off", "ctx": 8192},
        "system_prompt": system_prompt or None,
    })
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
