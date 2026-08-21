#!/usr/bin/env python3
"""LAB-CTX-001: RULER-inspired effective-context curve for a live llama-server.

This local suite is not claimed to reproduce NVIDIA RULER.  It adds the missing task
families beyond single-needle recall: multi-key retrieval, multi-hop composition, and
aggregation.  Prompts are calibrated with the server's applied chat template/tokenizer,
and scoring is strict exact reply.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from benchmark_harness_qa import strict_exact_reply, wilson_interval  # noqa: E402
from tools.benchmarks.qwen38_requal import (apply_template_token_count, chat,
                                             niah_units)  # noqa: E402


TASKS = ("retrieval", "multikey", "multihop", "aggregation")


def stable_code(seed: int, label: str) -> str:
    return hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:8].upper()


def task_spec(kind: str, seed: int) -> tuple[list[str], str, str]:
    if kind == "retrieval":
        code = "R-" + stable_code(seed, "retrieval")
        return [f"AUTHORITATIVE FACT: project Aster's release code is {code}."], code, (
            "What is project Aster's release code? Reply with ONLY the exact code.")
    if kind == "multikey":
        values = ["K" + stable_code(seed, f"key-{i}")[:5] for i in range(3)]
        facts = [f"AUTHORITATIVE FACT: vault {name} has key {value}."
                 for name, value in zip(("alpha", "beta", "gamma"), values)]
        expected = "|".join(values)
        question = ("Return the keys for vaults alpha, beta, gamma in that order, joined by |. "
                    "Reply with ONLY that exact string.")
        return facts, expected, question
    if kind == "multihop":
        middle = "NODE-" + stable_code(seed, "middle")[:6]
        terminal = "NODE-" + stable_code(seed, "terminal")[:6]
        code = "H-" + stable_code(seed, "answer")
        facts = [f"AUTHORITATIVE FACT: route Aster points to {middle}.",
                 f"AUTHORITATIVE FACT: {middle} points to {terminal}.",
                 f"AUTHORITATIVE FACT: {terminal} stores code {code}."]
        return facts, code, ("Follow the route starting at Aster. What final code is stored? "
                             "Reply with ONLY the exact code.")
    if kind == "aggregation":
        rng = random.Random(seed)
        counts = [rng.randrange(11, 90) for _ in range(5)]
        facts = [f"AUTHORITATIVE FACT: depot {name} shipped {count} cobalt units."
                 for name, count in zip(("A", "B", "C", "D", "E"), counts)]
        expected = str(sum(counts))
        return facts, expected, ("How many cobalt units did depots A through E ship in total? "
                                 "Reply with ONLY the integer total.")
    raise ValueError(kind)


def assemble(units: list[str], facts: list[str], question: str) -> str:
    material = list(units)
    depths = (0.14, 0.49, 0.83, 0.31, 0.69)
    inserts = []
    for index, fact in enumerate(facts):
        position = min(len(material), int(round(depths[index] * len(material))))
        inserts.append((position, index, fact))
    for position, _, fact in sorted(inserts, reverse=True):
        material.insert(position, fact)
    return "\n".join(material) + "\n\nQuestion: " + question


def calibrate(base_url: str, target: int, kind: str, seed: int) -> tuple[str, int, str]:
    facts, expected, question = task_spec(kind, seed)
    superset = niah_units(seed, max(768, target // 5))
    low, high, best_prompt, best_tokens = 0, len(superset), "", 0
    while low <= high:
        middle = (low + high) // 2
        candidate = assemble(superset[:middle], facts, question)
        tokens = apply_template_token_count(base_url, candidate)
        if tokens <= target:
            best_prompt, best_tokens = candidate, tokens
            low = middle + 1
        else:
            high = middle - 1
    return best_prompt, best_tokens, expected


def summarize(rows: list[dict]) -> dict:
    output = {}
    observed_tasks = [kind for kind in TASKS if any(row["task"] == kind for row in rows)]
    for target in sorted({row["target_tokens"] for row in rows}):
        target_rows = [row for row in rows if row["target_tokens"] == target]
        by_task = {}
        for kind in observed_tasks:
            group = [row for row in target_rows if row["task"] == kind]
            passed = sum(row["exact"] for row in group)
            lo, hi = wilson_interval(passed, len(group))
            by_task[kind] = {"passed": passed, "n": len(group), "rate": passed / len(group),
                             "wilson_95": [lo, hi]}
        passed = sum(row["exact"] for row in target_rows)
        lo, hi = wilson_interval(passed, len(target_rows))
        output[str(target)] = {"passed": passed, "n": len(target_rows),
                               "rate": passed / len(target_rows), "wilson_95": [lo, hi],
                               "by_task": by_task,
                               "actual_tokens": sorted({row["actual_tokens"] for row in target_rows})}
    return output


def selfcheck() -> None:
    for kind in TASKS:
        facts, expected, question = task_spec(kind, 123)
        assert facts and expected and question
        built = assemble(["filler"] * 20, facts, question)
        assert all(fact in built for fact in facts)
        assert built.count("Question:") == 1
    facts, expected, _ = task_spec("aggregation", 123)
    assert int(expected) == sum(int(fact.split(" shipped ")[1].split()[0]) for fact in facts)
    print("context suite v2 self-check OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--targets", type=int, nargs="+", default=[8000, 16000, 28000])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("runs/context/LAB-CTX-001-v2/results.json"))
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return 0
    rows = []
    for target_index, target in enumerate(args.targets):
        tasks = list(args.tasks) if target_index % 2 == 0 else list(reversed(args.tasks))
        for rep in range(args.reps):
            for kind in (tasks if rep % 2 == 0 else list(reversed(tasks))):
                # Paired curve: the exact same facts/problem repeat at every context length.
                # Only the amount of filler changes, so a within-cell delta is attributable
                # to context rather than to a different random arithmetic instance.
                seed = args.seed + rep * 10 + TASKS.index(kind)
                prompt, actual, expected = calibrate(args.base_url, target, kind, seed)
                # Same calibrated unit granularity accepted by qwen38_requal NIAH.
                if target - actual > 64:
                    raise RuntimeError(f"token calibration miss for {kind}/{target}: {actual}")
                result = chat(args.base_url, prompt, max_tokens=64, timeout=args.timeout)
                answer = result["response"].strip()
                row = {"target_tokens": target, "actual_tokens": actual, "task": kind,
                       "rep": rep, "seed": seed, "expected": expected,
                       "response": result["response"],
                       "exact": strict_exact_reply(result["response"], expected),
                       "finish_reason": result["finish_reason"], "usage": result["usage"],
                       "timings": result["timings"], "wall_s": result["wall_s"],
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                rows.append(row)
                print(f"t={target:5d} actual={actual:5d} {kind:<11} r{rep} "
                      f"{'PASS' if row['exact'] else 'FAIL'} answer={answer[:40]!r}", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"campaign": "LAB-CTX-001-v2",
                    "status": "RUNNING", "rows": rows}, indent=2), encoding="utf-8")
    report = {"campaign": "LAB-CTX-001-v2",
              "scope": "RULER-inspired local tasks; not NVIDIA RULER-comparable",
              "timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": args.base_url,
              "targets": args.targets, "reps": args.reps, "summary": summarize(rows),
              "rows": rows}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if all(row["exact"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
