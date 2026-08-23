#!/usr/bin/env python3
"""Generate independent tests with a local model and run a frozen mutation gate."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


MUTATIONS = [
    ("stale_digest", '        if delta.base_digest != self.digest:\n            raise ValueError("stale delta: base digest does not match current contract")\n', ""),
    ("cross_contract", '        if delta.contract_id != self.contract_id:\n            raise ValueError("delta targets a different contract")\n', ""),
    ("version_increment", "            version=self.version + 1,", "            version=self.version,"),
    ("parent_chain", "            parent_digest=self.digest,", "            parent_digest=None,"),
    ("evidence_append", "            evidence=self.evidence + tuple(delta.evidence_append),", "            evidence=tuple(delta.evidence_append),"),
    ("missing_test", "    passed = not regressions and (not require_same_tests or not missing)", "    passed = not regressions"),
    ("regression", "    passed = not regressions and (not require_same_tests or not missing)", "    passed = not missing"),
]


def ask(port: int, source: str) -> tuple[dict, float]:
    prompt = f"""You are an independent test writer. Write one self-contained Python unittest module for the source below.

Requirements:
- Import only TaskContract, ContractDelta, and test_baseline_non_weakening from model_lifecycle.agent_harness.
- Do not modify sys.path. Do not use network, subprocess, sleep, random, mocks, or third-party packages.
- Write at least 8 deterministic test methods.
- Test digest determinism; valid delta version increment; parent digest chaining; invariant objective/constraints/required_tests; evidence append rather than replacement; stale digest rejection; cross-contract rejection; status/next_action preservation and update; missing baseline-test rejection; passing-to-failing regression rejection; and harmless test additions.
- Output Python code only, in one fenced python block. Keep it under 180 non-empty lines.

SOURCE:
```python
{source}
```
"""
    payload = {
        "model": "independent-test-writer",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 2048,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip() + "\n"


def run_suite(source: str, tests: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        package = root / "model_lifecycle"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "agent_harness.py").write_text(source, encoding="utf-8")
        (root / "test_independent.py").write_text(tests, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "-v", "test_independent.py"],
            cwd=root, env=env, text=True, capture_output=True, timeout=60,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "passed": completed.returncode == 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source.read_text(encoding="utf-8")
    body, elapsed = ask(args.port, source)
    choice = body["choices"][0]
    message = choice["message"]
    completion = (message.get("content") or message.get("reasoning_content") or "").strip()
    (args.output / "MODEL_RESPONSE.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    tests = extract_code(completion)
    test_path = args.output / "test_independent_generated.py"
    test_path.write_text(tests, encoding="utf-8")

    syntax_valid = True
    syntax_error = None
    try:
        tree = ast.parse(tests)
    except SyntaxError as exc:
        syntax_valid = False
        syntax_error = str(exc)
        tree = ast.parse("pass")
    banned_names = sorted({
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        and node.id in {"subprocess", "requests", "urllib", "random", "sleep"}
    })
    test_methods = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
        for node in ast.walk(tree)
    )
    nonempty_lines = sum(bool(line.strip()) for line in tests.splitlines())

    baseline = run_suite(source, tests) if syntax_valid else {"passed": False, "returncode": None}
    rows = []
    for mutation_id, old, new in MUTATIONS:
        occurrences = source.count(old)
        if occurrences != 1:
            rows.append({"id": mutation_id, "valid": False, "occurrences": occurrences, "killed": False})
            continue
        mutated = source.replace(old, new, 1)
        execution = run_suite(mutated, tests)
        rows.append({
            "id": mutation_id, "valid": True, "occurrences": occurrences,
            "killed": not execution["passed"], "execution": execution,
        })
        print(f"{mutation_id}: {'KILLED' if not execution['passed'] else 'SURVIVED'}")
    killed = sum(row["killed"] for row in rows)
    required_killed = {
        row["id"]: row["killed"] for row in rows
        if row["id"] in {"stale_digest", "regression"}
    }
    static_pass = syntax_valid and not banned_names and nonempty_lines <= 180 and test_methods >= 8
    passed = (
        static_pass and baseline.get("passed") and killed >= 5
        and all(required_killed.values()) and len(required_killed) == 2
    )
    report = {
        "schema_version": 1, "model_elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"), "syntax_valid": syntax_valid,
        "syntax_error": syntax_error, "banned_names": banned_names,
        "nonempty_lines": nonempty_lines, "test_methods": test_methods,
        "static_pass": static_pass, "baseline": baseline, "mutations": rows,
        "killed": killed, "total_mutations": len(rows),
        "required_killed": required_killed,
        "decision": "PASS" if passed else "FAIL",
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "model_elapsed_seconds", "syntax_valid", "banned_names", "nonempty_lines",
        "test_methods", "static_pass", "killed", "total_mutations",
        "required_killed", "decision",
    )}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
