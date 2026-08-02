#!/usr/bin/env python3
"""Score a SUBSET samples.jsonl with evalplus (which otherwise demands all 164 problems).

Pads the subset to the full HumanEval+ set with empty solutions (they fail), runs the official
evalplus.evaluate (real sandbox + tests), then reads its per-problem results and reports pass@1
over ONLY the subset's task_ids -- base (HumanEval) and plus (HumanEval+). Run in the evalplus
venv:  /home/augus/evalplus-venv/bin/python score_subset.py <subset_samples.jsonl>
"""
import sys, json, subprocess, pathlib
from evalplus.data import get_human_eval_plus

samples = pathlib.Path(sys.argv[1])
mine = {}
for line in samples.read_text().splitlines():
    line = line.strip()
    if line:
        d = json.loads(line); mine[d["task_id"]] = d["solution"]
subset_ids = list(mine)
print(f"subset: {len(subset_ids)} problems from {samples.name}")

allp = list(get_human_eval_plus())
padded = samples.with_suffix(".padded.jsonl")
with padded.open("w") as f:
    for tid in allp:
        f.write(json.dumps({"task_id": tid, "solution": mine.get(tid, "")}) + "\n")

subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
                "--samples", str(padded)], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

res_path = padded.with_name(padded.stem + "_eval_results.json")  # evalplus: X.jsonl -> X_eval_results.json
res = json.loads(res_path.read_text())
evald = res["eval"]            # {task_id: [ {base_status, plus_status, ...} ]}

base_ok = plus_ok = 0
fails = []
for tid in subset_ids:
    e = evald.get(tid)
    entry = e[0] if isinstance(e, list) and e else (e or {})
    b = entry.get("base_status") == "pass"
    p = entry.get("plus_status") == "pass"
    base_ok += b; plus_ok += p
    if not p:
        fails.append(tid)
n = len(subset_ids)
print(f"pass@1 base (HumanEval)  = {base_ok}/{n} = {base_ok/n:.3f}")
print(f"pass@1 plus (HumanEval+) = {plus_ok}/{n} = {plus_ok/n:.3f}")
print(f"failed (plus): {sorted(fails)}")
# machine-readable line for the matrix aggregator
print(f"RESULT {samples.stem} base={base_ok}/{n} plus={plus_ok}/{n}")
