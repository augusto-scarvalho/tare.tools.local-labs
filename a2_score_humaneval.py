#!/usr/bin/env python3
"""Score a HumanEval+ samples.jsonl and dump PER-TASK pass/fail as {task_id: bool} JSON, so
a2_stats can do the paired (McNemar) accuracy test. score_subset.py already reports aggregate
pass@1; this variant additionally writes the per-problem booleans a2_stats' --ext-base/--ext-cap
consume. Uses the HumanEval+ (`plus_status`) verdict -- the harder, extended tests -- as the
correctness oracle.

Executes untrusted model-generated code through evalplus's sandbox, so it runs ONLY inside the
distro's evalplus venv, never on the Windows side:

    /home/augus/evalplus-venv/bin/python a2_score_humaneval.py <subset_samples.jsonl> [out.json]
"""
import json
import pathlib
import subprocess
import sys

from evalplus.data import get_human_eval_plus

samples = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else samples.with_name(
    samples.stem.replace("__samples", "") + "__scores.json")

mine = {}
for line in samples.read_text().splitlines():
    line = line.strip()
    if line:
        d = json.loads(line)
        mine[d["task_id"]] = d["solution"]
subset_ids = list(mine)

# evalplus insists on the full 164; pad the rest with empty (failing) solutions.
allp = list(get_human_eval_plus())
padded = samples.with_suffix(".padded.jsonl")
with padded.open("w") as f:
    for tid in allp:
        f.write(json.dumps({"task_id": tid, "solution": mine.get(tid, "")}) + "\n")

res_path = padded.with_name(padded.stem + "_eval_results.json")
# evalplus reuses <padded>_eval_results.json if present -> stale verdicts when re-scoring a
# corrected samples file under the same name. Delete it so the score is always current.
res_path.unlink(missing_ok=True)

subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
                "--samples", str(padded)], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
evald = json.loads(res_path.read_text())["eval"]

scores = {}
for tid in subset_ids:
    e = evald.get(tid)
    entry = e[0] if isinstance(e, list) and e else (e or {})
    scores[tid] = entry.get("plus_status") == "pass"   # HumanEval+ (extended tests)

out.write_text(json.dumps(scores, indent=2))
n = len(subset_ids)
passed = sum(scores.values())
print(f"pass@1 (HumanEval+) = {passed}/{n} = {passed/n:.3f}")
print(f"per-task scores -> {out}")
