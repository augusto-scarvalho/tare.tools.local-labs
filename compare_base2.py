#!/usr/bin/env python3
"""base vs base2: the non-determinism floor. Same config, same seed, twice."""
import json, pathlib
q = pathlib.Path("/mnt/c/projects/local-model-lifecycle/runs/quality")
M = "qwen36-35b-mtp-q4"
a = {r["task_id"]: r for r in json.loads((q / f"qm-base__{M}.json").read_text())}
b = {r["task_id"]: r for r in json.loads((q / f"qm-base2__{M}.json").read_text())}
ids = sorted(a)
ident = sum(1 for t in ids if a[t]["completion"] == b[t]["completion"])
ans_a = sum(1 for t in ids if a[t].get("answered"))
ans_b = sum(1 for t in ids if b[t].get("answered"))
diff_ids = [t for t in ids if a[t]["completion"] != b[t]["completion"]]
print(f"base vs base2 (SAME config, SAME seed): {ident}/{len(ids)} completions IDENTICAL")
print(f"  answered: base={ans_a}  base2={ans_b}")
print(f"  differing task_ids ({len(diff_ids)}): {diff_ids}")
