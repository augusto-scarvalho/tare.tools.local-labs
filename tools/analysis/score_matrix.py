#!/usr/bin/env python3
"""Aggregate the levers x quality matrix: pass@1 per config + what each lever changes.

Two-pronged, exploiting temp=0 determinism:
  * EXACT levers (MTP) must produce byte-identical completions to base -> quality-neutral by
    construction; we CHECK that identity directly (cheaper and stronger than comparing pass@1).
  * numerics-changing levers (expert cache, KV quant) produce different completions -> score
    pass@1 with evalplus and diff the FAILED SET vs base (which problems flip, and which way).

Run in the evalplus venv:  /home/augus/evalplus-venv/bin/python score_matrix.py <runs/quality dir>
"""
import sys, json, subprocess, pathlib
from evalplus.data import get_human_eval_plus

qdir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                    "/mnt/c/projects/local-model-lifecycle/runs/quality")
CONFIGS = ["qm-base", "qm-mtp", "qm-kvq4", "qm-kvf16", "qm-cache"]
MODEL = "qwen36-35b-mtp-q4"
ALL = list(get_human_eval_plus())


def load_records(tag):
    f = qdir / f"{tag}__{MODEL}.json"
    return {r["task_id"]: r for r in json.loads(f.read_text())} if f.exists() else None


def score(tag):
    """pass@1 (base, plus) + failed-plus set over the subset, via padded evalplus."""
    samples = qdir / f"{tag}__{MODEL}__samples.jsonl"
    mine = {json.loads(l)["task_id"]: json.loads(l)["solution"]
            for l in samples.read_text().splitlines() if l.strip()}
    ids = list(mine)
    padded = samples.with_name(samples.stem + ".padded.jsonl")
    with padded.open("w") as fh:
        for tid in ALL:
            fh.write(json.dumps({"task_id": tid, "solution": mine.get(tid, "")}) + "\n")
    subprocess.run([sys.executable, "-m", "evalplus.evaluate", "--dataset", "humaneval",
                    "--samples", str(padded)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ev = json.loads((padded.with_name(padded.stem + "_eval_results.json")).read_text())["eval"]
    base = {t for t in ids if ev.get(t, [{}])[0].get("base_status") == "pass"}
    plus = {t for t in ids if ev.get(t, [{}])[0].get("plus_status") == "pass"}
    return ids, base, plus


base_recs = load_records("qm-base")
if not base_recs:
    raise SystemExit("qm-base records missing -- matrix not finished?")

print("=" * 70)
print("LEVERS x QUALITY  (Qwen3.6-35B-A3B, HumanEval+ 40-subset, thinking, temp=0)")
print("=" * 70)

base_ids, base_b, base_p = score("qm-base")
n = len(base_ids)
print(f"\nqm-base (deploy: ncmoe=8 q8_0, no lever): "
      f"pass@1 base={len(base_b)}/{n} plus={len(base_p)}/{n}")

for tag in CONFIGS[1:]:
    recs = load_records(tag)
    if not recs:
        print(f"\n{tag}: MISSING (skipped)"); continue
    # completion identity vs base (the exact-lever check)
    ident = sum(1 for t in base_ids
                if (recs.get(t, {}).get("completion")) == base_recs[t]["completion"])
    ids, b, p = score(tag)
    dp = len(p) - len(base_p)
    newfail = sorted(base_p - p)   # regressions: passed on base, fails here
    newpass = sorted(p - base_p)   # gains
    print(f"\n{tag}:  pass@1 base={len(b)}/{n} plus={len(p)}/{n}  "
          f"(Δplus {dp:+d})   completions identical to base: {ident}/{n}")
    if ident == n:
        print(f"   -> BYTE-IDENTICAL to base at temp=0: quality-neutral by construction.")
    else:
        print(f"   -> {n-ident} completions differ.  regressions: {newfail or 'none'}"
              + (f"   gains: {newpass}" if newpass else ""))
