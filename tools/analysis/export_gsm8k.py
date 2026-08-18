#!/usr/bin/env python3
"""Export the GSM8K test split to workloads/gsm8k.jsonl -- inert prompt data, one line
per problem: {task_id, prompt, answer}.

Why GSM8K exists in this project at all (it did not before A2): the ThinkingCap
long-to-short claim is a REASONING-token reduction, and its headline numbers were measured
on math/knowledge benchmarks where the <think> block dominates the output. HumanEval+ is a
CODE benchmark; the long-to-short literature (arXiv 2510.06052) shows the token reduction
on code is far smaller (~8%) than on math (~46%), and is carried by the prose preamble.
So HumanEval+ alone would test the effect where it is weakest and could read as a null even
if the mechanism works. GSM8K is the cheap, exactly-scorable math axis that reproduces the
REGIME the claim was made in -- if the concision does not show up here, it will not show up
anywhere on our box, and that is the finding we want to be sure of.

Scoring is a pure numeric match (see `gsm8k_score`), so unlike HumanEval+ it executes NO
model-generated code -- it is safe to score on either side. This exporter still lives on the
WSL side purely because that is where `datasets` is installed; the OUTPUT is inert data the
Windows harness reads, exactly like humaneval_plus.jsonl.

    wsl -d Ubuntu-24.04 -- /home/augus/evalplus-venv/bin/python3 /mnt/c/projects/local-model-lifecycle/export_gsm8k.py
"""
from __future__ import annotations

import json
import pathlib
import re

OUT = pathlib.Path(__file__).parent / "workloads" / "gsm8k.jsonl"

# The gold answer in GSM8K sits after a literal "#### " marker at the end of the reference
# solution. Everything before it is the reference chain-of-thought, which we DISCARD -- the
# model produces its own reasoning; we only score the final number.
_GOLD = re.compile(r"####\s*([\-0-9\.,]+)")


def gold_answer(answer_field: str) -> str:
    m = _GOLD.search(answer_field)
    if not m:
        raise ValueError(f"no #### gold marker in: {answer_field[-80:]!r}")
    # Normalise: strip thousands separators and any trailing dot so "1,234" == "1234".
    return m.group(1).replace(",", "").rstrip(".")


def main() -> int:
    from datasets import load_dataset  # WSL-only import, hence inside main

    ds = load_dataset("openai/gsm8k", "main", split="test")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            rec = {
                "task_id": f"gsm8k/{i}",
                "prompt": ex["question"].strip(),
                "answer": gold_answer(ex["answer"]),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} problems -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
