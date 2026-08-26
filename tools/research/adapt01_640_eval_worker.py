#!/usr/bin/env python3
"""Behaviorally evaluate the fresh ADAPT-01 640-step arm omitted by its driver."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/benchmarks"))
from normal_qa_ab import load_tasks
from tools.probes.adapt01_lokr_scale import PROTECTED_IDS, arm_summary, load_target_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=pathlib.Path, required=True)
    parser.add_argument("--teacher", type=pathlib.Path, required=True)
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--qa", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    target_rows = load_target_rows(args.teacher, args.prompts, args.seed)
    by_id = {task["id"]: task for task in load_tasks(args.qa)}
    protected = [by_id[task_id] for task_id in PROTECTED_IDS]
    teacher_counts = [len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"]) for row in target_rows]
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    started = time.monotonic()
    result = arm_summary("lokr_5ep", model, tokenizer, target_rows, protected, statistics.median(teacher_counts), torch)
    result["elapsed_seconds_total"] = time.monotonic() - started
    result["seed"] = args.seed
    result["gpu"] = torch.cuda.get_device_name(0)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
