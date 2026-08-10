"""A2 -- ThinkingCap long-to-short: reasoning-token / wall-clock / quality, one record per
problem, for ONE arm. The paired comparison (base vs ThinkingCap) is two runs of this with
the same fixed seeded subset; `a2_stats.py` joins them by task_id and does the paired stats.

This is a sibling of quality_bench.py, not a replacement -- it exists because the A2 metric
is the REASONING-token count, which quality_bench never captured (it stored predicted_n, the
reasoning+answer total). The differences that matter, each deliberate:

1. SPECULATIVE DECODING IS OFF, and this is not a default to override. `--spec-type
   draft-mtp` is NOT verified-exact on the qwen35 arch (upstream #23335 closed / #23302 open,
   and our own §Q: draft-mtp is quality-neutral but NOT bit-identical). It changes which
   tokens get committed, hence the reasoning-token COUNT -- the exact quantity we measure. So
   the concision arm runs plain decode; MTP throughput is a SEPARATE question measured by the
   existing e4mtp/S3 A/Bs, never mixed into the token-count metric.

2. REASONING AND ANSWER TOKENS ARE COUNTED SEPARATELY, exactly, via the server's /tokenize.
   The server runs with `--reasoning-format deepseek`, which splits the <think> block into a
   distinct `reasoning_content` stream (collectors/request.py assembles it into
   `reasoning_text`); we then tokenize each side. predicted_n (server total) is kept too, as
   the cross-check: reasoning + answer should track it up to the <think> delimiter tokens.

3. TEMPERATURE 0 (greedy), not the vendor's temp=1.0 sampler. Determinism gives a tight
   paired comparison and reproducible token counts; the vendor's 5-seed temp=1.0 protocol is
   a SECOND arm we add only if the greedy result already shows an effect worth pinning down.

4. A RUNTIME LoRA is optional (`--lora-scaled` under the hood): for the reconstruction gate
   (adapter on its ThinkingCap-origin base) and the Frágil DavidAU transfer. Baking is wrong
   here -- lambda is a swept factor.

Scoring is a SEPARATE step and differs by workload: HumanEval+ writes an evalplus samples
file (untrusted code, scored in WSL); GSM8K is a pure numeric match scored offline by
a2_stats.py (no code execution). This script GENERATES and records; it never executes a
completion.

    python a2_concision_bench.py --model qwen36-27b-dense --workload gsm8k --subset 60 --tag a2r0
    python a2_concision_bench.py --model thinkingcap-27b   --workload gsm8k --subset 60 --tag a2r0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

from benchmark_harness_qa import assemble_humaneval_solution  # single source of truth (LAB-QA-001)

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.collectors.host import sample                 # noqa: E402
from model_lifecycle.collectors.request import chat_stream, count_tokens  # noqa: E402
from model_lifecycle.models import ADAPTERS, MODELS                # noqa: E402
from model_lifecycle.servers.llama_cpp import (                    # noqa: E402
    LlamaCppAdapter, ServerProfile)

# Same consolidated fork we deploy and measure quality on, so token counts and the tokenizer
# behind /tokenize are the ones that ship.
LOCAL_BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"

# The subset is drawn ONCE from this seed and reused by BOTH arms -- the same discipline
# quality_bench enforces, and here it is load-bearing twice over: a paired reduction needs
# the SAME problems on both sides, or the pairing is a lie.
SUBSET_SEED = 20260726

# Per-workload prompt + where the problem set lives. The instruction is part of the
# measured configuration: a different instruction is a different experiment, so it is a
# constant here, not a flag.
WORKLOADS = {
    # Code: reuse quality_bench's exact instruction so HumanEval+ numbers stay comparable to
    # the existing §Q runs.
    "humaneval": {
        "file": "workloads/humaneval_plus.jsonl",
        "instruction": (
            "Complete the following Python function. Reply with the COMPLETE function "
            "definition inside a single ```python code block, and nothing else: no "
            "explanation, no tests, no example usage.\n\n{prompt}"),
    },
    # Math: the regime the ThinkingCap claim was measured in. Ask for a machine-extractable
    # final line so scoring is an exact numeric match, not a fragile last-number heuristic.
    "gsm8k": {
        "file": "workloads/gsm8k.jsonl",
        "instruction": (
            "Solve the problem. Show your reasoning, then on the final line write only:\n"
            "#### <answer>\nwhere <answer> is the final number.\n\n{prompt}"),
    },
}


def load_problems(workload: str) -> list[dict]:
    spec = WORKLOADS[workload]
    path = pathlib.Path(__file__).parent / spec["file"]
    if not path.exists():
        raise SystemExit(f"problem set missing: {path}\n"
                         f"  humaneval: wsl ... export_humaneval.sh\n"
                         f"  gsm8k:     wsl ... export_gsm8k.py")
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append({"task_id": d["task_id"], "prompt": d["prompt"],
                            "answer": d.get("answer")})
    if not out:
        raise SystemExit(f"problem set is empty: {path}")
    return sorted(out, key=lambda d: d["task_id"])


def pick_subset(problems: list[dict], n: int) -> list[dict]:
    """Fixed, seeded, and NESTED: subset(n1) is a subset of subset(n2) for n1 < n2.

    Unlike quality_bench's `rng.sample(range, n)` (which draws a fresh set per n, so a pilot
    and a full run share no problems), this takes the first n of ONE fixed shuffle. That is
    the property fail-fast needs: a cheap pilot (n=12) is a strict prefix of the full run
    (n=60), so the pilot's records are reused verbatim when we escalate -- and a promising
    pilot's numbers must move only by ADDING problems, never by resampling. Sorted back into
    task order for stable logs; the nesting lives in the shuffle, not the display order."""
    if n >= len(problems):
        n = len(problems)
    rng = random.Random(SUBSET_SEED)
    order = list(range(len(problems)))
    rng.shuffle(order)
    chosen = sorted(order[:n])
    return [problems[i] for i in chosen]


def extract_code(text: str) -> str:
    """The HumanEval completion, from whatever the model wrapped it in (quality_bench's
    logic, kept identical so scores compare)."""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    block = parts[1] if len(parts) > 1 else text
    if block.startswith("python"):
        block = block[len("python"):]
    return block.strip()


def run(model_key: str, *, workload: str, subset: int, ncmoe: int, ctx: int,
        max_tokens: int, spec: str, reasoning_format: str,
        lora_key: str, lora_lambda: float, tag: str,
        out_path: pathlib.Path) -> list[dict]:
    gguf = MODELS[model_key].path
    env = {"GGML_CUDA_REGISTER_HOST": "1"}

    # --jinja is mandatory for the thinking chat template; --reasoning-format deepseek splits
    # the think block into reasoning_content so we can count it. spec is OFF unless explicitly
    # asked for (throughput only) -- see the module docstring for why it must not touch the
    # concision metric.
    extra = ["--jinja", "--reasoning-format", reasoning_format]
    if spec:
        extra += ["--spec-type", spec, "--spec-draft-n-max", "4"]
    lora_path = None
    if lora_key:
        lora_path = ADAPTERS[lora_key]
        # --lora-scaled applies W = W0 + lambda*(A*B). lambda is the swept factor; 1.0 is the
        # SVD baseline (the reconstruction-gate point). On this pinned build the flag takes a
        # SINGLE colon-joined "FNAME:SCALE" arg (verified against --help), NOT two args.
        extra += ["--lora-scaled", f"{lora_path}:{lora_lambda}"]

    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN, env=env)
    profile = ServerProfile(model_path=gguf, port=8080, n_cpu_moe=ncmoe, ctx_size=ctx,
                            extra_args=tuple(extra))

    problems = pick_subset(load_problems(workload), subset)
    wl = WORKLOADS[workload]

    # RESUME (fail-fast escalation): if this exact stem was run before at a smaller n, its
    # records are a prefix of ours (nested subset) -- load them and generate ONLY the missing
    # task_ids. Escalating a promising pilot from n=12 to n=60 then costs 48 problems, not 60.
    # Keyed on the output file, whose name already encodes tag/model/workload/lora -- the tag
    # is the config-cell id by convention, so a same-stem record is a same-config record.
    records: list[dict] = []
    if out_path.exists():
        records = json.loads(out_path.read_text(encoding="utf-8"))
    done = {r.get("task_id") for r in records if r.get("task_id")}
    todo = [p for p in problems if p["task_id"] not in done]
    print(f"  {len(problems)} problems [{workload}], {len(done)} already done, "
          f"{len(todo)} to run. model={model_key} "
          f"lora={lora_key or 'none'}@{lora_lambda if lora_key else '-'} "
          f"spec={spec or 'OFF'} ncmoe={ncmoe} ctx={ctx} max_tok={max_tokens}", flush=True)
    if not todo:
        print("  nothing to do (all requested problems already recorded)")
        return records

    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=1800):
            print("  SERVER NEVER HEALTHY. argv:")
            print("    " + " ".join(adapter.argv(profile)))
            for ln in h.stderr_tail[-12:]:
                print(f"    | {ln}")
            # Return whatever was already recorded (possibly empty) rather than a fabricated
            # error row -- the resume file must never gain a non-problem record.
            return records
        load_s = h.load_seconds
        for i, p in enumerate(todo):
            t0 = time.monotonic()
            r = chat_stream(h.base_url, wl["instruction"].format(prompt=p["prompt"]),
                            max_tokens=max_tokens, temperature=0.0, cache_prompt=False)
            wall = time.monotonic() - t0
            s = sample()
            text = r.text or ""
            # Exact token counts under THIS arm's own tokenizer, while the server is up.
            # None (not 0) if /tokenize failed, so a transport error is never averaged as
            # 'no reasoning'. Empty reasoning is a real 0.
            reasoning_tokens = count_tokens(h.base_url, r.reasoning_text)
            answer_tokens = count_tokens(h.base_url, text)
            completion = extract_code(text) if workload == "humaneval" else text
            records.append({
                "tag": tag, "model": model_key, "workload": workload,
                "task_id": p["task_id"], "gold": p.get("answer"),
                "lora": lora_key or None, "lora_lambda": lora_lambda if lora_key else None,
                "ncmoe": ncmoe, "ctx": ctx, "spec": spec or "off",
                "reasoning_format": reasoning_format,
                # THE A2 metric, raw: separate reasoning vs answer token counts...
                "reasoning_tokens": reasoning_tokens, "answer_tokens": answer_tokens,
                # ...plus the server's own total as the independent cross-check.
                "predicted_n": r.predicted_n, "predicted_ms": r.predicted_ms,
                "prompt_n": r.prompt_n, "prompt_ms": r.prompt_ms,
                "reasoning_chars": len(r.reasoning_text), "answer_chars": len(text),
                # RAW reasoning trace kept, not just its length: the reconstruction gate needs
                # it to measure output fidelity, the short-but-wrong inspection needs to read
                # WHAT was cut, and a stored count can never be re-tokenized or re-diffed.
                "reasoning_text": r.reasoning_text,
                "wall_s": round(wall, 3), "ttft_s": r.ttft_s,
                "answered": r.answered, "ok": r.ok, "error": r.error,
                "finish_reason": None,  # placeholder; finish_reason lives in the error string
                "fenced": "```" in text,
                "completion": completion,
                "load_seconds": load_s,
                "vram_free_mb": s.vram_free_mb, "ram_avail_mb": s.ram_available_mb,
            })
            # Incremental write after every problem: a long run that dies at problem 55 keeps
            # 54, and resume picks up from there. Cheap at these sizes (one small JSON dump).
            out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            if (i + 1) % 5 == 0:
                rt = [r["reasoning_tokens"] for r in records
                      if r.get("reasoning_tokens") is not None]
                med = sorted(rt)[len(rt) // 2] if rt else "?"
                print(f"    {len(done) + i + 1}/{len(problems)}  (reasoning median so far {med})",
                      flush=True)
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
        time.sleep(20)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    ap.add_argument("--subset", type=int, default=60)
    ap.add_argument("--ncmoe", type=int, default=None,
                    help="dense 27B has no experts; default 0 (all layers on GPU as VRAM allows)")
    ap.add_argument("--ctx", type=int, default=8192)
    # 4096: same reasoning-model headroom quality_bench settled on (at 1024 the model spends
    # the whole budget in <think> and never answers). Raise for hard math if truncation shows.
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--spec", default="", help="OFF by default; draft-mtp ONLY for a throughput arm, never the token metric")
    ap.add_argument("--reasoning-format", default="deepseek",
                    help="deepseek splits <think> into reasoning_content so it can be counted")
    ap.add_argument("--lora", default="", choices=[""] + sorted(ADAPTERS),
                    help="runtime adapter (reconstruction gate / DavidAU transfer)")
    ap.add_argument("--lora-lambda", type=float, default=1.0)
    ap.add_argument("--tag", required=True, help="run label; the paired-cell id joined by a2_stats")
    args = ap.parse_args()

    # Dense models carry no experts, so ncmoe is a no-op there; default 0. (Kept as a knob
    # only so the same script could later measure a MoE arm.)
    ncmoe = args.ncmoe if args.ncmoe is not None else 0

    out = pathlib.Path(__file__).parent / "runs" / "a2"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}__{args.model}__{args.workload}"
    if args.lora:
        stem += f"__{args.lora}-l{args.lora_lambda}"
    out_path = out / f"{stem}.json"

    # run() writes out_path incrementally and resumes from it if it already exists.
    recs = run(args.model, workload=args.workload, subset=args.subset, ncmoe=ncmoe,
               ctx=args.ctx, max_tokens=args.max_tokens, spec=args.spec,
               reasoning_format=args.reasoning_format, lora_key=args.lora,
               lora_lambda=args.lora_lambda, tag=args.tag, out_path=out_path)

    # HumanEval+ needs the evalplus samples file (scored in WSL, executes code). GSM8K is
    # scored offline by a2_stats from the recorded completion -- no samples file needed.
    if args.workload == "humaneval":
        # evalplus does NOT prepend the HumanEval prompt: `solution` must be a SELF-CONTAINED
        # program (imports + prompt-provided helpers + the target function). Our instruction
        # asks the model to "Complete the following Python function", so a CONCISE model
        # correctly continues the prompt -- returning ONLY the target function and relying on
        # the helpers/imports already in the prompt (e.g. HumanEval/10 uses the prompt's
        # `is_palindrome`). Storing the bare completion then scores those continuations as 0
        # (NameError), which zeroed the ThinkingCap models while sparing verbose ones that
        # happen to re-emit everything. Prepend the prompt so both styles score fairly; for a
        # model that already re-emits the signature this only adds a harmless redefinition.
        prompts = {p["task_id"]: p["prompt"] for p in load_problems(args.workload)}
        samples = out / f"{stem}__samples.jsonl"
        with samples.open("w", encoding="utf-8") as f:
            for r in recs:
                tid = r.get("task_id")
                if tid:
                    solution = assemble_humaneval_solution(prompts[tid], r["completion"])
                    f.write(json.dumps({"task_id": tid, "solution": solution}) + "\n")
        print(f"  samples -> {samples}")

    print(f"  records -> {out / f'{stem}.json'}")
    rt = [r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens") is not None]
    if rt:
        rt.sort()
        print(f"  reasoning tokens: median={rt[len(rt)//2]}  min={rt[0]}  max={rt[-1]}  "
              f"(n={len(rt)})")
    answered = sum(1 for r in recs if r.get("answered"))
    print(f"  {answered}/{len(recs)} answered")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # Subset is fixed across calls (both arms MUST see the same problems).
        fake = [{"task_id": f"gsm8k/{i}", "prompt": "p", "answer": str(i)} for i in range(200)]
        a, b = pick_subset(fake, 60), pick_subset(fake, 60)
        assert [x["task_id"] for x in a] == [x["task_id"] for x in b], "subset must be fixed"
        assert len(a) == 60 and pick_subset(fake, 999) == fake
        ids = [x["task_id"] for x in a]
        assert ids == sorted(ids, key=lambda s: fake.index(
            next(f for f in fake if f["task_id"] == s))), "subset keeps task order"
        # NESTED: the fail-fast property -- a pilot is a strict prefix of the full run.
        small = {x["task_id"] for x in pick_subset(fake, 12)}
        big = {x["task_id"] for x in pick_subset(fake, 60)}
        assert len(small) == 12 and small <= big, "pilot subset must nest into the full run"
        # Code extraction parity with quality_bench.
        assert extract_code("```python\ndef f(): return 1\n```") == "def f(): return 1"
        assert extract_code("bare") == "bare"
        # Both workloads resolve an instruction with a {prompt} slot.
        for w in WORKLOADS.values():
            assert "{prompt}" in w["instruction"]
        print("a2_concision_bench self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
