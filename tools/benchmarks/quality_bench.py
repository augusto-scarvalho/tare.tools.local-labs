"""Quality AND throughput from the same run, one record per problem.

The models have to be loaded and generated from anyway, so every run yields both
responses: whether the answer was correct, and how fast it arrived. Treating quality as
just another response variable is what makes the existing DOE machinery apply to it.

Three things this file is careful about, each for a reason that has already cost this
project time:

1. ONE RECORD PER PROBLEM, never a pre-aggregated pass@1. A stored quotient cannot be
   un-divided -- the same defect that forced a re-measurement when `gen_tps` was written
   into the raw payload as a derived value. Per-problem records let pass@1 be recomputed,
   given a CI, and inspected for WHICH problems discriminate between configurations.

2. THE SUBSET IS FIXED AND SEEDED. A different random subset per configuration would put
   sampling variance inside the factor effects, and no amount of replication recovers it.

3. EXECUTION IS NOT OURS. Scoring runs through `evalplus`, which already has process
   isolation and timeouts for untrusted model-generated code. Writing a sandbox for
   arbitrary generated code, on a machine that is the owner's workstation, is exactly the
   thing not to hand-roll. This script GENERATES and records; it never executes a
   completion.

    python quality_bench.py --model qwen36-35b --subset 40 --tag screen-r0
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.collectors.host import sample                 # noqa: E402
from model_lifecycle.collectors.request import chat_stream         # noqa: E402
from model_lifecycle.models import MODELS                          # noqa: E402
from model_lifecycle.servers.llama_cpp import (                    # noqa: E402
    LlamaCppAdapter, ServerProfile)

# The consolidated fork (branch `lifecycle` = 720d7fa40 + §B2b + prefetch + expert-cache), so
# quality is measured on the SAME binary we deploy and the lever knobs below actually engage.
LOCAL_BIN = "/home/augus/src/llama.cpp-master/build/bin/llama-server"

# MODELS is the shared registry (model_lifecycle.models), quant-keyed. Quantisations of the
# same weights stay separate entries on purpose: quant is a FACTOR in the screen, not a
# property of the model. Was a copied (path, block_count) dict here.

# The subset is drawn ONCE from this seed and reused by every configuration. Changing it
# invalidates comparison with every run already taken, so it is a constant and not a flag.
SUBSET_SEED = 20260726

INSTRUCTION = (
    "Complete the following Python function. Reply with the COMPLETE function definition "
    "inside a single ```python code block, and nothing else: no explanation, no tests, no "
    "example usage.\n\n{prompt}"
)


# Problem prompts, exported ONCE from inside the distro and read here as plain data.
#
# The first version imported `evalplus` directly. That fails, because generation runs on
# WINDOWS while evalplus was installed in WSL -- but installing it on both sides is the
# wrong repair. evalplus exists to EXECUTE untrusted model-generated code; putting it on
# the workstation, even unused, weakens the boundary for no gain.
#
# The two needs separate cleanly: Windows needs the prompts (public, inert data), the
# distro needs the executor. So the distro exports a JSONL once and this side reads it.
PROBLEMS_JSONL = pathlib.Path(__file__).parent / "workloads" / "humaneval_plus.jsonl"

EXPORT_HINT = (
    "Export it from inside the distro, where evalplus lives:\n"
    "  wsl -d Ubuntu-24.04 -- bash /mnt/c/export_humaneval.sh")


def load_problems() -> list[dict]:
    if not PROBLEMS_JSONL.exists():
        raise SystemExit(f"problem set missing: {PROBLEMS_JSONL}\n{EXPORT_HINT}")
    out = []
    with PROBLEMS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append({"task_id": d["task_id"], "prompt": d["prompt"]})
    if not out:
        raise SystemExit(f"problem set is empty: {PROBLEMS_JSONL}\n{EXPORT_HINT}")
    return sorted(out, key=lambda d: d["task_id"])


def pick_subset(problems: list[dict], n: int) -> list[dict]:
    """Fixed, seeded, and SORTED back into task order so the run is reproducible and the
    log reads in a stable sequence."""
    if n >= len(problems):
        return problems
    rng = random.Random(SUBSET_SEED)
    chosen = rng.sample(range(len(problems)), n)
    return [problems[i] for i in sorted(chosen)]


def extract_code(text: str) -> str:
    """The completion, from whatever the model wrapped it in.

    Models that cannot follow "reply with only a code block" fail this benchmark for a
    formatting reason, which is a REAL agentic failure and must not be papered over -- but
    it must also be distinguishable from a wrong answer. `fenced` is recorded so the two
    can be told apart afterwards.
    """
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    # parts[1] is the first fenced block; strip a leading language tag.
    block = parts[1] if len(parts) > 1 else text
    if block.startswith("python"):
        block = block[len("python"):]
    return block.strip()


def run(model_key: str, *, subset: int, ncmoe: int, ctx: int, ubatch: int | None,
        kv: str, flash: str | None, prefetch: str, max_tokens: int,
        spec: str, cache_slots: int, cache_profile: str,
        tag: str) -> list[dict]:
    gguf, blocks = MODELS[model_key].path, MODELS[model_key].block_count
    env = {"GGML_CUDA_REGISTER_HOST": "1"}
    if prefetch and prefetch != "0":
        env["GGML_SCHED_PREFETCH_EXPERTS"] = prefetch
    # Pass through GDN chunked-kernel knobs so this harness can bless it for quality-neutrality.
    # Set on the (Windows) quality_bench invocation; the adapter forwards them into the WSL server
    # env. MIN_TOKENS=2 forces the chunked TF32 prefill path even on short HumanEval prompts.
    for _k in ("GGML_CUDA_GDN_CHUNKED", "GGML_CUDA_GDN_CHUNKED_MIN_TOKENS", "GGML_CUDA_GDN_TC"):
        if os.environ.get(_k):
            env[_k] = os.environ[_k]

    # The lever knobs, threaded as server flags so the SAME quality machinery measures each
    # config. --jinja is mandatory (thinking-model chat template). spec=draft-mtp is the MTP
    # self-draft (EXACT by construction -> a quality-neutrality CHECK); the MoE expert cache
    # changes compute (hot/cold split) -> a real quality test. Off by default = pristine.
    extra = ["--jinja"]
    if spec:
        extra += ["--spec-type", spec, "--spec-draft-n-max", "4"]
    if cache_slots > 0 and cache_profile:
        extra += ["--moe-cache-slots", str(cache_slots), "--moe-cache-profile", cache_profile]

    adapter = LlamaCppAdapter(server_bin=LOCAL_BIN, env=env)
    profile = ServerProfile(model_path=gguf, port=8080, n_cpu_moe=ncmoe, ctx_size=ctx,
                            ubatch=ubatch, cache_type_k=kv, cache_type_v=kv,
                            flash_attn=flash, extra_args=tuple(extra))

    problems = pick_subset(load_problems(), subset)
    print(f"  {len(problems)} problems, config: ncmoe={ncmoe} ctx={ctx} ub={ubatch} "
          f"kv={kv} fa={flash} prefetch={prefetch} spec={spec or 'off'} "
          f"cache_slots={cache_slots}", flush=True)

    records: list[dict] = []
    h = adapter.start(profile)
    try:
        if not adapter.wait_until_healthy(h, timeout_s=1800):
            print("  SERVER NEVER HEALTHY. argv:")
            print("    " + " ".join(adapter.argv(profile)))
            for ln in h.stderr_tail[-10:]:
                print(f"    | {ln}")
            return [{"tag": tag, "model": model_key, "error": "unhealthy",
                     "ncmoe": ncmoe, "ctx": ctx, "ubatch": ubatch, "kv": kv,
                     "flash": flash, "prefetch": prefetch}]
        load_s = h.load_seconds
        for i, p in enumerate(problems):
            t0 = time.monotonic()
            r = chat_stream(h.base_url, INSTRUCTION.format(prompt=p["prompt"]),
                            max_tokens=max_tokens, temperature=0.0, cache_prompt=False)
            wall = time.monotonic() - t0
            s = sample()
            # A starved or empty answer is a recordable outcome, never silently a blank
            # completion -- `answered` and `error` below distinguish the two.
            text = r.text or ""
            completion = extract_code(text)
            records.append({
                "tag": tag, "model": model_key, "task_id": p["task_id"],
                "ncmoe": ncmoe, "ctx": ctx, "ubatch": ubatch, "kv": kv,
                "flash": flash, "prefetch": prefetch,
                "spec": spec or "off", "cache_slots": cache_slots,
                "completion": completion,
                "answered": r.answered, "ok": r.ok, "error": r.error,
                "fenced": "```" in (text or ""),
                # RAW, never a rate: the same discipline the throughput side learned the
                # hard way. Rates are recomputed from these.
                "prompt_n": r.prompt_n, "prompt_ms": r.prompt_ms,
                "predicted_n": r.predicted_n, "predicted_ms": r.predicted_ms,
                "cache_n": r.cache_n, "wall_s": round(wall, 2),
                "load_seconds": load_s,
                "vram_free_mb": s.vram_free_mb, "ram_avail_mb": s.ram_available_mb,
            })
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{len(problems)}", flush=True)
    finally:
        adapter.stop(h)
        adapter.force_stop(h)
        time.sleep(20)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--subset", type=int, default=40)
    ap.add_argument("--ncmoe", type=int, default=None)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ubatch", type=int, default=None)
    ap.add_argument("--kv", default="q8_0")
    ap.add_argument("--flash", default=None)
    ap.add_argument("--prefetch", default="0")
    # 4096, not 1024: Qwen3.6 is a THINKING model -- at 1024 it spends the whole budget in
    # <think> and never emits code (33/40 starved). 4096 clears think+code with headroom
    # (diagnostic: pred_n 1090-3475, none hit the ceiling).
    ap.add_argument("--max-tokens", type=int, default=4096)
    # Lever knobs for the config x quality matrix (all off by default = pristine fork path).
    ap.add_argument("--spec", default="", help="e.g. draft-mtp (MTP self-draft; EXACT)")
    ap.add_argument("--moe-cache-slots", type=int, default=0)
    ap.add_argument("--moe-cache-profile", default="",
                    help="routing profile CSV from llama-moe-trace (with --moe-cache-slots)")
    ap.add_argument("--tag", required=True,
                    help="run label; the DOE cell id, so records can be joined to a design")
    args = ap.parse_args()

    blocks = MODELS[args.model].block_count
    ncmoe = args.ncmoe if args.ncmoe is not None else max(1, round(blocks * 0.6))
    recs = run(args.model, subset=args.subset, ncmoe=ncmoe, ctx=args.ctx,
               ubatch=args.ubatch, kv=args.kv, flash=args.flash,
               prefetch=args.prefetch, max_tokens=args.max_tokens,
               spec=args.spec, cache_slots=args.moe_cache_slots,
               cache_profile=args.moe_cache_profile, tag=args.tag)

    out = pathlib.Path(__file__).parent / "runs" / "quality"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}__{args.model}.json").write_text(
        json.dumps(recs, indent=2), encoding="utf-8")

    # evalplus format: one {task_id, solution} per line. Scoring is a SEPARATE step, run
    # inside the distro, because it executes untrusted generated code.
    samples = out / f"{args.tag}__{args.model}__samples.jsonl"
    with samples.open("w", encoding="utf-8") as f:
        for r in recs:
            if r.get("task_id"):
                f.write(json.dumps({"task_id": r["task_id"],
                                    "solution": r["completion"]}) + "\n")
    print(f"\n  records -> {out / f'{args.tag}__{args.model}.json'}")
    print(f"  samples -> {samples}")
    print("  score with:  evalplus.evaluate --dataset humaneval --samples <samples.jsonl>")
    answered = sum(1 for r in recs if r.get("answered"))
    print(f"  {answered}/{len(recs)} produced content "
          f"(a low number here is a FORMAT failure, not a wrong answer)")
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # The subset must be identical across calls, or factor effects absorb sampling
        # variance and no replication recovers it.
        fake = [{"task_id": f"T/{i}", "prompt": "p"} for i in range(164)]
        a, b = pick_subset(fake, 40), pick_subset(fake, 40)
        assert [x["task_id"] for x in a] == [x["task_id"] for x in b], "subset must be fixed"
        assert len(a) == 40
        assert pick_subset(fake, 999) == fake, "n >= total returns everything"
        ids = [x["task_id"] for x in a]
        assert ids == sorted(ids, key=lambda s: fake.index(next(
            f for f in fake if f["task_id"] == s))), "subset keeps task order"

        # Fenced, tagged, bare, and prose-wrapped completions all have to come out clean.
        assert extract_code("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
        assert extract_code("```\ndef f(): pass\n```") == "def f(): pass"
        assert extract_code("def f(): pass") == "def f(): pass"
        assert extract_code("Sure!\n```python\ndef f(): pass\n```\nHope that helps") \
            == "def f(): pass"

        # The problem set is DATA on this side, not a package import. Verify the loader
        # reads what the exporter writes, and that a missing file fails with the fix in the
        # message rather than an ImportError from a package that should not be installed
        # here at all.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "hp.jsonl"
            p.write_text('{"task_id": "HumanEval/1", "prompt": "b"}\n'
                         '{"task_id": "HumanEval/0", "prompt": "a"}\n', encoding="utf-8")
            import builtins
            saved = PROBLEMS_JSONL
            globals()["PROBLEMS_JSONL"] = p
            got = load_problems()
            assert [d["task_id"] for d in got] == ["HumanEval/0", "HumanEval/1"], got
            globals()["PROBLEMS_JSONL"] = pathlib.Path(td) / "missing.jsonl"
            try:
                load_problems()
                raise AssertionError("a missing problem set must fail loudly")
            except SystemExit as e:
                assert "export_humaneval" in str(e), str(e)
            globals()["PROBLEMS_JSONL"] = saved
            del builtins

        # WIRING, not just parsing. The first version read `r.raw_text`, a field that does
        # not exist, so `hasattr` returned False and EVERY completion would have come back
        # empty -- while a self-check that only exercised `extract_code` passed happily.
        # A parser test that never touches the object it parses from is not a check.
        from model_lifecycle.collectors.request import RequestResult
        rr = RequestResult(ok=True, text="```python\ndef f(): return 2\n```")
        assert extract_code(rr.text) == "def f(): return 2", \
            "the field quality_bench reads must be the field chat_stream fills"
        assert RequestResult(ok=True).text == "", "text must default to empty, not None"

        print("quality_bench self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
