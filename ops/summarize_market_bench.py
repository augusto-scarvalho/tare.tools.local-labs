#!/usr/bin/env python3
"""Summarize the fable-tc market-comparison benchmark into one SUMMARY.md.

Reads the a2_concision_bench record JSONs (quality inputs + per-problem speed) plus the
HumanEval+ evalplus score, and emits a compact, market-comparable table: code pass@1
(HumanEval base + HumanEval+), GSM8K accuracy, decode t/s (levers ON = MTP), and median
reasoning tokens (concision). Absolute numbers meant to sit next to published model scores.

    PYTHONPATH=src python ops/summarize_market_bench.py --tag market-r0 --model fable-tc-l1.0-q4 --out runs/quality-market/SUMMARY.md
"""
import argparse, json, pathlib, statistics, sys, re

sys.path.insert(0, "src")
sys.path.insert(0, ".")
import a2_stats  # gsm8k_extract, numeric_equal — pure-python numeric match


def _tps(r):
    pn, pm = r.get("predicted_n"), r.get("predicted_ms")
    return pn / (pm / 1000.0) if (pn and pm) else None


def _load(tag, model, wl):
    p = pathlib.Path("runs/a2") / f"{tag}__{model}__{wl}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _speed(recs):
    xs = [_tps(r) for r in recs if _tps(r)]
    if not xs:
        return None
    return {"median": statistics.median(xs), "mean": statistics.mean(xs),
            "min": min(xs), "max": max(xs), "n": len(xs)}


def _reasoning_median(recs):
    rt = sorted(r["reasoning_tokens"] for r in recs if r.get("reasoning_tokens") is not None)
    return rt[len(rt) // 2] if rt else None


def _answered(recs):
    return sum(1 for r in recs if r.get("answered")), len([r for r in recs if r.get("task_id")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--he-score-file", default="runs/quality-market/humaneval_score.txt")
    a = ap.parse_args()

    he = _load(a.tag, a.model, "humaneval")
    gs = _load(a.tag, a.model, "gsm8k")

    # HumanEval+ pass@1 from the evalplus score line: "RESULT <stem> base=X/N plus=Y/N"
    he_line = ""
    p = pathlib.Path(a.he_score_file)
    if p.exists():
        raw = p.read_bytes()  # PowerShell redirects can be UTF-16; be BOM-aware
        if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
            txt = raw.decode("utf-16", errors="replace")
        else:
            txt = raw.decode("utf-8-sig", errors="replace")
        for ln in txt.splitlines():
            if ln.strip().startswith("RESULT"):
                he_line = ln.strip()
    m = re.search(r"base=(\d+)/(\d+)\s+plus=(\d+)/(\d+)", he_line)
    he_base = f"{int(m.group(1))}/{int(m.group(2))} ({100*int(m.group(1))/int(m.group(2)):.1f}%)" if m else "n/a"
    he_plus = f"{int(m.group(3))}/{int(m.group(4))} ({100*int(m.group(3))/int(m.group(4)):.1f}%)" if m else "n/a"

    # GSM8K accuracy — numeric match on the recorded completion
    gs_ok = [a2_stats.numeric_equal(a2_stats.gsm8k_extract(r.get("completion") or ""), r.get("gold"))
             for r in gs if r.get("task_id")]
    gs_n = len(gs_ok); gs_c = sum(1 for x in gs_ok if x)
    gs_acc = f"{gs_c}/{gs_n} ({100*gs_c/gs_n:.1f}%)" if gs_n else "n/a"

    he_sp, gs_sp = _speed(he), _speed(gs)
    he_ans = _answered(he); gs_ans = _answered(gs)

    def sp(s):
        return f"{s['median']:.1f} t/s (mean {s['mean']:.1f}, n={s['n']})" if s else "n/a"

    lines = [
        f"# fable-tc l1.0 - quality + speed benchmark (market comparison baseline)",
        "",
        f"Model: `{a.model}` (Qwen3.6-27B dense merge: concise TC + uncensored Fable). Deploy config,",
        f"**speed levers ON**: `--spec-type draft-mtp` (MTP self-draft) on the `lifecycle` fork binary,",
        f"CUDA graphs + MMQ (int8 TC) default-on, VRAM OC +350 (hardware). ctx 8192, max_tokens 4096,",
        f"temperature 0 (greedy). Seeded/nested subsets. Note: these are SHORT-context standard benchmarks",
        f"(prompts <500 tok); the fork's MoE/long-context levers (placement, prefetch, KV-host-pin, GDN)",
        f"do not apply to a dense short-context run (GDN is even -2 to -4% on the dense H=48).",
        "",
        "| Axis | Benchmark | Score | Answered | Decode speed (MTP) |",
        "|------|-----------|-------|----------|--------------------|",
        f"| Code | HumanEval (base) | {he_base} | {he_ans[0]}/{he_ans[1]} | {sp(he_sp)} |",
        f"| Code | HumanEval+ (plus)| {he_plus} | - | - |",
        f"| Reasoning/math | GSM8K | {gs_acc} | {gs_ans[0]}/{gs_ans[1]} | {sp(gs_sp)} |",
        "",
        f"Concision (median reasoning tokens): HumanEval {_reasoning_median(he)}, GSM8K {_reasoning_median(gs)}.",
        "",
        "Raw records: `runs/a2/{tag}__{model}__{{humaneval,gsm8k}}.json` (one record per problem, pass@1",
        "recomputable with a CI, per-problem t/s and reasoning trace kept). Reproduce: `ops/run_market_bench.ps1`.",
        f"Tag `{a.tag}`. Scale up by re-running the same tag with a larger --subset (nested, resumes, no rework).",
    ]
    outp = pathlib.Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {outp}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
