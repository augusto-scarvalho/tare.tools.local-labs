#!/usr/bin/env python3
"""A2 Gate 3 -- Claude WORKER judge bridge (Sonnet-5 @ medium, no Anthropic API key).

The Claude member of the quorum runs as a Claude Code subagent, not an HTTP call. This script is the
two-way bridge between the harness and those subagents:

  --emit-tasks   render the blind judging tasks (same rubric + order convention as a2_gate3_judge.py)
                 into batch files; the orchestrator hands each batch to a `model=sonnet, effort=medium`
                 subagent, which returns a JSON map {cell_id: verdict}.
  --fold         fold the collected verdicts back into a RESULTS_CLAUDE_*.json in the harness's exact
                 raws schema, so `a2_gate3_judge.py --merge` combines it with the HTTP judges and
                 `--rescore` aggregates the full quorum.

Cell id convention (stable join key back to the harness):
  pairwise:  "<idx>:<order>"   order in {cand_first, ref_first}   winner "1"->arm shown first
  pointwise: "<idx>:<arm>"     arm in {cand, ref}                 score 0-10

Usage:
  python a2_gate3_worker.py --emit-tasks --mode pairwise --batches 4
  #   -> runs/a2/gate3/claude_tasks_pairwise_b{0..3}.json   (feed each to a Sonnet-5 subagent)
  python a2_gate3_worker.py --fold verdicts.json --mode pairwise
  #   -> runs/a2/gate3/RESULTS_CLAUDE_pairwise_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from a2_gate3_judge import (ARM_CAND, ARM_REF, OUT, PROMPT_TEXT, SYSTEM_POINTWISE, SYSTEM_PROMPT,
                            _user_msg, _user_pointwise, comparison_set, load_arm, tier_of)

# The instruction wrapper every worker subagent receives, so its output maps back cleanly.
WORKER_INSTRUCTIONS_PAIRWISE = (
    "You are the Claude judge in a blind writing-quality panel. Read the tasks file, apply the RUBRIC "
    "below to EACH task independently, and return your verdicts.\n\n"
    "=== RUBRIC (identical for every task) ===\n" + SYSTEM_PROMPT + "\n\n"
    "=== OUTPUT ===\nReturn ONE strict-JSON object mapping each task's cell_id to its verdict:\n"
    '{"<cell_id>": {"winner": "1"|"2"|"tie", "reason": "<=35 words"}, ...}\n'
    "Judge every task in the file. No prose outside the JSON."
)
WORKER_INSTRUCTIONS_POINTWISE = (
    "You are the Claude judge scoring writing quality. Read the tasks file, apply the RUBRIC below to "
    "EACH task independently, and return your scores.\n\n"
    "=== RUBRIC (identical for every task) ===\n" + SYSTEM_POINTWISE + "\n\n"
    "=== OUTPUT ===\nReturn ONE strict-JSON object mapping each task's cell_id to its score:\n"
    '{"<cell_id>": {"score": <0-10>, "reason": "<=35 words"}, ...}\n'
    "Score every task in the file. No prose outside the JSON."
)


def build_tasks(mode: str, set_name: str):
    """Return (idxs, foldmap). foldmap holds FULL info (opaque id -> idx/order/arm/arm1); the
    judge-facing task carries ONLY {id, user} so nothing reveals which response is the candidate.
    Opaque ids + a shuffled order keep the panel truly blind (the earlier "<idx>:<order>" ids and the
    arm1/arm_cand fields de-blinded it -- a diligent judge could read the mapping)."""
    import random
    rng = random.Random(hash((mode, set_name, 424242)) & 0xffffffff)
    cand, ref = load_arm(ARM_CAND), load_arm(ARM_REF)
    idxs = comparison_set(cand, ref, set_name)
    foldmap = []
    if mode == "pairwise":
        for i in idxs:
            ct, rt = cand[i]["text"], ref[i]["text"]
            for order, (a, b, arm1) in (("cand_first", (ct, rt, "cand")), ("ref_first", (rt, ct, "ref"))):
                foldmap.append(dict(idx=i, tier=tier_of(i), order=order, arm1=arm1,
                                    user=_user_msg(PROMPT_TEXT[i], a, b)))
    else:
        for i in idxs:
            for arm, rec in (("cand", cand[i]), ("ref", ref[i])):
                foldmap.append(dict(idx=i, tier=tier_of(i), arm=arm,
                                    user=_user_pointwise(PROMPT_TEXT[i], rec["text"])))
    rng.shuffle(foldmap)                                   # break any positional pattern
    for k, t in enumerate(foldmap):
        t["id"] = f"t{k:03d}"                              # opaque, carries no arm/order signal
    return idxs, foldmap


def emit_tasks(mode: str, set_name: str, batches: int) -> None:
    idxs, foldmap = build_tasks(mode, set_name)
    instr = WORKER_INSTRUCTIONS_PAIRWISE if mode == "pairwise" else WORKER_INSTRUCTIONS_POINTWISE
    OUT.mkdir(parents=True, exist_ok=True)
    # fold-map (NEVER shown to a judge): opaque id -> arm/order mapping, kept for --fold
    mp = OUT / f"claude_foldmap_{mode}.json"
    mp.write_text(json.dumps(dict(mode=mode, set=set_name, idxs=idxs, map={
        t["id"]: {k: t[k] for k in t if k not in ("user",)} for t in foldmap}), indent=2), encoding="utf-8")

    groups = [foldmap[b::batches] for b in range(batches)]
    written = []
    for b, g in enumerate(groups):
        # judge-facing file: ONLY id + user (blind). No arm names, no idx, no order.
        p = OUT / f"claude_tasks_{mode}_b{b}.json"
        p.write_text(json.dumps(dict(
            meta=dict(mode=mode, batch=b, of=batches, note="blind panel; ids are opaque"),
            instructions=instr,
            tasks=[dict(id=t["id"], user=t["user"]) for t in g]), indent=2, ensure_ascii=False), encoding="utf-8")
        written.append((p, len(g)))
    print(f"emitted {len(foldmap)} {mode} tasks across {batches} blind batch files ({len(idxs)} prompts):")
    for p, n in written:
        print(f"  {p}  ({n} tasks)")
    print(f"fold-map (judges never see this): {mp}")
    print("\nHand each batch to a Claude subagent (model=sonnet, effort=medium): "
          "'Read <batch file>, follow its `instructions`, judge every task, return the JSON map.'")


def fold(verdicts_path: Path, mode: str, set_name: str) -> None:
    verdicts = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    mp = json.loads((OUT / f"claude_foldmap_{mode}.json").read_text(encoding="utf-8"))
    idxs, fmap = mp["idxs"], mp["map"]
    missing = [tid for tid in fmap if tid not in verdicts]
    if missing:
        print(f"  WARN: {len(missing)} tasks missing from verdicts (will be errored): {missing[:6]}...")

    raws = []
    for tid, t in fmap.items():
        v = verdicts.get(tid)
        if mode == "pairwise":
            if v and str(v.get("winner", "")).lower() in ("1", "2", "tie"):
                w, reason, err = str(v["winner"]).lower(), str(v.get("reason", ""))[:200], None
            else:
                w, reason, err = "err", "", "missing/invalid verdict"
            raws.append(dict(judge="claude", idx=t["idx"], tier=t["tier"], order=t["order"],
                             arm1=t["arm1"], winner=w, reason=reason, error=err))
        else:
            try:
                score = max(0.0, min(10.0, float(v["score"]))) if v else None
                err = None if score is not None else "missing"
            except (TypeError, ValueError, KeyError):
                score, err = None, "invalid score"
            raws.append(dict(judge="claude", idx=t["idx"], tier=t["tier"], arm=t["arm"],
                             score=score, reason=str((v or {}).get("reason", ""))[:200], error=err))

    OUT.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = OUT / f"RESULTS_CLAUDE_{mode}_{ts}.json"
    out.write_text(json.dumps(dict(
        meta=dict(ts=ts, mode=mode, set=set_name, idxs=idxs, arm_cand=ARM_CAND, arm_ref=ARM_REF,
                  judges=["claude"], models={"claude": "claude-sonnet-5 @ medium (worker)"}),
        raws=raws), indent=2), encoding="utf-8")
    ok = sum(1 for r in raws if not r["error"])
    print(f"folded {ok}/{len(raws)} verdicts -> {out}")
    print(f"next:  python a2_gate3_judge.py --rescore {out}    # claude-only")
    print(f"  or:  python a2_gate3_judge.py --merge <API_RESULTS.json> {out}   # full quorum")


def main() -> int:
    ap = argparse.ArgumentParser(description="A2 Gate 3 Claude-worker judge bridge")
    ap.add_argument("--emit-tasks", action="store_true")
    ap.add_argument("--fold", metavar="verdicts.json")
    ap.add_argument("--mode", default="pairwise", choices=["pairwise", "pointwise"])
    ap.add_argument("--set", default="prose", choices=["prose", "all", "mild"])
    ap.add_argument("--batches", type=int, default=4)
    args = ap.parse_args()
    if args.emit_tasks:
        emit_tasks(args.mode, args.set, args.batches)
    elif args.fold:
        fold(Path(args.fold), args.mode, args.set)
    else:
        ap.print_help(); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
