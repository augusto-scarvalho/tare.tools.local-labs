"""CLI — plan, run, analyze, report.

    python -m model_lifecycle.cli plan    --quants Q4_K_M,Q5_K_M --ncmoe 8,10
    python -m model_lifecycle.cli run     --plan-id sweep-1
    python -m model_lifecycle.cli analyze --plan-id sweep-1 --role infra
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .analysis.gates import Gates, evaluate
from .analysis.scoring import ROLE_WEIGHTS, score_all
from .control_plane.guard import Envelope
from .control_plane.planner import Axis, build_plan
from .servers.llama_cpp import LlamaCppAdapter, ServerProfile
from .storage.database import Store
from .workloads.throughput import run_config

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "runs" / "lifecycle.db"
MODEL_DIR = "/home/augus/models/qwen36-35b-a3b"
PROMPT = ("Explain, in about 120 words, why memory bandwidth rather than raw compute "
          "usually limits token generation on a single consumer GPU. Be concrete.")


def _axes(args) -> list[Axis]:
    return [
        Axis("quant", tuple(args.quants.split(","))),
        Axis("ncmoe", tuple(int(x) for x in args.ncmoe.split(","))),
        Axis("kv", tuple(args.kv.split(","))),
        Axis("ctx", (args.ctx,)),
    ]


def cmd_plan(args) -> int:
    plan = build_plan(_axes(args), plan_id=args.plan_id, seed=args.seed)
    store = Store(args.db)
    store.emit("plan_created", args.plan_id, configs=len(plan.configs), seed=args.seed)
    done = store.completed_config_ids(args.plan_id)
    print(f"plan {args.plan_id}: {len(plan.configs)} configs, {len(done)} already done")
    for c in plan.pending(done):
        print("  ", c.config_id)
    store.close()
    return 0


def cmd_run(args) -> int:
    plan = build_plan(_axes(args), plan_id=args.plan_id, seed=args.seed)
    store = Store(args.db)
    done = store.completed_config_ids(args.plan_id)
    pending = plan.pending(done)
    print(f"plan {args.plan_id}: {len(pending)} pending of {len(plan.configs)}")
    if args.dry_run:
        for c in pending:
            print("  would run:", c.config_id)
        store.close()
        return 0

    adapter = LlamaCppAdapter()
    env = Envelope()
    for i, cfg in enumerate(pending, 1):
        model = f"{MODEL_DIR}/Qwen3.6-35B-A3B-{cfg.get('quant')}.gguf"
        profile = ServerProfile(model_path=model, port=args.port,
                                n_cpu_moe=cfg.get("ncmoe"), ctx_size=cfg.get("ctx"),
                                cache_type_k=cfg.get("kv"), cache_type_v=cfg.get("kv"))
        print(f"[{i}/{len(pending)}] {cfg.config_id}", flush=True)
        result = run_config(adapter, profile, config_id=cfg.config_id, prompt=PROMPT,
                            repetitions=args.repetitions, max_tokens=args.max_tokens,
                            envelope=env)
        store.record_run(result.as_dict(), plan_id=args.plan_id, model_path=model)
        print(f"    -> {result.verdict} {result.reason or ''} "
              f"| vram {result.min_free_vram_mb}MB", flush=True)
    store.close()
    return 0


def cmd_analyze(args) -> int:
    store = Store(args.db)
    runs = store.runs(args.plan_id)
    if not runs:
        print("no runs recorded")
        store.close()
        return 1

    # Gates BEFORE scores, always: ranking something unusable is what gates prevent.
    eligible, rejected = [], []
    for r in runs:
        g = evaluate(r, Gates())
        (eligible if g.eligible else rejected).append((r, g))

    print(f"{len(runs)} runs | {len(eligible)} eligible | {len(rejected)} not")
    for r, g in rejected:
        print(f"  INELIGIBLE {r['config_id']}: {g.failures[0]}")

    scores = score_all([r for r, _ in eligible])
    ranked = sorted(scores.items(), key=lambda kv: kv[1].for_role(args.role), reverse=True)
    print(f"\nrank by role={args.role} (weights {ROLE_WEIGHTS[args.role]})")
    for cid, s in ranked:
        print(f"  {s.for_role(args.role):6.1f}  {cid}"
              f"   speed={s.speed:.0f} fit={s.fit:.0f} stab={s.stability:.0f}")
    if args.json:
        print(json.dumps({cid: s.as_dict() | {"role_score": s.for_role(args.role)}
                          for cid, s in ranked}, indent=2))
    store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="model_lifecycle")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("plan", cmd_plan), ("run", cmd_run)):
        sp = sub.add_parser(name)
        sp.add_argument("--plan-id", default="sweep-1")
        sp.add_argument("--quants", default="UD-Q4_K_M,UD-Q5_K_M")
        sp.add_argument("--ncmoe", default="8,10")
        sp.add_argument("--kv", default="q8_0")
        sp.add_argument("--ctx", type=int, default=8192)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--port", type=int, default=8080)
        sp.add_argument("--repetitions", type=int, default=3)
        sp.add_argument("--max-tokens", type=int, default=1500)
        sp.add_argument("--dry-run", action="store_true")
        sp.set_defaults(func=fn)

    sa = sub.add_parser("analyze")
    sa.add_argument("--plan-id", default="sweep-1")
    sa.add_argument("--role", default="infra", choices=sorted(ROLE_WEIGHTS))
    sa.add_argument("--json", action="store_true")
    sa.set_defaults(func=cmd_analyze)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
