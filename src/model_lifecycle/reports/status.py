"""Regenerate STATUS's empirical spine FROM THE STORE.

STATUS.md's numbers were hand-transcribed, which means they were born drifting from the data.
This produces the evidence table -- the noise floor and every paired comparison -- straight
out of the Store, where the backfill unified every experiment's records. It is the first
consumer to read what the ingest wrote, and the proof the ingest was worth doing: ask the
Store one question ("every A/B, paired, against the floor") instead of globbing nine record
shapes across `runs/ab-*/`.

What this OWNS: the numbers. What it does NOT own: the interpretation. The SETTLED prose in
STATUS.md -- what a doubled prefill MEANS, why the fork can be dropped -- is editorial and
stays authored by hand. This regenerates the evidence those paragraphs cite, so the prose can
never quietly disagree with the data again.

    python -m model_lifecycle.reports.status                 # print markdown
    python -m model_lifecycle.reports.status --json          # the machine blob
    python -m model_lifecycle.reports.status -o reports/EVIDENCE.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

from ..storage.database import Store
from .ab import analyse

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "runs" / "lifecycle.db"


def from_store(store: Store) -> dict:
    """Blob for the whole PLAN, sourced from the Store's `ingest:ab-*` plans."""
    return analyse(lambda d: store.runs(f"ingest:{d}"))


def _row_md(r: dict) -> str:
    flag = " ⚠" if r["within_floor"] else ""     # inside the last cell; legend explains ⚠
    ci = f"[{r['boot_ci'][0]:+.1f}, {r['boot_ci'][1]:+.1f}]"
    return (f"| `{r['a']} − {r['b']}` | {r['metric']} | {r['n']} | "
            f"{r['median_delta']:+.2f} ({r['median_pct']:+.2f}%) | "
            f"{r['sign_p']:.4f} | {ci} | {r['cliffs']:+.2f}{flag} |")


def render(blob: dict, *, db_name: str, n_runs: int, stamp: str) -> str:
    floor = blob["noise_floor_pct"]
    out: list[str] = []
    out.append("# STATUS — evidence (auto-generated)")
    out.append("")
    out.append(f"> Generated {stamp} by `model_lifecycle.reports.status` from **{db_name}** "
               f"({n_runs} runs). **Numbers only** — the interpretation lives in the authored "
               f"`STATUS.md`. Do not hand-edit this file; regenerate it.")
    out.append("")
    out.append("Every delta is `a − b`, paired by `(round, ncmoe)`, median (not mean). "
               "`sign_p` is the exact two-sided sign test (floor 0.031 at n=6); the CI is the "
               "seeded percentile bootstrap of the paired delta; `δ` is Cliff's delta.")
    out.append("")
    out.append(f"**Noise floor** (median |%| of the null A/B's prefill, true Δ = 0 by "
               f"construction): **{floor:.2f}%**. A prefill delta at or below this is flagged "
               f"⚠ and is not evidence, however tidy its median.")
    out.append("")
    for c in blob["comparisons"]:
        out.append(f"## {c['headline']}")
        out.append(f"`ingest:{c['dir']}` — {c['n_records']} records")
        out.append("")
        live = [r for r in c["rows"] if r]
        if live:
            out.append("| comparison | metric | n | Δ median | sign_p | boot CI95 | δ |")
            out.append("|---|---|---|---|---|---|---|")
            out.extend(_row_md(r) for r in live)
        elif c.get("rejection"):
            rej = c["rejection"]
            out.append(f"**{rej['rejected']}/{rej['of']} REJECTED** — no comparable metric "
                       f"produced. Dominant reason: `{rej['top_reason']}`.")
        else:
            out.append("_(no paired data)_")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--json", action="store_true", help="emit the machine blob")
    ap.add_argument("-o", "--out", help="write markdown to this path instead of stdout")
    args = ap.parse_args(argv)

    db = pathlib.Path(args.db)
    if not db.exists():
        print(f"no Store at {db} — run `python ingest_runs.py` first", file=sys.stderr)
        return 1
    store = Store(db)
    blob = from_store(store)
    n_runs = len(store.runs())
    store.close()

    if args.json:
        print(json.dumps(blob, indent=2))
        return 0

    stamp = datetime.date.today().isoformat()
    md = render(blob, db_name=db.name, n_runs=n_runs, stamp=stamp)
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md + "\n", encoding="utf-8")
        print(f"wrote {p} ({md.count(chr(10)) + 1} lines) from {db.name}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
