"""Backfill every JSON record in runs/ into the Store, as a unified queryable index.

The Store (control_plane record of truth) held 18 rows -- only the CLI sweep path ever
wrote to it. The research outgrew that path: the sharp one-off experiment scripts
(ab_isolate, the sweeps, taguchi, optimize, residency, quality, agentic_gate) each write
their own JSON to runs/ and never touch the Store. So a question like "every gpt-oss run
and its prefill, across all experiments" means globbing directories and knowing nine
different record shapes.

This ingester unifies them WITHOUT rewiring any experiment script (those work, and their
JSON-first form is defensible) and WITHOUT flattening the heterogeneity: each record is
stored with a synthesised config_id and a plan_id naming its source, while the record
itself is kept VERBATIM in the payload -- exactly the column the Store designed for it.

It honours the Store's rules:
  * IMMUTABLE runs -- it never updates or deletes; idempotency is by skipping a
    (config_id, plan_id) that is already present, so re-running ingests only what is new.
  * every source ingested emits an event, so the backfill is itself in the provenance log.

    python ingest_runs.py            # ingest into runs/lifecycle.db
    python ingest_runs.py --dry-run  # report what would be ingested, touch nothing
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from model_lifecycle.storage.database import Store                   # noqa: E402

ROOT = pathlib.Path(__file__).parent
RUNS = ROOT / "runs"
DB = RUNS / "lifecycle.db"

# Keys that DISCRIMINATE one config from another within a source, most-significant first --
# they make the config_id READABLE. They do NOT make it unique: replicates of one config
# (the L18 runs each config twice, with no round field) share every discriminator, and a
# config_id that collided would silently DROP the replicate as a duplicate. So the record's
# position in the file is ALWAYS appended: it is the true per-record key, stable because the
# experiment JSON is written once and never reordered, which keeps re-ingest idempotent.
_DISCRIMINATORS = ("tag", "arm", "label", "axis", "case", "task_id",
                   "model", "ncmoe", "ctx", "kv", "ubatch", "pin", "prefetch", "round")


def _config_id(rec: dict, stem: str, i: int) -> str:
    parts = [f"{k}={rec[k]}" for k in _DISCRIMINATORS if rec.get(k) is not None]
    body = "__".join(parts) if parts else "rec"
    return f"{stem}::{body}#{i}"


def _verdict(rec: dict) -> str:
    """Map each shape's success signal onto a verdict, without inventing one."""
    if rec.get("verdict"):
        return str(rec["verdict"])
    if "pass" in rec:                       # agentic_gate
        return "OK" if rec["pass"] else "FAIL"
    if rec.get("error"):
        return "FAIL"
    if "loaded" in rec:                     # optimize / residency
        return "OK" if rec["loaded"] else "REJECTED"
    return "OK"                             # sweeps/taguchi: a produced measurement


def _model_path(rec: dict) -> str | None:
    return rec.get("gguf") or rec.get("model_path") or rec.get("model")


def _records(path: pathlib.Path) -> list[dict]:
    """Normalise a source file to a list of record dicts. A dict-shaped file (agent_bench's
    phase_a/phase_b) is one record; a list is its elements."""
    d = json.load(open(path, encoding="utf-8"))
    if isinstance(d, list):
        return [r for r in d if isinstance(r, dict)]
    if isinstance(d, dict):
        return [d]
    return []


def _sources() -> list[tuple[str, pathlib.Path]]:
    """(plan_id, file). plan_id names the experiment so records group and query by it."""
    out = []
    for f in sorted(RUNS.glob("*.json")):
        out.append((f"ingest:{f.stem}", f))
    for f in sorted(RUNS.glob("ab-*/records.json")):
        out.append((f"ingest:{f.parent.name}", f))
    for f in sorted(RUNS.glob("quality/*.json")):
        if f.name.endswith("__samples.jsonl"):
            continue
        out.append((f"ingest:quality/{f.stem}", f))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = None if args.dry_run else Store(DB)
    total_new = total_seen = 0
    print(f"{'DRY-RUN: ' if args.dry_run else ''}ingesting runs/ -> {DB.name}\n")
    for plan_id, f in _sources():
        recs = _records(f)
        done = store.completed_config_ids(plan_id) if store else set()
        new = 0
        for i, rec in enumerate(recs):
            cid = _config_id(rec, f.stem, i)
            total_seen += 1
            if cid in done:
                continue
            done.add(cid)                    # guard against in-file id collisions too
            if store:
                # record_run reads the verdict column FROM this dict, and the column is
                # NOT NULL. Shapes like the sweeps and taguchi carry no verdict, so a
                # DERIVED one is injected -- only when absent, leaving existing verdicts
                # byte-identical. config_id is likewise injected for the column.
                payload = dict(rec, config_id=cid)
                if not payload.get("verdict"):
                    payload["verdict"] = _verdict(rec)
                store.record_run(payload, plan_id=plan_id, model_path=_model_path(rec))
            new += 1
        total_new += new
        if store and new:
            store.emit("source_ingested", plan_id, file=str(f.relative_to(ROOT)),
                       records=new)
        print(f"  {plan_id:34} {new:4d} new / {len(recs):4d} records")

    if store:
        native = len(store.runs()) - total_new
        print(f"\ningested {total_new} records across {len(_sources())} sources; "
              f"store now holds {len(store.runs())} runs "
              f"({native} pre-existing, {total_new} ingested)")
        store.close()
    else:
        print(f"\nDRY-RUN: {total_seen} records across {len(_sources())} sources "
              f"would be considered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
