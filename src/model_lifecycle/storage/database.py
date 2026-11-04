"""SQLite persistence + append-only event log.

Two rules from the plan, both structural rather than stylistic:

  * **Raw data is immutable.** Runs are inserted, never updated. Scores and reports
    are DERIVED and can be recomputed without re-running anything -- which is the
    whole reason a benchmark that costs minutes per config can afford to change its
    mind about scoring.
  * **Every state change emits an event.** Provenance is not a report you generate
    later; it is the log you kept while it happened.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    config_id     TEXT NOT NULL,
    plan_id       TEXT,
    model_path    TEXT,
    verdict       TEXT NOT NULL,
    reason        TEXT,
    payload       TEXT NOT NULL,          -- the full RunResult as JSON, verbatim
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_config ON runs(config_id);
CREATE INDEX IF NOT EXISTS runs_plan   ON runs(plan_id);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    subject    TEXT,
    data       TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind);
"""


class Store:
    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # --- events -------------------------------------------------------------
    def emit(self, kind: str, subject: str | None = None, **data: Any) -> None:
        self.conn.execute(
            "INSERT INTO events (kind, subject, data, created_at) VALUES (?,?,?,?)",
            (kind, subject, json.dumps(data, default=str), time.time()))
        self.conn.commit()

    # --- runs ---------------------------------------------------------------
    def record_run(self, result: dict, *, plan_id: str | None = None,
                   model_path: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (config_id, plan_id, model_path, verdict, reason, payload, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (result.get("config_id"), plan_id, model_path, result.get("verdict"),
             result.get("reason"), json.dumps(result, default=str), time.time()))
        self.conn.commit()
        self.emit("run_recorded", result.get("config_id"),
                  verdict=result.get("verdict"), plan_id=plan_id)
        return int(cur.lastrowid)

    def completed_config_ids(self, plan_id: str) -> set[str]:
        """Config ids already recorded for this plan.

        This is what makes a sweep resumable: an interrupted plan continues instead of
        restarting, and a config that already produced a verdict -- including REJECTED,
        which is a real answer -- is not paid for twice.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT config_id FROM runs WHERE plan_id = ?", (plan_id,))
        return {r["config_id"] for r in rows}

    def runs(self, plan_id: str | None = None) -> list[dict]:
        sql = "SELECT payload FROM runs"
        args: tuple = ()
        if plan_id:
            sql += " WHERE plan_id = ?"
            args = (plan_id,)
        sql += " ORDER BY id"
        return [json.loads(r["payload"]) for r in self.conn.execute(sql, args)]

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        s = Store(pathlib.Path(tmp) / "t.db")
        s.record_run({"config_id": "a", "verdict": "OK"}, plan_id="p1")
        s.record_run({"config_id": "b", "verdict": "REJECTED"}, plan_id="p1")
        assert s.completed_config_ids("p1") == {"a", "b"}, "resume set must include REJECTED"
        assert s.completed_config_ids("other") == set()
        assert len(s.runs("p1")) == 2
        s.emit("plan_started", "p1", configs=2)
        n = s.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        assert n == 3, f"expected 3 events (2 auto + 1 explicit), got {n}"
        s.close()
    print("storage self-check OK")
