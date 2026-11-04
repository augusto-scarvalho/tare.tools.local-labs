"""JSON -> TSV, so a reader spends tokens on numbers instead of punctuation.

The run records here are arrays of flat objects written with `indent=2`. Reading one costs
several KB of braces, quotes and repeated key names to recover a handful of values. TSV
carries the same information at roughly a tenth the size, and stays greppable.

    python json2tsv.py runs/ab-pinning-qwen36-35b/records.json
    python json2tsv.py records.json --cols arm,ncmoe,prompt_tps --where verdict=OK
    python json2tsv.py records.json --agg arm --mean prompt_tps

`--agg` exists because the usual question is not "show me 18 rows" but "what is the mean
per arm", and computing that here costs nothing while reading 18 rows to do it by eye
costs the whole file.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys


def flatten(obj: dict, prefix: str = "") -> dict:
    """One level of nesting is enough for these records; deeper structures are summarised
    rather than exploded, because a column that is itself a table defeats the purpose."""
    out = {}
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            # The stats blocks (`{"mean": ..., "cv": ...}`) are the common case: keep the
            # mean, which is what every caller wanted, and note the rest exists.
            if "mean" in v:
                out[key] = v.get("mean")
                if v.get("cv") is not None:
                    out[f"{key}_cv"] = v["cv"]
            else:
                out[key] = f"<{len(v)} keys>"
        elif isinstance(v, list):
            out[key] = f"<{len(v)} items>"
        else:
            out[key] = v
    return out


def load(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return [flatten(r) for r in data if isinstance(r, dict)]


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v).replace("\t", " ").replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--cols", help="comma-separated columns to keep, in order")
    ap.add_argument("--where", action="append", default=[],
                    help="key=value filter, repeatable")
    ap.add_argument("--agg", help="group by this column")
    ap.add_argument("--mean", action="append", default=[],
                    help="column to average within each --agg group, repeatable")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = load(pathlib.Path(args.path))
    for w in args.where:
        k, _, v = w.partition("=")
        rows = [r for r in rows if str(r.get(k)) == v]
    if not rows:
        print("(no rows after filters)")
        return 0

    if args.agg:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(str(r.get(args.agg)), []).append(r)
        metrics = args.mean or [c for c in rows[0]
                                if isinstance(rows[0][c], (int, float))
                                and c != args.agg]
        print("\t".join([args.agg, "n"] + metrics))
        for g, rs in sorted(groups.items()):
            cells = []
            for m in metrics:
                vals = [r[m] for r in rs if isinstance(r.get(m), (int, float))]
                cells.append(f"{st.fmean(vals):.4g}" if vals else "")
            print("\t".join([g, str(len(rs))] + cells))
        return 0

    cols = (args.cols.split(",") if args.cols
            else list(dict.fromkeys(k for r in rows for k in r)))
    print("\t".join(cols))
    for r in (rows[:args.max_rows] if args.max_rows else rows):
        print("\t".join(fmt(r.get(c)) for c in cols))
    return 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        import io
        import tempfile
        from contextlib import redirect_stdout

        recs = [{"arm": "base", "ncmoe": 24, "verdict": "OK",
                 "gen_tps": {"mean": 44.2, "cv": 0.01}, "failures": [1, 2],
                 "nested": {"a": 1, "b": 2}},
                {"arm": "pin", "ncmoe": 24, "verdict": "OK",
                 "gen_tps": {"mean": 44.4, "cv": 0.02}, "failures": [],
                 "nested": {"a": 1, "b": 2}},
                {"arm": "pin", "ncmoe": 24, "verdict": "REJECTED",
                 "gen_tps": {"mean": 40.0, "cv": 0.9}, "failures": [], "nested": {}}]
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "r.json"
            p.write_text(json.dumps(recs), encoding="utf-8")

            # A stats block collapses to its mean, which is what every caller wanted.
            flat = load(p)
            assert flat[0]["gen_tps"] == 44.2 and flat[0]["gen_tps_cv"] == 0.01
            # Lists and opaque dicts are summarised, never exploded into the row.
            assert flat[0]["failures"] == "<2 items>"
            assert flat[0]["nested"] == "<2 keys>"

            sys.argv = ["x", str(p), "--where", "verdict=OK", "--cols", "arm,gen_tps"]
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            out = buf.getvalue().strip().splitlines()
            assert out[0] == "arm\tgen_tps" and len(out) == 3, out

            sys.argv = ["x", str(p), "--where", "verdict=OK", "--agg", "arm",
                        "--mean", "gen_tps"]
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            got = buf.getvalue().strip().splitlines()
            assert got[0] == "arm\tn\tgen_tps"
            assert "base\t1\t44.2" in got[1] or "base\t1\t44.2" in got[2], got

            # A filter that matches nothing must say so, not print an empty table that
            # reads as "measured, and it was zero".
            sys.argv = ["x", str(p), "--where", "arm=nope"]
            buf = io.StringIO()
            with redirect_stdout(buf):
                main()
            assert "no rows after filters" in buf.getvalue()

        print("json2tsv self-check OK")
        raise SystemExit(0)
    raise SystemExit(main())
