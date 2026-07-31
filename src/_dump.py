import json
import pathlib
import sqlite3
import sys

db = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "../runs/lifecycle.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
print(f"{'config':<46}{'verdict':<10}{'gen t/s':>9}{'cv':>7}{'ttft p95':>10}{'vram':>8}{'load':>7}")
print("-" * 97)
for r in conn.execute("SELECT payload FROM runs ORDER BY id"):
    d = json.loads(r["payload"])
    g = d.get("gen_tps") or {}
    t = d.get("ttft") or {}
    print(f"{d['config_id'][:45]:<46}{d['verdict']:<10}"
          f"{(f'{g.get(chr(109)+chr(101)+chr(97)+chr(110)):.1f}' if g else '-'):>9}"
          f"{(f'{g.get(chr(99)+chr(118)):.3f}' if g else '-'):>7}"
          f"{(f'{t.get(chr(112)+chr(57)+chr(53)):.1f}' if t else '-'):>10}"
          f"{str(d.get('min_free_vram_mb') or '-'):>8}"
          f"{(f'{d.get(chr(108)+chr(111)+chr(97)+chr(100)+chr(95)+chr(115)+chr(101)+chr(99)+chr(111)+chr(110)+chr(100)+chr(115)):.1f}' if d.get('load_seconds') else '-'):>7}")
