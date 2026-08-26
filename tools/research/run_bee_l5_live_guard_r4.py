#!/usr/bin/env python3
"""Byte-piece normalization for frozen BEE-L5 R3 protocol."""
from __future__ import annotations
import argparse, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.research import run_bee_l5_live_guard as core
TASK_ID = "BACKLOG-BEE-L5-LIVE-GUARD-04"


def tokenize(text: str):
    response = core.post_json("/tokenize", {"content": text, "with_pieces": True})
    result = []
    for token in response.get("tokens") or []:
        piece = token.get("piece") if isinstance(token, dict) else None
        if isinstance(piece, str): result.append(piece)
        elif isinstance(piece, list) and all(isinstance(value, int) and 0 <= value <= 255 for value in piece): result.append(bytes(piece).decode("utf-8", errors="replace"))
        else: raise RuntimeError(f"unexpected token piece: {token}")
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);args=parser.parse_args()
    core.TASK_ID=TASK_ID;core.tokenize=tokenize
    core.EXPECTED.update({
        ROOT/"config/research_backlog_admissions/BACKLOG-BEE-L5-LIVE-GUARD-03.json":"88995a1931c9137d6d5a07a0fd1894d3f24801e36076f95c680c0d6e757f2020",
        ROOT/"runs/research/BACKLOG-BEE-L5-LIVE-GUARD-03/PRE_REGISTRATION.md":"89e7fb7ad4e521426f48c2dc8965fc692c21e09cfe3bf8104da6fe435b10055b",
        ROOT/"runs/research/BACKLOG-BEE-L5-LIVE-GUARD-03/ABORTED.md":"7d006d848d240ce27454381a35a8fd2559532062fbff49b916325e83664658b2",
        ROOT/"tools/research/run_bee_l5_live_guard_r3.py":"f32ab039a1e1b39f7c70b0d7ac87cc9b9f4ec6ef141848a3c59686142530cdea",
        ROOT/"config/research_backlog_admissions/BACKLOG-BEE-L5-LIVE-GUARD-04.json":"5b2e89b67efc0f1f30301df4cbeb8fa3629ab0745505cb64553af9c8d623d399",
        ROOT/"runs/research/BACKLOG-BEE-L5-LIVE-GUARD-04/PRE_REGISTRATION.md":"1695388605910a44dbf50bb5b0332d8519b169969b96776325cecc38cdb1ba59"})
    receipt,metrics=core.run(args.outdir.resolve());passed=all(row["pass"] for row in receipt["gates"].values());claim="BEE_L5_LIVE_GUARD_QUALIFIED_R4" if passed else "BEE_L5_FALSE_POSITIVE_CONFIRMED_R4";failed=[gate for gate,row in receipt["gates"].items() if not row["pass"]]
    (args.outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nReal teacher traces: {metrics['real_legitimate_traces']}; false alarms: {metrics['teacher_false_positives']} ({metrics['false_alarm_fpr']:.4%}). Live pathological baselines: {metrics['live_pathological_baselines']}; guard triggers/aborts: {metrics['stream_aborts_confirmed']}/25; median trigger token: {metrics['median_trigger_token']}; median savings: {metrics['median_token_savings']:.4%}; guard p95: {metrics['guard_p95_us_per_token']:.3f} us/token. Failed gates: {', '.join(failed) if failed else 'none'}. This is client-side streaming intervention, not server integration.\n",encoding="utf-8")
    print(json.dumps({"claim":claim,"metrics":metrics,"gates":receipt["gates"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
