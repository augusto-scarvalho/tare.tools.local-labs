#!/usr/bin/env python3
"""Recompute DISTILL-01 from process-isolated real adapter generations."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file
from tools.research.run_adapter_requalification import DEFAULT_QA_PATH, FROZEN_QA_IDS, extract_gsm8k_pred, grade_qa, is_gsm8k_correct, load_qa_panel

TASK_ID="BACKLOG-DISTILL01-FLEET-REAL-01"
SOURCE=ROOT/"runs/research/BACKLOG-ADAPT-REQUAL-02/raw/samples.jsonl"
EXPECTED={
ROOT/"config/research_backlog_admissions/BACKLOG-DISTILL01-FLEET-REAL-01.json":"cdc85abf2dd596df8643a563abddc12dfc5ad2c82b8e8bd0d01e7b41df4542e5",
ROOT/"runs/research/BACKLOG-DISTILL01-FLEET-REAL-01/PRE_REGISTRATION.md":"e5fb8463bd4f6a5792643e942f2588ecab46b0b68eb5143d0ff7b3a0e887d002",
ROOT/"runs/research/BACKLOG-ADAPT-REQUAL-02/PRE_REGISTRATION.md":"9b528e0f9542778fd8dbbc2dcddf730a3e63f46aa130a2b5849e1f99181f0ffd",
ROOT/"runs/research/BACKLOG-ADAPT-REQUAL-02/raw/receipt.json":"8bc38d1f2cb5ef60f53ddb989e5c0aa1104b81359efbc5a2c8e4bbd0d92bc876",
SOURCE:"8900194aa5abc38092f7e5d99122c7322de8781c5aff4ef402d812fb6dfb2a8c",
ROOT/"runs/research/BACKLOG-ADAPT-REQUAL-02/raw/artifact_hashes.json":"b19fa60e5d122219934a1563cdf231dac0a847393327d35b214763711582c5fc",
ROOT/"runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/PRE_REGISTRATION.md":"dd5889aaf3767b67bd24dd805775dce3c0bf20b3291e600311adc7793bf86c10",
ROOT/"runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/RESULT.md":"a663e7807c06faf5aaa47ee26de64eae7ccbe0bf05e0580cc3f3b8e8e21ab3ab",
ROOT/"runs/research/DISTILL-01-FLEET-DISTILLATION-2026-08-25/raw/receipt.json":"e71f1831345356b6e1dc5d20f960b533daf1dc7e2f61d15fc869d430e120a8a9",
ROOT/"tools/probes/distill01_fleet_distillation.py":"ccadd6e28e8ad8bbb9c40e7f512aa3cb5f260f335c21c2711b37ab25169008cc",
DEFAULT_QA_PATH:"56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f"}
ARMS=("target_mlp_only","target_attn_only","target_all_linear")

def write(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def score(rows):
    qa={row["id"]:row for row in load_qa_panel(DEFAULT_QA_PATH,FROZEN_QA_IDS)}; output=[]
    for row in rows:
        if row["panel"]=="math": recomputed=is_gsm8k_correct(extract_gsm8k_pred(row["output_text"]),row["gold"])
        else: recomputed=grade_qa(qa[row["task_id"]],row["output_text"])[0]
        output.append({**row,"recomputed_correct":recomputed})
    counts={arm:{panel:sum(r["recomputed_correct"] for r in output if r["arm"]==arm and r["panel"]==panel) for panel in ("math","qa")} for arm in ARMS}
    fleet_math=counts["target_mlp_only"]["math"]; fleet_qa=counts["target_attn_only"]["qa"]
    mono=sum(counts["target_all_linear"].values()); fleet=fleet_math+fleet_qa
    return output,{"counts":counts,"fleet_math_correct":fleet_math,"fleet_qa_correct":fleet_qa,"fleet_total":fleet,"monolith_total":mono,"fleet_gain_over_monolith":(fleet-mono)/mono}

def run(outdir):
    raw=outdir/"raw"; started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); mono=time.monotonic()
    if any(raw.iterdir()): raise RuntimeError("raw not empty")
    ledger={}
    for path,expected in EXPECTED.items():
        actual=sha256_file(path)
        if actual!=expected: raise ValueError(f"hash mismatch {path}: {actual}")
        ledger[str(path.relative_to(ROOT).as_posix())]={"bytes":path.stat().st_size,"sha256":actual}
    all_rows=[json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    rows=[row for row in all_rows if row["arm"] in ARMS]
    rescored,scores=score(rows); match=all(r["correct"]==r["recomputed_correct"] for r in rescored)
    coverage=len(rows)==144 and all(sum(r["arm"]==a for r in rows)==48 for a in ARMS)
    receipt_source=json.loads((ROOT/"runs/research/BACKLOG-ADAPT-REQUAL-02/raw/receipt.json").read_text(encoding="utf-8"))
    source_verified=receipt_source.get("provenance_complete") is True and coverage
    with (raw/"samples.jsonl").open("w",encoding="utf-8") as stream:
        for row in rescored: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    write(raw/"actual_scores.json",scores); write(raw/"artifact_hashes.json",ledger)
    write(raw/"dataset_hashes.json",{"source_samples_sha256":EXPECTED[SOURCE],"selected_semantic_sha256":canonical_json_sha256(rescored)})
    write(raw/"source_execution_receipt.json",{"verified":source_verified,"receipt_sha256":EXPECTED[ROOT/"runs/research/BACKLOG-ADAPT-REQUAL-02/raw/receipt.json"]})
    write(raw/"falsifiable_hypothesis.json",{"gain_threshold":.20,"math_threshold":15,"qa_threshold":5})
    write(raw/"invariant_controls.json",{"arms":ARMS,"samples_per_arm":48,"routing":{"math":"target_mlp_only","qa":"target_attn_only"},"monolith":"target_all_linear"})
    decision=scores["fleet_gain_over_monolith"]>=.20 and scores["fleet_math_correct"]>=15 and scores["fleet_qa_correct"]>=5
    write(raw/"invalidation_rules.json",{"all_original_rules_pass":decision})
    write(raw/"failure_reproduction.json",{"historical":{"fleet_total":22,"monolith_total":18,"gain":.2222},"clean":scores,"historical_promotion_reproduced":decision})
    second_rows,second_scores=score(rows); independent=match and canonical_json_sha256(scores)==canonical_json_sha256(second_scores)
    write(raw/"independent_evaluation.json",{"independent_rescore_match":independent,"scores":second_scores})
    write(raw/"semantic_parity.json",{"stored_flags_match":match,"two_pass_scores_match":canonical_json_sha256(scores)==canonical_json_sha256(second_scores)})
    obs={"source_real_execution_verified":source_verified,"complete_required_arms":3 if coverage else 0,"complete_required_samples":len(rows),**scores,"independent_rescore_match":independent}
    defs={"source_execution":("source_real_execution_verified","eq",True),"arm_coverage":("complete_required_arms","eq",3),"sample_coverage":("complete_required_samples","eq",144),"fleet_gain":("fleet_gain_over_monolith","ge",.20),"math_specialist":("fleet_math_correct","ge",15),"qa_specialist":("fleet_qa_correct","ge",5),"independent_scoring":("independent_rescore_match","eq",True)}
    ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b}; gates={g:{"metric":m,"operator":o,"threshold":t,"actual":obs[m],"pass":ops[o](obs[m],t)} for g,(m,o,t) in defs.items()}
    files=[raw/name for name in ("actual_scores.json","artifact_hashes.json","dataset_hashes.json","failure_reproduction.json","falsifiable_hypothesis.json","independent_evaluation.json","invalidation_rules.json","invariant_controls.json","samples.jsonl","semantic_parity.json","source_execution_receipt.json")]
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,*files],packages=["pytest"],runtime={"execution_mode":"deterministic_routing_over_bound_real_generations"})
    complete,errors=provenance_complete(provenance)
    if not complete: raise ValueError(errors)
    evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","receipt_fingerprint":"raw/receipt.json","semantic_parity":"raw/semantic_parity.json","source_execution_receipt":"raw/source_execution_receipt.json"}
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":complete,"gates":gates,"evidence":evidence}; receipt["receipt_fingerprint"]=canonical_json_sha256(receipt); write(raw/"receipt.json",receipt); return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();print(json.dumps(run(a.outdir.resolve())["gates"],indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
