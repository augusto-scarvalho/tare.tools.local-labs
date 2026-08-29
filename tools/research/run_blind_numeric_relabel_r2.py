#!/usr/bin/env python3
"""Blind target-policy amendment of the complete R1 numeric relabel."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any

ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.analysis.a2_stats import numeric_equal
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file

TASK_ID="BACKLOG-BLIND-NUMERIC-RELABEL-02"
SOURCE=ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01"
TARGETS={
    ("answer_only","gsm8k/608"):None,
    ("answer_only","gsm8k/633"):"1 5/7",
    ("answer_only","gsm8k/656"):None,
    ("answer_only","gsm8k/774"):None,
    ("full_trace","gsm8k/774"):None,
    ("answer_only","gsm8k/865"):None,
    ("full_trace","gsm8k/865"):None,
}
LABEL_RE=re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\d+/\d+|\d+\s+\d+/\d+)$")
HOST_INPUTS={
    "config/research_backlog_admissions/BACKLOG-BLIND-NUMERIC-RELABEL-02.json":"75b60e1b5aa6d420e6b3b0b39b2b4882fdce56fbf0e1efb21b366d206ee9c424",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/PRE_REGISTRATION.md":"534e84379c609be00db4b28347776fe91bf45720b59a8a2352c7db30a6debcad",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/receipt.json":"a5b79f6733e60864c7ea9d962376575b28bd36b2d2a2a56e19a23e25fd57e0e5",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/final_blind_labels.jsonl":"481e903710dad71a921ed5844197750f81fe053f1be7914cfadf823bf8bb177f",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/sealed_mapping.json":"409b2227f0e9906e2f605216a8f77694ab104dc284ff620b0fbc1b8a8d3b95b8",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/inter_rater_agreement.json":"8f20f41a379e6330fc07f7ef2a54cc5e1732bd81de22736df3fa352c1a755839",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/REVIEW.json":"9009c48834928ba6a647c5834be0a693dc001431f71e0f6cac53192c157dd7a5",
}

def write_json(path:pathlib.Path,value:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")

def write_jsonl(path:pathlib.Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")

def read_jsonl(path:pathlib.Path)->list[dict[str,Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def verify()->list[pathlib.Path]:
    paths=[]
    for relative,expected in HOST_INPUTS.items():
        path=ROOT/relative;actual=sha256_file(path)
        if actual!=expected:raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        paths.append(path)
    return paths

def selected(mapping:list[dict[str,Any]])->list[dict[str,Any]]:
    rows=[m for m in mapping if (m["arm"],m["task_id"]) in TARGETS and m["source"]=="trace"]
    if len(rows)!=7 or {(r["arm"],r["task_id"]) for r in rows}!=set(TARGETS):raise ValueError("amendment selection mismatch")
    return rows

def prepare(outdir:pathlib.Path)->None:
    verify();raw=outdir/"raw"
    if any(raw.iterdir()):raise RuntimeError("raw directory is not empty")
    mapping=json.loads((SOURCE/"raw/sealed_mapping.json").read_text(encoding="utf-8"));rows=selected(mapping)
    public=[{"record_id":r["record_id"],"question":r["question"],"response":r["response"]} for r in rows]
    write_jsonl(raw/"blind_inputs/amendment.jsonl",public)
    write_json(raw/"AMENDMENT_INSTRUCTIONS.json",{
        "task":"Extract only a numeric value explicitly concluded for the exact quantity asked by the question.",
        "frozen_policy":["Do not solve the problem and do not inspect gold, arm, mapping, sources, R1 labels or reviews.",
            "If the response concludes only a different intermediate or subquantity and never concludes the requested quantity, return null.",
            "A wrong value explicitly concluded for the requested quantity remains that wrong value.",
            "Preserve mixed numbers as written, normalizing whitespace only; do not convert them to improper fractions."],
        "output_schema":{"record_id":"opaque ID","concluded_value":"numeric string, mixed number, or null","confidence":"high|medium|low","rationale":"short target-policy explanation"},
        "records":len(public),"gold_or_arm_fields_exposed":False})
    write_json(raw/"blind_label_packet.json",{"records":len(public),"task_families":5,"exposed_fields":["record_id","question","response"],"gold_or_arm_fields_exposed":False})

def finalize(outdir:pathlib.Path)->None:
    sources=verify();raw=outdir/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic()
    mapping=json.loads((SOURCE/"raw/sealed_mapping.json").read_text(encoding="utf-8"));amended_rows=selected(mapping)
    output=raw/"blind_outputs/amendment.jsonl";labels=read_jsonl(output);by_id={r.get("record_id"):r for r in labels}
    expected_ids={r["record_id"] for r in amended_rows}
    if len(labels)!=len(by_id) or set(by_id)!=expected_ids:raise ValueError("amendment output coverage mismatch")
    for rid,row in by_id.items():
        value=row.get("concluded_value")
        if value is not None and (not isinstance(value,str) or not LABEL_RE.fullmatch(value.strip())):raise ValueError(f"invalid amendment value: {rid}: {value!r}")
        if row.get("confidence") not in {"high","medium","low"} or not str(row.get("rationale","")).strip():raise ValueError(f"invalid amendment metadata: {rid}")
        if value is not None:row["concluded_value"]=" ".join(value.split())
    policy_failures=[]
    for m in amended_rows:
        actual=by_id[m["record_id"]]["concluded_value"];expected=TARGETS[(m["arm"],m["task_id"])]
        if actual!=expected:policy_failures.append({"record_id":m["record_id"],"expected":expected,"actual":actual})
    base=read_jsonl(SOURCE/"raw/final_blind_labels.jsonl");base_by_id={r["record_id"]:r["concluded_value"] for r in base}
    before=dict(base_by_id)
    for rid,row in by_id.items():base_by_id[rid]=row["concluded_value"]
    final=[{"record_id":m["record_id"],"concluded_value":base_by_id[m["record_id"]]} for m in mapping]
    write_jsonl(raw/"final_blind_labels.jsonl",final)
    untouched=[rid for rid in before if rid not in expected_ids and before[rid]!=base_by_id[rid]]
    scored=[]
    for m in mapping:
        value=base_by_id[m["record_id"]];scored.append({"record_id":m["record_id"],"source":m["source"],"arm":m["arm"],"task_id":m["task_id"],"gold":m["gold"],"concluded_value":value,"correct":bool(numeric_equal(value,m["gold"]))})
    write_jsonl(raw/"sealed_scored_labels.jsonl",scored)
    aggregates={}
    for source,arms in (("trace",("answer_only","full_trace")),("q8",("f16","q8"))):
        aggregates[source]={arm:{"correct":sum(r["correct"] for r in scored if r["source"]==source and r["arm"]==arm),"n":sum(1 for r in scored if r["source"]==source and r["arm"]==arm)} for arm in arms}
    agreement=json.loads((SOURCE/"raw/inter_rater_agreement.json").read_text(encoding="utf-8"))
    metrics={"r1_sources_verified":True,"final_labels":len(final),"gold_or_arm_fields_exposed":False,
        "amended_policy_cases":len({m["task_id"] for m in amended_rows}),"amended_records":len(amended_rows),
        "inherited_exact_inter_rater_agreement":agreement["agreement"],"unresolved_target_policy_cases":len(policy_failures),
        "unregistered_numeric_recodings":int(by_id[next(m["record_id"] for m in amended_rows if m["task_id"]=="gsm8k/633")]["concluded_value"]!="1 5/7"),
        "unresolved_amendments":len(policy_failures)+len(untouched),"untouched_label_mutations":len(untouched),
        "descriptive_aggregates_not_scientifically_authorized":aggregates}
    write_json(raw/"actual_scores.json",metrics);write_json(raw/"policy_validation.json",{"failures":policy_failures,"untouched_mutations":untouched})
    write_json(raw/"inter_rater_agreement.json",{"inherited_from":SOURCE.name,"sha256":sha256_file(SOURCE/"raw/inter_rater_agreement.json"),**agreement})
    attestation=raw/"rater_attestation.json"
    write_json(raw/"rater_provenance.json",{"inherited_r1_receipt_sha256":sha256_file(SOURCE/"raw/receipt.json"),"amendment_attestation":json.loads(attestation.read_text(encoding="utf-8"))})
    write_json(raw/"independent_evaluation.json",{"descriptive_only":True,"aggregates":aggregates,"downstream_audit_required":True})
    write_json(raw/"scorer_hashes.json",{"protocol_runner":{"sha256":sha256_file(pathlib.Path(__file__).resolve())},"automated_numeric_scorer_used":False})
    defs={"source_integrity":("r1_sources_verified","eq",True),"label_coverage":("final_labels","eq",768),"amendment_blinding":("gold_or_arm_fields_exposed","eq",False),
        "amendment_coverage":("amended_policy_cases","eq",5),"inherited_agreement":("inherited_exact_inter_rater_agreement","ge",.85),
        "target_policy":("unresolved_target_policy_cases","eq",0),"representation_policy":("unregistered_numeric_recodings","eq",0),"adjudication":("unresolved_amendments","eq",0)}
    gates={}
    for gate,(metric,op,threshold) in defs.items():
        actual=metrics[metric];passed=actual==threshold if op=="eq" else actual>=threshold;gates[gate]={"metric":metric,"operator":op,"threshold":threshold,"actual":actual,"pass":passed}
    evidence={"acceptance_gates":"raw/receipt.json","blind_label_packet":"raw/blind_label_packet.json","independent_evaluation":"raw/independent_evaluation.json",
        "inter_rater_agreement":"raw/inter_rater_agreement.json","provenance":"raw/receipt.json","rater_provenance":"raw/rater_provenance.json",
        "raw_samples":"raw/final_blind_labels.jsonl","receipt_fingerprint":"raw/receipt.json","scorer_hashes":"raw/scorer_hashes.json"}
    inputs=[*sources,output,attestation,*sorted(p for p in raw.rglob("*") if p.is_file() and p.name!="receipt.json")]
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=inputs,packages=[],runtime={"execution_mode":"blind_target_policy_amendment","amended_records":7,"new_model_inference_under_test":False})
    complete,errors=provenance_complete(provenance)
    if not complete:raise RuntimeError(f"incomplete provenance: {errors}")
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write_json(raw/"receipt.json",receipt)
    failed=[name for name,g in gates.items() if not g["pass"]];claim="BLIND_NUMERIC_RELABEL_COMPLETED_R2" if not failed else "BLIND_NUMERIC_RELABEL_NOT_VALIDATED_R2"
    (outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nAmended `{len(amended_rows)}` records across five audited task families; preserved `{len(final)-len(amended_rows)}` labels. Failed gates: `{', '.join(failed) if failed else 'none'}`. Downstream aggregates remain descriptive.\n",encoding="utf-8",newline="\n")

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--finalize",action="store_true");p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args()
    prepare(a.outdir.resolve()) if a.prepare else finalize(a.outdir.resolve());return 0
if __name__=="__main__":raise SystemExit(main())
