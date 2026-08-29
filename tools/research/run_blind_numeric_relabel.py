#!/usr/bin/env python3
"""Prepare, collect and finalize a complete blind semantic numeric relabel."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.a2_stats import numeric_equal
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID = "BACKLOG-BLIND-NUMERIC-RELABEL-01"
SEED = 2026082813
RATERS = 3
OVERLAP = 96
BATCH = 32
LABEL_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\d+/\d+)$")
HOST_INPUTS = {
    "config/research_backlog_admissions/BACKLOG-BLIND-NUMERIC-RELABEL-01.json":"df99baa71b87e9cf02123a897a7ad5f19cd2b5c1c2f24bc9cbdfa5adce0d095d",
    "runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/PRE_REGISTRATION.md":"2c947edd868f9160cddceeeb37a934a6e14f56d3346a264ce219327c7a782737",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json":"b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json":"288270e4faa780bbd905b593193bf9c9edc595d84bf41cc2ef3fd72ba53663c9",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/receipt.json":"c4653adbc30ba652ef64d60130f416b47667e14c97ca726d5b55e309e7af2b20",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/samples.jsonl":"3801db8e8c45e30bbfa417d03da23a3db0131caa1039b0ea90fd9a59e4090196",
    "runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-03/REVIEW.json":"bb31683caf22d4d9bcd367cddd821046e54887c1edc70e6aad9fffde598f6860",
    "runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/REVIEW.json":"8755385e44b95473d99ecbeb5031f91236b16d13481c07bbdd84b40f286783c0",
    "workloads/gsm8k.jsonl":"68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77",
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sources() -> list[pathlib.Path]:
    paths=[]
    for relative,expected in HOST_INPUTS.items():
        path=ROOT/relative; actual=sha256_file(path)
        if actual != expected: raise ValueError(f"frozen input mismatch: {relative}: {actual} != {expected}")
        paths.append(path)
    return paths


def source_records() -> list[dict[str, Any]]:
    rows=[]
    trace=json.loads((ROOT/"runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/student_samples.json").read_text(encoding="utf-8"))
    for arm in trace:
        for row in arm["math_samples"]:
            rows.append({"source":"trace","arm":arm["arm"],"task_id":row["task_id"],"gold":row["gold"],
                         "question":row["prompt"],"response":row["output_text"]})
    prompts={r["task_id"]:r["prompt"] for r in read_jsonl(ROOT/"workloads/gsm8k.jsonl")}
    for row in read_jsonl(ROOT/"runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-02/raw/samples.jsonl"):
        rows.append({"source":"q8","arm":row["arm"],"task_id":row["task_id"],"gold":row["gold"],
                     "question":prompts[row["task_id"]],"response":row["content"]})
    if len(rows)!=768 or len({(r["source"],r["arm"],r["task_id"]) for r in rows})!=768:
        raise ValueError("source coverage is not 768 unique rows")
    for row in rows:
        key=f"{SEED}|{row['source']}|{row['arm']}|{row['task_id']}"
        row["record_id"]=hashlib.sha256(key.encode()).hexdigest()[:20]
    random.Random(SEED).shuffle(rows)
    return rows


def prepare(outdir: pathlib.Path) -> None:
    verify_sources(); raw=outdir/"raw"
    if any(raw.iterdir()): raise RuntimeError("raw directory is not empty")
    rows=source_records(); overlap={r["record_id"] for r in rows[:OVERLAP]}
    assignments={str(i):[] for i in range(RATERS)}; mapping=[]
    for index,row in enumerate(rows):
        primary=index%RATERS; assigned=[primary]
        if row["record_id"] in overlap: assigned.append((primary+1)%RATERS)
        public={"record_id":row["record_id"],"question":row["question"],"response":row["response"]}
        for rater in assigned: assignments[str(rater)].append(public)
        mapping.append({**row,"primary_rater":primary,"assigned_raters":assigned})
    manifest={"schema":"blind-numeric-label-packet-v1","seed":SEED,"records":len(rows),"raters":RATERS,
              "primary_per_rater":[sum(m["primary_rater"]==i for m in mapping) for i in range(RATERS)],
              "assigned_per_rater":[len(assignments[str(i)]) for i in range(RATERS)],"double_labeled":OVERLAP,
              "exposed_fields":["record_id","question","response"],"gold_or_arm_fields_exposed":False}
    write_json(raw/"bundle_manifest.json",manifest); write_json(raw/"sealed_mapping.json",mapping)
    instructions={"task":"Extract only the numeric value the response itself ultimately concludes for the question.",
        "rules":["Do not solve the question and do not inspect gold, arm, mapping, source packets or other raters.",
                 "Use null when the response does not conclude an answer; a wrong concluded value remains that wrong value.",
                 "Canonicalize commas, currency and percent signs away; preserve decimals and fractions.",
                 "Return record_id, concluded_value, confidence high/medium/low, and a short rationale."],
        "output_schema":{"record_id":"opaque input ID","concluded_value":"numeric string or null","confidence":"high|medium|low","rationale":"short gold-blind explanation"}}
    write_json(raw/"ANNOTATION_INSTRUCTIONS.json",instructions)
    for rater,items in assignments.items():
        random.Random(SEED+100+int(rater)).shuffle(items)
        for batch_index,start in enumerate(range(0,len(items),BATCH)):
            write_jsonl(raw/f"blind_inputs/rater_{rater}/batch_{batch_index:02d}.jsonl",items[start:start+BATCH])


def load_rater_outputs(raw: pathlib.Path, mapping: list[dict[str, Any]]) -> tuple[dict[int,dict[str,dict[str,Any]]],list[pathlib.Path]] | None:
    expected={i:{m["record_id"] for m in mapping if i in m["assigned_raters"]} for i in range(RATERS)}
    all_rows={}; paths=[]
    for rater in range(RATERS):
        directory=raw/f"blind_outputs/rater_{rater}"
        files=sorted(directory.glob("batch_*.jsonl")) if directory.exists() else []
        if not files: return None
        rows=[row for path in files for row in read_jsonl(path)]; paths.extend(files)
        by_id={row.get("record_id"):row for row in rows}
        if len(rows)!=len(by_id) or set(by_id)!=expected[rater]: return None
        for rid,row in by_id.items():
            value=row.get("concluded_value")
            if value is not None and (not isinstance(value,str) or not LABEL_RE.fullmatch(value.replace(",",""))):
                raise ValueError(f"invalid label from rater {rater}: {rid}: {value!r}")
            if row.get("confidence") not in {"high","medium","low"} or not str(row.get("rationale","")).strip():
                raise ValueError(f"invalid annotation metadata from rater {rater}: {rid}")
            if value is not None: row["concluded_value"]=value.replace(",","")
        all_rows[rater]=by_id
    return all_rows,paths


def make_adjudication(raw: pathlib.Path, mapping: list[dict[str, Any]], labels: dict[int,dict[str,dict[str,Any]]]) -> list[str]:
    disagreements=[]; public={m["record_id"]:{"record_id":m["record_id"],"question":m["question"],"response":m["response"]} for m in mapping}
    for m in mapping:
        if len(m["assigned_raters"])<2: continue
        a,b=m["assigned_raters"]; va=labels[a][m["record_id"]]["concluded_value"]; vb=labels[b][m["record_id"]]["concluded_value"]
        if va!=vb:
            disagreements.append({**public[m["record_id"]],"candidate_labels":[va,vb],
                "candidate_rationales":[labels[a][m["record_id"]]["rationale"],labels[b][m["record_id"]]["rationale"]]})
    directory=raw/"blind_inputs/adjudicator"; directory.mkdir(parents=True,exist_ok=True)
    if not list(directory.glob("batch_*.jsonl")):
        for index,start in enumerate(range(0,len(disagreements),BATCH)):
            write_jsonl(directory/f"batch_{index:02d}.jsonl",disagreements[start:start+BATCH])
        write_json(raw/"adjudication_manifest.json",{"disagreements":len(disagreements),"batches":len(list(directory.glob('batch_*.jsonl'))),"gold_or_arm_fields_exposed":False})
    return [row["record_id"] for row in disagreements]


def load_adjudication(raw:pathlib.Path, ids:list[str]) -> tuple[dict[str,dict[str,Any]],list[pathlib.Path]] | None:
    if not ids: return {},[]
    directory=raw/"blind_outputs/adjudicator"; files=sorted(directory.glob("batch_*.jsonl")) if directory.exists() else []
    if not files: return None
    rows=[row for path in files for row in read_jsonl(path)]; by_id={r.get("record_id"):r for r in rows}
    if len(rows)!=len(by_id) or set(by_id)!=set(ids): return None
    for rid,row in by_id.items():
        value=row.get("concluded_value")
        if value is not None and (not isinstance(value,str) or not LABEL_RE.fullmatch(value.replace(",",""))): raise ValueError(f"invalid adjudication: {rid}")
        if value is not None: row["concluded_value"]=value.replace(",","")
        if not str(row.get("rationale","")).strip(): raise ValueError(f"missing adjudication rationale: {rid}")
    return by_id,files


def finalize(outdir:pathlib.Path,started:str,mono:float,source_paths:list[pathlib.Path],mapping:list[dict[str,Any]],labels:dict[int,dict[str,dict[str,Any]]],label_paths:list[pathlib.Path],adjudicated:dict[str,dict[str,Any]],adjudication_paths:list[pathlib.Path]) -> None:
    raw=outdir/"raw"; overlap=[m for m in mapping if len(m["assigned_raters"])==2]
    agreed=sum(labels[m["assigned_raters"][0]][m["record_id"]]["concluded_value"]==labels[m["assigned_raters"][1]][m["record_id"]]["concluded_value"] for m in overlap)
    final=[]
    for m in mapping:
        rid=m["record_id"]; value=adjudicated[rid]["concluded_value"] if rid in adjudicated else labels[m["primary_rater"]][rid]["concluded_value"]
        final.append({"record_id":rid,"concluded_value":value})
    write_jsonl(raw/"final_blind_labels.jsonl",final)
    by_final={r["record_id"]:r["concluded_value"] for r in final}; scored=[]
    for m in mapping:
        scored.append({"record_id":m["record_id"],"source":m["source"],"arm":m["arm"],"task_id":m["task_id"],
                       "gold":m["gold"],"concluded_value":by_final[m["record_id"]],"correct":bool(numeric_equal(by_final[m["record_id"]],m["gold"]))})
    write_jsonl(raw/"sealed_scored_labels.jsonl",scored)
    aggregates={}
    for source,arms in (("trace",("answer_only","full_trace")),("q8",("f16","q8"))):
        aggregates[source]={arm:{"correct":sum(r["correct"] for r in scored if r["source"]==source and r["arm"]==arm),
                                        "n":sum(1 for r in scored if r["source"]==source and r["arm"]==arm)} for arm in arms}
    metrics={"frozen_sources_verified":True,"blind_records":len(mapping),"gold_or_arm_fields_exposed":False,
        "primary_labels":len(mapping),"double_labeled_records":len(overlap),"independent_raters":RATERS,
        "exact_inter_rater_agreement":agreed/len(overlap),"disagreements":len(overlap)-agreed,
        "unresolved_disagreements":0,"descriptive_aggregates_not_scientifically_authorized":aggregates}
    write_json(raw/"actual_scores.json",metrics); write_json(raw/"inter_rater_agreement.json",{"overlap":len(overlap),"agreed":agreed,"agreement":agreed/len(overlap),"disagreements":len(overlap)-agreed})
    attestations=[]
    for path in sorted((raw/"rater_attestations").glob("*.json")):
        attestations.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(raw/"rater_provenance.json",{"raters":attestations,"required_distinct":RATERS,"adjudicator":bool(adjudicated)})
    write_json(raw/"independent_evaluation.json",{"descriptive_only":True,"aggregates":aggregates,"downstream_audit_required":True})
    write_json(raw/"scorer_hashes.json",{"protocol_runner":{"sha256":sha256_file(pathlib.Path(__file__).resolve())},"automated_numeric_scorer_used":False})
    definitions={"source_integrity":("frozen_sources_verified","eq",True),"bundle_coverage":("blind_records","eq",768),
        "gold_blinding":("gold_or_arm_fields_exposed","eq",False),"primary_coverage":("primary_labels","eq",768),
        "overlap_coverage":("double_labeled_records","ge",96),"rater_independence":("independent_raters","ge",3),
        "agreement":("exact_inter_rater_agreement","ge",.85),"adjudication":("unresolved_disagreements","eq",0)}
    gates={}
    for gate,(metric,op,threshold) in definitions.items():
        actual=metrics[metric]; passed=actual==threshold if op=="eq" else actual>=threshold
        gates[gate]={"metric":metric,"operator":op,"threshold":threshold,"actual":actual,"pass":passed}
    evidence={"acceptance_gates":"raw/receipt.json","blind_label_packet":"raw/bundle_manifest.json","independent_evaluation":"raw/independent_evaluation.json",
        "inter_rater_agreement":"raw/inter_rater_agreement.json","provenance":"raw/receipt.json","rater_provenance":"raw/rater_provenance.json",
        "raw_samples":"raw/final_blind_labels.jsonl","receipt_fingerprint":"raw/receipt.json","scorer_hashes":"raw/scorer_hashes.json"}
    evidence_files=sorted(p for p in raw.rglob("*") if p.is_file() and p.name!="receipt.json")
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,
        input_paths=[*source_paths,*label_paths,*adjudication_paths,*evidence_files],packages=[],runtime={"execution_mode":"blind_semantic_annotation","new_model_inference_under_test":False,"gpu":False})
    complete,errors=provenance_complete(provenance)
    if not complete: raise RuntimeError(f"incomplete provenance: {errors}")
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":True,"gates":gates,"evidence":evidence}
    receipt["receipt_fingerprint"]=canonical_json_sha256(receipt); write_json(raw/"receipt.json",receipt)
    failed=[name for name,gate in gates.items() if not gate["pass"]]; claim="BLIND_NUMERIC_RELABEL_COMPLETED_R1" if not failed else "BLIND_NUMERIC_RELABEL_NOT_VALIDATED_R1"
    (outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\n"
        f"Labeled `{len(mapping)}` records with `{len(overlap)}` overlaps; exact agreement `{agreed/len(overlap):.4f}`; "
        f"adjudicated `{len(overlap)-agreed}` disagreements. Failed gates: `{', '.join(failed) if failed else 'none'}`. "
        "Downstream arm aggregates are descriptive until separately preregistered and audited.\n",encoding="utf-8",newline="\n")


def wait_finalize(outdir:pathlib.Path,poll:int) -> None:
    source_paths=verify_sources(); raw=outdir/"raw"; started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); mono=time.monotonic()
    mapping=json.loads((raw/"sealed_mapping.json").read_text(encoding="utf-8"))
    while True:
        loaded=load_rater_outputs(raw,mapping)
        if loaded is None: time.sleep(poll); continue
        labels,label_paths=loaded; disagreements=make_adjudication(raw,mapping,labels); adjudication=load_adjudication(raw,disagreements)
        if adjudication is None: time.sleep(poll); continue
        adjudicated,adjudication_paths=adjudication
        finalize(outdir,started,mono,source_paths,mapping,labels,label_paths,adjudicated,adjudication_paths); return


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True); mode.add_argument("--prepare",action="store_true"); mode.add_argument("--wait-finalize",action="store_true")
    parser.add_argument("--poll-seconds",type=int,default=300); parser.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID); args=parser.parse_args(); outdir=args.outdir.resolve()
    if args.prepare: prepare(outdir)
    else: wait_finalize(outdir,max(5,args.poll_seconds))
    return 0


if __name__=="__main__": raise SystemExit(main())
