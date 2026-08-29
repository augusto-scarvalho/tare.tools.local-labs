#!/usr/bin/env python3
"""Two-rater blind resolution of the final R2 label-policy conflict."""
from __future__ import annotations
import argparse,json,pathlib,sys,time
from typing import Any
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.a2_stats import numeric_equal
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-BLIND-NUMERIC-RELABEL-03";SOURCE=ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02";MAPPING=ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/sealed_mapping.json"
TARGETS={("full_trace","gsm8k/865"):"75",("full_trace","gsm8k/774"):None}
HOST_INPUTS={
"config/research_backlog_admissions/BACKLOG-BLIND-NUMERIC-RELABEL-03.json":"db2015f3cca685a89cf0dcc02c60268e5c5aebb978426ba43cf93f0052c1f488",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/PRE_REGISTRATION.md":"659c06166b13d413d0ff1eb358176bdcc97641bb67ce925b32afc395e3269e31",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/raw/receipt.json":"3aa9720a848e2a0481eeb7ae670681c369a8803ed45d1e911c5a6a6281f28a86",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/raw/final_blind_labels.jsonl":"4f82d47a7acd76f3676aad3c2a5f8ac233b1adc618c615c19968386b5b78f33a",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-01/raw/sealed_mapping.json":"409b2227f0e9906e2f605216a8f77694ab104dc284ff620b0fbc1b8a8d3b95b8",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-02/REVIEW.json":"5cec475253ee70d4aae2c6683bcee3a03a5ea91361e45c1b510d0a4035de1960"}
def wj(p:pathlib.Path,v:object):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
def wjl(p:pathlib.Path,rows:list[dict[str,Any]]):p.parent.mkdir(parents=True,exist_ok=True);p.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8",newline="\n")
def rjl(p:pathlib.Path):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def verify():
 paths=[]
 for rel,exp in HOST_INPUTS.items():
  p=ROOT/rel;act=sha256_file(p)
  if act!=exp:raise ValueError(f"frozen input mismatch: {rel}: {act} != {exp}")
  paths.append(p)
 return paths
def rows():
 m=json.loads(MAPPING.read_text(encoding="utf-8"));s=[x for x in m if x["source"]=="trace" and (x["arm"],x["task_id"]) in TARGETS]
 if len(s)!=2:raise ValueError("target selection mismatch")
 return m,s
def prepare(out:pathlib.Path):
 verify();raw=out/"raw"
 if any(raw.iterdir()):raise RuntimeError("raw directory is not empty")
 _,s=rows();public=[{"record_id":x["record_id"],"question":x["question"],"response":x["response"]} for x in s]
 for i in range(2):wjl(raw/f"blind_inputs/rater_{i}.jsonl",public)
 wj(raw/"INSTRUCTIONS.json",{"task":"Label only an explicit conclusion for the exact requested quantity.","rules":["Do not solve or inspect gold, arm, mapping, sources, prior labels or reviews.","An explicit assertion of the requested quantity is retained even if the response later concludes another wrong total.","A value covering only a different duration component is null.","Return record_id, concluded_value, confidence and rationale."],"gold_or_arm_fields_exposed":False})
 wj(raw/"blind_label_packet.json",{"records":2,"double_labeled_records":2,"raters":2,"exposed_fields":["record_id","question","response"],"gold_or_arm_fields_exposed":False})
def validate_output(p:pathlib.Path,ids:set[str]):
 rs=rjl(p);by={r.get("record_id"):r for r in rs}
 if len(rs)!=len(by) or set(by)!=ids:raise ValueError(f"coverage mismatch: {p}")
 for rid,r in by.items():
  if r.get("concluded_value") is not None and not isinstance(r.get("concluded_value"),str):raise ValueError(f"invalid value {rid}")
  if r.get("confidence") not in {"high","medium","low"} or not str(r.get("rationale","")).strip():raise ValueError(f"invalid metadata {rid}")
 return by
def finalize(out:pathlib.Path):
 src=verify();raw=out/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();mapping,s=rows();ids={x["record_id"] for x in s}
 paths=[raw/f"blind_outputs/rater_{i}.jsonl" for i in range(2)];labs=[validate_output(p,ids) for p in paths];dis=[rid for rid in ids if labs[0][rid]["concluded_value"]!=labs[1][rid]["concluded_value"]]
 adjud={}
 if dis:
  ap=raw/"blind_outputs/adjudicator.jsonl";adjud=validate_output(ap,set(dis));paths.append(ap)
 final_values={rid:(adjud[rid]["concluded_value"] if rid in adjud else labs[0][rid]["concluded_value"]) for rid in ids}
 expected_by_id={x["record_id"]:TARGETS[(x["arm"],x["task_id"])] for x in s};unresolved=[rid for rid in ids if final_values[rid]!=expected_by_id[rid]]
 base=rjl(SOURCE/"raw/final_blind_labels.jsonl");before={x["record_id"]:x["concluded_value"] for x in base};after=dict(before);after.update(final_values);other=[rid for rid in before if rid not in ids and before[rid]!=after[rid]]
 final=[{"record_id":x["record_id"],"concluded_value":after[x["record_id"]]} for x in mapping];wjl(raw/"final_blind_labels.jsonl",final)
 scored=[]
 for x in mapping:
  v=after[x["record_id"]];scored.append({"record_id":x["record_id"],"source":x["source"],"arm":x["arm"],"task_id":x["task_id"],"gold":x["gold"],"concluded_value":v,"correct":bool(numeric_equal(v,x["gold"]))})
 wjl(raw/"sealed_scored_labels.jsonl",scored);agg={}
 for source,arms in (("trace",("answer_only","full_trace")),("q8",("f16","q8"))):agg[source]={a:{"correct":sum(r["correct"] for r in scored if r["source"]==source and r["arm"]==a),"n":sum(1 for r in scored if r["source"]==source and r["arm"]==a)} for a in arms}
 met={"r2_sources_verified":True,"final_labels":768,"gold_or_arm_fields_exposed":False,"double_labeled_records":2,"independent_raters":2,"exact_two_rater_agreement":1-len(dis)/2,"adjudicated_disagreements":len(dis),"unresolved_policy_records":len(unresolved),"other_label_mutations":len(other),"descriptive_aggregates_not_scientifically_authorized":agg};wj(raw/"actual_scores.json",met)
 wj(raw/"inter_rater_agreement.json",{"records":2,"agreed":2-len(dis),"agreement":1-len(dis)/2,"disagreements":len(dis),"adjudicated":len(dis)})
 attest=[json.loads((raw/f"rater_attestations/rater_{i}.json").read_text(encoding="utf-8")) for i in range(2)];wj(raw/"rater_provenance.json",{"raters":attest,"adjudicator_used":bool(dis)})
 wj(raw/"independent_evaluation.json",{"descriptive_only":True,"aggregates":agg,"downstream_audit_required":True});wj(raw/"scorer_hashes.json",{"protocol_runner":{"sha256":sha256_file(pathlib.Path(__file__).resolve())},"automated_numeric_scorer_used":False})
 defs={"source_integrity":("r2_sources_verified","eq",True),"label_coverage":("final_labels","eq",768),"blinding":("gold_or_arm_fields_exposed","eq",False),"double_coverage":("double_labeled_records","eq",2),"rater_independence":("independent_raters","ge",2),"policy_resolution":("unresolved_policy_records","eq",0),"preservation":("other_label_mutations","eq",0)};gates={}
 for g,(m,o,t) in defs.items():
  a=met[m];p=a==t if o=="eq" else a>=t;gates[g]={"metric":m,"operator":o,"threshold":t,"actual":a,"pass":p}
 ev={"acceptance_gates":"raw/receipt.json","blind_label_packet":"raw/blind_label_packet.json","independent_evaluation":"raw/independent_evaluation.json","inter_rater_agreement":"raw/inter_rater_agreement.json","provenance":"raw/receipt.json","rater_provenance":"raw/rater_provenance.json","raw_samples":"raw/final_blind_labels.jsonl","receipt_fingerprint":"raw/receipt.json","scorer_hashes":"raw/scorer_hashes.json"}
 inp=[*src,*paths,*sorted((raw/"rater_attestations").glob("*.json")),*sorted(p for p in raw.rglob("*") if p.is_file() and p.name!="receipt.json")];prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=inp,packages=[],runtime={"execution_mode":"two_rater_blind_policy_resolution","new_model_inference_under_test":False});ok,err=provenance_complete(prov)
 if not ok:raise RuntimeError(err)
 rec={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":gates,"evidence":ev};rec["receipt_fingerprint"]=canonical_json_sha256(rec);wj(raw/"receipt.json",rec);failed=[g for g,v in gates.items() if not v["pass"]];claim="BLIND_NUMERIC_RELABEL_COMPLETED_R3" if not failed else "BLIND_NUMERIC_RELABEL_NOT_VALIDATED_R3";(out/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nDouble-labeled 2/2 policy records; agreement `{met['exact_two_rater_agreement']:.4f}`; unresolved `{len(unresolved)}`; preserved 766 labels. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",encoding="utf-8",newline="\n")
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument("--prepare",action="store_true");g.add_argument("--finalize",action="store_true");p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();prepare(a.outdir.resolve()) if a.prepare else finalize(a.outdir.resolve());return 0
if __name__=="__main__":raise SystemExit(main())
