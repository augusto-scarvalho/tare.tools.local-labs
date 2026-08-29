#!/usr/bin/env python3
"""Final trace aggregation from the promoted blind semantic labels."""
from __future__ import annotations
import argparse,json,pathlib,random,statistics,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-04";SEED=2026082814;REPS=20_000
INPUTS={
"config/research_backlog_admissions/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-04.json":"9b164d2be9c7e21e2628fd9db071ae785fb2f4cffa5485b6f226628220a3271a",
"runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-RESCORE-04/PRE_REGISTRATION.md":"690ab6db76e8f35c07a0bb42a99a6e254485d9f0639af4e27c3c9966f58c007d",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/receipt.json":"68015109f79a92e2054e21b1bad6a9b33fdf55f0de067b0e776cca59d1c103f7",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl":"2766c833a24533713014d2835fd4bb5df8427872bb94844566e302f34e8f1505",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/REVIEW.json":"dc2d0be45586d247133f876383866802dcb1698237f17c39fb50b2c1805c73d1",
"runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json":"b4fc924a1542e4913c3c1d70fdf77f8bb9be0e2662b8757d0d06f82b60d3f521",
"runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json":"c2ba817d9919d7c58e4d9ca33f6ce3105c9f25e3ccf105e96d94631944f3a18a"}
def wj(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
def rjl(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def boot(rows):
 arms={a:{r["task_id"]:int(r["correct"]) for r in rows if r["arm"]==a} for a in ("answer_only","full_trace")}
 if len(arms["answer_only"])!=256 or set(arms["answer_only"])!=set(arms["full_trace"]):raise ValueError("pair coverage")
 d=[arms["full_trace"][t]-arms["answer_only"][t] for t in sorted(arms["answer_only"])];rng=random.Random(SEED);b=sorted(sum(d[rng.randrange(256)] for _ in range(256))/256 for _ in range(REPS));return {"point":statistics.mean(d),"lower_95":b[int(.025*REPS)],"upper_95":b[int(.975*REPS)],"replicates":REPS,"seed":SEED,"trace_only_correct":sum(x==1 for x in d),"answer_only_correct":sum(x==-1 for x in d)}
def run(out):
 raw=out/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();paths=[]
 for rel,exp in INPUTS.items():
  p=ROOT/rel;act=sha256_file(p)
  if act!=exp:raise ValueError(f"source mismatch {rel}")
  paths.append(p)
 review=json.loads((ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/REVIEW.json").read_text(encoding="utf-8"))
 if review.get("verdict")!="APPROVED" or "BLIND_NUMERIC_RELABEL_COMPLETED_R3" not in review.get("claim_codes_permitted",[]):raise ValueError("labelset is not independently promoted")
 rows=[r for r in rjl(ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl") if r["source"]=="trace"]
 if len(rows)!=512:raise ValueError("trace rows")
 cmp=boot(rows);a=sum(r["correct"] for r in rows if r["arm"]=="answer_only");t=sum(r["correct"] for r in rows if r["arm"]=="full_trace");old=json.loads((ROOT/"runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/actual_scores.json").read_text(encoding="utf-8"))
 m={"promoted_blind_labels_verified":True,"trace_labeled_rows":512,"answer_correct":a,"trace_correct":t,"answer_accuracy":a/256,"trace_accuracy":t/256,"trace_minus_answer":cmp["point"],"paired_bootstrap":cmp,"paired_bootstrap_95ci_lower_trace_minus_answer":cmp["lower_95"],"imported_selected_seed_qa_regression":old["imported_selected_seed_qa_regression"]}
 defs={"source_integrity":("promoted_blind_labels_verified","eq",True),"evaluation_coverage":("trace_labeled_rows","eq",512),"finalist_gain":("paired_bootstrap_95ci_lower_trace_minus_answer","gt",0.0),"finalist_absolute":("trace_accuracy","ge",.40),"protected_retention":("imported_selected_seed_qa_regression","le",.05)};g={}
 for n,(k,o,x) in defs.items():v=m[k];p=v==x if o=="eq" else v>x if o=="gt" else v>=x if o=="ge" else v<=x;g[n]={"metric":k,"operator":o,"threshold":x,"actual":v,"pass":p}
 with (raw/"scored_labels.jsonl").open("w",encoding="utf-8",newline="\n") as f:
  for r in rows:f.write(json.dumps(r,separators=(",",":"))+"\n")
 wj(raw/"actual_scores.json",m);wj(raw/"independent_evaluation.json",{"comparison":cmp,"answer_correct":a,"trace_correct":t});wj(raw/"paired_baseline.json",{"baseline":"answer_only","treatment":"full_trace","paired_tasks":256,"comparison":cmp});wj(raw/"dataset_hashes.json",{"promoted_labels":{"sha256":INPUTS["runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl"]}});wj(raw/"model_hash.json",{"rescore_only":True,"source_receipt_sha256":INPUTS["runs/research/BACKLOG-ADAPT-TRACE-DEPLOY-FINALIST-01/raw/receipt.json"]});wj(raw/"student_samples.json",{"rows":512,"source":"raw/scored_labels.jsonl"});wj(raw/"teacher_samples.json",{"rows":0,"aggregation_only":True})
 ev={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","dataset_hashes":"raw/dataset_hashes.json","independent_evaluation":"raw/independent_evaluation.json","model_hash":"raw/model_hash.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/scored_labels.jsonl","receipt_fingerprint":"raw/receipt.json","student_samples":"raw/student_samples.json","teacher_samples":"raw/teacher_samples.json"};e=sorted(p for p in raw.rglob("*") if p.is_file());prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*paths,*e],packages=[],runtime={"execution_mode":"promoted_blind_label_aggregation","new_inference":False});ok,err=provenance_complete(prov)
 if not ok:raise RuntimeError(err)
 rec={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":g,"evidence":ev};rec["receipt_fingerprint"]=canonical_json_sha256(rec);wj(raw/"receipt.json",rec);failed=[n for n,v in g.items() if not v["pass"]];claim="TRACE_DISTILLATION_DEPLOYMENT_FINALIST_CONFIRMED_R4" if not failed else "TRACE_DISTILLATION_DEPLOYMENT_FINALIST_NOT_CONFIRMED_R4";(out/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nAnswer/trace `{a}/256` and `{t}/256`; delta `{cmp['point']:.6f}`, 95% CI `[{cmp['lower_95']:.6f}, {cmp['upper_95']:.6f}]`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",encoding="utf-8",newline="\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();run(a.outdir.resolve());return 0
if __name__=="__main__":raise SystemExit(main())
