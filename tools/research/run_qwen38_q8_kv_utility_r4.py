#!/usr/bin/env python3
"""Final Q8 KV utility aggregation from promoted blind labels."""
from __future__ import annotations
import argparse,json,pathlib,random,statistics,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-QWEN38-Q8-KV-UTILITY-04";SEED=2026082815;REPS=20_000
INPUTS={
"config/research_backlog_admissions/BACKLOG-QWEN38-Q8-KV-UTILITY-04.json":"e69902af0f70e4723f7903657fb3c56e82b04c53b0f45091e5ea702f0e611a99",
"runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-04/PRE_REGISTRATION.md":"a7568b244994efcfe6d0b083a3e7e92eb9281046886acca4779edcb0e5efb5df",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/receipt.json":"68015109f79a92e2054e21b1bad6a9b33fdf55f0de067b0e776cca59d1c103f7",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl":"2766c833a24533713014d2835fd4bb5df8427872bb94844566e302f34e8f1505",
"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/REVIEW.json":"dc2d0be45586d247133f876383866802dcb1698237f17c39fb50b2c1805c73d1",
"runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/raw/receipt.json":"55bc5facdbb4936c580c357c39c2bc6133362396d8bcca4b5ea0f09bad1d6524",
"runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/raw/actual_scores.json":"3aaeaf8deaf04619d1590274cf14093bbd3f82fa368ca6c28390dbd73f92d291",
"runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/REVIEW.json":"8755385e44b95473d99ecbeb5031f91236b16d13481c07bbdd84b40f286783c0"}
def wj(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
def rjl(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def boot(rows):
 arms={a:{r["task_id"]:int(r["correct"]) for r in rows if r["arm"]==a} for a in ("f16","q8")}
 if len(arms["f16"])!=128 or set(arms["f16"])!=set(arms["q8"]):raise ValueError("pair coverage")
 d=[arms["q8"][t]-arms["f16"][t] for t in sorted(arms["f16"])];rng=random.Random(SEED);b=sorted(sum(d[rng.randrange(128)] for _ in range(128))/128 for _ in range(REPS));return {"point":statistics.mean(d),"lower_95":b[int(.025*REPS)],"upper_95":b[int(.975*REPS)],"replicates":REPS,"seed":SEED,"q8_only_correct":sum(x==1 for x in d),"f16_only_correct":sum(x==-1 for x in d)}
def run(out):
 raw=out/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();paths=[]
 for rel,exp in INPUTS.items():
  p=ROOT/rel;act=sha256_file(p)
  if act!=exp:raise ValueError(f"source mismatch {rel}")
  paths.append(p)
 label_review=json.loads((ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/REVIEW.json").read_text(encoding="utf-8"))
 if label_review.get("verdict")!="APPROVED" or "BLIND_NUMERIC_RELABEL_COMPLETED_R3" not in label_review.get("claim_codes_permitted",[]):raise ValueError("labelset not promoted")
 rows=[r for r in rjl(ROOT/"runs/research/BACKLOG-BLIND-NUMERIC-RELABEL-03/raw/sealed_scored_labels.jsonl") if r["source"]=="q8"]
 if len(rows)!=256:raise ValueError("q8 rows")
 cmp=boot(rows);f=sum(r["correct"] for r in rows if r["arm"]=="f16");q=sum(r["correct"] for r in rows if r["arm"]=="q8");old=json.loads((ROOT/"runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/raw/actual_scores.json").read_text(encoding="utf-8"))
 m={"promoted_blind_labels_and_physical_sources_verified":True,"q8_labeled_rows":256,"f16_correct":f,"q8_correct":q,"f16_accuracy":f/128,"q8_accuracy":q/128,"f16_minus_q8_accuracy":(f-q)/128,"paired_q8_minus_f16_accuracy":cmp,"paired_bootstrap_95ci_lower_q8_minus_f16_accuracy":cmp["lower_95"],"vram_saving_mib":old["vram_saving_mib"],"q8_vs_f16_throughput_ratio":old["q8_vs_f16_throughput_ratio"],"service_and_embedding_restored":old["service_and_embedding_restored"]}
 defs={"source_integrity":("promoted_blind_labels_and_physical_sources_verified","eq",True),"request_coverage":("q8_labeled_rows","eq",256),"utility_noninferiority":("paired_bootstrap_95ci_lower_q8_minus_f16_accuracy","gt",-.05),"quality_regression":("f16_minus_q8_accuracy","le",.03),"physical_memory_saving":("vram_saving_mib","ge",500),"throughput_non_regression":("q8_vs_f16_throughput_ratio","ge",.95),"service_recovery":("service_and_embedding_restored","eq",True)};g={}
 for n,(k,o,x) in defs.items():v=m[k];p=v==x if o=="eq" else v>x if o=="gt" else v>=x if o=="ge" else v<=x;g[n]={"metric":k,"operator":o,"threshold":x,"actual":v,"pass":p}
 with (raw/"scored_labels.jsonl").open("w",encoding="utf-8",newline="\n") as stream:
  for r in rows:stream.write(json.dumps(r,separators=(",",":"))+"\n")
 wj(raw/"actual_scores.json",m);wj(raw/"independent_evaluation.json",{"comparison":cmp,"f16_correct":f,"q8_correct":q});wj(raw/"paired_baseline.json",{"baseline":"f16","treatment":"q8","paired_tasks":128,"comparison":cmp})
 for name in ("effective_route","hardware_metrics","recovery_state","service_identity"):
  wj(raw/f"{name}.json",{"imported_from":"BACKLOG-QWEN38-Q8-KV-UTILITY-03","source_receipt_sha256":INPUTS["runs/research/BACKLOG-QWEN38-Q8-KV-UTILITY-03/raw/receipt.json"],"physical_metrics":{k:m[k] for k in ("vram_saving_mib","q8_vs_f16_throughput_ratio","service_and_embedding_restored")}})
 ev={"acceptance_gates":"raw/receipt.json","effective_route":"raw/effective_route.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/scored_labels.jsonl","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","service_identity":"raw/service_identity.json"};e=sorted(p for p in raw.rglob("*") if p.is_file());prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*paths,*e],packages=[],runtime={"execution_mode":"promoted_blind_label_and_physical_metric_aggregation","new_inference":False,"service_mutation":False});ok,err=provenance_complete(prov)
 if not ok:raise RuntimeError(err)
 rec={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":g,"evidence":ev};rec["receipt_fingerprint"]=canonical_json_sha256(rec);wj(raw/"receipt.json",rec);failed=[n for n,v in g.items() if not v["pass"]];claim="QWEN38_Q8_KV_UTILITY_NONINFERIOR_R4" if not failed else "QWEN38_Q8_KV_UTILITY_NOT_NONINFERIOR_R4";(out/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nF16/Q8 `{f}/128` and `{q}/128`; delta `{cmp['point']:.6f}`, 95% CI `[{cmp['lower_95']:.6f}, {cmp['upper_95']:.6f}]`; physical `{m['vram_saving_mib']:.1f}` MiB and `{m['q8_vs_f16_throughput_ratio']:.4f}x`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",encoding="utf-8",newline="\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();run(a.outdir.resolve());return 0
if __name__=="__main__":raise SystemExit(main())
