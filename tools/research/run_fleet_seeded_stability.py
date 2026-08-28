#!/usr/bin/env python3
"""Fixed-seed temperature-0.2 stability matrix for qualified text routes."""
from __future__ import annotations
import argparse, json, os, pathlib, statistics, subprocess, sys, time
from typing import Any

ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
from tools.research import run_fleet_regression_screen as fleet

TASK_ID="BACKLOG-FLEET-SEEDED-STABILITY-01"; MODELS=("qwen38","hauhaucs","fable-tc","qwen36-moe")
PRE="026284a3288aa90700b46b409a2f945bbf92973f3e22d52991c78324c12cd5e3"
SOURCES={"config/qualified_model_fleet.json":"042fedf5907f031fb9993c03058f3cc9c8fe2c8d75a3235ea4b5e11c7412cd82","workloads/gsm8k.jsonl":"68a72276898a45dacb893154477621ff3d05ae7e043a10820644b691d8b63d77","runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl":"56434ebd53ff9f5adb477bd902566e28644fe9ea88619834f81417a06c66b84f","tools/research/run_fleet_regression_screen.py":"7cbf942375c120970ac78fb40f7f15050ebddb7ec5372cb40bd721009faf1de3"}

def wj(p,v):
 t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");t.replace(p)
def aj(p,v):
 with p.open("a",encoding="utf-8",newline="\n") as f:f.write(json.dumps(v,ensure_ascii=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())
def rj(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []
def verify():
 led={};paths=[]
 for rel,exp in SOURCES.items():
  p=ROOT/rel;a=sha256_file(p)
  if a!=exp:raise ValueError(f"source mismatch {rel}: {a}")
  led[rel]={"bytes":p.stat().st_size,"sha256":a};paths.append(p)
 p=ROOT/"runs/research"/TASK_ID/"PRE_REGISTRATION.md";a=sha256_file(p)
 if a!=PRE:raise ValueError(f"prereg mismatch: {a}")
 led[str(p.relative_to(ROOT)).replace("\\","/")]={"bytes":p.stat().st_size,"sha256":a};paths.append(p)
 return led,paths
def cases():
 math=fleet.read_jsonl(ROOT/"workloads/gsm8k.jsonl")[:16];qa=fleet.read_jsonl(ROOT/"runs/requalification/QWEN38-HAUHAUCS-NORMAL-QA-2026-08-23/tasks.jsonl")[:8]
 return [("math",str(x["task_id"]),x) for x in math]+[("qa",str(x["id"]),x) for x in qa]
def execute(out):
 raw=out/"raw";raw.mkdir(parents=True,exist_ok=True);sp=raw/"samples.jsonl";stp=raw/"runner_state.json";start=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();led,paths=verify();wj(raw/"frozen_inputs.json",led)
 rows=rj(sp);done={(x["model"],x["repeat"],x["suite"],x["case_id"]) for x in rows};panel=cases();svc=fleet.service_state();gw=fleet.gateway_status();initial=gw["current_model"]
 if svc["active_state"]!="active" or fleet.embedding_health()!=200:raise RuntimeError("service boundary unhealthy")
 state={"task_id":TASK_ID,"started_at_utc":start,"status":"running","initial_service":svc,"initial_gateway":gw,"initial_model":initial,"recorded_requests":len(rows)};wj(stp,state);routes=[];err=None
 try:
  for model in MODELS:
   status,gpu=fleet.switch_model(model);routes.append({"model":model,"status":status,"gpu":gpu,"embedding":fleet.embedding_health()});con=0
   for rep in range(3):
    for suite,cid,case in panel:
     key=(model,rep,suite,cid)
     if key in done:continue
     req=fleet.payload_for(model,suite,case);req["temperature"]=0.2;req["top_p"]=0.95;req["seed"]=20260826
     t=time.perf_counter();code,response=fleet.http_json(f"{fleet.BASE_URL}/v1/chat/completions",req);wall=round((time.perf_counter()-t)*1000,3);proj=fleet.semantic_projection(response);score=fleet.score_response(suite,case,response) if code==200 else {"pass":False}
     row={"model":model,"repeat":rep,"suite":suite,"case_id":cid,"http_status":code,"error":response.get("_error"),"wall_ms":wall,"score":score,"response":response,"semantic_projection":proj,"semantic_sha256":canonical_json_sha256(proj)};aj(sp,row);rows.append(row);done.add(key);con=con+1 if code!=200 else 0;state.update({"recorded_requests":len(rows),"last":list(key)});wj(stp,state);print(f"{len(rows):03d}/288 {model} r{rep} {suite}:{cid} http={code}",flush=True)
     if con>=3:raise RuntimeError(f"three consecutive errors on {model}")
 except Exception as x:err=f"{type(x).__name__}: {x}";state.update({"status":"aborted","error":err});wj(stp,state);raise
 finally:
  try:s,g=fleet.switch_model(initial);rest={"status":s,"gpu":g,"embedding":fleet.embedding_health()}
  except Exception as x:rest={"error":f"{type(x).__name__}: {x}"}
  state["restoration"]=rest;wj(stp,state)
 rows=rj(sp);base={(x["model"],x["suite"],x["case_id"]):x["semantic_sha256"] for x in rows if x["repeat"]==0};comp=[x["semantic_sha256"]==base[(x["model"],x["suite"],x["case_id"])] for x in rows if x["repeat"]>0];final=fleet.service_state();metrics={"routes_completed":len({x["model"] for x in rows if sum(y["model"]==x["model"] for y in rows)==72}),"recorded_requests":len(rows),"successful_response_rate":sum(x["http_status"]==200 for x in rows)/len(rows),"exact_seeded_repeat_rate":sum(comp)/len(comp),"service_restarts":final["n_restarts"]-svc["n_restarts"],"initial_model_restored":rest.get("status",{}).get("current_model")==initial and rest.get("embedding")==200};wj(raw/"actual_scores.json",metrics);wj(raw/"effective_route.json",{"routes":routes});wj(raw/"hardware_metrics.json",{"p50_ms":statistics.median(x["wall_ms"] for x in rows)});wj(raw/"independent_evaluation.json",{"metrics":metrics,"independent_review_pending":True});wj(raw/"paired_baseline.json",{"baseline_repeat":0,"comparison_repeats":[1,2],"pairs":len(comp)});wj(raw/"recovery_state.json",rest);wj(raw/"service_identity.json",{"initial":svc,"final":final});wj(raw/"service_maintenance.json",{"service_stopped":False,"restored":metrics["initial_model_restored"]});wj(raw/"treatment_controls.json",{"temperature":.2,"top_p":.95,"seed":20260826,"models":MODELS,"repeats":3})
 defs={"route_coverage":("routes_completed","eq",4),"request_coverage":("recorded_requests","eq",288),"request_integrity":("successful_response_rate","eq",1.0),"seeded_stability":("exact_seeded_repeat_rate","ge",.9),"service_integrity":("service_restarts","eq",0),"service_recovery":("initial_model_restored","eq",True)};gates={}
 for gid,(m,op,th) in defs.items():a=metrics[m];gates[gid]={"metric":m,"operator":op,"threshold":th,"actual":a,"pass":a==th if op=="eq" else a>=th}
 evfiles=sorted(p for p in raw.iterdir() if p.is_file());prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=start,started_monotonic=mono,input_paths=[*paths,*evfiles],packages=[],runtime={"execution_mode":"fixed_seed_sampling_stability","requests":len(rows)});ok,errors=provenance_complete(prov)
 if not ok:raise RuntimeError(errors)
 evidence={"acceptance_gates":"raw/receipt.json","effective_route":"raw/effective_route.json","service_identity":"raw/service_identity.json","paired_baseline":"raw/paired_baseline.json","recovery_state":"raw/recovery_state.json","hardware_metrics":"raw/hardware_metrics.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","receipt_fingerprint":"raw/receipt.json","independent_evaluation":"raw/independent_evaluation.json","treatment_controls":"raw/treatment_controls.json","service_maintenance":"raw/service_maintenance.json"};receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);wj(raw/"receipt.json",receipt);passed=all(x["pass"] for x in gates.values());claim="QUALIFIED_TEXT_FLEET_SEEDED_STABLE_R1" if passed else "QUALIFIED_TEXT_FLEET_SEEDED_UNSTABLE_R1";(out/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nRecorded `{len(rows)}` requests with exact seeded repeat rate `{metrics['exact_seeded_repeat_rate']:.6f}`.\n",encoding="utf-8",newline="\n");state.update({"status":"completed","claim":claim});wj(stp,state);return receipt
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);p.add_argument("--selfcheck",action="store_true");a=p.parse_args()
 if a.selfcheck:assert len(cases())==24;print("fleet seeded stability self-check OK");return 0
 r=execute(a.outdir.resolve());print(json.dumps(r["gates"],indent=2),flush=True);x=subprocess.run([sys.executable,str(ROOT/"tools/analysis/backlog_pipeline.py"),"advance",TASK_ID,"--to","EXECUTED","--actor","Codex executor"],cwd=ROOT,capture_output=True,text=True);print(x.stdout,flush=True);return 0 if x.returncode==0 else 2
if __name__=="__main__":raise SystemExit(main())
