#!/usr/bin/env python3
"""Paired physical K0/K2/K4 telemetry audit of BEE-L3."""
from __future__ import annotations
import argparse,json,pathlib,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.adaptive_mtp_controller import AdaptiveMTPController
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-BEE-L3-REAL-TELEMETRY-01";BASE="http://127.0.0.1:8080"
EXPECTED={ROOT/"config/research_backlog_admissions/BACKLOG-BEE-L3-REAL-TELEMETRY-01.json":"cf923541cb598cb28cbfa3acd0204bc4e2944825e930f13a66e60909462fb3a3",ROOT/"runs/research/BACKLOG-BEE-L3-REAL-TELEMETRY-01/PRE_REGISTRATION.md":"918ba8165584b64a0e89f1ff328615cc6deeeb91122c5a9b2c897103c2356a3b",ROOT/"runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/PRE_REGISTRATION.md":"cfb661dca692d6d857fa9e854e6f00a436e42675929ebfb740dd384015e067a4",ROOT/"runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/RESULT.md":"a26f33d2d93f097ed425be19723d6096ad7faba16f90080aa4dad938408d9720",ROOT/"runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/raw/receipt.json":"d13d51e10ceda38fe611cf4c4b6efaa12ac1fa22461b0aaa2a6fe6f1e61fee6d",ROOT/"tools/analysis/adaptive_mtp_controller.py":"7ede4879ddf8d94a5efcbd4b4b2ca2ab0ed70353004822d83fbe442b2a986e9e",ROOT/"tests/test_adaptive_mtp_controller.py":"c5074759e4cbb4c1379468a15ca4d051b7b2ac80dffc38f981f55d0db0a3f12a",ROOT/"runs/research/BACKLOG-ADAPT-TRAIN-01/raw/samples.jsonl":"243311c37ff240d97f63539c4e85f3a9ec7272ea8eaa1279d31c7c38d44d50c4"}
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def run(argv,timeout=600):
 d=subprocess.run(argv,capture_output=True,text=True,encoding="utf-8",errors="replace",check=False,timeout=timeout);return {"argv":argv,"returncode":d.returncode,"stdout":d.stdout.strip(),"stderr":d.stderr.strip()}
def health(port):
 try:
  with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=5) as r:return r.status
 except Exception:return None
def wait_health(want=200,timeout=240):
 end=time.time()+timeout
 while time.time()<end:
  value=health(8080)
  if value==want:return
  time.sleep(.5)
 raise RuntimeError(f"8080 did not reach health {want}")
def wait_down(timeout=30):
 end=time.time()+timeout
 while time.time()<end:
  if health(8080) is None:return
  time.sleep(.25)
 raise RuntimeError("8080 remained reachable")
def service():
 x=run(["wsl","-d","Ubuntu-24.04","--","systemctl","show","llm-inference.service","-p","MainPID","-p","NRestarts","-p","ActiveState","-p","ExecStart","--no-pager"]);return {"raw":x,"values":dict(line.split("=",1) for line in x["stdout"].splitlines() if "=" in line)}
def stable_exec(v):return v.split(" ; ignore_errors=",1)[0].strip()
def post(payload):
 req=urllib.request.Request(BASE+"/completion",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=180) as r:return json.loads(r.read().decode())
def collect(arm,prompts):
 rows=[]
 for i,item in enumerate(prompts):
  payload={"prompt":item["prompt"],"n_predict":64,"temperature":0.0,"top_k":1,"seed":0,"cache_prompt":False,"stream":False,"id_slot":i%4};start=time.perf_counter();response=post(payload);wall=(time.perf_counter()-start)*1000;timings=response.get("timings") or {};rows.append({"arm":arm,"index":i,"task_id":item["task_id"],"prompt":item["prompt"],"content":str(response.get("content") or ""),"wall_ms":wall,"predicted_n":int(timings.get("predicted_n") or response.get("tokens_predicted") or 0),"predicted_ms":float(timings.get("predicted_ms") or 0),"draft_n":int(timings.get("draft_n") or 0),"draft_n_accepted":int(timings.get("draft_n_accepted") or 0),"response":response})
 return rows
def execute(outdir):
 raw=outdir/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();ledger={}
 if any(raw.iterdir()):raise RuntimeError("raw not empty")
 for p,e in EXPECTED.items():
  a=sha256_file(p)
  if a!=e:raise ValueError(f"hash mismatch {p}: {a}")
  ledger[p.relative_to(ROOT).as_posix()]={"bytes":p.stat().st_size,"sha256":a}
 seen=set();prompts=[]
 for line in (ROOT/"runs/research/BACKLOG-ADAPT-TRAIN-01/raw/samples.jsonl").read_text(encoding="utf-8").splitlines():
  row=json.loads(line)
  if row.get("arm")=="base" and row["task_id"] not in seen:seen.add(row["task_id"]);prompts.append({"task_id":row["task_id"],"prompt":row["prompt"]})
  if len(prompts)==48:break
 if len(prompts)!=48:raise RuntimeError("did not resolve 48 frozen prompts")
 original=service();original_exec=original["values"].get("ExecStart","");binary=original_exec.split("path=",1)[1].split(" ;",1)[0];model=original_exec.split("-m ",1)[1].split(" ",1)[0];binary_hash=run(["wsl","-d","Ubuntu-24.04","--","sha256sum",binary])["stdout"].split()[0];model_hash=run(["wsl","-d","Ubuntu-24.04","--","sha256sum",model])["stdout"].split()[0]
 if "--spec-type draft-mtp" not in original_exec or "--spec-draft-n-max 4" not in original_exec:raise RuntimeError("original route is not K4 MTP")
 arms={"k4":collect("k4",prompts)};temp=None;handle=None;launches=[]
 try:
  stopped=run(["wsl","-d","Ubuntu-24.04","-u","root","--","systemctl","stop","llm-inference.service"],timeout=60)
  if stopped["returncode"]:raise RuntimeError(stopped)
  wait_down()
  if health(8081)!=200:raise RuntimeError("embedding down")
  common=["wsl","-d","Ubuntu-24.04","--",binary,"-m",model,"--alias","fable-tc-l1.0","--host","0.0.0.0","--port","8080","-ngl","99","-fa","on","--ctx-size","8192","--parallel","4","--jinja","--metrics"]
  for arm,extra in (("k0",[]),("k2",["--spec-type","draft-mtp","--spec-draft-n-max","2"])):
   args=common+extra;handle=(raw/f"server_{arm}.log").open("w",encoding="utf-8");temp=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));wait_health();launches.append(args);arms[arm]=collect(arm,prompts);temp.terminate()
   try:temp.wait(timeout=20)
   except subprocess.TimeoutExpired:temp.kill();temp.wait(timeout=10)
   handle.close();temp=None;handle=None;wait_down()
 finally:
  if temp is not None:
   temp.terminate()
   try:temp.wait(timeout=20)
   except subprocess.TimeoutExpired:temp.kill()
  if handle is not None:handle.close()
  start_result=run(["wsl","-d","Ubuntu-24.04","-u","root","--","systemctl","start","llm-inference.service"],timeout=60)
  if start_result["returncode"]:raise RuntimeError(start_result)
  wait_health();restored=service()
 if stable_exec(restored["values"].get("ExecStart",""))!=stable_exec(original_exec):raise RuntimeError("original path/argv not restored")
 maps={arm:{r["task_id"]:r for r in rows} for arm,rows in arms.items()};parity=sum(len({maps[a][p["task_id"]]["content"] for a in maps})==1 for p in prompts)/48
 controller=AdaptiveMTPController(window_size=16,gamma=.15,max_k=4,probe_interval=8);adaptive=[]
 for p in prompts:
  k4=maps["k4"][p["task_id"]];recommended=controller.get_recommended_depth();selected="k0" if recommended<=1 else "k2" if recommended<=3 else "k4";adaptive.append({"task_id":p["task_id"],"recommended_k":recommended,"selected_arm":selected,"wall_ms":maps[selected][p["task_id"]]["wall_ms"],"predicted_n":maps[selected][p["task_id"]]["predicted_n"]});controller.record_step(k4["draft_n_accepted"],k4["draft_n"])
 def throughput(rows):return sum(r["predicted_n"] for r in rows)/sum(r["wall_ms"] for r in rows)
 tp={arm:throughput(rows) for arm,rows in arms.items()};atp=throughput(adaptive);rates=sorted(((r["draft_n_accepted"]/r["draft_n"] if r["draft_n"] else 0,r["task_id"]) for r in arms["k4"]),key=lambda x:(x[0],x[1]));low_ids={task for _,task in rates[:12]};low_ad=[r for r in adaptive if r["task_id"] in low_ids];low_k0=[r for r in arms["k0"] if r["task_id"] in low_ids];low_protection=throughput(low_ad)/throughput(low_k0)
 metrics={"live_requests":sum(len(v) for v in arms.values()),"paired_exact_parity_rate":parity,"k4_requests_with_drafts":sum(r["draft_n"]>0 for r in arms["k4"]),"adaptive_replay_speedup_over_k0":atp/tp["k0"],"adaptive_replay_gain_over_k4":atp/tp["k4"]-1,"low_acceptance_protection":low_protection,"k0_throughput":tp["k0"],"k2_throughput":tp["k2"],"k4_throughput":tp["k4"],"adaptive_throughput":atp,"mean_recommended_k":sum(r["recommended_k"] for r in adaptive)/48,"original_service_restored":int(restored["values"].get("ActiveState")=="active" and health(8080)==200),"embedding_health":health(8081)}
 with (raw/"samples.jsonl").open("w",encoding="utf-8") as f:
  for arm in ("k0","k2","k4"):
   for row in arms[arm]:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 write(raw/"actual_scores.json",metrics);write(raw/"artifact_hashes.json",ledger|{"binary":{"sha256":binary_hash},"model":{"sha256":model_hash}});write(raw/"dataset_hashes.json",{"prompt_panel_semantic_sha256":canonical_json_sha256(prompts)});write(raw/"effective_route.json",{"original_k4":original_exec,"temporary_launches":launches});write(raw/"failure_reproduction.json",{"historical":"synthetic Bernoulli acceptance and abstract cost","successor":"paired live latency and draft telemetry"});write(raw/"falsifiable_hypothesis.json",{"physical_arms":[0,2,4],"prompts":48,"all_gates_required":True});write(raw/"hardware_metrics.json",{"throughput":tp|{"adaptive":atp},"wall_ms":{a:[r["wall_ms"] for r in rows] for a,rows in arms.items()}});write(raw/"independent_evaluation.json",{"adaptive_replay":adaptive,"low_acceptance_task_ids":sorted(low_ids)});write(raw/"invalidation_rules.json",{"any_output_mismatch_aborts_claim":True,"restore_failure_aborts":True});write(raw/"invariant_controls.json",{"decode":{"n_predict":64,"temperature":0,"top_k":1,"seed":0,"cache_prompt":False},"controller":{"window_size":16,"gamma":.15,"max_k":4,"probe_interval":8},"mapping":{"0..1":"k0","2..3":"k2","4":"k4"}});write(raw/"paired_baseline.json",{"task_ids":[p["task_id"] for p in prompts],"arms":["k0","k2","k4"]});write(raw/"real_implementation.json",{"physical_static_arms":True,"adaptive_counterfactual_replay":True,"live_request_switching":False});write(raw/"recovery_state.json",{"stable_exec_match":True,"restored":restored});write(raw/"semantic_parity.json",{"paired_exact_parity_rate":parity});write(raw/"service_identity.json",{"original":original,"binary_sha256":binary_hash,"model_sha256":model_hash,"restored":restored});write(raw/"service_maintenance.json",{"root_handoff":True,"original_service_restored":metrics["original_service_restored"],"embedding_health":metrics["embedding_health"]});write(raw/"source_execution_receipt.json",{"historical_receipt_sha256":EXPECTED[ROOT/"runs/research/BEE-L3-MTP-CONTROLLER-2026-08-25/raw/receipt.json"]})
 defs={"arm_coverage":("live_requests","eq",144),"semantic_parity":("paired_exact_parity_rate","eq",1.0),"mtp_telemetry":("k4_requests_with_drafts","ge",40),"global_speedup":("adaptive_replay_speedup_over_k0","ge",1.25),"static_gain":("adaptive_replay_gain_over_k4","ge",.15),"low_protection":("low_acceptance_protection","ge",.95),"service_restore":("original_service_restored","eq",1),"embedding_integrity":("embedding_health","eq",200)};ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b};gates={g:{"metric":m,"operator":o,"threshold":t,"actual":metrics[m],"pass":ops[o](metrics[m],t)} for g,(m,o,t) in defs.items()};evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","effective_route":"raw/effective_route.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","semantic_parity":"raw/semantic_parity.json","service_identity":"raw/service_identity.json","service_maintenance":"raw/service_maintenance.json","source_execution_receipt":"raw/source_execution_receipt.json"};files=sorted({raw/v.removeprefix("raw/") for v in evidence.values() if v!="raw/receipt.json"});prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,*files],packages=["pytest"],runtime={"execution_mode":"paired_live_k0_k2_k4","binary":binary,"model":model});ok,errors=provenance_complete(prov)
 if not ok:raise ValueError(errors)
 receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write(raw/"receipt.json",receipt);return receipt,metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();r,m=execute(a.outdir.resolve());passed=all(x["pass"] for x in r["gates"].values());claim="BEE_L3_REAL_TELEMETRY_QUALIFIED_R1" if passed else "BEE_L3_FALSE_POSITIVE_CONFIRMED_R1";failed=[g for g,x in r["gates"].items() if not x["pass"]];(a.outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\n144 physical requests, exact parity {m['paired_exact_parity_rate']:.4%}. Adaptive replay speedup vs K0 {m['adaptive_replay_speedup_over_k0']:.3f}x, gain vs K4 {m['adaptive_replay_gain_over_k4']:.3%}, low-acceptance protection {m['low_acceptance_protection']:.3%}, mean K {m['mean_recommended_k']:.3f}. Failed gates: {', '.join(failed) if failed else 'none'}. Replay uses physical paired arms but is not live per-request K switching. Original service restored.\n",encoding="utf-8");print(json.dumps({"claim":claim,"metrics":m,"gates":r["gates"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
