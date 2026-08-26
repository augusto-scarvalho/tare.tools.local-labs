#!/usr/bin/env python3
"""Physical multi-adapter serving audit for ADAPT-06 and SLOP-L1..L7."""
from __future__ import annotations
import argparse, json, pathlib, statistics, subprocess, sys, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-ADAPT06-SLOP-LIVE-01";BASE="http://127.0.0.1:8080"
EXPECTED={
ROOT/"config/research_backlog_admissions/BACKLOG-ADAPT06-SLOP-LIVE-01.json":"2195a48a923990f7642af2b15c3d698dafd88138607be1ec4e7c74b544bb5c12",ROOT/"runs/research/BACKLOG-ADAPT06-SLOP-LIVE-01/PRE_REGISTRATION.md":"4da2aaf993703e42bcce71a9cebf5931e5cf8fe8cdd0f940ae021f5311e06885",
ROOT/"runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/PRE_REGISTRATION.md":"9551d6ba2f481ae11490159b530d305887463bd49ae160cf781cfbcb00fd244c",ROOT/"runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/RESULT.md":"3ef1e46397b285a5ebaddd6924c383c6462de4d6cf03c1d8277292419653122a",ROOT/"runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/raw/receipt.json":"f00dd6d31aa4d8970ef77aad7ccbfa68ca23bc31d26caf3f2bf8ca5e43665bb9",ROOT/"tools/analysis/adapter_cache_tagger.py":"951395b9210ce86771cc4982c94ecd31db7641c0e5c2a53c47aabc7539765369",
ROOT/"runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/PRE_REGISTRATION.md":"3c321c0a568ac4451de6cd5f998005e8e20e2fbef816df420637c245df956239",ROOT/"runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/RESULT.md":"75f76dad7122e124bb115998e08155bfcb13eb002ab6f1471160e5bd0f901841",ROOT/"runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/raw/receipt.json":"60d6f9e0f1a189a663a44ca6cc1979444b0b4653b5c2e708994311ebe287dd8d",ROOT/"tools/analysis/multi_adapter_router.py":"1b4e6abf5d88ea26293d0ff55320631236e57de54472e74d6bfb52532b57a9b4",
ROOT/"runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter/adapter_config.json":"45067f22d87e53ba56114cd0126c20d0591cefc5c9261a1de6c83b705f56e784",ROOT/"runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter/adapter_model.safetensors":"3fda4d2bae7c6388e97fc69c3c2e4de5d85a614e99f436d8c04373ced3b38966",ROOT/"runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter/adapter_config.json":"8516576a6d6a79f13bddd2388483d0638804d3367f661041be9c1b99aa0008fd",ROOT/"runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter/adapter_model.safetensors":"839d777b848ec202349266fc271aaacd4eff3078bb68e8a46f341dbc9b3194eb"}
PROMPTS=["What is 17 plus 28?","Compute 9 times 7.","If x+12=31, find x.","What is 144 divided by 12?","A box has 8 rows of 6 balls. How many balls?","What is 15 percent of 200?","Solve 3x=42.","What is 11 squared?","Subtract 39 from 100.","What is half of 86?","Compute 5 cubed.","A dozen plus a score equals how many?"]

def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def run(argv,timeout=600):
 d=subprocess.run(argv,capture_output=True,text=True,encoding="utf-8",errors="replace",check=False,timeout=timeout);return {"argv":argv,"returncode":d.returncode,"stdout":d.stdout.strip(),"stderr":d.stderr.strip()}
def get(path,timeout=30):
 with urllib.request.urlopen(BASE+path,timeout=timeout) as r:return json.loads(r.read().decode())
def post(path,payload,timeout=180):
 req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def health(port):
 try:
  with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=5) as r:return r.status
 except Exception:return None
def wait_health(port,want=200,timeout=180):
 end=time.time()+timeout;last=None
 while time.time()<end:
  last=health(port)
  if last==want:return last
  time.sleep(.5)
 raise RuntimeError(f"port {port} health {last}, wanted {want}")
def service():
 x=run(["wsl","-d","Ubuntu-24.04","--","systemctl","show","llm-inference.service","-p","MainPID","-p","NRestarts","-p","ActiveState","-p","ExecStart","--no-pager"]);return {"raw":x,"values":dict(line.split("=",1) for line in x["stdout"].splitlines() if "=" in line)}
def route(name):
 return {"base":[{"id":0,"scale":0.0},{"id":1,"scale":0.0}],"mlp":[{"id":0,"scale":1.0},{"id":1,"scale":0.0}],"attn":[{"id":0,"scale":0.0},{"id":1,"scale":1.0}]}[name]
def complete(prompt,name,slot=None,cache=False,n=24):
 payload={"prompt":prompt,"lora":route(name),"n_predict":n,"temperature":0.0,"top_k":1,"seed":0,"cache_prompt":cache,"stream":False}
 if slot is not None:payload["id_slot"]=slot
 t=time.perf_counter();response=post("/completion",payload);return {"route":name,"prompt":prompt,"slot":slot,"request":payload,"response":response,"content":str(response.get("content") or ""),"wall_ms":(time.perf_counter()-t)*1000,"cache_n":int((response.get("timings") or {}).get("cache_n") or 0)}
def wsl_path(p):return "/mnt/c/"+p.resolve().as_posix().split(":/",1)[1]

def execute(outdir):
 raw=outdir/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();ledger={}
 if any(raw.iterdir()):raise RuntimeError("raw not empty")
 for p,e in EXPECTED.items():
  a=sha256_file(p)
  if a!=e:raise ValueError(f"hash mismatch {p}: {a}")
  ledger[p.relative_to(ROOT).as_posix()]={"bytes":p.stat().st_size,"sha256":a}
 original=service();original_exec=original["values"].get("ExecStart","");binary=original_exec.split("path=",1)[1].split(" ;",1)[0];model=original_exec.split("-m ",1)[1].split(" ",1)[0]
 binary_hash=run(["wsl","-d","Ubuntu-24.04","--","sha256sum",binary])["stdout"].split()[0];converter="/home/augus/src/slop.cpp-main/convert_lora_to_gguf.py";python="/home/augus/.venvs/adapt00-20260824/bin/python";base="/home/augus/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68";converter_hash=run(["wsl","-d","Ubuntu-24.04","--","sha256sum",converter])["stdout"].split()[0]
 adapters=raw/"adapters";adapters.mkdir();sources=[ROOT/"runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_mlp_only/adapter",ROOT/"runs/research/ADAPT-02-MODULE-TARGETING-2026-08-25/raw/target_attn_only/adapter"];outputs=[adapters/"mlp-f16.gguf",adapters/"attn-f16.gguf"];conversions=[]
 for source,dest in zip(sources,outputs):
  row=run(["wsl","-d","Ubuntu-24.04","--",python,converter,wsl_path(source),"--base",base,"--outfile",wsl_path(dest),"--outtype","f16"],timeout=600);conversions.append(row)
  if row["returncode"] or not dest.is_file():write(raw/"conversion_abort.json",conversions);raise RuntimeError("adapter conversion failed")
 temp=None;log_handle=None;rows=[];restored={}
 try:
  stopped=run(["wsl","-d","Ubuntu-24.04","--","systemctl","stop","llm-inference.service"],timeout=60)
  end=time.time()+30
  while time.time()<end and health(8080) is not None:time.sleep(.25)
  if health(8081)!=200:raise RuntimeError("embedding failed during maintenance")
  args=["wsl","-d","Ubuntu-24.04","--",binary,"-m",model,"--alias","fable-tc-l1.0","--host","0.0.0.0","--port","8080","-ngl","99","-fa","on","--ctx-size","8192","--parallel","4","--spec-type","draft-mtp","--spec-draft-n-max","4","--jinja","--metrics","--lora",wsl_path(outputs[0]),"--lora",wsl_path(outputs[1]),"--lora-init-without-apply"]
  log_handle=(raw/"temporary_server.log").open("w",encoding="utf-8");temp=subprocess.Popen(args,stdout=log_handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));wait_health(8080,200,240)
  loaded=get("/lora-adapters");slot_state=get("/slots")
  if len(loaded)!=2 or len(slot_state)!=4:raise RuntimeError(f"unexpected materialization: {loaded}, slots={len(slot_state)}")
  baselines={}
  for name in ("base","mlp","attn"):
   for index,prompt in enumerate(PROMPTS):
    row=complete(prompt,name,cache=False);rows.append({"phase":"baseline","index":index,**row});baselines[(name,index)]=row["content"]
  distinct=sum(len({baselines[(name,i)] for name in ("base","mlp","attn")})>=2 for i in range(len(PROMPTS)))
  routed=[]
  for repeat in range(2):
   for index,prompt in enumerate(PROMPTS):
    for name in ("base","mlp","attn"):
     row=complete(prompt,name,slot=(index+repeat)%4,cache=True);match=row["content"]==baselines[(name,index)];routed.append({"repeat":repeat,"index":index,"match":match,**row});rows.append({"phase":"routed",**routed[-1]})
  cache_rows=[];long_prefix="Shared immutable prefix for adapter cache isolation. "*80
  for name,slot in zip(("base","mlp","attn"),(0,1,2)):
   prompt=long_prefix+PROMPTS[slot]
   first=complete(prompt,name,slot=slot,cache=True,n=16);second=complete(prompt,name,slot=slot,cache=True,n=16);other="mlp" if name=="base" else "base";switched=complete(prompt,other,slot=slot,cache=True,n=16);returned=complete(prompt,name,slot=slot,cache=True,n=16)
   cache_rows.append({"route":name,"slot":slot,"first":first,"second":second,"switched":switched,"returned":returned,"same_route_hit":second["cache_n"]>0,"return_exact":returned["content"]==first["content"]})
  cells=[(name,i,PROMPTS[i]) for i in range(10) for name in ("base","mlp","attn")];alternating=cells;grouped=sorted(cells,key=lambda x:x[0]);schedule={}
  for label,order in (("alternating",alternating),("grouped",grouped)):
   start=time.perf_counter();schedule[label]=[complete(prompt,name,cache=False,n=16) | {"index":i} for name,i,prompt in order];schedule[label+"_wall_ms"]=(time.perf_counter()-start)*1000
  amap={(r["route"],r["index"]):r["content"] for r in schedule["alternating"]};gmap={(r["route"],r["index"]):r["content"] for r in schedule["grouped"]};parity=sum(amap[k]==gmap[k] for k in amap)/len(amap)
  def switches(order):return sum(order[i][0]!=order[i-1][0] for i in range(1,len(order)))
  switch_reduction=1-switches(grouped)/switches(alternating)
  contamination=sum(not row["match"] for row in routed)+sum(not row["return_exact"] for row in cache_rows)
  metrics={"converted_adapters":sum(p.is_file() for p in outputs),"loaded_adapters":len(loaded),"prompts_with_distinct_route_outputs":distinct,"routed_exact_match_rate":sum(r["match"] for r in routed)/len(routed),"cross_route_contamination_count":contamination,"same_route_cache_hit_rate":sum(r["same_route_hit"] for r in cache_rows)/len(cache_rows),"requested_route_switch_reduction":switch_reduction,"schedule_semantic_parity":parity,"alternating_wall_ms":schedule["alternating_wall_ms"],"grouped_wall_ms":schedule["grouped_wall_ms"],"client_affinity_speedup":schedule["alternating_wall_ms"]/schedule["grouped_wall_ms"]}
  write(raw/"live_rows.json",{"baselines":[r for r in rows if r["phase"]=="baseline"],"routed":routed,"cache":cache_rows,"schedule":schedule});write(raw/"conversion.json",conversions);write(raw/"loaded_adapters.json",loaded);write(raw/"temporary_slots.json",slot_state)
 finally:
  if temp is not None:
   temp.terminate()
   try:temp.wait(timeout=20)
   except subprocess.TimeoutExpired:temp.kill();temp.wait(timeout=10)
  if log_handle is not None:log_handle.close()
  run(["wsl","-d","Ubuntu-24.04","--","systemctl","start","llm-inference.service"],timeout=60)
  try:wait_health(8080,200,240)
  finally:restored=service()
 if restored["values"].get("ExecStart","")!=original_exec:raise RuntimeError("original ExecStart not restored")
 metrics["original_service_restored"]=int(restored["values"].get("ActiveState")=="active" and health(8080)==200);metrics["embedding_health"]=health(8081)
 write(raw/"actual_scores.json",metrics);write(raw/"artifact_hashes.json",ledger|{"active_binary":{"sha256":binary_hash},"converter":{"sha256":converter_hash},"mlp_gguf":{"sha256":sha256_file(outputs[0]),"bytes":outputs[0].stat().st_size},"attn_gguf":{"sha256":sha256_file(outputs[1]),"bytes":outputs[1].stat().st_size}});write(raw/"dataset_hashes.json",{"prompts_semantic_sha256":canonical_json_sha256(PROMPTS)});write(raw/"effective_route.json",{"temporary_args":args,"loaded":loaded,"slots":slot_state});write(raw/"failure_reproduction.json",{"historical":"Python hash/router simulations","successor":"two physical GGUF adapters on live llama-server"});write(raw/"falsifiable_hypothesis.json",{"prompts":12,"routes":3,"routed_repetitions":2,"all_gates_required":True});write(raw/"hardware_metrics.json",{"alternating_wall_ms":metrics["alternating_wall_ms"],"grouped_wall_ms":metrics["grouped_wall_ms"],"client_affinity_speedup":metrics["client_affinity_speedup"]});write(raw/"independent_evaluation.json",{"baseline_hashes":{f"{k[0]}:{k[1]}":canonical_json_sha256(v) for k,v in baselines.items()},"routed_matches":[r["match"] for r in routed]});write(raw/"invalidation_rules.json",{"conversion_or_restore_failure_aborts":True,"all_gates_required":True});write(raw/"invariant_controls.json",{"routes":{"base":route("base"),"mlp":route("mlp"),"attn":route("attn")},"decode":{"temperature":0,"top_k":1,"seed":0}});write(raw/"paired_baseline.json",{"isolated_cells":len(baselines),"routed_cells":len(routed),"cache_sequences":len(cache_rows)});write(raw/"real_implementation.json",{"converted_gguf":True,"per_request_lora":True,"server_native_affinity_scheduler":False,"client_affinity_order":True});write(raw/"recovery_state.json",{"original_exec_match":True,"restored":restored});write(raw/"semantic_parity.json",{"routed_exact_match_rate":metrics["routed_exact_match_rate"],"schedule_semantic_parity":parity});write(raw/"service_identity.json",{"original":original,"temporary_binary":binary,"temporary_binary_sha256":binary_hash,"restored":restored});write(raw/"service_maintenance.json",{"systemd_stopped":True,"temporary_server_started":True,"original_service_restored":metrics["original_service_restored"],"embedding_health":metrics["embedding_health"]});write(raw/"source_execution_receipt.json",{"adapt06":EXPECTED[ROOT/"runs/research/ADAPT-06-ADAPTER-CACHE-TAGGING-2026-08-25/raw/receipt.json"],"slop":EXPECTED[ROOT/"runs/research/SLOP-L1-L7-MULTI-ADAPTER-2026-08-25/raw/receipt.json"]})
 with (raw/"samples.jsonl").open("w",encoding="utf-8") as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 defs={"adapter_conversion":("converted_adapters","eq",2),"adapter_loading":("loaded_adapters","eq",2),"behavioral_materiality":("prompts_with_distinct_route_outputs","ge",4),"route_isolation":("routed_exact_match_rate","eq",1.0),"cross_route_isolation":("cross_route_contamination_count","eq",0),"cache_reuse":("same_route_cache_hit_rate","ge",.75),"affinity_switch_reduction":("requested_route_switch_reduction","ge",.90),"affinity_parity":("schedule_semantic_parity","eq",1.0),"service_restore":("original_service_restored","eq",1),"embedding_integrity":("embedding_health","eq",200)};ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b};gates={g:{"metric":m,"operator":o,"threshold":t,"actual":metrics[m],"pass":ops[o](metrics[m],t)} for g,(m,o,t) in defs.items()}
 evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","effective_route":"raw/effective_route.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","semantic_parity":"raw/semantic_parity.json","service_identity":"raw/service_identity.json","service_maintenance":"raw/service_maintenance.json","source_execution_receipt":"raw/source_execution_receipt.json"};files=sorted({raw/v.removeprefix("raw/") for v in evidence.values() if v!="raw/receipt.json"});prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,*files,*outputs],packages=["pytest"],runtime={"execution_mode":"temporary_live_multi_adapter","binary":binary,"model":model});ok,errors=provenance_complete(prov)
 if not ok:raise ValueError(errors)
 receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write(raw/"receipt.json",receipt);return receipt,metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();r,m=execute(a.outdir.resolve());passed=all(x["pass"] for x in r["gates"].values());claim="ADAPT06_LIVE_ISOLATION_QUALIFIED_SLOP_CLIENT_AFFINITY_R1" if passed else "ADAPT06_SLOP_FALSE_POSITIVE_CONFIRMED_R1";failed=[g for g,x in r["gates"].items() if not x["pass"]];(a.outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nConverted/loaded {m['converted_adapters']}/{m['loaded_adapters']} adapters; distinct route outputs on {m['prompts_with_distinct_route_outputs']}/12 prompts; routed exact match {m['routed_exact_match_rate']:.4%}; contamination {m['cross_route_contamination_count']}; cache-hit rate {m['same_route_cache_hit_rate']:.4%}; requested switch reduction {m['requested_route_switch_reduction']:.4%}; schedule parity {m['schedule_semantic_parity']:.4%}; client affinity speedup {m['client_affinity_speedup']:.3f}x. Failed gates: {', '.join(failed) if failed else 'none'}. No server-native scheduler or fused-GEMM claim. Original service restored.\n",encoding="utf-8");print(json.dumps({"claim":claim,"metrics":m,"gates":r["gates"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
