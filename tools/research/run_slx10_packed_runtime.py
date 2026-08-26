#!/usr/bin/env python3
"""Physical IQ2_XXS GGUF and live serving audit for SLX-10."""
from __future__ import annotations
import argparse,json,pathlib,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
from tools.analysis.a2_stats import gsm8k_extract,numeric_equal
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-SLX10-PACKED-RUNTIME-01";BASE="http://127.0.0.1:8080";F16=ROOT/"runs/research/BACKLOG-ADAPT06-SLOP-LIVE-05/raw/qwen3.5-0.8b-base-f16.gguf"
EXPECTED={ROOT/"config/research_backlog_admissions/BACKLOG-SLX10-PACKED-RUNTIME-01.json":"35172b24885ff8dee0d489675a9b98f2ecf7f9e2e0bddbafd600916217344a38",ROOT/"runs/research/BACKLOG-SLX10-PACKED-RUNTIME-01/PRE_REGISTRATION.md":"f1fdf716671ea11108c7628d5f12984c00c2231c4f2a64d6e81fcd039b12afcf",ROOT/"runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/PRE_REGISTRATION.md":"f985a570fc485956cc7408077f1a07ae0e135e7c3d46e8d1e6b2a6948d0ed9b7",ROOT/"runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/RESULT.md":"e33f412d2c3099844e10a202c80e41a9a65ada172c85df019b5f20876571469a",ROOT/"runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/raw/receipt.json":"ff7365541552058c5a13d2475b1b47c1c9a7acb877d52ce0575f53fcb3dabb34",ROOT/"tools/probes/slx10_physical_codec_bakeoff.py":"717f68023469179a65ad86cc4289de7b44670cf05c1de0c1fd31b87875a904d0",F16:"514133770c0e30367721334fb86a76a8647bf8ab4d51fedc62980ce86dda1ac1",ROOT/"runs/research/BACKLOG-ADAPT-TRAIN-01/raw/samples.jsonl":"243311c37ff240d97f63539c4e85f3a9ec7272ea8eaa1279d31c7c38d44d50c4"}
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def run(argv,timeout=1200):
 d=subprocess.run(argv,capture_output=True,text=True,encoding="utf-8",errors="replace",check=False,timeout=timeout);return {"argv":argv,"returncode":d.returncode,"stdout":d.stdout.strip(),"stderr":d.stderr.strip()}
def health(port):
 try:
  with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=5) as r:return r.status
 except Exception:return None
def wait_health(timeout=240):
 end=time.time()+timeout
 while time.time()<end:
  if health(8080)==200:return
  time.sleep(.5)
 raise RuntimeError("8080 health timeout")
def wait_down(timeout=30):
 end=time.time()+timeout
 while time.time()<end:
  if health(8080) is None:return
  time.sleep(.25)
 raise RuntimeError("8080 remained reachable")
def service():
 x=run(["wsl","-d","Ubuntu-24.04","-e","systemctl","show","llm-inference.service","-p","MainPID","-p","NRestarts","-p","ActiveState","-p","ExecStart","--no-pager"]);return {"raw":x,"values":dict(line.split("=",1) for line in x["stdout"].splitlines() if "=" in line)}
def stable_exec(v):return v.split(" ; ignore_errors=",1)[0].strip()
def vram_mib():
 row=run(["wsl","-d","Ubuntu-24.04","-e","/usr/lib/wsl/lib/nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"])
 if row["returncode"]:raise RuntimeError(row)
 return int(row["stdout"].splitlines()[0].strip())
def post(payload):
 req=urllib.request.Request(BASE+"/completion",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=180) as r:return json.loads(r.read().decode())
def collect(arm,panel):
 rows=[]
 for i,item in enumerate(panel):
  payload={"prompt":item["prompt"],"n_predict":128,"temperature":0.0,"top_k":1,"seed":0,"cache_prompt":False,"stream":False,"id_slot":i%4};start=time.perf_counter();response=post(payload);wall=(time.perf_counter()-start)*1000;text=str(response.get("content") or "");extracted=gsm8k_extract(text);timings=response.get("timings") or {};rows.append({"arm":arm,"index":i,"task_id":item["task_id"],"prompt":item["prompt"],"gold":item["gold"],"content":text,"extracted":extracted,"correct":numeric_equal(extracted,item["gold"]),"wall_ms":wall,"predicted_n":int(timings.get("predicted_n") or response.get("tokens_predicted") or 0),"response":response})
 return rows
def wsl_path(p):return "/mnt/c/"+p.resolve().as_posix().split(":/",1)[1]
def execute(outdir):
 raw=outdir/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();ledger={}
 if any(raw.iterdir()):raise RuntimeError("raw not empty")
 for p,e in EXPECTED.items():
  a=sha256_file(p)
  if a!=e:raise ValueError(f"hash mismatch {p}: {a}")
  ledger[p.relative_to(ROOT).as_posix()]={"bytes":p.stat().st_size,"sha256":a}
 panel=[];seen=set()
 for line in (ROOT/"runs/research/BACKLOG-ADAPT-TRAIN-01/raw/samples.jsonl").read_text(encoding="utf-8").splitlines():
  r=json.loads(line)
  if r.get("arm")=="base" and r.get("panel")=="math" and r["task_id"] not in seen:seen.add(r["task_id"]);panel.append({"task_id":r["task_id"],"prompt":r["prompt"],"gold":r["gold"]})
  if len(panel)==32:break
 if len(panel)!=32:raise RuntimeError("panel resolution failed")
 original=service();original_exec=original["values"].get("ExecStart","");binary=original_exec.split("path=",1)[1].split(" ;",1)[0];quantizer=str(pathlib.PurePosixPath(binary).with_name("llama-quantize"));binary_hash=run(["wsl","-d","Ubuntu-24.04","-e","sha256sum",binary])["stdout"].split()[0];quantizer_hash=run(["wsl","-d","Ubuntu-24.04","-e","sha256sum",quantizer])["stdout"].split()[0];iq2=raw/"qwen3.5-0.8b-base-iq2_xxs.gguf";q=run(["wsl","-d","Ubuntu-24.04","-e",quantizer,wsl_path(F16),wsl_path(iq2),"IQ2_XXS"])
 if q["returncode"] or not iq2.is_file():write(raw/"quantization_abort.json",q);raise RuntimeError("quantization failed")
 arms={};vrams={};launches=[];temp=None;handle=None
 try:
  stop=run(["wsl","-d","Ubuntu-24.04","-u","root","-e","systemctl","stop","llm-inference.service"],timeout=60)
  if stop["returncode"]:raise RuntimeError(stop)
  wait_down();background=vram_mib()
  if health(8081)!=200:raise RuntimeError("embedding down")
  for arm,model in (("f16",F16),("iq2",iq2)):
   args=["wsl","-d","Ubuntu-24.04","-e",binary,"-m",wsl_path(model),"--alias",arm,"--host","0.0.0.0","--port","8080","-ngl","99","-fa","on","--ctx-size","4096","--parallel","4","--jinja","--metrics"];launches.append(args);handle=(raw/f"server_{arm}.log").open("w",encoding="utf-8");temp=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));wait_health();time.sleep(1);vrams[arm]={"total_mib":vram_mib(),"net_mib":vram_mib()-background};arms[arm]=collect(arm,panel);temp.terminate()
   try:temp.wait(timeout=20)
   except subprocess.TimeoutExpired:temp.kill();temp.wait(timeout=10)
   handle.close();temp=None;handle=None;wait_down()
 finally:
  if temp is not None:
   temp.terminate()
   try:temp.wait(timeout=20)
   except subprocess.TimeoutExpired:temp.kill()
  if handle is not None:handle.close()
  start_result=run(["wsl","-d","Ubuntu-24.04","-u","root","-e","systemctl","start","llm-inference.service"],timeout=60)
  if start_result["returncode"]:raise RuntimeError(start_result)
  wait_health();restored=service()
 if stable_exec(restored["values"].get("ExecStart",""))!=stable_exec(original_exec):raise RuntimeError("service argv not restored")
 def throughput(rows):return sum(r["predicted_n"] for r in rows)/sum(r["wall_ms"] for r in rows)
 f16_acc=sum(r["correct"] for r in arms["f16"])/32;iq2_acc=sum(r["correct"] for r in arms["iq2"])/32;exact=sum(a["content"]==b["content"] for a,b in zip(arms["f16"],arms["iq2"]))/32;tp_f16=throughput(arms["f16"]);tp_iq2=throughput(arms["iq2"]);metrics={"iq2_file_ratio":iq2.stat().st_size/F16.stat().st_size,"loaded_arms":len(arms),"vram_reduction":1-vrams["iq2"]["net_mib"]/vrams["f16"]["net_mib"],"throughput_ratio":tp_iq2/tp_f16,"accuracy_regression":f16_acc-iq2_acc,"exact_output_rate":exact,"f16_accuracy":f16_acc,"iq2_accuracy":iq2_acc,"f16_throughput":tp_f16,"iq2_throughput":tp_iq2,"original_service_restored":int(restored["values"].get("ActiveState")=="active" and health(8080)==200),"embedding_health":health(8081)}
 with (raw/"samples.jsonl").open("w",encoding="utf-8") as f:
  for arm in ("f16","iq2"):
   for row in arms[arm]:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 write(raw/"actual_scores.json",metrics);write(raw/"artifact_hashes.json",ledger|{"binary":{"sha256":binary_hash},"quantizer":{"sha256":quantizer_hash},"iq2":{"sha256":sha256_file(iq2),"bytes":iq2.stat().st_size}});write(raw/"dataset_hashes.json",{"panel_semantic_sha256":canonical_json_sha256(panel)});write(raw/"effective_route.json",{"launches":launches,"arms":["F16","IQ2_XXS"]});write(raw/"failure_reproduction.json",{"historical":"allocated random matrices plus arithmetic model-size projection","successor":"immutable IQ2 GGUF loaded by live runtime"});write(raw/"falsifiable_hypothesis.json",{"codec":"IQ2_XXS","panel":32,"all_gates_required":True});write(raw/"hardware_metrics.json",{"background_vram_mib":background,"arm_vram":vrams,"throughput":{"f16":tp_f16,"iq2":tp_iq2}});write(raw/"independent_evaluation.json",{"f16_correct":sum(r["correct"] for r in arms["f16"]),"iq2_correct":sum(r["correct"] for r in arms["iq2"]),"exact_pairs":int(exact*32)});write(raw/"invalidation_rules.json",{"quantization_or_load_failure_aborts":True,"service_restore_failure_aborts":True});write(raw/"invariant_controls.json",{"decode":{"n_predict":128,"temperature":0,"top_k":1,"seed":0,"cache_prompt":False},"server":{"ctx_size":4096,"parallel":4}});write(raw/"paired_baseline.json",{"task_ids":[p["task_id"] for p in panel],"baseline":"F16","treatment":"IQ2_XXS"});write(raw/"real_implementation.json",{"physical_packed_gguf":True,"quantizer_type":"IQ2_XXS","live_load":True});write(raw/"recovery_state.json",{"stable_exec_match":True,"restored":restored});write(raw/"semantic_parity.json",{"exact_output_rate":exact,"accuracy_regression":metrics["accuracy_regression"]});write(raw/"service_identity.json",{"original":original,"binary_sha256":binary_hash,"restored":restored});write(raw/"service_maintenance.json",{"root_handoff":True,"original_service_restored":metrics["original_service_restored"],"embedding_health":metrics["embedding_health"]});write(raw/"source_execution_receipt.json",{"historical_receipt_sha256":EXPECTED[ROOT/"runs/research/SLX-10-PHYSICAL-CODEC-2026-08-25/raw/receipt.json"],"quantization_command":q})
 defs={"packed_artifact":("iq2_file_ratio","le",.18),"physical_load":("loaded_arms","eq",2),"memory_reduction":("vram_reduction","ge",.60),"throughput":("throughput_ratio","ge",.95),"quality":("accuracy_regression","le",.02),"semantic_stability":("exact_output_rate","ge",.80),"service_restore":("original_service_restored","eq",1),"embedding_integrity":("embedding_health","eq",200)};ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b,"le":lambda a,b:a<=b};gates={g:{"metric":m,"operator":o,"threshold":t,"actual":metrics[m],"pass":ops[o](metrics[m],t)} for g,(m,o,t) in defs.items()};evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","effective_route":"raw/effective_route.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","semantic_parity":"raw/semantic_parity.json","service_identity":"raw/service_identity.json","service_maintenance":"raw/service_maintenance.json","source_execution_receipt":"raw/source_execution_receipt.json"};files=sorted({raw/v.removeprefix("raw/") for v in evidence.values() if v!="raw/receipt.json"});prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,*files,iq2],packages=["pytest"],runtime={"execution_mode":"physical_iq2_live_serving","binary":binary,"quantizer":quantizer});ok,errors=provenance_complete(prov)
 if not ok:raise ValueError(errors)
 receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write(raw/"receipt.json",receipt);return receipt,metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();r,m=execute(a.outdir.resolve());passed=all(x["pass"] for x in r["gates"].values());claim="SLX10_IQ2_PHYSICAL_QUALIFIED_R1" if passed else "SLX10_FALSE_POSITIVE_CONFIRMED_R1";failed=[g for g,x in r["gates"].items() if not x["pass"]];(a.outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nIQ2/F16 file ratio {m['iq2_file_ratio']:.4%}; net VRAM reduction {m['vram_reduction']:.4%}; throughput ratio {m['throughput_ratio']:.4f}; F16/IQ2 accuracy {m['f16_accuracy']:.4%}/{m['iq2_accuracy']:.4%}; exact output rate {m['exact_output_rate']:.4%}. Failed gates: {', '.join(failed) if failed else 'none'}. Scope is the frozen 0.8B model only. Original service restored.\n",encoding="utf-8");print(json.dumps({"claim":claim,"metrics":m,"gates":r["gates"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
