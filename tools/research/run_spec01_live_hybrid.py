#!/usr/bin/env python3
"""Physical MTP plus n-gram serving audit for SPEC-01."""
from __future__ import annotations
import argparse,json,pathlib,re,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-SPEC01-LIVE-HYBRID-01";BASE="http://127.0.0.1:8080"
EXPECTED={ROOT/"config/research_backlog_admissions/BACKLOG-SPEC01-LIVE-HYBRID-01.json":"afda2b262a487f4dd7e0876f5c01f20925481b8258494c63105560924d1b0403",ROOT/"runs/research/BACKLOG-SPEC01-LIVE-HYBRID-01/PRE_REGISTRATION.md":"1fcbcaf5bc04773573f83f7c4a51a6a1fdbb09401cd317420889db6b4d724df4",ROOT/"runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/PRE_REGISTRATION.md":"62fecb2683c511546b8b37703698673f4806869dec7a7094ead33c61b77e16b1",ROOT/"runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/RESULT.md":"e4afcf36f830ff0ebbe32a8971065d68797beed169f9d928c56a025169c619fc",ROOT/"runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/raw/receipt.json":"26b0c2857396cb080659a0217dd5b88bebe08c876655caab5c57b82a0d4c7da0",ROOT/"tools/analysis/hybrid_speculative_engine.py":"7a3c8456498848f003476137a88b3e472ac5d1a91be476bd57ab8ac93f5d44ab",ROOT/"tests/test_hybrid_speculative_engine.py":"61c627f1d1874efbf42d25858105211042b55e182885fcbc34cf2697022f4c50"}
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def run(argv,timeout=600):
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
 raise RuntimeError("health timeout")
def wait_down(timeout=30):
 end=time.time()+timeout
 while time.time()<end:
  if health(8080) is None:return
  time.sleep(.25)
 raise RuntimeError("8080 remained reachable")
def service():
 x=run(["wsl","-d","Ubuntu-24.04","-e","systemctl","show","llm-inference.service","-p","MainPID","-p","NRestarts","-p","ActiveState","-p","ExecStart","--no-pager"]);return {"raw":x,"values":dict(line.split("=",1) for line in x["stdout"].splitlines() if "=" in line)}
def stable_exec(v):return v.split(" ; ignore_errors=",1)[0].strip()
def get_text(path):
 with urllib.request.urlopen(BASE+path,timeout=30) as r:return r.read().decode(errors="replace")
def post(payload):
 req=urllib.request.Request(BASE+"/completion",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=180) as r:return json.loads(r.read().decode())
def prompts():
 rows=[]
 for case in range(30):
  pattern=[{"case":case,"step":i%4,"value":f"V{case:02d}-{i%4}"} for i in range(24)]
  rows.append({"case":case,"prompt":"Continue this JSONL pattern with the next records and no explanation:\n"+"\n".join(json.dumps(x,separators=(",",":")) for x in pattern)+"\n"})
 return rows
def collect(arm,panel):
 rows=[]
 for i,item in enumerate(panel):
  payload={"prompt":item["prompt"],"n_predict":128,"temperature":0.0,"top_k":1,"seed":0,"cache_prompt":False,"stream":False,"id_slot":i%4};start=time.perf_counter();response=post(payload);wall=(time.perf_counter()-start)*1000;timings=response.get("timings") or {};settings=response.get("generation_settings") or {};rows.append({"arm":arm,"case":item["case"],"prompt":item["prompt"],"content":str(response.get("content") or ""),"wall_ms":wall,"predicted_n":int(timings.get("predicted_n") or response.get("tokens_predicted") or 0),"draft_n":int(timings.get("draft_n") or 0),"draft_n_accepted":int(timings.get("draft_n_accepted") or 0),"speculative_types":str(settings.get("speculative.types") or ""),"response":response})
 return rows
def execute(outdir):
 raw=outdir/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();ledger={}
 if any(raw.iterdir()):raise RuntimeError("raw not empty")
 for p,e in EXPECTED.items():
  a=sha256_file(p)
  if a!=e:raise ValueError(f"hash mismatch {p}: {a}")
  ledger[p.relative_to(ROOT).as_posix()]={"bytes":p.stat().st_size,"sha256":a}
 panel=prompts();original=service();original_exec=original["values"].get("ExecStart","");binary=original_exec.split("path=",1)[1].split(" ;",1)[0];model=original_exec.split("-m ",1)[1].split(" ",1)[0];binary_hash=run(["wsl","-d","Ubuntu-24.04","-e","sha256sum",binary])["stdout"].split()[0];model_hash=run(["wsl","-d","Ubuntu-24.04","-e","sha256sum",model])["stdout"].split()[0]
 if "--spec-type draft-mtp" not in original_exec:raise RuntimeError("baseline is not MTP")
 baseline=collect("mtp",panel);temp=None;handle=None
 try:
  stop=run(["wsl","-d","Ubuntu-24.04","-u","root","-e","systemctl","stop","llm-inference.service"],timeout=60)
  if stop["returncode"]:raise RuntimeError(stop)
  wait_down()
  args=["wsl","-d","Ubuntu-24.04","-e",binary,"-m",model,"--alias","fable-tc-l1.0","--host","0.0.0.0","--port","8080","-ngl","99","-fa","on","--ctx-size","8192","--parallel","4","--spec-type","draft-mtp,ngram-cache","--spec-draft-n-max","4","--jinja","--metrics"];handle=(raw/"server_hybrid.log").open("w",encoding="utf-8");temp=subprocess.Popen(args,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0));wait_health();hybrid=collect("hybrid",panel);metrics_text=get_text("/metrics")
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
 exact=sum(a["content"]==b["content"] for a,b in zip(baseline,hybrid))/30
 def throughput(rows):return sum(r["predicted_n"] for r in rows)/sum(r["wall_ms"] for r in rows)
 base_tp=throughput(baseline);hybrid_tp=throughput(hybrid);route_confirmed=sum("draft-mtp" in r["speculative_types"] and "ngram-cache" in r["speculative_types"] for r in hybrid);attribution=bool(re.search(r"ngram.{0,60}(accepted|draft).{0,120}mtp.{0,60}(accepted|draft)|mtp.{0,60}(accepted|draft).{0,120}ngram.{0,60}(accepted|draft)",metrics_text,re.I|re.S));metrics={"live_requests":len(baseline)+len(hybrid),"hybrid_route_confirmed":route_confirmed,"exact_output_rate":exact,"hybrid_speedup":hybrid_tp/base_tp,"per_proposer_attribution_available":int(attribution),"hybrid_requests_with_drafts":sum(r["draft_n"]>0 for r in hybrid),"baseline_throughput":base_tp,"hybrid_throughput":hybrid_tp,"baseline_draft_tokens":sum(r["draft_n"] for r in baseline),"hybrid_draft_tokens":sum(r["draft_n"] for r in hybrid),"original_service_restored":int(restored["values"].get("ActiveState")=="active" and health(8080)==200),"embedding_health":health(8081)}
 with (raw/"samples.jsonl").open("w",encoding="utf-8") as f:
  for row in baseline+hybrid:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 write(raw/"actual_scores.json",metrics);write(raw/"artifact_hashes.json",ledger|{"binary":{"sha256":binary_hash},"model":{"sha256":model_hash}});write(raw/"dataset_hashes.json",{"prompt_panel_semantic_sha256":canonical_json_sha256(panel)});write(raw/"effective_route.json",{"baseline":original_exec,"hybrid_args":args,"advertised_types":[r["speculative_types"] for r in hybrid]});write(raw/"failure_reproduction.json",{"historical":"synthetic target sequences and simulated engine","successor":"deployed combined proposer route"});write(raw/"falsifiable_hypothesis.json",{"prompts":30,"historical_speedup_gate":3.0,"all_gates_required":True});write(raw/"hardware_metrics.json",{"baseline_throughput":base_tp,"hybrid_throughput":hybrid_tp,"speedup":metrics["hybrid_speedup"]});write(raw/"independent_evaluation.json",{"exact_pairs":int(exact*30),"metrics_text":metrics_text,"per_proposer_attribution_available":attribution});write(raw/"invalidation_rules.json",{"aggregate_draft_n_not_per_proposer_attribution":True,"restore_failure_aborts":True});write(raw/"invariant_controls.json",{"decode":{"n_predict":128,"temperature":0,"top_k":1,"seed":0,"cache_prompt":False},"hybrid_types":["draft-mtp","ngram-cache"]});write(raw/"paired_baseline.json",{"cases":list(range(30)),"baseline":"draft-mtp","treatment":"draft-mtp,ngram-cache"});write(raw/"real_implementation.json",{"combined_runtime_route":True,"per_proposer_telemetry":attribution});write(raw/"recovery_state.json",{"stable_exec_match":True,"restored":restored});write(raw/"semantic_parity.json",{"exact_output_rate":exact});write(raw/"service_identity.json",{"original":original,"binary_sha256":binary_hash,"model_sha256":model_hash,"restored":restored});write(raw/"service_maintenance.json",{"root_handoff":True,"original_service_restored":metrics["original_service_restored"],"embedding_health":metrics["embedding_health"]});write(raw/"source_execution_receipt.json",{"historical_receipt_sha256":EXPECTED[ROOT/"runs/research/SPEC-01-SPECULATIVE-PIPELINE-2026-08-25/raw/receipt.json"]})
 defs={"request_coverage":("live_requests","eq",60),"hybrid_route":("hybrid_route_confirmed","eq",30),"semantic_parity":("exact_output_rate","eq",1.0),"historical_speedup":("hybrid_speedup","ge",3.0),"ngram_attribution":("per_proposer_attribution_available","eq",1),"draft_coverage":("hybrid_requests_with_drafts","ge",25),"service_restore":("original_service_restored","eq",1),"embedding_integrity":("embedding_health","eq",200)};ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b};gates={g:{"metric":m,"operator":o,"threshold":t,"actual":metrics[m],"pass":ops[o](metrics[m],t)} for g,(m,o,t) in defs.items()};evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","effective_route":"raw/effective_route.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","semantic_parity":"raw/semantic_parity.json","service_identity":"raw/service_identity.json","service_maintenance":"raw/service_maintenance.json","source_execution_receipt":"raw/source_execution_receipt.json"};files=sorted({raw/v.removeprefix("raw/") for v in evidence.values() if v!="raw/receipt.json"});prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,*files],packages=["pytest"],runtime={"execution_mode":"live_hybrid_ngram_mtp","binary":binary,"model":model});ok,errors=provenance_complete(prov)
 if not ok:raise ValueError(errors)
 receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write(raw/"receipt.json",receipt);return receipt,metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();r,m=execute(a.outdir.resolve());passed=all(x["pass"] for x in r["gates"].values());claim="SPEC01_LIVE_HYBRID_QUALIFIED_R1" if passed else "SPEC01_FALSE_POSITIVE_CONFIRMED_R1";failed=[g for g,x in r["gates"].items() if not x["pass"]];(a.outdir/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\n60 physical requests; hybrid route confirmed {m['hybrid_route_confirmed']}/30; exact parity {m['exact_output_rate']:.4%}; speedup {m['hybrid_speedup']:.4f}x; per-proposer attribution {bool(m['per_proposer_attribution_available'])}; draft coverage {m['hybrid_requests_with_drafts']}/30. Failed gates: {', '.join(failed) if failed else 'none'}. Original service restored.\n",encoding="utf-8");print(json.dumps({"claim":claim,"metrics":m,"gates":r["gates"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
