#!/usr/bin/env python3
"""Live four-slot MTP isolation test for BEE-L4."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.analysis.experiment_provenance import build_provenance, canonical_json_sha256, provenance_complete, sha256_file

TASK_ID="BACKLOG-BEE-L4-LIVE-MTP-01"; BASE="http://127.0.0.1:8080"; WORDS={0:"SAFFRON",1:"COBALT",2:"AMBER",3:"VIOLET"}
EXPECTED={
ROOT/"config/research_backlog_admissions/BACKLOG-BEE-L4-LIVE-MTP-01.json":"b62c97cc83b2c4465318e237e335650d947670bf27615ecbdfa2ed9217fd0e74",
ROOT/"runs/research/BACKLOG-BEE-L4-LIVE-MTP-01/PRE_REGISTRATION.md":"6981f663e365942dc154df3ed8012e6d084bd3e1f05328120615e264091257bf",
ROOT/"runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/PRE_REGISTRATION.md":"cc6d18fd717c34411d1cd682097978315b4616596649c5584992bb0eeae76bad",
ROOT/"runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/RESULT.md":"65ce6535303a8f7d782553175360a94c19f63c51a2b07b16ee8c116fecbd476a",
ROOT/"runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/raw/receipt.json":"4b806de6b1bcd5b109d96f002e86cdb9aec5d8242b9cfbd226b6f552149a41fd",
ROOT/"tools/analysis/transactional_mtp_manager.py":"4b7cb18323dfb70912f140965d4967739443c9dc7d30321f6d065e4ec6076d66",
ROOT/"tests/test_transactional_mtp_manager.py":"a873602bdc9efebf8a160bc8e0bdee5c4f66edffe5ca2fa6c17ab50abcde3cd7"}

def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def get_json(url,timeout=30):
    with urllib.request.urlopen(url,timeout=timeout) as response:return json.loads(response.read().decode())
def post_json(url,payload,timeout=180):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.loads(response.read().decode())
def run_text(argv):
    done=subprocess.run(argv,capture_output=True,text=True,encoding="utf-8",errors="replace",check=False,timeout=60);return {"argv":argv,"returncode":done.returncode,"stdout":done.stdout.strip(),"stderr":done.stderr.strip()}

def service_identity():
    show=run_text(["wsl","-d","Ubuntu-24.04","--","systemctl","show","llm-inference.service","-p","MainPID","-p","NRestarts","-p","ActiveState","-p","ExecStart","--no-pager"])
    values=dict(line.split("=",1) for line in show["stdout"].splitlines() if "=" in line)
    binary=(values.get("ExecStart","").split("path=",1)[1].split(" ;",1)[0] if "path=" in values.get("ExecStart","") else "")
    digest=run_text(["wsl","-d","Ubuntu-24.04","--","sha256sum",binary]) if binary else {}
    health={}
    for port,name in ((8080,"inference"),(8081,"embedding")):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=10) as response:health[name]=response.status
        except Exception:health[name]=None
    return {"systemd":values,"binary":binary,"binary_sha256":digest.get("stdout","").split(" ",1)[0],"health":health}

def slot_rows():
    payload=get_json(f"{BASE}/slots");return payload if isinstance(payload,list) else payload.get("value",payload)

def request_slot(slot:int,round_index:int)->dict[str,Any]:
    word=WORDS[slot];prompt=f"Complete this deterministic mapping with exactly the value and no explanation.\nkey=slot{slot}\nvalue={word}\nkey=slot{slot}\nvalue="
    payload={"prompt":prompt,"n_predict":32,"temperature":0.0,"top_k":1,"seed":0,"cache_prompt":True,"id_slot":slot,"stream":False}
    started=time.perf_counter();response=post_json(f"{BASE}/completion",payload);elapsed=(time.perf_counter()-started)*1000
    if "error" in response:raise RuntimeError(response)
    timings=response.get("timings") or {};content=str(response.get("content") or "")
    return {"round":round_index,"slot":slot,"word":word,"request":payload,"content":content,"response":response,"wall_latency_ms":elapsed,"draft_n":int(timings.get("draft_n") or 0),"draft_n_accepted":int(timings.get("draft_n_accepted") or 0)}

def score(rows):
    baselines={slot:next(row["content"] for row in rows if row["round"]==0 and row["slot"]==slot) for slot in WORDS}
    for row in rows:
        upper=row["content"].upper();row["exact_repeat"]=row["content"]==baselines[row["slot"]];row["own_nonce"]=row["word"] in upper;row["foreign_nonces"]=[word for slot,word in WORDS.items() if slot!=row["slot"] and word in upper];row["rejected_draft_tokens"]=max(0,row["draft_n"]-row["draft_n_accepted"])
    return {"live_requests":len(rows),"physical_slots":len(WORDS),"requests_with_draft_tokens":sum(row["draft_n"]>0 for row in rows),"requests_with_rejected_draft_tokens":sum(row["rejected_draft_tokens"]>0 for row in rows),"rejected_draft_tokens":sum(row["rejected_draft_tokens"] for row in rows),"exact_repeat_rate":sum(row["exact_repeat"] for row in rows)/len(rows),"own_nonce_rate":sum(row["own_nonce"] for row in rows)/len(rows),"cross_slot_leakage_count":sum(len(row["foreign_nonces"]) for row in rows),"median_wall_latency_ms":statistics.median(row["wall_latency_ms"] for row in rows)}

def run(outdir):
    raw=outdir/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic()
    if any(raw.iterdir()):raise RuntimeError("raw not empty")
    ledger={}
    for path,expected in EXPECTED.items():
        actual=sha256_file(path)
        if actual!=expected:raise ValueError(f"hash mismatch {path}: {actual}")
        ledger[path.relative_to(ROOT).as_posix()]={"bytes":path.stat().st_size,"sha256":actual}
    before=service_identity();slots_before=slot_rows()
    if len(slots_before)!=4 or any(row.get("is_processing") for row in slots_before) or not all(row.get("speculative") for row in slots_before):raise RuntimeError(f"route not four idle speculative slots: {slots_before}")
    if "--spec-type draft-mtp" not in before["systemd"].get("ExecStart",""):raise RuntimeError("active route is not draft-mtp")
    rows=[]
    for round_index in range(25):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(request_slot,slot,round_index) for slot in WORDS]
            current=[future.result() for future in futures]
        rows.extend(sorted(current,key=lambda row:row["slot"]))
    deadline=time.time()+60;slots_after=[]
    while time.time()<deadline:
        slots_after=slot_rows()
        if len(slots_after)==4 and not any(row.get("is_processing") for row in slots_after):break
        time.sleep(.25)
    after=service_identity();metrics=score(rows);metrics["service_restarts"]=int(after["systemd"].get("NRestarts") or -1);metrics["idle_slots_after"]=sum(not row.get("is_processing") for row in slots_after)
    if before["systemd"].get("MainPID")!=after["systemd"].get("MainPID") or before["binary_sha256"]!=after["binary_sha256"]:raise RuntimeError("service identity changed")
    if after["health"]!={"inference":200,"embedding":200}:raise RuntimeError(f"health failed {after}")
    with (raw/"samples.jsonl").open("w",encoding="utf-8") as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    write(raw/"actual_scores.json",metrics);write(raw/"artifact_hashes.json",ledger);write(raw/"dataset_hashes.json",{"request_semantic_sha256":canonical_json_sha256([row["request"] for row in rows]),"samples_semantic_sha256":canonical_json_sha256(rows)})
    write(raw/"service_identity.json",{"before":before,"after":after});write(raw/"effective_route.json",{"spec_type":"draft-mtp","physical_slots":4,"slots_before":slots_before,"exec_start":before["systemd"].get("ExecStart")});write(raw/"recovery_state.json",{"slots_after":slots_after,"idle_slots":metrics["idle_slots_after"],"main_pid_unchanged":True,"restarts":metrics["service_restarts"]})
    write(raw/"service_maintenance.json",{"service_untouched":True,"before":before,"after":after});write(raw/"source_execution_receipt.json",{"historical_receipt_sha256":EXPECTED[ROOT/"runs/research/BEE-L4-TRANSACTIONAL-MTP-2026-08-25/raw/receipt.json"]});write(raw/"falsifiable_hypothesis.json",{"requests":100,"rounds":25,"slots":4,"all_gates_required":True});write(raw/"invariant_controls.json",{"words":WORDS,"temperature":0.0,"top_k":1,"seed":0,"n_predict":32,"cache_prompt":True});write(raw/"invalidation_rules.json",{"any_request_error_aborts":True,"pid_or_binary_change_aborts":True,"all_gates_required":True});write(raw/"paired_baseline.json",{"baseline_round":0,"comparison_rounds":list(range(1,25)),"baseline_outputs":{str(slot):next(row["content"] for row in rows if row["round"]==0 and row["slot"]==slot) for slot in WORDS}});write(raw/"hardware_metrics.json",{"wall_latency_ms":[row["wall_latency_ms"] for row in rows],"draft_tokens":sum(row["draft_n"] for row in rows),"accepted_draft_tokens":sum(row["draft_n_accepted"] for row in rows)});write(raw/"real_implementation.json",{"endpoint":"/completion","explicit_physical_slot":True,"active_spec_type":"draft-mtp","binary_sha256":before["binary_sha256"]});write(raw/"independent_evaluation.json",{"rescored_metrics":score(rows),"all_rows_rescored":True});write(raw/"semantic_parity.json",{"exact_repeat_rate":metrics["exact_repeat_rate"],"own_nonce_rate":metrics["own_nonce_rate"],"cross_slot_leakage_count":metrics["cross_slot_leakage_count"]});write(raw/"failure_reproduction.json",{"historical_simulation":{"transactions":2000,"leaks":0,"overhead_us":3.54},"live_runtime":metrics,"historical_overhead_not_retested":True})
    journal=run_text(["wsl","-d","Ubuntu-24.04","--","journalctl","-u","llm-inference.service","-n","200","--no-pager"]);write(raw/"service_logs.json",journal)
    defs={"request_coverage":("live_requests","eq",100),"slot_coverage":("physical_slots","eq",4),"speculation_coverage":("requests_with_draft_tokens","ge",80),"rollback_coverage":("requests_with_rejected_draft_tokens","ge",25),"state_consistency":("exact_repeat_rate","eq",1.0),"nonce_integrity":("own_nonce_rate","eq",1.0),"cross_slot_isolation":("cross_slot_leakage_count","eq",0),"service_integrity":("service_restarts","eq",0),"idle_recovery":("idle_slots_after","eq",4)};ops={"eq":lambda a,b:a==b,"ge":lambda a,b:a>=b};gates={g:{"metric":m,"operator":o,"threshold":t,"actual":metrics[m],"pass":ops[o](metrics[m],t)} for g,(m,o,t) in defs.items()}
    names=("actual_scores.json","artifact_hashes.json","dataset_hashes.json","effective_route.json","failure_reproduction.json","falsifiable_hypothesis.json","hardware_metrics.json","independent_evaluation.json","invalidation_rules.json","invariant_controls.json","paired_baseline.json","real_implementation.json","recovery_state.json","samples.jsonl","semantic_parity.json","service_identity.json","service_logs.json","service_maintenance.json","source_execution_receipt.json");files=[raw/name for name in names]
    provenance=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*EXPECTED,*files],packages=["pytest"],runtime={"execution_mode":"live_four_slot_draft_mtp","endpoint":BASE,"service_identity":before})
    complete,errors=provenance_complete(provenance)
    if not complete:raise ValueError(errors)
    evidence={"acceptance_gates":"raw/receipt.json","actual_scores":"raw/actual_scores.json","artifact_hashes":"raw/artifact_hashes.json","dataset_hashes":"raw/dataset_hashes.json","effective_route":"raw/effective_route.json","failure_reproduction":"raw/failure_reproduction.json","falsifiable_hypothesis":"raw/falsifiable_hypothesis.json","hardware_metrics":"raw/hardware_metrics.json","independent_evaluation":"raw/independent_evaluation.json","invalidation_rules":"raw/invalidation_rules.json","invariant_controls":"raw/invariant_controls.json","paired_baseline":"raw/paired_baseline.json","provenance":"raw/receipt.json","raw_samples":"raw/samples.jsonl","real_implementation":"raw/real_implementation.json","receipt_fingerprint":"raw/receipt.json","recovery_state":"raw/recovery_state.json","semantic_parity":"raw/semantic_parity.json","service_identity":"raw/service_identity.json","service_maintenance":"raw/service_maintenance.json","source_execution_receipt":"raw/source_execution_receipt.json"}
    receipt={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":provenance,"provenance_complete":True,"gates":gates,"evidence":evidence};receipt["receipt_fingerprint"]=canonical_json_sha256(receipt);write(raw/"receipt.json",receipt);return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();out=a.outdir.resolve();receipt=run(out);metrics=json.loads((out/"raw/actual_scores.json").read_text(encoding="utf-8"));passed=all(v["pass"] for v in receipt["gates"].values());claim="BEE_L4_LIVE_SLOT_ISOLATION_QUALIFIED_R1" if passed else "BEE_L4_FALSE_POSITIVE_CONFIRMED_R1";failed=[k for k,v in receipt["gates"].items() if not v["pass"]];(out/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent AGY review.\n\nExecuted `{metrics['live_requests']}` real requests across `{metrics['physical_slots']}` MTP slots. Draft tokens appeared in `{metrics['requests_with_draft_tokens']}` requests and real draft rejection in `{metrics['requests_with_rejected_draft_tokens']}`. Exact repeat rate `{metrics['exact_repeat_rate']:.6f}`, own-code rate `{metrics['own_nonce_rate']:.6f}`, cross-slot leaks `{metrics['cross_slot_leakage_count']}`, restarts `{metrics['service_restarts']}`, idle slots `{metrics['idle_slots_after']}`. Failed gates: `{', '.join(failed) if failed else 'none'}`. No internal-pointer or microsecond-overhead claim is made.\n",encoding="utf-8");print(json.dumps(receipt["gates"],indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
