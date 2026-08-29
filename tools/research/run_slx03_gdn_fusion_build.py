#!/usr/bin/env python3
"""Build-only qualification of the immutable SLX-03 GDN fusion source."""
from __future__ import annotations
import argparse,json,pathlib,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.analysis.experiment_provenance import build_provenance,canonical_json_sha256,provenance_complete,sha256_file
TASK_ID="BACKLOG-SLX03-GDN-FUSION-BUILD-01";DISTRO="Ubuntu-24.04";SOURCE="/home/augus/src/slop.cpp-main";BUILD="/home/augus/src/slop.cpp-main/build-slx03-gdn-audit-01";COMMIT="87a416bd75d5a64e66e55846b779c0a54eca21bd"
INPUTS={"config/research_backlog_admissions/BACKLOG-SLX03-GDN-FUSION-BUILD-01.json":"c68da972121b67c35cb1510da6289f0008f9b3e224d64d43fefbbc21cf123e8a","runs/research/BACKLOG-SLX03-GDN-FUSION-BUILD-01/PRE_REGISTRATION.md":"e54dbe47561bd33c9013707b78db880dec342d0db8ff035cc570599abb438a16","runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/raw/receipt.json":"5ebe76094b02ed4557533fa6def8b64241a3b1ad4fc0124c7d10359df6e8589e","runs/research/BACKLOG-AGY-SYSTEM-BLOCKERS-03/REVIEW.json":"c865e2f08d9bf36f7366b8ed256999babc1356c97067f6f9ac42d1cfdbabdea8"}
def wj(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")
def cmd(*args,timeout=1800):
 p=subprocess.run(["wsl.exe","-d",DISTRO,*args],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout);return {"argv":["wsl.exe","-d",DISTRO,*args],"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
def at_source(*args,timeout=1800):return cmd("--cd",SOURCE,*args,timeout=timeout)
def health(url):
 try:
  with urllib.request.urlopen(url,timeout=10) as r:return r.status
 except Exception as e:return f"{type(e).__name__}:{e}"
def service():
 r=cmd("systemctl","show","llm-inference.service","-p","MainPID","-p","NRestarts","-p","ActiveState","-p","SubState","--no-pager",timeout=30);d={}
 for line in r["stdout"].splitlines():
  if "=" in line:k,v=line.split("=",1);d[k]=v
 d["returncode"]=r["returncode"];return d
def run(out):
 raw=out/"raw";started=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime());mono=time.monotonic();paths=[]
 for rel,exp in INPUTS.items():
  p=ROOT/rel;act=sha256_file(p)
  if act!=exp:raise ValueError(f"frozen source mismatch {rel}")
  paths.append(p)
 if cmd("test","!","-e",BUILD,timeout=30)["returncode"]!=0:raise RuntimeError(f"build directory already exists: {BUILD}")
 head=at_source("git","rev-parse","HEAD",timeout=30);clean=at_source("git","status","--porcelain","--untracked-files=no",timeout=30)
 exact=head["returncode"]==0 and head["stdout"].strip()==COMMIT;tracked=clean["returncode"]==0 and not clean["stdout"].strip()
 if not exact or not tracked:raise RuntimeError(f"source identity failed: {head} {clean}")
 before={"service":service(),"gateway":health("http://127.0.0.1:8080/health"),"embedding":health("http://127.0.0.1:8081/health")};wj(raw/"service_before.json",before)
 configure=at_source("cmake","-S",".","-B",BUILD.split("/")[-1],"-DCMAKE_BUILD_TYPE=Release","-DGGML_CUDA=ON","-DGGML_NATIVE=OFF","-DCMAKE_CUDA_ARCHITECTURES=86",timeout=1800)
 (raw/"configure.stdout.log").write_text(configure["stdout"],encoding="utf-8",newline="\n");(raw/"configure.stderr.log").write_text(configure["stderr"],encoding="utf-8",newline="\n")
 build={"returncode":-1,"stdout":"","stderr":"configure failed"}
 if configure["returncode"]==0:build=at_source("cmake","--build",BUILD.split("/")[-1],"--target","llama-server","-j","8",timeout=7200)
 (raw/"build.stdout.log").write_text(build["stdout"],encoding="utf-8",newline="\n");(raw/"build.stderr.log").write_text(build["stderr"],encoding="utf-8",newline="\n")
 binary=f"{BUILD}/bin/llama-server";lib=f"{BUILD}/bin/libggml-cuda.so"
 marker=cmd("strings",lib,timeout=120);fusion="fused gated_delta_net snapshot copies" in marker["stdout"]
 linkage=cmd("env",f"LD_LIBRARY_PATH={BUILD}/bin","ldd",binary,timeout=120);project=[]
 for line in linkage["stdout"].splitlines():
  if any(name in line for name in ("libllama","libggml","libmtmd","libllama-common")):project.append(line.strip())
 own=bool(project) and all((f"=> {BUILD}/bin/" in line) or line.startswith(f"{BUILD}/bin/") for line in project)
 version=cmd("env",f"LD_LIBRARY_PATH={BUILD}/bin",binary,"--version",timeout=120)
 hashes={}
 for name,path in (("llama_server",binary),("libggml_cuda",lib)):
  h=cmd("sha256sum",path,timeout=120);s=cmd("stat","-c","%s",path,timeout=120);hashes[name]={"path":path,"sha256":h["stdout"].split()[0] if h["returncode"]==0 else None,"bytes":int(s["stdout"].strip()) if s["returncode"]==0 else None}
 after={"service":service(),"gateway":health("http://127.0.0.1:8080/health"),"embedding":health("http://127.0.0.1:8081/health")};same=before==after
 metrics={"exact_source_commit":exact,"tracked_source_clean":tracked,"cmake_configure_exit":configure["returncode"],"llama_server_build_exit":build["returncode"],"gdn_fusion_marker_present":fusion,"project_libraries_resolve_to_new_build":own,"server_version_exit":version["returncode"],"gateway_and_embedding_unchanged":same}
 wj(raw/"actual_scores.json",metrics);wj(raw/"source_revision.json",{"repository":SOURCE,"commit":head["stdout"].strip(),"expected":COMMIT,"tracked_status":clean["stdout"]});wj(raw/"dependency_hashes.json",{"host_inputs":INPUTS});wj(raw/"build_receipts.json",{"configure":configure["returncode"],"build":build["returncode"],"command_configure":configure["argv"],"command_build":build["argv"]});wj(raw/"correctness_receipts.json",{"fusion_marker":fusion,"linkage":linkage,"project_lines":project,"server_version":version});wj(raw/"end_to_end_artifact.json",hashes);wj(raw/"independent_evaluation.json",{"build_only":True,"runtime_fusion_not_observed":True,"metrics":metrics});wj(raw/"service_after.json",after)
 defs={"source_revision":("exact_source_commit","eq",True),"tracked_clean":("tracked_source_clean","eq",True),"configure":("cmake_configure_exit","eq",0),"build":("llama_server_build_exit","eq",0),"fusion_marker":("gdn_fusion_marker_present","eq",True),"self_linkage":("project_libraries_resolve_to_new_build","eq",True),"callability":("server_version_exit","eq",0),"service_invariance":("gateway_and_embedding_unchanged","eq",True)};g={}
 for n,(k,o,x) in defs.items():v=metrics[k];g[n]={"metric":k,"operator":o,"threshold":x,"actual":v,"pass":v==x}
 ev={"acceptance_gates":"raw/receipt.json","build_receipts":"raw/build_receipts.json","correctness_receipts":"raw/correctness_receipts.json","dependency_hashes":"raw/dependency_hashes.json","end_to_end_artifact":"raw/end_to_end_artifact.json","independent_evaluation":"raw/independent_evaluation.json","provenance":"raw/receipt.json","raw_samples":"raw/build.stdout.log","receipt_fingerprint":"raw/receipt.json","source_revision":"raw/source_revision.json"};e=sorted(p for p in raw.rglob("*") if p.is_file());prov=build_provenance(script_path=pathlib.Path(__file__).resolve(),started_at_utc=started,started_monotonic=mono,input_paths=[*paths,*e],packages=[],runtime={"execution_mode":"isolated_cuda_build","wsl_source":SOURCE,"wsl_build":BUILD,"gpu_model_not_loaded":True});ok,err=provenance_complete(prov)
 if not ok:raise RuntimeError(err)
 rec={"schema":"local-labs-backlog-receipt-v1","task_id":TASK_ID,"provenance":prov,"provenance_complete":True,"gates":g,"evidence":ev};rec["receipt_fingerprint"]=canonical_json_sha256(rec);wj(raw/"receipt.json",rec);failed=[n for n,v in g.items() if not v["pass"]];claim="SLX03_GDN_FUSION_BUILD_CALLABLE_R1" if not failed else "SLX03_GDN_FUSION_BUILD_NOT_CONFIRMED_R1";(out/"RESULT.md").write_text(f"# {TASK_ID} result\n\n`{claim}` pending independent review.\n\nCommit `{COMMIT}`; configure/build/version `{configure['returncode']}/{build['returncode']}/{version['returncode']}`; fusion marker `{fusion}`; own linkage `{own}`; services unchanged `{same}`. Failed gates: `{', '.join(failed) if failed else 'none'}`.\n",encoding="utf-8",newline="\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--outdir",type=pathlib.Path,default=ROOT/"runs/research"/TASK_ID);a=p.parse_args();run(a.outdir.resolve());return 0
if __name__=="__main__":raise SystemExit(main())
