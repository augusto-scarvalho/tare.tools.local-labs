import json,sys,time,subprocess,pathlib,hashlib
from urllib.request import Request,urlopen
root=pathlib.Path('/home/augus/.local/share/tare-assessments/20260921-guarded-v2')
sys.path.insert(0,str(root/'labs_src'))
from model_lifecycle.gpu_lease import SharedGpuLease

def http(port,path,data=None):
 with urlopen(Request(f'http://127.0.0.1:{port}{path}',data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=45) as r:return json.load(r)

def queue_idle():
 q=http(8188,'/queue'); assert not q['queue_running'] and not q['queue_pending'];return q

def call(request):
 start=time.monotonic();r=subprocess.run([sys.executable,'-B',str(root/'decision_worker.py'),str(root/'config.json')],input=json.dumps(request),text=True,capture_output=True,timeout=55)
 return {'seconds':time.monotonic()-start,'returncode':r.returncode,'answer':json.loads(r.stdout),'stderr':r.stderr[-3000:]}

for name,sha in json.loads((root/'MANIFEST.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
out=root/'CANARY.json'
with out.open('x') as f:f.write('{}')
r={'schema':'tare.local-labs/decision-worker-canary/1','qualification':'UNQUALIFIED','authority':'NONE'}
lease=SharedGpuLease('/home/augus/.local/state/tare-qualified-models/gpu.lock',timeout=2)
request=json.loads(pathlib.Path('/home/augus/experiments/openjev-20260921/PROTOCOL.json').read_text())['cases'][0]['request']
initial=http(8080,'/health');r['before']=initial;assert initial['current_model'] is None and not initial['gpu_coordination']['held'];queue_idle()
r['services_before']=subprocess.check_output(['systemctl','show','llm-inference.service','llm-embedding.service','comfyui.service','-p','Id','-p','MainPID','-p','NRestarts','-p','ActiveState'],text=True)
try:
 r['profile']=http(8080,'/v1/fleet/profile',{'model':'qwen38'})
 r['gpu']=call(request);print(json.dumps({'gpu':r['gpu']['answer'],'seconds':r['gpu']['seconds']}),flush=True)
 queue_idle()
 with lease.hold('image','decision-fallback-canary') as proof:
  http(8080,'/internal/gpu/yield',{'nonce':proof['nonce']})
  r['cpu']=call(request)
  r['during_cpu']=http(8080,'/health')
  assert r['during_cpu']['current_model'] is None
  assert r['during_cpu']['gpu_coordination']['held']
  print(json.dumps({'cpu':r['cpu']['answer'],'seconds':r['cpu']['seconds']}),flush=True)
 r['status']='PASS'
except Exception as exc:r.update(status='FAILED',error=repr(exc))
finally:
 h=http(8080,'/health')
 if h['current_model']=='qwen38':
  with lease.hold('image','decision-canary-recovery') as proof:http(8080,'/internal/gpu/yield',{'nonce':proof['nonce']})
 r['after']=http(8080,'/health');r['queue_after']=queue_idle()
 r['services_after']=subprocess.check_output(['systemctl','show','llm-inference.service','llm-embedding.service','comfyui.service','-p','Id','-p','MainPID','-p','NRestarts','-p','ActiveState'],text=True)
 r['gpu_final']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader'],text=True)
 out.write_text(json.dumps(r,indent=2))
print(r['status'],flush=True)
