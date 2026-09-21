import base64,hashlib,json,os,pathlib,shutil,subprocess,sys,time
from urllib.request import urlopen
bundle=pathlib.Path('/home/augus/.local/share/tare-assessments/20260921-guarded-v2')
sys.path.insert(0,str(bundle/'labs_src'))
from model_lifecycle.gpu_lease import SharedGpuLease
old=pathlib.Path('/home/augus/.local/share/tare-qualified-models/releases/2d718355cc48e552e549')
raw=(bundle/'qualified_model_gateway.py').read_bytes();sha=hashlib.sha256(raw).hexdigest()
new=old.parent/('resident-'+sha[:20]);unit=pathlib.Path('/etc/systemd/system/llm-inference.service.d/zzz-shared-gpu.conf')
previous=unit.read_bytes()
assert str(old).encode() in previous and not new.exists()
manifest=json.loads((old/'release_manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest,name
shutil.copytree(old,new)
(new/'tools/serving/qualified_model_gateway.py').write_bytes(raw)
manifest['files']['tools/serving/qualified_model_gateway.py']=sha
manifest['parent_release']=str(old);manifest['change']='resident PID-bound decision readout'
(new/'release_manifest.json').write_text(json.dumps(manifest,indent=2))
r={'old_release':str(old),'new_release':str(new),'gateway_sha256':sha,'previous_override_base64':base64.b64encode(previous).decode(),'status':'PREPARED'}
out=bundle/'SERVICE_UPDATE.json'
def save():out.write_text(json.dumps(r,indent=2))
def http(port,path):
 with urlopen(f'http://127.0.0.1:{port}{path}',timeout=3) as f:return json.load(f)
def ctl(*args):return subprocess.check_output(['systemctl',*args],text=True,timeout=30)
def states():return ctl('show','llm-inference.service','llm-embedding.service','comfyui.service','-p','Id','-p','MainPID','-p','NRestarts','-p','ActiveState')
def wait_ready():
 end=time.monotonic()+15
 while time.monotonic()<end:
  try:
   h=http(8080,'/health')
   if h.get('resident_readout')=='pid-bound-v1':return h
  except Exception:pass
  time.sleep(.2)
 raise RuntimeError('NEW_GATEWAY_NOT_READY')
r['services_before']=states();save()
lease=SharedGpuLease('/home/augus/.local/state/tare-qualified-models/gpu.lock',timeout=2)
with lease.hold('text','resident-gateway-update'):
 q=http(8188,'/queue');h=http(8080,'/health')
 assert q['queue_running']==q['queue_pending']==[] and h['current_model'] is None,'BUSY_REFUSED'
 try:
  temp=unit.with_suffix('.tmp');temp.write_bytes(previous.replace(str(old).encode(),str(new).encode()));os.replace(temp,unit)
  ctl('daemon-reload');ctl('restart','llm-inference.service')
  r['health_after']=wait_ready();r['status']='INSTALLED'
 except BaseException as exc:
  unit.write_bytes(previous);ctl('daemon-reload');ctl('restart','llm-inference.service')
  r.update(status='ROLLED_BACK',error=repr(exc));save();raise
 r['services_after']=states();save()
print(json.dumps(r,indent=2))
