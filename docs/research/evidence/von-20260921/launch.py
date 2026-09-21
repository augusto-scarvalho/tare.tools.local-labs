import json, os, subprocess, time, shutil
from pathlib import Path
import psutil
root=Path(__file__).parent
if psutil.virtual_memory().available < 6*2**30 or shutil.disk_usage('C:/').free < 12*2**30:
    raise RuntimeError('Insufficient CPU experiment headroom')
cmd=['C:/Users/augus/AppData/Local/tare.tools/assessment-venv/Scripts/python.exe','-B',
'C:/projects/tare.tools.local-labs/tools/analysis/qualify_von.py','--kernel-root','C:/projects/tare.tools.kernel',
'--manifest','C:/Users/augus/AppData/Local/tare.tools/assessment-models/von-aa2fdc96/checkpoint.json',
'--protocol','C:/projects/tare.tools.local-labs/docs/research/evidence/von-20260921/PROTOCOL.json',
'--output',str(root/'RESULT.json')]
env=dict(os.environ,USE_TF='0',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
start=time.monotonic();peak=0;sampled=0;free=psutil.virtual_memory().available;reason=None
with (root/'process.log').open('x',encoding='utf-8') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    while p.poll() is None:
        try:
            m=psutil.Process(p.pid).memory_info();peak=max(peak,getattr(m,'peak_wset',m.rss));sampled=max(sampled,m.rss)
            free=min(free,psutil.virtual_memory().available)
            if time.monotonic()-start>300: reason='WALL_TIMEOUT'
            if free<2*2**30: reason='RAM_FLOOR'
            if reason: p.kill();break
        except psutil.NoSuchProcess: break
        time.sleep(.1)
    code=p.wait(timeout=10)
receipt={'command':cmd,'cpu_only':True,'elapsed_seconds':time.monotonic()-start,
'exit_code':code,'reason':reason,'peak_working_set_bytes':peak,'sampled_peak_rss_bytes':sampled,
'min_available_ram_bytes':free,'process_exited':p.poll() is not None,'host':os.environ.get('COMPUTERNAME')}
(root/'RESOURCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
print((root/'process.log').read_text())
