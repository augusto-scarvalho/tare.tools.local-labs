import json, os, subprocess, time, sys
from pathlib import Path
import psutil
root=Path(__file__).parent
backend=sys.argv[1]
cmd=['C:/Users/augus/AppData/Local/tare.tools/assessment-venv/Scripts/python.exe','-B',
'C:/projects/tare.tools.local-labs/tools/analysis/audit_von_contract.py','--backend',backend,
'--upstream-root',str(root/'upstream'),'--checkpoint','C:/Users/augus/AppData/Local/tare.tools/assessment-models/von-aa2fdc96',
'--protocol',str(root/'PROTOCOL.json'),'--output',str(root/(backend+'-RESULT.json'))]
env=dict(os.environ,PYTHONPATH='C:/Users/augus/AppData/Local/tare.tools/von-reference-deps',CUDA_VISIBLE_DEVICES='',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
if psutil.virtual_memory().available<6*2**30: raise RuntimeError('RAM headroom')
started=time.monotonic();seen={};peak=0;reason=None
with (root/(backend+'-process.log')).open('x',encoding='utf-8') as log:
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    while p.poll() is None:
        try: processes=[psutil.Process(p.pid)]+psutil.Process(p.pid).children(recursive=True)
        except psutil.NoSuchProcess: processes=[]
        current=0
        for process in processes:
            try:
                m=process.memory_info();current+=m.rss
                seen[process.pid]={'created':process.create_time(),'peak_wset':max(seen.get(process.pid,{}).get('peak_wset',0),getattr(m,'peak_wset',m.rss))}
            except psutil.NoSuchProcess: pass
        peak=max(peak,current)
        if time.monotonic()-started>600: reason='WALL_TIMEOUT'
        if psutil.virtual_memory().available<2*2**30: reason='RAM_FLOOR'
        if reason:
            for process in reversed(processes):
                try: process.kill()
                except psutil.NoSuchProcess: pass
            break
        time.sleep(.1)
    code=p.wait(timeout=10)
remaining=[]
for pid,row in seen.items():
    try:
        process=psutil.Process(pid)
        if process.create_time()==row['created'] and process.is_running():remaining.append(pid)
    except psutil.NoSuchProcess:pass
receipt={'backend':backend,'exit_code':code,'reason':reason,'wall_seconds':time.monotonic()-started,'sampled_tree_peak_rss_bytes':peak,'processes':seen,'unreaped_processes':remaining}
(root/(backend+'-RESOURCE.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
print((root/(backend+'-process.log')).read_text(encoding='utf-8')[-3500:])
