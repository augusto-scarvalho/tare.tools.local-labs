import json, os, subprocess, time
from pathlib import Path
import psutil
root=Path(__file__).parent
protocol=json.loads(Path('C:/projects/tare.tools.local-labs/docs/research/evidence/von-20260921/PROTOCOL.json').read_bytes())
request=next(c['request'] for c in protocol['cases'] if c['id']=='english-mechanical')
cmd=['C:/Users/augus/AppData/Local/tare.tools/assessment-venv/Scripts/python.exe','-B','-m','compute_plane.von_choice_worker','C:/Users/augus/AppData/Local/tare.tools/assessment-models/von-aa2fdc96/checkpoint.json']
env=dict(os.environ,PYTHONPATH='C:/projects/tare.tools.kernel',CUDA_VISIBLE_DEVICES='',USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
start=time.monotonic();seen={};peak=0;reason=None
if psutil.virtual_memory().available<6*2**30: raise RuntimeError('RAM headroom')
with (root/'canary.stdout.json').open('x') as out,(root/'canary.stderr.log').open('x') as err:
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=out,stderr=err,env=env)
    p.stdin.write(json.dumps(request).encode());p.stdin.close()
    while p.poll() is None:
        processes=[psutil.Process(p.pid)]
        try: processes+=processes[0].children(recursive=True)
        except psutil.NoSuchProcess: pass
        current=0
        for process in processes:
            try:
                m=process.memory_info();current+=m.rss
                seen[process.pid]={'pid':process.pid,'created':process.create_time(),'peak_wset':max(seen.get(process.pid,{}).get('peak_wset',0),getattr(m,'peak_wset',m.rss))}
            except psutil.NoSuchProcess: pass
        peak=max(peak,current)
        if time.monotonic()-start>60:
            reason='WALL_TIMEOUT'
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
        if process.create_time()==row['created'] and process.is_running(): remaining.append(pid)
    except psutil.NoSuchProcess: pass
answer=json.loads((root/'canary.stdout.json').read_bytes())
import sys
sys.path.insert(0,'C:/projects/tare.tools.kernel')
from compute_plane.choice_assessment import validate_answer
validate_answer(answer,request)
receipt={'scope':'Single repeated transport/resource canary; excluded from quality counts','exit_code':code,'reason':reason,'elapsed_seconds':time.monotonic()-start,'processes':list(seen.values()),'sampled_tree_peak_rss_bytes':peak,'unreaped_processes':remaining,'answer':answer,'request':request}
(root/'RESOURCE_CANARY.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
