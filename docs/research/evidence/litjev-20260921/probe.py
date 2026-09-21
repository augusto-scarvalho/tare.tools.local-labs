import json, hashlib, math, pathlib, subprocess, time, threading, sys, types, importlib.util
from urllib.request import Request, urlopen
P=pathlib.Path(__file__).resolve().parent

def http(port,path,data=None):
    req=Request(f'http://127.0.0.1:{port}{path}',data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=120) as r:return json.load(r)

def idle():
    q=http(8188,'/queue'); h=http(8080,'/health')
    if q['queue_running'] or q['queue_pending'] or h['gpu_coordination']['held']:raise RuntimeError('GPU_OR_IMAGE_QUEUE_BUSY')
    if h['current_model'] not in (None,'qwen38'):raise RuntimeError('OTHER_MODEL_RESIDENT')
    return h

def token(text):return http(18080,'/tokenize',{'content':text,'add_special':False,'parse_special':True})['tokens']

def load(name):
    spec=importlib.util.spec_from_file_location(name,P/(name+'.py')); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

prompting=load('prompting'); scoring=load('scoring')
report={'schema':'tare.local-labs/litjev-slop-probe/1','status':'STARTED','authority':'NONE','qualification':'UNQUALIFIED','method':'single-question LitJEV prompt with native first-token pre-sampling logprobs','upstream_revision':'e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3','hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in P.iterdir() if x.is_file()},'cases':[],'memory':[]}
out=P/'RESULT.json'
with out.open('x') as f:json.dump(report,f)
def save():
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(out)
stop=threading.Event();start=time.monotonic()
def monitor():
    import psutil
    while not stop.is_set():
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv,noheader,nounits'],timeout=5,text=True)
            used,free=map(float,raw.splitlines()[0].split(','))
            report['memory'].append({'seconds':time.monotonic()-start,'used_mib':used,'free_mib':free,'ram_available_bytes':psutil.virtual_memory().available})
        except Exception as exc:report['monitor_error']=str(exc)
        stop.wait(.2)

def evaluate(req):
    idle()
    if time.monotonic()-start>550:raise TimeoutError('TOTAL_DEADLINE')
    keys=list(req['options']);codes=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:len(keys)]
    question=types.SimpleNamespace(type='choice',instructions=req['question'],choices=keys,descriptions=list(req['options'].values()))
    prefix=http(18080,'/apply-template',{'messages':prompting.build_decision_messages(req['state']),'add_generation_prompt':True,'chat_template_kwargs':{'enable_thinking':False}})
    prefix=prefix['prompt'];suffix=prompting.question_suffix(question,codes)
    base=token(suffix);candidates=[]
    for code in codes:
        ids=token(suffix+' '+code)
        if ids[:-1]!=base or len(ids)!=len(base)+1:raise ValueError('CANDIDATE_NOT_ONE_TOKEN')
        candidates.append(ids[-1])
    ids=token(prefix)+base
    if len(ids)>4096:raise ValueError('INPUT_TOO_LONG')
    payload={'model':'qwen38','prompt':ids,'n_predict':1,'n_probs':128,'post_sampling_probs':False,'temperature':1.0,'top_k':0,'top_p':1.0,'min_p':0.0,'repeat_penalty':1.0,'presence_penalty':0.0,'frequency_penalty':0.0,'cache_prompt':False,'return_tokens':True,'stream':False,'seed':219,'speculative.n_max':0}
    before=time.monotonic();raw=http(8080,'/completion',payload);elapsed=time.monotonic()-before
    result={'seconds':elapsed,'prefix':prefix,'suffix':suffix,'input_tokens':len(ids),'candidate_token_ids':candidates,'payload':payload,'raw':raw,'valid':False}
    rows=raw.get('completion_probabilities',raw.get('probs',[]))
    if raw.get('truncated'):result['error']='TRUNCATED';return result
    if len(rows)!=1:result['error']='EXPECTED_ONE_LOGPROB_ROW';return result
    scores={v['id']:v['logprob'] for v in rows[0].get('top_logprobs',[])}
    if not all(t in scores and math.isfinite(scores[t]) for t in candidates):result['error']='MISSING_CANDIDATE_LOGPROBS';return result
    dist=scoring.calibrated_distribution([scores[t] for t in candidates],list(range(len(keys))),1.0)
    result.update(valid=True,choice=keys[dist.winner_index],probabilities=dict(zip(keys,dist.probabilities)),candidate_logprobs=[scores[t] for t in candidates])
    return result

initial=None;owned_pid=None
try:
    initial=idle();report['initial']=initial
    report['services_before']=subprocess.check_output(['systemctl','show','llm-embedding.service','comfyui.service','-p','Id','-p','MainPID','-p','NRestarts','-p','ActiveState'],text=True)
    t=threading.Thread(target=monitor,daemon=True);t.start()
    before=time.monotonic();report['profile']=http(8080,'/v1/fleet/profile',{'model':'qwen38'});report['profile_seconds']=time.monotonic()-before
    h=http(8080,'/health');owned_pid=h['backend_pid'];report['loaded_health']=h
    report['control']=evaluate({'state':'The required literal is beta.','question':'Select the required literal.','options':{'alpha':'alpha','beta':'beta','gamma':'gamma'}});save()
    if not report['control']['valid']:raise RuntimeError('CAPABILITY_CONTROL_FAILED')
    for suite in ['PROTOCOL.json','FRESH_CASES.json']:
        for case in json.loads((P/suite).read_text())['cases']:
            result=evaluate(case['request']);result.update(id=case['id'],suite=suite,request=case['request'],expected_identity=case['expected_identity'])
            result['chosen_identity']=case['choice_identity'].get(result.get('choice'))
            result['correct']=result['valid'] and result['chosen_identity']==case['expected_identity']
            report['cases'].append(result);save();print(json.dumps({'case':case['id'],'correct':result['correct'],'seconds':result['seconds'],'choice':result.get('choice'),'error':result.get('error')}),flush=True)
    report['status']='MEASURED'
except Exception as exc:
    report.update(status='FAILED',error=repr(exc));print(repr(exc),flush=True)
finally:
    try:
        report['before_recovery']=http(8080,'/health')
        if initial and initial['current_model'] is None and owned_pid and report['before_recovery']['backend_pid']==owned_pid:
            runner='/home/augus/experiments/openjev-20260921/lab/tools/serving/gpu_lease_run.py'
            r=subprocess.run(['python3',runner,'--gpu-wait','5','--stop-grace','5','--','true'],capture_output=True,text=True,timeout=120)
            report['recovery']={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        report['final_health']=http(8080,'/health');report['final_queue']=http(8188,'/queue')
        report['services_after']=subprocess.check_output(['systemctl','show','llm-embedding.service','comfyui.service','-p','Id','-p','MainPID','-p','NRestarts','-p','ActiveState'],text=True)
    except Exception as exc:report['recovery_error']=repr(exc)
    stop.set()
    if 't' in globals():t.join(timeout=6)
    report['total_seconds']=time.monotonic()-start;save()
