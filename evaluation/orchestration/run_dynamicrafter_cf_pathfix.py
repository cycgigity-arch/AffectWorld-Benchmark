#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/root/autodl-tmp')
BASE = ROOT / 'ewm_eval_local_cf223'
ADAPTER = BASE / 'ewm_eval_local.py'
CANONICAL = BASE / 'private_annotations_1498_normalized.json'
ORIG = ROOT / 'ewm_generated_dynamicrafter' / 'normal'
CF = ROOT / 'ewm_generated_dynamicrafter' / 'counterfactual'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, obj):
    p=Path(p); tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,p)

def launch(args):
    import start_full_new_models_eval as bundle
    mismatches=bundle.verify_evaluator_code(BASE)
    if mismatches:
        raise SystemExit('Evaluator verification failed: '+ '; '.join(mismatches))
    canonical=read(CANONICAL)
    by_key={r['video_key']:r for r in canonical}
    previous=read(args.previous/'launch_manifest.json')
    prev_cf={}
    for shard in previous['shards']:
        stage=shard['stages'].get('cf')
        if stage: prev_cf[shard['index']]=stage
    if len(prev_cf)!=8:
        raise SystemExit(f'Expected 8 previous CF shards, got {len(prev_cf)}')

    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run=ROOT/'new_model_eval_runs'/f'dynamicrafter_cf_pathfix_{tag}'
    run.mkdir(parents=True,exist_ok=False)
    shards=[]
    for i in range(8):
        stage=prev_cf[i]
        old_rows=read(stage['annotations'])
        if not old_rows:
            raise SystemExit(f'Empty CF shard {i}')
        rows=[]
        for original in old_rows:
            key=original['video_key']
            row=json.loads(json.dumps(by_key[key]))
            parts=Path(key).as_posix().split('/')
            branch=row.get('counterfactual')
            if not isinstance(branch,dict):
                branch=(row.get('ewm_unit') or {}).get('counterfactual')
            if len(parts)<2 or not isinstance(branch,dict):
                raise SystemExit(f'Bad CF annotation: {key}')
            rel=f'{parts[1]}/{hashlib.sha256(key.encode()).hexdigest()[:16]}.mp4'
            if not (CF/rel).is_file():
                raise SystemExit(f'Hashed CF file missing: {CF/rel}')
            branch['video_key']=rel
            rows.append(row)
        ann=run/f'cf_annotations_shard{i}.json'
        ann.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        pre=run/f'cf_preflight_shard{i}'
        cmd=[sys.executable,str(ADAPTER),'check','--annotations',str(ann),
             '--orig-dir',str(ORIG),'--cf-dir',str(CF),'--tasks','cf',
             '--limit','0','--output-dir',str(pre)]
        res=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        if res.returncode:
            raise SystemExit(f'Preflight failed shard{i}:\n{res.stdout}\n{res.stderr}')
        report=read(pre/'preflight.json')
        if report['errors'] or report['selected']!=len(rows) or report['ready']!=len(rows) or report['cf_selected']!=len(rows):
            raise SystemExit(f'Preflight mismatch shard{i}: {report}')
        out=run/f'cf_shard{i}'
        shards.append({'index':i,'annotations':str(ann),'preflight':str(pre),
                       'output':str(out),'expected':len(rows),
                       'previous_cache':str(Path(stage['output'])/'.ewh_cache'),
                       'log':str(run/f'worker{i}.log')})
    total=sum(s['expected'] for s in shards)
    if total!=223: raise SystemExit(f'Expected 223 CF samples, got {total}')
    manifest={'created_utc':tag,'state':'running','model':'dynamicrafter',
              'purpose':'retry CF path lookup failures only','previous_run':str(args.previous),
              'workers':8,'selected_cf':total,'shards':shards}
    mp=run/'launch_manifest.json'; write(mp,manifest)
    vlm=getpass.getpass('VLM API key (hidden): ').strip()
    llm=getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm or not llm: raise SystemExit('Both API keys are required')
    env=os.environ.copy(); env.update(EWM_VLM_API_KEY=vlm,EWM_LLM_API_KEY=llm)
    del vlm,llm
    for s in shards:
        cfg=run/f'worker{s["index"]}.json'
        write(cfg,{'root':str(ROOT),'base':str(BASE),'adapter':str(ADAPTER),
                   'orig':str(ORIG),'cf':str(CF),'stage':s})
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--config',str(cfg)]
        with open(s['log'],'w',encoding='utf-8') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        s['pid']=p.pid; write(mp,manifest)
        print(f'CF shard{s["index"]}: PID {p.pid}; log {s["log"]}',flush=True)
    del env
    print(f'CF_PATHFIX_STARTED; 223 samples; 8 workers; run={run}',flush=True)

def worker(config):
    c=read(config); s=c['stage']
    sys.path.insert(0,c['base'])
    import ewm_eval_local as adapter
    from types import SimpleNamespace
    args=SimpleNamespace(annotations=Path(s['annotations']),orig_dir=Path(c['orig']),
        cf_dir=Path(c['cf']),tasks=['cf'],limit=0,label_overrides=None,
        output_dir=Path(s['output']))
    work,report=adapter.prepare(args)
    if report['errors'] or report['cf_selected']!=s['expected']:
        raise SystemExit(f'Worker preflight mismatch: {report}')
    old_cache=Path(s['previous_cache'])
    new_cache=work/'.ewh_cache'
    if old_cache.is_dir():
        shutil.copytree(old_cache,new_cache,dirs_exist_ok=True)
        print(f'REUSED_CACHE {old_cache}',flush=True)
    print(f'STAGE_START cf expected={s["expected"]}',flush=True)
    adapter.run(args,work)
    print('STAGE_DONE cf',flush=True)

def status(args):
    m=read(args.target/'launch_manifest.json')
    for s in m['shards']:
        pid=s.get('pid')
        alive=bool(pid and Path(f'/proc/{pid}/stat').is_file() and Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z')
        p=Path(s['output'])/'cf_results.json'
        rows=read(p) if p.is_file() else []
        errs=sum('error' in r for r in rows)
        print(f'shard{s["index"]} pid={pid} alive={alive} success={len(rows)-errs}/{s["expected"]} errors={errs} pending={s["expected"]-len(rows)}')
    print('run='+str(args.target))

def main():
    p=argparse.ArgumentParser()
    sp=p.add_subparsers(dest='cmd',required=True)
    lp=sp.add_parser('launch'); lp.add_argument('--previous',type=Path,required=True)
    st=sp.add_parser('status'); st.add_argument('--target',type=Path,required=True)
    wp=sp.add_parser('worker'); wp.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='launch': launch(a)
    elif a.cmd=='status': status(a)
    else: worker(a.config)
if __name__=='__main__': main()
