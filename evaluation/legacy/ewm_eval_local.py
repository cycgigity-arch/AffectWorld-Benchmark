"""Offline preflight and local-video adapter for the supplied Task2/3/CF scorers.

Task2/3 scoring is unchanged; CF uses state0_emotion_family_v1. Annotation
flattening, explicit Task2 label normalization and local file mapping are saved
for inspection before API use.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import types

BUNDLE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def offline_parsers(bundle):
    """Run only the pure parsers/taxonomies, without imports, loggers or clients."""
    def select(name, functions, variables, scope):
        tree = ast.parse((bundle/name).read_text(encoding='utf-8-sig'))
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                names = {child.id for target in node.targets for child in ast.walk(target)
                         if isinstance(child, ast.Name)}
                if names.intersection(variables):
                    nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), scope)
        return types.SimpleNamespace(**{k:v for k,v in scope.items() if k!='__builtins__'})
    t2 = select('eval_task2.py', {'norm_key','_build','_normalize','normalize_emotion',
                                'normalize_action','entity_lines','parse_sample'},
                {'EMOTION_TAXONOMY','EMOTION_FAMILIES','ACTION_TAXONOMY','EMOTION_DISPLAY',
                 'EMOTION_GROUP','ACTION_DISPLAY','ACTION_GROUP','EMOTION_ALIAS','ACTION_ALIAS'},
                {'re':re})
    t3 = select('eval_task3.py', {'parse_constraints'},
                {'CONSTRAINT_TYPES','ID_PREFIX','NEGATIVE_PHRASE_RE'}, {'re':re,'t2':t2})
    cf = select('eval_cf.py', {'normalize_list','pick','state0_emotion_family',
                               'parse_branches'},set(),{'t2':t2,'t3':t3})
    return t2,t3,cf


def flatten(row):
    result = copy.deepcopy(row)
    unit = result.pop('ewm_unit', None)
    if unit is not None:
        if not isinstance(unit,dict):
            raise ValueError('ewm_unit must be an object')
        for key,value in unit.items():
            if key in result and result[key]!=value:
                raise ValueError(f'Conflicting top-level and ewm_unit fields: {key}')
            result[key]=value
    return result


def apply_reference_mapping(row, t2, override=None):
    changes=[]
    for section in ('orig','cf'):
        if section=='orig':
            ref_name='reference_state1' if row.get('reference_state1') else 'state1'
            ref=row.get(ref_name) or {}
            old=ref.get('emotion')
            supplied=(override or {}).get('orig_emotion')
            if supplied is not None:
                value=t2.normalize_emotion(supplied)
                if value is None:raise ValueError(f'Unknown orig_emotion override: {supplied}')
                row['reference_state1']=dict(row.get('reference_state1') or {})
                row['reference_state1']['emotion']=value
                changes.append({'field':'reference_state1.emotion','old':old,'new':value,'method':'explicit_override'})
            elif old and t2.normalize_emotion(old) is None:
                # Only use mappings actually recorded for this same label in this row.
                matches={t2.normalize_emotion(c.get('normalized'))
                         for c in row.get('normalization',{}).get('changes',[])
                         if c.get('original')==old and 'plausible_emotions[' in c.get('field','')}
                matches.discard(None)
                if len(matches)==1:
                    value=matches.pop()
                    row[ref_name]['emotion']=value
                    changes.append({'field':ref_name+'.emotion','old':old,'new':value,'method':'recorded_same_label_mapping'})
        elif isinstance(row.get('counterfactual'),dict):
            supplied=(override or {}).get('cf_emotion')
            if supplied is not None:
                value=t2.normalize_emotion(supplied)
                if value is None:raise ValueError(f'Unknown cf_emotion override: {supplied}')
                # parse_branches gives cf_emotion precedence over emotion_cf.
                old=row['counterfactual'].get('cf_emotion',row['counterfactual'].get('emotion_cf'))
                row['counterfactual']['cf_emotion']=value
                changes.append({'field':'counterfactual.cf_emotion','old':old,'new':value,'method':'explicit_override'})
    return changes


def resolve_video(root, key, branch=None):
    key=key.replace('\\','/')
    parts=PurePosixPath(key).parts
    if not parts or '..' in parts or PurePosixPath(key).is_absolute():
        raise ValueError(f'Unsafe video_key: {key}')
    candidates=[Path(*parts),Path(parts[-1])]
    if len(parts)>1:
        candidates.extend([Path(*parts[1:]),Path(parts[1])/parts[-1]])
    if len(parts)>2:candidates.append(Path(*parts[2:]))
    digest=hashlib.sha256(key.encode()).hexdigest()[:16]
    named=f'{PurePosixPath(key).stem[:90]}_{digest}_state0.mp4'
    candidates.extend([Path(named),Path(digest+'.mp4')])
    if len(parts)>1:candidates.extend([Path(parts[1])/named,Path(parts[1])/(digest+'.mp4')])
    explicit=(branch or {}).get('video_key')
    if explicit and explicit!=key:
        return resolve_video(root,explicit)
    root=Path(root).resolve()
    found={str((root/path).resolve()) for path in candidates
           if (root/path).is_file() and (root/path).resolve().is_relative_to(root)}
    if len(found)>1:raise ValueError(f'Ambiguous videos for {key}: {sorted(found)}')
    if not found:raise FileNotFoundError(f'No generated video matching {key} under {root}')
    return Path(found.pop())


def prepare(args):
    data=read_json(args.annotations)
    if not isinstance(data,list):raise ValueError('Annotation JSON must be a list')
    t2,t3,cf=offline_parsers(BUNDLE)
    overrides=read_json(args.label_overrides) if args.label_overrides else {}
    if not isinstance(overrides,dict):raise ValueError('Overrides must be keyed by video_key')
    if 'cf' in args.tasks and any(
            any(field in value for field in ('orig_emotion','cf_emotion'))
            for value in overrides.values() if isinstance(value,dict)):
        raise ValueError('CF emotion overrides cannot change the conditioning of already-generated videos; '
                         'correct annotations and regenerate affected videos instead')
    rows=[flatten(row) for row in data]
    keys=[row['video_key'] for row in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate video_key in annotations')
    # A CF-only run needs only paired records, including when --limit 0.
    if args.tasks == ['cf'] or (args.limit and 'cf' in args.tasks):
        rows=[row for row in rows if isinstance(row.get('counterfactual'),dict)]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('No selected samples')
    adapted=[]; errors=[]; mappings=[]; notes=[];cf_count=0;used_orig={};used_cf={}
    for row in rows:
        key=row['video_key']
        try:
            changes=apply_reference_mapping(row,t2,overrides.get(key))
            if changes:mappings.append({'video_key':key,'changes':changes})
            sample,warnings=t2.parse_sample(row)
            if not sample['emotion_candidates'] or not sample['action_candidates']:
                raise ValueError('No valid emotion/action candidates')
            t3.parse_constraints((row.get('transition') or {}).get('world_state_delta_items'),[])
            path=resolve_video(args.orig_dir,key)
            if str(path) in used_orig:raise ValueError(f'Original video also mapped to {used_orig[str(path)]}')
            used_orig[str(path)]=key
            row['_video_file']=str(path)
            if sample['target_emotion'] is None:
                notes.append({'video_key':key,'note':'Task2 target emotion is missing/unsupported; candidate scoring can run, target diagnostics unavailable'})
            if 'cf' in args.tasks and isinstance(row.get('counterfactual'),dict):
                cf_count+=1
                branches=cf.parse_branches(row,[])
                if branches['orig']['emotion_family']==branches['cf']['emotion_family']:
                    raise ValueError('state0.emotion and emotion_cf resolve to the same family')
                if not args.cf_dir:raise ValueError('--cf-dir is required for counterfactual samples')
                cf_path=resolve_video(args.cf_dir,key,row['counterfactual'])
                if cf_path==path:raise ValueError('Original and CF resolve to the same file')
                if str(cf_path) in used_cf:raise ValueError(f'CF video also mapped to {used_cf[str(cf_path)]}')
                used_cf[str(cf_path)]=key
                row['counterfactual']['_video_file']=str(cf_path)
            adapted.append(row)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append({'video_key':key,'error':str(exc)})
    if 'cf' in args.tasks and cf_count==0:
        errors.append({'video_key':None,'error':'No counterfactual samples selected'})
    output=args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    write=lambda name,value:(output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={'source':str(args.annotations.resolve()),'source_sha256':file_hash(args.annotations),
             'tasks':args.tasks,'total_annotations':len(data),'selected':len(rows),
             'cf_protocol_version':'state0_emotion_family_v1' if 'cf' in args.tasks else None,
             'ready':len(adapted),'cf_selected':cf_count,'errors':errors,
             'reference_label_mappings':mappings,'notes':notes,
             'orig_dir':str(args.orig_dir.resolve()),
             'cf_dir':str(args.cf_dir.resolve()) if args.cf_dir else None,
             'api_called':False}
    write('preflight.json',summary)
    write('eval_inputs.json',adapted)
    print(f'标注总数 {len(data)}；本次选择 {len(rows)}；通过检查 {len(adapted)}；问题 {len(errors)}',flush=True)
    for error in errors[:8]:print(f'  {error["video_key"]}: {error["error"]}')
    print(f'检查报告：{output/"preflight.json"}')
    print(f'本次数据（已展开并记录映射）：{output/"eval_inputs.json"}')
    return output,summary


@contextmanager
def runtime(bundle,work):
    previous=Path.cwd();argv=sys.argv[:]
    sys.path.insert(0,str(bundle));os.chdir(work)
    try:yield
    finally:
        os.chdir(previous);sys.argv=argv;sys.path.remove(str(bundle))


def run(args,work):
    required={'EWM_VLM_API_KEY','EWM_LLM_API_KEY'}
    missing=sorted(name for name in required if not os.environ.get(name))
    if missing:raise ValueError('Missing environment variables: '+', '.join(missing))
    with runtime(BUNDLE,work):
        t2=importlib.import_module('eval_task2')
        t3=importlib.import_module('eval_task3')
        cf=importlib.import_module('eval_cf')
        t2.DATA_PATH=str(work/'eval_inputs.json');t2.VIDEO_DIR='.';t2.LIMIT=None
        t2.TRANSPORT='base64';t2.CACHE_DIR=str(work/'.ewh_cache')
        t2.VLM_API_KEY=os.environ['EWM_VLM_API_KEY'];t2.LLM_API_KEY=os.environ['EWM_LLM_API_KEY']
        for name in ('VLM_MODEL','LLM_MODEL','VLM_BASE_URL','LLM_BASE_URL'):
            if os.environ.get('EWM_'+name):setattr(t2,name,os.environ['EWM_'+name])
        t3.JUDGE_CACHE_PATH=work/'.ewh_cache/task3_judge.json'
        cf.JUDGE_CACHE_PATH=work/'.ewh_cache/cf_judge.json'
        for module in (t2,t3,cf):
            for name in ('OUT_PATH','SUMMARY_PATH','REVIEW_PATH'):
                setattr(module,name,str(work/Path(getattr(module,name)).name))
        # Full identity/content prevents orig/CF cache collisions on identical basenames.
        def cache_key(path):
            return f'local:{file_hash(path)}|{t2.VLM_MODEL}|{t2.short_hash(t2.DESCRIBE_PROMPT)}'
        t2.video_cache_key=cache_key
        def local_branch(raw,prefix,section=None):
            if section=='counterfactual':return raw['counterfactual']['_video_file']
            return raw['_video_file']
        def local_description(path,clients,caches):
            return t2.get_description({'video_file':path},clients,caches)
        def local_config():
            t2.check_config()
            if cf.SAME_SETTING_GATE and not cf.RUN_DIFF:
                raise ValueError('SAME_SETTING_GATE requires RUN_DIFF')
        cf.branch_oss_key=local_branch;cf.get_oss_description=local_description
        cf.make_bucket=lambda:None;cf.check_cf_config=local_config
        cf.CF_ORIG_SOURCE='task2'
        # Check payload sizes before making even the first API call.
        inputs=read_json(t2.DATA_PATH)
        for row in inputs:
            paths=[row['_video_file']]
            if 'cf' in args.tasks and isinstance(row.get('counterfactual'),dict):
                paths.append(row['counterfactual']['_video_file'])
            for path in paths:
                if 4*((Path(path).stat().st_size+2)//3)>t2.MAX_BASE64_BYTES:
                    raise ValueError(f'Video exceeds evaluator base64 limit: {path}; use an explicit transport plan')
        modules={'task2':t2,'task3':t3,'cf':cf}
        for name in args.tasks:
            print(f'运行 {name}；结果目录 {work}',flush=True)
            sys.argv=[name+'.py']
            modules[name].main()
            failures=[row for row in read_json(modules[name].OUT_PATH) if 'error' in row]
            if failures:raise ValueError(f'{name}: {len(failures)} samples failed; inspect output before continuing')
        report=importlib.import_module('run_all')
        report.REPORT_PATH=work/'ewh_report.json'
        sys.argv=['run_all.py','--report-only']
        report.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['check','run'])
    parser.add_argument('--annotations',type=Path,default=BUNDLE/'private_annotations_1498_normalized.json')
    parser.add_argument('--orig-dir',type=Path,required=True)
    parser.add_argument('--cf-dir',type=Path)
    parser.add_argument('--tasks',nargs='+',choices=['task2','task3','cf'],default=['task2','task3','cf'])
    parser.add_argument('--limit',type=int,default=5,help='5 for trial; 0 for all annotations')
    parser.add_argument('--label-overrides',type=Path,help='Explicit per-video orig_emotion/cf_emotion overrides')
    parser.add_argument('--output-dir',type=Path,default=Path('eval_runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args()
    if args.limit<0:parser.error('--limit must be nonnegative')
    if len(args.tasks)!=len(set(args.tasks)):parser.error('Duplicate tasks')
    try:
        work,summary=prepare(args)
        if summary['errors']:
            print('存在数据或路径问题，未调用 API。请先处理 preflight.json 中的问题。')
            return 2
        if args.mode=='run':run(args,work)
        return 0
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        print(f'无法继续：{exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
