#!/usr/bin/env python3
"""Retry failed DynamiCrafter Task 2 rows and run the pending Task 3/CF cohort."""

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/autodl-tmp')
ADAPTER_DIR = ROOT / 'ewm_eval_local_cf223'
ADAPTER = ADAPTER_DIR / 'ewm_eval_local.py'
ANNOTATIONS = ADAPTER_DIR / 'private_annotations_1498_normalized.json'
ORIG = ROOT / 'ewm_generated_dynamicrafter' / 'normal'
CF = ROOT / 'ewm_generated_dynamicrafter' / 'counterfactual'
SOURCE_RUN = ROOT / 'new_model_eval_runs/dynamicrafter_20261102T091154037326Z'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def verify():
    if not ADAPTER.is_file() or not ANNOTATIONS.is_file():
        raise SystemExit('Missing evaluator or canonical annotations')
    sys.path.insert(0, str(ROOT))
    import start_full_new_models_eval as bundle
    mismatches = bundle.verify_evaluator_code(ADAPTER_DIR)
    if mismatches:
        raise SystemExit('Evaluator bundle mismatch: ' + '; '.join(mismatches))
    if not ORIG.is_dir() or not CF.is_dir():
        raise SystemExit('DynamiCrafter video directories missing')


def run_check(python, ann, tasks, out):
    cmd = [str(python), str(ADAPTER), 'check', '--annotations', str(ann),
           '--orig-dir', str(ORIG), '--cf-dir', str(CF), '--tasks', *tasks,
           '--limit', '0', '--output-dir', str(out)]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f'Offline preflight failed for {ann}:\n{result.stdout}\n{result.stderr}')
    report = read(out / 'preflight.json')
    if report['errors'] or report['selected'] != report['ready']:
        raise SystemExit(f'Preflight reports errors for {ann}: {report["errors"][:3]}')
    return report


def launch(_args):
    verify()
    python = Path(sys.executable).resolve()
    subprocess.run([str(python), '-c', 'import openai, tqdm'], check=True)
    rows = read(ANNOTATIONS)
    if len(rows) != 1498 or len({r['video_key'] for r in rows}) != 1498:
        raise SystemExit('Expected 1498 unique canonical annotations')
    if not SOURCE_RUN.is_dir():
        raise SystemExit('Original DynamiCrafter run not found')

    old_task2 = []
    for i in range(4):
        old = read(SOURCE_RUN / f'shard{i}/task2_results.json')
        old_task2.extend(old)
    if len(old_task2) != 1498 or len({r['video_key'] for r in old_task2}) != 1498:
        raise SystemExit('Original Task 2 results do not cover 1498 unique keys')
    failed = {r['video_key'] for r in old_task2 if 'error' in r}
    by_key = {r['video_key']: r for r in rows}
    if not failed or not failed <= by_key.keys():
        raise SystemExit('No valid failed Task 2 rows found to retry')
    failed_rows = [r for r in rows if r['video_key'] in failed]

    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = ROOT / 'new_model_eval_runs' / f'dynamicrafter_recovery_{tag}'
    run.mkdir(parents=True, exist_ok=False)
    shards = []
    for i in range(4):
        normal_rows = rows[i::4]
        retry_rows = failed_rows[i::4]
        normal_ann = run / f'annotations_shard{i}.json'
        retry_ann = run / f'task2_retry_annotations_shard{i}.json'
        normal_ann.write_text(json.dumps(normal_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        retry_ann.write_text(json.dumps(retry_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        stages = {}
        if retry_rows:
            p = run / f'preflight_task2_retry_shard{i}'
            report = run_check(python, retry_ann, ['task2'], p)
            if report['selected'] != len(retry_rows):
                raise SystemExit(f'Task 2 retry preflight count mismatch in shard{i}')
            stages['task2_retry'] = {'annotations': str(retry_ann), 'preflight': str(p),
                                     'output': str(run / f'task2_retry_shard{i}'),
                                     'expected': len(retry_rows), 'task': 'task2'}
        for task in ('task3', 'cf'):
            p = run / f'preflight_{task}_shard{i}'
            report = run_check(python, normal_ann, [task], p)
            expected_cf = report['cf_selected']
            expected = expected_cf if task == 'cf' else len(normal_rows)
            if task == 'cf' and expected_cf == 0:
                raise SystemExit(f'CF shard{i} has no selected videos')
            stages[task] = {'annotations': str(normal_ann), 'preflight': str(p),
                            'output': str(run / f'{task}_shard{i}'),
                            'expected': expected, 'task': task}
        shards.append({'index': i, 'normal_count': len(normal_rows),
                       'retry_count': len(retry_rows), 'stages': stages,
                       'log': str(run / f'worker{i}.log')})
    if sum(s['stages']['cf']['expected'] for s in shards) != 223:
        raise SystemExit('CF shard cohort must contain exactly 223 samples')

    vlm = getpass.getpass('VLM API key (hidden): ').strip()
    llm = getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm or not llm:
        raise SystemExit('Both API keys are required')
    env = os.environ.copy()
    env.update(EWM_VLM_API_KEY=vlm, EWM_LLM_API_KEY=llm)
    del vlm, llm
    manifest = {'created_utc': tag, 'state': 'running', 'model': 'dynamicrafter',
                'source_run': str(SOURCE_RUN), 'failed_task2_rows': len(failed),
                'workers': 4, 'shards': shards}
    manifest_path = run / 'launch_manifest.json'
    write(manifest_path, manifest)
    for shard in shards:
        config = run / f'worker{shard["index"]}.json'
        write(config, {'root': str(ROOT), 'adapter': str(ADAPTER), 'orig': str(ORIG),
                       'cf': str(CF), 'stages': shard['stages']})
        cmd = [str(python), '-u', str(Path(__file__).resolve()), 'worker', '--config', str(config)]
        with open(shard['log'], 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        shard['pid'] = proc.pid
        write(manifest_path, manifest)
        print(f"shard{shard['index']}: PID {proc.pid}; log {shard['log']}", flush=True)
    del env
    print(f'RECOVERY_STARTED; failed Task 2={len(failed)}; 4 workers; run={run}', flush=True)


def worker(config_path):
    cfg = read(config_path)
    for stage_name, stage in cfg['stages'].items():
        print(f"STAGE_START {stage_name} expected={stage['expected']}", flush=True)
        cmd = [sys.executable, '-u', cfg['adapter'], 'run', '--annotations', stage['annotations'],
               '--orig-dir', cfg['orig'], '--cf-dir', cfg['cf'], '--tasks', stage['task'],
               '--limit', '0', '--output-dir', stage['output']]
        try:
            result = subprocess.run(cmd, cwd=cfg['root'], check=False)
            print(f'STAGE_DONE {stage_name} exit={result.returncode}', flush=True)
        except Exception as exc:
            print(f'STAGE_EXCEPTION {stage_name} {exc!r}', flush=True)
    print('WORKER_DONE', flush=True)


def status(args):
    target = args.target
    if target is None:
        candidates = sorted((ROOT / 'new_model_eval_runs').glob('dynamicrafter_recovery_*/launch_manifest.json'))
        if not candidates:
            raise SystemExit('No DynamiCrafter recovery run found')
        target = candidates[-1].parent
    manifest = read(target / 'launch_manifest.json')
    for shard in manifest['shards']:
        pid = shard.get('pid')
        alive = bool(pid and Path(f'/proc/{pid}/stat').is_file()
                     and Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z')
        bits = []
        for name, stage in shard['stages'].items():
            f = Path(stage['output']) / f"{stage['task']}_results.json"
            records = read(f) if f.is_file() else []
            errors = sum('error' in r for r in records)
            bits.append(f"{name}={len(records)-errors}/{stage['expected']},errors={errors},pending={stage['expected']-len(records)}")
        print(f"shard{shard['index']} pid={pid} alive={alive} " + ' | '.join(bits))
    print('run=' + str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('launch')
    stat = sub.add_parser('status')
    stat.add_argument('--target', type=Path)
    worker_parser = sub.add_parser('worker')
    worker_parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'launch':
        launch(args)
    elif args.command == 'status':
        status(args)
    else:
        worker(args.config)


if __name__ == '__main__':
    main()
