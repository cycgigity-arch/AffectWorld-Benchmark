#!/usr/bin/env python3
"""Preflight and launch four resumable DynamiCrafter evaluation shards."""

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
TASKS = ('task2', 'task3', 'cf')
EXPECTED = {'task2': 1498, 'task3': 1498, 'cf': 223}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_sources():
    if not ADAPTER.is_file() or not ANNOTATIONS.is_file():
        raise SystemExit('Missing EWM evaluator or canonical annotations')
    sys.path.insert(0, str(ROOT))
    import start_full_new_models_eval as source_bundle
    mismatches = source_bundle.verify_evaluator_code(ADAPTER_DIR)
    if mismatches:
        raise SystemExit('Evaluator bundle mismatch: ' + '; '.join(mismatches))
    if not ORIG.is_dir() or not CF.is_dir():
        raise SystemExit('DynamiCrafter normal/counterfactual directories missing')


def launch(args):
    verify_sources()
    python = Path(sys.executable).resolve()
    subprocess.run([str(python), '-c', 'import openai, tqdm'], check=True)
    source = load(ANNOTATIONS)
    if len(source) != 1498 or len({r['video_key'] for r in source}) != 1498:
        raise SystemExit('Canonical annotation cohort must have 1498 unique videos')
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = ROOT / 'new_model_eval_runs' / ('dynamicrafter_' + tag)
    run.mkdir(parents=True, exist_ok=False)
    shards = []
    for i in range(4):
        rows = source[i::4]
        ann = run / f'annotations_shard{i}.json'
        ann.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        work = run / f'shard{i}'
        preflight = run / f'preflight_shard{i}'
        cmd = [str(python), str(ADAPTER), 'check', '--annotations', str(ann),
               '--orig-dir', str(ORIG), '--cf-dir', str(CF), '--tasks', *TASKS,
               '--limit', '0', '--output-dir', str(preflight)]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            raise SystemExit(f'shard{i} offline preflight failed:\n{result.stdout}\n{result.stderr}')
        report = load(preflight / 'preflight.json')
        expected_cf = report['cf_selected']
        if (report['errors'] or report['selected'] != len(rows)
                or report['ready'] != len(rows)):
            raise SystemExit(f'shard{i} preflight counts/errors mismatch')
        shards.append({'index': i, 'annotations': str(ann), 'preflight': str(preflight),
                       'output': str(work), 'log': str(run / f'shard{i}.log'),
                       'expected_normal': len(rows), 'expected_cf': expected_cf})
    if sum(shard['expected_cf'] for shard in shards) != EXPECTED['cf']:
        raise SystemExit('Shard preflight cohort must contain exactly 223 CF samples')

    vlm = getpass.getpass('VLM API key (hidden): ').strip()
    llm = getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm or not llm:
        raise SystemExit('Both API keys are required')
    env = os.environ.copy()
    env['EWM_VLM_API_KEY'] = vlm
    env['EWM_LLM_API_KEY'] = llm
    del vlm, llm

    manifest = {'created_utc': tag, 'state': 'running', 'model': 'dynamicrafter',
                'tasks': list(TASKS), 'workers': 4, 'shards': shards}
    manifest_path = run / 'launch_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for shard in shards:
        cmd = [str(python), '-u', str(ADAPTER), 'run', '--annotations', shard['annotations'],
               '--orig-dir', str(ORIG), '--cf-dir', str(CF), '--tasks', *TASKS,
               '--limit', '0', '--output-dir', shard['output']]
        with open(shard['log'], 'w', encoding='utf-8') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        shard['pid'] = proc.pid
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f"shard{shard['index']}: PID {proc.pid}; log {shard['log']}", flush=True)
    del env
    print('DynamiCrafter full scoring launched; run=' + str(run), flush=True)


def status(args):
    target = args.target
    if target is None:
        candidates = sorted((ROOT / 'new_model_eval_runs').glob('dynamicrafter_*/launch_manifest.json'))
        if not candidates:
            raise SystemExit('No DynamiCrafter evaluation run found')
        target = candidates[-1].parent
    manifest = load(target / 'launch_manifest.json')
    for shard in manifest['shards']:
        out = Path(shard['output'])
        alive = False
        pid = shard.get('pid')
        if pid and Path(f'/proc/{pid}/stat').is_file():
            alive = Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
        parts = []
        for task in TASKS:
            path = out / f'{task}_results.json'
            rows = load(path) if path.is_file() else []
            errors = sum('error' in row for row in rows)
            total = shard['expected_cf'] if task == 'cf' else shard['expected_normal']
            parts.append(f'{task}={len(rows)-errors}/{total},errors={errors},pending={total-len(rows)}')
        print(f"shard{shard['index']} pid={pid} alive={alive} " + ' | '.join(parts))
    print('run=' + str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('launch')
    stat = sub.add_parser('status')
    stat.add_argument('--target', type=Path)
    args = parser.parse_args()
    if args.command == 'launch':
        launch(args)
    else:
        status(args)


if __name__ == '__main__':
    main()
