#!/usr/bin/env python3
"""Resume only unfinished Hunyuan Task2 and Task3 in four existing shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time

import start_hunyuan_full_eval as baseline

ROOT = Path('/root/autodl-tmp')
TARGET = ROOT / 'hunyuan_full_eval_runs/hunyuan_1721_20261105'
SOURCE = ROOT / 'ewm_eval_local_cf223/private_annotations_1498_normalized.json'
TASKS = ('task2', 'task3')
STOP_PHRASES = ('access denied', 'arrearage', 'overdue-payment')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, data) -> None:
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def counts(work: Path, task: str, expected: int) -> dict:
    path = work / f'{task}_results.json'
    records = load(path) if path.exists() else []
    success = {r['video_key'] for r in records if isinstance(r, dict) and 'error' not in r}
    errors = {r['video_key'] for r in records if isinstance(r, dict) and 'error' in r} - success
    return {'success': len(success), 'errors': len(errors),
            'unattempted': expected - len(success) - len(errors),
            'remaining': expected - len(success)}


def moderation_only_failures(work: Path, task: str, expected: int) -> bool:
    """Allow later tasks when every remaining item is provider-blocked video content."""
    state = counts(work, task, expected)
    if state['remaining'] == 0 or state['unattempted'] != 0:
        return False
    records = load(work / f'{task}_results.json')
    failures = [r for r in records if isinstance(r, dict) and 'error' in r]
    return len(failures) == state['errors'] and all(
        'data_inspection_failed' in str(r['error']).lower() for r in failures)


def validate() -> dict:
    manifest = load(TARGET / 'launch_manifest.json')
    if sha(SOURCE) != manifest['source_sha256']:
        raise SystemExit('Original annotation hash differs from the existing run')
    if len(manifest['shards']) != 4:
        raise SystemExit('Expected four existing Hunyuan shards')
    for shard in manifest['shards']:
        work = Path(shard['work'])
        inputs = load(work / 'eval_inputs.json')
        if len(inputs) != shard['normal'] or len(inputs) not in (374, 375):
            raise SystemExit(f'Input count mismatch: {work}')
        for task in TASKS:
            if counts(work, task, shard['normal'])['success'] > shard['normal']:
                raise SystemExit(f'Too many results: {work} {task}')
    return manifest


def snapshot() -> dict:
    manifest = validate()
    backup_dir = ROOT / 'hunyuan_full_eval_runs/backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = backup_dir / f'task23_before_resume_{stamp}.tar'
    if archive.exists():
        raise SystemExit(f'Backup already exists: {archive}')
    with tarfile.open(archive, 'w') as tar:
        tar.add(TARGET, arcname=TARGET.name)
    info = {'created_utc': stamp, 'archive': str(archive),
            'archive_sha256': sha(archive), 'source_sha256': sha(SOURCE),
            'target': str(TARGET), 'shards': len(manifest['shards'])}
    save(backup_dir / f'task23_before_resume_{stamp}.json', info)
    return info


def alive(pid: int | None) -> bool:
    return bool(pid) and baseline.pid_alive(pid)


def start() -> None:
    baseline_manifest = validate()
    run_manifest_path = TARGET / 'task23_resume_manifest.json'
    if run_manifest_path.exists():
        previous = load(run_manifest_path)
        if any(alive(s.get('pid')) for s in previous['shards']):
            raise SystemExit('A Task2/3 resume worker is already running')
    backup = snapshot()
    print('Backup verified:', backup['archive'], backup['archive_sha256'], flush=True)
    vlm = getpass.getpass('VLM API key (hidden): ').strip()
    llm = getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm or not llm:
        raise SystemExit('Both API keys are required')
    env = os.environ.copy()
    env['EWM_VLM_API_KEY'] = vlm
    env['EWM_LLM_API_KEY'] = llm
    del vlm, llm
    shards = []
    for shard in baseline_manifest['shards']:
        index = shard['index']
        work = Path(shard['work'])
        if all(counts(work, task, shard['normal'])['remaining'] == 0 for task in TASKS):
            print(f'shard{index} already complete', flush=True)
            continue
        log = TARGET / f'task23_shard{index}_resume.log'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker',
               '--shard', str(index)]
        with log.open('ab') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        shards.append({'index': index, 'pid': child.pid, 'log': str(log),
                       'expected': shard['normal']})
        print(f'started shard{index} pid={child.pid} log={log}', flush=True)
    save(run_manifest_path, {'created_utc': datetime.now(timezone.utc).isoformat(),
                             'backup': backup, 'source_sha256': sha(SOURCE),
                             'tasks': list(TASKS), 'shards': shards})
    print('Task2/3 resume launched', flush=True)


def billing_watch(work: Path, task: str, started: float, stop: threading.Event) -> None:
    path = work / f'{task}_results.json'
    while not stop.wait(3):
        try:
            if path.stat().st_mtime < started:
                continue
            records = load(path)
            denied = sum(any(term in str(r.get('error', '')).lower() for term in STOP_PHRASES)
                         for r in records if isinstance(r, dict) and 'error' in r)
            if denied >= 3:
                print(f'BILLING_BLOCK: {task} got {denied} access/balance errors; stopping this shard', flush=True)
                os._exit(75)
        except (OSError, ValueError, TypeError):
            continue


def worker(index: int) -> None:
    manifest = validate()
    shard = next(s for s in manifest['shards'] if s['index'] == index)
    work = Path(shard['work'])
    adapter = baseline.load_adapter()
    for task in TASKS:
        before = counts(work, task, shard['normal'])
        print(f'shard{index} {task} before={before}', flush=True)
        if before['remaining'] == 0:
            continue
        if task == 'task2' and moderation_only_failures(work, task, shard['normal']):
            print(f'shard{index} task2 has only provider content-inspection failures; '
                  'preserving those error records and continuing to task3', flush=True)
            continue
        stopped = threading.Event()
        watch = threading.Thread(target=billing_watch,
                                 args=(work, task, time.time(), stopped), daemon=True)
        watch.start()
        try:
            adapter.run(argparse.Namespace(tasks=[task], cf_dir=TARGET / 'cf_runtime'), work)
        except Exception as exc:
            print(f'shard{index} {task} run ended with {type(exc).__name__}: {exc}', flush=True)
        finally:
            stopped.set()
            watch.join(timeout=5)
        after = counts(work, task, shard['normal'])
        print(f'shard{index} {task} after={after}', flush=True)
        if after['remaining']:
            if task == 'task2' and moderation_only_failures(work, task, shard['normal']):
                print(f'shard{index} task2 has only provider content-inspection failures; '
                      'continuing to task3', flush=True)
                continue
            print(f'shard{index} stopping before next task; {task} incomplete', flush=True)
            return
    print(f'shard{index} Task2/3 complete', flush=True)


def status() -> None:
    manifest = validate()
    running = {}
    path = TARGET / 'task23_resume_manifest.json'
    if path.exists():
        running = {s['index']: alive(s.get('pid')) for s in load(path)['shards']}
    totals = {task: {'success': 0, 'remaining': 0, 'errors': 0, 'unattempted': 0}
              for task in TASKS}
    for shard in manifest['shards']:
        index = shard['index']
        parts = []
        for task in TASKS:
            c = counts(Path(shard['work']), task, shard['normal'])
            for k in totals[task]:
                totals[task][k] += c[k]
            parts.append(f'{task}={c["success"]}/{shard["normal"]} remaining={c["remaining"]}')
        print(f'shard{index} alive={running.get(index, False)} ' + ' '.join(parts))
    for task in TASKS:
        print(f'TOTAL {task}: {totals[task]}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'worker', 'status'])
    parser.add_argument('--shard', type=int, choices=range(4))
    args = parser.parse_args()
    if args.command == 'start':
        start()
    elif args.command == 'status':
        status()
    else:
        if args.shard is None:
            parser.error('--shard is required for worker')
        worker(args.shard)


if __name__ == '__main__':
    main()
