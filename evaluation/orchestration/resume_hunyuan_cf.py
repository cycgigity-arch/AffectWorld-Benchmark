#!/usr/bin/env python3
"""Resume only the unfinished Hunyuan CF evaluations on the existing four shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import start_hunyuan_full_eval as baseline


ROOT = Path('/root/autodl-tmp')
TARGET = ROOT / 'hunyuan_full_eval_runs/hunyuan_1721_20261105'
SOURCE = ROOT / 'ewm_eval_local_cf223/private_annotations_1498_normalized.json'
STOP_PHRASES = ('access denied', 'arrearage', 'overdue-payment', '余额')


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def counts(work: Path, expected: int) -> dict:
    path = work / 'cf_results.json'
    rows = load(path) if path.is_file() else []
    good = {x['video_key'] for x in rows if isinstance(x, dict) and 'video_key' in x and 'error' not in x}
    errors = {x['video_key'] for x in rows if isinstance(x, dict) and 'video_key' in x and 'error' in x} - good
    return {'success': len(good), 'errors': len(errors),
            'unattempted': expected - len(good) - len(errors),
            'remaining': expected - len(good)}


def validate() -> dict:
    manifest = load(TARGET / 'launch_manifest.json')
    if sha(SOURCE) != manifest['source_sha256']:
        raise SystemExit('Annotation hash differs from original run')
    if len(manifest['shards']) != 4 or baseline.create_cf_runtime(TARGET) != 223:
        raise SystemExit('Hunyuan CF source or runtime mapping is incomplete')
    for shard in manifest['shards']:
        if counts(Path(shard['work']), shard['cf'])['success'] > shard['cf']:
            raise SystemExit(f'Invalid CF result count: {shard["work"]}')
    return manifest


def alive(pid: int | None) -> bool:
    return bool(pid) and baseline.pid_alive(pid)


def backup(manifest: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder = ROOT / 'hunyuan_full_eval_runs/backups' / f'cf_before_resume_{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for shard in manifest['shards']:
        source = Path(shard['work']) / 'cf_results.json'
        target = folder / f'shard{shard["index"]}_cf_results.json'
        shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise SystemExit(f'Backup mismatch: {source}')
        entries.append({'source': str(source), 'backup': str(target), 'sha256': sha(target)})
    (folder / 'manifest.json').write_text(json.dumps({'created_utc': stamp, 'entries': entries}, indent=2) + '\n')
    return folder


def start() -> None:
    manifest = validate()
    launch = TARGET / 'cf_resume_manifest.json'
    if launch.exists() and any(alive(x.get('pid')) for x in load(launch)['shards']):
        raise SystemExit('CF resume workers are already running')
    saved = backup(manifest)
    print('CF backup verified:', saved, flush=True)
    vlm = getpass.getpass('VLM API key (hidden): ').strip()
    llm = getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm or not llm:
        raise SystemExit('Both API keys are required')
    env = os.environ.copy()
    env['EWM_VLM_API_KEY'] = vlm
    env['EWM_LLM_API_KEY'] = llm
    del vlm, llm
    workers = []
    for shard in manifest['shards']:
        index = shard['index']
        work = Path(shard['work'])
        if counts(work, shard['cf'])['remaining'] == 0:
            print(f'shard{index} CF already complete', flush=True)
            continue
        log = TARGET / f'cf_shard{index}_resume.log'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--shard', str(index)]
        with log.open('ab') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        workers.append({'index': index, 'pid': child.pid, 'log': str(log), 'expected': shard['cf']})
        print(f'started CF shard{index} pid={child.pid}', flush=True)
    launch.write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(),
                                  'backup': str(saved), 'source_sha256': sha(SOURCE),
                                  'shards': workers}, indent=2) + '\n')


def billing_watch(work: Path, original_errors: set[tuple[str, str]], started: float,
                  stop: threading.Event) -> None:
    path = work / 'cf_results.json'
    while not stop.wait(3):
        try:
            if path.stat().st_mtime < started:
                continue
            rows = load(path)
            denied = sum(
                (str(x.get('video_key')), str(x.get('error'))) not in original_errors
                and any(term in str(x.get('error', '')).lower() for term in STOP_PHRASES)
                for x in rows if isinstance(x, dict) and 'error' in x)
            if denied >= 3:
                print(f'BILLING_BLOCK: {denied} new balance errors; stopping shard', flush=True)
                os._exit(75)
        except (OSError, ValueError, TypeError):
            continue


def worker(index: int) -> None:
    manifest = validate()
    shard = next(x for x in manifest['shards'] if x['index'] == index)
    work = Path(shard['work'])
    previous = load(work / 'cf_results.json')
    old_errors = {(str(x.get('video_key')), str(x.get('error'))) for x in previous
                  if isinstance(x, dict) and 'error' in x}
    print(f'shard{index} before={counts(work, shard["cf"])}', flush=True)
    stop = threading.Event()
    watch = threading.Thread(target=billing_watch,
                             args=(work, old_errors, time.time(), stop), daemon=True)
    watch.start()
    adapter = baseline.load_adapter()
    try:
        adapter.run(argparse.Namespace(tasks=['cf'], cf_dir=TARGET / 'cf_runtime'), work)
    except Exception as error:
        print(f'shard{index} CF ended with {type(error).__name__}: {error}', flush=True)
    finally:
        stop.set()
        watch.join(timeout=5)
    print(f'shard{index} after={counts(work, shard["cf"])}', flush=True)


def status() -> None:
    manifest = validate()
    path = TARGET / 'cf_resume_manifest.json'
    running = {x['index']: alive(x.get('pid')) for x in load(path)['shards']} if path.exists() else {}
    totals = {'success': 0, 'errors': 0, 'unattempted': 0, 'remaining': 0}
    for shard in manifest['shards']:
        index = shard['index']
        state = counts(Path(shard['work']), shard['cf'])
        for key in totals:
            totals[key] += state[key]
        print(f'shard{index} alive={running.get(index, False)} {state}')
    print('TOTAL CF:', totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('start', 'status', 'worker'))
    parser.add_argument('--shard', type=int, choices=range(4))
    args = parser.parse_args()
    if args.command == 'start':
        start()
    elif args.command == 'status':
        status()
    else:
        if args.shard is None:
            parser.error('--shard required')
        worker(args.shard)


if __name__ == '__main__':
    main()
