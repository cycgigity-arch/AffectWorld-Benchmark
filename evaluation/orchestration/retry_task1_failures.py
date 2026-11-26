#!/usr/bin/env python3
"""Retry only failed Task1 videos for the four newly evaluated models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path('/root/autodl-tmp')
RUNNER = ROOT / 'ewm_eval_local_cf223/run_hunyuan_task1_scores.py'
ANNOTATIONS = ROOT / 'ewm_eval_local_cf223/private_annotations_1498_normalized.json'
VFI_REPO = ROOT / 'VFIMamba-main'
RETRY = ROOT / 'task1_eval_runs/retry_failed_20261108'
MODELS = {
    'hunyuan': (ROOT / 'ewm_generated_hunyuan_full_5s/normal',
                ROOT / 'hunyuan_full_eval_runs/task1_full_5s_20261108'),
    'svd': (ROOT / 'ewm_generated_svd/normal',
            ROOT / 'task1_eval_runs/svd_full_5s_20261108'),
    'easyanimate': (ROOT / 'ewm_generated_easyanimate/normal',
                    ROOT / 'task1_eval_runs/easyanimate_full_5s_20261108'),
    'dynamicrafter': (ROOT / 'ewm_generated_dynamicrafter/normal',
                      ROOT / 'task1_eval_runs/dynamicrafter_full_5s_20261108'),
}
PROGRESS = re.compile(r'PROGRESS (\d+)/(\d+)')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def counts(path: Path) -> dict:
    scored: set[str] = set()
    errors: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get('video_key')
            if isinstance(key, str):
                (scored if record.get('status') == 'scored' else errors).add(key)
    return {'scored': len(scored), 'failed': len(errors - scored),
            'covered': len(scored | errors)}


def alive(pid: int) -> bool:
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ', 1)[1][0] != 'Z'
    except (OSError, IndexError):
        return False


def progress(log: Path) -> tuple[int, int]:
    if not log.exists():
        return (0, 0)
    matches = PROGRESS.findall(log.read_text(encoding='utf-8', errors='replace'))
    return tuple(map(int, matches[-1])) if matches else (0, 0)


def retry_keys(output: Path, backup: Path) -> set[str]:
    before = len(backup.read_text(encoding='utf-8').splitlines())
    lines = (output / 'results.jsonl').read_text(encoding='utf-8').splitlines()
    keys = set()
    for line in lines[before:]:
        try:
            key = json.loads(line).get('video_key')
        except json.JSONDecodeError:
            continue
        if isinstance(key, str):
            keys.add(key)
    return keys


def start() -> None:
    if RETRY.exists():
        raise SystemExit(f'Retry manifest directory already exists: {RETRY}')
    if sha(ANNOTATIONS) != 'aa09b47e6cdc17fd0729a5f0a9dae057f77e780c2c93895733284364fd7c5aa3':
        raise SystemExit('Annotation file changed; refusing to mix evaluation protocols')
    pending = {}
    for model, (video_root, output) in MODELS.items():
        if not video_root.is_dir() or not (output / 'results.jsonl').is_file():
            raise SystemExit(f'Missing {model} source or first-pass results')
        original_pid = output / 'pid'
        if original_pid.is_file() and alive(int(original_pid.read_text().strip())):
            raise SystemExit(f'{model} first-pass scorer is still running')
        state = counts(output / 'results.jsonl')
        if state['covered'] != 1498:
            raise SystemExit(f'{model} first pass does not cover 1498 keys: {state}')
        pending[model] = state
    RETRY.mkdir(parents=True)
    backup_dir = RETRY / 'backups'
    backup_dir.mkdir()
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'annotation_sha256': sha(ANNOTATIONS), 'models': {}}
    for model, (video_root, output) in MODELS.items():
        source = output / 'results.jsonl'
        backup = backup_dir / f'{model}_results_before_retry.jsonl'
        shutil.copy2(source, backup)
        if sha(source) != sha(backup):
            raise SystemExit(f'{model} backup hash mismatch')
        manifest['models'][model] = {'before': pending[model], 'backup': str(backup),
                                     'backup_sha256': sha(backup), 'output': str(output),
                                     'video_root': str(video_root)}
    for model, (video_root, output) in MODELS.items():
        if pending[model]['failed'] == 0:
            continue
        log = RETRY / f'{model}_retry.log'
        cmd = [sys.executable, '-u', str(RUNNER), '--video-root', str(video_root),
               '--annotations', str(ANNOTATIONS), '--vfi-repo', str(VFI_REPO),
               '--output-dir', str(output)]
        with log.open('ab') as stream:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        manifest['models'][model].update({'pid': child.pid, 'log': str(log)})
        print(f'{model}: retrying {pending[model]["failed"]} failures, pid={child.pid}', flush=True)
    (RETRY / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def status() -> None:
    path = RETRY / 'manifest.json'
    if not path.exists():
        raise SystemExit('Retry not launched')
    manifest = json.loads(path.read_text())
    for model, detail in manifest['models'].items():
        state = counts(Path(detail['output']) / 'results.jsonl')
        retried = retry_keys(Path(detail['output']), Path(detail['backup']))
        pid = detail.get('continue_pid', detail.get('pid', 0))
        print(f'{model} alive={alive(pid)} '
              f'retry_attempted={len(retried)}/{detail["before"]["failed"]} '
              f'scored={state["scored"]}/1498 failed={state["failed"]}')


def continue_retry() -> None:
    path = RETRY / 'manifest.json'
    if not path.exists():
        raise SystemExit('Initial retry and backup must exist first')
    manifest = json.loads(path.read_text())
    for model, detail in manifest['models'].items():
        if alive(detail.get('pid', 0)) or alive(detail.get('continue_pid', 0)):
            raise SystemExit(f'{model} retry is still running')
    for model, detail in manifest['models'].items():
        retried = retry_keys(Path(detail['output']), Path(detail['backup']))
        remaining = detail['before']['failed'] - len(retried)
        if remaining <= 0:
            continue
        log = RETRY / f'{model}_retry_remaining.log'
        with log.open('ab') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      'worker', '--model', model], stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        detail['continue_pid'] = child.pid
        detail['continue_log'] = str(log)
        print(f'{model}: retrying {remaining} still-unretried failures, pid={child.pid}', flush=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def worker(model: str) -> None:
    sys.path.insert(0, str(RUNNER.parent))
    import run_hunyuan_task1_scores as original

    detail = json.loads((RETRY / 'manifest.json').read_text())['models'][model]
    video_root = Path(detail['video_root'])
    output = Path(detail['output'])
    backup = Path(detail['backup'])
    original_rows = [json.loads(line) for line in backup.read_text(encoding='utf-8').splitlines()]
    failed = {row['video_key'] for row in original_rows if row.get('status') != 'scored'}
    args = argparse.Namespace(
        score_python=ROOT / 'task1-eval-env/bin/python',
        scorer=RUNNER.parent / 'task1_full_score.py',
        protocol=RUNNER.parent / 'task1_full_score_trial_protocol.json',
        vfi_python=ROOT / 'vfimamba-env/bin/python',
        vfi_repo=VFI_REPO,
        hf_home=ROOT / 'cache/huggingface',
        artifacts=output / 'artifacts',
        logs=output / 'logs',
    )
    with (output / 'task1_hunyuan.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = original.plan(ANNOTATIONS, video_root)
        attempted = retry_keys(output, backup)
        remaining = [(key, video) for key, video in jobs if key in failed and key not in attempted]
        print(f'{model}: already retried={len(attempted)}/{len(failed)}, pending={len(remaining)}', flush=True)
        index = output / 'results.jsonl'
        for key, video in remaining:
            try:
                original.score_one(key, video, args, index)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                record = {'video_key': key, 'kind': 'generated_future', 'source': str(video),
                          'status': 'error', 'error': f'{type(error).__name__}: {error}'[:1500],
                          'retry_exception': True}
                with index.open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f'ERROR {key}: {record["error"]}', flush=True)
            attempted.add(key)
            print(f'RETRY_PROGRESS {len(attempted)}/{len(failed)}', flush=True)
        print(f'RETRY_FINISHED {model}: {len(attempted)}/{len(failed)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'continue', 'worker', 'status'))
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    if args.action == 'start':
        start()
    elif args.action == 'continue':
        continue_retry()
    elif args.action == 'worker':
        if not args.model:
            parser.error('--model required for worker')
        worker(args.model)
    else:
        status()


if __name__ == '__main__':
    main()
