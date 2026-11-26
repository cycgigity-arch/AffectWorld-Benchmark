#!/usr/bin/env python3
"""Queue unscored EWM Task1 models after the current Hunyuan Task1 run.

Runs on the AutoDL evaluation machine. Each model keeps independent Task1
results and logs; per-video errors are recorded by the existing scorer.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path('/root/autodl-tmp')
QUEUE = ROOT / 'task1_eval_queue_20261108'
HUNYUAN = ROOT / 'hunyuan_full_eval_runs/task1_full_5s_20261108'
BUNDLE = ROOT / 'ewm_eval_local_cf223'
RUNNER = BUNDLE / 'run_hunyuan_task1_scores.py'
ANNOTATIONS = BUNDLE / 'private_annotations_1498_normalized.json'
VFI_REPO = ROOT / 'VFIMamba-main'
MODELS = ('svd', 'easyanimate', 'dynamicrafter')
EXPECTED = 1498
PROGRESS = re.compile(r'PROGRESS (\d+)/(\d+)')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def progress(log: Path) -> tuple[int, int]:
    if not log.is_file():
        return (0, 0)
    # These logs are small; reading the whole file also handles a partial final line.
    matches = PROGRESS.findall(log.read_text(encoding='utf-8', errors='replace'))
    return tuple(map(int, matches[-1])) if matches else (0, 0)


def running(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text().strip())
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ', 1)[1][0] != 'Z'
    except (OSError, ValueError, IndexError):
        return False


def result_counts(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return (0, 0)
    successes: set[str] = set()
    failures: set[str] = set()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get('video_key')
            if isinstance(key, str):
                (successes if row.get('status') == 'scored' else failures).add(key)
    return (len(successes), len(failures - successes))


def announce(message: str) -> None:
    print(f'{now()} {message}', flush=True)


def wait_for_hunyuan() -> None:
    log = HUNYUAN / 'run.log'
    pid_path = HUNYUAN / 'pid'
    while running(pid_path):
        done, total = progress(log)
        announce(f'waiting for Hunyuan Task1: attempted={done}/{total or EXPECTED}')
        time.sleep(60)
    done, total = progress(log)
    content = log.read_text(encoding='utf-8', errors='replace') if log.exists() else ''
    if (done, total) != (EXPECTED, EXPECTED) or 'RESULTS ' not in content:
        raise RuntimeError(f'Hunyuan Task1 ended early at {done}/{total}; inspect {log}')
    scored, failed = result_counts(HUNYUAN / 'results.jsonl')
    announce(f'Hunyuan Task1 attempted all {EXPECTED}: scored={scored}, failed={failed}')


def model_paths(model: str) -> tuple[Path, Path]:
    return (ROOT / f'ewm_generated_{model}/normal',
            ROOT / f'task1_eval_runs/{model}_full_5s_20261108')


def command(model: str, output: Path, check_only: bool = False) -> list[str]:
    video_root, _ = model_paths(model)
    args = [sys.executable, '-u', str(RUNNER), '--video-root', str(video_root),
            '--annotations', str(ANNOTATIONS), '--vfi-repo', str(VFI_REPO),
            '--output-dir', str(output)]
    if check_only:
        args.append('--check-only')
    return args


def run_model(model: str, manifest: dict) -> bool:
    video_root, output = model_paths(model)
    if not video_root.is_dir():
        raise RuntimeError(f'{model} video directory missing: {video_root}')
    preflight = subprocess.run(command(model, output, True), capture_output=True,
                              text=True, check=False)
    if preflight.returncode or 'available 1498/1498, missing 0, invalid 0' not in preflight.stdout:
        raise RuntimeError(f'{model} preflight failed: {preflight.stdout} {preflight.stderr}')
    output.mkdir(parents=True, exist_ok=True)
    log = output / 'run.log'
    manifest['models'][model] = {'state': 'running', 'started_utc': now(),
                                 'video_root': str(video_root), 'output': str(output),
                                 'log': str(log)}
    write_json(QUEUE / 'manifest.json', manifest)
    announce(f'{model} Task1 started: {video_root}; output={output}')
    with log.open('a', encoding='utf-8') as stream:
        finished = subprocess.run(command(model, output), stdin=subprocess.DEVNULL,
                                  stdout=stream, stderr=subprocess.STDOUT, check=False)
    attempted, total = progress(log)
    scored, failed = result_counts(output / 'results.jsonl')
    complete_attempt = (attempted, total) == (EXPECTED, EXPECTED) and 'RESULTS ' in log.read_text(encoding='utf-8', errors='replace')
    manifest['models'][model].update({'state': 'attempted_all' if complete_attempt else 'stopped_early',
                                       'finished_utc': now(), 'exit_code': finished.returncode,
                                       'attempted': attempted, 'scored': scored, 'failed': failed})
    write_json(QUEUE / 'manifest.json', manifest)
    announce(f'{model} Task1 finished: attempted={attempted}/{total}, scored={scored}, failed={failed}, exit={finished.returncode}')
    return complete_attempt


def run_queue() -> None:
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = QUEUE / 'manifest.json'
        manifest = read_json(path) if path.is_file() else {'created_utc': now(), 'models': {}}
        if manifest.get('state') == 'parallel_running':
            raise RuntimeError('Parallel Task1 workers are active; use status instead')
        manifest['state'] = 'waiting_for_hunyuan'
        write_json(path, manifest)
        try:
            wait_for_hunyuan()
            manifest['state'] = 'running'
            write_json(path, manifest)
            for model in MODELS:
                previous = manifest['models'].get(model, {})
                if previous.get('state') == 'attempted_all':
                    announce(f'{model} already attempted all; skipping')
                    continue
                if not run_model(model, manifest):
                    raise RuntimeError(f'{model} stopped before attempting all {EXPECTED}; inspect its run.log')
            manifest['state'] = 'finished'
            manifest['finished_utc'] = now()
            write_json(path, manifest)
            announce('All queued Task1 models attempted')
        except Exception as error:
            manifest['state'] = 'stopped'
            manifest['error'] = f'{type(error).__name__}: {error}'
            write_json(path, manifest)
            announce(manifest['error'])
            raise


def start_parallel() -> None:
    """Run the three new models beside the already running Hunyuan scorer."""
    QUEUE.mkdir(parents=True, exist_ok=True)
    with (QUEUE / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = QUEUE / 'manifest.json'
        manifest = read_json(path) if path.is_file() else {'created_utc': now(), 'models': {}}
        if manifest.get('state') == 'parallel_running':
            raise RuntimeError('Parallel Task1 workers already launched')
        # Validate every source before launching any expensive GPU process.
        for model in MODELS:
            video_root, output = model_paths(model)
            if not video_root.is_dir() or output.exists():
                raise RuntimeError(f'{model} source missing or output already exists: {video_root}, {output}')
            preflight = subprocess.run(command(model, output, True), capture_output=True,
                                      text=True, check=False)
            if preflight.returncode or 'available 1498/1498, missing 0, invalid 0' not in preflight.stdout:
                raise RuntimeError(f'{model} preflight failed: {preflight.stdout} {preflight.stderr}')
        for model in MODELS:
            video_root, output = model_paths(model)
            output.mkdir(parents=True)
            log = output / 'run.log'
            with log.open('ab') as stream:
                child = subprocess.Popen(command(model, output), stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            (output / 'pid').write_text(f'{child.pid}\n', encoding='ascii')
            manifest['models'][model] = {'state': 'running', 'started_utc': now(),
                                         'video_root': str(video_root), 'output': str(output),
                                         'log': str(log), 'pid': child.pid}
            write_json(path, manifest)
            announce(f'{model} Task1 started concurrently: pid={child.pid}, output={output}')
        manifest['state'] = 'parallel_running'
        manifest['parallel_started_utc'] = now()
        write_json(path, manifest)


def status() -> None:
    done, total = progress(HUNYUAN / 'run.log')
    scored, failed = result_counts(HUNYUAN / 'results.jsonl')
    print(f'hunyuan alive={running(HUNYUAN / "pid")} attempted={done}/{total or EXPECTED} scored={scored} failed={failed}')
    path = QUEUE / 'manifest.json'
    manifest = read_json(path) if path.is_file() else {}
    print(f'queue state={manifest.get("state", "not_started")}')
    for model in MODELS:
        _, output = model_paths(model)
        done, total = progress(output / 'run.log')
        scored, failed = result_counts(output / 'results.jsonl')
        state = manifest.get('models', {}).get(model, {}).get('state', 'pending')
        alive = running(output / 'pid')
        if state == 'running' and not alive:
            content = (output / 'run.log').read_text(encoding='utf-8', errors='replace') if (output / 'run.log').exists() else ''
            state = 'attempted_all' if done == total == EXPECTED and 'RESULTS ' in content else 'stopped_early'
        print(f'{model} alive={alive} state={state} attempted={done}/{total or EXPECTED} scored={scored} failed={failed}')
    if manifest.get('error'):
        print('queue error:', manifest['error'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'start-parallel', 'status'))
    args = parser.parse_args()
    if args.action == 'run':
        run_queue()
    elif args.action == 'start-parallel':
        start_parallel()
    else:
        status()


if __name__ == '__main__':
    main()
