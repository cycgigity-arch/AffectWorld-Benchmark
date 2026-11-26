#!/usr/bin/env python3
"""Evaluate only missing Task2/Task3 keys of the four original EWM models."""

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
import threading
import time


ROOT = Path('/root/autodl-tmp')
BUNDLE = ROOT / 'ewm_eval_local_cf223'
SOURCE = BUNDLE / 'private_annotations_1498_normalized.json'
HISTORIC = ROOT / 'ewm_eval_local/eval_runs'
TARGET = ROOT / 'legacy_task23_resume_20261108'
ROOTS = {
    'cog': ROOT / 'ewm_generated_cog/cogvideox_i2v_state0_context_6s',
    'cosmos': ROOT / 'ewm_generated_cosmos/visual_only_6s',
    'ltx': ROOT / 'ewm_generated_LTX/ltx_i2v_state0_context_6s',
    'wan': ROOT / 'ewm_generated_wan/vace_visual_only_6s',
}
TASKS = ('task2', 'task3')
SOURCE_SHA = 'aa09b47e6cdc17fd0729a5f0a9dae057f77e780c2c93895733284364fd7c5aa3'
STOP_PHRASES = ('access denied', 'arrearage', 'overdue-payment', '余额')


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def classify(path: Path, rows: list[dict]) -> str | None:
    name = path.parent.name.lower()
    for model in ROOTS:
        if model in name:
            return model
    for row in rows:
        source = str(row.get('video_file', '')).lower()
        for model in ROOTS:
            if f'ewm_generated_{model}' in source:
                return model
    return None


def historic_success() -> dict[tuple[str, str], set[str]]:
    result = {(model, task): set() for model in ROOTS for task in TASKS}
    for task in TASKS:
        for path in HISTORIC.rglob(f'{task}_results.json'):
            rows = load(path)
            if not isinstance(rows, list):
                continue
            model = classify(path, rows)
            if model is None:
                continue
            result[(model, task)].update(
                row['video_key'] for row in rows
                if isinstance(row, dict) and 'video_key' in row and 'error' not in row)
    return result


def prepare() -> None:
    if TARGET.exists():
        raise SystemExit(f'Target already exists: {TARGET}')
    if sha(SOURCE) != SOURCE_SHA:
        raise SystemExit('Original annotation hash changed')
    sys.path.insert(0, str(BUNDLE))
    import ewm_eval_local as adapter
    annotations = load(SOURCE)
    if len(annotations) != 1498:
        raise SystemExit('Expected 1498 canonical annotations')
    keys = {row['video_key'] for row in annotations}
    if len(keys) != 1498:
        raise SystemExit('Duplicate annotation keys')
    old = historic_success()
    TARGET.mkdir(parents=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'source': str(SOURCE), 'source_sha256': SOURCE_SHA,
                'models': {}}
    for model, root in ROOTS.items():
        if not root.is_dir():
            raise SystemExit(f'Missing video root: {root}')
        manifest['models'][model] = {'video_root': str(root), 'tasks': {}}
        for task in TASKS:
            prior = old[(model, task)]
            if not prior <= keys:
                raise SystemExit(f'{model} {task}: historic results include foreign keys')
            selected = [row for row in annotations if row['video_key'] not in prior]
            folder = TARGET / model
            folder.mkdir(exist_ok=True)
            subset = folder / f'{task}_missing_annotations.json'
            save(subset, selected)
            work = folder / task
            args = argparse.Namespace(annotations=subset, orig_dir=root, cf_dir=None,
                                      tasks=[task], limit=0, label_overrides=None,
                                      output_dir=work)
            _, report = adapter.prepare(args)
            if report['errors'] or report['ready'] != len(selected):
                raise SystemExit(f'{model} {task} preflight failed: {report["ready"]}/{len(selected)}')
            manifest['models'][model]['tasks'][task] = {
                'historic_success': len(prior), 'selected': len(selected),
                'subset': str(subset), 'subset_sha256': sha(subset), 'work': str(work)}
            print(f'{model} {task}: existing={len(prior)}, to_test={len(selected)}, ready={report["ready"]}', flush=True)
    save(TARGET / 'manifest.json', manifest)
    print('OFFLINE PREFLIGHT PASS:', TARGET, flush=True)


def validate() -> dict:
    manifest = load(TARGET / 'manifest.json')
    if sha(SOURCE) != manifest['source_sha256']:
        raise SystemExit('Annotation hash changed')
    for model, detail in manifest['models'].items():
        for task, state in detail['tasks'].items():
            if sha(Path(state['subset'])) != state['subset_sha256']:
                raise SystemExit(f'{model} {task} subset hash changed')
            preflight = load(Path(state['work']) / 'preflight.json')
            if preflight['errors'] or preflight['ready'] != state['selected']:
                raise SystemExit(f'{model} {task} preflight no longer valid')
    return manifest


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.split(') ', 1)[1][0] != 'Z'
    except (OSError, IndexError):
        return False


def counts(work: Path, expected: int) -> dict:
    path = work / 'task2_results.json' if work.name == 'task2' else work / 'task3_results.json'
    rows = load(path) if path.is_file() else []
    good = {row['video_key'] for row in rows if isinstance(row, dict) and 'video_key' in row and 'error' not in row}
    errors = {row['video_key'] for row in rows if isinstance(row, dict) and 'video_key' in row and 'error' in row} - good
    return {'success': len(good), 'errors': len(errors),
            'unattempted': expected - len(good) - len(errors),
            'remaining': expected - len(good)}


def billing_watch(work: Path, task: str, started: float, stop: threading.Event) -> None:
    path = work / f'{task}_results.json'
    while not stop.wait(3):
        try:
            if path.stat().st_mtime < started:
                continue
            rows = load(path)
            denied = sum(any(term in str(row.get('error', '')).lower() for term in STOP_PHRASES)
                         for row in rows if isinstance(row, dict) and 'error' in row)
            if denied >= 3:
                print(f'BILLING_BLOCK: {task} got {denied} balance errors; stopping worker', flush=True)
                os._exit(75)
        except (OSError, ValueError, TypeError):
            continue


def worker(model: str) -> None:
    manifest = validate()
    sys.path.insert(0, str(BUNDLE))
    import ewm_eval_local as adapter
    detail = manifest['models'][model]
    for task in TASKS:
        state = detail['tasks'][task]
        work = Path(state['work'])
        before = counts(work, state['selected'])
        print(f'{model} {task} before={before}', flush=True)
        if before['remaining'] == 0:
            continue
        stop = threading.Event()
        watch = threading.Thread(target=billing_watch,
                                 args=(work, task, time.time(), stop), daemon=True)
        watch.start()
        try:
            adapter.run(argparse.Namespace(tasks=[task], cf_dir=None), work)
        except Exception as error:
            print(f'{model} {task} ended with {type(error).__name__}: {error}', flush=True)
        finally:
            stop.set()
            watch.join(timeout=5)
        print(f'{model} {task} after={counts(work, state["selected"])}', flush=True)
    print(f'{model} Task2/Task3 worker finished', flush=True)


def start() -> None:
    manifest = validate()
    launch_path = TARGET / 'launch_manifest.json'
    if launch_path.exists() and any(alive(x.get('pid')) for x in load(launch_path)['models']):
        raise SystemExit('Legacy Task2/Task3 workers already running')
    vlm = getpass.getpass('VLM API key (hidden): ').strip()
    llm = getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm or not llm:
        raise SystemExit('Both API keys are required')
    env = os.environ.copy()
    env['EWM_VLM_API_KEY'] = vlm
    env['EWM_LLM_API_KEY'] = llm
    del vlm, llm
    workers = []
    for model, detail in manifest['models'].items():
        if all(counts(Path(detail['tasks'][task]['work']), detail['tasks'][task]['selected'])['remaining'] == 0
               for task in TASKS):
            continue
        log = TARGET / model / 'resume.log'
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--model', model]
        with log.open('ab') as stream:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        workers.append({'model': model, 'pid': child.pid, 'log': str(log)})
        print(f'started {model} pid={child.pid}', flush=True)
    save(launch_path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'models': workers})


def status() -> None:
    manifest = validate()
    launch_path = TARGET / 'launch_manifest.json'
    running = {x['model']: alive(x.get('pid')) for x in load(launch_path)['models']} if launch_path.exists() else {}
    for model, detail in manifest['models'].items():
        parts = []
        for task in TASKS:
            state = detail['tasks'][task]
            current = counts(Path(state['work']), state['selected'])
            parts.append(f'{task}={state["historic_success"] + current["success"]}/1498 '
                         f'new={current["success"]}/{state["selected"]} errors={current["errors"]}')
        print(f'{model} alive={running.get(model, False)} ' + ' '.join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'start', 'status', 'worker'))
    parser.add_argument('--model', choices=ROOTS)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'start':
        start()
    elif args.command == 'status':
        status()
    else:
        if not args.model:
            parser.error('--model required')
        worker(args.model)


if __name__ == '__main__':
    main()
