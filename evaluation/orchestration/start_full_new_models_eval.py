#!/usr/bin/env python3
"""Prompt for API keys, then start full SVD/EasyAnimate Task 2/3/CF scoring.

Secrets are passed only through child process environments. They are never written
to source files, shell history, logs, or the launch manifest.
"""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MODELS = ('svd', 'easyanimate')
EXPECTED_ANNOTATIONS = 1498
EXPECTED_CF = 223
REFERENCE_EVALUATOR_HASHES = {
    'ewm_eval_local.py': '6ab38cf7c187d72977fc52ea87188b2446bdb1ff65b5ad6218a7ceea77991725',
    'run_all.py': '5e8d695221b34a07322703de06607879f63d142bb4dc5a508a76bcfc27f08e27',
    'eval_task2.py': '82d1d1c652d14e8e9778baba854c9472a57c5a659dcbd28dab53da79c2832342',
    'eval_task3.py': 'bf5d6e8618fbf5579ffa78e3f8594e5afdbb3f6fd7189f8e2ca340a4127caef9',
    'eval_cf.py': '9e7dfbc155bb115eb98cac403150202ed5b3b0e901b64ad5b2b73b6c43aa869e',
}


def verify_evaluator_code(folder):
    """Compare cloud evaluator files to the checked five-model source bundle."""
    mismatches = []
    for filename, expected in REFERENCE_EVALUATOR_HASHES.items():
        path = folder / filename
        if not path.is_file():
            mismatches.append(filename + ': missing')
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(filename + ': differs from five-model source bundle')
    return mismatches


def preflight(root, model):
    path = root / f'{model}_preflight_20261027' / 'preflight.json'
    if not path.is_file():
        raise ValueError(f'Missing preflight report: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    orig = (root / f'ewm_generated_{model}' / 'normal').resolve()
    cf = (root / f'ewm_generated_{model}' / 'counterfactual').resolve()
    if (data.get('total_annotations'), data.get('selected'), data.get('ready'), data.get('cf_selected')) != (1498, 1498, 1498, 223):
        raise ValueError(f'{model}: preflight counts are not 1498/223')
    if data.get('errors') or data.get('api_called') is not False:
        raise ValueError(f'{model}: preflight has errors or is not an offline check')
    if data.get('tasks') != ['task2', 'task3', 'cf']:
        raise ValueError(f'{model}: preflight task list differs')
    if Path(data.get('orig_dir', '')).resolve() != orig or Path(data.get('cf_dir', '')).resolve() != cf:
        raise ValueError(f'{model}: preflight paths differ from current model paths')
    if not orig.is_dir() or not cf.is_dir():
        raise ValueError(f'{model}: video directories are missing')
    return path, data, orig, cf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--python', type=Path, default=Path(sys.executable), help='Python environment containing openai and tqdm')
    args = parser.parse_args()
    root = args.root.resolve()
    adapter = root / 'ewm_eval_local_cf223' / 'ewm_eval_local.py'
    if not adapter.is_file():
        parser.error('Evaluator not found: ' + str(adapter))
    mismatches = verify_evaluator_code(adapter.parent)
    if mismatches:
        parser.error('Evaluator version is not verified: ' + '; '.join(mismatches) +
                     '. Do not launch until the code difference is reviewed.')
    for key, expected in (('EWM_VLM_MODEL', 'qwen3.8-max'), ('EWM_LLM_MODEL', 'gpt-5.6-sol')):
        override = os.environ.get(key)
        if override and override != expected:
            parser.error(f'{key}={override!r} differs from the paper model {expected!r}')
    if len(args.models) != len(set(args.models)):
        parser.error('Duplicate model name')
    checked = {}
    for model in args.models:
        try:
            checked[model] = preflight(root, model)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    package_check = subprocess.run([str(args.python), '-c', 'import openai, tqdm'],
                                   capture_output=True, text=True)
    if package_check.returncode:
        parser.error('Selected Python cannot import openai and tqdm; install requirements in that environment')

    print('Preflight passed for:', ', '.join(args.models), flush=True)
    print('Evaluator files match the five-model source bundle.', flush=True)
    vlm_key = getpass.getpass('VLM API key (hidden): ').strip()
    llm_key = getpass.getpass('LLM API key (hidden): ').strip()
    if not vlm_key or not llm_key:
        parser.error('Both API keys are required')
    child_env = os.environ.copy()
    child_env['EWM_VLM_API_KEY'] = vlm_key
    child_env['EWM_LLM_API_KEY'] = llm_key
    del vlm_key, llm_key

    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    parent = root / 'new_model_eval_runs' / tag
    parent.mkdir(parents=True, exist_ok=False)
    launches = []
    for model in args.models:
        report_path, report, orig, cf = checked[model]
        output_dir = parent / model
        log_path = parent / (model + '.log')
        command = [str(args.python), '-u', str(adapter), 'run',
                   '--orig-dir', str(orig), '--cf-dir', str(cf),
                   '--tasks', 'task2', 'task3', 'cf', '--limit', '0',
                   '--output-dir', str(output_dir)]
        with log_path.open('xb') as log:
            child = subprocess.Popen(command, cwd=root, env=child_env,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        launches.append(dict(model=model, pid=child.pid, log=str(log_path),
                             results=str(output_dir), preflight=str(report_path),
                             annotations_sha256=report.get('source_sha256'),
                             expected_normal=EXPECTED_ANNOTATIONS, expected_counterfactual=EXPECTED_CF))
        print(f'{model}: started PID {child.pid}; log {log_path}', flush=True)
    del child_env
    (parent / 'launch_manifest.json').write_text(json.dumps(dict(created_utc=tag, launches=launches),
                                                              ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Full scoring launched. Follow logs with: tail -f ' + str(parent / '*.log'), flush=True)
    print('Launch manifest: ' + str(parent / 'launch_manifest.json'), flush=True)


if __name__ == '__main__':
    main()
