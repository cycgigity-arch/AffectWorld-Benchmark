#!/usr/bin/env python3
"""Move the running SVD/EasyAnimate evaluation to disjoint, resumable shards.

The scorer files and settings are unchanged.  ``prepare`` calls only the
evaluator's offline preflight; ``cutover`` checkpoints the live runs, hands
their API environment to gated workers, stops the two serial processes, and
then releases the workers.  Credentials are never written or printed.
"""

import argparse
from datetime import datetime, timezone
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import start_full_new_models_eval as original


MODEL_NAMES = ("svd", "easyanimate")
TASKS = ["task2", "task3", "cf"]
ENV_NAMES = ("EWM_VLM_API_KEY", "EWM_LLM_API_KEY", "EWM_VLM_MODEL",
             "EWM_LLM_MODEL", "EWM_VLM_BASE_URL", "EWM_LLM_BASE_URL")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def adapter_from(path):
    spec = importlib.util.spec_from_file_location("ewm_eval_local_parallel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pid_alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().split()[2] != "Z"


def check_source_pid(pid, output):
    if not pid_alive(pid):
        raise ValueError(f"Original process {pid} is no longer running")
    parts = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    command = [p.decode("utf-8", "replace") for p in parts if p]
    if ("ewm_eval_local.py" not in " ".join(command)
            or "--output-dir" not in command
            or command[command.index("--output-dir") + 1] != str(output)):
        raise ValueError(f"PID {pid} does not match the original evaluation")
    return os.readlink(f"/proc/{pid}/exe")


def check_worker_pid(pid, output):
    if not pid_alive(pid):
        raise ValueError(f"Parallel worker {pid} is no longer running")
    parts = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    command = [p.decode("utf-8", "replace") for p in parts if p]
    if ("parallel_new_models_eval.py" not in " ".join(command)
            or "worker" not in command or "--work" not in command
            or command[command.index("--work") + 1] != str(output)):
        raise ValueError(f"PID {pid} does not match the active parallel worker")
    return os.readlink(f"/proc/{pid}/exe")


def private_environment(pid):
    entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    values = {}
    for entry in entries:
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            name = key.decode("utf-8", "replace")
            if name in ENV_NAMES:
                values[name] = value.decode("utf-8", "replace")
    if not values.get("EWM_VLM_API_KEY") or not values.get("EWM_LLM_API_KEY"):
        raise ValueError(f"PID {pid} lacks an API key in its private environment")
    return values


def valid_snapshot(folder):
    """Read a complete JSON checkpoint even if the old writer is active."""
    folder = Path(folder)
    for attempt in range(12):
        try:
            rows = read_json(folder / "task2_results.json")
            caches = {}
            for name in ("descriptions.json", "judge.json"):
                blob = (folder / ".ewh_cache" / name).read_bytes()
                if not isinstance(json.loads(blob), dict):
                    raise ValueError(f"Unexpected cache format: {name}")
                caches[name] = blob
            if not isinstance(rows, list):
                raise ValueError("Task 2 checkpoint is not a list")
            return rows, caches
        except (OSError, ValueError, json.JSONDecodeError):
            if attempt == 11:
                raise
            time.sleep(0.2)


def success_count(snapshot):
    return sum("error" not in row and "Score_T2" in row for row in snapshot[0])


def merge_snapshots(snapshots):
    rows = []
    seen = set()
    cache_maps = {"descriptions.json": {}, "judge.json": {}}
    for records, caches in snapshots:
        for row in records:
            key = row.get("video_key")
            if not isinstance(key, str) or key in seen:
                raise ValueError(f"Duplicate or invalid checkpoint key: {key}")
            seen.add(key)
            rows.append(row)
        for name, blob in caches.items():
            current = json.loads(blob)
            merged = cache_maps[name]
            for key, value in current.items():
                if key in merged and merged[key] != value:
                    raise ValueError(f"Conflicting cache entry: {name} {key}")
                merged[key] = value
    merged_caches = {name: (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
                     for name, value in cache_maps.items()}
    return rows, merged_caches


def seed_shards(manifest, model, snapshot):
    rows, caches = snapshot
    success = {}
    for row in rows:
        key = row.get("video_key")
        if not isinstance(key, str):
            raise ValueError("Checkpoint has a row without video_key")
        if "error" not in row and "Score_T2" in row:
            if key in success:
                raise ValueError(f"Duplicate completed Task 2 key: {key}")
            success[key] = row
    selected = set()
    for shard in manifest["shards"]:
        if shard["model"] != model:
            continue
        keys = [row["video_key"] for row in read_json(shard["annotations"])]
        selected.update(keys)
        output = Path(shard["output"])
        write_json(output / "task2_results.json",
                   [success[key] for key in keys if key in success])
        cache_dir = output / ".ewh_cache"
        cache_dir.mkdir(exist_ok=True)
        for name, blob in caches.items():
            (cache_dir / name).write_bytes(blob)
    if not set(success) <= selected:
        raise ValueError(f"{model}: completed records are outside the shard cohort")
    return len(success)


def prepare(args):
    root = args.root.resolve()
    source_parallel = args.source_parallel is not None
    source = (args.source_parallel if source_parallel else args.source_run).resolve()
    adapter_path = root / "ewm_eval_local_cf223" / "ewm_eval_local.py"
    mismatches = original.verify_evaluator_code(adapter_path.parent)
    if mismatches:
        raise ValueError("Evaluator code differs: " + "; ".join(mismatches))
    adapter = adapter_from(adapter_path)
    source_launch = read_json(source / "launch_manifest.json")
    if source_parallel:
        if source_launch.get("state") != "running":
            raise ValueError("Source parallel run is not active")
        launches = {model: [row for row in source_launch["shards"]
                            if row["model"] == model] for model in MODEL_NAMES}
        if any(not launches[model] for model in MODEL_NAMES):
            raise ValueError("Source parallel run does not contain both models")
    else:
        launches = {row["model"]: row for row in source_launch["launches"]}
        if set(launches) != set(MODEL_NAMES):
            raise ValueError("Original launch manifest does not contain both models")
    annotations = root / "ewm_eval_local_cf223" / "private_annotations_1498_normalized.json"
    raw = read_json(annotations)
    if not isinstance(raw, list) or len(raw) != 1498:
        raise ValueError("Expected exactly 1498 source annotations")
    if len({row["video_key"] for row in raw}) != len(raw):
        raise ValueError("Duplicate source annotation video_key")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = root / "parallel_eval_runs" / tag
    target.mkdir(parents=True, exist_ok=False)
    manifest = {"created_utc": tag, "state": "prepared", "source_run": str(source),
                "source_kind": "parallel" if source_parallel else "serial",
                "target": str(target), "adapter": str(adapter_path),
                "source_annotations": str(annotations), "workers_per_model": args.workers_per_model,
                "original": {}, "shards": []}
    for model in MODEL_NAMES:
        report, info, orig, cf = original.preflight(root, model)
        if info["source_sha256"] != adapter.file_hash(annotations):
            raise ValueError(f"{model}: source annotations changed")
        source_inputs = ([row for shard in launches[model]
                          for row in read_json(Path(shard["output"]) / "eval_inputs.json")]
                         if source_parallel else read_json(source / model / "eval_inputs.json"))
        if len(source_inputs) != 1498 or len({row["video_key"] for row in source_inputs}) != 1498:
            raise ValueError(f"{model}: source run is not the full 1498-key cohort")
        cf_keys = {row["video_key"] for row in source_inputs
                   if isinstance(row.get("counterfactual"), dict)}
        if len(cf_keys) != 223:
            raise ValueError(f"{model}: original run does not contain 223 CF keys")
        if source_parallel:
            workers = [{"pid": int(shard["pid"]), "output": shard["output"]}
                       for shard in launches[model]]
            for worker in workers:
                check_worker_pid(worker["pid"], worker["output"])
        else:
            pid = int(launches[model]["pid"])
            check_source_pid(pid, source / model)
            workers = [{"pid": pid, "output": str(source / model)}]
        manifest["original"][model] = {"workers": workers,
                                         "orig_dir": str(orig), "cf_dir": str(cf)}
        for index in range(args.workers_per_model):
            subset = raw[index::args.workers_per_model]
            ann_path = target / f"{model}_shard{index}_annotations.json"
            write_json(ann_path, subset)
            output = target / model / f"shard{index}"
            output.parent.mkdir(exist_ok=True)
            local_args = argparse.Namespace(annotations=ann_path, orig_dir=orig,
                cf_dir=cf, tasks=TASKS[:], limit=0, label_overrides=None,
                output_dir=output)
            work, preflight = adapter.prepare(local_args)
            expected_cf = sum(row["video_key"] in cf_keys for row in subset)
            if (preflight["errors"] or preflight["selected"] != len(subset)
                    or preflight["ready"] != len(subset)
                    or preflight["cf_selected"] != expected_cf):
                raise ValueError(f"{model} shard{index}: preflight failed")
            worker_config = {"model": model, "source": str(source), "output": str(work),
                             "orig_dir": str(orig), "cf_dir": str(cf)}
            write_json(work / "worker_config.json", worker_config)
            manifest["shards"].append({"model": model, "index": index,
                "annotations": str(ann_path), "output": str(work),
                "log": str(target / f"{model}_shard{index}.log"),
                "expected_normal": len(subset), "expected_cf": expected_cf})
    write_json(target / "launch_manifest.json", manifest)
    print(f"PREPARED {target}: {len(manifest['shards'])} disjoint shards; no API calls", flush=True)
    return target


def wait_for_ready(processes, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(child.poll() is not None for child, _ in processes):
            raise RuntimeError("A gated worker exited before cutover")
        if all((Path(work) / ".worker_ready").is_file() for _, work in processes):
            return
        time.sleep(0.2)
    raise TimeoutError("Gated workers did not become ready")


def cutover(args):
    target = args.target.resolve()
    manifest_path = target / "launch_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["state"] != "prepared" or Path(manifest["target"]) != target:
        raise ValueError("Target is not a prepared, unused parallel run")
    adapter_path = Path(manifest["adapter"])
    if original.verify_evaluator_code(adapter_path.parent):
        raise ValueError("Evaluator code changed after preparation")
    credentials = {}
    python_paths = {}
    before = {}
    before_parts = {}
    source_workers = {}
    parallel_source = manifest.get("source_kind") == "parallel"
    for model in MODEL_NAMES:
        entry = manifest["original"][model]
        source_workers[model] = entry["workers"]
        parts = []
        for worker in entry["workers"]:
            pid = worker["pid"]
            python = (check_worker_pid(pid, worker["output"]) if parallel_source
                      else check_source_pid(pid, worker["output"]))
            env = private_environment(pid)
            if model in python_paths and python_paths[model] != python:
                raise ValueError(f"{model}: source workers use different Python environments")
            if model in credentials and credentials[model] != env:
                raise ValueError(f"{model}: source workers use different API settings")
            python_paths[model] = python
            credentials[model] = env
            parts.append(valid_snapshot(worker["output"]))
        before_parts[model] = parts
        before[model] = merge_snapshots(parts)
        seed_shards(manifest, model, before[model])
    processes = []
    try:
        for shard in manifest["shards"]:
            model = shard["model"]
            env = os.environ.copy()
            env.update(credentials[model])
            with Path(shard["log"]).open("xb") as log:
                child = subprocess.Popen([python_paths[model], "-u", str(Path(__file__).resolve()),
                    "worker", "--work", shard["output"], "--gate", str(target / ".start")],
                    cwd=target, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            shard["pid"] = child.pid
            processes.append((child, shard["output"]))
        wait_for_ready(processes)
    except BaseException:
        for child, _ in processes:
            if child.poll() is None:
                child.terminate()
        raise
    source_pids = [worker["pid"] for workers in source_workers.values() for worker in workers]
    for pid in source_pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if all(not pid_alive(pid) for pid in source_pids):
            break
        time.sleep(0.2)
    else:
        for child, _ in processes:
            if child.poll() is None:
                child.terminate()
        raise RuntimeError("Original process did not exit; workers remain gated")
    for model in MODEL_NAMES:
        after_parts = []
        for worker, old_part in zip(source_workers[model], before_parts[model]):
            try:
                after_parts.append(valid_snapshot(worker["output"]))
            except (OSError, ValueError, json.JSONDecodeError):
                after_parts.append(old_part)
        after = merge_snapshots(after_parts)
        snapshot = after if success_count(after) >= success_count(before[model]) else before[model]
        manifest["original"][model]["completed_task2_seeded"] = seed_shards(manifest, model, snapshot)
    manifest["state"] = "running"
    manifest["cutover_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    (target / ".start").touch(exist_ok=False)
    if parallel_source:
        old_manifest_path = Path(manifest["source_run"]) / "launch_manifest.json"
        old_manifest = read_json(old_manifest_path)
        old_manifest["state"] = "superseded"
        old_manifest["superseded_by"] = str(target)
        write_json(old_manifest_path, old_manifest)
    print(f"STARTED {len(processes)} workers; manifest: {manifest_path}", flush=True)
    for shard in manifest["shards"]:
        print(f"{shard['model']} shard{shard['index']}: PID {shard['pid']} "
              f"seeded={manifest['original'][shard['model']]['completed_task2_seeded']} "
              f"log={shard['log']}", flush=True)


def worker(args):
    work = args.work.resolve()
    config = read_json(work / "worker_config.json")
    if args.resume:
        if not args.gate.is_file():
            raise ValueError("The original cutover gate is missing")
        (work / f".resume_ready_{os.getpid()}").touch(exist_ok=False)
        print(f"RESUME {config['model']} {work}", flush=True)
    else:
        (work / ".worker_ready").touch(exist_ok=False)
        print(f"READY {config['model']} {work}", flush=True)
        deadline = time.monotonic() + 1800
        while not args.gate.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("Cutover gate was never released")
            time.sleep(0.2)
    print("GATE OPEN; scoring started", flush=True)
    adapter = adapter_from(Path(config["source"]).parent.parent /
                           "ewm_eval_local_cf223" / "ewm_eval_local.py")
    selected_tasks = ["cf"] if args.cf_only else TASKS[:]
    run_args = argparse.Namespace(tasks=selected_tasks[:], cf_dir=Path(config["cf_dir"]))
    failures = []
    remaining = selected_tasks[:]
    while remaining:
        run_args.tasks = remaining
        try:
            adapter.run(run_args, work)
            break
        except ValueError as exc:
            match = re.fullmatch(r"(task2|task3|cf): (\d+) samples failed;.*", str(exc))
            if not match or match.group(1) not in remaining:
                raise
            failures.append((match.group(1), int(match.group(2))))
            print(f"STAGE_ERRORS {match.group(1)} {match.group(2)}; continuing later stages",
                  flush=True)
            remaining = remaining[remaining.index(match.group(1)) + 1:]
    print("WORKER_DONE", config["model"], "failures", failures, flush=True)
    return 1 if failures else 0


def live_workers(target):
    """Find workers for this run even if the manifest contains stale PIDs."""
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = [part.decode("utf-8", "replace") for part in
                       cmdline.read_bytes().split(b"\0") if part]
            if ("parallel_new_models_eval.py" in " ".join(command)
                    and "worker" in command and "--work" in command):
                work = Path(command[command.index("--work") + 1]).resolve()
                if work.is_relative_to(target):
                    found.append(int(cmdline.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def validate_resume_target(target, manifest):
    if (manifest.get("state") != "running"
            or Path(manifest.get("target", "")).resolve() != target
            or manifest.get("workers_per_model") != 4):
        raise ValueError("Expected the active four-shards-per-model run")
    if not (target / ".start").is_file():
        raise ValueError("The original cutover gate is missing")
    adapter = Path(manifest["adapter"])
    mismatches = original.verify_evaluator_code(adapter.parent)
    if mismatches:
        raise ValueError("Evaluator code changed: " + "; ".join(mismatches))
    source_keys = [row["video_key"] for row in read_json(manifest["source_annotations"])]
    if len(source_keys) != 1498 or len(set(source_keys)) != 1498:
        raise ValueError("Source annotation cohort is not 1498 unique keys")
    if len(manifest["shards"]) != 8:
        raise ValueError("Expected exactly eight shards")
    for model in MODEL_NAMES:
        shards = [s for s in manifest["shards"] if s["model"] == model]
        if len(shards) != 4 or {s["index"] for s in shards} != set(range(4)):
            raise ValueError(f"{model}: expected four distinct shards")
        all_keys, all_cf = [], []
        for shard in shards:
            work = Path(shard["output"])
            inputs = read_json(work / "eval_inputs.json")
            keys = [row["video_key"] for row in inputs]
            cf_keys = [row["video_key"] for row in inputs
                       if isinstance(row.get("counterfactual"), dict)]
            annotations = [row["video_key"] for row in read_json(shard["annotations"])]
            if (keys != annotations or len(keys) != shard["expected_normal"]
                    or len(cf_keys) != shard["expected_cf"]):
                raise ValueError(f"{model} shard{shard['index']}: inputs changed")
            all_keys.extend(keys)
            all_cf.extend(cf_keys)
            valid_keys = {"task2": set(keys), "task3": set(keys), "cf": set(cf_keys)}
            for task in TASKS:
                path = work / f"{task}_results.json"
                if not path.is_file():
                    continue
                rows = read_json(path)
                result_keys = [row["video_key"] for row in rows]
                if (not isinstance(rows, list) or len(result_keys) != len(set(result_keys))
                        or not set(result_keys) <= valid_keys[task]):
                    raise ValueError(f"{model} shard{shard['index']}: invalid {task} checkpoint")
            for name in ("descriptions.json", "judge.json"):
                if not isinstance(read_json(work / ".ewh_cache" / name), dict):
                    raise ValueError(f"{model} shard{shard['index']}: invalid {name} cache")
        if (len(all_keys) != 1498 or set(all_keys) != set(source_keys)
                or len(all_cf) != 223 or len(set(all_cf)) != 223):
            raise ValueError(f"{model}: shard cohort is not 1498/223")


def resume(args):
    target = args.target.resolve()
    manifest_path = target / "launch_manifest.json"
    manifest = read_json(manifest_path)
    validate_resume_target(target, manifest)
    active = live_workers(target)
    if active:
        raise ValueError(f"Run already has active workers: {active}; refusing duplicate launch")
    import fcntl
    with (target / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = live_workers(target)
        if active:
            raise ValueError(f"Run already has active workers: {active}; refusing duplicate launch")
        if subprocess.run([sys.executable, "-c", "import openai, tqdm"],
                          capture_output=True).returncode:
            raise ValueError("This Python environment cannot import openai and tqdm")
        print("Checkpoints valid: SVD and EasyAnimate each have 1498 normal / 223 CF items.",
              flush=True)
        vlm_key = getpass.getpass("VLM API key (hidden): ").strip()
        llm_key = getpass.getpass("LLM API key (hidden): ").strip()
        if not vlm_key or not llm_key:
            raise ValueError("Both API keys are required")
        child_env = os.environ.copy()
        for name in ENV_NAMES:
            child_env.pop(name, None)
        child_env["EWM_VLM_API_KEY"] = vlm_key
        child_env["EWM_LLM_API_KEY"] = llm_key
        del vlm_key, llm_key
        processes = []
        try:
            for shard in manifest["shards"]:
                with Path(shard["log"]).open("a", encoding="utf-8") as log:
                    log.write("\n=== RESUME " + datetime.now(timezone.utc).isoformat() + " ===\n")
                    log.flush()
                    child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                        "worker", "--work", shard["output"], "--gate", str(target / ".start"),
                        "--resume"], cwd=target, env=child_env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((child, Path(shard["output"])))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if any(child.poll() is not None for child, _ in processes):
                    raise RuntimeError("A resumed worker exited during startup; inspect shard logs")
                if all((work / f".resume_ready_{child.pid}").is_file()
                       for child, work in processes):
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("Resumed workers did not become ready")
            for shard, (child, _) in zip(manifest["shards"], processes):
                shard["pid"] = child.pid
            manifest.setdefault("resume_history", []).append({
                "resumed_utc": datetime.now(timezone.utc).isoformat(),
                "pids": [child.pid for child, _ in processes]})
            write_json(manifest_path, manifest)
        except BaseException:
            for child, _ in processes:
                if child.poll() is None:
                    child.terminate()
            raise
        finally:
            del child_env
    print(f"RESUMED {len(processes)} workers; manifest: {manifest_path}", flush=True)
    for shard in manifest["shards"]:
        print(f"{shard['model']} shard{shard['index']}: PID {shard['pid']} log={shard['log']}",
              flush=True)


def corrected_cf_inputs(work, cf_root):
    """Use the offline preflight's exact CF file, without changing score logic."""
    rows = read_json(work / "eval_inputs.json")
    corrected = 0
    for row in rows:
        cf = row.get("counterfactual")
        if not isinstance(cf, dict):
            continue
        mapped = Path(cf["_video_file"]).resolve()
        relative = mapped.relative_to(cf_root).as_posix()
        if not mapped.is_file() or not relative.lower().endswith(".mp4"):
            raise ValueError(f"Missing mapped CF video: {mapped}")
        existing = cf.get("video_key")
        if existing not in (None, relative):
            raise ValueError(f"Conflicting CF video override for {row['video_key']}")
        cf["video_key"] = relative
        corrected += 1
    return rows, corrected


def repair_cf(args):
    """Correct CF paths, then retry CF alone or all unfinished stages."""
    target = args.target.resolve()
    manifest_path = target / "launch_manifest.json"
    manifest = read_json(manifest_path)
    validate_resume_target(target, manifest)
    if live_workers(target):
        raise ValueError("Run still has active workers; refusing duplicate launch")
    import fcntl
    with (target / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if live_workers(target):
            raise ValueError("Run still has active workers; refusing duplicate launch")
        if subprocess.run([sys.executable, "-c", "import openai, tqdm"],
                          capture_output=True).returncode:
            raise ValueError("This Python environment cannot import openai and tqdm")
        prepared = []
        by_model = {model: 0 for model in MODEL_NAMES}
        for shard in manifest["shards"]:
            work = Path(shard["output"])
            config = read_json(work / "worker_config.json")
            cf_root = Path(config["cf_dir"]).resolve()
            rows, count = corrected_cf_inputs(work, cf_root)
            if count != shard["expected_cf"]:
                raise ValueError(f"{shard['model']} shard{shard['index']}: CF count changed")
            previous = read_json(work / "cf_results.json")
            if len(previous) > count:
                raise ValueError(f"{shard['model']} shard{shard['index']}: too many CF results")
            by_model[shard["model"]] += count
            prepared.append((shard, work, rows))
        if any(value != 223 for value in by_model.values()):
            raise ValueError("Expected exactly 223 CF items per model")
        mode = "retry failed Task 2/3 and CF" if args.retry_normal else "retry CF only"
        print(f"CF mapping verified: 223 real video files per model; {mode}.", flush=True)
        vlm_key = getpass.getpass("VLM API key (hidden): ").strip()
        llm_key = getpass.getpass("LLM API key (hidden): ").strip()
        if not vlm_key or not llm_key:
            raise ValueError("Both API keys are required")
        child_env = os.environ.copy()
        for name in ENV_NAMES:
            child_env.pop(name, None)
        child_env["EWM_VLM_API_KEY"] = vlm_key
        child_env["EWM_LLM_API_KEY"] = llm_key
        del vlm_key, llm_key
        processes = []
        try:
            for shard, work, rows in prepared:
                backup = work / "eval_inputs.before_cf_path_fix.json"
                if not backup.exists():
                    with backup.open("xb") as stream:
                        stream.write((work / "eval_inputs.json").read_bytes())
                results_backup = work / "cf_results.before_path_fix.json"
                if not results_backup.exists():
                    with results_backup.open("xb") as stream:
                        stream.write((work / "cf_results.json").read_bytes())
                if args.retry_normal:
                    for task in ("task2", "task3"):
                        baseline = work / f"{task}_results.before_retry.json"
                        if not baseline.exists():
                            with baseline.open("xb") as stream:
                                stream.write((work / f"{task}_results.json").read_bytes())
                write_json(work / "eval_inputs.json", rows)
                prefix = "retry_all" if args.retry_normal else "cf_repair"
                log_path = target / f"{prefix}_{shard['model']}_shard{shard['index']}.log"
                with log_path.open("a", encoding="utf-8") as log:
                    log.write("\n=== " + mode.upper() + " " + datetime.now(timezone.utc).isoformat()
                              + " ===\n")
                    log.flush()
                    command = [sys.executable, "-u", str(Path(__file__).resolve()),
                        "worker", "--work", str(work), "--gate", str(target / ".start"),
                        "--resume"]
                    if not args.retry_normal:
                        command.append("--cf-only")
                    child = subprocess.Popen(command, cwd=target, env=child_env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                processes.append((child, work, log_path))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if any(child.poll() is not None for child, _, _ in processes):
                    raise RuntimeError("A CF worker exited during startup; inspect CF repair logs")
                if all((work / f".resume_ready_{child.pid}").is_file()
                       for child, work, _ in processes):
                    break
                time.sleep(0.2)
            else:
                raise TimeoutError("CF workers did not become ready")
            for (shard, _, _), (child, _, log_path) in zip(prepared, processes):
                shard["pid"] = child.pid
                shard["retry_log"] = str(log_path)
            manifest.setdefault("retry_history", []).append({
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "pids": [child.pid for child, _, _ in processes],
                "mode": mode,
                "mapping": "counterfactual.video_key = relative preflight _video_file"})
            write_json(manifest_path, manifest)
        except BaseException:
            for child, _, _ in processes:
                if child.poll() is None:
                    child.terminate()
            raise
        finally:
            del child_env
    print(f"RETRY_STARTED {mode}; 8 workers, 4 per model", flush=True)
    for shard in manifest["shards"]:
        print(f"{shard['model']} shard{shard['index']}: PID {shard['pid']} "
              f"log={shard['retry_log']}", flush=True)


def status(args):
    manifest = read_json(args.target.resolve() / "launch_manifest.json")
    for model in MODEL_NAMES:
        shards = [s for s in manifest["shards"] if s["model"] == model]
        alive = 0
        counts = {task: {"rows": 0, "errors": 0, "expected": 0} for task in TASKS}
        for shard in shards:
            try:
                check_worker_pid(shard["pid"], shard["output"])
                alive += 1
            except (KeyError, ValueError, OSError):
                pass
            for task in TASKS:
                counts[task]["expected"] += (shard["expected_cf"] if task == "cf"
                                             else shard["expected_normal"])
                path = Path(shard["output"]) / f"{task}_results.json"
                if path.is_file():
                    rows = read_json(path)
                    counts[task]["rows"] += len(rows)
                    counts[task]["errors"] += sum("error" in row for row in rows)
        parts = []
        for task in TASKS:
            c = counts[task]
            parts.append(f"{task}: success={c['rows']-c['errors']}/{c['expected']} "
                         f"errors={c['errors']} pending={c['expected']-c['rows']}")
        print(f"{model} workers={alive}/{len(shards)} | " + " | ".join(parts))
    if args.summary_only:
        return
    print("MANIFEST_STATE", manifest["state"])
    for shard in manifest["shards"]:
        work = Path(shard["output"])
        parts = []
        for task in TASKS:
            path = work / f"{task}_results.json"
            if path.is_file():
                try:
                    rows = read_json(path)
                    parts.append(f"{task}={len(rows)}(errors={sum('error' in r for r in rows)})")
                except (OSError, ValueError):
                    parts.append(f"{task}=unreadable")
        pid = shard.get("pid")
        print(shard["model"], f"shard{shard['index']}",
              f"pid={pid}", f"alive={pid_alive(pid) if pid else False}",
              f"expected={shard['expected_normal']}/{shard['expected_cf']}", *parts)


def merge(args):
    """Combine completed shards, rejecting duplicate or missing sample keys."""
    target = args.target.resolve()
    manifest = read_json(target / "launch_manifest.json")
    if manifest["state"] != "running":
        raise ValueError("Parallel run has not started")
    if any(pid_alive(shard["pid"]) for shard in manifest["shards"]):
        raise ValueError("Workers are still running; merge only after they finish")
    source_rows = read_json(manifest["source_annotations"])
    source_order = [row["video_key"] for row in source_rows]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "parallel_run": str(target), "models": {}}
    combined = {}
    for model in MODEL_NAMES:
        shards = [shard for shard in manifest["shards"] if shard["model"] == model]
        expected_by_task = {"task2": set(source_order), "task3": set(source_order),
                            "cf": set()}
        for shard in shards:
            inputs = read_json(Path(shard["output"]) / "eval_inputs.json")
            expected_by_task["cf"].update(row["video_key"] for row in inputs
                if isinstance(row.get("counterfactual"), dict))
        if len(expected_by_task["cf"]) != 223:
            raise ValueError(f"{model}: CF cohort does not contain 223 unique keys")
        model_report = {}
        for task in TASKS:
            records = {}
            for shard in shards:
                path = Path(shard["output"]) / f"{task}_results.json"
                if not path.is_file():
                    raise ValueError(f"Missing shard result: {path}")
                rows = read_json(path)
                if not isinstance(rows, list):
                    raise ValueError(f"Result is not a list: {path}")
                for row in rows:
                    key = row.get("video_key")
                    if key in records or key not in expected_by_task[task]:
                        raise ValueError(f"{model} {task}: duplicate or unexpected key {key}")
                    records[key] = row
            if set(records) != expected_by_task[task]:
                raise ValueError(f"{model} {task}: {len(expected_by_task[task] - set(records))} missing keys")
            ordered = [records[key] for key in source_order if key in records]
            combined[model, task] = ordered
            model_report[task] = {"total": len(ordered),
                                  "errors": [key for key, row in records.items()
                                             if "error" in row]}
        report["models"][model] = model_report
    output = target / "combined"
    output.mkdir(exist_ok=False)
    for (model, task), rows in combined.items():
        folder = output / model
        folder.mkdir(exist_ok=True)
        write_json(folder / f"{task}_results.json", rows)
    write_json(output / "merge_report.json", report)
    print(f"MERGED {output}", flush=True)
    for model, tasks in report["models"].items():
        print(model, " ".join(f"{task}={info['total']} errors={len(info['errors'])}"
                              for task, info in tasks.items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, default=Path("/root/autodl-tmp"))
    p.add_argument("--source-run", type=Path,
                   default=Path("/root/autodl-tmp/new_model_eval_runs/20261027T064119Z"))
    p.add_argument("--source-parallel", type=Path,
                   help="Reshard an active parallel run without repeating its completed work")
    p.add_argument("--workers-per-model", type=int, choices=(2, 4), default=2)
    p = sub.add_parser("cutover")
    p.add_argument("--target", type=Path, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cf-only", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--summary-only", action="store_true")
    p = sub.add_parser("merge")
    p.add_argument("--target", type=Path, required=True)
    p = sub.add_parser("resume")
    p.add_argument("--target", type=Path, required=True)
    p = sub.add_parser("repair-cf")
    p.add_argument("--target", type=Path, required=True)
    p.set_defaults(retry_normal=False)
    p = sub.add_parser("retry-failed-and-cf")
    p.add_argument("--target", type=Path, required=True)
    p.set_defaults(retry_normal=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "cutover":
        cutover(args)
    elif args.command == "worker":
        sys.exit(worker(args))
    elif args.command == "merge":
        merge(args)
    elif args.command == "resume":
        resume(args)
    elif args.command in ("repair-cf", "retry-failed-and-cf"):
        repair_cf(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
