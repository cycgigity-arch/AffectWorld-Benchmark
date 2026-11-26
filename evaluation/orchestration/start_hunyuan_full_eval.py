#!/usr/bin/env python3
"""Resume the paper's Task 2, Task 3 and CF scorer on all Hunyuan 5s videos.

The evaluator modules are imported from the existing CF223 bundle. Historical
scores are reused only when both the adapted annotation and evaluated video
bytes match the new full cohort. API credentials stay in process memory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import time


ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "ewm_eval_local_cf223" / "private_annotations_1498_normalized.json"
ADAPTER = ROOT / "ewm_eval_local_cf223" / "ewm_eval_local.py"
VIDEOS = ROOT / "ewm_generated_hunyuan_full_5s"
HISTORY = ROOT / "ewm_eval_local" / "eval_runs"
TASK_FILES = {
    "task2": ("hunyuan_task2_*", "task2_results.json"),
    "task3": ("resume_task3_*hunyuan_*", "task3_results.json"),
    "cf": ("hunyuan_cf223_*", "cf_results.json"),
}
TASKS = tuple(TASK_FILES)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_adapter():
    spec = importlib.util.spec_from_file_location("hunyuan_full_adapter", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def file_hash(path: str | Path, cache: dict[str, str]) -> str | None:
    path = Path(path)
    key = str(path)
    if key in cache:
        return cache[key]
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    cache[key] = digest.hexdigest()
    return cache[key]


def without_video_paths(value):
    if isinstance(value, dict):
        return {k: without_video_paths(v) for k, v in value.items() if k != "_video_file"}
    if isinstance(value, list):
        return [without_video_paths(v) for v in value]
    return value


def same_video(old_path: str | Path | None, new_path: str | Path,
               hashes: dict[str, str]) -> bool:
    if not old_path:
        return False
    old_hash = file_hash(old_path, hashes)
    return old_hash is not None and old_hash == file_hash(new_path, hashes)


def reusable(task: str, record: dict, old_row: dict, new_row: dict,
             hashes: dict[str, str]) -> bool:
    if "error" in record or without_video_paths(old_row) != without_video_paths(new_row):
        return False
    new_orig = new_row["_video_file"]
    old_orig = record.get("orig_video") if task == "cf" else record.get("video_file")
    old_orig = old_orig or old_row.get("_video_file")
    if not same_video(old_orig, new_orig, hashes):
        return False
    if task == "cf":
        old_cf = (old_row.get("counterfactual") or {}).get("_video_file")
        new_cf = (new_row.get("counterfactual") or {}).get("_video_file")
        if not new_cf or not same_video(old_cf, new_cf, hashes):
            return False
    return True


def historical_scores(new_rows: dict[str, dict]) -> dict[str, dict[str, dict]]:
    hashes: dict[str, str] = {}
    seeded: dict[str, dict[str, dict]] = {task: {} for task in TASKS}
    for task, (pattern, filename) in TASK_FILES.items():
        directories = sorted(HISTORY.glob(pattern), key=lambda p: p.stat().st_mtime)
        for directory in directories:
            result_path = directory / filename
            input_path = directory / "eval_inputs.json"
            if not result_path.is_file() or not input_path.is_file():
                continue
            try:
                old_inputs = {r["video_key"]: r for r in read_json(input_path)}
                results = read_json(result_path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if not isinstance(results, list):
                continue
            for record in results:
                if not isinstance(record, dict):
                    continue
                key = record.get("video_key")
                if key not in new_rows or key not in old_inputs:
                    continue
                if not reusable(task, record, old_inputs[key], new_rows[key], hashes):
                    continue
                copy = deepcopy(record)
                if task == "cf":
                    copy["orig_video"] = new_rows[key]["_video_file"]
                else:
                    copy["video_file"] = new_rows[key]["_video_file"]
                seeded[task][key] = copy
    return seeded


def create_cf_runtime(target: Path) -> int:
    """Provide the scorer's video_key paths without copying or changing videos."""
    runtime = target / "cf_runtime"
    runtime.mkdir(exist_ok=True)
    count = 0
    for index in range(4):
        rows = read_json(target / f"shard{index}" / "eval_inputs.json")
        for row in rows:
            branch = row.get("counterfactual")
            if not isinstance(branch, dict):
                continue
            key = branch.get("video_key") or row["video_key"]
            parts = PurePosixPath(key).parts
            if not parts or ".." in parts or PurePosixPath(key).is_absolute():
                raise ValueError(f"Unsafe CF video key: {key}")
            source = Path(branch["_video_file"])
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = runtime.joinpath(*parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not destination.samefile(source):
                    raise ValueError(f"Conflicting CF video path: {destination}")
            else:
                os.link(source, destination)
            count += 1
    if count != 223:
        raise ValueError(f"Expected 223 CF paths, found {count}")
    return count


def prepare(target: Path) -> None:
    adapter = load_adapter()
    source = read_json(SOURCE)
    if len(source) != 1498 or len({r["video_key"] for r in source}) != 1498:
        raise ValueError("Source annotation cohort is not 1498 unique keys")
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    shards = []
    new_rows: dict[str, dict] = {}
    for index in range(4):
        subset = source[index::4]
        annotation = target / f"shard{index}_annotations.json"
        write_json(annotation, subset)
        work = target / f"shard{index}"
        args = argparse.Namespace(annotations=annotation,
                                  orig_dir=VIDEOS / "normal",
                                  cf_dir=VIDEOS / "counterfactual",
                                  tasks=list(TASKS), limit=0,
                                  label_overrides=None, output_dir=work)
        _, report = adapter.prepare(args)
        if report["errors"] or report["ready"] != len(subset):
            raise ValueError(f"Shard {index} failed preflight: {report['errors'][:3]}")
        rows = read_json(work / "eval_inputs.json")
        new_rows.update({r["video_key"]: r for r in rows})
        cf_count = sum(isinstance(r.get("counterfactual"), dict) for r in rows)
        shards.append({"index": index, "work": str(work),
                       "annotations": str(annotation), "normal": len(rows),
                       "cf": cf_count, "pid": None})
        print(f"preflight shard{index}: normal={len(rows)} cf={cf_count}", flush=True)
    if len(new_rows) != 1498 or sum(s["cf"] for s in shards) != 223:
        raise ValueError("Full cohort is incomplete")
    create_cf_runtime(target)
    previous = historical_scores(new_rows)
    for shard in shards:
        rows = read_json(Path(shard["work"]) / "eval_inputs.json")
        keys = [r["video_key"] for r in rows]
        shard["seeded"] = {}
        for task, filename in (("task2", "task2_results.json"),
                               ("task3", "task3_results.json"),
                               ("cf", "cf_results.json")):
            records = [previous[task][key] for key in keys if key in previous[task]]
            write_json(Path(shard["work"]) / filename, records)
            shard["seeded"][task] = len(records)
        print(f"seed shard{shard['index']}: {shard['seeded']}", flush=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(SOURCE), "source_sha256": file_hash(SOURCE, {}),
                "video_root": str(VIDEOS), "tasks": list(TASKS),
                "expected_normal": 1498, "expected_cf": 223,
                "shards": shards}
    write_json(target / "launch_manifest.json", manifest)
    print("PREPARED", target, flush=True)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return stat.is_file() and stat.read_text().split()[2] != "Z"


def start(target: Path) -> None:
    manifest_path = target / "launch_manifest.json"
    manifest = read_json(manifest_path)
    if any(pid_alive(shard.get("pid")) for shard in manifest["shards"]):
        raise RuntimeError("A shard is already running")
    if create_cf_runtime(target) != 223:
        raise RuntimeError("CF runtime mapping is incomplete")
    vlm = getpass.getpass("VLM API key (hidden): ").strip()
    llm = getpass.getpass("LLM API key (hidden): ").strip()
    if not vlm or not llm:
        raise ValueError("Both API keys are required")
    env = os.environ.copy()
    env["EWM_VLM_API_KEY"] = vlm
    env["EWM_LLM_API_KEY"] = llm
    del vlm, llm
    for shard in manifest["shards"]:
        index = shard["index"]
        log_path = target / f"shard{index}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "worker", "--target", str(target), "--shard", str(index)]
        with log_path.open("ab") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        shard["pid"] = child.pid
        shard["log"] = str(log_path)
        print(f"started shard{index} pid={child.pid} log={log_path}", flush=True)
    del env
    write_json(manifest_path, manifest)
    print("STARTED", target, flush=True)


def worker(target: Path, index: int) -> None:
    adapter = load_adapter()
    work = target / f"shard{index}"
    if not (work / "eval_inputs.json").is_file():
        raise FileNotFoundError(work / "eval_inputs.json")
    errors = []
    for task in TASKS:
        args = argparse.Namespace(tasks=[task], cf_dir=target / "cf_runtime")
        for attempt in (1, 2):
            print(f"shard{index} {task} attempt={attempt}", flush=True)
            try:
                adapter.run(args, work)
            except Exception as exc:
                print(f"shard{index} {task} attempt={attempt} failed: {exc}", flush=True)
                if attempt == 1:
                    time.sleep(10)
            else:
                break
        else:
            errors.append(task)
    write_json(work / "worker_done.json", {"finished_utc": datetime.now(timezone.utc).isoformat(),
                                           "incomplete_tasks": errors})
    print(f"shard{index} finished; incomplete_tasks={errors}", flush=True)


def status(target: Path) -> None:
    manifest = read_json(target / "launch_manifest.json")
    totals = {task: {"success": 0, "errors": 0, "expected": 0} for task in TASKS}
    for shard in manifest["shards"]:
        work = Path(shard["work"])
        values = []
        for task, (_, filename) in TASK_FILES.items():
            rows = read_json(work / filename) if (work / filename).is_file() else []
            success = len({r["video_key"] for r in rows if "error" not in r})
            errors = len({r["video_key"] for r in rows if "error" in r})
            expected = shard["cf"] if task == "cf" else shard["normal"]
            totals[task]["success"] += success
            totals[task]["errors"] += errors
            totals[task]["expected"] += expected
            values.append(f"{task}={success}/{expected} err={errors}")
        print(f"shard{shard['index']} alive={pid_alive(shard.get('pid'))} " + " ".join(values))
    for task, counts in totals.items():
        print(f"TOTAL {task}: {counts['success']}/{counts['expected']} "
              f"errors_to_retry={counts['errors']} "
              f"unattempted={counts['expected']-counts['success']-counts['errors']} "
              f"remaining={counts['expected']-counts['success']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "start", "status", "worker", "fix-cf-runtime"):
        part = sub.add_parser(command)
        part.add_argument("--target", type=Path, required=True)
        if command == "worker":
            part.add_argument("--shard", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.target)
    elif args.command == "start":
        start(args.target)
    elif args.command == "status":
        status(args.target)
    elif args.command == "fix-cf-runtime":
        print("CF runtime paths:", create_cf_runtime(args.target))
    else:
        worker(args.target, args.shard)


if __name__ == "__main__":
    main()
