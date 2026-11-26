"""Run available Hunyuan Task2 videos, then score available Hunyuan CF pairs.

This scheduler is independent of the four-model CF run. Both stages use the
existing CF223 evaluator and its original annotations. Successful scores are
reused by video_key after a restart.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import continue_task3_cf223_after_task2 as old
import run_hunyuan_cf223 as cf_runner


SHARDS = 20
PREFIX = "hunyuan_task2_20_"


def select_task2_rows(orig_root: Path) -> list[dict]:
    sys.path.insert(0, str(old.EVAL_BASE))
    from ewm_eval_local import resolve_video

    by_ten, _ = old.shards()
    rows = [row for part in by_ten.values() for row in part]
    selected, seen, problems = [], {}, []
    for row in rows:
        key = row["video_key"]
        try:
            path = old.find_source(resolve_video, [orig_root], key).resolve()
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            problems.append(f"{key}: {exc}")
            continue
        try:
            if path in seen:
                raise ValueError(f"file also maps to {seen[path]}")
            if not path.stat().st_size:
                raise ValueError("empty video")
            seen[path] = key
            selected.append(row)
        except (OSError, ValueError) as exc:
            problems.append(f"{key}: {exc}")
    old.say(f"HUNYUAN TASK2 SOURCE: available={len(selected)}/1451 "
            f"missing={1451-len(selected)-len(problems)} invalid={len(problems)}")
    for problem in problems[:10]:
        old.say(f"HUNYUAN TASK2 INVALID {problem}")
    if problems:
        raise RuntimeError("Invalid Hunyuan Task2 source mapping; no API calls made")
    if not selected:
        raise RuntimeError("No Hunyuan original video matches the Task2 cohort")
    return selected


def scored_keys(targets: set[str]) -> set[str]:
    done = set()
    for folder in old.RUNS.glob(PREFIX + "*"):
        if not re.fullmatch(PREFIX + r"[A-Za-z0-9]+_\d{2}_a\d+", folder.name):
            continue
        scored, _ = old.read_scores(folder / "task2_results.json", "Score_T2")
        done.update(scored)
    unknown = done - targets
    if unknown:
        raise RuntimeError(f"Unexpected Hunyuan Task2 score keys: {sorted(unknown)[:5]}")
    return done


def run_task2(rows: list[dict], overlay: Path, work: Path, epoch: str,
              parallel: int, poll: int, max_attempts: int) -> tuple[int, list[str]]:
    parts = {number: rows[number::SHARDS] for number in range(SHARDS)}
    targets = {row["video_key"] for row in rows}
    current_prefix = PREFIX + epoch + "_"
    while True:
        active = old.active_outputs()
        other = sorted(name for name in active
                       if name.startswith(PREFIX) and not name.startswith(current_prefix))
        if other:
            raise RuntimeError("Another Hunyuan Task2 run is active: " + ", ".join(other[:5]))
        active_count = sum(name.startswith(current_prefix) for name in active)
        done = scored_keys(targets)
        exhausted = []
        for number, shard_rows in parts.items():
            pending = [row for row in shard_rows if row["video_key"] not in done]
            if not pending:
                continue
            base_name = f"{current_prefix}{number:02d}"
            if any(name.startswith(base_name + "_a") for name in active):
                continue
            attempts = sorted((work / "attempts").glob(base_name + "_a*.json"))
            if len(attempts) >= max_attempts:
                exhausted.extend(row["video_key"] for row in pending)
                continue
            if active_count >= parallel:
                continue
            run_name = f"{base_name}_a{len(attempts)}"
            annotation = work / "annotations" / f"{run_name}.json"
            old.atomic_json(annotation, pending)
            output = old.RUNS / run_name
            if output.exists():
                raise RuntimeError(f"Output directory already exists: {output}")
            log = old.BASE / "logs" / f"{run_name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(old.PROGRAM), "run",
                       "--annotations", str(annotation),
                       "--orig-dir", str(overlay),
                       "--tasks", "task2", "--limit", "0",
                       "--output-dir", str(output)]
            old.atomic_json(work / "attempts" / f"{run_name}.json",
                            {"annotation": str(annotation), "output": str(output),
                             "log": str(log), "command": command})
            with log.open("ab") as stream:
                child = subprocess.Popen(command, stdout=stream,
                                         stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL,
                                         start_new_session=True)
            active.add(run_name)
            active_count += 1
            old.say(f"START {run_name}: pending={len(pending)} pid={child.pid} log={log}")
        old.say(f"STATUS task2 hunyuan={len(done)}/{len(targets)} "
                f"coverage={len(targets)}/1451 active={active_count} "
                f"exhausted={len(exhausted)}")
        if len(done) + len(exhausted) == len(targets):
            old.say(f"END task2 hunyuan={len(done)}/{len(targets)} "
                    f"failed={len(exhausted)}")
            for key in exhausted[:10]:
                old.say(f"TASK2 FAILED {key}")
            return len(done), exhausted
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-root", type=Path,
                        default=old.ROOT / "ewm_generated_hunyuan/hunyuan_i2v_5s")
    parser.add_argument("--cf-root", type=Path,
                        default=old.ROOT / "ewm_generated_hunyuan/counterfactual")
    parser.add_argument("--expected-cf-files", type=int, default=30)
    parser.add_argument("--work", type=Path,
                        default=old.BASE / "hunyuan_task2_cf_pipeline")
    parser.add_argument("--task2-epoch", default="r1")
    parser.add_argument("--cf-epoch", default="r1")
    parser.add_argument("--task2-parallel", type=int, default=4)
    parser.add_argument("--cf-parallel", type=int, default=4)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-task2-failures", type=int, default=10,
                        help="Stop before CF if more Task2 samples remain unscored")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for epoch in (args.task2_epoch, args.cf_epoch):
        if not re.fullmatch(r"[A-Za-z0-9]+", epoch):
            parser.error("Epoch identifiers must contain only ASCII letters and digits")
    if not 1 <= args.task2_parallel <= SHARDS or not 1 <= args.cf_parallel <= SHARDS:
        parser.error("--task2-parallel and --cf-parallel must be 1..20")
    if args.poll < 10 or args.max_attempts < 1 or args.max_task2_failures < 0:
        parser.error("--poll >= 10, --max-attempts >= 1, --max-task2-failures >= 0 required")
    if args.expected_cf_files < 1:
        parser.error("--expected-cf-files must be positive")
    orig_root, cf_root = args.orig_root.resolve(), args.cf_root.resolve()
    for root in (orig_root, cf_root):
        if not root.is_dir():
            raise RuntimeError(f"Hunyuan video directory missing: {root}")
    if not old.PROGRAM.is_file():
        raise RuntimeError(f"CF223 evaluator missing: {old.PROGRAM}")
    task2_rows = select_task2_rows(orig_root)
    cf_rows = cf_runner.select_pairs(orig_root, cf_root, args.expected_cf_files)
    task2_targets = {row["video_key"] for row in task2_rows}
    cf_targets = {row["video_key"] for row in cf_rows}
    old.say(f"HUNYUAN SCORED task2={len(scored_keys(task2_targets))}/{len(task2_rows)} "
            f"cf223={len(cf_runner.scored_keys(cf_targets))}/{len(cf_rows)}")
    if args.check_only:
        old.say("Both source checks complete; no API calls made")
        return
    for key in ("EWM_VLM_API_KEY", "EWM_LLM_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing environment variable: {key}")
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old.atomic_json(args.work / "task2_selected_keys.json",
                        [row["video_key"] for row in task2_rows])
        old.ORIG["hunyuan"] = orig_root
        from ewm_eval_local import resolve_video

        overlay = args.work / "media/task2_orig"
        old.make_overlay("hunyuan", task2_rows, "orig", [orig_root],
                         overlay, resolve_video)
        old.preflight("hunyuan", "task2", task2_rows, overlay, None, args.work)
        scored, failed = run_task2(task2_rows, overlay, args.work,
                                   args.task2_epoch, args.task2_parallel,
                                   args.poll, args.max_attempts)
        if failed and len(failed) > args.max_task2_failures:
            raise RuntimeError(f"Task2 has {len(failed)} unscored samples, above the "
                               f"allowed {args.max_task2_failures}; inspect logs "
                               "before starting CF")
        old.say(f"TASK2 COMPLETE ENOUGH: {scored}/{len(task2_rows)} scored; "
                f"{len(failed)} failed; starting Hunyuan CF")
        command = [sys.executable, "-u", str(Path(__file__).with_name("run_hunyuan_cf223.py")),
                   "--orig-root", str(orig_root), "--cf-root", str(cf_root),
                   "--expected-files", str(args.expected_cf_files),
                   "--epoch", args.cf_epoch, "--parallel", str(args.cf_parallel),
                   "--poll", str(args.poll), "--max-attempts", str(args.max_attempts)]
        subprocess.run(command, check=True)
        old.say("DONE Hunyuan Task2 and available CF pairs")


if __name__ == "__main__":
    main()
