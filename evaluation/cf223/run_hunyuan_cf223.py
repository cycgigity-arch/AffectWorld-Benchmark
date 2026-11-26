"""Score the available Hunyuan counterfactual pairs with the CF223 protocol.

This is an independent scheduler. It can run while the four-model CF scheduler
continues, and it never edits the existing models' outputs or worker processes.
Run it from /root/autodl-tmp/ewm_eval_local_cf223 on the AutoDL host.
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

import continue_task3_cf223_after_task2 as old


MODEL = "hunyuan"
SHARDS = 20
RESULT = "cf_results.json"
SCORE = "S_counter"
PREFIX = "hunyuan_cf223_20_"


def select_pairs(orig_root: Path, cf_root: Path, expected_files: int) -> list[dict]:
    """Find every generated CF video and select those with Hunyuan originals."""
    sys.path.insert(0, str(old.EVAL_BASE))
    from ewm_eval_local import resolve_video

    _, annotated = old.all_cf_shards()
    physical = {path.resolve() for path in cf_root.rglob("*")
                if path.is_file() and path.suffix.lower() == ".mp4"}
    if len(physical) != expected_files:
        raise RuntimeError(f"Expected {expected_files} Hunyuan CF MP4 files, found "
                           f"{len(physical)} under {cf_root}")

    selected, matched_cf, matched_orig = [], {}, {}
    errors, unpaired = [], []
    for row in annotated:
        key = row["video_key"]
        try:
            cf = old.find_source(resolve_video, [cf_root], key,
                                 old.cf_branch(row)).resolve()
        except FileNotFoundError:
            continue  # One of the 223 annotated CF videos was not generated.
        except (OSError, ValueError) as exc:
            errors.append(f"CF {key}: {exc}")
            continue
        if cf in matched_cf:
            errors.append(f"CF file maps to both {matched_cf[cf]} and {key}: {cf}")
            continue
        matched_cf[cf] = key
        try:
            orig = old.find_source(resolve_video, [orig_root], key).resolve()
        except FileNotFoundError:
            unpaired.append(key)
            continue
        try:
            if orig in matched_orig:
                raise ValueError(f"original file also maps to {matched_orig[orig]}")
            if not orig.stat().st_size or not cf.stat().st_size:
                raise ValueError("empty original or CF video")
            matched_orig[orig] = key
            selected.append(row)
        except (OSError, ValueError) as exc:
            errors.append(f"Original {key}: {exc}")

    unmatched = sorted(physical - set(matched_cf))
    for path in unmatched[:10]:
        errors.append(f"Unmatched generated CF file: {path}")
    old.say(f"HUNYUAN CF SOURCE: physical={len(physical)} matched={len(matched_cf)} "
            f"paired={len(selected)}/223 unpaired={len(unpaired)}")
    for key in unpaired[:10]:
        old.say(f"HUNYUAN CF NO ORIGINAL {key}")
    for message in errors[:12]:
        old.say(f"HUNYUAN CF INVALID {message}")
    if errors:
        raise RuntimeError(f"Source mapping failed: {len(errors)} reported problem(s), "
                           f"including {len(unmatched)} unmatched CF file(s); "
                           "no API calls made")
    if not selected:
        raise RuntimeError("No generated Hunyuan CF video has a matching original")
    return selected


def scored_keys(targets: set[str]) -> set[str]:
    done = set()
    for folder in old.RUNS.glob(PREFIX + "*"):
        if not re.fullmatch(PREFIX + r"[A-Za-z0-9]+_\d{2}_a\d+", folder.name):
            continue
        scored, _ = old.read_scores(folder / RESULT, SCORE)
        done.update(scored)
    unknown = done - targets
    if unknown:
        raise RuntimeError(f"Unexpected Hunyuan CF score keys: {sorted(unknown)[:5]}")
    return done


def run_shards(rows: list[dict], orig_overlay: Path, cf_overlay: Path,
               work: Path, epoch: str, parallel: int, poll: int,
               max_attempts: int) -> None:
    parts = {number: rows[number::SHARDS] for number in range(SHARDS)}
    targets = {row["video_key"] for row in rows}
    current_prefix = PREFIX + epoch + "_"
    while True:
        active = old.active_outputs()
        other = sorted(name for name in active
                       if name.startswith(PREFIX) and not name.startswith(current_prefix))
        if other:
            raise RuntimeError("Another Hunyuan CF run is active: " + ", ".join(other[:5]))
        active_count = sum(name.startswith(current_prefix) for name in active)
        done = scored_keys(targets)
        exhausted = 0
        for number, shard_rows in parts.items():
            pending = [row for row in shard_rows if row["video_key"] not in done]
            if not pending:
                continue
            base_name = f"{current_prefix}{number:02d}"
            if any(name.startswith(base_name + "_a") for name in active):
                continue
            attempts = sorted((work / "attempts").glob(base_name + "_a*.json"))
            if len(attempts) >= max_attempts:
                exhausted += len(pending)
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
                       "--orig-dir", str(orig_overlay),
                       "--cf-dir", str(cf_overlay),
                       "--tasks", "cf", "--limit", "0",
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
        old.say(f"STATUS cf223 hunyuan={len(done)}/{len(targets)} "
                f"coverage={len(targets)}/223 active={active_count} exhausted={exhausted}")
        if len(done) + exhausted == len(targets):
            if exhausted:
                raise RuntimeError(f"{exhausted} Hunyuan CF pair(s) remain unscored; "
                                   "inspect shard logs, then retry with a new --epoch")
            old.say(f"DONE cf223 hunyuan={len(done)}/{len(targets)}")
            return
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-root", type=Path,
                        default=old.ROOT / "ewm_generated_hunyuan/hunyuan_i2v_5s")
    parser.add_argument("--cf-root", type=Path,
                        default=old.ROOT / "ewm_generated_hunyuan/counterfactual")
    parser.add_argument("--expected-files", type=int, default=30)
    parser.add_argument("--work", type=Path,
                        default=old.BASE / "hunyuan_cf223")
    parser.add_argument("--epoch", default="r1")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Concurrent Hunyuan shards; cohort is always split into 20")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9]+", args.epoch):
        parser.error("--epoch must contain only ASCII letters and digits")
    if args.expected_files < 1 or not 1 <= args.parallel <= SHARDS:
        parser.error("--expected-files >= 1 and --parallel 1..20 required")
    if args.poll < 10 or args.max_attempts < 1:
        parser.error("--poll >= 10 and --max-attempts >= 1 required")
    orig_root, cf_root = args.orig_root.resolve(), args.cf_root.resolve()
    for root in (orig_root, cf_root):
        if not root.is_dir():
            raise RuntimeError(f"Hunyuan video directory missing: {root}")
    if not old.PROGRAM.is_file():
        raise RuntimeError(f"CF223 evaluator missing: {old.PROGRAM}")
    rows = select_pairs(orig_root, cf_root, args.expected_files)
    targets = {row["video_key"] for row in rows}
    old.say(f"HUNYUAN CF SCORED: {len(scored_keys(targets))}/{len(rows)}")
    if args.check_only:
        old.say("Source check complete; no API calls made")
        return
    for key in ("EWM_VLM_API_KEY", "EWM_LLM_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing environment variable: {key}")
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "scheduler.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old.atomic_json(args.work / "selected_keys.json",
                        [row["video_key"] for row in rows])
        old.ORIG[MODEL] = orig_root
        from ewm_eval_local import resolve_video

        orig_overlay = args.work / "media/orig"
        cf_overlay = args.work / "media/cf"
        old.make_overlay(MODEL, rows, "orig", [orig_root], orig_overlay,
                         resolve_video)
        old.make_overlay(MODEL, rows, "cf", [cf_root], cf_overlay,
                         resolve_video)
        old.preflight(MODEL, "cf", rows, orig_overlay, cf_overlay, args.work)
        run_shards(rows, orig_overlay, cf_overlay, args.work, args.epoch,
                   args.parallel, args.poll, args.max_attempts)


if __name__ == "__main__":
    main()
