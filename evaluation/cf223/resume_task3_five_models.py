"""Resume Task3 on available Hunyuan videos and full old-model cohorts, then CF.

Run on the AutoDL host after stopping the previous CF223 scheduler/workers.
All successful old scores are reused by video_key; old files are never edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import continue_task3_cf223_after_task2 as old


MODELS = (*old.MODELS, "hunyuan")
RUNS = old.RUNS
OUTPUT_RE = re.compile(
    r"(?:follow_(?:task3|cf223)_(?:20_)?(?:cog|cosmos|ltx|wan)|"
    r"resume_(?:task3|cf223)_20_[A-Za-z0-9]+_(?:cog|cosmos|ltx|wan|hunyuan))_\d{2}_a\d+"
)


def previous_scores(stage_name: str, model: str) -> set[str]:
    result = "task3_results.json" if stage_name == "task3" else "cf_results.json"
    score = "Score_T3" if stage_name == "task3" else "S_counter"
    folders = [
        *RUNS.glob(f"follow_{stage_name}_{model}_[0-9][0-9]_a*"),
        *RUNS.glob(f"follow_{stage_name}_20_{model}_[0-9][0-9]_a*"),
        *RUNS.glob(f"resume_{stage_name}_20_*_{model}_[0-9][0-9]_a*"),
    ]
    done: set[str] = set()
    for folder in folders:
        scored, _ = old.read_scores(folder / result, score)
        done.update(scored)
    return done


def check_hunyuan_source(root: Path, task3_rows: list[dict]) -> set[str]:
    sys.path.insert(0, str(old.EVAL_BASE))
    from ewm_eval_local import resolve_video

    source = old.EVAL_BASE / "private_annotations_1498_normalized.json"
    rows = json.loads(source.read_text(encoding="utf-8-sig"))
    if len(rows) != 1498:
        raise RuntimeError(f"Expected 1498 source annotations: {source}")
    found: dict[Path, str] = {}
    missing, invalid = [], []
    for row in rows:
        key = row["video_key"]
        try:
            path = old.find_source(resolve_video, [root], key).resolve()
            if path in found:
                raise ValueError(f"same Hunyuan file also maps to {found[path]}")
            if path.stat().st_size == 0:
                raise ValueError(f"empty Hunyuan video: {path}")
            found[path] = key
        except FileNotFoundError:
            missing.append(key)
        except (OSError, ValueError) as exc:
            invalid.append((key, str(exc)))
    available = set(found.values())
    eligible = {row["video_key"] for row in task3_rows} & available
    old.say(f"HUNYUAN SOURCE: available={len(available)}/1498 "
            f"task3_eligible={len(eligible)}/1451 missing={len(missing)} invalid={len(invalid)}")
    for key, reason in invalid[:10]:
        old.say(f"HUNYUAN INVALID {key}: {reason}")
    if invalid:
        raise RuntimeError("Ambiguous or invalid Hunyuan videos need inspection before scoring")
    if not eligible:
        raise RuntimeError("No Hunyuan videos match the Task3 cohort")
    return eligible


def stage(stage_name: str, rows_by_model: dict[str, dict[int, list[dict]]],
          orig_overlays: dict[str, Path], cf_overlays: dict[str, Path],
          work: Path, epoch: str, poll: int, parallel: int, attempts_limit: int) -> bool:
    models = tuple(rows_by_model)
    targets = {model: {row["video_key"] for rows in parts.values() for row in rows}
               for model, parts in rows_by_model.items()}
    expected = {model: len(keys) for model, keys in targets.items()}
    for model in models:
        required = 223 if stage_name == "cf223" else 1451
        if model == "hunyuan" and stage_name == "task3":
            if not 1 <= expected[model] <= required:
                raise RuntimeError(f"Invalid Hunyuan Task3 cohort: {expected[model]}")
        elif expected[model] != required:
            raise RuntimeError(f"Expected {required} {stage_name} keys for {model}, got {expected[model]}")
        source_count = sum(len(rows) for rows in rows_by_model[model].values())
        if source_count != expected[model]:
            raise RuntimeError(f"Duplicate {stage_name} annotation keys for {model}")
    prefix = f"resume_{stage_name}_20_{epoch}_"
    while True:
        active = old.active_outputs()
        unexpected = sorted(name for name in active if OUTPUT_RE.fullmatch(name)
                            and not name.startswith(prefix))
        if unexpected:
            raise RuntimeError("Another Task3/CF evaluator is still active; stop it first: " +
                               ", ".join(unexpected[:5]))
        active_count = Counter()
        for model in models:
            active_count[model] = sum(name.startswith(prefix + model + "_") for name in active)
        scored = {}
        for model in models:
            done = previous_scores(stage_name, model)
            if done - targets[model]:
                raise RuntimeError(f"Unexpected {stage_name} score keys for {model}: "
                                   f"{sorted(done-targets[model])[:3]}")
            scored[model] = done
        exhausted = Counter()
        for model in models:
            for number, rows in rows_by_model[model].items():
                pending = [row for row in rows if row["video_key"] not in scored[model]]
                if not pending:
                    continue
                name = f"{prefix}{model}_{number:02d}"
                attempts = sorted(RUNS.glob(name + "_a*"))
                if any(folder.name in active for folder in attempts):
                    continue
                if len(attempts) >= attempts_limit:
                    exhausted[model] += len(pending)
                    continue
                if active_count[model] >= parallel:
                    continue
                run_name = f"{name}_a{len(attempts)}"
                annotation = work / "annotations" / f"{run_name}.json"
                old.atomic_json(annotation, pending)
                output = RUNS / run_name
                log = old.BASE / "logs" / f"{run_name}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(old.PROGRAM), "run",
                           "--annotations", str(annotation), "--orig-dir", str(orig_overlays[model]),
                           "--tasks", "task3" if stage_name == "task3" else "cf",
                           "--limit", "0", "--output-dir", str(output)]
                if stage_name == "cf223":
                    command.extend(["--cf-dir", str(cf_overlays[model])])
                with log.open("ab") as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True)
                active.add(run_name)
                active_count[model] += 1
                old.say(f"START {run_name}: pending={len(pending)} pid={child.pid} log={log}")
        old.say(f"STATUS {stage_name} " + " ".join(
            f"{m}={len(scored[m])}/{expected[m]} "
            + (f"coverage={expected[m]}/1451 " if m == "hunyuan" else "")
            + f"active={active_count[m]} exhausted={exhausted[m]}"
            for m in models))
        if all(len(scored[m]) + exhausted[m] == expected[m] for m in models):
            if any(exhausted.values()):
                old.say(f"INCOMPLETE {stage_name}: exhausted samples remain; inspect per-shard logs, then "
                        "restart with a fresh --epoch to retry only unscored video_key values")
                return False
            old.say(f"DONE {stage_name}: all {len(models)} available cohorts scored")
            return True
        time.sleep(poll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hunyuan-root", type=Path,
                        default=old.ROOT / "ewm_generated_hunyuan/hunyuan_i2v_5s")
    parser.add_argument("--epoch", default="r1", help="New identifier for another retry round")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max-per-model", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--work", type=Path, default=old.BASE / "follow_task2")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9]+", args.epoch):
        parser.error("--epoch must contain only ASCII letters and digits")
    if args.poll < 10 or not 1 <= args.max_per_model <= 20 or args.max_attempts < 1:
        parser.error("--poll >= 10, --max-per-model 1..20, --max-attempts >= 1 required")
    root = args.hunyuan_root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Hunyuan video directory missing: {root}")
    by_ten, _ = old.shards()
    by_twenty = old.task3_shards(by_ten)
    # Keep the original order so existing overlay/check manifests compare equal.
    rows = [row for part in by_ten.values() for row in part]
    hunyuan_keys = check_hunyuan_source(root, rows)
    hunyuan_rows = [row for row in rows if row["video_key"] in hunyuan_keys]
    task3_by_model = {model: by_twenty for model in old.MODELS}
    task3_by_model["hunyuan"] = {
        number: [row for row in part if row["video_key"] in hunyuan_keys]
        for number, part in by_twenty.items()
    }
    if args.check_only:
        for model in MODELS:
            denominator = len(hunyuan_keys) if model == "hunyuan" else 1451
            old.say(f"SCORED task3/{model}: {len(previous_scores('task3', model))}/{denominator}")
        for model in old.MODELS:
            old.say(f"SCORED cf223/{model}: {len(previous_scores('cf223', model))}/223")
        old.say("Offline check complete; no API calls made")
        return
    if not old.PROGRAM.is_file():
        raise RuntimeError(f"Evaluator missing: {old.PROGRAM}")
    for key in ("EWM_VLM_API_KEY", "EWM_LLM_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing environment variable: {key}")
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "scheduler_cf223.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = old.active_outputs()
        leftover = sorted(name for name in active if OUTPUT_RE.fullmatch(name)
                          and not name.startswith(f"resume_task3_20_{args.epoch}_"))
        if leftover:
            raise RuntimeError("Stop the previous Task3 workers first: " + ", ".join(leftover[:5]))
        old.ORIG["hunyuan"] = root
        sys.path.insert(0, str(old.EVAL_BASE))
        from ewm_eval_local import resolve_video
        overlays = {}
        for model in MODELS:
            source = old.ORIG[model]
            if not source.is_dir():
                raise RuntimeError(f"Original video root missing: {source}")
            overlay = args.work / "media" / model / "orig"
            selected = hunyuan_rows if model == "hunyuan" else rows
            old.make_overlay(model, selected, "orig", [source], overlay, resolve_video)
            old.preflight(model, "task3", selected, overlay, None, args.work)
            overlays[model] = overlay
        task3_ok = stage("task3", task3_by_model, overlays, {}, args.work,
                         args.epoch, args.poll, args.max_per_model, args.max_attempts)
        cf_by_twenty, cf_rows = old.all_cf_shards()
        cf_orig, cf_overlays = {}, {}
        for model in old.MODELS:
            original_overlay = args.work / "media" / model / "cf223_orig"
            old.make_overlay(model, cf_rows, "orig", [old.ORIG[model]],
                             original_overlay, resolve_video)
            cf_orig[model] = original_overlay
            cf_overlay = args.work / "media" / model / "cf223"
            old.make_overlay(model, cf_rows, "cf", old.CF[model], cf_overlay, resolve_video)
            old.preflight(model, "cf", cf_rows, original_overlay, cf_overlay, args.work)
            cf_overlays[model] = cf_overlay
        cf_ok = stage("cf223", {model: cf_by_twenty for model in old.MODELS},
                      cf_orig, cf_overlays,
                      args.work, args.epoch, args.poll, args.max_per_model, args.max_attempts)
        if not task3_ok or not cf_ok:
            raise RuntimeError("Some Task3/CF scores remain unscored; inspect logs and use a new --epoch")


if __name__ == "__main__":
    main()
