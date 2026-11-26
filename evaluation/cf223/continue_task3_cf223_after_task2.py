"""Reuse four-model Task2, then run Task3 and all 223 attached-protocol CF pairs.

Upload this file and the cf223 evaluator bundle to /root/autodl-tmp. Running
Task2 processes and their evaluator bundle are left untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
import time


ROOT = Path("/root/autodl-tmp")
BASE = ROOT / "ewm_eval_local"
EVAL_BASE = ROOT / "ewm_eval_local_cf223"
PROGRAM = EVAL_BASE / "ewm_eval_local.py"
RUNS = BASE / "eval_runs"
SHARDS = BASE / "shards/20261010T171904052041Z"
TAG = "20261011T024038Z"
LIMIT = 10 * 1024 * 1024
TARGET_BYTES = 6.5 * 1024 * 1024
MODELS = ("cog", "cosmos", "ltx", "wan")
SPLITS = 20
ORIG = {
    "cog": ROOT / "ewm_generated_cog/cogvideox_i2v_state0_context_6s",
    "cosmos": ROOT / "ewm_generated_cosmos/visual_only_6s",
    "ltx": ROOT / "ewm_generated_LTX/ltx_i2v_state0_context_6s",
    "wan": ROOT / "ewm_generated_wan/vace_visual_only_6s",
}
CF = {
    "cog": [ROOT / "ewm_generated_cog/cogvideox_counterfactual_6s",
            ROOT / "ewm_generated/cogvideox_counterfactual_6s"],
    "cosmos": [ROOT / f"ewm_generated_cosmos/{name}" for name in (
        "counterfactual_6s_3gpu", "counterfactual_6s_4gpu",
        "counterfactual_6s_1gpu", "counterfactual_6s")] + [
        ROOT / f"ewm_generated/{name}" for name in (
            "counterfactual_6s_3gpu", "counterfactual_6s_4gpu",
            "counterfactual_6s_1gpu", "counterfactual_6s")],
    "ltx": [ROOT / "ewm_generated_LTX/ltx_counterfactual_6s",
            ROOT / "ewm_generated/ltx_counterfactual_6s"],
    "wan": [ROOT / "ewm_generated_wan/vace_counterfactual_6s/480p_96f16fps_50steps",
            ROOT / "ewm_generated_wan/vace_counterfactual_6s",
            ROOT / "ewm_generated/vace_counterfactual_6s/480p_96f16fps_50steps"],
}
_CACHE = {}


def now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def say(message):
    print(f"[{now()}] {message}", flush=True)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != value:
            raise RuntimeError(f"Existing file differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def shards():
    result = {}
    all_keys = []
    for i in range(10):
        path = SHARDS / f"all_{i:02d}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Empty or invalid shard: {path}")
        result[i] = rows
        all_keys.extend(row["video_key"] for row in rows)
    if len(all_keys) != 1451 or len(set(all_keys)) != len(all_keys):
        raise RuntimeError(f"Expected 1451 distinct samples, got {len(all_keys)}")
    return result, set(all_keys)


def cf_branch(row):
    """Accept both normalized ewm_unit rows and flattened eval_inputs rows."""
    direct = row.get("counterfactual")
    if isinstance(direct, dict):
        return direct
    unit = row.get("ewm_unit")
    nested = unit.get("counterfactual") if isinstance(unit, dict) else None
    return nested if isinstance(nested, dict) else None


def all_cf_shards():
    source = BASE / "private_annotations_1498_normalized.json"
    bundled = EVAL_BASE / source.name
    if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(bundled.read_bytes()).digest():
        raise RuntimeError("Old and CF223 bundles contain different source annotations")
    rows = json.loads(source.read_text(encoding="utf-8-sig"))
    cf_rows = [row for row in rows if cf_branch(row) is not None]
    keys = [row["video_key"] for row in cf_rows]
    if len(rows) != 1498 or len(keys) != 223 or len(set(keys)) != 223:
        raise RuntimeError(f"Expected 1498 source rows and 223 unique CF rows: {source}")
    return {i: cf_rows[i::SPLITS] for i in range(SPLITS)}, cf_rows


def task3_shards(by_shard):
    rows = [row for part in by_shard.values() for row in part]
    if len(rows) != 1451 or len({row["video_key"] for row in rows}) != 1451:
        raise RuntimeError("Task3 source shards must contain 1451 distinct rows")
    return {i: rows[i::SPLITS] for i in range(SPLITS)}


def read_scores(path, score_field):
    """Read only complete top-level JSON records; cache unchanged files."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return set(), set()
    identity = (stat.st_size, stat.st_mtime_ns)
    cached = _CACHE.get((path, score_field))
    if cached and cached[0] == identity:
        return cached[1], cached[2]
    scored, errors = set(), set()
    try:
        with path.open(encoding="utf-8") as stream:
            record = {}
            for line in stream:
                if line.startswith("  {"):
                    record = {}
                elif line.startswith('    "'):
                    key, _, value = line.strip().partition(":")
                    key = json.loads(key)
                    if key in ("video_key", score_field, "error"):
                        record[key] = json.loads(value.rstrip(","))
                elif line.startswith("  }"):
                    key = record.get("video_key")
                    if key and "error" not in record and isinstance(
                        record.get(score_field), (int, float)
                    ):
                        scored.add(key)
                    elif key and "error" in record:
                        errors.add(key)
    except (OSError, ValueError, json.JSONDecodeError):
        return cached[1:3] if cached else (set(), set())
    _CACHE[(path, score_field)] = (identity, scored, errors)
    return scored, errors


def task2_dirs(model):
    if model == "cog":
        original = [RUNS / "20261010T171904052041Z", *sorted(RUNS.glob("cog_task2_*"))]
    else:
        original = sorted(RUNS.glob(f"{model}_task2_{TAG}_*"))
    return [*original, *sorted(RUNS.glob(f"task2_cf223_recovery_{model}_a*"))]


def wait_task2(expected, by_shard, work, poll):
    """Finish unscored Task2 rows before launching Task3, reusing earlier scores."""
    all_rows = [row for part in by_shard.values() for row in part]
    row_by_key = {row["video_key"]: row for row in all_rows}
    sys.path.insert(0, str(EVAL_BASE))
    from ewm_eval_local import resolve_video
    while True:
        counts, pending_by_model = {}, {}
        for model in MODELS:
            done = set()
            for directory in task2_dirs(model):
                scored, _ = read_scores(directory / "task2_results.json", "Score_T2")
                done.update(scored)
            unknown = done - expected
            if unknown:
                raise RuntimeError(f"Task2 {model} has unexpected keys: {list(unknown)[:3]}")
            counts[model] = len(done)
            pending_by_model[model] = [row_by_key[key] for key in sorted(expected - done)]
        say("WAIT Task2 " + " ".join(f"{m}={n}/1451" for m, n in counts.items()))
        if all(n == len(expected) for n in counts.values()):
            return
        running = active_outputs()
        for model in MODELS:
            pending = pending_by_model[model]
            if not pending:
                continue
            prefixes = (("cog_task2_",) if model == "cog" else
                        (f"{model}_task2_{TAG}_",))
            recovery_prefix = f"task2_cf223_recovery_{model}_a"
            if any(name.startswith((*prefixes, recovery_prefix)) for name in running):
                continue
            attempts = sorted(RUNS.glob(recovery_prefix + "*"))
            if len(attempts) >= 3:
                missing = [row["video_key"] for row in pending]
                raise RuntimeError(f"Task2 {model} still lacks {len(missing)} scores "
                                   f"after 3 recovery attempts: {missing[:5]}; "
                                   f"inspect {BASE / 'logs' / (recovery_prefix + '*.log')}")
            attempt = len(attempts)
            name = recovery_prefix + str(attempt)
            annotation = work / "annotations" / f"{name}.json"
            atomic_json(annotation, pending)
            overlay = work / "media" / model / name
            make_overlay(model, pending, "orig", [ORIG[model]], overlay, resolve_video)
            preflight(model, "task2", pending, overlay, None, work, check_tag=name)
            output = RUNS / name
            log = BASE / "logs" / f"{name}.log"
            command = [sys.executable, "-u", str(PROGRAM), "run",
                       "--annotations", str(annotation), "--orig-dir", str(overlay),
                       "--tasks", "task2", "--limit", "0", "--output-dir", str(output)]
            with log.open("ab") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            running.add(name)
            say(f"REPAIR Task2 {model}: pending={len(pending)} pid={child.pid} log={log}")
        time.sleep(poll)


def too_large(path):
    return 4 * ((path.stat().st_size + 2) // 3) > LIMIT


def encode(source, dest):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are needed for oversized videos")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        check=True, capture_output=True, text=True,
    )
    seconds = float(result.stdout.strip())
    if seconds <= 0:
        raise RuntimeError(f"Invalid video duration: {source}")
    bitrate = min(8_000_000, max(300_000, int(TARGET_BYTES * 8 / seconds) - 128_000))
    temporary = dest.with_name(dest.stem + ".encoding.mp4")
    for factor in (1.0, 0.8, 0.6, 0.4):
        rate = max(200_000, int(bitrate * factor))
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-b:v", str(rate),
            "-maxrate", str(rate), "-bufsize", str(rate), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(temporary),
        ], check=True)
        if not too_large(temporary):
            os.replace(temporary, dest)
            return
    raise RuntimeError(f"Cannot compress below Base64 limit: {source}")


def find_source(resolve_video, roots, key, branch=None):
    for root in roots:
        if not root.is_dir():
            continue
        try:
            return resolve_video(root, key, branch)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(f"Video missing for {key} under {', '.join(map(str, roots))}")


def overlay_video(source, output, manifest):
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        if too_large(source):
            encode(source, output)
            method = "encoded"
        else:
            try:
                os.link(source, output)
                method = "hardlink"
            except OSError:
                shutil.copy2(source, output)
                method = "copy"
    else:
        method = "existing"
    if too_large(output):
        raise RuntimeError(f"Overlay video exceeds Base64 limit: {output}")
    manifest.append({"source": str(source), "evaluation_file": str(output),
                     "method": method, "source_bytes": source.stat().st_size,
                     "evaluation_bytes": output.stat().st_size})


def make_overlay(model, rows, stage, root_candidates, overlay_root, resolve_video):
    existing = overlay_root / "source_manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text(encoding="utf-8"))
        if len(manifest) != len(rows) or any(
            not Path(item["evaluation_file"]).is_file() for item in manifest
        ):
            raise RuntimeError(f"Incomplete existing overlay manifest: {existing}")
        say(f"OVERLAY {model}/{stage}: reusing {len(manifest)} videos")
        return
    manifest = []
    for row in rows:
        branch = cf_branch(row) if stage == "cf" else None
        key = (branch or {}).get("video_key") or row["video_key"]
        dest = (overlay_root / key).resolve()
        if not dest.is_relative_to(overlay_root.resolve()):
            raise RuntimeError(f"Unsafe video key: {key}")
        source = find_source(resolve_video, root_candidates, row["video_key"], branch)
        if stage == "orig" and too_large(source):
            mirror = (BASE / "base64_recovery_videos" / model /
                      source.relative_to(ORIG[model].resolve()))
            if mirror.is_file() and not too_large(mirror):
                source = mirror
        overlay_video(source, dest, manifest)
    atomic_json(overlay_root / "source_manifest.json", manifest)
    say(f"OVERLAY {model}/{stage}: {len(manifest)} videos, "
        f"{sum(x['method'] == 'encoded' for x in manifest)} encoded")


def preflight(model, stage, rows, orig_dir, cf_dir, work, check_tag=None):
    name = check_tag or f"{stage}_{model}"
    check_input = work / "checks" / f"{name}.json"
    atomic_json(check_input, rows)
    output = work / "checks" / f"{name}_report_{time.time_ns()}"
    command = [sys.executable, str(PROGRAM), "check",
               "--annotations", str(check_input), "--orig-dir", str(orig_dir),
               "--tasks", stage, "--limit", "0", "--output-dir", str(output)]
    if stage == "cf":
        command.extend(["--cf-dir", str(cf_dir)])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Preflight {stage}/{model} failed:\n{result.stdout}\n{result.stderr}")
    report = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    if report["ready"] != len(rows) or report["errors"]:
        raise RuntimeError(f"Preflight {stage}/{model} incomplete: {output / 'preflight.json'}")
    say(f"CHECK {stage}/{model}: ready={report['ready']} errors=0")


def active_outputs():
    active = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = [os.fsdecode(x) for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
            if "--output-dir" in args:
                i = args.index("--output-dir") + 1
                if i < len(args):
                    active.add(Path(args[i]).name)
        except (OSError, ValueError):
            pass
    return active


def old_scheduler_pids():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = [os.fsdecode(item) for item in
                    (proc / "cmdline").read_bytes().split(b"\0") if item]
            if any(Path(item).name == "continue_task3_cf_after_task2.py" for item in argv):
                found.append(proc.name)
        except OSError:
            pass
    return found


def run_stage(stage, by_shard, orig_overlays, cf_overlays, work, poll,
              max_per_model, allow_failed=False):
    score = "Score_T3" if stage == "task3" else "S_counter"
    stage_label = "task3" if stage == "task3" else "cf223"
    result_name = "task3_results.json" if stage == "task3" else "cf_results.json"
    prefix = f"follow_{stage_label}_20_"
    tasks = []
    for model in MODELS:
        for number, all_rows in by_shard.items():
            rows = (all_rows if stage == "task3" else [
                r for r in all_rows if cf_branch(r) is not None])
            if rows:
                tasks.append((model, number, rows))
    target_keys = {model: {row["video_key"] for name, _, rows in tasks
                           if name == model for row in rows} for model in MODELS}
    while True:
        active = active_outputs()
        legacy_active = [name for name in active if re.fullmatch(
            rf"follow_{stage_label}_(?:cog|cosmos|ltx|wan)_\d{{2}}_a\d+", name)]
        if legacy_active:
            raise RuntimeError("Old 10-shard evaluators are still active: " +
                               ", ".join(sorted(legacy_active)[:5]))
        active_count = Counter(name.split("_")[3] for name in active
                               if name.startswith(prefix) and len(name.split("_")) > 3)
        scored_by_model = {}
        for model in MODELS:
            directories = [
                *RUNS.glob(f"follow_{stage_label}_{model}_[0-9][0-9]_a*"),
                *RUNS.glob(f"{prefix}{model}_[0-9][0-9]_a*"),
            ]
            scored = set()
            for folder in directories:
                complete_keys, _ = read_scores(folder / result_name, score)
                scored.update(complete_keys)
            if scored - target_keys[model]:
                raise RuntimeError(f"Unexpected {stage} result keys for {model}: "
                                   f"{list(scored - target_keys[model])[:3]}")
            scored_by_model[model] = scored
        complete, unresolved = Counter(), Counter()
        complete.update({model: len(keys) for model, keys in scored_by_model.items()})
        unresolved_keys = {model: [] for model in MODELS}
        for model, number, rows in tasks:
            name = f"{prefix}{model}_{number:02d}"
            attempts = sorted(RUNS.glob(name + "_a*"))
            pending = [r for r in rows if r["video_key"] not in scored_by_model[model]]
            if not pending:
                continue
            if any(folder.name in active for folder in attempts):
                continue
            if len(attempts) >= 3:
                if not allow_failed:
                    raise RuntimeError(f"{name}: {len(pending)} still missing after 3 attempts; inspect logs")
                unresolved[model] += len(pending)
                unresolved_keys[model].extend(row["video_key"] for row in pending)
                continue
            if active_count[model] >= max_per_model:
                continue
            attempt = len(attempts)
            run_name = name + f"_a{attempt}"
            annotation = work / "annotations" / f"{run_name}.json"
            atomic_json(annotation, pending)
            output = RUNS / run_name
            log = BASE / "logs" / f"{run_name}.log"
            command = [sys.executable, "-u", str(PROGRAM), "run",
                       "--annotations", str(annotation), "--orig-dir", str(orig_overlays[model]),
                       "--tasks", stage, "--limit", "0", "--output-dir", str(output)]
            if stage == "cf":
                command.extend(["--cf-dir", str(cf_overlays[model])])
            with log.open("ab") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            active.add(run_name)
            active_count[model] += 1
            say(f"START {run_name}: pending={len(pending)} pid={child.pid} log={log}")
        targets = Counter()
        for model, _, rows in tasks:
            targets[model] += len(rows)
        say(f"STATUS {stage}: " + " ".join(
            f"{m}=scored:{complete[m]}/{targets[m]},failed:{unresolved[m]}"
            for m in MODELS))
        if all(complete[m] + unresolved[m] == targets[m] for m in MODELS):
            for model in MODELS:
                if unresolved[model]:
                    say(f"FAILED {stage}/{model}: {unresolved[model]} sample(s), "
                        f"keys={unresolved_keys[model][:10]}; inspect {BASE / 'logs' / (prefix + model + '_*.log')}")
            return
        time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--max-per-model", type=int, default=20)
    parser.add_argument("--work", type=Path, default=BASE / "follow_task2")
    parser.add_argument("--cf-root", action="append", default=[], metavar="MODEL=PATH",
                        help="Put an actual CF video directory first for one model")
    parser.add_argument("--check-only", action="store_true",
                        help="Check all 223 original/CF source-video pairs; no API calls")
    parser.add_argument("--skip-task2-wait", action="store_true",
                        help="Start Task3 immediately with existing Task2 scores; record later failures and proceed to CF")
    args = parser.parse_args()
    if args.poll < 10 or not 1 <= args.max_per_model <= SPLITS:
        parser.error(f"--poll must be >= 10 and --max-per-model must be 1..{SPLITS}")
    for spec in args.cf_root:
        model, separator, path = spec.partition("=")
        if not separator or model not in MODELS or not path:
            parser.error("--cf-root must look like cosmos=/absolute/video/directory")
        CF[model].insert(0, Path(path))
    if not PROGRAM.is_file():
        raise RuntimeError(f"CF223 evaluator missing: {PROGRAM}")
    by_shard, expected = shards()
    cf_by_shard, cf_rows = all_cf_shards()
    if args.check_only:
        sys.path.insert(0, str(EVAL_BASE))
        from ewm_eval_local import flatten, offline_parsers, resolve_video
        t2, t3, cf = offline_parsers(EVAL_BASE)
        for row in cf_rows:
            item = flatten(row)
            t2.parse_sample(item)
            cf.parse_branches(item, [])
        problems = []
        for model in MODELS:
            found_orig = found_cf = 0
            for row in cf_rows:
                try:
                    find_source(resolve_video, [ORIG[model]], row["video_key"])
                    found_orig += 1
                except (OSError, ValueError) as exc:
                    problems.append((model, "orig", row["video_key"], str(exc)))
                try:
                    find_source(resolve_video, CF[model], row["video_key"], cf_branch(row))
                    found_cf += 1
                except (OSError, ValueError) as exc:
                    problems.append((model, "cf", row["video_key"], str(exc)))
            say(f"SOURCE CHECK {model}: orig={found_orig}/223 cf={found_cf}/223")
        for model, kind, key, reason in problems[:20]:
            say(f"MISSING {model}/{kind} {key}: {reason}")
        if problems:
            raise RuntimeError(f"{len(problems)} source-video matches missing or ambiguous; no API calls made")
        say("Source-video check passed; no API calls made")
        return
    old_pids = old_scheduler_pids()
    if old_pids:
        raise RuntimeError("Old 176-CF scheduler is still running (PIDs " +
                           ", ".join(old_pids) + "); stop only that scheduler first")
    for key in ("EWM_VLM_API_KEY", "EWM_LLM_API_KEY"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing environment variable: {key}")
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / "scheduler_cf223.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_outputs = sorted(name for name in active_outputs() if re.fullmatch(
            r"follow_(?:task3|cf223)_(?:cog|cosmos|ltx|wan)_\d{2}_a\d+", name))
        if old_outputs:
            raise RuntimeError("Stop the previous 10-shard evaluator processes "
                               "before starting 20 shards: " + ", ".join(old_outputs[:5]))
        say(f"ANNOTATIONS: Task3={len(expected)} CF={len(cf_rows)} "
            f"({SPLITS} shards per model; CF uses full source annotations)")
        for model, root in ORIG.items():
            if not root.is_dir():
                raise RuntimeError(f"Original-video directory missing for {model}: {root}")
            say(f"SOURCE {model}: orig={root}")
        for model, roots in CF.items():
            existing = [root for root in roots if root.is_dir()]
            say(f"SOURCE {model}: CF candidates=" + ", ".join(map(str, existing)))
            if not existing:
                raise RuntimeError(f"No CF directory found for {model}; supply --cf-root {model}=PATH")
        if args.skip_task2_wait:
            counts = {}
            for model in MODELS:
                done = set()
                for directory in task2_dirs(model):
                    scored, _ = read_scores(directory / "task2_results.json", "Score_T2")
                    done.update(scored)
                if done - expected:
                    raise RuntimeError(f"Task2 {model} has unexpected result keys")
                counts[model] = len(done)
            say("PARTIAL Task2 accepted: " + " ".join(
                f"{model}={counts[model]}/1451" for model in MODELS))
            say("Starting Task3 now; failed Task2 rows remain excluded from Task2 means")
        else:
            say("Scheduler ready; waiting for all four Task2 scores")
            wait_task2(expected, by_shard, args.work, args.poll)
            say("All Task2 samples scored; preparing Task3 media")
        sys.path.insert(0, str(EVAL_BASE))
        from ewm_eval_local import resolve_video
        all_rows = [row for rows in by_shard.values() for row in rows]
        orig_overlays, cf_overlays = {}, {}
        for model in MODELS:
            overlay = args.work / "media" / model / "orig"
            make_overlay(model, all_rows, "orig", [ORIG[model]], overlay, resolve_video)
            orig_overlays[model] = overlay
            preflight(model, "task3", all_rows, overlay, None, args.work)
        run_stage("task3", task3_shards(by_shard), orig_overlays, cf_overlays, args.work,
                  args.poll, args.max_per_model, allow_failed=args.skip_task2_wait)
        say("All Task3 samples scored; preparing CF media")
        cf_orig_overlays = {}
        for model in MODELS:
            original_overlay = args.work / "media" / model / "cf223_orig"
            make_overlay(model, cf_rows, "orig", [ORIG[model]], original_overlay, resolve_video)
            cf_orig_overlays[model] = original_overlay
            overlay = args.work / "media" / model / "cf223"
            make_overlay(model, cf_rows, "cf", CF[model], overlay, resolve_video)
            cf_overlays[model] = overlay
            preflight(model, "cf", cf_rows, original_overlay, overlay, args.work)
        run_stage("cf", cf_by_shard, cf_orig_overlays, cf_overlays, args.work,
                  args.poll, args.max_per_model, allow_failed=args.skip_task2_wait)
        say("DONE: Task3 and CF attempted for all selected samples; inspect STATUS for scored and failed counts")


if __name__ == "__main__":
    main()
