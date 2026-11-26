"""Score available Hunyuan videos with the existing experimental Task1 scorer.

One video is scored at a time on the local GPU. Successful results are indexed
by the original annotation video_key and skipped on restart. No VLM/LLM API is
used. The scorer's per-video artifacts remain available for mask review.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ewm_eval_local import resolve_video


ROOT = Path("/root/autodl-tmp")
BUNDLE = Path(__file__).resolve().parent


def plan(annotations: Path, video_root: Path) -> list[tuple[str, Path]]:
    rows = json.loads(annotations.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or len(rows) != 1498:
        raise ValueError(f"Expected 1498 annotations: {annotations}")
    found, paths, missing, invalid = [], set(), [], []
    for row in rows:
        key = row["video_key"]
        try:
            path = resolve_video(video_root, key).resolve()
            if path in paths:
                raise ValueError(f"same Hunyuan file maps to multiple annotation keys: {path}")
            if path.stat().st_size == 0:
                raise ValueError(f"empty video: {path}")
            found.append((key, path))
            paths.add(path)
        except FileNotFoundError:
            missing.append(key)
        except (OSError, ValueError) as error:
            invalid.append((key, str(error)))
    print(f"Hunyuan Task1 source: available {len(found)}/1498, missing {len(missing)}, "
          f"invalid {len(invalid)}", flush=True)
    for key, reason in invalid[:12]:
        print(f"INVALID {key}: {reason}", flush=True)
    if invalid:
        raise RuntimeError("Ambiguous or invalid Hunyuan videos need inspection before scoring")
    if not found:
        raise RuntimeError("No Hunyuan video matches the Task1 annotations")
    return found


def successful_keys(index: Path) -> set[str]:
    keys = set()
    if not index.is_file():
        return keys
    with index.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A later run can recover an interrupted final line.
            if record.get("status") == "scored" and isinstance(record.get("video_key"), str):
                keys.add(record["video_key"])
    return keys


def score_one(key: str, video: Path, args: argparse.Namespace, index: Path) -> bool:
    count_before = {path for path in args.artifacts.iterdir() if path.is_dir()}
    command = [str(args.score_python), "-u", str(args.scorer),
               "--video", str(video), "--protocol", str(args.protocol),
               "--vfi-python", str(args.vfi_python), "--vfi-repo", str(args.vfi_repo),
               "--output-dir", str(args.artifacts)]
    env = os.environ.copy()
    env["HF_HOME"] = str(args.hf_home)
    env["HF_HUB_OFFLINE"] = "1"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    log = args.logs / f"{digest}.log"
    with log.open("ab") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, check=False, env=env)
    new_dirs = sorted((path for path in args.artifacts.iterdir()
                       if path.is_dir() and path not in count_before),
                      key=lambda path: path.stat().st_mtime_ns)
    if len(new_dirs) != 1:
        raise RuntimeError(f"Expected one Task1 scorer output directory for {key}; log={log}")
    report = new_dirs[0] / "results.jsonl"
    lines = report.read_text(encoding="utf-8").splitlines() if report.is_file() else []
    if len(lines) != 1:
        raise RuntimeError(f"Expected one Task1 result for {key}; log={log} report={report}")
    record = json.loads(lines[0])
    record["video_key"] = key
    record["kind"] = "generated_future"
    record["source"] = str(video)
    record["scorer_result"] = str(report)
    record["scorer_exit_code"] = completed.returncode
    record["scorer_log"] = str(log)
    if completed.returncode and record.get("status") == "scored":
        raise RuntimeError(f"Scorer exited {completed.returncode} despite a scored record: {log}")
    if index.is_file() and index.stat().st_size:
        with index.open("rb") as existing:
            existing.seek(-1, 2)
            if existing.read(1) != b"\n":
                with index.open("ab") as repair:
                    repair.write(b"\n")
    with index.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    if record.get("status") != "scored":
        print(f"ERROR {key}: {record.get('error', 'scorer returned no error detail')}; "
              f"scorer_exit_code={completed.returncode}; log={log}", flush=True)
    return record.get("status") == "scored"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path,
                        default=ROOT / "ewm_generated_hunyuan/hunyuan_i2v_5s")
    parser.add_argument("--annotations", type=Path,
                        default=BUNDLE / "private_annotations_1498_normalized.json")
    parser.add_argument("--score-python", type=Path,
                        default=ROOT / "task1-eval-env/bin/python")
    parser.add_argument("--scorer", type=Path, default=BUNDLE / "task1_full_score.py")
    parser.add_argument("--protocol", type=Path,
                        default=BUNDLE / "task1_full_score_trial_protocol.json")
    parser.add_argument("--vfi-python", type=Path,
                        default=ROOT / "vfimamba-env/bin/python")
    parser.add_argument("--vfi-repo", type=Path, default=ROOT / "VFIMamba")
    parser.add_argument("--hf-home", type=Path,
                        default=ROOT / "cache/huggingface",
                        help="Reuse the local model cache used by the previous Task1 runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hunyuan_task1_scores")
    parser.add_argument("--limit", type=int, help="Score only the first N pending videos")
    parser.add_argument("--check-only", action="store_true",
                        help="Report available file mappings without loading models")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not args.video_root.is_dir():
        raise RuntimeError(f"Hunyuan video root missing: {args.video_root}")
    jobs = plan(args.annotations, args.video_root)
    if args.check_only:
        print("Source check complete; no GPU or API calls made", flush=True)
        return
    for path in (args.score_python, args.scorer, args.protocol, args.vfi_python):
        if not path.is_file():
            raise RuntimeError(f"Required Task1 file missing: {path}")
    if not args.vfi_repo.is_dir():
        raise RuntimeError(f"VFIMamba repository missing: {args.vfi_repo}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts = args.output_dir / "artifacts"
    args.logs = args.output_dir / "logs"
    args.artifacts.mkdir(exist_ok=True)
    args.logs.mkdir(exist_ok=True)
    index = args.output_dir / "results.jsonl"
    with (args.output_dir / "task1_hunyuan.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = successful_keys(index)
        available_keys = {key for key, _ in jobs}
        remaining = [(key, video) for key, video in jobs if key not in done]
        if args.limit is not None:
            remaining = remaining[:args.limit]
        print(f"Task1 Hunyuan: already scored={len(done & available_keys)}/{len(jobs)} "
              f"(coverage={len(jobs)}/1498), this run={len(remaining)}",
              flush=True)
        print(f"HF_HOME={args.hf_home}; HF_HUB_OFFLINE=1", flush=True)
        errors = 0
        consecutive_failures = 0
        for ordinal, (key, video) in enumerate(remaining, 1):
            try:
                scored = score_one(key, video, args, index)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                scored = False
                print(f"ERROR {key}: {exc}", flush=True)
            if not scored:
                errors += 1
                consecutive_failures += 1
            else:
                done.add(key)
                consecutive_failures = 0
            print(f"PROGRESS {ordinal}/{len(remaining)}; "
                  f"total scored={len(done & available_keys)}/{len(jobs)} "
                  f"coverage={len(jobs)}/1498; "
                  f"errors_this_run={errors}", flush=True)
            if consecutive_failures >= 3:
                raise RuntimeError("Three consecutive Task1 failures; inspect the latest logs before retrying")
        print(f"RESULTS {index}", flush=True)
        if errors:
            raise RuntimeError(f"{errors} Task1 samples failed; inspect logs and rerun to retry")


if __name__ == "__main__":
    main()
