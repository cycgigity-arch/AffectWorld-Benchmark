#!/usr/bin/env python3
"""Summarize completed EWM Task2/Task3 runs for Cog, Cosmos, LTX, Wan, Hunyuan.

Read-only. Standard library only. No API keys or model downloads are needed.
Each successful video_key contributes once, even when it appears in retry runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


MODELS = ("cog", "cosmos", "ltx", "wan", "hunyuan")
FIELDS = {
    "task2": ("Score_T2", "M_emo", "S_action"),
    "task3": ("Score_T3",),
}
RESULT_FILE = {"task2": "task2_results.json", "task3": "task3_results.json"}
DEFAULT_BASE = Path("/root/autodl-tmp/ewm_eval_local")


def source_keys(base: Path) -> set[str]:
    shard_dir = base / "shards/20261010T171904052041Z"
    paths = [shard_dir / f"all_{i:02d}.json" for i in range(10)]
    keys = []
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list: {path}")
        keys.extend(row["video_key"] for row in rows)
    if len(keys) != 1451 or len(set(keys)) != 1451:
        raise ValueError(f"Expected 1451 distinct Task2/Task3 video_keys, got {len(keys)}")
    return set(keys)


def run_dirs(runs: Path, stage: str, model: str) -> list[Path]:
    if stage == "task2":
        patterns = [f"{model}_task2_*", f"task2_cf223_recovery_{model}_a*"]
        paths = [runs / "20261010T171904052041Z"] if model == "cog" else []
    else:
        patterns = [
            f"follow_task3_{model}_[0-9][0-9]_a*",
            f"follow_task3_20_{model}_[0-9][0-9]_a*",
            f"resume_task3_20_*_{model}_[0-9][0-9]_a*",
        ]
        paths = []
    for pattern in patterns:
        paths.extend(runs.glob(pattern))
    return sorted({path for path in paths if path.is_dir()},
                  key=lambda path: (path.stat().st_mtime_ns, str(path)))


def valid_score(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def read_stage(runs: Path, stage: str, model: str, expected: set[str]):
    """Newest successful retry wins; an earlier error never hides later success."""
    successful = {}
    error_keys = set()
    duplicates = set()
    conflicting = set()
    files = []
    for directory in run_dirs(runs, stage, model):
        path = directory / RESULT_FILE[stage]
        if not path.is_file():
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"Expected a list: {path}")
        files.append(path)
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("video_key"), str):
                raise ValueError(f"Invalid result record in {path}")
            key = record["video_key"]
            if key not in expected:
                raise ValueError(f"Unexpected video_key {key!r} in {path}")
            if "error" in record:
                error_keys.add(key)
                continue
            score_field = FIELDS[stage][0]
            if not valid_score(record.get(score_field)):
                raise ValueError(f"Missing/invalid {score_field} for {key} in {path}")
            for field in FIELDS[stage][1:]:
                if not valid_score(record.get(field)):
                    raise ValueError(f"Missing/invalid {field} for {key} in {path}")
            if key in successful:
                duplicates.add(key)
                if any(abs(successful[key][field] - record[field]) > 1e-8
                       for field in FIELDS[stage]):
                    conflicting.add(key)
            successful[key] = record
    if not files:
        raise FileNotFoundError(f"No {RESULT_FILE[stage]} found for {model} under {runs}")
    return successful, error_keys - successful.keys(), duplicates, conflicting, len(files)


def hunyuan_target(base: Path, expected: set[str]) -> set[str] | None:
    path = base / "hunyuan_task2_cf_pipeline/task2_selected_keys.json"
    if not path.is_file():
        return None
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
        raise ValueError(f"Invalid Hunyuan selected keys: {path}")
    keys = set(values)
    if len(keys) != len(values) or not keys or keys - expected:
        raise ValueError(f"Invalid Hunyuan selected cohort: {path}")
    return keys


def mean(records: dict, keys: set[str], field: str) -> float | None:
    values = [records[key][field] for key in keys]
    return sum(values) / len(values) if values else None


def metric(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def video_rows(data: dict, failed_keys: dict, expected: set[str],
               available_hunyuan: set[str] | None, common: set[str]):
    """One line per available model/video pair, including failed or missing scores."""
    task2_fields = ("Score_T2", "M_emo", "S_action", "e_star", "a_star")
    task3_fields = ("Score_T3", "S_causal", "S_must", "S_must_not",
                    "S_direction", "S_exists", "K_must", "K_must_not", "K_dir")
    for model in MODELS:
        t2, t3 = data[model]["task2"], data[model]["task3"]
        target = (available_hunyuan if model == "hunyuan" else expected)
        if target is None:
            target = (set(t2) | set(t3) | failed_keys[model]["task2"]
                      | failed_keys[model]["task3"])
        for key in sorted(target):
            row = {"model": model, "video_key": key,
                   "in_five_model_common_cohort": key in common}
            for stage, records, fields in (("task2", t2, task2_fields),
                                           ("task3", t3, task3_fields)):
                record = records.get(key)
                row[f"{stage}_status"] = (
                    "scored" if record is not None else
                    "error" if key in failed_keys[model][stage] else "missing")
                for field in fields:
                    row[field] = record.get(field) if record is not None else None
            yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path,
                        help="Output directory (default: BASE/reports/five_model_task2_task3)")
    parser.add_argument("--include-full-records", action="store_true",
                        help="Also export complete original Task2/Task3 records to JSONL")
    args = parser.parse_args()
    base = args.base.resolve()
    out = (args.out or base / "reports/five_model_task2_task3").resolve()
    expected = source_keys(base)
    available_hunyuan = hunyuan_target(base, expected)
    data, failed_keys, audit = {}, {}, {}
    for model in MODELS:
        data[model], failed_keys[model], audit[model] = {}, {}, {}
        for stage in FIELDS:
            records, errors, duplicates, conflicts, file_count = read_stage(
                base / "eval_runs", stage, model, expected)
            data[model][stage] = records
            failed_keys[model][stage] = errors
            audit[model][stage] = {
                "result_files": file_count,
                "error_keys_not_recovered": len(errors),
                "duplicate_success_keys": len(duplicates),
                "conflicting_success_keys": len(conflicts),
            }
    if available_hunyuan is not None:
        for stage in FIELDS:
            unknown = data["hunyuan"][stage].keys() - available_hunyuan
            if unknown:
                raise ValueError(f"Hunyuan {stage} scored outside its selected cohort: {list(unknown)[:3]}")

    # Use one identical set of video_keys for all ten model/task combinations.
    common = set.intersection(*(set(data[m][s]) for m in MODELS for s in FIELDS))
    own_rows, common_rows = [], []
    for model in MODELS:
        t2, t3 = data[model]["task2"], data[model]["task3"]
        target = available_hunyuan if model == "hunyuan" else expected
        own_rows.append({
            "model": model,
            "available_videos": len(target) if target is not None else "",
            "task2_scored": len(t2),
            "task2_unscored": len(target - t2.keys()) if target is not None else "",
            "M_emo": metric(mean(t2, set(t2), "M_emo")),
            "S_action": metric(mean(t2, set(t2), "S_action")),
            "Score_T2": metric(mean(t2, set(t2), "Score_T2")),
            "task3_scored": len(t3),
            "task3_unscored": len(target - t3.keys()) if target is not None else "",
            "Score_T3": metric(mean(t3, set(t3), "Score_T3")),
        })
        common_rows.append({
            "model": model,
            "common_videos": len(common),
            "M_emo": metric(mean(t2, common, "M_emo")),
            "S_action": metric(mean(t2, common, "S_action")),
            "Score_T2": metric(mean(t2, common, "Score_T2")),
            "Score_T3": metric(mean(t3, common, "Score_T3")),
        })
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "scores_all_available.csv", own_rows,
             list(own_rows[0]))
    save_csv(out / "scores_five_model_common.csv", common_rows,
             list(common_rows[0]))
    with (out / "video_scores.jsonl").open("w", encoding="utf-8") as stream:
        for row in video_rows(data, failed_keys, expected, available_hunyuan, common):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    if args.include_full_records:
        with (out / "video_results_full.jsonl").open("w", encoding="utf-8") as stream:
            for row in video_rows(data, failed_keys, expected, available_hunyuan, common):
                model, key = row["model"], row["video_key"]
                full = {"model": model, "video_key": key,
                        "task2_status": row["task2_status"],
                        "task3_status": row["task3_status"],
                        "task2": data[model]["task2"].get(key),
                        "task3": data[model]["task3"].get(key)}
                stream.write(json.dumps(full, ensure_ascii=False, allow_nan=False) + "\n")
    (out / "audit.json").write_text(json.dumps({
        "protocol_videos": len(expected),
        "hunyuan_available_videos": len(available_hunyuan) if available_hunyuan else None,
        "five_model_two_task_common_videos": len(common),
        "details": audit,
        "notes": [
            "All scores are arithmetic means over unique, successfully scored video_keys.",
            "A later successful retry replaces an earlier successful record for the same key.",
            "scores_all_available.csv uses each model's own scored cohort.",
            "scores_five_model_common.csv uses the same keys for all five models and both tasks.",
            "video_scores.jsonl has one line per available model/video pair; null scores mean unscored.",
            "With --include-full-records, video_results_full.jsonl contains complete original records.",
            "Task2 and Task3 remain separate scores; no cross-task composite is computed.",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("各自已评分样本的均分（空白表示无可用值）：")
    print("model     可用  T2成功  T2缺  M_emo  S_action  Score_T2  T3成功  T3缺  Score_T3")
    for r in own_rows:
        print(f"{r['model']:<8} {str(r['available_videos']):>4} {r['task2_scored']:>7} "
              f"{str(r['task2_unscored']):>5} {r['M_emo']:>6} {r['S_action']:>9} "
              f"{r['Score_T2']:>9} {r['task3_scored']:>7} "
              f"{str(r['task3_unscored']):>5} {r['Score_T3']:>9}")
    print(f"\n五模型、两任务共同样本：{len(common)} 条")
    print("model     M_emo  S_action  Score_T2  Score_T3")
    for r in common_rows:
        print(f"{r['model']:<8} {r['M_emo']:>6} {r['S_action']:>9} "
              f"{r['Score_T2']:>9} {r['Score_T3']:>9}")
    for model in MODELS:
        for stage in FIELDS:
            if audit[model][stage]["conflicting_success_keys"]:
                print(f"注意：{model}/{stage} 有 "
                      f"{audit[model][stage]['conflicting_success_keys']} 条重复成功结果分数不同；"
                      "已采用最新结果，详见 audit.json")
    if available_hunyuan is None:
        print("注意：未找到 Hunyuan 可用样本清单，因此其可用数/未评分数留空。")
    print(f"\n已写入：{out}")


if __name__ == "__main__":
    main()
