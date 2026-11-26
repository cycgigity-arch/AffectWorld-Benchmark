"""
EWH-Bench 顺序执行：Task 2 → Task 3 → Counterfactual，并汇总各任务的分项得分

用法：
  python run_all.py                  # 断点续跑：已成功的样本跳过
  python run_all.py --fresh          # 各步删除自己的旧结果后重评（缓存保留）
  python run_all.py --only task3 cf  # 只跑指定步骤（仍按固定顺序执行）
  python run_all.py --keep-going     # 某一步异常退出时继续执行后续步骤
  python run_all.py --report-only    # 不调用 API，只根据已有结果生成报告

报告只列各任务的得分与分项指标，不计算跨任务的综合分。
各步输出写入 run_all.log；报告写入 ewh_report.json。
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STEPS = [("task2", "eval_task2.py", "eval_task2"),
         ("task3", "eval_task3.py", "eval_task3"),
         ("cf",    "eval_cf.py",    "eval_cf")]
LOG_PATH = HERE / "run_all.log"
REPORT_PATH = HERE / "ewh_report.json"

# 各任务的得分指标（第一个为该任务的总分），取自各自的 summary 文件
SCORE_KEYS = {
    "task2": ["Score_T2", "M_emo", "S_action"],
    "task3": ["Score_T3", "S_must", "S_must_not", "S_direction", "S_exists"],
    "cf":    ["S_counter", "plausibility", "sel_orig", "sel_cf",
              "R_oo", "R_cc", "R_co", "R_oc"],
}
# 诊断指标：不属于得分，用于检查评分器与数据
DIAG_KEYS = {
    "task2": ["S_action_wo_expression", "e_star_is_target", "emotion_tie",
              "judge_instability", "open_emotion_in_candidates",
              "open_action_in_candidates"],
    "task3": ["judge_instability", "entity_not_found_rate"],
    "cf":    ["L_mean", "same_setting_rate"],
}
# 样本级表格的列：(表头, results 中的字段)
SAMPLE_COLS = {
    "task2": [("T2", "Score_T2"), ("M_emo", "M_emo"), ("S_action", "S_action"),
              ("e*", "e_star"), ("a*", "a_star")],
    "task3": [("T3", "S_causal"), ("must", "S_must"), ("must_not", "S_must_not"),
              ("dir", "S_direction"), ("exists", "S_exists"),
              ("dir_unobs", "K_dir_unobservable")],
    "cf":    [("S_cf", "S_counter"), ("sel_o", "sel_orig"), ("sel_c", "sel_cf"),
              ("R_oo", "R_oo"), ("R_cc", "R_cc"), ("R_co", "R_co"), ("R_oc", "R_oc")],
}
TASK_TITLES = {"task2": "Task 2: Emotion-Action Consistency",
               "task3": "Task 3: Causal Consistency",
               "cf":    "Counterfactual Chain Evaluation"}


# ---------------------------------------------------------------- 工具
def log(msg, fh=None):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def result_paths(module):
    """从各模块读取 OUT_PATH / SUMMARY_PATH，保证与脚本配置一致（导入不会调用 API）。"""
    mod = __import__(module)
    return HERE / mod.OUT_PATH, HERE / mod.SUMMARY_PATH


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def fmt(x, width=0):
    if isinstance(x, int) and not isinstance(x, bool):
        s = str(x)                      # 计数类字段按整数显示
    elif is_num(x):
        s = f"{x:.4f}"
    else:
        s = "-" if x is None else str(x)
    return f"{s:>{width}}" if width else s

# ---------------------------------------------------------------- 运行
def run_step(script, extra_args, fh):
    """运行一个评测脚本，按字节实时转发（保留 tqdm 进度条），日志只记每行最终状态。"""
    proc = subprocess.Popen([sys.executable, "-u", script, *extra_args], cwd=HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=0)
    fd, pending = proc.stdout.fileno(), b""
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            fh.write(line.decode("utf-8", errors="replace").split("\r")[-1] + "\n")
    if pending:
        fh.write(pending.decode("utf-8", errors="replace").split("\r")[-1] + "\n")
    fh.flush()
    return proc.wait()


def count_failed(out_path):
    return [r.get("video_key") for r in load_json(out_path, []) if "error" in r]


# ---------------------------------------------------------------- 报告
def build_report(paths):
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "tasks": {}}
    for name, (out_path, summary_path) in paths.items():
        summary = load_json(summary_path, {})
        results = load_json(out_path, [])
        ok = [r for r in results if "error" not in r]
        task = {
            "n": summary.get("n", len(ok)),
            "n_failed": len(results) - len(ok),
            "scores": {k: summary.get(k) for k in SCORE_KEYS[name]},
            "diagnostics": {k: summary.get(k) for k in DIAG_KEYS[name]
                            if summary.get(k) is not None},
            "samples": {r["video_key"]: {col: r.get(key)
                                         for col, key in SAMPLE_COLS[name]}
                        for r in ok},
        }
        # Task 3：各约束类别的得分与不可观测比例
        if name == "task3" and summary.get("by_constraint_kind"):
            task["by_constraint_kind"] = {
                k: {"n": v.get("n"), "item_score": v.get("item_score"),
                    "not_observable_rate": v.get("not_observable_rate")}
                for k, v in summary["by_constraint_kind"].items()}
        if name == "cf":
            task["note"] = "computed only on samples with counterfactual videos"
        report["tasks"][name] = task
    return report


def print_report(rep, fh=None):
    log("=" * 78, fh)
    log("EWH-Bench report (per-task scores; no composite index)", fh)
    for name, t in rep["tasks"].items():
        log("", fh)
        log(f"==== {TASK_TITLES[name]}   n={t['n']}  failed={t['n_failed']}"
            + (f"   ({t['note']})" if t.get("note") else ""), fh)
        keys = list(t["scores"]) + list(t["diagnostics"])
        kw = max(len(k) for k in keys) + 2 if keys else 24
        log("  scores:", fh)
        for k, v in t["scores"].items():
            log(f"    {k:<{kw}}{fmt(v)}", fh)
        if t["diagnostics"]:
            log("  diagnostics (not part of the score):", fh)
            for k, v in t["diagnostics"].items():
                log(f"    {k:<{kw}}{fmt(v)}", fh)
        if t.get("by_constraint_kind"):
            log("  by constraint category / kind:", fh)
            for k, v in t["by_constraint_kind"].items():
                log(f"    {k:<36} n={v['n']:<3} item_score={fmt(v['item_score']):<8}"
                    f"  not_observable={fmt(v['not_observable_rate'])}", fh)
        if t["samples"]:
            cols = [c for c, _ in SAMPLE_COLS[name]]
            rows = t["samples"]
            # 每列宽度取表头与所有取值中最长者 + 2
            widths = {c: max([len(c)] + [len(fmt(r[c])) for r in rows.values()]) + 2
                      for c in cols}
            vw = max(len(vk) for vk in rows) + 2
            log("  per sample:", fh)
            log(f"    {'video_key':<{vw}}" + "".join(f"{c:>{widths[c]}}" for c in cols), fh)
            for vk, row in rows.items():
                log(f"    {vk:<{vw}}" + "".join(fmt(row[c], widths[c]) for c in cols), fh)
    log("", fh)
    log(f"Report written to {REPORT_PATH.name}", fh)


def finish(all_paths, fh=None):
    rep = build_report(all_paths)
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print_report(rep, fh)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="各步删除自己的旧结果（results/summary/review）后重评，缓存保留")
    ap.add_argument("--only", nargs="+", choices=[s[0] for s in STEPS],
                    help="只运行指定步骤")
    ap.add_argument("--keep-going", action="store_true",
                    help="某一步异常退出时继续执行后续步骤")
    ap.add_argument("--report-only", action="store_true",
                    help="不调用 API，只根据已有结果生成报告")
    args = ap.parse_args()

    # 报告始终覆盖全部任务；--only 只限定本次运行的步骤
    all_paths = {name: result_paths(mod) for name, _, mod in STEPS}

    if args.report_only:
        finish(all_paths)
        return

    steps = [s for s in STEPS if not args.only or s[0] in args.only]
    extra = ["--fresh"] if args.fresh else []

    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        log("=" * 78, fh)
        log(f"run_all: steps={[s[0] for s in steps]} fresh={args.fresh}", fh)

        status = {}
        for name, script, _ in steps:
            log(f"---- {name}: {script} {' '.join(extra)} ----", fh)
            t0 = time.time()
            code = run_step(script, extra, fh)
            failed = count_failed(all_paths[name][0])
            log(f"{name} exited with code {code} in {time.time() - t0:.0f}s; "
                f"failed samples: {len(failed)}", fh)
            if failed:
                log(f"  failed: {', '.join(map(str, failed))}", fh)
            status[name] = (code, len(failed))
            if code != 0 and not args.keep_going:
                log(f"stopped: {name} exited abnormally; later steps skipped", fh)
                break

        log("==== run status ====", fh)
        for name, (code, n_failed) in status.items():
            s = ("ok" if code == 0 and n_failed == 0 else
                 "sample failures" if code == 0 else "exited abnormally")
            log(f"  {name:<6} {s}  (exit={code}, failed_samples={n_failed})", fh)
        skipped = [s[0] for s in steps if s[0] not in status]
        if skipped:
            log(f"  skipped: {', '.join(skipped)}", fh)

        finish(all_paths, fh)

    sys.exit(0 if all(c == 0 for c, _ in status.values()) and not skipped else 1)


if __name__ == "__main__":
    main()
