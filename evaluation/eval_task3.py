"""
EWH-Bench Task 3: Emotion-driven Social World Evolution
流程：读取视频 → VLM 描述（复用 Task 2 缓存）→ LLM 先列变化、再逐条核对约束
     → S_must / S_must_not / S_direction / S_exists → S_causal

依赖：与 eval_task2.py 放在同一目录。Key、模型、VLM prompt、路径等配置全部复用 eval_task2.py，
     本文件只包含 Task 3 专属的少量配置。
运行：python eval_task3.py
对照：RUN_GROUPS 默认依次运行正常组与错配组；验证完成后改为 [("none",)] 只跑正常评测。
"""
import re, json, math, time, random, logging, argparse
from collections import defaultdict, Counter
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import eval_task2 as t2      # 若 Task 2 文件改名，这里同步修改


# #####################################################################
# 1. Task 3 专属配置（其余配置见 eval_task2.py）
# #####################################################################

OUT_PATH = "task3_results.json"
SUMMARY_PATH = "task3_results.summary.json"
JUDGE_CACHE_PATH = Path(t2.CACHE_DIR) / "task3_judge.json"
LOG_FILE = "task3_eval.log"
REVIEW_PATH = "task3_review.json"   # 精简评审文件，贴这个即可

# 对照实验：依次运行下列各组。
#   ("none",)     正常组 A：视频与样本匹配
#   ("mismatch",) 错配组 B：每个样本换一个目标情绪大类不同的视频，S_causal 应明显下降
# 验证完成后改为 [("none",)]
# RUN_GROUPS = [("none",), ("mismatch",)]
RUN_GROUPS = [("none",)]
CONTROL_MODE = "none"       # 运行时由 RUN_GROUPS 自动设置，不用改

SHUFFLE_ITEMS = True        # 每次评分打乱约束顺序
QC_LOW_SCORE = 0.5          # 低于该值的样本 / 约束会被标记（参考视频用于标注质检）

# 判定结果 → 数值（None 表示该约束不可观测，不计入分母）
MUST_SCORE = {"satisfied": 1.0, "partial": 0.5, "not_satisfied": 0.0}
MUST_NOT_VIOLATION = {"violated": 1.0, "partially_violated": 0.5, "not_violated": 0.0}
DIRECTION_SCORE = {"correct": 1.0, "no_change": 0.0, "opposite": 0.0,
                   "not_observable": None}
VERDICT_TABLE = {"must": MUST_SCORE, "must_not": MUST_NOT_VIOLATION,
                 "directional": DIRECTION_SCORE}
# 找不到实体、主体不可见或判定无效时的降级结果
FALLBACK = {"must": "not_satisfied", "must_not": "not_violated",
            "directional": "no_change"}
# 这些判定不需要证据
NO_EVIDENCE_OK = {"not_satisfied", "not_violated", "no_change", "not_observable"}

CONSTRAINT_TYPES = ("must", "must_not", "directional")
ID_PREFIX = {"must": "M", "must_not": "N", "directional": "D"}
WORLD_DIMENSIONS = {"physical_object", "spatial_relation", "agent_internal",
                    "social_atmosphere"}
NEGATIVE_PHRASE_RE = re.compile(r"(^|[_\s])(no|not|without|never)([_\s])",
                                re.IGNORECASE)

logger = logging.getLogger("task3")
logger.setLevel(logging.INFO)
logger.propagate = False
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_fh)


# #####################################################################
# 2. Prompt
# #####################################################################

JUDGE_SYSTEM = """
You are a strict, evidence-bound evaluator for an emotion-aware world-model
benchmark. You judge only from a written description of a video. You never
invent observations, never complete the story from common sense, never assume
that a constraint must be satisfied, and you apply the verdict definitions
literally. Output one valid JSON object and nothing else.
"""

GROUNDING_RULES = """
GROUNDING RULES
1. The video description is the ONLY source of observations. The context, the
   event, and the initial state describe the situation at or before the start
   of the video; use them to identify entities and to understand the starting
   point, never as evidence that a change happened in the video.
2. Every evidence item must start with a [T=XX%] anchor that appears in the
   description and must paraphrase what the description states at that
   anchor. Never cite anything the description does not state.
3. A change that is not mentioned did not happen.
4. Camera movement, cuts, zooms, reframing, and subtitles are not world
   changes. An entity that becomes visible or invisible only because of
   framing has not changed in the world.
5. Constraints name entities with identifiers. If the INPUT lists entity
   identifiers, use those definitions; otherwise map each identifier to the
   most plausible described entity. If no described entity matches, set
   "entity_found" to false. If the INPUT names a Main subject, "main_subject"
   in the output refers to that person; set "subject_visible" to false if no
   described person matches it.
6. Match on CORE MEANING, not on wording: a reported state that realizes the
   same core as the constraint in different words counts (e.g., "her gaze
   drops to the table" realizes "downcast"). A state that only looks similar
   but has a different function does not count.
7. The description was written under a rule that forbids asserting speech
   when audio is unclear. An open or moving mouth directed at another person
   is a valid cue of vocal or verbal activity.
8. Judge each constraint independently. The listing order is random and
   carries no meaning.
9. For a spatial relation whose subject is a single entity, judge the change
   produced by the subject's own movement, gaze, or orientation (e.g.,
   "mother -> son, distance, decrease" is realized when the mother moves or
   leans closer). For a group subject, judge the relation as a whole.
10. "observable as" lists acceptable observable indicators. A constraint is
   realized when any listed indicator, or a cue with the same core meaning,
   is reported; the indicators are not all required.
11. "from" describes the expected starting state for reference only; the
   video does not need to show it. However, a "to" state that is already
   present at the start and does not change does not satisfy a must item.
   This exception does not apply when the entity is not visible at the start
   (e.g., it first appears after a cut): then reaching the "to" state is
   judged on its own.
12. The description reports gaze and orientation as directions, not targets.
   A gaze or orientation target may be established from the description's
   own spatial information: the reported direction matches the reported
   location of the target, or the subject acts toward that target (points,
   turns, moves) while keeping that direction. Do not use the context or the
   event to establish the target.
"""


# 必须包含变量：{input_block} {grounding_rules} {constraint_block}
CAUSAL_PROMPT = """
TASK: Check whether the video, as described, satisfies world-state constraints
describing how the scene should evolve after the event. Work in two steps:
first list the observed world-state changes WITHOUT looking at the
constraints, then judge each constraint.

{input_block}

{grounding_rules}

STEP 1 - OBSERVED CHANGES (constraint-independent)
List every change in the world state that the description reports during the
video, each with its anchor and exactly one dimension:
- physical_object: position, attributes, existence, or contact of a movable
  object
- spatial_relation: gaze, distance, contact, occlusion, or orientation between
  two or more entities
- agent_internal: facial expression, posture, action pattern, speech, or
  attention focus of a person
- social_atmosphere: conversation rhythm, group distribution, volume, energy,
  or interaction mode among several people
Set "world_state_changed" to true if at least one such change is reported.

STEP 2 - CONSTRAINTS
Each constraint has a category (must, must_not, directional), a kind
(agent_state, spatial_relation, object_state, social_atmosphere, ...), and
structured fields. "from" is the expected state at or before the start
(reference only); "to" is the required end state; "change" is the required
change or direction; "observable as" lists acceptable observable indicators.
{constraint_block}

VERDICT DEFINITIONS
must (a change that any plausible future should contain):
  satisfied     = the description reports the matched entity reaching the
                  stated end state, or a state with the same core meaning,
                  and holding it for more than a moment during the video,
                  and that state was not already present unchanged from the
                  start. Leaving the state later does not undo this, unless
                  the constraint explicitly concerns the final state.
  partial       = the entity moves toward the stated state, but the change is
                  incomplete, momentary, or ambiguous, or only part of a
                  multi-part statement is realized.
  not_satisfied = not reported, the entity is not found, or a contradicting
                  state is reported.
must_not (a change that no plausible future should contain; judge whether this
change HAPPENS):
  violated           = the stated change is clearly reported.
  partially_violated = a weaker, brief, or partial form of it is reported.
  not_violated       = it is not reported.
  must_not concerns a change that HAPPENS after the event. A prohibited state
  that is already present at the start is not a violation, whether the entity
  then leaves it or keeps it unchanged; a violation requires the entity to
  enter the prohibited state, or to return to it after having left it.

directional (a quantity that should move in the stated direction; the exact
end state is not required):
  correct        = a change in the stated direction is reported.
  no_change      = the entity is reported both before and after the event,
                   and the quantity does not change in the stated direction.
  opposite       = a change in the opposite direction is reported.
  not_observable = the change cannot be judged from the description: the
                   relevant entity is never reported after the event; or its
                   state before the event is never reported (e.g., it first
                   appears after a cut) and no change is reported while it
                   is visible; or the only possible cue would depend on audio.

VERDICT CONSTRAINTS
- Every verdict other than not_satisfied, not_violated, no_change, or
  not_observable requires at least one evidence item with a valid anchor.
- If "entity_found" is false, the verdict must be not_satisfied (must),
  not_violated (must_not), or no_change (directional).

OUTPUT (exactly this JSON structure; one entry per constraint; copy the ids
exactly)
{
  "main_subject": "<short description of the main subject>",
  "subject_visible": true,
  "observed_changes": [
    {"dimension": "<physical_object | spatial_relation | agent_internal | social_atmosphere>",
     "change": "[T=XX%] ..."}
  ],
  "world_state_changed": false,
  "constraints": [
    {
      "id": "<constraint id>",
      "entity_found": true,
      "entity_match": "<the described entity this identifier refers to>",
      "analysis": "<which reported change, if any, realizes or contradicts this constraint, at the level of core meaning>",
      "evidence": ["[T=XX%] ..."],
      "counter_evidence": ["[T=XX%] ..."],
      "verdict": "<one verdict from the definitions for this category>"
    }
  ]
}
"""



# #####################################################################
# 3. 约束解析与展示
# #####################################################################

def humanize(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v).replace("_", " ").strip()


def parse_constraints(raw_items, warnings):
    """返回 [{"id", "ctype", "kind", "item"}]。只接受 {must, must_not, directional} 字典格式。"""
    if not isinstance(raw_items, dict):
        raise ValueError("world_state_delta_items must be a dict with keys "
                         "must / must_not / directional "
                         f"(got {type(raw_items).__name__})")
    cats = {}
    for k, v in raw_items.items():
        nk = t2.norm_key(k)
        if nk in CONSTRAINT_TYPES:
            cats[nk] = v
        else:
            warnings.append(f"ignored unknown constraint category {k!r}")

    items = []
    for ctype in CONSTRAINT_TYPES:
        for i, it in enumerate(cats.get(ctype) or [], 1):
            cid = f"{ID_PREFIX[ctype]}{i}"
            if not isinstance(it, dict) or not it:
                warnings.append(f"{cid}: dropped empty or non-object item")
                continue
            if ctype == "must_not" and any(
                    NEGATIVE_PHRASE_RE.search(str(v))
                    for k, v in it.items() if k not in ("type", "observable_as")):
                warnings.append(f"{cid}: must_not item is phrased negatively; "
                                f"verify polarity in the annotation")
            items.append({"id": cid, "ctype": ctype,
                          "kind": it.get("type", "unspecified"), "item": it})
    return items



def render_constraint(c):
    fields = " | ".join(f"{k}: {humanize(v)}"
                        for k, v in c["item"].items() if k != "type")
    return f"- [{c['id']}] {c['ctype']} | {humanize(c['kind'])} | {fields}"


def constraint_block(items, seed_text):
    order = list(items)
    if SHUFFLE_ITEMS:
        random.Random(seed_text).shuffle(order)
    return "\n".join(render_constraint(c) for c in order)


def build_input_block(sample, description):
    return "\n".join([
        "INPUT",
        f"Scene: {sample['scene_category']} ({sample['scene_mode']})",
        f"Context (before the video): {sample['context']}",
        f"Event (before or at the start of the video): "
        f"{sample['event'] or 'No distinct event annotated.'}",
        f"Initial state (at or before the start of the video; reference only, "
        f"not evidence): {sample['initial_state'] or 'Not annotated.'}",
        *sample.get("entity_lines", []),
        "",
        "Video description (the only source of observations):",
        '"""', description, '"""',
    ])

def static_description(desc):
    """静态对照：只保留 [T=0%] 段，并声明之后没有任何变化。"""
    parts = [p.strip() for p in re.split(r"(?=\[T=\d+%\])", desc.strip()) if p.strip()]
    first = next((p for p in parts if p.startswith("[T=0%]")), parts[0] if parts else "[T=0%]")
    return (first + " [T=100%] The video ends with every person and object in "
            "exactly the same position, posture, expression, and state as at the "
            "start; no movement or change of any kind occurs.")


# #####################################################################
# 4. LLM 调用与缓存
# #####################################################################

def call_llm_json(client, user_prompt, temperature):
    kwargs = dict(
        model=t2.LLM_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM.strip()},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature, max_tokens=t2.LLM_MAX_TOKENS)
    if t2.USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    out = t2.extract_json(resp.choices[0].message.content)
    if not isinstance(out, dict):
        raise ValueError("judge json parse failed")
    return out


def cached_judge(clients, caches, stage, sample, prompt, run):
    # prompt 含视频描述，错配组与正常组不会互相命中缓存
    temp = t2.judge_temperature()
    key = (f"{stage}|{sample['video_key']}|{t2.LLM_MODEL}|"
           f"{t2.short_hash(JUDGE_SYSTEM, prompt)}|t{temp}|run{run}")
    out = caches["judge"].get(key)
    if out is None:
        out = t2.with_retry(call_llm_json, clients["llm"], prompt, temp)
        caches["judge"][key] = out
        t2.save_json(caches["judge_path"], caches["judge"])
    return out


# #####################################################################
# 5. 单次评分结果校验（代码侧兜底）
# #####################################################################

def clean_id(x):
    return re.sub(r"[\[\]\s]", "", str(x or "")).upper()


def score_run(out, items, anchors, warnings):
    """
    返回 verdicts {cid: {"verdict", "value", "valid_evidence", "entity_found"}} 与 exists（0/1）。
    value：must/directional 为得分（not_observable 为 None），must_not 为违反度。
    """
    subject_ok = out.get("subject_visible") is not False
    if not subject_ok:
        warnings.append("subject not visible: constraints set to fallback, exists=0")

    entries = {}
    for ent in out.get("constraints") or []:
        if isinstance(ent, dict):
            entries.setdefault(clean_id(ent.get("id")), ent)

    verdicts = {}
    for c in items:
        cid, ctype = c["id"], c["ctype"]
        table, fallback = VERDICT_TABLE[ctype], FALLBACK[ctype]
        ent = entries.get(cid)
        n_valid, found = 0, None
        if ent is None:
            warnings.append(f"{cid}: missing in judge output, set to {fallback}")
            v = fallback
        else:
            found = ent.get("entity_found")
            v = t2.norm_key(ent.get("verdict"))
            if v not in table:
                warnings.append(f"{cid}: invalid verdict {ent.get('verdict')!r}, "
                                f"set to {fallback}")
                v = fallback
            n_valid, _ = t2.valid_evidence_anchors(ent.get("evidence"), anchors)
            if not subject_ok:
                v = fallback
            elif found is False and v != fallback:
                warnings.append(f"{cid}: capped {v}->{fallback} (entity not found)")
                v = fallback
            elif v not in NO_EVIDENCE_OK and n_valid == 0:
                if ctype == "must_not":   # 无证据的违反判定降一档，而不是直接清零
                    nv = "partially_violated" if v == "violated" else "not_violated"
                else:
                    nv = fallback
                warnings.append(f"{cid}: capped {v}->{nv} (no valid anchored evidence)")
                v = nv
        verdicts[cid] = {"verdict": v, "value": table[v],
                         "valid_evidence": n_valid, "entity_found": found}

    changes = out.get("observed_changes") or []
    valid_changes = [
        ch for ch in changes
        if isinstance(ch, dict)
        and t2.norm_key(ch.get("dimension")) in WORLD_DIMENSIONS
        and t2.valid_evidence_anchors([ch.get("change")], anchors)[0] > 0]
    claimed = out.get("world_state_changed") is True
    if subject_ok and claimed and not valid_changes:
        warnings.append("exists: capped 1->0 (no valid anchored change)")
    exists = 1 if (subject_ok and claimed and valid_changes) else 0
    return verdicts, exists



# #####################################################################
# 6. S_causal 计算（与分支无关，反事实可复用）
# #####################################################################

def causal_components(values, items, exists):
    """values：{cid: 数值}（must/directional 为得分，must_not 为违反度；None 为不可观测）。"""
    must = [values[c["id"]] for c in items if c["ctype"] == "must"]
    mnot = [values[c["id"]] for c in items if c["ctype"] == "must_not"]
    dirs = [values[c["id"]] for c in items
            if c["ctype"] == "directional" and values[c["id"]] is not None]

    s_must = math.prod(must) if must else None
    s_must_not = math.prod(1 - v for v in mnot) if mnot else 1.0
    s_dir = sum(dirs) / len(dirs) if dirs else None
    k_must, k_dir = len(must), len(dirs)

    s_causal = s_must_not * (exists + (s_must or 0.0) * k_must
                             + (s_dir or 0.0) * k_dir) / (1 + k_must + k_dir)
    return {"S_must": s_must, "S_must_not": s_must_not, "S_direction": s_dir,
            "S_exists": exists, "S_causal": s_causal,
            "K_must": k_must, "K_must_not": len(mnot), "K_dir": k_dir,
            "K_dir_unobservable": sum(1 for c in items if c["ctype"] == "directional"
                                      and values[c["id"]] is None)}



def merge_values(vals):
    """合并一条约束在多次评分中的数值：多数评分判为不可观测时返回 None，否则对可观测的取 agg。"""
    obs = [x for x in vals if x is not None]
    if not obs or (len(vals) > 1 and len(obs) * 2 <= len(vals)):
        return None
    return t2.agg(obs)


def evaluate_causal(sample, items, description, clients, caches, warnings,
                    stage="task3"):
    """对一个分支的约束做多次评分并合并：先对每条约束合并，再代入公式。"""
    anchors = set(t2.ANCHOR_RE.findall(description))
    if not anchors:
        warnings.append("description has no [T=XX%] anchors; "
                        "anchor check degraded to non-empty check")
    input_block = build_input_block(sample, description)

    runs = []
    for run in range(t2.JUDGE_RUNS):
        prompt = t2.fill(CAUSAL_PROMPT, input_block=input_block,
                         grounding_rules=GROUNDING_RULES.strip(),
                         constraint_block=constraint_block(
                             items, f"{sample['video_key']}|{stage}|{run}"))
        out = cached_judge(clients, caches, stage, sample, prompt, run)
        verdicts, exists = score_run(out, items, anchors, warnings)
        per_run = causal_components({k: v["value"] for k, v in verdicts.items()},
                                    items, exists)
        runs.append({"verdicts": verdicts, "exists": exists,
                     "S_causal": per_run["S_causal"], "raw": out})

    merged = {c["id"]: merge_values([r["verdicts"][c["id"]]["value"] for r in runs])
              for c in items}
    merged_exists = t2.agg([r["exists"] for r in runs])
    comps = causal_components(merged, items, merged_exists)

    item_results = []
    for c in items:
        v = merged[c["id"]]
        vs = [r["verdicts"][c["id"]]["verdict"] for r in runs]
        item_results.append({
            "id": c["id"], "ctype": c["ctype"], "kind": c["kind"],
            "statement": render_constraint(c)[2:],
            "mean_value": v,
            # 统一为"越高越好"；不可观测为 None
            "item_score": None if v is None else (1 - v if c["ctype"] == "must_not" else v),
            # must/directional：判为最低档的比例；must_not：判为违反（含部分违反）的比例
            "fallback_rate": sum(x == FALLBACK[c["ctype"]] for x in vs) / len(vs),
            "violation_rate": (sum(x != "not_violated" for x in vs) / len(vs)
                               if c["ctype"] == "must_not" else None),
            "not_observable_rate": sum(x == "not_observable" for x in vs) / len(vs),
            "verdicts": vs,
        })

    # 实体未找到比例：错配时应明显升高（诊断用，不影响分数）
    nf = [r["verdicts"][c["id"]]["entity_found"] is False
          for r in runs for c in items]
    comps["entity_not_found_rate"] = sum(nf) / len(nf) if nf else 0.0
    comps["judge_instability"] = t2.spread([r["S_causal"] for r in runs])
    return comps, item_results, runs



# #####################################################################
# 7. 单样本评测
# #####################################################################

def build_qc_flags(record):
    """仅正常组有意义。参考视频：用于标注质检；生成视频：标记低分样本与未满足的约束。"""
    if CONTROL_MODE != "none":
        return []
    flags = []
    if record["S_causal"] < QC_LOW_SCORE:
        flags.append(f"low_S_causal={record['S_causal']:.2f}")
    for it in record["items"]:
        if it["item_score"] is None:
            flags.append(f"{it['id']}({it['ctype']})=not_observable")
        elif it["item_score"] < QC_LOW_SCORE:
            flags.append(f"{it['id']}({it['ctype']})={'/'.join(it['verdicts'])}")
    if record["S_exists"] < 1:
        flags.append(f"S_exists={record['S_exists']:.2f}")
    return flags



def evaluate_sample(raw, clients, caches):
    sample, warnings = t2.parse_sample(raw)   # 复用：视频文件、场景、事件、实体、目标情绪
    sample["initial_state"] = (raw.get("state0") or {}).get("text", "")

    items = parse_constraints(
        (raw.get("transition") or {}).get("world_state_delta_items"), warnings)
    if not items:
        warnings.append("no constraints; S_causal reduces to S_exists")

    description = t2.get_description(sample, clients, caches)
    if CONTROL_MODE == "static":
        description = static_description(description)
    comps, item_results, runs = evaluate_causal(sample, items, description,
                                                clients, caches, warnings)

    record = {
        "video_key": sample["video_key"],
        "video_file": sample["video_file"],
        "control_mode": CONTROL_MODE,
        "scene_category": sample["scene_category"],
        "scene_mode": sample["scene_mode"],
        "target_emotion": sample["target_emotion"],
        **{k: comps[k] for k in ("S_must", "S_must_not", "S_direction",
                                 "S_exists", "S_causal", "K_must",
                                 "K_must_not", "K_dir", "K_dir_unobservable")},
        "Score_T3": comps["S_causal"],
        "judge_instability": comps["judge_instability"],
        "entity_not_found_rate": comps["entity_not_found_rate"],
        "items": item_results,
        "description": description,
        "judge": {"runs": runs},
        "warnings": list(dict.fromkeys(warnings)),
    }
    record["qc_flags"] = build_qc_flags(record)
    return record




# #####################################################################
# 8. 汇总
# #####################################################################

def cap_reason(w):
    m = re.search(r"\((.*?)\)", w)
    return m.group(1) if m else w


def summarize(results):
    ok = [r for r in results if "error" not in r]
    summary = {"n": len(ok), "n_failed": len(results) - len(ok),
               "vlm_model": t2.VLM_MODEL, "llm_model": t2.LLM_MODEL,
               "judge_runs": t2.JUDGE_RUNS, "aggregate": t2.AGGREGATE,
               "control_mode": CONTROL_MODE}
    if not ok:
        return summary

    summary["Score_T3"] = t2.mean([r["S_causal"] for r in ok])
    # 各分量只在含对应约束的样本上平均
    summary["S_must"] = t2.mean([r["S_must"] for r in ok if r["K_must"]])
    summary["S_must_not"] = t2.mean([r["S_must_not"] for r in ok if r["K_must_not"]])
    summary["S_direction"] = t2.mean([r["S_direction"] for r in ok if r["K_dir"]])
    summary["S_exists"] = t2.mean([r["S_exists"] for r in ok])
    summary["judge_instability"] = t2.mean([r["judge_instability"] for r in ok])
    summary["entity_not_found_rate"] = t2.mean([r["entity_not_found_rate"] for r in ok])

    # 分组报告：用于发现系统性偏差
    def group(key_fn):
        g = defaultdict(list)
        for r in ok:
            g[key_fn(r)].append(r["S_causal"])
        return {k: {"n": len(v), "S_causal": t2.mean(v)} for k, v in sorted(g.items())}

    summary["by_scene"] = group(lambda r: r["scene_category"])
    summary["by_emotion_family"] = group(
        lambda r: t2.EMOTION_GROUP.get(r["target_emotion"], "UNKNOWN"))

    item_groups = defaultdict(lambda: {"n": 0, "score": [], "fallback": [],
                                       "violation": [], "unobs": []})
    for r in ok:
        for it in r["items"]:
            g = item_groups[f"{it['ctype']} / {it['kind']}"]
            g["n"] += 1
            g["score"].append(it["item_score"])            # t2.mean 会跳过 None
            g["fallback"].append(it["fallback_rate"])
            g["violation"].append(it["violation_rate"])
            g["unobs"].append(it["not_observable_rate"])
    summary["by_constraint_kind"] = {
        k: {"n": v["n"], "item_score": t2.mean(v["score"]),
            "fallback_rate": t2.mean(v["fallback"]),
            "violation_rate": t2.mean(v["violation"]),
            "not_observable_rate": t2.mean(v["unobs"])}
        for k, v in sorted(item_groups.items())}

    caps = Counter(cap_reason(w) for r in ok for w in r["warnings"] if "capped" in w)
    summary["cap_stats"] = dict(caps.most_common())
    ann_keys = ("verify polarity", "no constraints", "no target_agent",
                "unknown constraint category")
    summary["annotation_warnings"] = {
        r["video_key"]: [w for w in r["warnings"] if any(k in w for k in ann_keys)]
        for r in ok if any(any(k in w for k in ann_keys) for w in r["warnings"])}
    summary["flagged"] = {r["video_key"]: r["qc_flags"] for r in ok if r["qc_flags"]}
    return summary



def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


def print_summary(s):
    print("\n" + "=" * 60)
    print(f"Task 3 | VLM={t2.VLM_MODEL} | LLM={t2.LLM_MODEL} | "
          f"runs={t2.JUDGE_RUNS} ({t2.AGGREGATE})")
    print(f"Mode: control={CONTROL_MODE}")
    print(f"Evaluated: {s['n']} | Failed: {s['n_failed']}")
    if not s["n"]:
        return
    print(f"Score_T3 (S_causal): {fmt(s['Score_T3'])}")
    for k in ("S_must", "S_must_not", "S_direction", "S_exists"):
        print(f"  {k:<12}: {fmt(s[k])}")
    print(f"  judge_instability    : {fmt(s['judge_instability'])}")
    print(f"  entity_not_found_rate: {fmt(s['entity_not_found_rate'])}")
    for title, key in (("By scene category", "by_scene"),
                       ("By target emotion family", "by_emotion_family")):
        print(f"\n{title}:")
        for g, v in s[key].items():
            print(f"  {g}: n={v['n']} S_causal={fmt(v['S_causal'])}")
    print("\nBy constraint category / kind:")
    for g, v in s["by_constraint_kind"].items():
        extra = (f"violation_rate={fmt(v['violation_rate'])}"
                 if g.startswith("must_not") else
                 f"fallback_rate={fmt(v['fallback_rate'])}")
        print(f"  {g}: n={v['n']} item_score={fmt(v['item_score'])} {extra} "
              f"not_observable={fmt(v['not_observable_rate'])}")
    if s.get("cap_stats"):
        print("\nCode-side verdict caps:")
        for k, v in s["cap_stats"].items():
            print(f"  {k}: {v}")
    if s.get("annotation_warnings"):
        print("\nAnnotation warnings:")
        for k, v in s["annotation_warnings"].items():
            print(f"  {k}: {'; '.join(v)}")
    if s.get("flagged"):
        print("\nFlagged samples (reference videos: annotation QC; "
              "generated videos: low-score cases):")
        for k, v in s["flagged"].items():
            print(f"  {k}: {'; '.join(v)}")

def _short(s, n=240):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def review_record(r):
    if "error" in r:
        return {"video_key": r.get("video_key"), "error": r["error"]}
    runs = r["judge"]["runs"]
    items = []
    for it in r["items"]:
        vs = it["verdicts"]
        common = Counter(vs).most_common(1)[0][0]
        ent = {}
        for run in runs:   # 取判定为多数结果的那次评分的理由
            if run["verdicts"][it["id"]]["verdict"] == common:
                ent = next((e for e in run["raw"].get("constraints") or []
                            if isinstance(e, dict) and clean_id(e.get("id")) == it["id"]), {})
                break
        items.append({
            "statement": it["statement"],
            "verdicts": vs,
            "item_score": it["item_score"],
            "analysis": _short(ent.get("analysis")),
            "evidence": [_short(x, 160) for x in (ent.get("evidence") or [])][:3],
        })
    return {
        "video_key": r["video_key"],
        "S_causal": r["S_causal"],
        **{k: r[k] for k in ("S_must", "S_must_not", "S_direction", "S_exists",
                             "K_dir_unobservable")},
        "subject_not_visible_runs": sum(run["raw"].get("subject_visible") is False
                                        for run in runs),
        "items": items,
        "caps": [w for w in r["warnings"] if "capped" in w],
        "description": r["description"],
    }


def write_review(results, summary, path):
    keep = ("n", "n_failed", "Score_T3", "S_must", "S_must_not", "S_direction",
            "S_exists", "judge_instability", "entity_not_found_rate",
            "by_constraint_kind", "cap_stats")
    t2.save_json(path, {
        "control_mode": CONTROL_MODE,
        "judge_runs": t2.JUDGE_RUNS,
        "summary": {k: summary[k] for k in keep if k in summary},
        "samples": [review_record(r) for r in results],
    })


# #####################################################################
# 9. 对照实验
# #####################################################################

GROUP_NAMES = {"none": "A_normal", "mismatch": "B_mismatch"}


def with_tag(path):
    """错配组写入独立文件；正常组使用原文件名。"""
    if CONTROL_MODE == "none":
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}.{CONTROL_MODE}{p.suffix}"))


def assign_mismatch(samples):
    """为每个样本挑一个视频：其目标情绪大类不在本样本候选情绪的大类中。"""
    info = [t2.parse_sample(raw)[0] for raw in samples]
    n = len(samples)
    for i, (raw, s) in enumerate(zip(samples, info)):
        cand_fams = {t2.EMOTION_GROUP[e] for e in s["emotion_candidates"]}
        pick = next((j for j in range(n) if j != i and info[j]["target_emotion"]
                     and t2.EMOTION_GROUP[info[j]["target_emotion"]] not in cand_fams),
                    (i + 1) % n)
        raw["_video_file"] = samples[pick]["video_key"]
        print(f"  [mismatch] {raw['video_key']}  <-  {raw['_video_file']}")


def print_comparison(summaries):
    print("\n" + "#" * 60)
    print("各组对比")
    cols = ("Score_T3", "S_must", "S_must_not", "S_direction", "entity_not_found_rate")
    heads = ("S_causal", "must", "must_not", "dir", "entity_nf")
    print(f"{'group':<14}" + "".join(f"{h:>11}" for h in heads))
    for name, s in summaries.items():
        if not s.get("n"):
            print(f"{name:<14}  (no result)")
            continue
        print(f"{name:<14}" + "".join(
            f"{s[c]:>11.3f}" if isinstance(s.get(c), (int, float)) else f"{'-':>11}"
            for c in cols))

    a, b = summaries.get("A_normal"), summaries.get("B_mismatch")
    if a and b and a.get("n") and b.get("n"):
        gap = a["Score_T3"] - b["Score_T3"]
        gap_must = (a["S_must"] or 0) - (b["S_must"] or 0)
        print(f"\nA - B 的 S_causal 差值：{gap:.3f}  "
              f"{'✓ 有区分度' if gap >= 0.3 else '✗ 区分度不足，约束判定可能偏宽松'}")
        print(f"A - B 的 S_must   差值：{gap_must:.3f}  "
              f"（must 连乘，对错配最敏感）")
        print("注：错配时 S_exists 通常仍为 1、must_not 通常未违反，"
              "因此 B 组 S_causal 不会降到 0。")



# #####################################################################
# 10. 主流程
# #####################################################################

def run_group(clients, caches):
    out_path, summary_path, review_path = (with_tag(OUT_PATH), with_tag(SUMMARY_PATH),
                                           with_tag(REVIEW_PATH))
    samples = t2.load_json(t2.DATA_PATH, [])
    if t2.LIMIT:
        samples = samples[:t2.LIMIT]

    # 断点续跑：保留成功记录，失败记录重跑
    results = [r for r in t2.load_json(out_path, []) if "error" not in r]
    done = {r["video_key"] for r in results}
    pending = [s for s in samples if s["video_key"] not in done]
    print(f"Control: {CONTROL_MODE} | Total: {len(samples)} | Done: {len(done)} | "
          f"Pending: {len(pending)}")
    print(f"Output: {out_path}")

    for raw in tqdm(pending, desc=f"Task 3 ({CONTROL_MODE})"):
        try:
            record = evaluate_sample(raw, clients, caches)
            tqdm.write(f" ✓ {raw['video_key']}  "
                       f"must={fmt(record['S_must'])} "
                       f"must_not={fmt(record['S_must_not'])} "
                       f"dir={fmt(record['S_direction'])} "
                       f"exists={fmt(record['S_exists'])}  "
                       f"S_causal={record['S_causal']:.3f}")
        except Exception as e:
            logger.exception(f"[FAILED] {raw.get('video_key')}")
            tqdm.write(f" ✗ {raw.get('video_key')}: {e}")
            record = {"video_key": raw.get("video_key"), "error": str(e)}
        results.append(record)
        t2.save_json(out_path, results)
        time.sleep(t2.SLEEP_SEC)

    summary = summarize(results)
    t2.save_json(summary_path, summary)
    write_review(results, summary, review_path)
    print_summary(summary)
    print(f"\nReview file (paste this one): {review_path}")
    return summary


def main():
    global CONTROL_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["none", "static"], default="none",
                    help="none：正常评测；static：静态对照（描述只保留开场）")
    ap.add_argument("--fresh", action="store_true",
                    help="删除本组旧结果后重评（缓存保留）")
    ap.add_argument("--review-only", action="store_true",
                    help="不调用 API，只根据已有结果重新生成评审文件")
    args = ap.parse_args()
    CONTROL_MODE = args.control
    out_path, summary_path, review_path = (with_tag(OUT_PATH), with_tag(SUMMARY_PATH),
                                           with_tag(REVIEW_PATH))

    if args.review_only:
        results = t2.load_json(out_path, [])
        if not results:
            raise SystemExit(f"找不到结果文件：{out_path}")
        write_review(results, summarize(results), review_path)
        print(f"Review file: {review_path}")
        return

    t2.check_config()          # 复用 Task 2 的 Key / prompt / 路径检查
    if args.fresh:
        for p in (out_path, summary_path, review_path):
            if Path(p).exists():
                Path(p).unlink()
                print(f"removed {p}")

    clients = {"vlm": OpenAI(api_key=t2.VLM_API_KEY, base_url=t2.VLM_BASE_URL),
               "llm": OpenAI(api_key=t2.LLM_API_KEY, base_url=t2.LLM_BASE_URL)}
    desc_path = Path(t2.CACHE_DIR) / "descriptions.json"   # 与 Task 2 共用描述缓存
    caches = {"desc_path": desc_path, "desc": t2.load_json(desc_path, {}),
              "judge_path": JUDGE_CACHE_PATH,
              "judge": t2.load_json(JUDGE_CACHE_PATH, {})}
    run_group(clients, caches)



if __name__ == "__main__":
    main()
