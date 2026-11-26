"""
EWH-Bench Counterfactual Chain Evaluation, state0-emotion protocol.
The factual emotion is state0.emotion; the counterfactual intervention is
counterfactual.emotion_cf. Both are judged at the same broad-family level.
Each video's action/causal fit is cross-checked against both branch specs.
The description-difference judge is used for the shared-setting gate.

依赖：与 eval_task2.py、eval_task3.py 放在同一目录。Key、模型、VLM prompt、路径等配置复用
     eval_task2.py；OSS Key 使用 eval_task2.py 中的 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET。
原分支的因果约束评分复用 Task 3 缓存；动作评分改为两分支对称的粗类情绪条件。
运行：python eval_cf.py
"""
import re, time, logging, argparse
from collections import defaultdict, Counter
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import eval_task2 as t2      # 若文件改名，这里同步修改
import eval_task3 as t3


# #####################################################################
# 1. 反事实专属配置（其余配置见 eval_task2.py）
# #####################################################################

OUT_PATH = "cf_results.json"
SUMMARY_PATH = "cf_results.summary.json"
JUDGE_CACHE_PATH = Path(t2.CACHE_DIR) / "cf_judge.json"
LOG_FILE = "cf_eval.log"
REVIEW_PATH = "cf_review.json"  # 精简评审文件，贴这个即可

# 反事实分支视频：OSS 上的文件夹；文件名默认与 video_key 相同（可由 counterfactual.video_key 覆盖）
CF_OSS_PREFIX = "videos_cf"
# 原分支视频来源："task2" 表示复用 Task 2 评测的视频与视频描述/因果约束缓存；
# 评测生成模型时填 OSS 前缀（以 e_orig 为条件生成的视频），如 "videos_orig_cond"
CF_ORIG_SOURCE = "task2"
CF_SIGN_EXPIRE_SEC = 3600

# 差异评分 L 仅作诊断，同初始场景判断用于门控；S_counter 由交叉矩阵计算
RUN_DIFF = True
DIFF_RUNS_PER_ORDER = None      # 每种顺序的重复次数；None 表示使用 JUDGE_RUNS

# 两段生成视频必须保留同一初始场景，才是同一 state0 的反事实比较。
SAME_SETTING_GATE = True
SAME_SETTING_MIN = 0.5          # same_setting_rate 低于该值时 S_counter 置 0

PROTOCOL_VERSION = "state0_emotion_family_v1"
EMOTION_FAMILY_CUES = {
    "SAD": "withdrawn, low-energy, resigned, disappointed, or self-blaming tone",
    "FEAR": "vigilant, cautious, startled, avoidant, or insecure tone",
    "ANGER": "tense, sharp, oppositional, hurt, or confrontational tone",
    "SURPRISE": "abrupt reorientation, startle, wonder, or confusion",
    "HAPPY": "warm, engaged, confident, affectionate, or positive tone",
    "DISGUST": "recoil, rejection, aversion, judgment, or disapproval",
    "NEUTRAL": "ordinary behavior without a directional emotional tone",
}

QC_LOW_R = 0.5                  # 对角分支合理性低于该值时标记
QC_LOW_SEL = 0.3                # 情绪选择性低于该值时标记
QC_ORDER_GAP = 2                # 两种顺序的等级相差达到该值时标记

DIFF_DIMENSIONS = {"subject_action", "subject_expression", "spatial_relation",
                   "physical_object", "others_response", "social_atmosphere"}
DIFF_TYPES = {"different_direction", "present_vs_absent", "intensity_only"}


logger = logging.getLogger("cf")
logger.setLevel(logging.INFO)
logger.propagate = False
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_fh)


# #####################################################################
# 2. 差异评分 Prompt
# #####################################################################

DIFF_SYSTEM = """
You are a strict, evidence-bound evaluator for an emotion-aware world-model
benchmark. You compare two written descriptions of videos that continue the
same initial scene. You never invent observations, never guess what the
videos were intended to show, and you apply the rubric literally. Output one
valid JSON object and nothing else.
"""

# 必须包含变量：{input_block}
DIFF_PROMPT = """
TASK: Two videos, A and B, continue the same initial scene after the same
event. Compare how the MAIN SUBJECT behaves and how the world state evolves in
A versus B, and rate the size of the difference. You are not told what either
video was meant to show; do not guess, and do not judge which one is better.

{input_block}

RULES
1. The two descriptions are the ONLY source of observations. Each description
   has its own [T=XX%] anchors. An item for A must cite an anchor from
   description A; an item for B must cite an anchor from description B.
2. A behavior or change that is not mentioned did not happen.
3. Count only differences in: the main subject's actions (e.g., approaching
   vs. avoiding, engaging vs. withdrawing, expressing vs. holding back,
   continuing vs. stopping a task), the main subject's facial or bodily
   expression, spatial relations between entities, object states, other
   people's responses, and group atmosphere.
4. Do NOT count: differences in wording, camera movement, framing or cuts,
   appearance or rendering details, background elements unrelated to the
   subject, differences already present at [T=0%], or small shifts in timing.
5. Match on core meaning. The same behavior described in different words is
   not a difference.
6. The description was written under a rule that forbids asserting speech
   when audio is unclear. An open or moving mouth directed at another person
   is a valid cue of vocal or verbal activity.

PROCEDURE
Step 1. For each video, summarize the main subject's behavior trajectory and
        the final world state, citing anchors.
Step 2. List every difference. Each item has one dimension and one type:
        dimensions: subject_action, subject_expression, spatial_relation,
                    physical_object, others_response, social_atmosphere
        types: different_direction (both videos change this dimension, but in
               different or opposite ways), present_vs_absent (a change or
               behavior appears in one video only), intensity_only (same
               behavior and direction, differing only in strength, duration,
               or detail)
        For present_vs_absent, write "absent" on the side where it does not
        occur.
Step 3. State the dominant behavior function of the main subject in each video
        (e.g., withdrawing from the other person, confronting the other person,
        celebrating with the other person) and whether the two are the same.

RUBRIC (level)
0 = No difference in the main subject's behavior or in the world-state
    outcome beyond wording, framing, or negligible detail.
1 = Minor. Same dominant behavior function and same direction of world-state
    change; differences only in intensity, timing, or secondary details.
2 = Clear. The dominant behavior function differs, OR at least one
    world-state dimension changes in a different direction or is present in
    one video and absent in the other, while the two trajectories still share
    their main structure.
3 = Significant. The dominant behavior functions differ or are opposite AND
    the resulting world states differ in at least two dimensions.

LEVEL CONSTRAINTS
- Level 1 or higher requires at least one difference item with a valid anchor.
- Level 2 or higher requires at least one different_direction or
  present_vs_absent item.
- Level 3 requires "same": false for the dominant behavior and structural
  differences in at least two dimensions.

Also report "same_setting": compare ONLY the [T=0%] parts of the two
descriptions. It is true only if both start with the same-looking main
subject (clothing, hair), the same place, and the same key objects. A
different room, different clothing, or different key objects make it false.

OUTPUT (exactly this JSON structure)
{
  "same_setting": true,
  "summary_a": "[T=XX%] ...",
  "summary_b": "[T=XX%] ...",
  "differences": [
    {"dimension": "<dimension>", "type": "<type>",
     "a": "[T=XX%] ... or absent", "b": "[T=XX%] ... or absent"}
  ],
  "dominant": {"a": "<dominant behavior function in A>",
               "b": "<dominant behavior function in B>",
               "same": true},
  "level": 0
}
"""


PROMPT_SPECS = {"DIFF_PROMPT": ["input_block"]}


# #####################################################################
# 3. 视频读取
# #####################################################################

def make_bucket():
    import oss2
    auth = oss2.Auth(t2.OSS_ACCESS_KEY_ID, t2.OSS_ACCESS_KEY_SECRET)
    return oss2.Bucket(auth, t2.OSS_ENDPOINT, t2.OSS_BUCKET)


def branch_oss_key(raw, prefix, section=None):
    """分支视频在 OSS 上的路径；section（如 "counterfactual"）中的 video_key 可覆盖默认文件名。"""
    name = ((raw.get(section) or {}).get("video_key") if section else None) \
        or raw["video_key"]
    return f"{prefix.strip('/')}/{Path(name).name}"


def get_oss_description(oss_key, clients, caches):
    """从 OSS 读取分支视频，签名 URL 交给 VLM；描述共用 Task 2 缓存。"""
    bucket = clients["bucket"]
    try:
        meta = bucket.head_object(oss_key)
    except Exception as e:
        raise FileNotFoundError(f"video not found on OSS: {oss_key} ({e})")
    dkey = (f"oss:{oss_key}|{meta.etag}|{t2.VLM_MODEL}|"
            f"{t2.short_hash(t2.DESCRIBE_PROMPT)}")
    description = caches["desc"].get(dkey)
    if description is None:
        url = bucket.sign_url("GET", oss_key, CF_SIGN_EXPIRE_SEC, slash_safe=True)
        description = t2.with_retry(t2.describe_video, clients["vlm"], url)
        caches["desc"][dkey] = description
        t2.save_json(caches["desc_path"], caches["desc"])
    return description



# #####################################################################
# 4. 分支解析与分支合理性 R
# #####################################################################

def normalize_list(labels, fn, kind, warnings, branch):
    out = []
    for lab in labels or []:
        k = fn(lab)
        if k is None:
            warnings.append(f"{branch}: dropped unknown {kind} {lab!r}")
        elif k not in out:
            out.append(k)
    return out


def pick(d, *paths):
    """按顺序查找第一个非空字段；path 可以是 "a" 或 "a.b"。"""
    for p in paths:
        cur = d
        for k in p.split("."):
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur not in (None, "", [], {}):
            return cur
    return None


def parse_branches(raw, warnings):
    tr = raw.get("transition") or {}
    state0 = raw.get("state0") or {}
    cf = raw.get("counterfactual") or {}
    specs = {
        "orig": (state0.get("emotion"), tr.get("plausible_actions"),
                 tr.get("world_state_delta_items")),
        "cf": (
            pick(cf, "cf_emotion", "emotion_cf", "e_cf", "emotion",
                 "target_emotion", "state1_cf.emotion", "state1.emotion",
                 "reference_state1.emotion"),
            pick(cf, "plausible_actions", "designed_actions", "actions",
                 "diff.plausible_actions", "transition.plausible_actions"),
            pick(cf, "world_state_delta_items", "world_state_constraints",
                 "constraints", "diff.world_state_delta_items",
                 "transition.world_state_delta_items"),
        ),
    }
    branches = {}
    for b, (emo, acts, cons) in specs.items():
        where = (f"; available counterfactual keys: {sorted(cf.keys())}"
                 if b == "cf" else "")
        e = t2.norm_key(emo) if b == "orig" else t2.normalize_emotion(emo)
        family = state0_emotion_family(e) if b == "orig" else t2.EMOTION_GROUP.get(e)
        if family is None:
            raise ValueError(f"{b}: unknown or missing emotion {emo!r}{where}")
        actions = normalize_list(acts, t2.normalize_action, "action", warnings, b)
        if not actions:
            raise ValueError(f"{b}: no valid action candidates{where}")
        w = []
        try:
            items = t3.parse_constraints(cons, w)
        except ValueError as ex:
            raise ValueError(f"{b}: {ex}{where}")
        if not items:
            raise ValueError(f"{b}: no constraints; branch plausibility would "
                             f"reduce to S_exists{where}")
        warnings.extend(f"{b}: {x}" for x in w)
        branches[b] = {"emotion": e, "emotion_family": family,
                       "actions": actions, "items": items}
    if branches["orig"]["emotion_family"] == branches["cf"]["emotion_family"]:
        raise ValueError("state0.emotion and emotion_cf have the same emotion family; "
                         "the intervention cannot be distinguished from this annotation")
    return branches



def state0_emotion_family(emotion):
    broad = {"sadness": "SAD", "fear": "FEAR", "anger": "ANGER",
             "surprise": "SURPRISE", "happiness": "HAPPY",
             "disgust": "DISGUST", "neutral": "NEUTRAL"}
    return broad.get(emotion) or t2.EMOTION_GROUP.get(t2.normalize_emotion(emotion))


def run_family_action_judging(sample, family, input_block, anchors, clients,
                              caches, warnings):
    """Score both branches using the same family-level emotion rubric."""
    candidates = sample["action_candidates"]
    runs = []
    for run in range(t2.JUDGE_RUNS):
        order = t2.presented_order(candidates,
                                   f"{sample['video_key']}|cf|{family}|act|{run}")
        prompt = t2.fill(
            t2.ACTION_SCORE_PROMPT, input_block=input_block,
            grounding_rules=t2.GROUNDING_RULES.strip(),
            candidate_block=t2.candidate_block(order, t2.ACTION_DISPLAY,
                                               t2.ACTION_DEFS),
            emotion=family.lower(),
            emotion_core=t2.EMOTION_FAMILY_DEFS[family],
            emotion_cues=EMOTION_FAMILY_CUES[family])
        out = t2.cached_judge(clients, caches, f"cf_action_family|{family}",
                              sample, prompt, run)
        scores, details = t2.score_action_output(out, candidates, anchors,
                                                  warnings)
        runs.append({"order": order, "scores": scores,
                     "details": details, "raw": out})
    return runs


def evaluate_branch(video, spec, sample, branch, description, clients, caches,
                    warnings):
    """
    R(V_video, e_spec) = S_action(V_video, e_spec) · S_causal(V_video; spec 约束)。
    video / spec ∈ {"orig", "cf"}：用 spec 分支的情绪、动作候选与约束评 video 分支的视频。
    """
    tag = f"{video}_on_{spec}"
    bsample = dict(sample, action_candidates=branch["actions"])
    anchors = set(t2.ANCHOR_RE.findall(description))
    w = []
    input_block = t2.build_input_block(bsample, description)

    # 因果约束评分可以复用 Task 3 缓存；动作评分使用独立的情绪家族 prompt/cache。
    if video == spec == "orig" and CF_ORIG_SOURCE == "task2":
        act_caches, causal_caches, stage = caches["t2"], caches["t3"], "task3"
    else:
        act_caches, causal_caches, stage = caches, caches, f"cf_{tag}"

    act_runs = run_family_action_judging(
        bsample, branch["emotion_family"], input_block,
        anchors, clients, act_caches, w)
    act_scores = {a: t2.agg([r["scores"][a] for r in act_runs])
                  for a in branch["actions"]}
    a_star = max(branch["actions"], key=lambda a: act_scores[a])
    s_action = act_scores[a_star]

    comps, item_results, causal_runs = t3.evaluate_causal(
        bsample, branch["items"], description, clients, causal_caches, w,
        stage=stage)
    warnings.extend(f"{tag}: {x}" for x in dict.fromkeys(w))

    return {
        "video": video,
        "spec": spec,
        "emotion": branch["emotion"],
        "emotion_family": branch["emotion_family"],
        "action_candidates": branch["actions"],
        "action_scores": act_scores,
        "a_star": a_star,
        "S_action": s_action,
        **{k: comps[k] for k in ("S_must", "S_must_not", "S_direction",
                                 "S_exists", "S_causal", "K_must",
                                 "K_must_not", "K_dir", "K_dir_unobservable")},
        "R": s_action * comps["S_causal"],
        "items": item_results,
        "judge": {"action_runs": act_runs, "causal_runs": causal_runs},
    }



# #####################################################################
# 5. 差异评分 L
# #####################################################################

def build_diff_input(sample, desc_a, desc_b):
    return "\n".join([
        "INPUT",
        f"Scene: {sample['scene_category']} ({sample['scene_mode']})",
        f"Context (before the videos): {sample['context']}",
        f"Event (before or at the start of the videos): "
        f"{sample['event'] or 'No distinct event annotated.'}",
        f"Initial state (shared by both videos; reference only): "
        f"{sample['initial_state'] or 'Not annotated.'}",
        "",
        "Description of video A:", '"""', desc_a, '"""',
        "",
        "Description of video B:", '"""', desc_b, '"""',
    ])


def call_diff_llm(client, user_prompt, temperature):
    kwargs = dict(
        model=t2.LLM_MODEL,
        messages=[{"role": "system", "content": DIFF_SYSTEM.strip()},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature, max_tokens=t2.LLM_MAX_TOKENS)
    if t2.USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    out = t2.extract_json(resp.choices[0].message.content)
    if not isinstance(out, dict):
        raise ValueError("diff judge json parse failed")
    return out


def cached_diff(clients, caches, sample, order, prompt, run):
    temp = t2.judge_temperature()
    key = (f"cf_diff|{order}|{sample['video_key']}|{t2.LLM_MODEL}|"
           f"{t2.short_hash(DIFF_SYSTEM, prompt)}|t{temp}|run{run}")
    out = caches["judge"].get(key)
    if out is None:
        out = t2.with_retry(call_diff_llm, clients["llm"], prompt, temp)
        caches["judge"][key] = out
        t2.save_json(caches["judge_path"], caches["judge"])
    return out


def score_diff_run(out, anchors_a, anchors_b, warnings, tag):
    """按细则执行等级约束：无有效差异→0；无结构性差异→最高 1；3 级需主导行为不同且≥2 维结构性差异。"""
    valid, struct_dims = 0, set()
    for it in out.get("differences") or []:
        if not isinstance(it, dict):
            continue
        dim, typ = t2.norm_key(it.get("dimension")), t2.norm_key(it.get("type"))
        if dim not in DIFF_DIMENSIONS or typ not in DIFF_TYPES:
            continue
        a_ok = t2.valid_evidence_anchors([it.get("a")], anchors_a)[0] > 0
        b_ok = t2.valid_evidence_anchors([it.get("b")], anchors_b)[0] > 0
        if not (a_ok or b_ok):
            continue
        valid += 1
        if typ == "present_vs_absent" or (typ == "different_direction" and a_ok and b_ok):
            struct_dims.add(dim)

    level = t2.to_level(out.get("level"), 0, 3)
    if level is None:
        warnings.append(f"diff[{tag}]: invalid level, set to 0")
        level = 0
    dominant = out.get("dominant") if isinstance(out.get("dominant"), dict) else {}
    if level >= 1 and valid == 0:
        warnings.append(f"diff[{tag}]: capped {level}->0 (no valid anchored difference)")
        level = 0
    if level >= 2 and not struct_dims:
        warnings.append(f"diff[{tag}]: capped {level}->1 (no structural difference)")
        level = 1
    if level == 3 and (dominant.get("same") is not False or len(struct_dims) < 2):
        warnings.append(f"diff[{tag}]: capped 3->2 (dominant behavior not different "
                        f"or fewer than two structural dimensions)")
        level = 2
    return {"level": level, "valid_differences": valid,
            "structural_dimensions": sorted(struct_dims),
            "same_setting": out.get("same_setting")}


def evaluate_difference(sample, desc_orig, desc_cf, clients, caches, warnings):
    anchors = {"orig": set(t2.ANCHOR_RE.findall(desc_orig)),
               "cf": set(t2.ANCHOR_RE.findall(desc_cf))}
    descs = {"orig": desc_orig, "cf": desc_cf}
    n = DIFF_RUNS_PER_ORDER or t2.JUDGE_RUNS
    runs = []
    for a_side, b_side in (("orig", "cf"), ("cf", "orig")):   # 交换顺序消除位置偏置
        order = f"A={a_side}"
        prompt = t2.fill(DIFF_PROMPT, input_block=build_diff_input(
            sample, descs[a_side], descs[b_side]))
        for run in range(n):
            out = cached_diff(clients, caches, sample, order, prompt, run)
            res = score_diff_run(out, anchors[a_side], anchors[b_side],
                                 warnings, f"{order}#{run}")
            runs.append({"order": order, **res, "raw": out})

    by_order = defaultdict(list)
    for r in runs:
        by_order[r["order"]].append(r["level"])
    order_means = {k: t2.mean(v) for k, v in by_order.items()}
    same_setting = [r["same_setting"] for r in runs if isinstance(r["same_setting"], bool)]
    l_mean = t2.mean([r["level"] for r in runs])
    return {
        "L_mean": l_mean,
        "D": l_mean / 3,
        "levels": [r["level"] for r in runs],
        "order_means": order_means,
        "order_gap": abs(order_means["A=orig"] - order_means["A=cf"]),
        "same_setting_rate": (sum(same_setting) / len(same_setting)
                              if same_setting else None),
        "runs": runs,
    }


# #####################################################################
# 6. 单样本评测
# #####################################################################

def selectivity(own, other):
    """同一套标注下，把视频换成另一情绪分支的视频后，合理性下降的比例（0~1）。"""
    if own <= 0:
        return 0.0
    return max(0.0, min(1.0, (own - other) / own))


def build_qc_flags(record):
    """参考视频：用于标注质检；生成视频：标记低分样本。"""
    flags = []
    for k in ("orig_on_orig", "cf_on_cf"):
        br = record["branches"][k]
        if br["R"] < QC_LOW_R:
            flags.append(f"low_R_{k}={br['R']:.2f}(S_action={br['S_action']:.2f},"
                         f"S_causal={br['S_causal']:.2f})")
    for k in ("sel_orig", "sel_cf"):
        if record[k] < QC_LOW_SEL:
            flags.append(f"low_{k}={record[k]:.2f}")
    d = record.get("difference")
    if d:
        if d["order_gap"] >= QC_ORDER_GAP:
            flags.append(f"order_gap={d['order_gap']:.2f}")
        if d["same_setting_rate"] is not None and d["same_setting_rate"] < 1:
            flags.append(f"same_setting_rate={d['same_setting_rate']:.2f}")
    if record.get("gated"):
        flags.append("S_counter gated to 0 (different initial setting)")
    return flags



def evaluate_sample(raw, clients, caches):
    warnings = []
    sample, w0 = t2.parse_sample(raw)
    warnings.extend(w for w in w0 if "target_agent" in w)
    sample["initial_state"] = (raw.get("state0") or {}).get("text", "")
    branches = parse_branches(raw, warnings)

    # ---- 视频与描述 ----
    cf_video = branch_oss_key(raw, CF_OSS_PREFIX, "counterfactual")
    if CF_ORIG_SOURCE == "task2":
        orig_video = sample["video_file"]
        desc_orig = t2.get_description(sample, clients, caches)
    else:
        orig_video = branch_oss_key(raw, CF_ORIG_SOURCE)
        desc_orig = get_oss_description(orig_video, clients, caches)
    desc_cf = get_oss_description(cf_video, clients, caches)
    descs = {"orig": desc_orig, "cf": desc_cf}
    for name, d in descs.items():
        if not t2.ANCHOR_RE.search(d):
            warnings.append(f"{name}: description has no [T=XX%] anchors")

    # ---- 交叉矩阵：两段视频 × 两套分支标注 ----
    res = {f"{v}_on_{s}": evaluate_branch(v, s, sample, branches[s], descs[v],
                                          clients, caches, warnings)
           for v in ("orig", "cf") for s in ("orig", "cf")}
    R_oo, R_cc = res["orig_on_orig"]["R"], res["cf_on_cf"]["R"]
    R_co, R_oc = res["cf_on_orig"]["R"], res["orig_on_cf"]["R"]
    sel_orig = selectivity(R_oo, R_co)     # e_orig 的标注是否更适合 V_orig
    sel_cf = selectivity(R_cc, R_oc)       # e_cf 的标注是否更适合 V_cf
    plausibility = (R_oo * R_cc) ** 0.5
    s_counter = plausibility * (sel_orig + sel_cf) / 2

    # ---- 差异评分 L（诊断）与同初始场景门控 ----
    diff = (evaluate_difference(sample, desc_orig, desc_cf, clients, caches, warnings)
            if RUN_DIFF else None)
    gated = False
    if (SAME_SETTING_GATE and diff and (diff["same_setting_rate"] is None
            or diff["same_setting_rate"] < SAME_SETTING_MIN)):
        s_counter, gated = 0.0, True
        warnings.append("S_counter gated to 0: same_setting_rate="
                        f"{diff['same_setting_rate']}")

    record = {
        "protocol_version": PROTOCOL_VERSION,
        "video_key": sample["video_key"],
        "orig_video": orig_video,
        "cf_video": cf_video,
        "scene_category": sample["scene_category"],
        "scene_mode": sample["scene_mode"],
        "e_orig": branches["orig"]["emotion"],
        "e_cf": branches["cf"]["emotion"],
        "e_orig_family": branches["orig"]["emotion_family"],
        "e_cf_family": branches["cf"]["emotion_family"],
        "R_oo": R_oo, "R_cc": R_cc, "R_co": R_co, "R_oc": R_oc,
        "sel_orig": sel_orig,
        "sel_cf": sel_cf,
        "plausibility": plausibility,
        "S_counter": s_counter,
        "gated": gated,
        "L_mean": diff["L_mean"] if diff else None,
        "branches": res,
        "difference": diff,
        "descriptions": descs,
        "warnings": list(dict.fromkeys(warnings)),
    }
    record["qc_flags"] = build_qc_flags(record)
    return record



# #####################################################################
# 7. 汇总
# #####################################################################

def cap_reason(w):
    m = re.search(r"\((.*?)\)", w)
    return m.group(1) if m else w


BRANCH_KEYS = ("orig_on_orig", "cf_on_cf", "cf_on_orig", "orig_on_cf")


def summarize(results, n_skipped):
    ok = [r for r in results if "error" not in r]
    s = {"protocol_version": PROTOCOL_VERSION,
         "n": len(ok), "n_failed": len(results) - len(ok),
         "n_without_counterfactual": n_skipped,
         "vlm_model": t2.VLM_MODEL, "llm_model": t2.LLM_MODEL,
         "judge_runs": t2.JUDGE_RUNS, "aggregate": t2.AGGREGATE,
         "orig_source": CF_ORIG_SOURCE}
    if not ok:
        return s

    for k in ("S_counter", "plausibility", "sel_orig", "sel_cf",
              "R_oo", "R_cc", "R_co", "R_oc", "L_mean"):
        s[k] = t2.mean([r[k] for r in ok])
    s["n_gated"] = sum(1 for r in ok if r.get("gated"))
    for b in BRANCH_KEYS:
        for k in ("S_action", "S_causal", "S_must"):
            s[f"{k}[{b}]"] = t2.mean([r["branches"][b][k] for r in ok])

    diffs = [r["difference"] for r in ok if r.get("difference")]
    s["order_gap"] = t2.mean([d["order_gap"] for d in diffs])
    s["same_setting_rate"] = t2.mean([d["same_setting_rate"] for d in diffs])

    def group(key_fn):
        g = defaultdict(list)
        for r in ok:
            g[key_fn(r)].append(r["S_counter"])
        return {k: {"n": len(v), "S_counter": t2.mean(v)} for k, v in sorted(g.items())}

    s["by_scene"] = group(lambda r: r["scene_category"])
    s["by_emotion_family_pair"] = group(
        lambda r: f"{r['e_orig_family']} -> {r['e_cf_family']}")

    caps = Counter(cap_reason(w) for r in ok for w in r["warnings"] if "capped" in w)
    s["cap_stats"] = dict(caps.most_common())
    ann_keys = ("verify polarity", "no constraints", "e_cf equals",
                "no target_agent", "unknown constraint category")
    s["annotation_warnings"] = {
        r["video_key"]: [w for w in r["warnings"] if any(k in w for k in ann_keys)]
        for r in ok if any(any(k in w for k in ann_keys) for w in r["warnings"])}
    s["flagged"] = {r["video_key"]: r["qc_flags"] for r in ok if r["qc_flags"]}
    return s



def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


def print_summary(s):
    print("\n" + "=" * 60)
    print(f"Counterfactual | VLM={t2.VLM_MODEL} | LLM={t2.LLM_MODEL} | "
          f"runs={t2.JUDGE_RUNS} ({t2.AGGREGATE}) | orig={s['orig_source']}")
    print(f"Evaluated: {s['n']} | Failed: {s['n_failed']} | "
          f"Without counterfactual (skipped): {s['n_without_counterfactual']}")
    if not s["n"]:
        return
    print(f"S_counter    : {fmt(s['S_counter'])}   (gated: {s['n_gated']})")
    print(f"plausibility : {fmt(s['plausibility'])}   (sqrt(R_oo * R_cc))")
    print(f"selectivity  : orig={fmt(s['sel_orig'])}  cf={fmt(s['sel_cf'])}")
    print("\nCross matrix R (rows: video, columns: annotation branch)")
    print(f"{'':12}{'e_orig':>10}{'e_cf':>10}")
    print(f"  {'V_orig':<10}{fmt(s['R_oo']):>10}{fmt(s['R_oc']):>10}")
    print(f"  {'V_cf':<10}{fmt(s['R_co']):>10}{fmt(s['R_cc']):>10}")
    print(f"\n{'':16}{'S_action':>10}{'S_causal':>10}{'S_must':>9}")
    for b in BRANCH_KEYS:
        print(f"  {b:<14}" + "".join(f"{fmt(s[f'{k}[{b}]']):>10}"
                                     for k in ("S_action", "S_causal", "S_must")))
    if s.get("L_mean") is not None:
        print(f"\nDiagnostics: L_mean={fmt(s['L_mean'])}  order_gap={fmt(s['order_gap'])}  "
              f"same_setting_rate={fmt(s['same_setting_rate'])}")
    for title, key in (("By scene category", "by_scene"),
                       ("By emotion family pair (orig -> cf)", "by_emotion_family_pair")):
        print(f"\n{title}:")
        for g, v in s[key].items():
            print(f"  {g}: n={v['n']} S_counter={fmt(v['S_counter'])}")
    if s.get("cap_stats"):
        print("\nCode-side caps:")
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

def _branch_brief(b):
    """一个交叉分支的精简信息：动作的 实现度/一致性、a* 的方式理由、每条约束的判定。"""
    runs = b["judge"]["action_runs"]
    ent = t2._rep_entry(runs, b["a_star"], b["action_scores"][b["a_star"]],
                        t2.normalize_action)
    return {
        "R": round(b["R"], 3),
        "S_action": b["S_action"],
        "a_star": b["a_star"],
        "actions": {a: [f"{r['details'][a]['realization']}/{r['details'][a]['coherence']}"
                        for r in runs] for a in b["action_candidates"]},
        "a_star_manner": t2._short(ent.get("coherence_reason"), 200),
        "S_causal": round(b["S_causal"], 3),
        "items": [f"{it['statement'].split(' | observable_as')[0]} => {'/'.join(it['verdicts'])}"
                  for it in b["items"]],
    }


def review_record(r):
    if "error" in r:
        return {"video_key": r.get("video_key"), "error": r["error"]}
    return {
        "video_key": r["video_key"],
        "e_orig": r["e_orig"], "e_cf": r["e_cf"],
        "matrix": {"R_oo": round(r["R_oo"], 3), "R_oc": round(r["R_oc"], 3),
                   "R_co": round(r["R_co"], 3), "R_cc": round(r["R_cc"], 3)},
        "sel_orig": round(r["sel_orig"], 3), "sel_cf": round(r["sel_cf"], 3),
        "S_counter": round(r["S_counter"], 3),
        "L_mean": r.get("L_mean"),
        "branches": {k: _branch_brief(r["branches"][k]) for k in BRANCH_KEYS},
        "caps": [w for w in r["warnings"] if "capped" in w],
        # orig 描述与 task2_review 中相同，这里只放 cf 描述
        "cf_description": r["descriptions"]["cf"],
    }


def write_review(results, summary, path):
    keep = ("n", "n_failed", "S_counter", "plausibility", "sel_orig", "sel_cf",
            "R_oo", "R_cc", "R_co", "R_oc", "L_mean", "same_setting_rate",
            *[f"{k}[{b}]" for b in BRANCH_KEYS for k in ("S_action", "S_causal")],
            "cap_stats", "annotation_warnings")
    t2.save_json(path, {
        "judge_runs": t2.JUDGE_RUNS,
        "summary": {k: summary[k] for k in keep if k in summary},
        "samples": [review_record(r) for r in results],
    })


# #####################################################################
# 8. 主流程
# #####################################################################

def check_cf_config():
    t2.check_config()           # 复用 Task 2 的 Key / prompt / 路径检查
    errors = []
    for name in ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET"):
        val = getattr(t2, name)
        if not val or val.startswith("<YOUR_"):
            errors.append(f"反事实视频从 OSS 读取，需要在 eval_task2.py 中填写 {name}")
    for name, required in PROMPT_SPECS.items():
        for var in required:
            if "{" + var + "}" not in globals()[name]:
                errors.append(f"{name} 缺少模板变量 {{{var}}}")
    if SAME_SETTING_GATE and not RUN_DIFF:
        errors.append("SAME_SETTING_GATE = True 需要 RUN_DIFF = True")
    if not CF_ORIG_SOURCE:
        errors.append("CF_ORIG_SOURCE 不能为空：填 \"task2\" 或 OSS 前缀")
    try:
        import oss2  # noqa: F401
    except ImportError:
        errors.append("缺少依赖 oss2：pip install oss2")
    if errors:
        raise SystemExit("配置检查未通过：\n  - " + "\n  - ".join(errors))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="删除旧结果后重评（缓存保留）")
    ap.add_argument("--review-only", action="store_true",
                    help="不调用 API，只根据已有结果重新生成评审文件")
    args = ap.parse_args()

    if args.review_only:
        results = t2.load_json(OUT_PATH, [])
        if not results:
            raise SystemExit(f"找不到结果文件：{OUT_PATH}")
        write_review(results, summarize(results, 0), REVIEW_PATH)
        print(f"Review file: {REVIEW_PATH}")
        return

    check_cf_config()
    if args.fresh:
        for p in (OUT_PATH, SUMMARY_PATH, REVIEW_PATH):
            if Path(p).exists():
                Path(p).unlink()
                print(f"removed {p}")

    clients = {"vlm": OpenAI(api_key=t2.VLM_API_KEY, base_url=t2.VLM_BASE_URL),
               "llm": OpenAI(api_key=t2.LLM_API_KEY, base_url=t2.LLM_BASE_URL),
               "bucket": make_bucket()}

    desc_path = Path(t2.CACHE_DIR) / "descriptions.json"
    t2_judge_path = Path(t2.CACHE_DIR) / "judge.json"
    caches = {
        "desc_path": desc_path, "desc": t2.load_json(desc_path, {}),
        "judge_path": JUDGE_CACHE_PATH,
        "judge": t2.load_json(JUDGE_CACHE_PATH, {}),
        "t2": {"desc_path": desc_path, "desc": None,
               "judge_path": t2_judge_path,
               "judge": t2.load_json(t2_judge_path, {})},
        "t3": {"desc_path": desc_path, "desc": None,
               "judge_path": t3.JUDGE_CACHE_PATH,
               "judge": t2.load_json(t3.JUDGE_CACHE_PATH, {})},
    }

    samples = t2.load_json(t2.DATA_PATH, [])
    if t2.LIMIT:
        samples = samples[:t2.LIMIT]
    with_cf = [s for s in samples if isinstance(s.get("counterfactual"), dict)]
    n_skipped = len(samples) - len(with_cf)

    previous = t2.load_json(OUT_PATH, [])
    incompatible = [r for r in previous if "error" not in r
                    and r.get("protocol_version") != PROTOCOL_VERSION]
    if incompatible:
        raise ValueError("Existing CF scores use another protocol; preserve them "
                         "and choose a new output directory")
    results = [r for r in previous if "error" not in r]
    done = {r["video_key"] for r in results}
    pending = [s for s in with_cf if s["video_key"] not in done]
    print(f"Total: {len(samples)} | With counterfactual: {len(with_cf)} | "
          f"Done: {len(done)} | Pending: {len(pending)} | RUN_DIFF={RUN_DIFF}")

    for raw in tqdm(pending, desc="Counterfactual"):
        try:
            record = evaluate_sample(raw, clients, caches)
            tqdm.write(f" ✓ {raw['video_key']}  {record['e_orig']}->{record['e_cf']}  "
                       f"R_oo={record['R_oo']:.2f} R_cc={record['R_cc']:.2f} "
                       f"R_co={record['R_co']:.2f} R_oc={record['R_oc']:.2f}  "
                       f"sel={record['sel_orig']:.2f}/{record['sel_cf']:.2f}  "
                       f"S_counter={record['S_counter']:.3f}")
        except Exception as e:
            logger.exception(f"[FAILED] {raw.get('video_key')}")
            tqdm.write(f" ✗ {raw.get('video_key')}: {e}")
            record = {"video_key": raw.get("video_key"), "error": str(e)}
        results.append(record)
        t2.save_json(OUT_PATH, results)
        time.sleep(t2.SLEEP_SEC)

    summary = summarize(results, n_skipped)
    t2.save_json(SUMMARY_PATH, summary)
    write_review(results, summary, REVIEW_PATH)
    print_summary(summary)
    print(f"\nReview file (paste this one): {REVIEW_PATH}")



if __name__ == "__main__":
    main()
