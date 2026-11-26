"""
EWH-Bench Task 2: Emotional Agent Dynamics Prediction
流程：读取视频 → VLM 细节描述 → LLM 按细则评分 → M_emo / S_action / Score_T2

使用方法：
  1. 在第 1 节填写 Key（替换 <YOUR_...>）
  2. 在第 2 节把视频描述脚本的 PROMPT_TEXT 粘贴到 DESCRIBE_PROMPT
  3. 直接运行：python task2_eval.py
"""
import os, re, json, time, base64, hashlib, logging, random, argparse
from collections import defaultdict, Counter
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# #####################################################################
# 1. 配置（只需修改这里）
# #####################################################################

# ---------- API Key ----------
VLM_API_KEY = "<YOUR_ENVIRONMENT_VALUE>"
LLM_API_KEY = "<YOUR_ENVIRONMENT_VALUE>"
OSS_ACCESS_KEY_ID = "<YOUR_ENVIRONMENT_VALUE>"
OSS_ACCESS_KEY_SECRET = "<YOUR_ENVIRONMENT_VALUE>"

# ---------- 模型 ----------
VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VLM_MODEL = "qwen3.8-max"

LLM_BASE_URL = "https://api.gpt.ge/v1/"
LLM_MODEL = "gpt-5.6-sol"
LLM_MAX_TOKENS = 4000
USE_JSON_MODE = False       # 接口支持 response_format={"type":"json_object"} 时可设为 True

# ---------- 评分 ----------
JUDGE_RUNS = 3              # 每个评分重复次数，>1 时按 AGGREGATE 合并（候选顺序每次重新打乱）
JUDGE_TEMPERATURE = 0.3     # 仅 JUDGE_RUNS > 1 时使用；JUDGE_RUNS = 1 时固定为 0
AGGREGATE = "median"        # 多次评分的合并方式："median" 或 "mean"
SHUFFLE_CANDIDATES = True   # 打乱候选顺序，避免目标情绪总排第一带来的位置偏好
RUN_OPEN_SET = True         # 开放集两步分类诊断（不计入分数）

# 等级 → 分数映射
EMOTION_LEVEL_SCORE = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
REALIZATION_SCORE = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}
# coherence 只用于排除与情绪冲突的实现方式：0 = 冲突；1、2 都视为不冲突
COHERENCE_WEIGHT = {0: 0.0, 1: 0.75, 2: 1.0}
REVIEW_PATH = "task2_review.json"   # 精简评审文件，贴这个即可
CONTROL_MODE = "none"               # 由命令行 --control 设置，不用改

# ---------- 路径 ----------
DATA_PATH = "test.json"
VIDEO_DIR = "."                                # 视频文件名与 video_key 一致
OUT_PATH = "task2_results.json"
SUMMARY_PATH = "task2_results.summary.json"
CACHE_DIR = ".ewh_cache"
LOG_FILE = "task2_eval.log"

# ---------- 运行选项 ----------
LIMIT = None               # 只跑前 N 条，None 表示全部
TRANSPORT = "base64"       # "base64" 或 "oss"
MAX_BASE64_BYTES = 10 * 1024 * 1024
MAX_RETRIES = 3
SLEEP_SEC = 1.0

# ---------- OSS（仅 TRANSPORT = "oss" 时使用）----------
OSS_ENDPOINT = "https://oss-cn-beijing.aliyuncs.com"
OSS_BUCKET = "emotionbenchvideo01"
OSS_PREFIX = "ewh_eval/task2"

# ---------- 对照实验 ----------
# 一次执行会依次运行下面每一组；只想跑正常评测时改成 [("none", False)]
# RUN_GROUPS = [
#     ("none", False),      # A 正常：视频与样本匹配
#     ("mismatch", False),  # B 负对照：给每个样本配一个情绪完全不同的视频，分数应明显下降
#     ("none", True),       # C 隐藏上下文：评分时不给 context/event，分数应与 A 接近
# ]
# CONTROL_MODE = "none"     # 运行时由 RUN_GROUPS 自动设置，不用改
# HIDE_CONTEXT = False      # 运行时由 RUN_GROUPS 自动设置，不用改

# ---------- 质检阈值（仅用于标记需要人工复核的样本，不影响分数）----------
QC_LOW_SCORE = 0.75


# #####################################################################
# 2. Prompt
#    只有 DESCRIBE_PROMPT 需要粘贴，其余已写好
#    模板变量写成 {变量名}；JSON 示例中的花括号直接写单个 { }
# #####################################################################

# ---- VLM：视频细节描述 ----
DESCRIBE_PROMPT = """
You are a grounded video narrative model.

Your task is to produce one detailed, coherent, chronological description of
what is actually shown in the video. A separate reasoning model will later use
your description to infer the video's initial state, important event, possible
emotional change, behavioral response, social interaction, object changes,
and final world state.

The video may be AI-generated and may contain incomplete actions, unusual
transitions, identity inconsistencies, object changes, missing story steps, or
periods in which no person is visible. Describe the video as it is shown. Do
not repair, complete, or normalize its story.

Do NOT output JSON.
Do NOT output headings or bullet points.
Do NOT classify the video's plot.
Do NOT infer hidden thoughts, intentions, motivations, or emotions.

==================================================
1. WRITING STYLE: GROUNDED VISUAL NARRATIVE
==================================================

Write the description as a coherent visual narrative rather than a disconnected
list of frames.

The narrative should make the observable sequence easy to follow:

initial observable state
→ important visible event
→ immediate visible response
→ subsequent action or interaction
→ observable effects on people, objects, or the scene
→ final observable state

Use temporal connections such as:
- "then,"
- "after this,"
- "as the action continues,"
- "at the same time,"
- "a moment later,"
- "the video then cuts to..."

These phrases express observable temporal order only.

Do not convert temporal order into unsupported causality.

Allowed:
"The claw retracts without carrying the toy. The woman then looks down,
removes her hands from the machine, and takes out a wallet."

Not allowed:
"The failed attempt disappoints the woman, so she decides to spend the rest
of her money on another try."

Do not use unsupported psychological or causal expressions such as:
- "because,"
- "therefore,"
- "realizes,"
- "decides,"
- "wants,"
- "intends,"
- "hopes,"
- "regrets,"
- "cannot resist,"
- "acts impulsively,"
- "tries one last time,"
- "is determined to..."

Such information may be included only when it is explicitly conveyed through
clearly readable text, intelligible speech, or a directly observable completed
action. When it comes from text or speech, identify that source explicitly.

The narrative must be coherent, but coherence must come from accurate temporal
connections, not from invented explanations.

Keep each temporal phase readable and information-dense. Prefer one to three
short sentences per temporal marker. Avoid placing the main person's action,
several background people, environmental details, subtitles, and camera
information into one long compound sentence. Separate these information types
when necessary to keep entity-action relationships clear.

==================================================
2. MAIN PERSON AND SOCIAL FOCUS
==================================================

First determine whether the video contains one visually stable MAIN PERSON.

A stable main person is someone who:
- performs the most salient or continuous activity;
- drives the main visible event or interaction;
- receives the most sustained visual attention;
- undergoes the most important observable state change;
- remains visually identifiable across the relevant continuous human-centered
  part of the video.

When a stable main person exists, select that person and keep the same person
as the narrative center.

If the main person is visible at the beginning, begin the paragraph by
introducing the person and the immediate activity, for example:
- "A woman stands at a claw machine..."
- "A man sits across from another person..."
- "A person walks through a store..."

If no person is visible at the beginning, briefly describe the opening shot and
state that no person is visible. Introduce the main person at the first marker
where that person appears.

If no single person remains visually stable enough to serve as the main person,
do not force unrelated people or shots into one identity. Describe the shots in
chronological order and explicitly state that no stable main person is visible
across the video. Do not invent continuity merely to satisfy the main-person
requirement.

When a main person appears only during part of the video:
- keep that person as the narrative center while visible;
- explicitly state when the person first appears;
- explicitly state when a cut removes the person from view;
- state that the person is no longer visible in later landscape, vehicle,
  title-card, or unrelated-person shots;
- do not describe those later shots as the person's bodily or behavioral final
  state;
- do not imply that a landscape or scene cut was caused by the person's action.

Do not switch the narrative focus to another person. However, describe other
people whenever they:
- directly interact with the main person;
- look toward or respond to the main person;
- approach, withdraw from, touch, or follow the main person;
- give an object to or receive an object from the main person;
- visibly change their activity after the main person's action;
- contribute to a group-level interaction change.

Describe these people in relation to the main person.

Good:
"The woman approaches a person behind the counter, who looks up and extends
one hand toward her."

Avoid changing the main subject:
"A worker stands at the counter and begins handling several objects..."

If another person becomes more visually prominent after entering the frame,
continue grounding the description in the selected main person whenever that
person remains visible. Describe the other person's actions in relation to the
main person rather than silently replacing the main person as the narrative
center.

If the identity of the apparent main person becomes visually inconsistent
across an AI-generated transition or an ordinary cut, do not silently assume
perfect continuity. Briefly state the visible inconsistency or uncertainty, for
example:
- "After the cut, a similarly dressed woman appears at the counter, although
  her facial appearance is not fully consistent with the earlier shot."
- "The same apparent woman continues walking, but the color of her clothing
  changes between shots."
- "A similarly dressed person appears after the cut, but the video does not
  provide enough visual continuity to confirm that this is the same person."

Do not infer that two people are the same solely because they have similar
clothing, hair, or location.

Do not invent names, relationships, occupations, or character identities.

Use terms such as "employee," "mother," "child," "doctor," or "teacher" only
when the role is clearly established by visible context, clothing, readable
text, subtitles, or intelligible speech. Otherwise use neutral descriptions
such as:
- "a person behind the counter,"
- "another woman,"
- "a seated man."

==================================================
3. TEMPORAL STRUCTURE
==================================================

Write one paragraph with embedded temporal markers in this format:

[T=XX%]

Requirements:
- The output must begin exactly with [T=0%].
- Do not place any text before [T=0%].
- The output must end with a [T=100%] description of the final visible state.
- Percentages must be strictly increasing.
- Use approximate video-time percentages.
- Add markers only when a meaningful visible phase begins.
- Do not divide the video into arbitrary equal intervals.
- Do not create a new phase when nothing relevant changes.
- Use enough markers to preserve all major stages, normally about 4 to 10,
  depending on the complexity of the video.

Insert a new marker when there is a meaningful change in:
- action or task stage;
- visible task outcome;
- posture or body orientation;
- gaze or attention;
- facial configuration;
- hand or arm activity;
- movement direction, speed, amplitude, or force;
- object use, contact, possession, location, or condition;
- interaction target;
- another person's response;
- interpersonal distance or physical contact;
- group interaction;
- shot, time, or location.

A subtitle, caption, logo, or on-screen text change alone does NOT define a new
temporal phase. Do not insert a new marker only because the subtitle text
changes while the visible person, action, posture, interaction, and setting
remain substantially unchanged. Include changing subtitle content within the
current phase when it is relevant.

A camera movement or reframing alone does not require a new marker unless it:
- reveals a new relevant person, object, or spatial relation;
- begins a new shot, location, or time;
- changes what can be observed about the main action;
- accompanies a meaningful change in behavior or interaction.

The final marker may describe:
- the main person's settled final state;
- an incomplete action still in progress;
- the moment the main person is no longer visible after a cut;
- a final landscape, vehicle, title card, or unrelated shot, provided it is
  explicitly distinguished from the main person's state.

Do not invent a new final action.

==================================================
4. INFORMATION THAT MUST BE PRESERVED
==================================================

Describe the following whenever it is visible and relevant.

--------------------------------------------------
A. Initial observable state
--------------------------------------------------

At the beginning of the relevant human-centered portion, establish:
- the main person's visible appearance sufficiently for tracking;
- the physical setting;
- the main person's location;
- posture and body orientation;
- gaze or attention target, when visible;
- current action or task;
- relevant objects being held, touched, watched, or used;
- nearby people who are already interacting with the main person;
- the initial state of important objects and relationships.

If no person is visible at [T=0%], describe the opening setting briefly and
explicitly state that no person is visible. Establish the main person's initial
state when that person first appears.

Do not overload the description with decorative details.

Include environmental details only when they help establish:
- the task;
- the event;
- person or object identity;
- spatial relationships;
- a later state change.

For example, the claw machine, plush toy, wallet, cash, counter, and possible
tokens may be important. Decorative lights, unrelated machine logos, wall
colors, and background toys are usually unnecessary unless they later matter.

--------------------------------------------------
B. Important observable events
--------------------------------------------------

Clearly describe events that occur immediately before a meaningful change in
the main person's behavior or the social scene.

Relevant events may include:
- success or failure of a visible task;
- receiving, reading, or looking at a message;
- another person entering, leaving, speaking, gesturing, or making contact;
- an object falling, breaking, moving, or being handed over;
- a phone or screen changing;
- a payment or exchange beginning or completing;
- a sudden environmental event;
- readable subtitles or intelligible speech.

A task result may be described as success or failure only when:
- the task goal is visually clear;
- the outcome is directly visible.

For example:
"The claw closes around the toy but retracts without carrying it, and the toy
remains inside the machine."

Do not describe an unseen event merely because a later action would make it
plausible.

A shot change, title card, scenic montage, or unrelated location change is a
visible editing transition, not automatically an event caused by the main
person. Describe it as a cut or transition unless the video visibly establishes
a direct action connection.

--------------------------------------------------
C. Immediate response to the event
--------------------------------------------------

Preserve the main person's first visible response after an important event,
even when it is subtle.

Describe:
- whether the person remains still or moves immediately;
- gaze shifts;
- head movement;
- change in posture;
- hands leaving or gripping an object;
- stepping forward or backward;
- turning away or toward something;
- visible facial movements;
- a pause, repeated action, or abrupt redirection.

The timing and manner of the response are important.

If clearly visible, distinguish:
- immediate action;
- action after a pause;
- hesitant or repeated movement;
- slow versus rapid movement;
- smooth versus abrupt redirection;
- continued task engagement versus disengagement.

Use concrete observations:
- "She remains still briefly and looks down."
- "She immediately turns away without pausing."
- "She looks between the wallet and the machine twice."
- "She slowly removes her hands from the controls."
- "She abruptly turns toward the counter and walks quickly."

Do not translate these observations into emotion, impulsiveness, intention, or
motivation.

--------------------------------------------------
D. Facial and bodily evidence
--------------------------------------------------

When sufficiently visible, describe concrete physical facial cues:
- eyebrows raising, lowering, or drawing together;
- eyes widening, narrowing, blinking, or looking downward;
- lips pressing together or parting;
- mouth corners rising or lowering;
- jaw tightening or opening;
- smiling, laughing, crying, or visible tears;
- facial tension increasing or relaxing.

Also describe relevant bodily cues:
- shoulder position;
- body rigidity or relaxation;
- hand tension;
- repeated hand movement;
- freezing;
- recoiling;
- leaning forward or backward;
- movement speed and amplitude.

Describe only cues that are clearly visible.

Prefer changes over isolated configurations.

More useful:
"After the claw retracts, her eyebrows lower and she presses her lips together."

Less useful:
"Her eyebrows are slightly low."

If the face is visible but no reliable facial change occurs, do not invent one.
The later reasoning model can use posture, action, timing, and context instead.

Do not directly label the person as:
- happy,
- sad,
- disappointed,
- angry,
- afraid,
- anxious,
- concerned,
- surprised,
- disgusted,
- frustrated,
- impulsive,
- determined.

--------------------------------------------------
E. Action and task trajectory
--------------------------------------------------

Describe the main person's continuous observable action chain.

Preserve:
- the task being performed;
- whether the task continues, stops, changes, or resumes;
- movement from one task location to another;
- changes in attention or engagement target;
- preparation for a new action;
- completed and incomplete actions;
- return to a previous task, if shown.

Use ordinary action terms when the action is visually established:
- operating a claw machine;
- taking out a wallet;
- opening a wallet;
- removing cash;
- approaching a counter;
- extending money;
- receiving tokens;
- returning to a machine;
- inserting a token.

Distinguish action stages carefully.

For a transaction:
- approaching a counter is not a completed purchase;
- showing or extending money is not a completed payment;
- another person holding tokens is not proof that the tokens were received;
- an exchange is completed only when the transfer is visibly shown.

If the action remains incomplete at the end of the video, state its final
visible stage rather than completing it through common sense.

Example:
"The woman continues holding the bills toward the person behind the counter,
while the possible tokens remain in the other person's hand."

If a cut interrupts the action:
- describe the last completed visible stage before the cut;
- do not assume that the next shot completes the action;
- connect the shots only when the same person, object, and task remain visually
  identifiable.

--------------------------------------------------
F. Objects and physical world changes
--------------------------------------------------

For every important object, preserve:
- its initial state or location;
- who touches or holds it;
- the action performed on it;
- changes in location, possession, contact, or condition;
- its final visible state.

Important examples include:
- a toy remaining inside a machine;
- a wallet moving from a pocket to a hand and then to a counter;
- a wallet being opened;
- cash being removed and extended;
- tokens changing hands;
- a phone screen lighting up;
- an object being placed, dropped, opened, or broken.

Do not skip intermediate object changes when they help connect the action chain.

For example, do not jump directly from:
"She puts the wallet on the counter"
to:
"She holds three bills."

If visible, include:
"She opens the wallet, removes three folded bills, and extends them toward the
person behind the counter."

When an object's identity is uncertain, describe its visible properties and
state the uncertainty briefly:
- "several small round objects that may be arcade tokens";
- "a rectangular object that resembles a phone."

Do not turn an uncertain object into a definite story element.

Do not infer who owns an object merely from its location or from the viewpoint.
For example, write "a light-colored shoe enters the bottom of the frame," not
"the narrator's foot," unless visual continuity clearly establishes whose foot
it is.

--------------------------------------------------
G. Other people's responses and social changes
--------------------------------------------------

Describe socially relevant changes such as:
- another person looking up;
- turning toward the main person;
- approaching or moving away;
- extending a hand;
- handing over or receiving an object;
- stopping or beginning visible speech;
- changing interpersonal distance;
- beginning, ending, or redirecting an interaction.

For groups, describe concrete collective changes:
- several people turn toward the main person;
- visible conversation activity decreases;
- the group becomes still;
- people move closer together or farther apart;
- turn-taking becomes one-sided;
- multiple people stop their previous activity.

Do not replace these observations with abstract claims such as:
- "the atmosphere becomes tense";
- "the interaction becomes awkward";
- "their relationship worsens";
- "everyone becomes concerned."

Describe the evidence from which the reasoning model may infer such changes.

--------------------------------------------------
H. Text, subtitles, screens, audio, and camera viewpoint
--------------------------------------------------

If readable text, subtitles, screen content, or intelligible speech provides
important information, include it accurately and identify the source.

Examples:
- 'A subtitle reads, "I only have enough money for one more try."'
- 'The phone screen shows an available balance of "$5.00".'
- 'A message on the phone reads, "The exam result is below passing."'
- 'The woman says, "One last try," if the speech is clearly audible.'

Do not reconstruct blurred, cropped, hidden, or unreadable content.

Do not treat a subtitle as proof that a visible person is speaking unless:
- the person's mouth movement and the audio are clearly synchronized;
- the subtitle is explicitly associated with that speaker;
- or the video otherwise makes the speaker identity clear.

If the speaker is not visible or cannot be identified, write:
- 'A subtitle reads, "...".'
- 'An off-screen voice says, "...",' only if the speech is intelligible.
- 'The speaker is not visible or cannot be identified.'

If only mouth movement is visible and the audio source is unclear, describe the
mouth movement without writing "speaks" or "as if speaking."

If audio is unavailable or unclear:
- do not infer dialogue;
- do not infer vocal tone;
- do not claim that the scene becomes quiet;
- do not invent conversation content.

Do not infer who is holding or operating the camera from framing alone.

A selfie-style composition does not prove that the visible person is holding
the camera. Unless a hand, arm, camera, selfie stick, reflection, or clear
holding action is directly visible, describe only the composition, for example:
- "The man fills the foreground in a close selfie-style shot."
- "The camera shows a moving first-person view."
- "A handheld view follows the person."

Do not write:
- "he holds the camera at arm's length";
- "she films herself";
- "the narrator points the camera";
unless the holding or filming action is directly visible.

Irrelevant logos, decorative text, machine branding, and background signs
should normally be omitted.

==================================================
5. AI-GENERATED VIDEO HANDLING
==================================================

The video may contain visual or temporal inconsistencies.

Do not silently repair:
- a person's face or clothing changing;
- hands or fingers deforming;
- an object changing shape, color, or identity;
- an object appearing or disappearing without a visible action;
- a person or object changing location across a cut;
- an incomplete handover;
- physically impossible contact;
- duplicated people or objects;
- a movement that begins but does not finish;
- a discontinuity between two shots.

If such an inconsistency affects the interpretation of the event or state
change, describe it briefly and neutrally.

Examples:
- "Across the cut, the wallet changes from black to brown without a visible
  exchange."
- "The small round objects appear in the person's hand between frames without
  a clearly visible pickup."
- "The woman's right hand becomes briefly distorted as she reaches toward the
  counter."
- "The apparent exchange is visually discontinuous, so it is unclear whether
  the objects change hands."

Do not overemphasize minor generation artifacts that do not affect the action
or world-state interpretation.

Most importantly, do not use story expectations to fill missing steps.

If the video shows:
failed claw attempt
→ wallet taken out
→ movement toward counter
→ cash extended

but does not show:
payment completed
→ tokens received
→ return to machine
→ another attempt

then describe only what is shown.

==================================================
6. UNCERTAINTY AND ABSENT INFORMATION
==================================================

Do not guess.

Use concise uncertainty statements only when uncertainty affects an important
event or state transition:
- "The phone screen is not readable."
- "The object may be an arcade token, but its identity is not fully clear."
- "The transfer is obscured by the person's hand."
- "It is unclear whether the exchange is completed."
- "The video ends before the other person takes the money."
- "No return to the machine is shown."
- "The similarly dressed person after the cut cannot be confirmed as the same
  person."
- "The main person is no longer visible after the cut."

Do not list every story element that is absent.

Mention an absent step only when:
- the visible action strongly approaches that step;
- falsely assuming completion would change the interpretation;
- a cut removes the main person before the action reaches a visible outcome.

For example, when cash and tokens are extended but no transfer occurs, it is
useful to state that the exchange is not completed.

Do not mention unrelated absent information.

==================================================
7. RELEVANCE CONTROL
==================================================

Prioritize details that help reconstruct:

1. whether a stable central person exists and when that person is visible;
2. the initial person, object, spatial, and social state;
3. the main visible event;
4. the person's immediate visible response;
5. the subsequent action trajectory;
6. relevant facial and bodily evidence;
7. object-state and possession changes;
8. interactions and reactions from other people;
9. the final visible world state;
10. important uncertainty or discontinuity.

Usually omit:
- decorative lighting;
- irrelevant colors;
- background machine names;
- unrelated objects;
- clothing details beyond what is needed to track identity;
- camera descriptions that do not affect continuity;
- repeated statements of an unchanged state;
- repeated subtitle changes that do not correspond to visible behavioral
  changes.

The description should be detailed but information-dense.

==================================================
8. OUTPUT FORMAT
==================================================

Output ONLY one coherent chronological paragraph.

The paragraph must:
- begin exactly with [T=0%];
- introduce the main person, setting, and initial activity immediately when a
  main person is visible;
- if no person is visible at the beginning, briefly describe the opening and
  explicitly state that no person is visible;
- remain centered on the same main person whenever a stable main person exists;
- explicitly state when no stable main person exists rather than inventing
  identity continuity;
- explicitly state when a cut removes the main person from view;
- use embedded [T=XX%] temporal markers;
- create markers for meaningful visible phases, not subtitle changes alone;
- describe all meaningful visible phases;
- preserve relevant other-person responses;
- preserve important object and world-state changes;
- clearly distinguish completed from incomplete actions;
- distinguish editing transitions from person-driven world changes;
- mention important AI-video inconsistencies only when relevant;
- use temporal narrative connections without invented causality;
- use clear, reasonably short sentences;
- contain no JSON;
- contain no headings;
- contain no bullet points;
- contain no emotion labels;
- contain no hidden-intention or motivation claims;
- end with [T=100%] and the final observable state.

Before answering, silently verify that the paragraph contains, when visible:

- whether one stable main person exists;
- the interval during which that person remains visible;
- initial main-person state;
- initial task and important objects;
- central event or task outcome;
- immediate visible response;
- later behavior and movement;
- object changes;
- other-person reactions;
- final person, object, spatial, and interaction state;
- any important incomplete action or AI-generated discontinuity;
- an explicit statement if the main person disappears before the video ends;
- no marker created solely because a subtitle changed;
- no unsupported claim about who holds the camera, owns a visible body part or
  object, or speaks a subtitle.
"""

# ---- LLM 评分器：system prompt ----
JUDGE_SYSTEM = """
You are a strict, evidence-bound evaluator for an emotion-aware world-model
benchmark. You judge only from a written description of a video. You never
invent observations, never complete the story from common sense, never favor a
candidate because of its position in a list or because it fits the event, and
you apply the rubric literally. Output one valid JSON object and nothing else.
"""

# ---- 通用取证规则（插入到各评分 prompt 中）----
GROUNDING_RULES = """
GROUNDING RULES (apply to every judgment)
1. The video description is the ONLY source of observations. The context and
   event tell you what happened before or at the start of the video; use them
   only to interpret the meaning of cues the description actually reports,
   never as a source of cues.
2. Every evidence item must start with a [T=XX%] anchor that appears in the
   description and must paraphrase what the description states at that
   anchor. Never cite anything the description does not state.
3. A cue that is not mentioned is absent. Do not assume an expression, gaze,
   voice, or action that "would be expected" in this situation.
4. If the INPUT names a Main subject, judge only that person: map it to the
   described person who best matches its description, even if another person
   is more prominent. Otherwise, the main subject is the person who reacts to
   the event. If no described person matches, or no stable identifiable main
   subject exists, set "subject_visible" to false.
5. Judge each candidate independently against the rubric. Do not rank the
   candidates against each other, do not assume that any candidate must be
   correct, and ignore the listing order. Several candidates may receive the
   same level, including 0.
6. Match on CORE MEANING, not on surface form. Typical cues are examples, not
   requirements: a behavior that realizes the same core in a different form is
   a match. A behavior that only looks similar but serves a different core is
   not a match.
7. The description was written under a rule that forbids asserting speech
   when audio is unclear. An open or moving mouth directed at another person
   is therefore a valid cue of vocal or verbal activity; do not treat the
   absence of the word "speaks" as absence of communication.
"""

# ---- 开放集：先大类后小类 ----
OPEN_SET_PROMPT = """
TASK: Blind classification of the main subject's emotion and dominant action
tendency after the event. No candidate answers are given. Choose from the
closed taxonomy with a two-step, top-down procedure.

{input_block}

{grounding_rules}

EMOTION TAXONOMY (family -> subcategories with their core meaning)
{emotion_taxonomy_block}

ACTION TENDENCY TAXONOMY (category -> subcategories with their core function)
{action_taxonomy_block}

PROCEDURE
Emotion:
  E1. List every emotion cue the description reports for the main subject
      (face, gaze, voice or speech, posture), each with its anchor.
  E2. Choose exactly ONE family whose definition best fits those cues,
      interpreted in light of the event. Choose NEUTRAL when the cues show no
      directional emotion. Name the runner-up family and state why it was
      rejected.
  E3. Only inside the chosen family, choose exactly ONE subcategory whose core
      best fits the cues. The subcategory MUST belong to the family chosen in
      E2. If no subcategory fits, revise E2 instead of taking a subcategory
      from another family.
Action:
  A1. List every observable behavior of the main subject (movement, gesture,
      posture change, spatial movement, object handling, interaction), each
      with its anchor.
  A2. Choose exactly ONE category that describes the FUNCTION of the dominant
      behavior, i.e., the most sustained or most consequential one. Name the
      runner-up category and state why it was rejected.
  A3. Only inside the chosen category, choose exactly ONE subcategory. It MUST
      belong to the category chosen in A2.
Use the exact spelling of family, category, and subcategory names as listed.

OUTPUT (exactly this JSON structure)
{
  "main_subject": "<short description of the main subject>",
  "subject_visible": true,
  "emotion": {
    "cues": ["[T=XX%] ..."],
    "family": "<family name>",
    "family_reason": "<why this family; which family was runner-up and why it was rejected>",
    "label": "<subcategory inside the chosen family>",
    "label_reason": "<why this subcategory's core fits better than its siblings>"
  },
  "action": {
    "cues": ["[T=XX%] ..."],
    "category": "<category name>",
    "category_reason": "<why this category; which category was runner-up and why it was rejected>",
    "label": "<subcategory inside the chosen category>",
    "label_reason": "<why this subcategory's core fits better than its siblings>"
  }
}
"""

# ---- 情绪匹配评分 ----
EMOTION_SCORE_PROMPT = """
TASK: For each candidate emotion, rate how well the main subject's emotional
tone after the event matches the candidate's CORE meaning.

{input_block}

{grounding_rules}

CHANNEL PRIORITY
Judge emotional tone primarily from facial expression, gaze, and voice or
speech content. Posture and body movement may support a judgment, but on their
own they cannot justify a level above 2.

CORE-MEANING MATCHING
Counts as a match:
- Disappointed: the subject quietly closes the laptop and looks away from the
  presenter, even though no sigh or frown is reported.
- Mad: the subject slams the document onto the table and speaks sharply to the
  other person, even though no furrowed brow is reported.
Does NOT count as a match:
- Tears while laughing are not sadness.
- Looking down to read a document is not dejection.
- Smiling at the camera while posing is not joy about the event.

CANDIDATES (the order is random and carries no meaning)
{candidate_block}

RUBRIC (one integer level per candidate)
4 = Strong match. At least TWO distinct reported cues from different channels
    (face, gaze, voice/speech, posture, action) realize the candidate's core,
    at least one of them from face, gaze, or voice/speech; together they form
    a sensible reaction to the event; no reported cue contradicts the
    candidate.
3 = Clear match. Exactly ONE clear cue from face, gaze, or voice/speech
    realizes the core, or several cues realize it but one of them is
    ambiguous; no reported cue contradicts the candidate.
2 = Partial match. The reported cues fit the candidate's family (same valence,
    similar energy) but do not distinguish its core from emotions outside the
    candidate list; OR only posture/body cues support the core; OR the core is
    supported but one reported cue points to a different emotion.
1 = Weak. The candidate is plausible only from the event or context; the
    description reports no cue that realizes its core, or the reported cues
    are neutral.
0 = No match. Reported cues contradict the candidate (opposite valence or
    clearly another family), or the main subject is not visible or cannot be
    identified.

DISCRIMINATION CHECK (required before assigning level 3 or 4)
All candidates are acceptable answers for this sample, so you do NOT need to
separate a candidate from the OTHER CANDIDATES; two candidates may both be
distinguished and both receive a high level. You must separate each candidate
from emotions OUTSIDE the candidate list.
For each candidate, name its closest alternative, chosen ONLY from this list
of non-candidate emotions:
{outside_label_list}
Then decide whether at least one reported cue realizes something in the
candidate's core that this closest alternative lacks. The event may help
interpret a reported cue, but it can never replace one. If the candidate is
not distinguished from its closest non-candidate alternative, set
"distinguished" to false and give a level of at most 2.


LEVEL CONSTRAINTS
- Levels 2, 3, and 4 require at least one evidence item with a valid anchor.
- Level 4 requires at least two evidence items with different anchors.
- Levels 3 and 4 require "distinguished": true.
- If a listed contradiction is as strong as the supporting evidence, the level
  is at most 2.

OUTPUT (exactly this JSON structure; one entry per candidate; use the candidate
names exactly as listed)
{
  "main_subject": "<short description>",
  "subject_visible": true,
  "reported_emotion_cues": ["[T=XX%] ..."],
  "candidates": [
    {
      "label": "<candidate name>",
      "core_analysis": "<which reported cue, if any, realizes this candidate's core, and why at the level of core meaning>",
      "evidence": ["[T=XX%] ..."],
      "contradictions": ["[T=XX%] ..."],
      "closest_alternative": "<emotion name from the non-candidate list>",
      "distinguishing_cue": "[T=XX%] ... or empty if none",
      "distinguished": false,
      "level": 0
    }
  ]
}
"""

# ---- 动作匹配评分（以 e* 为条件）----
ACTION_SCORE_PROMPT = """
TASK: Assume the main subject feels {emotion} (core: {emotion_core}). For each
candidate action tendency, rate (A) whether the subject's observable behavior
after the event realizes the candidate's CORE FUNCTION, and (B) whether the
MANNER of that behavior is consistent with {emotion}.

{input_block}

{grounding_rules}

CHANNEL PRIORITY
Judge from body movement, gesture, posture change, spatial movement, object
handling, and interaction rhythm. Facial expression alone does not realize any
action tendency except Expression or Suppression.

CORE-FUNCTION MATCHING
Counts as a match:
- Withdrawal: the subject turns the chair away from the person talking to them
  and stops responding, even though the subject does not leave the room.
- Approach: the subject leans across the table toward the other person, even
  though the subject does not stand up.
Does NOT count as a match:
- Walking to the door to fetch an object is not Escape.
- Putting down a pen in order to pick up a document is not Disengagement.
- Pausing because the other person starts talking is not Freeze.

DEFINITIONS THAT APPLY TO ALL CANDIDATES
- Target. Social Interaction and Conflict Dynamics tendencies (Engagement,
  Withdrawal, Seek Support, Provide Support, Confrontation, Escalation,
  De-escalation) require another person reported present as their target.
  For Withdrawal the target may also be a group or an ongoing social situation
  the subject was part of (e.g., leaving the court where teammates are, or
  falling silent and lowering the head in a meeting), even if no one-to-one
  exchange is reported. Task & Environment tendencies target a task or an
  object. Attention & Cognition tendencies target a person, object, or event
  in the world. Approach-Avoidance tendencies target a person, object, or
  place. The camera and the viewer are never a target.
- Valence of interaction. Engagement, Seek Support, and Provide Support are
  neutral or positive interaction; Confrontation and Escalation are
  oppositional; De-escalation reduces opposition.
- Opposite pairs. Approach / Avoidance, Engagement / Withdrawal, Expression /
  Suppression, Escalation / De-escalation, Task Persistence / Disengagement.
  A behavior whose function is one member of a pair gives realization 0 for
  the other member.
- Change. A tendency is realized by what the subject does after or in
  response to the event. A state present from the start of the video that
  does not change does not by itself realize a tendency, unless maintaining
  that state is the tendency's core (e.g., Task Persistence).
- Co-occurrence. Several tendencies may be realized by the same episode when
  each core function is independently served. Do not lower a candidate only
  because another candidate is also realized.

FUNCTION CHECK
Set "function_established" to true only if a reported behavior serves exactly
this candidate's core function, not merely co-occurs with it or shares a
surface component, after applying the definitions above. If false,
realization is at most 2.

CANDIDATES (the order is random and carries no meaning)
{candidate_block}

RUBRIC A - realization (integer 0-4)
4 = Two or more reported behaviors, or one completed decisive act, whose
    function is exactly the candidate's core (e.g., the subject walks out of
    the room -> Escape).
3 = One clear reported behavior realizes the core, but it is brief, partial,
    or unfinished.
2 = A reported behavior shares a component with the candidate but fits another
    tendency equally well, or its function is ambiguous.
1 = Only a faint hint, or the tendency is merely implied by the event or
    context without a reported behavior.
0 = Not reported, or the reported behavior has the opposite function.

RUBRIC B - coherence of the MANNER with {emotion} (integer 0-2)
The candidate tendencies are already annotated as plausible for this event.
Do NOT judge whether the tendency category suits {emotion}. Judge only HOW
the realizing behavior is carried out: its speed, force, amplitude, the
accompanying facial and bodily tone, and how it treats other people.
Typical cues of {emotion} (examples only, not requirements): {emotion_cues}
2 = The manner carries cues consistent with {emotion}'s core.
1 = The manner is neutral: it carries no cue specific to {emotion} and none
    specific to a contrasting emotion, or the description does not report
    how the behavior is carried out.
0 = The manner carries cues typical of a contrasting emotion (e.g., a gentle,
    comforting approach attributed to anger; a celebratory gesture attributed
    to disappointment), or realization is 0.

LEVEL CONSTRAINTS
- Realization 2, 3, or 4 requires at least one evidence item with a valid
  anchor.
- Realization 3 or 4 requires "function_established": true.
- If realization is 0, coherence must be 0.
- Coherence 0 or 2 must cite the reported cue of manner in "coherence_reason".

OUTPUT (exactly this JSON structure; one entry per candidate; use the candidate
names exactly as listed)
{
  "main_subject": "<short description>",
  "subject_visible": true,
  "reported_behaviors": ["[T=XX%] ..."],
  "candidates": [
    {
      "label": "<candidate name>",
      "function_analysis": "<which reported behavior, if any, realizes this candidate's core function, and which definition above it was checked against>",
      "evidence": ["[T=XX%] ..."],
      "function_established": false,
      "realization": 0,
      "coherence_reason": "<which reported cue of manner, if any, is consistent or inconsistent with the assumed emotion>",
      "coherence": 0
    }
  ]
}
"""



# 各 prompt 必须包含的模板变量（启动自检用）
PROMPT_SPECS = {
    "DESCRIBE_PROMPT": [],
    "JUDGE_SYSTEM": [],
    "OPEN_SET_PROMPT": ["input_block", "grounding_rules",
                        "emotion_taxonomy_block", "action_taxonomy_block"],
    "EMOTION_SCORE_PROMPT": ["input_block", "grounding_rules",
                             "candidate_block", "outside_label_list"],
    "ACTION_SCORE_PROMPT": ["input_block", "grounding_rules", "candidate_block",
                            "emotion", "emotion_core", "emotion_cues"],
}



# #####################################################################
# 3. 封闭集合、标签释义与归一化
# #####################################################################

EMOTION_TAXONOMY = {
    "SAD": ["Guilty", "Despair", "Lonely", "Bored", "Disappointed", "Awful"],
    "FEAR": ["Scared", "Anxious", "Insecure", "Rejected"],
    "ANGER": ["Mad", "Hurt", "Threatened", "Distant"],
    "SURPRISE": ["Confused", "Startled", "Amazed", "Excited"],
    "HAPPY": ["Joyful", "Proud", "Optimistic", "Peaceful", "Loving", "Hopeful"],
    "DISGUST": ["Avoidance", "Hesitant", "Aversion", "Judgmental", "Revolted",
                "Disapproval"],
}
EMOTION_FAMILIES = dict(EMOTION_TAXONOMY, NEUTRAL=["Neutral"])

ACTION_TAXONOMY = {
    "Approach-Avoidance": ["Approach", "Avoidance", "Freeze"],
    "Stress Response": ["Escape", "Appeasement"],
    "Emotion Regulation": ["Expression", "Suppression", "Self-Calming"],
    "Social Interaction": ["Engagement", "Withdrawal", "Seek Support",
                           "Provide Support"],
    "Conflict Dynamics": ["Confrontation", "Escalation", "De-escalation"],
    "Attention & Cognition": ["Shift", "Seek Info"],
    "Task & Environment": ["Object Interaction", "Task Persistence",
                           "Disengagement"],
}

EMOTION_FAMILY_DEFS = {
    "SAD": "negative, usually low-energy states arising from loss, failure, unmet expectations, isolation, or self-blame",
    "FEAR": "negative states arising from an anticipated threat or uncertainty concerning oneself, including social threat",
    "ANGER": "negative states directed at another agent perceived as causing a wrong, an obstruction, hurt, or a challenge",
    "SURPRISE": "states triggered by something unexpected or novel; valence may be positive, negative, or mixed",
    "HAPPY": "positive states arising from goal attainment, pleasant situations, affection, or positive expectations",
    "DISGUST": "states of rejecting something or someone perceived as offensive, unpleasant, or unacceptable",
    "NEUTRAL": "no directional emotional state; ordinary baseline behavior",
}

ACTION_CATEGORY_DEFS = {
    "Approach-Avoidance": "basic movement toward, away from, or stopping in relation to a stimulus",
    "Stress Response": "leaving the situation or placating others to reduce pressure on oneself",
    "Emotion Regulation": "how the subject handles the display and intensity of their own emotion",
    "Social Interaction": "increasing, decreasing, or support-oriented interaction with other people",
    "Conflict Dynamics": "opposing someone and changing the intensity of a conflict",
    "Attention & Cognition": "redirecting attention or actively gathering information",
    "Task & Environment": "the subject's relation to objects and to an ongoing task",
}

# core：核心含义（匹配依据）；cues：典型线索（仅为示例）
EMOTION_DEFS = {
    "guilty": {"core": "self-blame for one's own wrongdoing or for harm caused to others",
               "cues": "averted gaze, lowered head, apologetic words or gestures, attempts to make amends"},
    "despair": {"core": "belief that a bad situation cannot be changed; giving up",
                "cues": "collapsed posture, face in hands, crying, stopping all effort"},
    "lonely": {"core": "painful sense of being socially isolated or disconnected",
               "cues": "staying apart, watching others from a distance, sitting alone, unanswered attempts to connect"},
    "bored": {"core": "lack of interest or stimulation in the current activity",
              "cues": "slack posture, yawning, fidgeting, unfocused gaze, idly checking a phone"},
    "disappointed": {"core": "an expected or hoped-for outcome did not happen",
                     "cues": "face falls, gaze drops, shoulders sag, sighing, stopping an expectant activity"},
    "awful": {"core": "diffuse, strongly unpleasant negative state without a specific target",
              "cues": "tense or pained face, heavy slow movements, holding the head"},
    "scared": {"core": "appraisal of an immediate threat to oneself",
               "cues": "wide eyes, recoiling, freezing, flinching, shielding, fleeing"},
    "anxious": {"core": "worry about an uncertain future outcome",
                "cues": "restlessness, fidgeting, repeated checking, tense face, pacing"},
    "insecure": {"core": "self-doubt about one's own adequacy or standing",
                 "cues": "hesitant speech or movement, shrinking posture, avoiding eye contact, seeking reassurance"},
    "rejected": {"core": "feeling excluded, dismissed, or unwanted by others",
                 "cues": "withdrawing after being turned away, looking down after a refusal, pulling back an extended hand"},
    "mad": {"core": "blaming someone or something for a wrong or obstruction, with an urge to oppose it",
            "cues": "furrowed brows, glaring, clenched jaw, raised voice, forceful gestures, pointing"},
    "hurt": {"core": "emotionally wounded by another person's words or actions (not physical pain)",
             "cues": "pained look, tearfulness, trembling lips, pulling back from that person"},
    "threatened": {"core": "one's position, status, or safety is challenged by someone",
                   "cues": "guarded or defensive posture, crossed arms, stepping back while watching the other, vigilance"},
    "distant": {"core": "emotional detachment or coolness toward others",
                "cues": "turning away, flat responses, reduced engagement, keeping physical distance"},
    "confused": {"core": "inability to make sense of what is happening",
                 "cues": "frowning, head tilt, looking around, questioning gestures, rereading"},
    "startled": {"core": "abrupt reaction to a sudden unexpected stimulus",
                 "cues": "flinching, jumping, sudden eye widening, quick head turn"},
    "amazed": {"core": "wonder or awe at something impressive or extraordinary",
               "cues": "open mouth, wide eyes, staring, admiring gestures, pointing it out to others"},
    "excited": {"core": "high-energy positive anticipation or enthusiasm",
                "cues": "animated gestures, bouncing, rapid lively speech, eager approach"},
    "joyful": {"core": "pleasure at a good outcome or situation",
               "cues": "broad smile, laughter, open relaxed body, celebratory gestures"},
    "proud": {"core": "satisfaction with one's own achievement or that of someone close",
              "cues": "upright chest, raised chin, showing off a result, self-satisfied smile"},
    "optimistic": {"core": "confident positive expectation about the future",
                   "cues": "encouraging words, confident posture, leaning forward, resuming effort with energy"},
    "peaceful": {"core": "calm contentment with no pressing concern",
                 "cues": "relaxed posture, soft smile, slow unhurried movement, even breathing"},
    "loving": {"core": "affection and warmth toward a specific person",
               "cues": "warm gaze, hugging, gentle touch, closeness, caring gestures"},
    "hopeful": {"core": "anticipation that a desired but still uncertain outcome may happen",
                "cues": "attentive watching, expectant smile, leaning in, holding still in anticipation"},
    "avoidance": {"core": "disgust-driven urge to stay away from something unpleasant",
                  "cues": "turning away, keeping distance, refusing to touch"},
    "hesitant": {"core": "reluctance to proceed because something feels wrong or unappealing",
                 "cues": "pausing, halting movements, starting and stopping, backing off"},
    "aversion": {"core": "dislike or sensory rejection of a stimulus",
                 "cues": "grimacing, pulling back, pushing away, covering nose or ears"},
    "judgmental": {"core": "critical contempt toward a person; looking down on them",
                   "cues": "eye roll, smirk, crossed arms, dismissive glance"},
    "revolted": {"core": "strong physical disgust at something offensive",
                 "cues": "wrinkled nose, gagging, recoiling, spitting out"},
    "disapproval": {"core": "disagreement with or rejection of someone's conduct",
                    "cues": "head shake, frown, pursed lips, disapproving remarks"},
    "neutral": {"core": "no directional emotional state",
                "cues": "ordinary baseline expression and behavior"},
}

ACTION_DEFS = {
    "approach": {"core": "reduces distance to a person, object, or place in the world",
                 "cues": "stepping toward, leaning toward, reaching toward"},
    "avoidance": {"core": "keeps away from a person, object, or situation without leaving the scene, including keeping the gaze or body turned away from it",
                  "cues": "sidestepping, angling the body away, keeping the gaze away, not engaging with something"},
    "freeze": {"core": "sudden cessation of movement or interruption of an ongoing action",
               "cues": "stopping mid-action, holding still, stiffening"},
    "escape": {"core": "leaves or flees the situation",
               "cues": "walking away, running off, exiting the room"},
    "appeasement": {"core": "placates others to reduce tension or pressure directed at oneself",
                    "cues": "apologizing, bowing, submissive smile, conciliatory gestures"},
    "expression": {"core": "openly displays one's emotion",
                   "cues": "crying, laughing, shouting, animated expressive gestures"},
    "suppression": {"core": "visibly restrains or minimizes one's own emotional display",
                    "cues": "an emotional display that starts and is then restrained; a controlled, minimal display after a strong event; tight lips; stiff, controlled movement"},
    "self_calming": {"core": "regulates one's own arousal",
                     "cues": "deep breaths, self-touch, closing eyes, pausing, rubbing hands"},
    "engagement": {"core": "initiates or increases neutral or positive interaction with another person",
                   "cues": "turning toward, talking to, gesturing to, sharing with someone"},
    "withdrawal": {"core": "reduces involvement with other people or with an ongoing social situation",
                   "cues": "turning away from others, falling silent, breaking gaze, leaving the group or conversation area, sitting apart"},
    "seek_support": {"core": "turns to others for comfort or help",
                     "cues": "looking to someone for reassurance, reaching out, asking for help"},
    "provide_support": {"core": "comforts or helps another person",
                        "cues": "hugging, patting, consoling words, helping with a task"},
    "confrontation": {"core": "directly opposes a specific person",
                      "cues": "facing off, leaning in toward them, pointing, arguing, challenging"},
    "escalation": {"core": "increases the intensity of a conflict",
                   "cues": "louder voice, more forceful gestures, stepping in aggressively"},
    "de_escalation": {"core": "reduces the intensity of a conflict",
                      "cues": "softening tone, lowering the voice, backing down, open palms"},
    "shift": {"core": "redirects attention to a new person, object, or event in the world",
              "cues": "gaze moves to a new target, turning head or body toward it"},
    "seek_info": {"core": "actively looks for information",
                  "cues": "checking a phone or document, asking questions, inspecting closely"},
    "object_interaction": {"core": "meaningfully handles or manipulates an object",
                           "cues": "picking up, throwing, operating, examining, handing over an object"},
    "task_persistence": {"core": "continues or resumes a task after a setback",
                         "cues": "trying again, keeps working despite failure"},
    "disengagement": {"core": "stops or abandons a task or activity",
                      "cues": "putting tools down, stopping work or play, walking away from the task"},
}



def norm_key(raw):
    """'Seek Info' / 'seek-info' / 'SEEK_INFO' -> 'seek_info'"""
    if not isinstance(raw, str):
        return None
    v = re.sub(r"[\s\-/]+", "_", raw.strip().lower())
    v = re.sub(r"[^a-z_]", "", v)
    return re.sub(r"_+", "_", v).strip("_") or None


def _build(taxonomy):
    display, group = {}, {}
    for g, items in taxonomy.items():
        for it in items:
            display[norm_key(it)] = it
            group[norm_key(it)] = g
    return display, group


# 情绪与动作分开建表，"Avoidance" 在两边互不干扰
EMOTION_DISPLAY, EMOTION_GROUP = _build(EMOTION_FAMILIES)
ACTION_DISPLAY, ACTION_GROUP = _build(ACTION_TAXONOMY)
EMOTION_FAMILY_KEYS = {norm_key(f): f for f in EMOTION_FAMILIES}
ACTION_CATEGORY_KEYS = {norm_key(c): c for c in ACTION_TAXONOMY}

EMOTION_ALIAS = {
    "guilt": "guilty", "loneliness": "lonely", "boredom": "bored",
    "disappointment": "disappointed", "bad_feeling": "awful",
    "anxiety": "anxious", "unease": "insecure",
    "rejection": "rejected", "threat": "threatened",
    "alienation": "distant", "confusion": "confused",
    "amazement": "amazed", "excitement": "excited", "joy": "joyful",
    "pride": "proud", "optimism": "optimistic", "peacefulness": "peaceful",
    "love": "loving", "hope": "hopeful",
    "hesitation": "hesitant", "judgment": "judgmental",
}

ACTION_ALIAS = {
    "emotional_expression": "expression", "emotional_suppression": "suppression",
    "social_engagement": "engagement", "support_seeking": "seek_support",
    "support_providing": "provide_support", "attention_shift": "shift",
    "information_seeking": "seek_info", "deescalation": "de_escalation",
    "self_soothing": "self_calming",
}


def _normalize(raw, display, alias):
    k = norm_key(raw)
    if k is None:
        return None
    k = alias.get(k, k)
    return k if k in display else None


def normalize_emotion(raw):
    return _normalize(raw, EMOTION_DISPLAY, EMOTION_ALIAS)


def normalize_action(raw):
    return _normalize(raw, ACTION_DISPLAY, ACTION_ALIAS)


assert set(EMOTION_DEFS) == set(EMOTION_DISPLAY), "EMOTION_DEFS 与分类不一致"
assert set(ACTION_DEFS) == set(ACTION_DISPLAY), "ACTION_DEFS 与分类不一致"


# #####################################################################
# 4. 通用工具
# #####################################################################

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("task2")
for _n in ("httpx", "openai", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)

ANCHOR_RE = re.compile(r"\[T=(\d+)%\]")
# 证据中引用的锚点：兼容单点 [T=40%] 与区间 [T=40%-60%] / [T=40%–60%]
EVIDENCE_ANCHOR_RE = re.compile(r"\[T=(\d+)%(?:\s*[-–—~]\s*(?:T=)?(\d+)%)?\]")


def cited_anchors(text):
    out = []
    for a, b in EVIDENCE_ANCHOR_RE.findall(text):
        out.append(a)
        if b:
            out.append(b)
    return out


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def extract_json(text):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def fill(template, **kwargs):
    """单次扫描替换 {变量}：插入的内容不会被再次替换，JSON 花括号不受影响。"""
    return re.sub(r"\{(\w+)\}",
                  lambda m: str(kwargs[m.group(1)]) if m.group(1) in kwargs
                  else m.group(0),
                  template.strip())


def short_hash(*texts):
    return hashlib.md5("".join(texts).encode("utf-8")).hexdigest()[:12]


def is_fatal_error(e):
    code = (getattr(e, "status_code", None)
            or getattr(getattr(e, "response", None), "status_code", None))
    return code in (400, 401, 402, 403, 404)


def with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if is_fatal_error(e) or attempt == MAX_RETRIES - 1:
                raise
            wait = SLEEP_SEC * 2 ** attempt
            logger.warning(f"[RETRY] {fn.__name__} attempt={attempt + 1} "
                           f"error={e} wait={wait:.1f}s")
            time.sleep(wait)


def to_level(x, lo, hi):
    try:
        v = int(round(float(x)))
    except (TypeError, ValueError):
        return None
    return max(lo, min(hi, v))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def spread(xs):
    return max(xs) - min(xs) if xs else 0.0

def agg(xs):
    """多次评分的合并：AGGREGATE = "median" 时取中位数，否则取平均。"""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    if AGGREGATE == "mean":
        return sum(xs) / len(xs)
    s, n = sorted(xs), len(xs)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# #####################################################################
# 5. 启动自检
# #####################################################################

def check_config():
    errors = []
    keys = {"VLM_API_KEY": VLM_API_KEY, "LLM_API_KEY": LLM_API_KEY}
    if TRANSPORT == "oss":
        keys.update({"OSS_ACCESS_KEY_ID": OSS_ACCESS_KEY_ID,
                     "OSS_ACCESS_KEY_SECRET": OSS_ACCESS_KEY_SECRET})
    for name, val in keys.items():
        if not val or val.startswith("<YOUR_"):
            errors.append(f"未填写 {name}")

    g = globals()
    for name, required in PROMPT_SPECS.items():
        text = g[name]
        if not text.strip() or "<PASTE_" in text:
            errors.append(f"未粘贴 prompt：{name}")
            continue
        for var in required:
            if "{" + var + "}" not in text:
                errors.append(f"{name} 缺少模板变量 {{{var}}}")

    if TRANSPORT not in ("base64", "oss"):
        errors.append(f"TRANSPORT 只能是 base64 或 oss，当前为 {TRANSPORT!r}")
    if JUDGE_RUNS < 1:
        errors.append("JUDGE_RUNS 至少为 1")
    if not Path(DATA_PATH).exists():
        errors.append(f"找不到测试集：{DATA_PATH}")

    if errors:
        raise SystemExit("配置检查未通过：\n  - " + "\n  - ".join(errors))


# #####################################################################
# 6. 样本解析与视频读取
# #####################################################################

def entity_lines(raw):
    """标注中的目标主体与实体定义，写入评分 prompt 的 INPUT 段。"""
    tr = raw.get("transition") or {}
    ta, ents = tr.get("target_agent") or {}, tr.get("entities") or {}
    lines = []
    if ta:
        lines.append(f"Main subject (judge only this person): "
                     f"{ta.get('id')} = {ta.get('description')}")
    if ents:
        lines.append("Entity identifiers: " +
                     "; ".join(f"{k} = {v}" for k, v in ents.items()))
    return lines


def parse_sample(raw):
    warnings = []
    tr = raw.get("transition", {})
    ref = raw.get("reference_state1") or raw.get("state1") or {}

    target = normalize_emotion(ref.get("emotion"))
    if target is None:
        warnings.append(f"unknown target emotion {ref.get('emotion')!r}")

    # 候选情绪 = 目标情绪 ∪ plausible_emotions
    emotions = []
    for lab in [ref.get("emotion")] + list(tr.get("plausible_emotions") or []):
        e = normalize_emotion(lab)
        if e is None:
            warnings.append(f"dropped unknown emotion {lab!r}")
        elif e not in emotions:
            emotions.append(e)

    actions = []
    for lab in tr.get("plausible_actions") or []:
        a = normalize_action(lab)
        if a is None:
            warnings.append(f"dropped unknown action {lab!r}")
        elif a not in actions:
            actions.append(a)

    hide = raw.get("_hide_entities", False)   # 错配对照：视频换了人，不给实体定义
    if not tr.get("target_agent") and not hide:
        warnings.append("no target_agent annotated; main subject inferred by the judge")

    meta = raw.get("meta", {})
    return {
        "video_key": raw["video_key"],
        "video_file": raw.get("_video_file", raw["video_key"]),
        "scene_category": meta.get("scene_category", ""),
        "scene_mode": meta.get("scene_mode", ""),
        "context": tr.get("context", ""),
        "event": tr.get("event"),
        "entity_lines": [] if hide else entity_lines(raw),
        "target_emotion": target,
        "emotion_candidates": emotions,
        "action_candidates": actions,
    }, warnings




def video_to_url(path):
    """直接读取已处理好的视频，不做切片。"""
    if TRANSPORT == "oss":
        import oss2
        auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
        key = f"{OSS_PREFIX}/{path.name}"
        bucket.put_object_from_file(key, str(path))
        return bucket.sign_url("GET", key, 3600, slash_safe=True)

    b64 = base64.b64encode(path.read_bytes()).decode()
    if len(b64) > MAX_BASE64_BYTES:
        raise ValueError(f"{path.name} base64 后 {len(b64) / 1e6:.1f}MB，"
                         f"超过上限，请把 TRANSPORT 改为 \"oss\"")
    return f"data:video/mp4;base64,{b64}"


def video_cache_key(path):
    st = path.stat()
    return (f"{path.name}|{st.st_size}|{int(st.st_mtime)}|{VLM_MODEL}|"
            f"{short_hash(DESCRIBE_PROMPT)}")


# #####################################################################
# 7. Stage 1：VLM 描述
# #####################################################################

def describe_video(client, url):
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": url}},
            {"type": "text", "text": DESCRIBE_PROMPT.strip()},
        ]}])
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty description")
    if not text.startswith("[T=0%]"):
        logger.warning("[DESCRIBE] description does not start with [T=0%]")
    return text


def get_description(sample, clients, caches):
    video_path = Path(VIDEO_DIR) / sample["video_file"]
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    dkey = video_cache_key(video_path)
    description = caches["desc"].get(dkey)
    if description is None:
        url = video_to_url(video_path)
        description = with_retry(describe_video, clients["vlm"], url)
        caches["desc"][dkey] = description
        save_json(caches["desc_path"], caches["desc"])
    return description


# #####################################################################
# 8. Stage 2：LLM 评分——prompt 组装
# #####################################################################

def build_input_block(sample, description):
    lines = ["INPUT",
             f"Scene: {sample['scene_category']} ({sample['scene_mode']})",
             f"Context (before the video): {sample['context']}",
             f"Event (before or at the start of the video): "
             f"{sample['event'] or 'No distinct event annotated.'}",
             *sample.get("entity_lines", [])]
    lines += ["", "Video description (the only source of observations):",
              '"""', description, '"""']
    return "\n".join(lines)



def emotion_taxonomy_block():
    lines = []
    for fam, items in EMOTION_FAMILIES.items():
        lines.append(f"{fam}: {EMOTION_FAMILY_DEFS[fam]}")
        for it in items:
            lines.append(f"    - {it}: {EMOTION_DEFS[norm_key(it)]['core']}")
    return "\n".join(lines)


def action_taxonomy_block():
    lines = []
    for cat, items in ACTION_TAXONOMY.items():
        lines.append(f"{cat}: {ACTION_CATEGORY_DEFS[cat]}")
        for it in items:
            lines.append(f"    - {it}: {ACTION_DEFS[norm_key(it)]['core']}")
    return "\n".join(lines)


def outside_label_list(candidates):
    """候选集之外的情绪（含释义），供区分性检查使用。"""
    return "\n".join(f"- {EMOTION_DISPLAY[k]}: {EMOTION_DEFS[k]['core']}"
                     for k in EMOTION_DISPLAY
                     if k not in candidates and k != "neutral")

def contrast_set(e, candidates):
    """e 的对照集：同大类中不在候选里的兄弟情绪；若为空，退回全部非候选情绪。"""
    sib = [k for k in EMOTION_DISPLAY
           if EMOTION_GROUP[k] == EMOTION_GROUP[e] and k not in candidates
           and k != "neutral"]
    return sib or [k for k in EMOTION_DISPLAY
                   if k not in candidates and k != "neutral"]


def emotion_candidate_block(order, candidates):
    return "\n".join(
        f"- {EMOTION_DISPLAY[k]}\n"
        f"    core: {EMOTION_DEFS[k]['core']}\n"
        f"    typical cues (examples only, not requirements): {EMOTION_DEFS[k]['cues']}\n"
        f"    contrast set (choose closest_alternative ONLY from these): "
        + "; ".join(f"{EMOTION_DISPLAY[c]} ({EMOTION_DEFS[c]['core']})"
                    for c in contrast_set(k, candidates))
        for k in order)


def candidate_block(keys, display, defs):
    return "\n".join(
        f"- {display[k]}\n"
        f"    core: {defs[k]['core']}\n"
        f"    typical cues (examples only, not requirements): {defs[k]['cues']}"
        for k in keys)


def presented_order(keys, seed_text):
    order = list(keys)
    if SHUFFLE_CANDIDATES:
        random.Random(seed_text).shuffle(order)
    return order


# #####################################################################
# 9. Stage 2：LLM 评分——调用与缓存
# #####################################################################

def judge_temperature():
    return JUDGE_TEMPERATURE if JUDGE_RUNS > 1 else 0.0


def call_llm_json(client, user_prompt, temperature):
    kwargs = dict(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM.strip()},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature, max_tokens=LLM_MAX_TOKENS)
    if USE_JSON_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    out = extract_json(resp.choices[0].message.content)
    if not isinstance(out, dict):
        raise ValueError("judge json parse failed")
    return out


def cached_judge(clients, caches, stage, sample, prompt, run):
    temp = judge_temperature()
    key = (f"{stage}|{sample['video_key']}|{LLM_MODEL}|"
           f"{short_hash(JUDGE_SYSTEM, prompt)}|t{temp}|run{run}")
    out = caches["judge"].get(key)
    if out is None:
        out = with_retry(call_llm_json, clients["llm"], prompt, temp)
        caches["judge"][key] = out
        save_json(caches["judge_path"], caches["judge"])
    return out


# #####################################################################
# 10. Stage 2：LLM 评分——结果校验（代码侧防幻觉兜底）
# #####################################################################

def valid_evidence_anchors(items, anchors):
    """返回有效证据数与其引用的锚点集合。有效：非空，且引用的锚点（含区间两端）都真实存在于描述中。"""
    if not isinstance(items, list):
        return 0, set()
    n, used = 0, set()
    for it in items:
        if not isinstance(it, str) or not it.strip():
            continue
        found = cited_anchors(it)
        if not anchors:                         # 描述无锚点时退化为非空检查
            n += 1
        elif found and all(a in anchors for a in found):
            n += 1
            used.update(found)
    return n, used


def index_entries(entries, candidates, normalize_fn, warnings, stage):
    out = {}
    if not isinstance(entries, list):
        warnings.append(f"{stage}: 'candidates' is not a list")
        return out
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        k = normalize_fn(ent.get("label"))
        if k in candidates and k not in out:
            out[k] = ent
        elif k not in candidates:
            warnings.append(f"{stage}: ignored non-candidate label "
                            f"{ent.get('label')!r}")
    return out


def score_emotion_output(out, candidates, anchors, warnings):
    visible = out.get("subject_visible") is not False
    entries = index_entries(out.get("candidates"), candidates,
                            normalize_emotion, warnings, "emotion")
    scores, details = {}, {}
    for e in candidates:
        ent, level, n_valid, dist, alt = entries.get(e), 0, 0, None, None
        if ent is None:
            warnings.append(f"emotion: missing candidate {e}, level=0")
        else:
            level = to_level(ent.get("level"), 0, 4)
            if level is None:
                warnings.append(f"emotion: invalid level for {e}, level=0")
                level = 0
            n_valid, used = valid_evidence_anchors(ent.get("evidence"), anchors)
            dist = ent.get("distinguished")
            alt = ent.get("closest_alternative")
            alt_inside = normalize_emotion(alt) in candidates  # 替代项在可接受集合内：区分判断无意义

            if not visible and level > 0:
                warnings.append(f"emotion: {e} capped to 0 (subject not visible)")
                level = 0
            elif level >= 2 and n_valid == 0:
                warnings.append(f"emotion: {e} capped {level}->1 (no valid anchored evidence)")
                level = 1
            else:
                if level >= 3 and dist is not True:
                    if alt_inside:
                        warnings.append(f"emotion: {e} closest_alternative {alt!r} "
                                        f"is a candidate; discrimination flag ignored")
                    else:
                        warnings.append(f"emotion: {e} capped {level}->2 "
                                        f"(not distinguished from {alt!r})")
                        level = 2
                if level == 4 and (n_valid < 2 or (anchors and len(used) < 2)):
                    warnings.append(f"emotion: {e} capped 4->3 "
                                    f"(fewer than two anchored evidence items)")
                    level = 3
        scores[e] = EMOTION_LEVEL_SCORE[level]
        details[e] = {"level": level, "valid_evidence": n_valid,
                      "distinguished": dist, "closest_alternative": alt}
    return scores, details



def score_action_output(out, candidates, anchors, warnings):
    visible = out.get("subject_visible") is not False
    entries = index_entries(out.get("candidates"), candidates,
                            normalize_action, warnings, "action")
    scores, details = {}, {}
    for a in candidates:
        ent, real, coh, n_valid, fn_ok = entries.get(a), 0, 0, 0, None
        if ent is None:
            warnings.append(f"action: missing candidate {a}, score=0")
        else:
            real = to_level(ent.get("realization"), 0, 4)
            coh = to_level(ent.get("coherence"), 0, 2)
            if real is None or coh is None:
                warnings.append(f"action: invalid level for {a}, treated as 0")
                real, coh = real or 0, coh or 0
            n_valid, used = valid_evidence_anchors(ent.get("evidence"), anchors)
            fn_ok = ent.get("function_established")
            if not visible and real > 0:
                warnings.append(f"action: {a} capped to 0 (subject not visible)")
                real = 0
            elif real >= 2 and n_valid == 0:
                warnings.append(f"action: {a} capped {real}->1 (no valid anchored evidence)")
                real = 1
            elif real >= 3 and fn_ok is not True:
                warnings.append(f"action: {a} capped {real}->2 (function not established)")
                real = 2
            elif real == 4 and anchors and len(used) < 2:
                warnings.append(f"action: {a} capped 4->3 (fewer than two anchored evidence items)")
                real = 3
            if real == 0:
                coh = 0
        scores[a] = REALIZATION_SCORE[real] * COHERENCE_WEIGHT[coh]
        details[a] = {"realization": real, "coherence": coh,
                      "valid_evidence": n_valid, "function_established": fn_ok}
    return scores, details



def parse_open_set(out, sample):
    emo = out.get("emotion") if isinstance(out.get("emotion"), dict) else {}
    act = out.get("action") if isinstance(out.get("action"), dict) else {}

    fam = EMOTION_FAMILY_KEYS.get(norm_key(emo.get("family")))
    elab = normalize_emotion(emo.get("label"))
    cat = ACTION_CATEGORY_KEYS.get(norm_key(act.get("category")))
    alab = normalize_action(act.get("label"))

    emo_ok = bool(fam and elab and EMOTION_GROUP[elab] == fam)
    act_ok = bool(cat and alab and ACTION_GROUP[alab] == cat)

    open_fam = EMOTION_GROUP.get(elab) if elab else fam
    open_cat = ACTION_GROUP.get(alab) if alab else cat
    cand_fams = {EMOTION_GROUP[e] for e in sample["emotion_candidates"]}
    cand_cats = {ACTION_GROUP[a] for a in sample["action_candidates"]}

    return {
        "open_emotion_family_chosen": fam,
        "open_emotion": elab,
        "open_emotion_hierarchy_ok": emo_ok,
        "open_emotion_in_candidates": elab in sample["emotion_candidates"],
        "open_emotion_family_in_candidates": open_fam in cand_fams,
        "open_action_category_chosen": cat,
        "open_action": alab,
        "open_action_hierarchy_ok": act_ok,
        "open_action_in_candidates": alab in sample["action_candidates"],
        "open_action_category_in_candidates": open_cat in cand_cats,
    }


def build_qc_flags(record):
    """
    低分与候选外标记。
    - 评测参考视频时：用于标注质检（参考视频本应高分）。
    - 评测生成视频时：低分代表模型表现差，不代表标注有问题。
    """
    d, flags = record["diagnostics"], []
    if record["M_emo"] < QC_LOW_SCORE:
        flags.append(f"low_M_emo={record['M_emo']:.2f}")
    if record["S_action"] < QC_LOW_SCORE:
        flags.append(f"low_S_action={record['S_action']:.2f}")
    ts = d.get("target_score")
    if ts is not None and ts < QC_LOW_SCORE:
        flags.append(f"low_target_score={ts:.2f}")
    if d.get("open_emotion") and not d.get("open_emotion_in_candidates"):
        flags.append(f"open_emotion_not_in_candidates={d['open_emotion']}")
    if d.get("open_action") and not d.get("open_action_in_candidates"):
        flags.append(f"open_action_not_in_candidates={d['open_action']}")
    return flags


# #####################################################################
# 11. 单样本评测
# #####################################################################

def run_emotion_judging(sample, input_block, anchors, clients, caches, warnings):
    cands = sample["emotion_candidates"]
    runs = []
    for run in range(JUDGE_RUNS):
        order = presented_order(cands, f"{sample['video_key']}|emo|{run}")
        prompt = fill(EMOTION_SCORE_PROMPT, input_block=input_block,
                      grounding_rules=GROUNDING_RULES.strip(),
                      candidate_block=candidate_block(order, EMOTION_DISPLAY,
                                                      EMOTION_DEFS),
                      outside_label_list=outside_label_list(cands))
        out = cached_judge(clients, caches, "emotion", sample, prompt, run)
        scores, details = score_emotion_output(out, cands, anchors, warnings)
        runs.append({"order": order, "scores": scores,
                     "details": details, "raw": out})
    return runs




def run_action_judging(sample, e_star, input_block, anchors, clients, caches,
                       warnings):
    cands = sample["action_candidates"]
    runs = []
    for run in range(JUDGE_RUNS):
        order = presented_order(cands, f"{sample['video_key']}|act|{run}")
        prompt = fill(ACTION_SCORE_PROMPT, input_block=input_block,
                      grounding_rules=GROUNDING_RULES.strip(),
                      candidate_block=candidate_block(order, ACTION_DISPLAY,
                                                      ACTION_DEFS),
                      emotion=EMOTION_DISPLAY[e_star],
                      emotion_core=EMOTION_DEFS[e_star]["core"],
                      emotion_cues=EMOTION_DEFS[e_star]["cues"])
        out = cached_judge(clients, caches, f"action|{e_star}", sample,
                           prompt, run)
        scores, details = score_action_output(out, cands, anchors, warnings)
        runs.append({"order": order, "scores": scores,
                     "details": details, "raw": out})
    return runs



def evaluate_sample(raw, clients, caches):
    sample, warnings = parse_sample(raw)
    if not sample["emotion_candidates"] or not sample["action_candidates"]:
        raise ValueError(f"no valid candidates; warnings={warnings}")

    description = get_description(sample, clients, caches)
    anchors = set(ANCHOR_RE.findall(description))
    if not anchors:
        warnings.append("description has no [T=XX%] anchors; "
                        "anchor check degraded to non-empty check")
    input_block = build_input_block(sample, description)
    judge_raw = {}

    # ---- A. 开放集两步分类（诊断，不计分）----
    open_diag = {}
    if RUN_OPEN_SET:
        prompt = fill(OPEN_SET_PROMPT, input_block=input_block,
                      grounding_rules=GROUNDING_RULES.strip(),
                      emotion_taxonomy_block=emotion_taxonomy_block(),
                      action_taxonomy_block=action_taxonomy_block())
        out = cached_judge(clients, caches, "open", sample, prompt, 0)
        judge_raw["open_set"] = out
        open_diag = parse_open_set(out, sample)

    # ---- B. 情绪评分 → M_emo, e* ----
    emo_cands = sample["emotion_candidates"]
    emo_runs = run_emotion_judging(sample, input_block, anchors, clients,
                                   caches, warnings)
    emo_scores = {e: agg([r["scores"][e] for r in emo_runs]) for e in emo_cands}
    e_star = max(emo_cands, key=lambda e: emo_scores[e])   # 平分时目标情绪优先
    m_emo = emo_scores[e_star]

    # ---- C. 动作评分（以 e* 为条件）→ S_action ----
    act_cands = sample["action_candidates"]
    act_runs = run_action_judging(sample, e_star, input_block, anchors,
                                  clients, caches, warnings)
    act_scores = {a: agg([r["scores"][a] for r in act_runs]) for a in act_cands}
    a_star = max(act_cands, key=lambda a: act_scores[a])
    s_action = act_scores[a_star]

    judge_raw["emotion_runs"] = emo_runs
    judge_raw["action_runs"] = act_runs

    # 评分波动：各候选在多次评分间的最大分差
    instability = max(
        [spread([r["scores"][e] for r in emo_runs]) for e in emo_cands]
        + [spread([r["scores"][a] for r in act_runs]) for a in act_cands])

    target = sample["target_emotion"]
    top = max(emo_scores.values())
    n_at_max = sum(1 for v in emo_scores.values() if v == top)
    wo_expr = [act_scores[a] for a in act_cands if a != "expression"]
    diagnostics = {
        "e_star_is_target": None if n_at_max > 1 else e_star == target,
        "e_star_family_match": (target is not None and
                                EMOTION_GROUP.get(e_star) == EMOTION_GROUP.get(target)),
        "target_score": emo_scores.get(target),
        "n_emotion_at_max": n_at_max,
        # 多个候选并列最高分时，e* 由平分规则决定，e_star_is_target 没有诊断意义
        "emotion_tie": n_at_max > 1,
        "judge_instability": instability,
        # 去掉 Expression 后的动作分：Expression 仅凭面部即可实现，单列以便观察
        "S_action_wo_expression": max(wo_expr) if wo_expr else None,
        **open_diag,
    }


    record = {
        "video_key": sample["video_key"],
        "video_file": sample["video_file"],
        "scene_category": sample["scene_category"],
        "scene_mode": sample["scene_mode"],
        "event": sample["event"],
        "target_emotion": target,
        "emotion_candidates": emo_cands,
        "action_candidates": act_cands,
        "emotion_scores": emo_scores,
        "e_star": e_star,
        "M_emo": m_emo,
        "action_scores": act_scores,
        "a_star": a_star,
        "S_action": s_action,
        "Score_T2": (m_emo + s_action) / 2,
        "diagnostics": diagnostics,
        "description": description,
        "judge": judge_raw,
        "warnings": list(dict.fromkeys(warnings)),
    }
    record["qc_flags"] = build_qc_flags(record)
    return record



# #####################################################################
# 12. 汇总
# #####################################################################

DIAG_KEYS = ("e_star_is_target", "e_star_family_match", "target_score",
             "n_emotion_at_max", "emotion_tie", "judge_instability",
             "S_action_wo_expression",
             "open_emotion_hierarchy_ok", "open_emotion_in_candidates",
             "open_emotion_family_in_candidates",
             "open_action_hierarchy_ok", "open_action_in_candidates",
             "open_action_category_in_candidates")


def cap_reason(w):
    m = re.search(r"\((.*?)\)", w)
    kind = "emotion" if w.startswith("emotion") else "action"
    return f"{kind}: {m.group(1) if m else w}"


def group_mean(ok, key_fn, score_key="Score_T2"):
    groups = defaultdict(list)
    for r in ok:
        groups[key_fn(r)].append(r[score_key])
    return {g: {"n": len(v), score_key: mean(v)} for g, v in sorted(groups.items())}


def summarize(results):
    ok = [r for r in results if "error" not in r]
    summary = {"n": len(ok), "n_failed": len(results) - len(ok),
               "vlm_model": VLM_MODEL, "llm_model": LLM_MODEL,
               "judge_runs": JUDGE_RUNS}
    if not ok:
        return summary

    for k in ("M_emo", "S_action", "Score_T2"):
        summary[k] = mean([r[k] for r in ok])
    for k in DIAG_KEYS:
        vals = [float(r["diagnostics"][k]) for r in ok
                if r["diagnostics"].get(k) is not None]
        if vals:
            summary[k] = mean(vals)

    # 分组报告：用于发现系统性偏差（某类场景 / 情绪大类整体偏严或偏松）
    summary["by_scene"] = group_mean(ok, lambda r: r["scene_category"])
    summary["by_emotion_family"] = group_mean(
        ok, lambda r: EMOTION_GROUP.get(r["target_emotion"], "UNKNOWN"))

    caps = Counter(cap_reason(w) for r in ok for w in r["warnings"]
                   if "capped" in w)
    summary["cap_stats"] = dict(caps.most_common())
    summary["flagged"] = {r["video_key"]: r["qc_flags"]
                          for r in ok if r.get("qc_flags")}
    return summary


def print_summary(s):
    print("\n" + "=" * 60)
    print(f"Task 2 | VLM={VLM_MODEL} | LLM={LLM_MODEL} | runs={JUDGE_RUNS}")
    print(f"Evaluated: {s['n']} | Failed: {s['n_failed']}")
    if not s["n"]:
        return
    print(f"M_emo    : {s['M_emo']:.4f}")
    print(f"S_action : {s['S_action']:.4f}")
    print(f"Score_T2 : {s['Score_T2']:.4f}")
    print("\nDiagnostics (not part of the score):")
    for k in DIAG_KEYS:
        if k in s:
            print(f"  {k}: {s[k]:.3f}")
    for title, key in (("By scene category", "by_scene"),
                       ("By target emotion family", "by_emotion_family")):
        print(f"\n{title}:")
        for g, v in s[key].items():
            print(f"  {g}: n={v['n']} Score_T2={v['Score_T2']:.4f}")
    if s.get("cap_stats"):
        print("\nCode-side level caps:")
        for k, v in s["cap_stats"].items():
            print(f"  {k}: {v}")
    if s.get("flagged"):
        print("\nFlagged samples (reference videos: annotation QC; "
              "generated videos: low-score cases):")
        for k, v in s["flagged"].items():
            print(f"  {k}: {'; '.join(v)}")

REASON_CHARS = 260


def _short(s, n=REASON_CHARS):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _rep_entry(runs, key, agg_score, norm):
    """取分数最接近合并结果的那次评分，返回其中该候选的原始条目。"""
    run = min(runs, key=lambda r: abs((r["scores"].get(key) or 0) - (agg_score or 0)))
    for ent in run["raw"].get("candidates") or []:
        if isinstance(ent, dict) and norm(ent.get("label")) == key:
            return ent
    return {}


def review_record(r):
    if "error" in r:
        return {"video_key": r.get("video_key"), "error": r["error"]}
    emo_runs, act_runs = r["judge"]["emotion_runs"], r["judge"]["action_runs"]
    d = r["diagnostics"]
    emotions = {}
    for e in r["emotion_candidates"]:
        ent = _rep_entry(emo_runs, e, r["emotion_scores"][e], normalize_emotion)
        emotions[e] = {
            "score": r["emotion_scores"][e],
            "levels": [run["details"][e]["level"] for run in emo_runs],
            "alt": [run["details"][e]["closest_alternative"] for run in emo_runs],
            "reason": _short(ent.get("core_analysis")),
        }
    actions = {}
    for a in r["action_candidates"]:
        ent = _rep_entry(act_runs, a, r["action_scores"][a], normalize_action)
        actions[a] = {
            "score": r["action_scores"][a],
            "real/coh": [f"{run['details'][a]['realization']}/{run['details'][a]['coherence']}"
                         for run in act_runs],
            "function": _short(ent.get("function_analysis")),
            "manner": _short(ent.get("coherence_reason")),
        }
    visible = [run["raw"].get("subject_visible") for run in emo_runs + act_runs]
    return {
        "video_key": r["video_key"],
        "video_file": r["video_file"],
        "event": r.get("event"),
        "target": r["target_emotion"],
        "e_star": r["e_star"], "M_emo": r["M_emo"],
        "a_star": r["a_star"], "S_action": r["S_action"],
        "Score_T2": r["Score_T2"],
        "subject_not_visible_runs": sum(v is False for v in visible),
        "open_set": f"{d.get('open_emotion')} / {d.get('open_action')}",
        "emotion_tie": d.get("emotion_tie"),
        "emotions": emotions,
        "actions": actions,
        "caps": [w for w in r["warnings"] if "capped" in w],
        "description": r["description"],
    }


def write_review(results, summary, path):
    keep = ("n", "n_failed", "M_emo", "S_action", "Score_T2", *DIAG_KEYS, "cap_stats")
    save_json(path, {
        "control_mode": CONTROL_MODE,
        "judge_runs": JUDGE_RUNS,
        "summary": {k: summary[k] for k in keep if k in summary},
        "samples": [review_record(r) for r in results],
    })


# #####################################################################
# 13. 主流程
# #####################################################################

# ---- 对照实验（已停用，评分器合理性已验证；如需重新验证可取消注释）----
# 负对照：为每个样本换一个目标情绪大类不在其候选大类中的视频
def assign_mismatch(samples):
    """负对照：为每个样本换一个"目标情绪大类不在本样本候选大类中"的视频，并隐藏实体定义。"""
    info = [parse_sample(raw)[0] for raw in samples]
    n = len(samples)
    for i, (raw, s) in enumerate(zip(samples, info)):
        cand_fams = {EMOTION_GROUP[e] for e in s["emotion_candidates"]}
        pick = next((j for j in range(n) if j != i and info[j]["target_emotion"]
                     and EMOTION_GROUP[info[j]["target_emotion"]] not in cand_fams), None)
        if pick is None:
            pick = (i + 1) % n
            print(f"  [mismatch] {raw['video_key']}: no video outside candidate "
                  f"families, fallback to neighbour")
        raw["_video_file"] = samples[pick]["video_key"]
        raw["_hide_entities"] = True
        print(f"  [mismatch] {raw['video_key']}  <-  {raw['_video_file']}")


def tagged(path):
    """对照组结果写入独立文件，不覆盖正常组。"""
    if CONTROL_MODE == "none":
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}.{CONTROL_MODE}{p.suffix}"))



def main():
    global CONTROL_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["none", "mismatch"], default="none",
                    help="none：正常评测；mismatch：错配负对照")
    ap.add_argument("--fresh", action="store_true",
                    help="删除本组旧结果后重评（缓存保留）")
    ap.add_argument("--review-only", action="store_true",
                    help="不调用 API，只根据已有结果重新生成评审文件")
    args = ap.parse_args()
    CONTROL_MODE = args.control
    out_path, summary_path, review_path = (tagged(OUT_PATH), tagged(SUMMARY_PATH),
                                           tagged(REVIEW_PATH))

    if args.review_only:
        results = load_json(out_path, [])
        if not results:
            raise SystemExit(f"找不到结果文件：{out_path}")
        write_review(results, summarize(results), review_path)
        print(f"Review file: {review_path}")
        return

    check_config()
    if args.fresh:
        for p in (out_path, summary_path, review_path):
            if Path(p).exists():
                Path(p).unlink()
                print(f"removed {p}")

    clients = {
        "vlm": OpenAI(api_key=VLM_API_KEY, base_url=VLM_BASE_URL),
        "llm": OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL),
    }
    cache_dir = Path(CACHE_DIR)
    caches = {"desc_path": cache_dir / "descriptions.json",
              "judge_path": cache_dir / "judge.json"}
    caches["desc"] = load_json(caches["desc_path"], {})
    caches["judge"] = load_json(caches["judge_path"], {})

    samples = load_json(DATA_PATH, [])
    if LIMIT:
        samples = samples[:LIMIT]
    if CONTROL_MODE == "mismatch":
        if len(samples) < 2:
            raise SystemExit("mismatch 模式至少需要 2 个样本")
        assign_mismatch(samples)

    # 断点续跑：保留成功记录，失败记录重跑
    results = [r for r in load_json(out_path, []) if "error" not in r]
    done = {r["video_key"] for r in results}
    pending = [s for s in samples if s["video_key"] not in done]
    print(f"Control: {CONTROL_MODE} | Total: {len(samples)} | Done: {len(done)} | "
          f"Pending: {len(pending)}")
    print(f"Output: {out_path}")

    for raw in tqdm(pending, desc=f"Task 2 ({CONTROL_MODE})"):
        try:
            record = evaluate_sample(raw, clients, caches)
            tqdm.write(f" ✓ {raw['video_key']} (video={record['video_file']})  "
                       f"e*={record['e_star']} a*={record['a_star']}  "
                       f"M_emo={record['M_emo']:.2f} "
                       f"S_action={record['S_action']:.2f}  "
                       f"T2={record['Score_T2']:.3f}")
        except Exception as e:
            logger.exception(f"[FAILED] {raw.get('video_key')}")
            tqdm.write(f" ✗ {raw.get('video_key')}: {e}")
            record = {"video_key": raw.get("video_key"), "error": str(e)}
        results.append(record)
        save_json(out_path, results)
        time.sleep(SLEEP_SEC)

    summary = summarize(results)
    summary["control_mode"] = CONTROL_MODE
    save_json(summary_path, summary)
    write_review(results, summary, review_path)
    print_summary(summary)
    print(f"\nReview file (paste this one): {review_path}")



if __name__ == "__main__":
    main()
