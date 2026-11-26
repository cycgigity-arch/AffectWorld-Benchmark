Task1 remains separate. This bundle runs Task2/3/CF on local generated videos.
CF PROTOCOL: state0_emotion_family_v1. Original emotion = state0.emotion (broad family).
CF emotion = counterfactual.emotion_cf, evaluated at the same family granularity.
The two families must differ. Old CF scores are incompatible and must stay in their old output directories.
The supplied legacy 223 CF records were selected against a different factual-emotion definition.
Do not treat their CF scores as final until annotations and affected videos are reviewed/regenerated.
Install: python -m pip install -r requirements.txt
Offline check (does not call APIs):
 python ewm_eval_local.py check --orig-dir /path/to/generated/orig --cf-dir /path/to/generated/cf --tasks cf --limit 0
Set your own EWM_VLM_API_KEY and EWM_LLM_API_KEY environment variables.
Optional: EWM_VLM_BASE_URL, EWM_LLM_BASE_URL, EWM_VLM_MODEL, EWM_LLM_MODEL.
CF run requires a clean preflight and matching generation prompts/video provenance.
Full preflight: use --limit 0. Same-family pairs require reannotation and video regeneration.
CF emotion overrides are refused because relabeling does not change an already-generated video.
The example override applies only to Task2's post-event reference label, not CF state0.emotion.
Only a unique normalization mapping recorded for the same label in that sample is applied automatically.
Task2 and Task3 original scoring code is unchanged. CF uses a new family-level action rubric,
the original cross-matrix formula, and the shared-setting gate. Transport is local base64.
Outputs live under eval_runs/<timestamp>: preflight.json, eval_inputs.json, task*_results.json,
cf_results.json, summaries, reviews, and ewh_report.json. Every run uses a fresh output directory.
Different models must be evaluated against their own paired generated video folders.
The shipped original run_all.py still has its original configuration; use ewm_eval_local.py.
