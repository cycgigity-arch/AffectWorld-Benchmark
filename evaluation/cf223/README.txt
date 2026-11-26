CF protocol: exact runtime logic of user-supplied eval_cf (1)(3).py
Original emotion = transition.plausible_emotions[0].
CF emotion = counterfactual.emotion_cf. Equal emotions cause a warning, not exclusion.
R = S_action * S_causal; S_counter = sqrt(R_oo*R_cc) * (sel_orig+sel_cf)/2. RUN_DIFF=False; SAME_SETTING_GATE=False.
The source docstring's D*R_orig*R_cf formula is stale; the code above is executed.
Task2/3 source files are copied from the previous bundle unchanged.
This bundle is for new Task3/CF runs; keep running Task2 workers on the old bundle.
1. Upload ZIP to /root/autodl-tmp and extract into /root/autodl-tmp/ewm_eval_local_cf223.
2. Run: python3 continue_task3_cf223_after_task2.py --check-only
3. Stop the old continue_task3_cf_after_task2.py scheduler only; leave Task2 workers.
4. Export EWM_VLM_API_KEY and EWM_LLM_API_KEY in the launching shell.
5. Run the new scheduler with nohup, then tail its log. It uses 20 shards per model for Task3 and CF.
If Task2 has permanent failed rows, use --skip-task2-wait to run Task3 and CF now. Stage failures are reported separately.
When upgrading from the 10-shard scheduler, run stop_cf223_10_shards.py first. Completed score files are reused by video_key.
Use the adapter ewm_eval_local.py with a fresh output directory per run.
