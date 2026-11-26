# Hunyuan Task2, then CF223

Upload `hunyuan_task2_cf_sequential_20261012.zip` to `/root/autodl-tmp/`.
It contains two Python files and this README. Extract into the existing
CF223 evaluator directory:

```bash
unzip -oq /root/autodl-tmp/hunyuan_task2_cf_sequential_20261012.zip -d /root/autodl-tmp/ewm_eval_local_cf223
python3 /root/autodl-tmp/ewm_eval_local_cf223/run_hunyuan_task2_then_cf.py --check-only
```

The check prints the Hunyuan Task2 cohort found among the 1451 protocol rows,
the number of generated Hunyuan CF videos, and the number with Hunyuan
originals. It makes no API calls. The generated CF directory is expected to
contain 30 MP4 files; use `--expected-cf-files N` if that count has changed.

In the new terminal, enter the two API keys. If both endpoints use the same
key, the same value can be entered twice:

```bash
read -rsp 'VLM API key: ' EWM_VLM_API_KEY; echo
read -rsp 'LLM API key: ' EWM_LLM_API_KEY; echo
export EWM_VLM_API_KEY EWM_LLM_API_KEY
```

Start the single sequential scheduler:

```bash
mkdir -p /root/autodl-tmp/ewm_eval_local/logs
nohup python3 -u /root/autodl-tmp/ewm_eval_local_cf223/run_hunyuan_task2_then_cf.py \
  --task2-parallel 4 --cf-parallel 4 \
  > /root/autodl-tmp/ewm_eval_local/logs/hunyuan_task2_then_cf.log 2>&1 < /dev/null &
tail -f /root/autodl-tmp/ewm_eval_local/logs/hunyuan_task2_then_cf.log
```

Task2 is split into 20 shards, and CF is split into 20 shards. Four concurrent
workers per stage keep extra API load modest while the existing four-model CF
run continues. Both stages can be configured for up to 20 concurrent workers.

Task2 successful scores are reused by `video_key` on restart. The scheduler
retries pending samples up to three times. With at most 10 remaining Task2
failures it proceeds to CF and reports those failures. If more than 10 remain,
it stops before CF so a broad API outage does not waste CF calls. Change this
limit with `--max-task2-failures N` if needed. CF scores are also reusable.

Task2 results: `/root/autodl-tmp/ewm_eval_local/eval_runs/hunyuan_task2_20_*`.
CF results: `/root/autodl-tmp/ewm_eval_local/eval_runs/hunyuan_cf223_20_*`.
Worker logs: `/root/autodl-tmp/ewm_eval_local/logs/hunyuan_{task2,cf223}_20_*`.
