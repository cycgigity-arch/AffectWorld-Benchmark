# Evaluation code

These scripts were copied from the evaluation machine under `/root/autodl-tmp`.

- `legacy/`: original local Task 2, Task 3 and counterfactual evaluation implementation.
- `cf223/`: the later 223-counterfactual implementation and its Task 1 scorer. This is a distinct version; keep it separate from `legacy/`.
- `orchestration/`: scripts used to launch, resume, retry and summarize model evaluations.

Task 1 uses local vision models and requires additional dependencies, checkpoints, and GPU resources beyond the minimal requirements files. Task 2/3 and counterfactual evaluation use VLM/LLM APIs. Provide `EWM_VLM_API_KEY` and `EWM_LLM_API_KEY` through environment variables or the launcher prompt; never commit credentials. Some scripts contain paths specific to the original AutoDL machine and must be configured for a new environment.

The annotation file in the repository may differ from the annotation versions used in historical runs. Reproducing reported scores requires matching the original annotation version, input videos, model versions, configurations, and evaluation run records. In the original LTX Task 1 raw results, 502 pre-retry backup videos were included; filter them when calculating results for the 1498 current videos.
