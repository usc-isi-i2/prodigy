#!/usr/bin/env bash
set -euo pipefail

# NM + classification use the episodic evaluator because support/query context is
# part of the prediction.  Override every variable from the environment.
MODEL_LIST=${MODEL_LIST:?Set MODEL_LIST to name/checkpoint rows}
DATASETS=${DATASETS:-midterm,covid_political}
TASKS=${TASKS:-neighbor_matching,classification}
SHOTS=${SHOTS:-3}
GPUS=${GPUS:-0}

python scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "$MODEL_LIST" \
  --datasets "$DATASETS" \
  --tasks "$TASKS" \
  --shots "$SHOTS" \
  --gpus "$GPUS" \
  --nm-n-way "${NM_N_WAY:-30}" \
  --parquet-val-cap "${VAL_CAP:-500}" \
  --parquet-test-cap "${TEST_CAP:-500}" \
  -- \
  --export_predictions True \
  --prediction_context_neighbors "${CONTEXT_NEIGHBORS:-3}" \
  --prediction_support_per_label "${SUPPORTS_PER_LABEL:-3}"
