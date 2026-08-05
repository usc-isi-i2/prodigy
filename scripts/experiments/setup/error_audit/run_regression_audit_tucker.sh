#!/usr/bin/env bash
set -euo pipefail

MODEL_LIST=${MODEL_LIST:?Set MODEL_LIST to name/checkpoint rows}
DATASET=${DATASET:-midterm}
GRAPH=${GRAPH:-/dataMeR1/phil/data/${DATASET}/graphs/retweet_graph_parquet.pt}
OUT_DIR=${OUT_DIR:-/dataMeR1/phil/gfm/error_audit/regression/${DATASET}}

python scripts/eval/regression_probe_sweep.py \
  --graph "$GRAPH" \
  --dataset "$DATASET" \
  --model-list "$MODEL_LIST" \
  --out-dir "$OUT_DIR" \
  --targets "${TARGETS:-followers_count,statuses_count,account_age_days}" \
  --shots "${SHOTS:-10}" \
  --n-query "${N_QUERY:-12}" \
  --episodes "${EPISODES:-500}" \
  --background-view "${BACKGROUND_VIEW:-static_background}" \
  --n-hop "${N_HOP:-1}" \
  --device "${DEVICE:-cuda}" \
  --export-examples \
  --context-neighbors "${CONTEXT_NEIGHBORS:-3}"
