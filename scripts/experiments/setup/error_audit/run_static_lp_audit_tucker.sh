#!/usr/bin/env bash
set -euo pipefail

MODEL_LIST=${MODEL_LIST:?Set MODEL_LIST to name/checkpoint rows}
DATASET=${DATASET:-midterm}
GRAPH=${GRAPH:-/dataMeR1/phil/data/${DATASET}/graphs/retweet_graph_parquet.pt}
OUT_DIR=${OUT_DIR:-/dataMeR1/phil/gfm/error_audit/static_lp/${DATASET}}

python scripts/eval/pair_link_sweep.py \
  --graph "$GRAPH" \
  --dataset "$DATASET" \
  --model-list "$MODEL_LIST" \
  --out-dir "$OUT_DIR" \
  --background-view static_background \
  --holdout-view static_holdout \
  --negative-kinds "${NEGATIVE_KINDS:-degree_matched}" \
  --max-positives "${MAX_POSITIVES:-2000}" \
  --n-hop "${N_HOP:-1}" \
  --device "${DEVICE:-cuda}" \
  --export-examples \
  --context-neighbors "${CONTEXT_NEIGHBORS:-3}"
