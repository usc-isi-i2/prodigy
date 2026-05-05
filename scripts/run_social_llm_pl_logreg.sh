#!/bin/bash
set -euo pipefail

FEATURE_SUBSET="${1:-emb_only}"
SAFE_FEATURE_SUBSET="$(printf '%s' "$FEATURE_SUBSET" | tr -c 'A-Za-z0-9._-' '_')"

cd /home1/eibl/gfm/prodigy
mkdir -p logs state/analysis

python3 scripts/analysis/social_llm_pl_logistic_regression.py \
  --dataset election2020 \
  --feature_subset "$FEATURE_SUBSET" \
  --out "state/analysis/election2020_pl_logreg_${SAFE_FEATURE_SUBSET}.json"

python3 scripts/analysis/social_llm_pl_logistic_regression.py \
  --dataset ukr_rus_suspended \
  --feature_subset "$FEATURE_SUBSET" \
  --out "state/analysis/ukr_rus_suspended_pl_logreg_${SAFE_FEATURE_SUBSET}.json"

python3 scripts/analysis/social_llm_pl_logistic_regression.py \
  --dataset covid_political \
  --feature_subset "$FEATURE_SUBSET" \
  --out "state/analysis/covid_political_pl_logreg_${SAFE_FEATURE_SUBSET}.json"
