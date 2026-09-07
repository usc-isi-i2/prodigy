#!/usr/bin/env bash
# End-to-end overnight matched-budget training, replay, and analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260907v1}"
TOTAL_STEPS="${TOTAL_STEPS:-2500}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/trace_health_guided}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_health_guided}"
SCHEDULE_MODEL_LIST="${SCHEDULE_MODEL_LIST:-/dataMeR1/phil/gfm/prodigy-trace-schedule/log/trace_schedule_scaling/launch/model_list_20260907v1.tsv}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

PHASE=full RUN_STAMP="$RUN_STAMP" TOTAL_STEPS="$TOTAL_STEPS" \
  STATE_ROOT="$STATE_ROOT" LOG_ROOT="$LOG_ROOT" \
  bash "$SCRIPT_DIR/run_stage1_tucker.sh"

"$PYTHON" -m scripts.experiments.setup.trace_health_guided.verify_stage1 \
  --state-root "$STATE_ROOT" --log-root "$LOG_ROOT" \
  --run-stamp "$RUN_STAMP" --total-steps "$TOTAL_STEPS" \
  --output "$LOG_ROOT/launch/stage1_verification_${RUN_STAMP}.json"

RUN_STAMP="$RUN_STAMP" LOG_ROOT="$LOG_ROOT" GPU=0 \
  bash "$SCRIPT_DIR/run_specialist_replay_tucker.sh"

STAGE1_MODEL_LIST="$LOG_ROOT/launch/stage1_model_list_${RUN_STAMP}.tsv"
HEALTH_PLAN="$LOG_ROOT/launch/health_plan_${RUN_STAMP}.tsv"
"$PYTHON" -m scripts.experiments.setup.trace_health_guided.derive_health_schedule \
  --replay-directory "$LOG_ROOT/health_replay_${RUN_STAMP}/original/facebook_page_reference/facebook_page_reference" \
  --model-list "$STAGE1_MODEL_LIST" --output "$HEALTH_PLAN" \
  --total-steps "$TOTAL_STEPS"

RUN_STAMP="$RUN_STAMP" TOTAL_STEPS="$TOTAL_STEPS" \
  STATE_ROOT="$STATE_ROOT" LOG_ROOT="$LOG_ROOT" PLAN="$HEALTH_PLAN" \
  bash "$SCRIPT_DIR/run_stage2_tucker.sh"

"$PYTHON" -m scripts.experiments.setup.trace_health_guided.verify_stage2 \
  --plan "$HEALTH_PLAN" --state-root "$STATE_ROOT" --log-root "$LOG_ROOT" \
  --run-stamp "$RUN_STAMP" --total-steps "$TOTAL_STEPS" \
  --output "$LOG_ROOT/launch/stage2_verification_${RUN_STAMP}.json"

FINAL_MODEL_LIST="$LOG_ROOT/launch/final_model_list_${RUN_STAMP}.tsv"
"$PYTHON" -m scripts.experiments.setup.trace_health_guided.build_final_model_list \
  --schedule-model-list "$SCHEDULE_MODEL_LIST" \
  --stage1-model-list "$STAGE1_MODEL_LIST" \
  --health-model-list "$LOG_ROOT/launch/health_model_list_${RUN_STAMP}.tsv" \
  --output "$FINAL_MODEL_LIST"

RUN_STAMP="$RUN_STAMP" LOG_ROOT="$LOG_ROOT" MODEL_LIST="$FINAL_MODEL_LIST" \
  bash "$SCRIPT_DIR/run_final_replay_tucker.sh"

"$PYTHON" -m scripts.experiments.setup.trace_health_guided.analyze_results \
  --original-directory "$LOG_ROOT/final_replay_${RUN_STAMP}/original/facebook_page_reference/facebook_page_reference" \
  --fresh-directory "$LOG_ROOT/final_replay_${RUN_STAMP}/fresh/facebook_page_reference/facebook_page_reference" \
  --health-plan "$HEALTH_PLAN" \
  --output "$LOG_ROOT/analysis_${RUN_STAMP}"

echo "TRACE_HEALTH_PIPELINE_COMPLETE results=$LOG_ROOT/analysis_${RUN_STAMP}"
