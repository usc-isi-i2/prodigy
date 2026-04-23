#!/bin/bash
# Submit a train1 (single-task, from scratch) job.
#
# Usage: submit_train1.sh <dataset> <task> [extra_sbatch_args...]
#
#   dataset: midterm | ukr_rus_twitter | covid19_twitter
#   task:    nm | lp | pl  (or full names)
#
# Extra sbatch args are forwarded verbatim, e.g.:
#   submit_train1.sh midterm nm --time=02:00:00

set -euo pipefail

usage() {
  echo "Usage: $0 <dataset> <task> [extra_sbatch_args...]" >&2
  echo "  dataset: midterm | ukr_rus_twitter | covid19_twitter" >&2
  echo "  task:    nm | lp | pl" >&2
  exit 1
}

[[ $# -lt 2 ]] && usage

DATASET="$1"; TASK="$2"; shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "No config found for dataset '${DATASET}' (looked at ${CONFIG})" >&2
  exit 1
fi
source "$CONFIG"

JOB_NAME="train1_${DATASET}_${TASK}"

sbatch \
  --job-name="$JOB_NAME" \
  --output="${LOG_DIR}/%x_%j.out" \
  --error="${LOG_DIR}/%x_%j.err" \
  --mem="${SLURM_MEM}" \
  --export="ALL,DATASET=${DATASET},TASK=${TASK},PREFIX_OVERRIDE=${JOB_NAME}" \
  "$@" \
  "${SCRIPT_DIR}/train_single_task.sbatch"
