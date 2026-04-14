#!/bin/bash
# Submit all leave-one-out combo experiments for a dataset.
#
# For each held-out task C, both orderings of the remaining two tasks A and B
# are submitted:  A → fine-tune B → eval C
#                 B → fine-tune A → eval C
#
# This produces 6 jobs (3 held-out tasks × 2 orderings), covering the full
# leave-one-out matrix across tasks.
#
# Usage:
#   submit_train_combo_all.sh <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt> [shots_csv]
#
#   dataset:   midterm | ukr_rus_twitter | covid19_twitter
#   nm_ckpt:   train1 checkpoint for neighbor_matching
#   lp_ckpt:   train1 checkpoint for temporal_link_prediction
#   pl_ckpt:   train1 checkpoint for classification (pass "" if unsupported)
#   shots_csv: comma-separated shot counts (default: 0,1,2,5,10)
#
# Example — all 6 midterm combos:
#   submit_train_combo_all.sh midterm \
#     state/train1_midterm_nm_.../state_dict \
#     state/train1_midterm_lp_.../state_dict \
#     state/train1_midterm_pl_.../state_dict
#
# Example — covid19_twitter (no pl checkpoint):
#   submit_train_combo_all.sh covid19_twitter \
#     state/train1_covid19_twitter_nm_.../state_dict \
#     state/train1_covid19_twitter_lp_.../state_dict \
#     ""

set -euo pipefail

usage() {
  echo "Usage: $0 <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt> [shots_csv]" >&2
  exit 1
}

[[ $# -lt 4 || $# -gt 5 ]] && usage

DATASET="$1"
NM_CKPT="$2"
LP_CKPT="$3"
PL_CKPT="$4"
SHOTS_CSV="${5:-0,1,2,5,10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "No config found for dataset '${DATASET}'" >&2
  exit 1
fi
source "$CONFIG"

# ---------------------------------------------------------------------------
# submit <src_task> <src_ckpt> <finetune_task> <eval_task>
# Skips silently when src_ckpt is empty or the eval task is unsupported.
# ---------------------------------------------------------------------------
submit() {
  local src_task="$1"
  local src_ckpt="$2"
  local ft_task="$3"
  local eval_task="$4"

  # Skip if either checkpoint is absent
  if [[ -z "$src_ckpt" ]]; then
    echo "Skipping ${src_task}→${ft_task} eval ${eval_task}: no ${src_task} checkpoint." >&2
    return
  fi

  # Resolve full task name for the SUPPORTED_TASKS check
  local eval_full
  case "$eval_task" in
    nm) eval_full=neighbor_matching ;;
    lp) eval_full=temporal_link_prediction ;;
    pl) eval_full=classification ;;
  esac

  if [[ ! " ${SUPPORTED_TASKS[*]} " =~ " ${eval_full} " ]]; then
    echo "Skipping ${src_task}→${ft_task} eval ${eval_task}: '${eval_task}' not supported by ${DATASET}." >&2
    return
  fi

  "${SCRIPT_DIR}/submit_train_combo.sh" \
    "$DATASET" "$src_task" "$src_ckpt" "$ft_task" "$eval_task" "$SHOTS_CSV"
}

echo "Submitting leave-one-out combo experiments for dataset: ${DATASET}"
echo ""

# ---------------------------------------------------------------------------
# Held-out: PL — train on NM + LP (both orderings)
# ---------------------------------------------------------------------------
echo "--- Held-out: pl ---"
submit nm "$NM_CKPT" lp pl
submit lp "$LP_CKPT" nm pl

# ---------------------------------------------------------------------------
# Held-out: LP — train on NM + PL (both orderings)
# ---------------------------------------------------------------------------
echo "--- Held-out: lp ---"
submit nm "$NM_CKPT" pl lp
submit pl "$PL_CKPT" nm lp

# ---------------------------------------------------------------------------
# Held-out: NM — train on LP + PL (both orderings)
# ---------------------------------------------------------------------------
echo "--- Held-out: nm ---"
submit lp "$LP_CKPT" pl nm
submit pl "$PL_CKPT" lp nm
