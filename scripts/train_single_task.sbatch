#!/bin/bash
# Generic single-task training job.
# Submit via submit_train1.sh or submit_train2.sh — do not submit directly.
#
# Required env vars:
#   DATASET   — one of: midterm, ukr_rus_twitter, covid19_twitter
#   TASK      — nm | lp | pl  (or full names)
#
# Optional env vars:
#   CKPT_PATH       — pretrained checkpoint path (train2 only)
#   SHOTS_OVERRIDE  — override default n_shots from config
#   PREFIX_OVERRIDE — override auto-generated --prefix

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G        # overridden by submit wrapper via --mem=
#SBATCH --time=02:00:00  # overridden by submit wrapper via --time=

set -euo pipefail

: "${DATASET:?DATASET env var is required}"
: "${TASK:?TASK env var is required}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown dataset '${DATASET}' — no config at ${CONFIG}" >&2
  exit 1
fi
source "$CONFIG"

module purge 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Normalise task to short tag
case "$TASK" in
  neighbor_matching|nm)        TASK=neighbor_matching;        TAG=nm ;;
  temporal_link_prediction|lp) TASK=temporal_link_prediction; TAG=lp ;;
  classification|pl)           TASK=classification;           TAG=pl ;;
  *) echo "Unknown task '${TASK}'. Use: nm | lp | pl" >&2; exit 1 ;;
esac

# Build task-specific args from config values
case "$TAG" in
  nm)
    GRAPH_FILENAME="${NM_GRAPH}"
    N_SHOTS="${SHOTS_OVERRIDE:-${NM_TRAIN_SHOTS}}"
    ZERO_SHOT="$([[ "$N_SHOTS" == "0" ]] && echo True || echo False)"
    TASK_ARGS=(
      --n_way  "${NM_N_WAY}"
      --n_shots "$N_SHOTS"
      --n_query "${NM_N_QUERY}"
      --zero_shot "$ZERO_SHOT"
      --val_len_cap 500 --test_len_cap 500
      --eval_step 1000 --checkpoint_step 1000
    )
    ;;
  lp)
    GRAPH_FILENAME="${LP_GRAPH}"
    N_SHOTS="${SHOTS_OVERRIDE:-${LP_TRAIN_SHOTS}}"
    ZERO_SHOT="$([[ "$N_SHOTS" == "0" ]] && echo True || echo False)"
    TASK_ARGS=(
      --n_way  "${LP_N_WAY}"
      --n_shots "$N_SHOTS"
      --n_query "${LP_N_QUERY}"
      --zero_shot "$ZERO_SHOT"
      --val_len_cap 500 --test_len_cap 500
      --epochs 3 --eval_step 1000 --checkpoint_step 1000
      --midterm_target_edge_view "${LP_TARGET_EDGE_VIEW}"
    )
    ;;
  pl)
    if [[ ! " ${SUPPORTED_TASKS[*]} " =~ " classification " ]]; then
      echo "Dataset '${DATASET}' does not support classification." >&2
      exit 1
    fi
    GRAPH_FILENAME="${PL_GRAPH}"
    N_SHOTS="${SHOTS_OVERRIDE:-${PL_TRAIN_SHOTS}}"
    ZERO_SHOT="$([[ "$N_SHOTS" == "0" ]] && echo True || echo False)"
    TASK_ARGS=(
      --n_way  "${PL_N_WAY}"
      --n_shots "$N_SHOTS"
      --n_query "${PL_N_QUERY}"
      --zero_shot "$ZERO_SHOT"
      --dataset_len_cap "${PL_DATASET_LEN_CAP}"
      --val_len_cap 500 --test_len_cap 500
      --epochs 3 --eval_step 1000 --checkpoint_step 1000
      --midterm_label_downsample "${PL_LABEL_DOWNSAMPLE}"
      --ignore_label_embeddings False --linear_probe False
    )
    ;;
esac

PREFIX="${PREFIX_OVERRIDE:-train_${DATASET}_${TAG}}"

COMMON_ARGS=(
  --dataset         "$DATASET"
  --root            "$GRAPH_ROOT"
  --graph_filename  "$GRAPH_FILENAME"
  --task_name       "$TASK"
  --midterm_feature_subset "$FEATURE_SUBSET"
  --input_dim       "$INPUT_DIM"
  --original_features True
  --workers         "$WORKERS"
  --device 0
  --seed   0
  --prefix "$PREFIX"
)

# Edge view is optional; midterm omits it to use the framework default.
if [[ -n "${EDGE_VIEW:-}" ]]; then
  COMMON_ARGS+=(--midterm_edge_view "$EDGE_VIEW")
fi

# Pretrained checkpoint is only set for train2 jobs.
if [[ -n "${CKPT_PATH:-}" ]]; then
  COMMON_ARGS+=(--pretrained_model_run "$CKPT_PATH")
fi

echo "=== Training: dataset=${DATASET} task=${TASK} shots=${N_SHOTS} prefix=${PREFIX} ==="
python3 experiments/run_single_experiment.py "${COMMON_ARGS[@]}" "${TASK_ARGS[@]}"
