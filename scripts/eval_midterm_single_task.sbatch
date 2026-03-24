#!/bin/bash
#SBATCH --job-name=eval_midterm
#SBATCH --output=/home/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd "$(dirname "$0")/.."
mkdir -p logs

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${TASK_NAME:?TASK_NAME is required}"
: "${CKPT_PATH:?CKPT_PATH is required}"

ROOT="/scratch1/eibl/data/midterm/graphs"
GRAPH="retweet_graph_5050_all_future_political_leaning.pt"

COMMON_ARGS=(
  --dataset midterm
  --root "$ROOT"
  --graph_filename "$GRAPH"
  --midterm_feature_subset emb_only
  --input_dim 384
  --original_features True
  --workers 8
  --device 0
  --seed 0
  --eval_only True
  --pretrained_model_run "$CKPT_PATH"
  --task_name "$TASK_NAME"
  --prefix "eval_${MODEL_NAME}_on_${TASK_NAME}"
)

case "$TASK_NAME" in
  neighbor_matching)
    EXTRA_ARGS=(
      --n_way 3
      --n_shots 3
      --n_query 24
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --checkpoint_step 1000
    )
    ;;
  temporal_link_prediction)
    EXTRA_ARGS=(
      --n_way 1
      --n_shots 1
      --n_query 3
      --zero_shot False
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --checkpoint_step 1000
      --epochs 3
    )
    ;;
  classification)
    EXTRA_ARGS=(
      --ignore_label_embeddings False
      --linear_probe False
      --n_way 2
      --n_shots 4
      --n_query 3
      --zero_shot False
      --dataset_len_cap 2000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --checkpoint_step 1000
      --epochs 3
    )
    ;;
  *)
    echo "Unknown TASK_NAME='$TASK_NAME'" >&2
    exit 1
    ;;
esac

echo "=== Evaluating ${MODEL_NAME} checkpoint on task ${TASK_NAME} ==="
python3 experiments/run_single_experiment.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
