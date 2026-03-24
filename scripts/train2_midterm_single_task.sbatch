#!/bin/bash
#SBATCH --job-name=train2_midterm
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p logs

: "${SOURCE_MODEL:?SOURCE_MODEL is required}"
: "${TARGET_TASK:?TARGET_TASK is required}"
: "${CKPT_PATH:?CKPT_PATH is required}"

COMMON_ARGS=(
  --dataset midterm
  --root /scratch1/eibl/data/midterm/graphs
  --graph_filename retweet_graph_5050_all_future_political_leaning.pt
  --midterm_feature_subset emb_only
  --input_dim 384
  --original_features True
  --workers 8
  --device 0
  --seed 0
  --pretrained_model_run "$CKPT_PATH"
)

case "$TARGET_TASK" in
  neighbor_matching)
    EXTRA_ARGS=(
      --task_name neighbor_matching
      --n_way 3
      --n_shots 3
      --n_query 24
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train2_${SOURCE_MODEL}_to_nm"
    )
    ;;
  temporal_link_prediction)
    EXTRA_ARGS=(
      --task_name temporal_link_prediction
      --n_way 1
      --n_shots 1
      --n_query 3
      --zero_shot False
      --val_len_cap 500
      --test_len_cap 500
      --epochs 3
      --eval_step 1000
      --checkpoint_step 1000
      --midterm_target_edge_view default
      --prefix "train2_${SOURCE_MODEL}_to_lp"
    )
    ;;
  classification)
    EXTRA_ARGS=(
      --task_name classification
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
      --epochs 3
      --checkpoint_step 1000
      --midterm_label_downsample 50:50
      --prefix "train2_${SOURCE_MODEL}_to_pl"
    )
    ;;
  *)
    echo "Unknown TARGET_TASK='$TARGET_TASK'" >&2
    exit 1
    ;;
esac

echo "=== Training ${SOURCE_MODEL} checkpoint on target task ${TARGET_TASK} ==="
python3 experiments/run_single_experiment.py "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
