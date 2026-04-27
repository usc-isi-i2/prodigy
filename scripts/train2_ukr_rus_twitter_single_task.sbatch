#!/bin/bash
#SBATCH --job-name=train2_ukr_rus_twitter
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/data/ukr_rus_twitter/logs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

: "${SOURCE_MODEL:?SOURCE_MODEL is required}"
: "${TARGET_TASK:?TARGET_TASK is required}"
: "${CKPT_PATH:?CKPT_PATH is required}"

COMMON_ARGS=(
  --dataset ukr_rus_twitter
  --root /scratch1/eibl/data/ukr_rus_twitter/graphs
  --midterm_feature_subset emb_only
  --midterm_edge_view temporal_history
  --input_dim 384
  --original_features True
  --workers 4
  --device 0
  --seed 0
  --pretrained_model_run "$CKPT_PATH"
)

case "$TARGET_TASK" in
  neighbor_matching)
    EXTRA_ARGS=(
      --graph_filename retweet_graph_150files_minilm_hf03.pt
      --task_name neighbor_matching
      --n_way 2
      --n_shots 3
      --n_query 8
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train2_${SOURCE_MODEL}_to_nm"
    )
    ;;
  temporal_link_prediction)
    EXTRA_ARGS=(
      --graph_filename retweet_graph_150files_minilm_hf03.pt
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
      --midterm_target_edge_view temporal_new
      --prefix "train2_${SOURCE_MODEL}_to_lp"
    )
    ;;
  classification)
    EXTRA_ARGS=(
      --graph_filename retweet_graph_150files_minilm_hf03_political_labels.pt
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
