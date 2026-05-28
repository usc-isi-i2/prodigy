#!/bin/bash
#SBATCH --job-name=train3_same_dataset
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
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
mkdir -p /home1/eibl/gfm/prodigy/logs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

: "${SOURCE_MODEL:?SOURCE_MODEL is required}"
: "${SOURCE_DATASET:?SOURCE_DATASET is required}"
: "${TARGET_TASK:?TARGET_TASK is required}"
: "${CKPT_PATH:?CKPT_PATH is required}"

COMMON_ARGS=(
  --dataset "$SOURCE_DATASET"
  --feature_subset emb_only
  --input_dim 384
  --original_features True
  --device 0
  --seed 0
  --pretrained_model_run "$CKPT_PATH"
)

case "$SOURCE_DATASET" in
  midterm)
    DATASET_ARGS=(
      --root /scratch1/eibl/data/midterm/graphs
      --graph_filename retweet_graph_1p5m.pt
      --edge_view temporal_history
      --workers 8
    )
    DATASET_PREFIX="midterm"
    ;;
  covid19_twitter)
    DATASET_ARGS=(
      --root /scratch1/eibl/data/covid19_twitter/graphs
      --graph_filename retweet_graph_1p5m_hf03_labeled.pt
      --edge_view temporal_history
      --workers 4
    )
    DATASET_PREFIX="covid19_twitter"
    ;;
  *)
    echo "Unknown SOURCE_DATASET='$SOURCE_DATASET'" >&2
    exit 1
    ;;
esac

case "$TARGET_TASK" in
  neighbor_matching)
    EXTRA_ARGS=(
      --task_name neighbor_matching
      --n_way 3
      --n_shots 3
      --n_query 12
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_${DATASET_PREFIX}_nm"
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
      --target_edge_view temporal_new
      --prefix "train3_${SOURCE_MODEL}_to_${DATASET_PREFIX}_lp"
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
      --prefix "train3_${SOURCE_MODEL}_to_${DATASET_PREFIX}_pl"
    )
    ;;
  *)
    echo "Unknown TARGET_TASK='$TARGET_TASK'" >&2
    exit 1
    ;;
esac

echo "=== train3: ${SOURCE_MODEL} -> ${SOURCE_DATASET} ${TARGET_TASK} ==="
python3 experiments/run_single_experiment.py \
  "${COMMON_ARGS[@]}" \
  "${DATASET_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
