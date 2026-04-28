#!/bin/bash
#SBATCH --job-name=train3_nm
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
: "${CKPT_PATH:?CKPT_PATH is required}"
: "${TARGET_DATASET:?TARGET_DATASET is required}"

BASE_ARGS=(
  --task_name neighbor_matching
  --feature_subset emb_only
  --input_dim 384
  --original_features True
  --use_edge_features False
  --pretrained_model_run "$CKPT_PATH"
  --seed 0
  --device 0
)

case "$TARGET_DATASET" in
  midterm)
    DATASET_ARGS=(
      --dataset midterm
      --root /scratch1/eibl/data/midterm/graphs
      --graph_filename retweet_graph_5050_all_future_political_leaning.pt
      --n_way 3
      --n_shots 3
      --n_query 24
      --workers 8
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_midterm_nm"
    )
    ;;
  covid19_twitter)
    DATASET_ARGS=(
      --dataset covid19_twitter
      --root /scratch1/eibl/data/covid19_twitter/graphs
      --graph_filename retweet_graph_minilm_first100_hf03.pt
      --edge_view temporal_history
      --n_way 3
      --n_shots 3
      --n_query 8
      --workers 4
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_covid_nm"
    )
    ;;
  ukr_rus_twitter)
    DATASET_ARGS=(
      --dataset ukr_rus_twitter
      --root /scratch1/eibl/data/ukr_rus_twitter/graphs
      --graph_filename retweet_graph_150files_minilm_hf03.pt
      --edge_view temporal_history
      --n_way 2
      --n_shots 3
      --n_query 8
      --workers 4
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_ukr_rus_nm"
    )
    ;;
  election2020)
    DATASET_ARGS=(
      --dataset election2020
      --root /scratch1/eibl/data/election2020/graphs
      --graph_filename retweet_graph.pt
      --n_way 3
      --n_shots 3
      --n_query 12
      --workers 4
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_election2020_nm"
    )
    ;;
  ukr_rus_suspended)
    DATASET_ARGS=(
      --dataset ukr_rus_suspended
      --root /scratch1/eibl/data/ukr_rus_suspended/graphs
      --graph_filename retweet_graph.pt
      --n_way 3
      --n_shots 3
      --n_query 12
      --workers 4
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_ukr_rus_suspended_nm"
    )
    ;;
  covid_political)
    DATASET_ARGS=(
      --dataset covid_political
      --root /scratch1/eibl/data/covid_political/graphs
      --graph_filename retweet_graph.pt
      --n_way 3
      --n_shots 3
      --n_query 12
      --workers 4
      --checkpoint_step 1000
      --val_len_cap 500
      --test_len_cap 500
      --eval_step 1000
      --prefix "train3_${SOURCE_MODEL}_to_covid_political_nm"
    )
    ;;
  *)
    echo "Unknown TARGET_DATASET='$TARGET_DATASET'" >&2
    exit 1
    ;;
esac

echo "=== train3: ${SOURCE_MODEL} -> ${TARGET_DATASET} NM ==="
python3 experiments/run_single_experiment.py "${BASE_ARGS[@]}" "${DATASET_ARGS[@]}"
