#!/bin/bash
#SBATCH --job-name=train1_ukr_rus_twitter_pl
#SBATCH --output=/scratch1/eibl/data/ukr_rus_twitter/logs/%x_%j.out
#SBATCH --error=/scratch1/eibl/data/ukr_rus_twitter/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p /scratch1/eibl/data/ukr_rus_twitter/logs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python3 experiments/run_single_experiment.py \
  --dataset ukr_rus_twitter \
  --root /scratch1/eibl/data/ukr_rus_twitter/graphs \
  --graph_filename retweet_graph_150files_minilm_hf03_political_labels.pt \
  --task_name classification \
  --midterm_feature_subset emb_only \
  --midterm_edge_view temporal_history \
  --input_dim 384 \
  --original_features True \
  --ignore_label_embeddings False \
  --linear_probe False \
  --n_way 2 \
  --n_shots 4 \
  --n_query 3 \
  --zero_shot False \
  --dataset_len_cap 2000 \
  --val_len_cap 500 \
  --test_len_cap 500 \
  --eval_step 1000 \
  --epochs 3 \
  --checkpoint_step 1000 \
  --workers 8 \
  --device 0 \
  --seed 0 \
  --midterm_label_downsample 50:50 \
  --prefix train1_ukr_rus_twitter_pl
