#!/bin/bash
#SBATCH --job-name=pretrain_midterm_nm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd "$(dirname "$0")/.."

mkdir -p logs

python3 experiments/run_single_experiment.py \
  --dataset midterm \
  --root /scratch1/eibl/data/midterm/graphs \
  --graph_filename retweet_graph_5050_all_future_political_leaning.pt \
  --task_name neighbor_matching \
  --midterm_feature_subset emb_only \
  --input_dim 384 \
  --n_way 3 \
  --n_shots 3 \
  --n_query 24 \
  --checkpoint_step 1000 \
  --original_features True \
  --val_len_cap 500 \
  --test_len_cap 500 \
  --eval_step 1000 \
  --workers 8 \
  --device 0 \
  --seed 0 \
  --prefix train1_midterm_nm
