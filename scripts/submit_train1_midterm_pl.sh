#!/bin/bash
#SBATCH --job-name=midterm_pl_sanity_emb_only
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p logs

python3 experiments/run_single_experiment.py \
  --dataset midterm \
  --root /scratch1/eibl/data/midterm/graphs \
  --graph_filename retweet_graph_5050_all_future_political_leaning.pt \
  --task_name classification \
  --midterm_feature_subset emb_only \
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
  --prefix train1_midterm_pl
