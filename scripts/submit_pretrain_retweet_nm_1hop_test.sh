#!/bin/bash
#SBATCH --job-name=pretrain_retweet_nm_1hop_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd "$(dirname "$0")/.."

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph_retweet \
    --input_dim 98 \
    --original_features True \
    --task neighbor_matching \
    --n_hop 1 \
    --device 0 \
    --dataset_len_cap 1000 \
    --epochs 1 \
    --eval_step 500 \
    --checkpoint_step 500 \
    -val_cap 100 \
    -test_cap 100 \
    --workers 10 \
    --prefix pretrain_retweet_nm_1hop_test
