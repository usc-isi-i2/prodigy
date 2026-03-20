#!/bin/bash
#SBATCH --job-name=pretrain_retweet_nm_1hop
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph_retweet \
    --input_dim 98 \
    --original_features True \
    --task neighbor_matching \
    --n_hop 1 \
    --device 0 \
    -val_cap 500 \
    -test_cap 500 \
    --workers 10 \
    --epochs 4 \
    --dataset_len_cap 10000 \
    --prefix pretrain_retweet_nm_1hop
