#!/bin/bash
#SBATCH --job-name=pretrain_co_retweet_nm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd /home/eibl/gfm/prodigy

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph_co_retweet \
    --input_dim 98 \
    --original_features True \
    --task neighbor_matching \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --workers 10 \
    --prefix pretrain_co_retweet_nm
