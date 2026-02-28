#!/bin/bash
#SBATCH --job-name=exp5_midterm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph \
    --input_dim 89 \
    --original_features True \
    --task classification \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --prefix exp5_train_midterm
