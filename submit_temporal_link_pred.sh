#!/bin/bash
#SBATCH --job-name=temporal_link_pred
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph_temporal \
    --input_dim 98 \
    --original_features True \
    --task temporal_link_prediction \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --prefix temporal_link_pred_scratch
