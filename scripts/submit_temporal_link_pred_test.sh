#!/bin/bash
#SBATCH --job-name=temporal_link_pred_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd "$(dirname "$0")/.."

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph_temporal \
    --input_dim 98 \
    --original_features True \
    --task temporal_link_prediction \
    --device 0 \
    --dataset_len_cap 1000 \
    --epochs 1 \
    --eval_step 500 \
    --checkpoint_step 500 \
    -val_cap 100 \
    -test_cap 100 \
    --prefix temporal_link_pred_test
