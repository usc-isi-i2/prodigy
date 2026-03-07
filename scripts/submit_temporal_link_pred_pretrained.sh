#!/bin/bash
#SBATCH --job-name=temporal_link_pred_pretrained
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd "$(dirname "$0")/.."

mkdir -p logs

# Replace with actual run directory from your NM pretraining job
PRETRAINED_RUN="state/pretrain_co_retweet_nm_<RUN_ID>/state_dict"

python experiments/run_single_experiment.py \
    --dataset midterm \
    --root midterm/graph_temporal \
    --input_dim 98 \
    --original_features True \
    --task temporal_link_prediction \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --pretrained_model_run "$PRETRAINED_RUN" \
    --workers 10 \
    --prefix temporal_link_pred_pretrained
