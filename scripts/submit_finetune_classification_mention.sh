#!/bin/bash
#SBATCH --job-name=finetune_cls_mention
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
    --root midterm/graph_mention \
    --input_dim 98 \
    --original_features True \
    --task classification \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --pretrained_model_run state/pretrain_midterm_nm_28_02_2026_10_43_36/checkpoint/state_dict_6000.ckpt \
    --workers 10 \
    --prefix finetune_cls_mention
