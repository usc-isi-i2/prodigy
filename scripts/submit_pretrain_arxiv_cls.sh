#!/bin/bash
#SBATCH --job-name=pretrain_arxiv_cls
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset arxiv \
    --root dataset \
    --input_dim 768 \
    --original_features False \
    --task classification \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --workers 10 \
    --dataset_len_cap 10000 \
    --epochs 4 \
    -way 3 \
    -shot 3 \
    -qry 24 \
    --prefix pretrain_arxiv_cls
