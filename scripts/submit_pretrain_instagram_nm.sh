#!/bin/bash
#SBATCH --job-name=pretrain_instagram_nm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd "$(dirname "$0")/.."

mkdir -p logs

python experiments/run_single_experiment.py \
    --dataset instagram_mention \
    --root data/graphs/ukr_ru/instagram \
    --graph_filename mention_graph_bge.pt \
    --input_dim 1024 \
    --original_features True \
    --task neighbor_matching \
    --device 0 \
    -val_cap 1000 \
    -test_cap 1000 \
    --workers 10 \
    --prefix pretrain_instagram_nm
