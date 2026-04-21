#!/bin/bash
#SBATCH --job-name=train1_ukr_rus_suspended_nm
#SBATCH --output=/scratch1/eibl/logs/ukr_rus_suspended/%x_%j.out
#SBATCH --error=/scratch1/eibl/logs/ukr_rus_suspended/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/logs/ukr_rus_suspended
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python3 experiments/run_single_experiment.py \
  --dataset ukr_rus_suspended \
  --root /scratch1/eibl/data/ukr_rus_suspended/graphs \
  --graph_filename retweet_graph.pt \
  --task_name neighbor_matching \
  --midterm_feature_subset emb_only \
  --input_dim 384 \
  --n_way 3 \
  --n_shots 3 \
  --n_query 8 \
  --checkpoint_step 1000 \
  --original_features True \
  --val_len_cap 500 \
  --test_len_cap 500 \
  --dataset_len_cap 2000 \
  --epochs 20 \
  --eval_step 1000 \
  --workers 4 \
  --device 0 \
  --seed 0 \
  --prefix train1_ukr_rus_suspended_nm
