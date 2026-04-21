#!/bin/bash
#SBATCH --job-name=train1_election2020_pl
#SBATCH --output=/scratch1/eibl/logs/election2020/%x_%j.out
#SBATCH --error=/scratch1/eibl/logs/election2020/%x_%j.err
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
mkdir -p /scratch1/eibl/logs/election2020
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python3 experiments/run_single_experiment.py \
  --dataset election2020 \
  --root /scratch1/eibl/data/election2020/graphs \
  --graph_filename retweet_graph.pt \
  --task_name classification \
  --midterm_feature_subset emb_only \
  --input_dim 384 \
  --n_way 2 \
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
  --prefix train1_election2020_pl
