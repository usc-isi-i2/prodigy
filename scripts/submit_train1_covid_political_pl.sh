#!/bin/bash
#SBATCH --job-name=train1_covid_political_pl
#SBATCH --output=/scratch1/eibl/logs/covid_political/%x_%j.out
#SBATCH --error=/scratch1/eibl/logs/covid_political/%x_%j.err
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
mkdir -p /scratch1/eibl/logs/covid_political
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python3 experiments/run_single_experiment.py \
  --dataset covid_political \
  --root /home1/eibl/gfm/prodigy/data/data/covid_political/graphs \
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
  --eval_step 1000 \
  --workers 4 \
  --device 0 \
  --seed 0 \
  --prefix train1_covid_political_pl
