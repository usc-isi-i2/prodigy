#!/bin/bash
#SBATCH --job-name=exp3_train1_covid_nm
#SBATCH --output=/home1/singhama/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/singhama/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

module purge || true
module load conda

cd /home1/singhama/gfm/prodigy
mkdir -p logs

echo "=== [Exp3] Pretraining on covid19_twitter NM ==="
conda run -p /home1/singhama/.conda/envs/prodigy-env --no-capture-output python3 experiments/run_single_experiment.py \
  --dataset              covid19_twitter \
  --root                 /scratch1/eibl/data/covid19_twitter/graphs \
  --graph_filename       retweet_graph_1p5m_hf03_labeled.pt \
  --task_name            neighbor_matching \
  --midterm_feature_subset emb_only \
  --midterm_edge_view    temporal_history \
  --input_dim            384 \
  --original_features    True \
  --n_way                2 \
  --n_shots              3 \
  --n_query              8 \
  --epochs               3 \
  --val_len_cap          500 \
  --test_len_cap         500 \
  --eval_step            1000 \
  --checkpoint_step      1000 \
  --workers              4 \
  --device               0 \
  --seed                 0 \
  --prefix               "exp3_train1_covid_nm"
