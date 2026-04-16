#!/bin/bash
#SBATCH --job-name=exp6_train1_covid_lp
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

echo "=== [Exp6] Pretraining on covid19_twitter LP ==="
conda run -p /home1/singhama/.conda/envs/prodigy-env --no-capture-output python3 experiments/run_single_experiment.py \
  --dataset                covid19_twitter \
  --root                   /scratch1/eibl/data/covid19_twitter/graphs \
  --graph_filename         retweet_graph_1p5m_hf03_labeled.pt \
  --task_name              temporal_link_prediction \
  --midterm_feature_subset emb_only \
  --midterm_edge_view      temporal_history \
  --midterm_target_edge_view temporal_new \
  --input_dim              384 \
  --original_features      True \
  --n_way                  1 \
  --n_shots                1 \
  --n_query                3 \
  --zero_shot              False \
  --epochs                 3 \
  --val_len_cap            500 \
  --test_len_cap           500 \
  --eval_step              1000 \
  --checkpoint_step        1000 \
  --workers                4 \
  --device                 0 \
  --seed                   0 \
  --prefix                 "exp6_train1_covid_lp"
