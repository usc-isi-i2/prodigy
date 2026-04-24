#!/bin/bash
# Step 2: Fine-tune a covid LP checkpoint on ukr_rus_twitter NM.
# Submit via step2_submit_finetune_ukr_rus.sh — do not submit directly.
#
# Required env vars:
#   CKPT_PATH — path to covid LP checkpoint file
#SBATCH --job-name=exp12_train2_covid_lp_to_ukr_rus_nm
#SBATCH --output=/home1/singhama/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/singhama/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${CKPT_PATH:?CKPT_PATH is required}"

module purge || true
module load conda

cd /home1/singhama/gfm/prodigy
source scripts/cross_dataset/env.sh
mkdir -p "$LOGS_DIR"

echo "=== [Exp12] Fine-tuning covid LP checkpoint on ukr_rus_twitter NM ==="
conda run -p "$CONDA_ENV" --no-capture-output python3 experiments/run_single_experiment.py \
  --dataset              ukr_rus_twitter \
  --root                 "$UKR_RUS_ROOT" \
  --graph_filename       "$UKR_RUS_GRAPH" \
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
  --pretrained_model_run "$CKPT_PATH" \
  --prefix               "exp12_train2_covid_lp_to_ukr_rus_nm"
