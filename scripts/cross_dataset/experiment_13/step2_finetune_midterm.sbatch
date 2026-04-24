#!/bin/bash
# Step 2: Fine-tune a covid NM checkpoint on midterm NM.
# Submit via step2_submit_finetune_midterm.sh — do not submit directly.
#
# Required env vars:
#   CKPT_PATH — path to covid NM checkpoint file
#SBATCH --job-name=exp13_train2_covid_nm_to_midterm_nm
#SBATCH --output=/home1/singhama/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/singhama/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${CKPT_PATH:?CKPT_PATH is required}"

module purge || true
module load conda

cd /home1/singhama/gfm/prodigy
source scripts/cross_dataset/env.sh
mkdir -p "$LOGS_DIR"

echo "=== [Exp13] Fine-tuning covid NM checkpoint on midterm NM ==="
conda run -p "$CONDA_ENV" --no-capture-output python3 experiments/run_single_experiment.py \
  --dataset              midterm \
  --root                 "$MIDTERM_ROOT" \
  --graph_filename       "$MIDTERM_GRAPH" \
  --task_name            neighbor_matching \
  --midterm_feature_subset emb_only \
  --input_dim            384 \
  --original_features    True \
  --n_way                3 \
  --n_shots              3 \
  --n_query              24 \
  --val_len_cap          500 \
  --test_len_cap         500 \
  --epochs               3 \
  --eval_step            1000 \
  --checkpoint_step      1000 \
  --workers              8 \
  --device               0 \
  --seed                 0 \
  --pretrained_model_run "$CKPT_PATH" \
  --prefix               "exp13_train2_covid_nm_to_midterm_nm"
