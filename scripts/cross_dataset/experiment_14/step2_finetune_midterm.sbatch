#!/bin/bash
# Step 2: Fine-tune a ukr_rus NM checkpoint on midterm NM.
# Submit via step2_submit_finetune_midterm.sh — do not submit directly.
#
# Required env vars:
#   CKPT_PATH — path to ukr_rus NM checkpoint file
#SBATCH --job-name=exp14_train2_ukr_rus_nm_to_midterm_nm
#SBATCH --output=/home1/singhama/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/singhama/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

: "${CKPT_PATH:?CKPT_PATH is required}"

module purge || true
module load conda

cd /home1/singhama/gfm/prodigy
mkdir -p /home1/singhama/gfm/prodigy/logs

echo "=== [Exp14] Fine-tuning ukr_rus NM checkpoint on midterm NM ==="
conda run -p /home1/singhama/.conda/envs/prodigy-env --no-capture-output python3 experiments/run_single_experiment.py \
  --dataset              midterm \
  --root                 /scratch1/eibl/data/midterm/graphs \
  --graph_filename       retweet_graph_1p5m.pt \
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
  --prefix               "exp14_train2_ukr_rus_nm_to_midterm_nm"
