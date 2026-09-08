#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data/prodigy_public_original}"
CKPT_ROOT="${CKPT_ROOT:-/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/log/social_to_fb15k_adapter}"
PYTHON="${PYTHON:-python}"

export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$OUT_ROOT"
cd "$REPO_ROOT"

models=(
  "ukr_rus:$CKPT_ROOT/finalcore_ss_ukr_rus_s0_20260807/checkpoint/state_dict_2500.ckpt"
  "covid:$CKPT_ROOT/finalcore_ss_covid_s0_20260807/checkpoint/state_dict_2500.ckpt"
  "twibot20:$CKPT_ROOT/finalcore_ss_twibot20_s0_20260807/checkpoint/state_dict_2500.ckpt"
  "midterm:$CKPT_ROOT/finalcore_ss_midterm_s0_20260807/checkpoint/state_dict_2500.ckpt"
)

label_set=()
for ((i=0; i<200; i++)); do label_set+=("$i"); done

pids=()
for i in "${!models[@]}"; do
  IFS=: read -r name checkpoint <<< "${models[$i]}"
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
  gpu="$i"
  "$PYTHON" -u experiments/run_single_experiment.py \
    --dataset FB15K-237 --root "$DATA_ROOT" \
    --task_name multiway_classification --eval_only True \
    --pretrained_model_run "$checkpoint" \
    --kg_social_checkpoint_adapter True \
    --layers S,U,M --input_dim 768 --emb_dim 256 --gnn_type sage \
    --n_way 20 --n_shots 3 --n_query 4 --batch_size 1 \
    --dataset_len_cap 1 --val_len_cap 1 --test_len_cap 500 \
    --workers 7 --device "$gpu" --seed 0 \
    --no_split_labels True --label_set "${label_set[@]}" \
    --ignore_label_embeddings True --n_hop 1 \
    --prefix "social_fb15k_${name}" \
    --log_dir "$OUT_ROOT/log" --state_dir "$OUT_ROOT/state" \
    > "$OUT_ROOT/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
