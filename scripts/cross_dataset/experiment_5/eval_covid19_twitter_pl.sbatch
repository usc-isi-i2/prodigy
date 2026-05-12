#!/bin/bash
#SBATCH --job-name=eval_covid19_twitter_pl
#SBATCH --output=/home1/singhama/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/singhama/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

module purge || true
module load conda

cd /home1/singhama/gfm/prodigy
source scripts/cross_dataset/env.sh
mkdir -p "$LOGS_DIR"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: sbatch $0 <model_list.txt> [shots_csv]" >&2
  exit 1
fi

MODEL_LIST="$1"
SHOTS_CSV="${2:-1,5,10}"
if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"

COMMON_ARGS=(
  --dataset covid19_twitter
  --root "$COVID_ROOT"
  --graph_filename "$COVID_GRAPH"
  --midterm_feature_subset emb_only
  --midterm_edge_view temporal_history
  --input_dim 384
  --original_features True
  --workers 4
  --device 0
  --seed 0
  --eval_only True
  --eval_test_before_train True
  --eval_val_before_train True
  --save_roc_curve True
  --task_name classification
  --ignore_label_embeddings False
  --linear_probe False
  --n_way 2
  --n_query 3
  --dataset_len_cap 2000
  --val_len_cap 500
  --test_len_cap 500
  --epochs 3
  --eval_step 1000
  --checkpoint_step 1000
  --midterm_label_downsample 50:50
)

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue

  read -r first second extra <<< "$line"
  if [[ -n "${second:-}" ]]; then
    model_name="$first"
    ckpt_path="$second"
  else
    ckpt_path="$first"
    model_name="$(basename "$(dirname "$ckpt_path")")"
  fi

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint not found for model '${model_name}': $ckpt_path" >&2
    exit 1
  fi

  for shots in "${SHOT_LIST[@]}"; do
    echo "=== Evaluating model=${model_name} task=classification(PL) shots=${shots} ==="
    conda run -p "$CONDA_ENV" --no-capture-output \
      python3 experiments/run_single_experiment.py \
      "${COMMON_ARGS[@]}" \
      --n_shots "$shots" \
      --zero_shot "$([[ "$shots" == "0" ]] && echo True || echo False)" \
      --pretrained_model_run "$ckpt_path" \
      --prefix "trained_on_${model_name}_eval_on_pl_${shots}_shot"
  done
done < "$MODEL_LIST"
