#!/bin/bash
#SBATCH --job-name=eval_midterm_all
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch $0 <model_list.txt>" >&2
  echo "Each non-empty, non-comment line may be either:" >&2
  echo "  <ckpt_path>" >&2
  echo "or" >&2
  echo "  <model_name> <ckpt_path>" >&2
  exit 1
fi

MODEL_LIST="$1"
if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

COMMON_ARGS=(
  --dataset midterm
  --root /scratch1/eibl/data/midterm/graphs
  --graph_filename retweet_graph_5050_all_future_political_leaning.pt
  --midterm_feature_subset emb_only
  --input_dim 384
  --original_features True
  --workers 8
  --device 0
  --seed 0
  --eval_only True
  --eval_test_before_train True
  --eval_val_before_train True
  --save_roc_curve True
)

run_eval() {
  local model_name="$1"
  local ckpt_path="$2"
  local task_name="$3"

  local extra_args=()
  case "$task_name" in
    neighbor_matching)
      extra_args=(
        --task_name neighbor_matching
        --n_way 3
        --n_shots 3
        --n_query 24
        --val_len_cap 500
        --test_len_cap 500
        --eval_step 1000
        --checkpoint_step 1000
        --prefix "eval_${model_name}_on_nm"
      )
      ;;
    temporal_link_prediction)
      extra_args=(
        --task_name temporal_link_prediction
        --n_way 1
        --n_shots 1
        --n_query 3
        --zero_shot False
        --val_len_cap 500
        --test_len_cap 500
        --epochs 3
        --eval_step 1000
        --checkpoint_step 1000
        --midterm_target_edge_view default
        --prefix "eval_${model_name}_on_lp"
      )
      ;;
    classification)
      extra_args=(
        --task_name classification
        --ignore_label_embeddings False
        --linear_probe False
        --n_way 2
        --n_shots 4
        --n_query 3
        --zero_shot False
        --dataset_len_cap 2000
        --val_len_cap 500
        --test_len_cap 500
        --epochs 3
        --eval_step 1000
        --checkpoint_step 1000
        --midterm_label_downsample 50:50
        --prefix "eval_${model_name}_on_pl"
      )
      ;;
    *)
      echo "Unknown task: $task_name" >&2
      exit 1
      ;;
  esac

  echo "=== Evaluating model=${model_name} task=${task_name} ckpt=${ckpt_path} ==="
  python3 experiments/run_single_experiment.py \
    "${COMMON_ARGS[@]}" \
    --pretrained_model_run "$ckpt_path" \
    "${extra_args[@]}"
}

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue

  model_name=""
  ckpt_path=""

  read -r first second extra <<< "$line"
  if [[ -n "${second:-}" ]]; then
    model_name="$first"
    ckpt_path="$second"
    if [[ -n "${extra:-}" ]]; then
      echo "Invalid line in ${MODEL_LIST}: '$raw_line'" >&2
      echo "Expected either '<ckpt_path>' or '<model_name> <ckpt_path>'." >&2
      exit 1
    fi
  else
    ckpt_path="$first"
    model_name="$(basename "$(dirname "$ckpt_path")")"
  fi

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint not found for model '${model_name}': $ckpt_path" >&2
    exit 1
  fi

  run_eval "$model_name" "$ckpt_path" neighbor_matching
  run_eval "$model_name" "$ckpt_path" temporal_link_prediction
  run_eval "$model_name" "$ckpt_path" classification
done < "$MODEL_LIST"

