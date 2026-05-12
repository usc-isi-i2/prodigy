#!/bin/bash
#SBATCH --job-name=eval_covid19_twitter_all
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
  echo "Each non-empty, non-comment line may be either:" >&2
  echo "  <ckpt_path>" >&2
  echo "or" >&2
  echo "  <model_name> <ckpt_path>" >&2
  echo "Example shots_csv: 0,1,2,5,10" >&2
  exit 1
fi

MODEL_LIST="$1"
SHOTS_CSV="${2:-0,1,2,5,10}"
if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"
if [[ ${#SHOT_LIST[@]} -eq 0 ]]; then
  echo "No shots provided via shots_csv='${SHOTS_CSV}'" >&2
  exit 1
fi

COMMON_ARGS=(
  --dataset covid19_twitter
  --root "$COVID_ROOT"
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
)

run_eval() {
  local model_name="$1"
  local ckpt_path="$2"
  local task_name="$3"
  local shots="$4"

  local task_tag=""
  local extra_args=()
  case "$task_name" in
    neighbor_matching)
      task_tag="nm"
      extra_args=(
        --graph_filename "$COVID_GRAPH"
        --task_name neighbor_matching
        --n_way 2
        --n_shots "$shots"
        --n_query 8
        --zero_shot "$([[ "$shots" == "0" ]] && echo True || echo False)"
        --val_len_cap 500
        --test_len_cap 500
        --eval_step 1000
        --checkpoint_step 1000
        --prefix "trained_on_${model_name}_eval_on_${task_tag}_${shots}_shot"
      )
      ;;
    temporal_link_prediction)
      task_tag="lp"
      extra_args=(
        --graph_filename "$COVID_GRAPH"
        --task_name temporal_link_prediction
        --n_way 1
        --n_shots "$shots"
        --n_query 3
        --zero_shot "$([[ "$shots" == "0" ]] && echo True || echo False)"
        --val_len_cap 500
        --test_len_cap 500
        --epochs 3
        --eval_step 1000
        --checkpoint_step 1000
        --midterm_target_edge_view temporal_new
        --prefix "trained_on_${model_name}_eval_on_${task_tag}_${shots}_shot"
      )
      ;;
    *)
      echo "Unknown or unsupported task for covid19_twitter: $task_name" >&2
      exit 1
      ;;
  esac

  echo "=== Evaluating model=${model_name} task=${task_name} shots=${shots} ckpt=${ckpt_path} ==="
  conda run -p "$CONDA_ENV" --no-capture-output \
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

  for shots in "${SHOT_LIST[@]}"; do
    run_eval "$model_name" "$ckpt_path" neighbor_matching "$shots"
    run_eval "$model_name" "$ckpt_path" temporal_link_prediction "$shots"
  done
done < "$MODEL_LIST"
