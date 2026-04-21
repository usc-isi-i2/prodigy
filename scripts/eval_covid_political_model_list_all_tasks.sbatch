#!/bin/bash
#SBATCH --job-name=eval_covid_political_all
#SBATCH --output=/scratch1/eibl/logs/covid_political/%x_%j.out
#SBATCH --error=/scratch1/eibl/logs/covid_political/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/logs/covid_political
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: sbatch $0 <model_list.txt> [shots_csv]" >&2
  echo "Each non-empty, non-comment line: <model_name> <ckpt_path>  or  <ckpt_path>" >&2
  exit 1
fi

MODEL_LIST="$1"
SHOTS_CSV="${2:-0,1,3,10}"
if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"

COMMON_ARGS=(
  --dataset covid_political
  --root /home1/eibl/gfm/prodigy/data/data/covid_political/graphs
  --graph_filename retweet_graph.pt
  --midterm_feature_subset emb_only
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
  local target_dataset="covid_political"

  local task_tag=""
  local extra_args=()
  case "$task_name" in
    neighbor_matching)
      task_tag="nm"
      extra_args=(
        --task_name neighbor_matching
        --n_way 3
        --n_shots "$shots"
        --n_query 12
        --zero_shot "$([[ "$shots" == "0" ]] && echo True || echo False)"
        --dataset_len_cap 5000
        --val_len_cap 1000
        --test_len_cap 1000
        --epochs 12
        --eval_step 2000
        --checkpoint_step 2000
        --prefix "eval_${model_name}_to_${target_dataset}_${task_tag}_${shots}shot"
      )
      ;;
    classification)
      task_tag="pl"
      extra_args=(
        --task_name classification
        --ignore_label_embeddings False
        --linear_probe False
        --n_way 2
        --n_shots "$shots"
        --n_query 12
        --zero_shot "$([[ "$shots" == "0" ]] && echo True || echo False)"
        --dataset_len_cap 2000
        --val_len_cap 1000
        --test_len_cap 1000
        --epochs 12
        --eval_step 2000
        --checkpoint_step 2000
        --midterm_label_downsample 50:50
        --prefix "eval_${model_name}_to_${target_dataset}_${task_tag}_${shots}shot"
      )
      ;;
    *)
      echo "Unknown task: $task_name" >&2
      exit 1
      ;;
  esac

  echo "=== model=${model_name} task=${task_name} shots=${shots} ckpt=${ckpt_path} ==="
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

  read -r first second extra <<< "$line"
  if [[ -n "${second:-}" ]]; then
    model_name="$first"
    ckpt_path="$second"
    if [[ -n "${extra:-}" ]]; then
      echo "Invalid line: '$raw_line'" >&2; exit 1
    fi
  else
    ckpt_path="$first"
    model_name="$(basename "$(dirname "$ckpt_path")")"
  fi

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint not found: $ckpt_path" >&2; exit 1
  fi

  for shots in "${SHOT_LIST[@]}"; do
    run_eval "$model_name" "$ckpt_path" neighbor_matching "$shots"
    run_eval "$model_name" "$ckpt_path" classification "$shots"
  done
done < "$MODEL_LIST"
