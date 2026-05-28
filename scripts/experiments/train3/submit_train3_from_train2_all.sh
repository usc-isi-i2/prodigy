#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dataset> <train2_model_list.txt>" >&2
  echo "Datasets: midterm | covid19_twitter" >&2
  exit 1
fi

SOURCE_DATASET="$1"
MODEL_LIST="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$MODEL_LIST" ]] || { echo "Model list not found: $MODEL_LIST" >&2; exit 1; }

case "$SOURCE_DATASET" in
  midterm|covid19_twitter) ;;
  *)
    echo "Unsupported dataset: $SOURCE_DATASET" >&2
    exit 1
    ;;
esac

dataset_key() {
  case "$1" in
    midterm) echo "midterm" ;;
    covid) echo "covid19_twitter" ;;
    ukr_rus) echo "ukr_rus_twitter" ;;
    *)
      echo ""
      ;;
  esac
}

submit_job() {
  local source_model="$1"
  local ckpt="$2"
  local target_task="$3"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY: sbatch --job-name=train3_${source_model}_${target_task} --export=ALL,SOURCE_MODEL=${source_model},SOURCE_DATASET=${SOURCE_DATASET},TARGET_TASK=${target_task},CKPT_PATH=${ckpt} ${SCRIPT_DIR}/train3_same_dataset_single_task.sbatch"
    return
  fi

  sbatch \
    --job-name="train3_${source_model}_${target_task}" \
    --export="ALL,SOURCE_MODEL=${source_model},SOURCE_DATASET=${SOURCE_DATASET},TARGET_TASK=${target_task},CKPT_PATH=${ckpt}" \
    "${SCRIPT_DIR}/train3_same_dataset_single_task.sbatch"
}

found_any=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  read -r model_name ckpt_path extra <<< "$line"
  [[ -n "${extra:-}" ]] && { echo "Invalid line in ${MODEL_LIST}: '$raw_line'" >&2; exit 1; }

  if [[ ! "$model_name" =~ ^(exp[0-9]+)_([a-z_]+)_(nm|lp|pl)_to_([a-z_]+)_(nm|lp|pl)$ ]]; then
    continue
  fi

  exp_id="${BASH_REMATCH[1]}"
  start_dataset_token="${BASH_REMATCH[2]}"
  start_task="${BASH_REMATCH[3]}"
  final_dataset_token="${BASH_REMATCH[4]}"
  final_task="${BASH_REMATCH[5]}"

  final_dataset="$(dataset_key "$final_dataset_token")"
  start_dataset="$(dataset_key "$start_dataset_token")"

  [[ "$final_dataset" == "$SOURCE_DATASET" ]] || continue

  found_any=1
  source_model="${exp_id}_${start_dataset}_${start_task}_to_${final_dataset}_${final_task}"

  case "$final_task" in
    nm)
      submit_job "$source_model" "$ckpt_path" temporal_link_prediction
      submit_job "$source_model" "$ckpt_path" classification
      ;;
    lp)
      submit_job "$source_model" "$ckpt_path" neighbor_matching
      submit_job "$source_model" "$ckpt_path" classification
      ;;
    pl)
      submit_job "$source_model" "$ckpt_path" neighbor_matching
      submit_job "$source_model" "$ckpt_path" temporal_link_prediction
      ;;
    *)
      echo "Unexpected final task '$final_task' in $model_name" >&2
      exit 1
      ;;
  esac
done < "$MODEL_LIST"

if [[ "$found_any" != "1" ]]; then
  echo "No train2 checkpoints in ${MODEL_LIST} end on dataset '${SOURCE_DATASET}'" >&2
  exit 1
fi
