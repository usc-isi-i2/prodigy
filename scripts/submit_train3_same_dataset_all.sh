#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dataset> <model_list.txt>" >&2
  echo "Datasets: midterm | covid19_twitter" >&2
  exit 1
fi

SOURCE_DATASET="$1"
MODEL_LIST="$2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

[[ -f "$MODEL_LIST" ]] || { echo "Model list not found: $MODEL_LIST" >&2; exit 1; }

case "$SOURCE_DATASET" in
  midterm|covid19_twitter) ;;
  *)
    echo "Unsupported dataset: $SOURCE_DATASET" >&2
    exit 1
    ;;
esac

DRY_RUN="${DRY_RUN:-0}"
NM_CKPT=""
LP_CKPT=""
PL_CKPT=""

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  read -r model_name ckpt_path extra <<< "$line"
  [[ -n "${extra:-}" ]] && { echo "Invalid line in ${MODEL_LIST}: '$raw_line'" >&2; exit 1; }
  [[ -n "${model_name:-}" && -n "${ckpt_path:-}" ]] || { echo "Invalid line in ${MODEL_LIST}: '$raw_line'" >&2; exit 1; }

  case "$model_name" in
    "${SOURCE_DATASET}_nm") NM_CKPT="$ckpt_path" ;;
    "${SOURCE_DATASET}_lp") LP_CKPT="$ckpt_path" ;;
    "${SOURCE_DATASET}_pl") PL_CKPT="$ckpt_path" ;;
  esac
done < "$MODEL_LIST"

[[ -n "$NM_CKPT" ]] || { echo "Missing ${SOURCE_DATASET}_nm entry in ${MODEL_LIST}" >&2; exit 1; }
[[ -n "$LP_CKPT" ]] || { echo "Missing ${SOURCE_DATASET}_lp entry in ${MODEL_LIST}" >&2; exit 1; }
[[ -n "$PL_CKPT" ]] || { echo "Missing ${SOURCE_DATASET}_pl entry in ${MODEL_LIST}" >&2; exit 1; }

submit_job() {
  local source_task="$1"
  local target_task="$2"
  local source_model="${SOURCE_DATASET}_${source_task}"
  local ckpt=""

  case "$source_task" in
    nm) ckpt="$NM_CKPT" ;;
    lp) ckpt="$LP_CKPT" ;;
    pl) ckpt="$PL_CKPT" ;;
    *)
      echo "Unknown source_task='$source_task'" >&2
      exit 1
      ;;
  esac

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY: sbatch --job-name=train3_${source_model}_${target_task} --export=ALL,SOURCE_MODEL=${source_model},SOURCE_DATASET=${SOURCE_DATASET},TARGET_TASK=${target_task},CKPT_PATH=${ckpt} ${SCRIPT_DIR}/train3_same_dataset_single_task.sbatch"
    return
  fi

  sbatch \
    --job-name="train3_${source_model}_${target_task}" \
    --export="ALL,SOURCE_MODEL=${source_model},SOURCE_DATASET=${SOURCE_DATASET},TARGET_TASK=${target_task},CKPT_PATH=${ckpt}" \
    "${SCRIPT_DIR}/train3_same_dataset_single_task.sbatch"
}

submit_job nm temporal_link_prediction
submit_job nm classification

submit_job lp neighbor_matching
submit_job lp classification

submit_job pl neighbor_matching
submit_job pl temporal_link_prediction
