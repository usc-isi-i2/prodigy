#!/bin/bash
set -euo pipefail

# Reads train2_all_models.txt, finds NM->NM experiments, and trains each checkpoint
# on its third unseen train dataset plus the 3 social LLM datasets.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_LIST="${1:-${SCRIPT_DIR}/model_lists/train2_all_models.txt}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$MODEL_LIST" ]] || { echo "Model list not found: $MODEL_LIST" >&2; exit 1; }

SOCIAL_LLM_DATASETS=(election2020 ukr_rus_suspended covid_political)

# Maps a dataset token in the experiment name to the canonical dataset key.
dataset_key() {
  case "$1" in
    midterm)    echo "midterm" ;;
    covid)      echo "covid19_twitter" ;;
    ukr_rus)    echo "ukr_rus_twitter" ;;
    *)          echo "" ;;
  esac
}

# Returns the third train dataset (the one not in d1 or d2).
third_dataset() {
  local d1="$1" d2="$2"
  for candidate in midterm covid19_twitter ukr_rus_twitter; do
    [[ "$candidate" == "$d1" || "$candidate" == "$d2" ]] && continue
    echo "$candidate"
    return
  done
  echo ""
}

submit_job() {
  local source_model="$1" ckpt="$2" target_dataset="$3"
  local job_name="train3_${source_model}_to_${target_dataset}_nm"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY: sbatch --job-name=${job_name} SOURCE_MODEL=${source_model} TARGET_DATASET=${target_dataset} CKPT_PATH=${ckpt}"
    return
  fi

  sbatch \
    --job-name="${job_name}" \
    --export="ALL,SOURCE_MODEL=${source_model},CKPT_PATH=${ckpt},TARGET_DATASET=${target_dataset}" \
    "${SCRIPT_DIR}/train3_nm_single_task.sbatch"
}

mkdir -p /home1/eibl/gfm/prodigy/logs

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  read -r model_name ckpt_path <<< "$line"
  [[ -z "${ckpt_path:-}" ]] && { echo "Skipping malformed line: '$raw_line'" >&2; continue; }

  # Only process NM->NM experiments: name contains _nm_to_*_nm
  [[ "$model_name" =~ _nm_to_([a-z_]+)_nm$ ]] || continue

  # Extract the two dataset tokens from the name, e.g. exp1_midterm_nm_to_covid_nm
  # Pattern: exp{N}_{d1}_nm_to_{d2}_nm
  if [[ "$model_name" =~ ^exp[0-9]+_([a-z_]+)_nm_to_([a-z_]+)_nm$ ]]; then
    d1_token="${BASH_REMATCH[1]}"
    d2_token="${BASH_REMATCH[2]}"
  else
    echo "Could not parse dataset tokens from '${model_name}', skipping" >&2
    continue
  fi

  d1="$(dataset_key "$d1_token")"
  d2="$(dataset_key "$d2_token")"

  if [[ -z "$d1" || -z "$d2" ]]; then
    echo "Unknown dataset token in '${model_name}' (d1='${d1_token}' d2='${d2_token}'), skipping" >&2
    continue
  fi

  d3="$(third_dataset "$d1" "$d2")"
  if [[ -z "$d3" ]]; then
    echo "Could not determine third dataset for '${model_name}', skipping" >&2
    continue
  fi

  echo "=== ${model_name}: d1=${d1} d2=${d2} -> train3 on: ${d3} + social LLM datasets ==="

  # Third train dataset
  submit_job "$model_name" "$ckpt_path" "$d3"

  # 3 social LLM datasets
  for social in "${SOCIAL_LLM_DATASETS[@]}"; do
    submit_job "$model_name" "$ckpt_path" "$social"
  done

done < "$MODEL_LIST"
