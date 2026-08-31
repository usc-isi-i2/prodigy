#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/social_specificity_pilot}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/social_specificity_pilot}"
FINAL_CORE_STATE_ROOT="${FINAL_CORE_STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core/state/final_core}"
FINAL_CORE_STAMP="${FINAL_CORE_STAMP:-20260807}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
GPUS="${GPUS:-2,3}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/validate_plan.py" --check-data

IFS=, read -r GPU2 GPU3 GPU_EXTRA <<< "$GPUS"
[[ -n "${GPU2:-}" && -n "${GPU3:-}" && -z "${GPU_EXTRA:-}" ]] || {
  echo "GPUS must contain exactly two ids, for example GPUS=2,3" >&2
  exit 2
}
for gpu in "$GPU2" "$GPU3"; do
  [[ "$gpu" == 2 || "$gpu" == 3 ]] || {
    echo "refusing non-owned GPU $gpu; use only GPUs 2 and 3" >&2
    exit 2
  }
done

train_one() {
  local dataset="$1" gpu="$2" config="$SCRIPT_DIR/${dataset}_nm.yaml"
  local run_name="socialspec_${dataset}_s0_${RUN_STAMP}"
  local checkpoint="$STATE_ROOT/$run_name/checkpoint/state_dict_2500.ckpt"
  local -a cmd=("$PYTHON" -u experiments/run_single_experiment.py
    --config "$config" --device "$gpu" --timestamp "$RUN_STAMP"
    --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT")
  if [[ -f "$checkpoint" ]]; then
    echo "SKIP complete $checkpoint"
  elif [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY train gpu=%s' "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
  elif [[ -e "$STATE_ROOT/$run_name" ]]; then
    echo "refusing incomplete existing run $STATE_ROOT/$run_name" >&2
    return 1
  else
    "${cmd[@]}" > "$LOG_ROOT/train/${run_name}.log" 2>&1
    [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; return 1; }
  fi
}

train_one cora "$GPU2" & p1=$!
train_one pubmed "$GPU3" & p2=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
(( status == 0 )) || exit "$status"

MODEL_LIST="$LOG_ROOT/launch/model_list_${RUN_STAMP}.txt"
UKR_CKPT="$FINAL_CORE_STATE_ROOT/finalcore_ss_ukr_rus_s0_${FINAL_CORE_STAMP}/checkpoint/state_dict_2500.ckpt"
FB_CKPT="$FINAL_CORE_STATE_ROOT/finalcore_ss_facebook_page_reference_s0_${FINAL_CORE_STAMP}/checkpoint/state_dict_2500.ckpt"
CORA_CKPT="$STATE_ROOT/socialspec_cora_s0_${RUN_STAMP}/checkpoint/state_dict_2500.ckpt"
PUBMED_CKPT="$STATE_ROOT/socialspec_pubmed_s0_${RUN_STAMP}/checkpoint/state_dict_2500.ckpt"

if [[ "$DRY_RUN" == 1 ]]; then
  printf '%s %s\n' ss_ukr_rus_twitter "$UKR_CKPT" ss_facebook_page_reference "$FB_CKPT" \
    ss_cora "$CORA_CKPT" ss_pubmed "$PUBMED_CKPT" > "$MODEL_LIST"
else
  for checkpoint in "$UKR_CKPT" "$FB_CKPT" "$CORA_CKPT" "$PUBMED_CKPT"; do
    [[ -f "$checkpoint" ]] || { echo "missing checkpoint $checkpoint" >&2; exit 1; }
  done
  printf '%s %s\n' ss_ukr_rus_twitter "$UKR_CKPT" ss_facebook_page_reference "$FB_CKPT" \
    ss_cora "$CORA_CKPT" ss_pubmed "$PUBMED_CKPT" > "$MODEL_LIST"
fi

eval_cmd=("$PYTHON" -u scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
  --model-list "$MODEL_LIST" --data-root "$DATA_ROOT"
  --datasets ukr_rus_twitter,facebook_page_reference,cora,pubmed
  --tasks nm --shots 3 --nm-n-way 30 --nm-dataset-len-cap 1
  --workers 2 --batch-size 4 --gpus "$GPUS" --seed 0
  -- --n_hop 2 --edge_view static_train --target_edge_view static_test
  --neighbor_matching_edge_split True --neighbor_sampling_hop_sizes 9,9
  --neighbor_sampling_node_limit 101 --log_dir "$LOG_ROOT/eval" --state_dir "$STATE_ROOT/eval")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY eval'; printf ' %q' "${eval_cmd[@]}"; printf '\n'
  exit 0
fi
"${eval_cmd[@]}" | tee "$LOG_ROOT/launch/eval_${RUN_STAMP}.log"
echo "SOCIAL_SPECIFICITY_PILOT_COMPLETE run_stamp=$RUN_STAMP log_root=$LOG_ROOT"
