#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PRETRAIN_STATE_ROOT="${PRETRAIN_STATE_ROOT:-${REPO_ROOT}/state/rq1_label_efficiency_loto/pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/state/rq1_label_efficiency_loto/adapt}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/rq1_label_efficiency_loto/adapt}"
SEED="${SEED:?set SEED to 0, 1, or 2}"
RUN_STAMP="${RUN_STAMP:-20260828}"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPUS <<< "$GPUS_TEXT"
[[ "$SEED" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[23]$ ]] || { echo "only Tucker GPUs 2 and 3 are authorized" >&2; exit 2; }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT"

mapfile -t target_rows < <("$PYTHON" "$SCRIPT_DIR/plan.py" | tail -n +2)
jobs=()
for row in "${target_rows[@]}"; do
  IFS=$'\t' read -r target _excluded _sources <<< "$row"
  checkpoint="$PRETRAIN_STATE_ROOT/rq1_loto_${target}_pretrain_s${SEED}_${RUN_STAMP}/state_dict"
  [[ -f "$checkpoint" ]] || { echo "missing pretraining checkpoint $checkpoint" >&2; exit 3; }
  for budget in 1 10 100 1000; do
    jobs+=("$target:scratch:$budget:")
    jobs+=("$target:pretrained:$budget:$checkpoint")
  done
done

worker() {
  local worker_index="$1" gpu="$2" item target arm budget checkpoint out log cmd index=0
  for item in "${jobs[@]}"; do
    if (( index % ${#GPUS[@]} == worker_index )); then
      IFS=: read -r target arm budget checkpoint <<< "$item"
      out="$OUTPUT_ROOT/seed_${SEED}/${target}/${budget}/${arm}"
      log="$LOG_ROOT/seed_${SEED}_${target}_${budget}_${arm}.log"
      if [[ -f "$out/result.json" ]]; then
        echo "[gpu $gpu] SKIP $target $budget $arm"
        ((index+=1)); continue
      fi
      cmd=("$PYTHON" -u -m scripts.experiments.setup.rq1_label_efficiency_loto.adapt
        --target "$target" --arm "$arm" --budget "$budget" --seed "$SEED"
        --output "$out" --device cuda:0)
      [[ "$arm" == pretrained ]] && cmd+=(--pretrained-checkpoint "$checkpoint")
      printf '[gpu %s] cmd=' "$gpu"; printf '%q ' "${cmd[@]}"; printf '\n'
      [[ "$DRY_RUN" == 1 ]] || CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1
    fi
    ((index+=1))
  done
}

pids=()
for index in "${!GPUS[@]}"; do worker "$index" "${GPUS[$index]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
