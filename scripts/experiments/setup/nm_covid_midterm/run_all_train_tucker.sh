#!/usr/bin/env bash
# Train all 5 cov/mid regimes across GPUs 0-3 (only those are available).
# Schedule (one lane per GPU; lanes run in parallel, configs within a lane run
# sequentially):
#   GPU 0: midterm_nm -> covid_nm        (two fast single-source runs)
#   GPU 1: merged_nm                     (naive)
#   GPU 2: merged_within_nm              (within, proportional)
#   GPU 3: merged_within_balanced_nm     (within, balanced)
#
# Run it INSIDE tmux so it survives closing your laptop:
#   tmux new -s cm
#   ./run_all_train_tucker.sh
#   (Ctrl-b d to detach; tmux attach -t cm to return)
#
# Per-config logs go to run_logs/. DRY_RUN=1 previews. Override GPUs with
#   GPUS="0 1 2 3" ./run_all_train_tucker.sh
# Extra args are forwarded to every run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

read -r -a G <<< "${GPUS:-0 1 2 3}"
if [[ ${#G[@]} -lt 4 ]]; then echo "need 4 GPU ids, got '${GPUS:-0 1 2 3}'" >&2; exit 2; fi

# lane i runs on GPU G[i]; space-separated configs run sequentially in the lane
LANES=(
  "midterm_nm.yaml covid_nm.yaml"
  "merged_nm.yaml"
  "merged_within_nm.yaml"
  "merged_within_balanced_nm.yaml"
)

declare -a PIDS LANEGPU
for i in 0 1 2 3; do
  gpu="${G[$i]}"
  configs="${LANES[$i]}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    for cfg in $configs; do DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "$cfg" --device "$gpu" "$@"; done
    continue
  fi
  echo "lane GPU ${gpu}: ${configs}" >&2
  (
    for cfg in $configs; do
      echo "[GPU ${gpu}] starting ${cfg}" >&2
      "${SCRIPT_DIR}/train_nm_tucker.sh" "$cfg" --device "$gpu" "$@"
    done
  ) &
  PIDS+=("$!"); LANEGPU+=("$gpu")
done

[[ "${DRY_RUN:-0}" == "1" ]] && exit 0

rc=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "OK   lane GPU ${LANEGPU[$i]}" >&2
  else
    echo "FAIL lane GPU ${LANEGPU[$i]} (see run_logs/)" >&2; rc=1
  fi
done
exit "${rc}"
