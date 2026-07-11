#!/usr/bin/env bash
# Launch the full 5-point partial-cross-source sweep, one detached tmux session per
# config. Each NM job is tiny (~2 GB, per-step-overhead bound), so co-locating on a
# busy-but-not-full GPU is fine; check `nvidia-smi` first and pass a GPU per config.
#
#   GPUS="0 1 2 3 2" ./run_all_train_tucker.sh          # default map (5 cfgs -> these GPUs)
#   GPUS="2 2 2 2 2" ./run_all_train_tucker.sh          # all on GPU 2
#   DRY_RUN=1 ./run_all_train_tucker.sh                 # print the tmux commands only
#
# Sessions: pxsrc_p000 pxsrc_p010 pxsrc_p025 pxsrc_p050 pxsrc_p100
# Watch one:  tmux attach -t pxsrc_p025      Kill all: tmux kill-session -t pxsrc_p0XX
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS=(p000_within.yaml p010.yaml p025.yaml p050.yaml p100_naive.yaml)
NAMES=(pxsrc_p000 pxsrc_p010 pxsrc_p025 pxsrc_p050 pxsrc_p100)
read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3 2}"
[[ ${#GPU_ARR[@]} -eq ${#CONFIGS[@]} ]] || { echo "need ${#CONFIGS[@]} GPUs in \$GPUS, got '${GPUS:-0 1 2 3 2}'" >&2; exit 2; }

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"; name="${NAMES[$i]}"; gpu="${GPU_ARR[$i]}"
  log="${SCRIPT_DIR}/run_logs/${cfg%.yaml}_launch.log"
  inner="export PATH=\"/home/mhchu/miniconda3/bin:\$PATH\"; bash ${SCRIPT_DIR}/train_tucker.sh ${cfg} --device ${gpu} > ${log} 2>&1"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "tmux new-session -d -s ${name} '${inner}'"
  else
    tmux new-session -d -s "${name}" "${inner}"
    echo "launched ${name}: ${cfg} on GPU ${gpu} -> ${log}"
  fi
done
