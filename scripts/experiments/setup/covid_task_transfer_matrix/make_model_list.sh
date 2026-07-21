#!/usr/bin/env bash
# Build model_list.txt for the COVID task-transfer matrix.
#
# Default rows use the latest best checkpoints:
#   task_transfer_covid_{nm,cl,fp}_smoke_*/state_dict
#
# Optional:
#   CHECKPOINT=latest  -> highest state_dict_<step>.ckpt under checkpoint/
#   CHECKPOINT=step:N  -> checkpoint/state_dict_N.ckpt
#
# Usage:
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"
CHECKPOINT="${CHECKPOINT:-best}"

run_dir_of() {
  ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true
}

latest_step_ckpt() {
  local run_dir="$1"
  ls "${run_dir}/checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' \
    | sort -n -k1,1 \
    | tail -n1 \
    | cut -d' ' -f2-
}

select_ckpt() {
  local prefix="$1"
  local run_dir
  run_dir="$(run_dir_of "${prefix}")"
  [[ -n "${run_dir}" ]] || return 1

  case "${CHECKPOINT}" in
    best)
      [[ -f "${run_dir}/state_dict" ]] && echo "${run_dir}/state_dict"
      ;;
    latest)
      latest_step_ckpt "${run_dir}"
      ;;
    step:*)
      local step="${CHECKPOINT#step:}"
      local ckpt="${run_dir}/checkpoint/state_dict_${step}.ckpt"
      [[ -f "${ckpt}" ]] && echo "${ckpt}"
      ;;
    *)
      echo "ERROR: unsupported CHECKPOINT=${CHECKPOINT}; use best, latest, or step:N" >&2
      return 2
      ;;
  esac
}

: > "${OUT}"

for task in nm cl fp; do
  prefix="task_transfer_covid_${task}_smoke"
  ckpt="$(select_ckpt "${prefix}" || true)"
  if [[ -z "${ckpt:-}" ]]; then
    echo "WARN: no ${CHECKPOINT} checkpoint found for ${prefix} under ${STATE_DIR}" >&2
    continue
  fi
  echo "${task} ${ckpt}" >> "${OUT}"
done

echo "wrote ${OUT}:" >&2
cat "${OUT}" >&2
