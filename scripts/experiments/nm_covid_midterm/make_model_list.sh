#!/usr/bin/env bash
# Write model_list.txt pointing at each regime's final checkpoint.
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

: > "${OUT}"
for prefix in nm_cm_midterm nm_cm_covid nm_cm_merged nm_cm_within nm_cm_within_balanced; do
  run_dir="$(ls -dt "${STATE_DIR}/${prefix}_"*/ 2>/dev/null | head -n1 || true)"
  [[ -z "${run_dir}" ]] && { echo "WARN: no run dir for ${prefix}" >&2; continue; }
  ckpt="$(ls "${run_dir}checkpoint/"state_dict_*.ckpt 2>/dev/null | sort -t_ -k3 -n | tail -n1 || true)"
  [[ -z "${ckpt}" ]] && { echo "WARN: no checkpoint in ${run_dir}checkpoint/" >&2; continue; }
  echo "${prefix} ${ckpt}" >> "${OUT}"
done
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
