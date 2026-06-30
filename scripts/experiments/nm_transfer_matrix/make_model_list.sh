#!/usr/bin/env bash
# Build model_list.txt pointing at the FINAL checkpoint of each trained source.
# Picks the newest run dir per prefix under STATE_DIR and its highest-step ckpt.
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

: > "${OUT}"
for prefix in nm_matrix_ukr nm_matrix_covid nm_matrix_merged; do
  run_dir="$(ls -dt "${STATE_DIR}/${prefix}_"*/ 2>/dev/null | head -n1 || true)"
  if [[ -z "${run_dir}" ]]; then
    echo "WARN: no run dir for ${prefix} under ${STATE_DIR}" >&2
    continue
  fi
  ckpt="$(ls "${run_dir}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2- || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "WARN: no checkpoint in ${run_dir}checkpoint/" >&2
    continue
  fi
  echo "${prefix} ${ckpt}" >> "${OUT}"
done

echo "wrote ${OUT}:" >&2
cat "${OUT}" >&2
