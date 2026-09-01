#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"
: > "${OUT}"
for arm in MT NM NM_MT; do
  for dataset in covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended; do
    run="$(ls -dt "${STATE_DIR}/mtpilot_${arm}_${dataset}_"[0-9]*/ 2>/dev/null | head -n1 || true)"
    [[ -n "${run}" ]] || { echo "missing ${arm}/${dataset}" >&2; exit 1; }
    ckpt="${run}checkpoint/state_dict_900.ckpt"
    [[ -f "${ckpt}" ]] || { echo "missing ${ckpt}" >&2; exit 1; }
    echo "${arm}_${dataset} ${ckpt}" >> "${OUT}"
  done
done
echo "wrote ${OUT}" >&2
