#!/usr/bin/env bash
# Resolve every rung to the exact matched-40k checkpoint. Refuse partial lists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
STEP="${STEP:-40000}"
OUT="${OUT:-${SCRIPT_DIR}/model_list.txt}"
PREFIXES=(
  nm_ladder_gatv2_r1_1src nm_ladder_gatv2_r2_2src
  nm_ladder_gatv2_r3_3src nm_ladder_gatv2_r4_4src
  nm_ladder_gatv2_r5_5src nm_ladder_gatv2_r6_6src
  nm_ladder_gatv2_r7_7src nm_ladder_gatv2_r8_8src
)

if [[ "${STEP}" != "40000" ]]; then
  echo "refusing STEP=${STEP}: this experiment is registered at matched-40k" >&2
  exit 2
fi

tmp="${OUT}.tmp"
: > "${tmp}"
missing=0
for prefix in "${PREFIXES[@]}"; do
  run_dir=""
  while IFS= read -r -d '' candidate; do
    if [[ -z "${run_dir}" || "${candidate}" -nt "${run_dir}" ]]; then
      run_dir="${candidate}"
    fi
  done < <(find "${STATE_DIR}" -maxdepth 1 -type d -name "${prefix}_*" -print0 2>/dev/null)
  if [[ -z "${run_dir}" ]]; then
    echo "MISSING run directory for ${prefix} under ${STATE_DIR}" >&2
    missing=1
    continue
  fi
  checkpoint="${run_dir}/checkpoint/state_dict_40000.ckpt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "MISSING ${checkpoint}" >&2
    missing=1
    continue
  fi
  printf '%s %s\n' "${prefix}" "${checkpoint}" >> "${tmp}"
done

if [[ "${missing}" == "1" ]]; then
  rm -f "${tmp}"
  echo "model list not written: all eight matched-40k checkpoints are required" >&2
  exit 1
fi
mv "${tmp}" "${OUT}"
echo "wrote ${OUT} with 8 matched-40k checkpoints" >&2
