#!/usr/bin/env bash
# Write model_list.txt pointing each trained rung at its MATCHED-40k checkpoint
# (state_dict_40000.ckpt) -- the budget the published ladder and the single-source
# matrix were evaluated at, so new rows drop straight into the table.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
#   GATE=1 STATE_DIR=... ./make_model_list.sh   # gate run only -> model_list_gate.txt
#   STEP=40000 ./make_model_list.sh             # override the eval step
#
# Prefixes are read from the generated configs, so this stays in sync with
# make_configs.py automatically. A run with no state_dict_<STEP>.ckpt is reported and
# skipped (WARN), and the script exits nonzero so a caller does not silently eval a
# partial set.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
STEP="${STEP:-40000}"

if [[ "${GATE:-0}" == "1" ]]; then
  CONFIG_GLOB="${SCRIPT_DIR}/train_gate_*.yaml"
  OUT="${SCRIPT_DIR}/model_list_gate.txt"
else
  CONFIG_GLOB="${SCRIPT_DIR}/train_ord*.yaml"
  OUT="${SCRIPT_DIR}/model_list.txt"
fi

PREFIXES=()
for cfg in ${CONFIG_GLOB}; do
  [[ -f "$cfg" ]] || continue
  p="$(awk '/^prefix:/{print $2; exit}' "$cfg")"
  [[ -n "$p" ]] && PREFIXES+=("$p")
done
[[ ${#PREFIXES[@]} -eq 0 ]] && { echo "no configs matched ${CONFIG_GLOB}" >&2; exit 2; }

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }

: > "${OUT}"
missing=0
for prefix in "${PREFIXES[@]}"; do
  d="$(run_dir_of "$prefix")"
  if [[ -z "$d" ]]; then
    echo "WARN: no run dir for ${prefix} under ${STATE_DIR}" >&2
    missing=1; continue
  fi
  ckpt="${d}checkpoint/state_dict_${STEP}.ckpt"
  if [[ -f "$ckpt" ]]; then
    echo "${prefix} ${ckpt}" >> "${OUT}"
  else
    echo "WARN: ${prefix}: no state_dict_${STEP}.ckpt in ${d}checkpoint/ (did it reach ${STEP} steps?)" >&2
    missing=1
  fi
done

echo "wrote ${OUT} (${#PREFIXES[@]} expected):" >&2
cat "${OUT}" >&2
if [[ "${missing}" == "1" ]]; then
  echo "ERROR: some models missing at step ${STEP} -- see WARN lines above." >&2
  exit 1
fi
