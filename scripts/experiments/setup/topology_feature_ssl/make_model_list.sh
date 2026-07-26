#!/usr/bin/env bash
# Write model_list.txt for the topology_feature_ssl eval sweep: one line per arm
# that has a trained checkpoint, "<ARM> <final_ckpt>", keyed by the ARM name so
# the benchmark CSVs carry model=B0/B1/... and reading-chain deltas are direct
# table subtractions.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
#   ARMS="B0 B1" STATE_DIR=... ./make_model_list.sh   # restrict to some arms
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"
ARMS="${ARMS:-B0 B1 E1 E2 E3 E4}"

run_dir_of() { ls -dt "${STATE_DIR}/tfssl_$1_"*/ 2>/dev/null | head -n1 || true; }
final_ckpt() {  # ARM -> highest-step ckpt path (numeric sort on basename step)
  local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-
}

: > "${OUT}"
for arm in ${ARMS}; do
  c="$(final_ckpt "$arm")" || { echo "WARN: no run for tfssl_${arm}_*" >&2; continue; }
  [[ -z "$c" ]] && { echo "WARN: no checkpoint for tfssl_${arm}_*" >&2; continue; }
  echo "${arm} ${c}" >> "${OUT}"
done

[[ -s "${OUT}" ]] || { echo "ERROR: no checkpoints found under ${STATE_DIR}/tfssl_*" >&2; exit 1; }
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
