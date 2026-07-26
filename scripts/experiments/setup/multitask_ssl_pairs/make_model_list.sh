#!/usr/bin/env bash
# Write model_list.txt for the multitask_ssl_pairs eval sweep: one line per arm
# with a trained checkpoint, "<ARM> <final_ckpt>", keyed by ARM so the benchmark
# CSVs carry model=NMCL/NMFP/CLFP (and, for the merged 7-arm table, NM/CL/FP/MIX).
#
# Arm -> checkpoint-prefix mapping (a run dir is <STATE_DIR>/<prefix><ARM>_<ts>/):
#   NMCL NMFP CLFP  -> mtp_   (this experiment;      searched in STATE_DIR)
#   NM CL FP MIX    -> mtr_   (multitask_ssl_rotation; searched in ROTATION_STATE_DIR,
#                             falling back to STATE_DIR if that is unset)
# This lets ONE eval sweep score all 7 arms together under identical conditions —
# the cleanest way to merge the pairs with the single/MIX controls (§ EXECUTION.md).
#
#   # pairs only (default):
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
#   # full 7-arm merge (pairs here + rotation arms from the mtr worktree):
#   ARMS="NMCL NMFP CLFP NM CL FP MIX" \
#     STATE_DIR=/dataMeR1/phil/gfm/prodigy/state \
#     ROTATION_STATE_DIR=/dataMeR1/phil/gfm/prodigy-mtr/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
ROTATION_STATE_DIR="${ROTATION_STATE_DIR:-${STATE_DIR}}"
OUT="${SCRIPT_DIR}/model_list.txt"
ARMS="${ARMS:-NMCL NMFP CLFP}"

prefix_of() { case "$1" in NM|CL|FP|MIX) echo "mtr";; *) echo "mtp";; esac; }
state_dir_of() { case "$1" in NM|CL|FP|MIX) echo "${ROTATION_STATE_DIR}";; *) echo "${STATE_DIR}";; esac; }

run_dir_of() {  # ARM -> newest matching run dir
  local p d; p="$(prefix_of "$1")"; d="$(state_dir_of "$1")"
  ls -dt "${d}/${p}_$1_"*/ 2>/dev/null | head -n1 || true
}
final_ckpt() {  # ARM -> highest-step ckpt path (numeric sort on basename step)
  local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-
}

: > "${OUT}"
for arm in ${ARMS}; do
  c="$(final_ckpt "$arm")" || { echo "WARN: no run for $(prefix_of "$arm")_${arm}_*" >&2; continue; }
  [[ -z "$c" ]] && { echo "WARN: no checkpoint for $(prefix_of "$arm")_${arm}_*" >&2; continue; }
  echo "${arm} ${c}" >> "${OUT}"
done

[[ -s "${OUT}" ]] || { echo "ERROR: no checkpoints found (STATE_DIR=${STATE_DIR}, ROTATION_STATE_DIR=${ROTATION_STATE_DIR})" >&2; exit 1; }
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
