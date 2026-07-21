#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${OUT:-${SCRIPT_DIR}/model_list_matched.txt}"
MATCH_STEP="${MATCH_STEP:-50000}"

run_dir_of() {
  ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true
}

final_ckpt() {
  local d
  d="$(run_dir_of "$1")"
  [[ -z "${d}" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' \
    | sort -n -k1,1 \
    | tail -n1 \
    | cut -d' ' -f2-
}

ckpt_at() {
  local d p
  d="$(run_dir_of "$1")"
  [[ -z "${d}" ]] && return 1
  p="${d}checkpoint/state_dict_$2.ckpt"
  [[ -f "${p}" ]] && echo "${p}"
}

write_final() {
  local name="$1"
  local prefix="$2"
  local ckpt
  ckpt="$(final_ckpt "${prefix}")" || {
    echo "WARN: no run for ${prefix}" >&2
    return 0
  }
  [[ -n "${ckpt}" ]] && printf "%s %s\n" "${name}" "${ckpt}" >> "${OUT}"
}

write_matched() {
  local name="$1"
  local prefix="$2"
  local ckpt
  ckpt="$(ckpt_at "${prefix}" "${MATCH_STEP}")" || {
    echo "WARN: no ${prefix} checkpoint at step ${MATCH_STEP}" >&2
    return 0
  }
  [[ -n "${ckpt}" ]] && printf "%s %s\n" "${name}" "${ckpt}" >> "${OUT}"
}

: > "${OUT}"

write_final nm_matrix_ukr nm_matrix_ukr
write_final nm_matrix_covid nm_matrix_covid
write_matched nm_matrix_merged_match nm_matrix_merged
write_matched nm_xsrc_within_source_match nm_xsrc_within_source

write_final nm_cm_midterm nm_cm_midterm
write_final nm_cm_covid nm_cm_covid
write_matched nm_cm_merged_match nm_cm_merged
write_matched nm_cm_within_match nm_cm_within
write_matched nm_cm_within_balanced_match nm_cm_within_balanced

write_final nm_twibot20 nm_twibot20

echo "wrote ${OUT}:" >&2
cat "${OUT}" >&2
