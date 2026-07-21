#!/usr/bin/env bash
# Experiment (b): build model_list_merged.txt from the merged-vs-single NM study
# checkpoints (final checkpoint of each run), to evaluate on TwiBot-20.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list_merged.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list_merged.txt"

# ukr/covid experiment (nm_transfer_matrix + nm_cross_source_shortcut) and
# covid/midterm experiment (nm_covid_midterm).
PREFIXES=(
  nm_matrix_ukr nm_matrix_covid nm_matrix_merged nm_xsrc_within_source
  nm_cm_covid nm_cm_midterm nm_cm_merged nm_cm_within nm_cm_within_balanced
)

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }
final_ckpt() {
  local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-
}

: > "${OUT}"
for prefix in "${PREFIXES[@]}"; do
  c="$(final_ckpt "$prefix")" || { echo "WARN: no run for $prefix" >&2; continue; }
  [[ -n "$c" ]] && echo "$prefix $c" >> "${OUT}" || echo "WARN: no ckpt for $prefix" >&2
done

echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
