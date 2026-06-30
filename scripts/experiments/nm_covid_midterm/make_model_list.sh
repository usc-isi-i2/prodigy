#!/usr/bin/env bash
# Write model_list.txt for the cov/mid eval.
#
# Single-source runs train 60k steps (final checkpoint ~50k due to the 0-indexed
# checkpoint cadence); merged runs train 120k (final ~110k). For a FAIR comparison
# we eval each merged model at TWO checkpoints:
#   *_match : the same step count as the single-source models  -> matched total compute
#   *_full  : the merged model's final checkpoint              -> matched per-domain exposure
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }
final_ckpt() {  # prefix -> highest-step ckpt path
  # Sort by the numeric step in the BASENAME (the full path has underscores in the
  # run-dir timestamp, so `sort -t_ -k3` would sort by the wrong field).
  local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-
}
step_of() { basename "$1" | sed -E 's/state_dict_([0-9]+)\.ckpt/\1/'; }
ckpt_at() {  # prefix step -> path (if exists)
  local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  local p="${d}checkpoint/state_dict_$2.ckpt"; [[ -f "$p" ]] && echo "$p"
}

: > "${OUT}"

# Single-source: final checkpoint.
for prefix in nm_cm_midterm nm_cm_covid; do
  c="$(final_ckpt "$prefix")" || { echo "WARN: no run for $prefix" >&2; continue; }
  echo "$prefix $c" >> "${OUT}"
done

# Matched-compute step = single-source final step (use covid's; midterm should match).
match_step="$(step_of "$(final_ckpt nm_cm_covid)")"
echo "matched-compute step = ${match_step}" >&2

# Merged variants: emit *_match (at match_step) and *_full (final).
for prefix in nm_cm_merged nm_cm_within nm_cm_within_balanced; do
  full="$(final_ckpt "$prefix")" || { echo "WARN: no run for $prefix" >&2; continue; }
  matched="$(ckpt_at "$prefix" "$match_step" || true)"
  if [[ -n "$matched" ]]; then echo "${prefix}_match $matched" >> "${OUT}";
  else echo "WARN: no ${prefix} checkpoint at step ${match_step}" >&2; fi
  echo "${prefix}_full $full" >> "${OUT}"
done

echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
