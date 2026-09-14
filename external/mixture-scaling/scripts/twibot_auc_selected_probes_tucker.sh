#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CFG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPL="${ROOT}/state/twibot_strict_pilot/splits"
PRE="${ROOT}/state/twibot_strict_pilot/pretrain"
OUT="${ROOT}/results/twibot_strict_pilot/raw_auc_selected"
LOG="${ROOT}/log/twibot_strict_pilot/auc_selected"
export PYTHONPATH="${ROOT}/src"
mkdir -p "${OUT}" "${LOG}"

probe_one() {
  local name="$1" checkpoint="$2" seed="$3" gpu="$4"
  "${PY}" -u -m mixture_scaling.probe_strict \
    --config "${CFG}" --split-root "${SPL}" --checkpoint "${checkpoint}" \
    --target twibot20 --device "${gpu}" --seed "${seed}" --selection-metric auc \
    --output "${OUT}/probe_${name}.json" >"${LOG}/probe_${name}.log" 2>&1
}

run_queue() {
  local gpu="$1"; shift
  local spec name checkpoint seed running=0 status=0
  for spec in "$@"; do
    IFS='|' read -r name checkpoint seed <<<"${spec}"
    probe_one "${name}" "${checkpoint}" "${seed}" "${gpu}" &
    ((running+=1))
    if (( running >= 4 )); then
      wait -n || status=1
      running=$((running - 1))
    fi
  done
  while (( running > 0 )); do wait -n || status=1; running=$((running - 1)); done
  return "${status}"
}

gpu2=(
  "scratch_s0|scratch|0" "twibot_target_s0|${PRE}/twibot_target_s0/best.pt|0"
  "election_single_s0|${PRE}/election_single_s0/best.pt|0" "all_non_target_s0|${PRE}/all_non_target_s0/best.pt|0"
  "covid_single_s0|${PRE}/covid_single_s0/best.pt|0" "ukrrus_single_s0|${PRE}/ukrrus_single_s0/best.pt|0"
  "facebook_single_s0|${PRE}/facebook_single_s0/best.pt|0" "cora_single_s0|${PRE}/cora_single_s0/best.pt|0"
  "pubmed_single_s0|${PRE}/pubmed_single_s0/best.pt|0"
)
gpu3=(
  "scratch_s1|scratch|1" "scratch_s2|scratch|2"
  "twibot_target_s1|${PRE}/twibot_target_s1/best.pt|1" "twibot_target_s2|${PRE}/twibot_target_s2/best.pt|2"
  "election_single_s1|${PRE}/election_single_s1/best.pt|1" "election_single_s2|${PRE}/election_single_s2/best.pt|2"
  "all_non_target_s1|${PRE}/all_non_target_s1/best.pt|1" "all_non_target_s2|${PRE}/all_non_target_s2/best.pt|2"
)

run_queue 2 "${gpu2[@]}" & q2=$!
run_queue 3 "${gpu3[@]}" & q3=$!

"${PY}" -m mixture_scaling.raw_logistic_baseline \
  --config "${CFG}" --split-root "${SPL}" --target twibot20 --seed 0 --selection-metric auc \
  --output "${OUT}/raw_logistic.json" >"${LOG}/raw_logistic.log" 2>&1 & qr=$!

status=0
wait "${q2}" || status=1
wait "${q3}" || status=1
wait "${qr}" || status=1
(( status == 0 )) || { echo "one or more AUC-selected probes failed" >&2; exit 1; }
echo "complete $(date -u +%FT%TZ)"
