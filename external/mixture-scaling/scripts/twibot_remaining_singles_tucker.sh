#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CFG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPL="${ROOT}/state/twibot_strict_pilot/splits"
PRE="${ROOT}/state/twibot_strict_pilot/pretrain"
OUT="${ROOT}/results/twibot_strict_pilot/raw_remaining_singles"
LOG="${ROOT}/log/twibot_strict_pilot"
export PYTHONPATH="${ROOT}/src"
mkdir -p "${OUT}" "${LOG}"

train_one() {
  local name="$1" source="$2" gpu="$3"
  "${PY}" -u -m mixture_scaling.train_strict \
    --config "${CFG}" --split-root "${SPL}" --sources "${source}" \
    --run-id "${name}_single_s0" --device "${gpu}" --seed 0 --output-root "${PRE}" \
    >"${LOG}/pretrain_${name}_single_s0.log" 2>&1
}

train_one covid covid_political 2 & p1=$!
train_one election election2020 2 & p2=$!
train_one pubmed pubmed 3 & p3=$!
wait "${p1}" "${p2}" "${p3}"

"${PY}" -c '
from pathlib import Path
from mixture_scaling.audit_strict_pretrain import audit_run
root = Path("state/twibot_strict_pilot/pretrain")
cases = [
    ("covid_single_s0", ["covid_political"]),
    ("election_single_s0", ["election2020"]),
    ("pubmed_single_s0", ["pubmed"]),
]
print([audit_run(root / name, sources, 0) for name, sources in cases])
' >"${LOG}/remaining_singles_audit.log"

probe_one() {
  local name="$1" gpu="$2"
  "${PY}" -u -m mixture_scaling.probe_strict \
    --config "${CFG}" --split-root "${SPL}" \
    --checkpoint "${PRE}/${name}_single_s0/best.pt" \
    --target twibot20 --device "${gpu}" --seed 0 \
    --output "${OUT}/probe_${name}_single.json" \
    >"${LOG}/probe_${name}_single.log" 2>&1
}

probe_one covid 2 & q1=$!
probe_one election 2 & q2=$!
probe_one pubmed 3 & q3=$!
wait "${q1}" "${q2}" "${q3}"
echo "complete $(date -u +%FT%TZ)"
