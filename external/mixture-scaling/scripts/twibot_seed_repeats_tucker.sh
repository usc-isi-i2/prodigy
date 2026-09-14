#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PY="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CFG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPL="${ROOT}/state/twibot_strict_pilot/splits"
PRE="${ROOT}/state/twibot_strict_pilot/pretrain"
OUT="${ROOT}/results/twibot_strict_pilot/raw_seed_repeats"
LOG="${ROOT}/log/twibot_strict_pilot/seed_repeats"
export PYTHONPATH="${ROOT}/src"
mkdir -p "${OUT}" "${LOG}"

run_pretrained_arm() {
  local name="$1" sources="$2" seed="$3" gpu="$4"
  local run_id="${name}_s${seed}"
  "${PY}" -u -m mixture_scaling.train_strict \
    --config "${CFG}" --split-root "${SPL}" --sources "${sources}" \
    --run-id "${run_id}" --device "${gpu}" --seed "${seed}" --output-root "${PRE}" \
    >"${LOG}/pretrain_${run_id}.log" 2>&1
  RUN_ID="${run_id}" SOURCES="${sources}" SEED="${seed}" PRE="${PRE}" "${PY}" -c '
import os
from pathlib import Path
from mixture_scaling.audit_strict_pretrain import audit_run
print(audit_run(Path(os.environ["PRE"]) / os.environ["RUN_ID"], os.environ["SOURCES"].split(","), int(os.environ["SEED"])))
' >"${LOG}/audit_${run_id}.log"
  "${PY}" -u -m mixture_scaling.probe_strict \
    --config "${CFG}" --split-root "${SPL}" --checkpoint "${PRE}/${run_id}/best.pt" \
    --target twibot20 --device "${gpu}" --seed "${seed}" \
    --output "${OUT}/probe_${run_id}.json" >"${LOG}/probe_${run_id}.log" 2>&1
}

run_scratch_arm() {
  local seed="$1" gpu="$2"
  "${PY}" -u -m mixture_scaling.probe_strict \
    --config "${CFG}" --split-root "${SPL}" --checkpoint scratch \
    --target twibot20 --device "${gpu}" --seed "${seed}" \
    --output "${OUT}/probe_scratch_s${seed}.json" >"${LOG}/probe_scratch_s${seed}.log" 2>&1
}

# Four independent pipelines per allocated GPU. Each pretrained arm proceeds
# directly from convergence to audit to probe; there is no cross-arm barrier.
run_pretrained_arm twibot_target twibot20 1 2 & p1=$!
run_pretrained_arm election_single election2020 1 2 & p2=$!
run_pretrained_arm all_non_target covid_political,ukr_rus_suspended,election2020,facebook_page_reference,cora,pubmed 1 2 & p3=$!
run_scratch_arm 1 2 & p4=$!

run_pretrained_arm twibot_target twibot20 2 3 & p5=$!
run_pretrained_arm election_single election2020 2 3 & p6=$!
run_pretrained_arm all_non_target covid_political,ukr_rus_suspended,election2020,facebook_page_reference,cora,pubmed 2 3 & p7=$!
run_scratch_arm 2 3 & p8=$!

status=0
for pid in "${p1}" "${p2}" "${p3}" "${p4}" "${p5}" "${p6}" "${p7}" "${p8}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || { echo "one or more seed-repeat arms failed" >&2; exit 1; }
echo "complete $(date -u +%FT%TZ)"
