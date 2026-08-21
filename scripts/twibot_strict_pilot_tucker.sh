#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CONFIG="${ROOT}/configs/twibot_strict_pilot.yaml"
BASE="${ROOT}/state/twibot_strict_pilot"
SPLITS="${BASE}/splits"
PRETRAIN="${BASE}/pretrain"
FINETUNE="${BASE}/finetune"
RESULTS="${ROOT}/results/twibot_strict_pilot/raw"
LOGS="${ROOT}/log/twibot_strict_pilot"
mkdir -p "${LOGS}" "${RESULTS}" "${PRETRAIN}" "${FINETUNE}"

export PYTHONPATH="${ROOT}/src"
if [[ ! -f "${SPLITS}/manifest.json" ]]; then
  "${PYTHON_BIN}" -m mixture_scaling.prepare_strict_splits \
    --config "${CONFIG}" --output-root "${SPLITS}" >"${LOGS}/prepare_splits.log" 2>&1
fi

mapfile -t PRETRAIN_ROWS < <(tail -n +2 "${ROOT}/manifests/twibot_strict_pretrain.tsv")
pretrain_worker() {
  local worker="$1" gpu="$2" index=0 row run_id kind target sources seed
  for row in "${PRETRAIN_ROWS[@]}"; do
    if (( index % 4 == worker )); then
      IFS=$'\t' read -r run_id kind target sources seed <<<"${row}"
      if [[ ! -f "${PRETRAIN}/${run_id}/summary.json" ]]; then
        "${PYTHON_BIN}" -u -m mixture_scaling.train_strict \
          --config "${CONFIG}" --split-root "${SPLITS}" --sources "${sources}" \
          --run-id "${run_id}" --device "${gpu}" --seed "${seed}" --output-root "${PRETRAIN}" \
          >"${LOGS}/pretrain_${run_id}.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}
pretrain_worker 0 2 & p0=$!
pretrain_worker 1 2 & p1=$!
pretrain_worker 2 3 & p2=$!
pretrain_worker 3 3 & p3=$!
status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "pretraining failed" >&2; exit 1; }

probe() {
  local name="$1" checkpoint="$2" gpu="$3"
  local output="${RESULTS}/probe_${name}.json"
  [[ -f "${output}" ]] && return 0
  "${PYTHON_BIN}" -u -m mixture_scaling.probe_strict \
    --config "${CONFIG}" --split-root "${SPLITS}" --checkpoint "${checkpoint}" \
    --target twibot20 --device "${gpu}" --seed 0 --output "${output}" \
    >"${LOGS}/probe_${name}.log" 2>&1
}
probe scratch scratch 2 & q0=$!
probe twibot_target "${PRETRAIN}/twibot_target_s0/best.pt" 2 & q1=$!
probe ukrrus_single "${PRETRAIN}/ukrrus_single_s0/best.pt" 2 & q2=$!
probe cora_single "${PRETRAIN}/cora_single_s0/best.pt" 3 & q3=$!
probe facebook_single "${PRETRAIN}/facebook_single_s0/best.pt" 3 & q4=$!
probe all_non_target "${PRETRAIN}/all_non_target_s0/best.pt" 3 & q5=$!
status=0
for pid in "$q0" "$q1" "$q2" "$q3" "$q4" "$q5"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "frozen probes failed" >&2; exit 1; }

mapfile -t FINETUNE_ROWS < <(tail -n +2 "${ROOT}/manifests/twibot_strict_finetune.tsv")
finetune_worker() {
  local worker="$1" gpu="$2" index=0 row run_id initialization checkpoint seed
  for row in "${FINETUNE_ROWS[@]}"; do
    if (( index % 4 == worker )); then
      IFS=$'\t' read -r run_id initialization checkpoint seed <<<"${row}"
      [[ "${checkpoint}" == scratch ]] || checkpoint="${ROOT}/${checkpoint}"
      if [[ ! -f "${FINETUNE}/${run_id}/summary.json" ]]; then
        "${PYTHON_BIN}" -u -m mixture_scaling.finetune_strict \
          --config "${CONFIG}" --split-root "${SPLITS}" --checkpoint "${checkpoint}" \
          --target twibot20 --run-id "${run_id}" --device "${gpu}" --seed "${seed}" \
          --output-root "${FINETUNE}" >"${LOGS}/finetune_${run_id}.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}
finetune_worker 0 2 & f0=$!
finetune_worker 1 2 & f1=$!
finetune_worker 2 3 & f2=$!
finetune_worker 3 3 & f3=$!
status=0
for pid in "$f0" "$f1" "$f2" "$f3"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "fine-tuning failed" >&2; exit 1; }

"${PYTHON_BIN}" -m mixture_scaling.aggregate_strict_pilot \
  --probe-root "${RESULTS}" --finetune-root "${FINETUNE}" \
  --output "${ROOT}/results/twibot_strict_pilot/results.csv"
echo "complete $(date -u +%FT%TZ)"
