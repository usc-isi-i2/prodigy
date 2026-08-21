#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODIGY_ROOT="${PRODIGY_ROOT:-/dataMeR1/phil/gfm/prodigy}"
PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${ROOT}/log/pipeline_s0}"
mkdir -p "${PIPELINE_LOG_ROOT}"

gpu_memory_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

echo "waiting for GPUs 2 and 3" | tee "${PIPELINE_LOG_ROOT}/status.txt"
while (( $(gpu_memory_mib 2) > 4096 || $(gpu_memory_mib 3) > 4096 )); do
  sleep 30
done

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"

build_citation_graph() {
  local dataset="$1" gpu="$2" artifact
  artifact="/dataMeR1/phil/data/${dataset}/graphs/citation_graph.pt"
  if [[ -f "${artifact}" ]]; then
    echo "citation artifact exists: ${artifact}"
    return 0
  fi
  conda run -n bio-embeddings-v001 env CUDA_VISIBLE_DEVICES="${gpu}" \
    python -u "${PRODIGY_ROOT}/scripts/graph_construction/generate_tag_citation_graph.py" \
      --dataset "${dataset}" --download --device cuda:0 \
      >"${PIPELINE_LOG_ROOT}/build_${dataset}.log" 2>&1
  [[ -f "${artifact}" ]] || { echo "builder did not create ${artifact}" >&2; return 1; }
}

build_citation_graph cora 2 & p1=$!
build_citation_graph pubmed 3 & p2=$!
wait "${p1}"; wait "${p2}"

conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${ROOT}"
PYTHONPATH="${ROOT}/src" python3 -c \
  'import sklearn, torch, torch_geometric, yaml; from mixture_scaling.train import make_loader; print("runtime imports ok")' \
  >"${PIPELINE_LOG_ROOT}/tests.log" 2>&1

smoke_id="smoke_covid_s0_$(date -u +%Y%m%dT%H%M%SZ)"
PYTHONPATH="${ROOT}/src" python3 -u -m mixture_scaling.train \
  --config "${ROOT}/configs/graphs.yaml" --sources covid_political \
  --run-id "${smoke_id}" --device 2 --seed 0 --max-steps 2 \
  --output-root "${ROOT}/state/smoke" >"${PIPELINE_LOG_ROOT}/smoke.log" 2>&1
grep -q '"status": "complete"' "${ROOT}/state/smoke/${smoke_id}/summary.json"

GPUS="2 3" WORKERS_PER_GPU=2 bash "${ROOT}/scripts/run_primary_tucker.sh" \
  >"${PIPELINE_LOG_ROOT}/training.log" 2>&1
GPUS="2 3" WORKERS_PER_GPU=2 bash "${ROOT}/scripts/eval_primary_tucker.sh" \
  >"${PIPELINE_LOG_ROOT}/evaluation.log" 2>&1
PYTHONPATH="${ROOT}/src" python3 -m mixture_scaling.aggregate \
  --raw-root "${ROOT}/results/primary_s0/raw" \
  --output "${ROOT}/results/primary_s0/primary_results.csv" --expected 56 \
  >"${PIPELINE_LOG_ROOT}/aggregate.log" 2>&1

echo "primary complete; starting follow-ups" | tee "${PIPELINE_LOG_ROOT}/status.txt"
PIPELINE_LOG_ROOT="${ROOT}/log/pipeline_followups" \
  bash "${ROOT}/scripts/pipeline_followups_tucker.sh" \
  >"${PIPELINE_LOG_ROOT}/followups.log" 2>&1

echo "all experiments complete $(date -u +%FT%TZ)" | tee "${PIPELINE_LOG_ROOT}/status.txt"
