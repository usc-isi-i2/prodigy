#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ARTIFACT="${DATA_ROOT:-/dataMeR1/phil/data}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt"
REPORT_ROOT="${REPO_ROOT}/log/nm_ladder_global_finalcore/preflight"

python3 "${SCRIPT_DIR}/make_configs.py" --check
[[ -s "${ARTIFACT}" ]] || { echo "missing final-core split artifact: ${ARTIFACT}" >&2; exit 1; }
mkdir -p "${REPORT_ROOT}"
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
cd "${REPO_ROOT}"
"${CONDA_PREFIX}/bin/python" -u "${SCRIPT_DIR}/probe_global_subset.py" \
  --config "${SCRIPT_DIR}/configs/train_r2.yaml" \
  --config "${SCRIPT_DIR}/configs/train_r9.yaml" \
  --episodes "${PROBE_EPISODES:-100}" \
  --output "${REPORT_ROOT}/active_union.json"
echo "OK: final-core artifact and global active-union probes passed"
